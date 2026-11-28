// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: See the NOTICE file distributed with this work for additional information regarding copyright ownership
// SPDX-FileContributor: Licensed to the Apache Software Foundation (ASF) under one or more contributor license agreements

use std::collections::BTreeMap;
use std::sync::Arc;
use std::sync::atomic::Ordering;
use std::time::Duration;

use arrow::array::{Int32Array, RecordBatch};
use arrow::compute::concat_batches;
use arrow::datatypes::{DataType, Field, Schema};
use datafusion_ai::MediaRefType;
use datafusion_catalog::{Session, TableProvider};
use datafusion_lake::backfill::{
    BackfillPlan, BackfillReceipt, BackfillTarget, BackfillWorkUnit, finalize_backfill,
    prepare_backfill, write_backfill_unit,
};
use datafusion_lake::dist::{
    LakeRuntimeConfig, bind_split, decode_split, encode_scan, encode_splits, rebuild_scan,
};
use datafusion_lake::{
    LakeObjectStores, LakeSplitExt, LakeSplitPayload, LakeTableOptions,
    LanceReadOnlyTable,
};
use datafusion_physical_plan::collect;
use futures::{future::try_join_all, stream};
use lance_file::version::LanceFileVersion;
use lance_testing::datafusion::TestSession;
use object_store::memory::InMemory;
use object_store::path::Path;
use object_store::{ObjectStore, ObjectStoreExt, PutMode};
use tokio::sync::Barrier;
use url::Url;

#[path = "support/s3.rs"]
mod s3;

use s3::{
    IDS, ManifestProbe, ObservedStore, PREFIX, URI, bound_stores, file_config, ids,
    is_blob, next_manifest, object_states, sort_by_id, worker_context, write_fixture,
};

#[test]
fn resolves_most_specific_scope() {
    let mut stores = LakeObjectStores::default();
    let scheme: Arc<dyn ObjectStore> = Arc::new(InMemory::new());
    let bucket: Arc<dyn ObjectStore> = Arc::new(InMemory::new());
    let prefix: Arc<dyn ObjectStore> = Arc::new(InMemory::new());
    stores.register("s3://", Arc::clone(&scheme)).unwrap();
    stores
        .register("s3://test-bucket", Arc::clone(&bucket))
        .unwrap();
    stores.register(PREFIX, Arc::clone(&prefix)).unwrap();

    for (uri, expected) in [
        (URI, &prefix),
        ("s3://test-bucket/lake/nested%20prefix2/data", &bucket),
        ("s3://test-bucket.evil/lake/nested%20prefix/data", &scheme),
    ] {
        let resolved = stores.resolve(&Url::parse(uri).unwrap()).unwrap();
        assert!(Arc::ptr_eq(&resolved, expected));
    }
    for invalid in [
        "s3://test-bucket/root?token=secret",
        "s3://user:secret@test-bucket/root",
        "s3://test-bucket/root#fragment",
    ] {
        assert!(stores.resolve(&Url::parse(invalid).unwrap()).is_none());
        let error = stores.register(invalid, Arc::clone(&prefix)).unwrap_err();
        assert!(error.to_string().contains("credential-free URL prefix"));
        assert!(!error.to_string().contains("secret"));
    }
    assert!(
        stores
            .resolve(&Url::parse("unknown://bucket/key").unwrap())
            .is_none()
    );
}

#[tokio::test]
async fn s3_scan_replay_and_atomic_backfill() {
    let inner = Arc::new(InMemory::new());
    let baseline = write_fixture(Arc::clone(&inner)).await;
    let version = baseline.version_id();
    assert_eq!(baseline.manifest().fragments.len(), 3);
    assert_eq!(
        baseline.manifest().data_storage_format.lance_file_format(),
        LanceFileVersion::V2_2.resolve()
    );
    let files = object_states(&inner, &baseline.data_dir()).await;
    assert_eq!(files.keys().filter(|path| is_blob(path)).count(), IDS.len());

    let observed = Arc::new(ObservedStore {
        inner: Arc::clone(&inner),
        manifest: Some(ManifestProbe {
            path: next_manifest(&baseline),
            barrier: Some(Barrier::new(2)),
            ..Default::default()
        }),
        ..Default::default()
    });
    let mut stores = bound_stores(observed.clone());
    stores
        .register("s3://test-bucket/lake/", Arc::new(InMemory::new()))
        .unwrap();
    let table = open(&stores).await;
    assert_eq!(table.dataset_version(), version);
    table
        .schema()
        .field(1)
        .try_extension_type::<MediaRefType>()
        .unwrap();

    let coordinator = TestSession::default();
    let direct_scan = table.scan(&coordinator, None, &[], None).await.unwrap();
    let schema = direct_scan.schema();
    let scan = encode_scan(file_config(direct_scan.as_ref()))
        .unwrap()
        .unwrap();
    let splits = encode_splits(file_config(direct_scan.as_ref())).unwrap();
    assert_eq!(splits.len(), 3);
    let direct = collect(direct_scan, coordinator.task_ctx()).await.unwrap();
    let direct = concat_batches(&schema, &direct).unwrap();
    observed.assert_no_blob_reads();

    let worker = worker_context(
        LakeRuntimeConfig::default()
            .with_object_stores(bound_stores(observed.clone()))
            .with_lance_builder(|_| panic!("registered store must take precedence")),
    );
    let before = observed.read_ops.load(Ordering::Relaxed);
    let replay = rebuild_scan(&scan, Arc::clone(&schema), &worker).unwrap();
    let config = file_config(replay.as_ref());
    assert!(config.file_groups.is_empty());
    let bound = splits
        .iter()
        .map(|payload| {
            let split = decode_split(payload).unwrap();
            let LakeSplitPayload::Lance { fragment_id, .. } =
                split.extension::<LakeSplitExt>().unwrap().payload
            else {
                panic!("expected Lance split");
            };
            (fragment_id, bind_split(config, split).unwrap())
        })
        .collect::<Vec<_>>();
    assert_eq!(observed.read_ops.load(Ordering::Relaxed), before);
    let replay = try_join_all(bound.into_iter().map(|(fragment, plan)| {
        let worker = Arc::clone(&worker);
        let schema = Arc::clone(&schema);
        async move {
            let batches = collect(plan, worker).await?;
            Ok::<_, datafusion_common::DataFusionError>((
                fragment,
                concat_batches(&schema, &batches)?,
            ))
        }
    }))
    .await
    .unwrap()
    .into_iter()
    .collect::<BTreeMap<_, _>>();
    let replay_rows = concat_batches(&schema, replay.values()).unwrap();
    assert_eq!(ids(&replay_rows).values().as_ref(), &IDS);
    let direct = sort_by_id(&direct);
    assert_eq!(sort_by_id(&replay_rows), direct);
    assert!(observed.read_ops.load(Ordering::Relaxed) > before);
    observed.assert_no_blob_reads();

    let columns = derived_columns(&IDS, 0).schema();
    let (plan, units) = prepare_backfill(BackfillTarget::Lance(URI), &columns, &stores)
        .await
        .unwrap();
    assert_eq!(plan.dataset_version(), version);
    assert_eq!(units.len(), replay.len());
    let worker_stores = bound_stores(observed.clone());
    let attempts = |attempt_id, offset| {
        let plan = &plan;
        let worker_stores = &worker_stores;
        let replay = &replay;
        try_join_all(units.iter().map(move |unit| {
            let physical = ids(&replay[&unit.fragment_id]).values();
            assert_eq!(physical.len() as u64, unit.row_count);
            write_backfill_unit(
                plan,
                unit,
                attempt_id,
                stream::iter([Ok(derived_columns(physical, offset))]),
                worker_stores,
            )
        }))
    };
    let (first, second) =
        tokio::try_join!(attempts("first", 1), attempts("second", 2)).unwrap();
    assert!(
        first
            .iter()
            .zip(&second)
            .all(|(a, b)| a.artifact_uri != b.artifact_uri)
    );
    let unit = &units[0];
    let duplicate = write_backfill_unit(
        &plan,
        unit,
        "first",
        stream::iter([Ok(derived_columns(
            ids(&replay[&unit.fragment_id]).values(),
            3,
        ))]),
        &worker_stores,
    )
    .await
    .unwrap_err();
    assert!(
        duplicate
            .to_string()
            .to_lowercase()
            .contains("already exists"),
        "{duplicate}"
    );
    assert_eq!(open(&stores).await.dataset_version(), version);

    let (result1, result2) = tokio::time::timeout(Duration::from_secs(15), async {
        tokio::join!(
            Box::pin(finalize_backfill(&plan, &first, &stores)),
            Box::pin(finalize_backfill(&plan, &second, &stores))
        )
    })
    .await
    .expect("both finalizers must reach the same manifest write");
    let (new_version, conflict, winners, offset) = match (result1, result2) {
        (Ok(new_version), Err(error)) => (new_version, error, first, 1),
        (Err(error), Ok(new_version)) => (new_version, error, second, 2),
        results => panic!("expected one successful manifest creation: {results:?}"),
    };
    assert_eq!(new_version, version + 1);
    assert!(
        conflict
            .to_string()
            .to_lowercase()
            .contains("commit conflict"),
        "{conflict}"
    );
    let probe = observed.manifest.as_ref().unwrap();
    {
        let writes = probe.writes.lock().unwrap();
        assert_eq!(writes.len(), 2);
        assert!(writes.iter().all(|(mode, _)| *mode == PutMode::Create));
        assert_eq!(writes.iter().filter(|(_, success)| *success).count(), 1);
    }

    let updated = open(&stores).await;
    assert_eq!(updated.dataset_version(), new_version);
    let scan = updated.scan(&coordinator, None, &[], None).await.unwrap();
    let schema = scan.schema();
    let batches = collect(scan, coordinator.task_ctx()).await.unwrap();
    let updated_rows = sort_by_id(&concat_batches(&schema, &batches).unwrap());
    assert_eq!(updated_rows.project(&[0, 1]).unwrap(), direct);
    assert_eq!(
        updated_rows.project(&[2]).unwrap(),
        derived_columns(ids(&direct).values(), offset)
    );
    observed.assert_no_blob_reads();
    for winner in winners {
        let uri = Url::parse(&winner.artifact_uri).unwrap();
        let path = Path::from_url_path(uri.path()).unwrap();
        assert!(inner.head(&path).await.is_ok());
    }
    let current = object_states(&inner, &baseline.data_dir()).await;
    for (path, state) in files {
        assert_eq!(current.get(&path), Some(&state), "changed object {path}");
    }
}

#[tokio::test]
async fn unmatched_storage_fails_closed() {
    let observed = Arc::new(ObservedStore::default());
    let stores = bound_stores(observed.clone());
    let options = LakeTableOptions::default();
    let unmatched = "s3://test-bucket/lake/nested%20prefix2/source.lance";
    let wrong_scheme = "unregistered://test-bucket/lake/nested%20prefix/source.lance";
    for (uri, expected) in [
        (unmatched, "No Lake object store is registered for scheme"),
        (wrong_scheme, "No object store provider found for scheme"),
    ] {
        assert!(stores.resolve(&Url::parse(uri).unwrap()).is_none());
        let error = LanceReadOnlyTable::open_with_stores(uri, None, &options, &stores)
            .await
            .unwrap_err();
        assert!(error.to_string().contains(expected), "{error}");
        let error = prepare_backfill(
            BackfillTarget::Lance(uri),
            &derived_columns(&[], 0).schema(),
            &stores,
        )
        .await
        .unwrap_err();
        assert!(error.to_string().contains(expected), "{error}");
    }
    assert_eq!(observed.read_ops.load(Ordering::Relaxed), 0);

    let inner = Arc::new(InMemory::new());
    write_fixture(Arc::clone(&inner)).await;
    let source = open(&bound_stores(inner)).await;
    let coordinator = TestSession::default();
    let plan = source.scan(&coordinator, None, &[], None).await.unwrap();
    let config = file_config(plan.as_ref());
    let scan = encode_scan(config).unwrap().unwrap();
    let splits = encode_splits(config).unwrap();
    let mut worker_stores = LakeObjectStores::default();
    worker_stores.register(unmatched, observed.clone()).unwrap();
    let worker =
        worker_context(LakeRuntimeConfig::default().with_object_stores(worker_stores));
    let replay = rebuild_scan(&scan, plan.schema(), &worker).unwrap();
    let split = decode_split(&splits[0]).unwrap();
    let bound = bind_split(file_config(replay.as_ref()), split).unwrap();
    let error = collect(bound, worker).await.unwrap_err();
    assert!(
        error
            .to_string()
            .contains("No Lake object store is registered for scheme"),
        "{error}"
    );
    assert_eq!(observed.read_ops.load(Ordering::Relaxed), 0);
}

#[tokio::test]
async fn s3_backfill_requires_conditional_writes() {
    let inner = Arc::new(InMemory::new());
    let baseline = write_fixture(Arc::clone(&inner)).await;
    let target = next_manifest(&baseline);
    let stores = bound_stores(Arc::new(ObservedStore {
        inner: Arc::clone(&inner),
        reject_conditional_copy: true,
        ..Default::default()
    }));
    let (plan, units) = prepare_backfill(
        BackfillTarget::Lance(URI),
        &derived_columns(&[], 0).schema(),
        &stores,
    )
    .await
    .unwrap();
    let error = write_test_unit(&plan, &units[0], &stores)
        .await
        .unwrap_err();
    assert!(
        error.to_string().contains("S3CopyIfNotExists::Multipart"),
        "{error}"
    );

    let observed = Arc::new(ObservedStore {
        inner: Arc::clone(&inner),
        manifest: Some(ManifestProbe {
            path: target.clone(),
            reject: true,
            ..Default::default()
        }),
        ..Default::default()
    });
    let stores = bound_stores(observed.clone());
    let receipts = try_join_all(
        units
            .iter()
            .map(|unit| write_test_unit(&plan, unit, &stores)),
    )
    .await
    .unwrap();
    let error = finalize_backfill(&plan, &receipts, &stores)
        .await
        .unwrap_err();
    assert!(
        error.to_string().contains("atomic PutMode::Create"),
        "{error}"
    );
    assert_eq!(open(&stores).await.dataset_version(), baseline.version_id());
    assert!(matches!(
        inner.head(&target).await,
        Err(object_store::Error::NotFound { .. })
    ));
    assert_eq!(
        *observed.manifest.as_ref().unwrap().writes.lock().unwrap(),
        [(PutMode::Create, false)]
    );
}

async fn write_test_unit(
    plan: &BackfillPlan,
    unit: &BackfillWorkUnit,
    stores: &LakeObjectStores,
) -> datafusion_common::Result<BackfillReceipt> {
    let batch = derived_columns(&vec![0; unit.row_count as usize], 0);
    write_backfill_unit(plan, unit, "test", stream::iter([Ok(batch)]), stores).await
}

async fn open(stores: &LakeObjectStores) -> LanceReadOnlyTable {
    LanceReadOnlyTable::open_with_stores(URI, None, &LakeTableOptions::default(), stores)
        .await
        .unwrap()
}

fn derived_columns(ids: &[i32], offset: i32) -> RecordBatch {
    let values = Int32Array::from_iter_values(ids.iter().map(|id| id * 10 + offset));
    RecordBatch::try_new(
        Arc::new(Schema::new(vec![Field::new(
            "derived",
            DataType::Int32,
            false,
        )])),
        vec![Arc::new(values)],
    )
    .unwrap()
}
