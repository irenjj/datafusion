// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: See the NOTICE file distributed with this work for additional information regarding copyright ownership
// SPDX-FileCopyrightText: Licensed to the Apache Software Foundation (ASF) under one or more contributor license agreements

use std::sync::Arc;
use std::sync::atomic::Ordering;

use arrow::array::{
    Array, AsArray, FixedSizeListArray, Float32Array, RecordBatch, StringArray,
    StructArray,
};
use arrow::compute::concat_batches;
use arrow::datatypes::{DataType, Field, Float32Type, SchemaRef, UInt64Type};
use datafusion_ai::MediaRefType;
use datafusion_catalog::{Session, TableProvider};
use datafusion_lake::backfill::{
    BackfillPlan, BackfillReceipt, BackfillTarget, BackfillWorkUnit,
    bridge_backfill_stage, finalize_backfill, prepare_backfill, write_backfill_unit,
};
use datafusion_lake::dist::{
    LakeRuntimeConfig, bind_split, decode_split, encode_scan, encode_splits, rebuild_scan,
};
use datafusion_lake::{LakeObjectStores, LakeTableOptions, LanceReadOnlyTable};
use datafusion_physical_plan::collect;
use futures::{StreamExt, stream};
use lance::dataset::transaction::{Operation, Transaction};
use lance::table::format::Fragment;
use lance_file::version::LanceFileVersion;
use lance_table_runtime::{CommitBuilder, FileFragment, FragReadConfig, Table};
use lance_testing::datafusion::TestSession;
use object_store::PutMode;
use object_store::memory::InMemory;
use object_store::path::Path;
use paimon::arrow::{build_read_arrow_schema, build_target_arrow_schema};
use sha2::{Digest, Sha256};
use tokio::sync::Barrier;
use url::Url;

#[path = "support/parquet.rs"]
mod parquet;
#[path = "support/s3.rs"]
mod s3;

use parquet::{open_parquet, parquet_schema, stage_catalog, write_parquet_winner};
use s3::{
    IDS, ManifestProbe, ObservedStore, PREFIX, URI, bound_stores, file_config, ids,
    is_blob, next_manifest, object_state, object_states, sort_by_id, worker_context,
    write_fixture,
};

const DELETED_OFFSETS: [&[u32]; 3] = [&[1], &[0, 2], &[]];
const VISIBLE_IDS: [&[i32]; 3] = [&[42, 81], &[9], &[30]];
const STAGE_SPLITS: [u64; 3] = [50, 9, 101];
const DIM: i32 = 3;

#[tokio::test]
async fn s3_scan_replay_backfills_deleted_fragments_from_parquet() {
    let inner = Arc::new(InMemory::new());
    let source = write_fixture(Arc::clone(&inner)).await;
    let baseline = delete_rows(&source).await;
    let version = baseline.version_id();
    assert_eq!(version, source.version_id() + 1);
    assert_eq!(baseline.manifest().fragments.len(), STAGE_SPLITS.len());
    assert_eq!(
        baseline.manifest().data_storage_format.lance_file_format(),
        LanceFileVersion::V2_2.resolve()
    );
    let original_data = object_states(&inner, &baseline.data_dir()).await;
    let original_dvs = object_states(&inner, &baseline.deletions_dir()).await;
    assert_eq!(
        original_data.keys().filter(|path| is_blob(path)).count(),
        IDS.len()
    );
    assert_eq!(
        original_data
            .keys()
            .filter(|path| path.extension() == Some("lance"))
            .count(),
        STAGE_SPLITS.len()
    );
    assert_eq!(original_dvs.len(), 2);

    let observed = Arc::new(ObservedStore {
        inner: Arc::clone(&inner),
        manifest: Some(ManifestProbe {
            path: next_manifest(&baseline),
            ..Default::default()
        }),
        ..Default::default()
    });
    let coordinator_stores = bound_stores(observed.clone());
    let worker_stores = bound_stores(observed.clone());
    let options = LakeTableOptions::default();
    let coordinator = TestSession::default();
    let source = LanceReadOnlyTable::open_with_stores(
        URI,
        Some(version),
        &options,
        &coordinator_stores,
    )
    .await
    .unwrap();
    let scan = source.scan(&coordinator, None, &[], None).await.unwrap();
    let schema = scan.schema();
    let config = file_config(scan.as_ref());
    let scan_bytes = encode_scan(config).unwrap().unwrap();
    let split_bytes = encode_splits(config).unwrap();
    assert_eq!(split_bytes.len(), STAGE_SPLITS.len());
    let splits = STAGE_SPLITS
        .into_iter()
        .zip(split_bytes.iter().map(Vec::as_slice));
    let direct = collect(scan, coordinator.task_ctx()).await.unwrap();
    let direct = sort_by_id(&concat_batches(&schema, &direct).unwrap());
    observed.assert_no_blob_reads();
    assert_eq!(ids(&direct).values().as_ref(), &[9, 30, 42, 81]);
    direct
        .schema()
        .field(1)
        .try_extension_type::<MediaRefType>()
        .unwrap();
    let media = direct["payload"].as_struct();
    assert_eq!(media.null_count(), 0);
    let uris = media.column_by_name("uri").unwrap().as_string::<i32>();
    let offsets = media
        .column_by_name("byte_offset")
        .unwrap()
        .as_primitive::<UInt64Type>();
    let lengths = media
        .column_by_name("byte_len")
        .unwrap()
        .as_primitive::<UInt64Type>();
    for (row, id) in ids(&direct).values().iter().enumerate() {
        assert!(uris.value(row).starts_with(PREFIX));
        let path = s3_path(uris.value(row));
        assert!(is_blob(&path));
        let (size, _, digest) = original_data[&path];
        assert!(offsets.is_valid(row) && lengths.is_valid(row));
        assert_eq!(offsets.value(row), 0);
        assert_eq!(lengths.value(row), size);
        let expected: [u8; 32] = Sha256::digest([u8::try_from(*id).unwrap(); 128]).into();
        assert_eq!(digest, expected, "media reference for row {id}");
    }

    let mut unmatched = LakeObjectStores::default();
    unmatched
        .register("s3://test-bucket/lake/sibling/", observed.clone())
        .unwrap();
    let missing_worker =
        worker_context(LakeRuntimeConfig::default().with_object_stores(unmatched));
    let before = observed.read_ops.load(Ordering::Relaxed);
    let rebuilt = rebuild_scan(&scan_bytes, schema.clone(), &missing_worker).unwrap();
    let split = decode_split(&split_bytes[0]).unwrap();
    let bound = bind_split(file_config(rebuilt.as_ref()), split).unwrap();
    let error = collect(bound, missing_worker).await.unwrap_err();
    assert!(
        error
            .to_string()
            .contains("No Lake object store is registered"),
        "{error}"
    );
    assert_eq!(observed.read_ops.load(Ordering::Relaxed), before);

    let worker_session = TestSession::new(worker_context(
        LakeRuntimeConfig::default().with_object_stores(worker_stores.clone()),
    ));
    assert_ne!(worker_session.session_id(), coordinator.session_id());
    let worker = worker_session.task_ctx();
    let rebuilt = rebuild_scan(&scan_bytes, schema.clone(), &worker).unwrap();
    let config = file_config(rebuilt.as_ref());
    assert!(config.file_groups.is_empty());
    let mut replay = Vec::new();
    for (bytes, expected) in split_bytes.iter().zip(VISIBLE_IDS) {
        let plan = bind_split(config, decode_split(bytes).unwrap()).unwrap();
        let batches = collect(plan, Arc::clone(&worker)).await.unwrap();
        let rows = concat_batches(&schema, &batches).unwrap();
        assert_eq!(ids(&rows).values().as_ref(), expected);
        replay.push(rows);
    }
    let worker_rows = concat_batches(&schema, &replay).unwrap();
    assert_eq!(sort_by_id(&worker_rows), direct);
    observed.assert_no_blob_reads();

    let stage_schema = parquet_schema(&[
        Field::new(
            "embedding",
            DataType::FixedSizeList(
                Arc::new(Field::new("item", DataType::Float32, true)),
                DIM,
            ),
            true,
        ),
        Field::new(
            "annotation",
            DataType::Struct(
                vec![
                    Field::new("score", DataType::Float32, true),
                    Field::new("label", DataType::Utf8, true),
                ]
                .into(),
            ),
            true,
        ),
        Field::new("tag", DataType::Utf8, true),
    ]);
    let write_schema = build_target_arrow_schema(stage_schema.fields()).unwrap();
    let read_schema = build_read_arrow_schema(stage_schema.fields()).unwrap();
    let stage_dir = tempfile::tempdir().unwrap();
    let catalog = stage_catalog(stage_dir.path()).await;
    let mut selected = Vec::new();
    for (split_id, rows) in STAGE_SPLITS.into_iter().zip(&replay) {
        let batch = features(rows, write_schema.clone(), 0);
        let file = write_parquet_winner(
            &catalog,
            &format!("winner_{split_id}"),
            &stage_schema,
            &batch,
        )
        .await;
        selected.push((split_id, file));
    }
    let wrong = features(&replay[0], write_schema, 1000);
    let loser_file = write_parquet_winner(&catalog, "loser", &stage_schema, &wrong).await;
    let mut rejected = selected.clone();
    rejected[0].1 = loser_file;
    selected.reverse();

    let (plan, units) = prepare_backfill(
        BackfillTarget::Lance(URI),
        &read_schema,
        &coordinator_stores,
    )
    .await
    .unwrap();
    assert_eq!(plan.dataset_version(), version);
    assert_eq!(
        units.iter().map(|unit| unit.row_count).collect::<Vec<_>>(),
        [3, 3, 1]
    );
    let plan_bytes = plan.to_json_bytes().unwrap();
    let worker_plan = BackfillPlan::from_json_bytes(&plan_bytes).unwrap();
    let mut tampered: serde_json::Value = serde_json::from_slice(&plan_bytes).unwrap();
    tampered["payload"]["plan_fingerprint"] = "0".repeat(64).into();
    let error = BackfillPlan::from_json_bytes(&serde_json::to_vec(&tampered).unwrap())
        .unwrap_err();
    assert!(
        error.to_string().contains("fingerprint does not match"),
        "{error}"
    );

    let bridge = |winners| {
        bridge_backfill_stage(&worker_plan, splits.clone(), winners, open_parquet)
    };
    let missing = bridge(selected[1..].to_vec()).err().unwrap();
    assert!(
        missing.to_string().contains("Missing winning Parquet"),
        "{missing}"
    );
    let mut duplicate = selected.clone();
    duplicate[1] = duplicate[0].clone();
    let duplicate = bridge(duplicate).err().unwrap();
    assert!(
        duplicate.to_string().contains("Duplicate winning Parquet"),
        "{duplicate}"
    );

    let mut work = bridge(selected.clone()).unwrap();
    assert_eq!(
        work.iter().map(|(unit, _)| *unit).collect::<Vec<_>>(),
        units
    );
    let (unit, winner) = work.remove(0);
    let (losing_unit, loser) = bridge(rejected).unwrap().remove(0);
    assert_eq!(unit, losing_unit);
    let unit = BackfillWorkUnit::from_json_bytes(
        &unit.to_json_bytes(&plan).unwrap(),
        &worker_plan,
    )
    .unwrap();
    let ready = Barrier::new(2);
    let concurrent = |batches| {
        let ready = &ready;
        stream::once(async move {
            ready.wait().await;
            batches
        })
        .flatten()
    };
    let (loser, winner) = tokio::try_join!(
        write_backfill_unit(
            &worker_plan,
            &unit,
            "attempt-0",
            concurrent(loser),
            &worker_stores
        ),
        write_backfill_unit(
            &worker_plan,
            &unit,
            "attempt-1",
            concurrent(winner),
            &worker_stores
        ),
    )
    .unwrap();
    let loser_path = s3_path(&loser.artifact_uri);
    assert_ne!(loser_path, s3_path(&winner.artifact_uri));
    let loser_state = object_state(&inner, &loser_path).await;
    let (retry_unit, retry) = bridge(selected).unwrap().remove(0);
    assert_eq!(unit, retry_unit);
    let duplicate_attempt =
        write_backfill_unit(&worker_plan, &unit, "attempt-0", retry, &worker_stores)
            .await
            .unwrap_err();
    assert!(
        duplicate_attempt
            .to_string()
            .to_lowercase()
            .contains("already exists"),
        "{duplicate_attempt}"
    );

    let mut receipts = vec![winner];
    for (unit, batches) in work {
        receipts.push(
            write_backfill_unit(
                &worker_plan,
                &unit,
                "attempt-1",
                batches,
                &worker_stores,
            )
            .await
            .unwrap(),
        );
    }
    let receipts = receipts
        .iter()
        .map(|receipt| {
            BackfillReceipt::from_json_bytes(
                &receipt.to_json_bytes(&worker_plan).unwrap(),
                &plan,
            )
            .unwrap()
        })
        .collect::<Vec<_>>();
    assert_eq!(baseline.latest_version_id().await.unwrap(), version);
    let committed_version = finalize_backfill(&plan, &receipts, &coordinator_stores)
        .await
        .unwrap();
    assert_eq!(committed_version, version + 1);
    let stale = finalize_backfill(&plan, &receipts, &coordinator_stores)
        .await
        .unwrap_err();
    assert!(stale.to_string().contains("is stale"), "{stale}");
    assert_eq!(
        *observed.manifest.as_ref().unwrap().writes.lock().unwrap(),
        [(PutMode::Create, true)]
    );
    observed.assert_no_blob_reads();

    let updated =
        LanceReadOnlyTable::open_with_stores(URI, None, &options, &coordinator_stores)
            .await
            .unwrap();
    assert_eq!(updated.dataset_version(), committed_version);
    let scan = updated.scan(&coordinator, None, &[], None).await.unwrap();
    let schema = scan.schema();
    let batches = collect(scan, coordinator.task_ctx()).await.unwrap();
    let updated = sort_by_id(&concat_batches(&schema, &batches).unwrap());
    assert_eq!(updated.project(&[0, 1]).unwrap(), direct);
    let expected = features(&direct, read_schema, 0);
    // Lance normalizes the list child field name on readback.
    let embedding = updated["embedding"].as_fixed_size_list();
    let expected_embedding = expected["embedding"].as_fixed_size_list();
    assert_eq!(embedding.value_length(), DIM);
    assert_eq!(embedding.null_count(), 0);
    assert_eq!(
        embedding.values().as_primitive::<Float32Type>(),
        expected_embedding.values().as_primitive::<Float32Type>(),
    );
    for name in ["annotation", "tag"] {
        assert_eq!(updated[name].to_data(), expected[name].to_data());
    }
    observed.assert_no_blob_reads();

    let committed = Arc::new(baseline.checkout_version(committed_version).await.unwrap());
    assert_eq!(committed.manifest().fragments.len(), receipts.len());
    for (((before, after), receipt), offsets) in baseline
        .manifest()
        .fragments
        .iter()
        .zip(committed.manifest().fragments.iter())
        .zip(&receipts)
        .zip(DELETED_OFFSETS)
    {
        assert_eq!(after.id, before.id);
        assert_eq!(after.id, u64::from(receipt.work_unit.fragment_id));
        assert_eq!(after.physical_rows, before.physical_rows);
        assert_eq!(after.deletion_file, before.deletion_file);
        let (file, old_files) = after.files.split_last().unwrap();
        assert_eq!(old_files, &before.files);
        assert_eq!(
            Some(file.path.as_str()),
            s3_path(&receipt.artifact_uri).filename()
        );
        assert!(
            after
                .files
                .iter()
                .all(|file| Some(file.path.as_str()) != loser_path.filename())
        );
        if !offsets.is_empty() {
            let mut physical_only = Fragment::new(after.id);
            physical_only.files.push(file.clone());
            let padding = FileFragment::new(Arc::clone(&committed), physical_only)
                .take_rows(
                    offsets,
                    &file.schema(committed.schema()),
                    FragReadConfig::default(),
                )
                .await
                .unwrap();
            assert_eq!(padding.num_rows(), offsets.len());
            assert!(
                padding
                    .columns()
                    .iter()
                    .all(|column| column.null_count() == offsets.len())
            );
        }
    }
    for (path, state) in original_data.into_iter().chain(original_dvs) {
        assert_eq!(
            object_state(&inner, &path).await,
            state,
            "changed source object {path}"
        );
    }
    assert_eq!(object_state(&inner, &loser_path).await, loser_state);
}

async fn delete_rows(source: &Table) -> Table {
    let mut updated_fragments = Vec::new();
    for (fragment, offsets) in source.get_fragments().into_iter().zip(DELETED_OFFSETS) {
        if !offsets.is_empty() {
            let deleted = fragment
                .extend_deletions(offsets.iter().copied())
                .await
                .unwrap()
                .unwrap();
            updated_fragments.push(deleted.into());
        }
    }
    CommitBuilder::new(source)
        .execute(Transaction::new(
            source.version_id(),
            Operation::Delete {
                updated_fragments,
                deleted_fragment_ids: Vec::new(),
                predicate: "id IN (7, 15, 61)".into(),
            },
            None,
        ))
        .await
        .unwrap()
}

fn features(rows: &RecordBatch, schema: SchemaRef, offset: i32) -> RecordBatch {
    let DataType::FixedSizeList(item, size) = schema.field(0).data_type() else {
        panic!("expected a fixed-size vector");
    };
    let DataType::Struct(fields) = schema.field(1).data_type() else {
        panic!("expected a struct");
    };
    assert_eq!(*size, DIM);
    let ids = ids(rows).values().iter().map(|id| id + offset);
    let scores = ids
        .clone()
        .map(|id| f32::from(i16::try_from(id).unwrap()) * 0.25);
    let embedding = FixedSizeListArray::try_new(
        item.clone(),
        DIM,
        Arc::new(Float32Array::from_iter_values(
            scores.clone().flat_map(|s| [s, -s, s + 0.5]),
        )),
        None,
    )
    .unwrap();
    let annotation = StructArray::try_new(
        fields.clone(),
        vec![
            Arc::new(Float32Array::from_iter_values(scores)),
            Arc::new(StringArray::from_iter_values(
                ids.clone().map(|id| format!("label-{id}")),
            )),
        ],
        None,
    )
    .unwrap();
    let tags = StringArray::from_iter_values(ids.map(|id| format!("tag-{id}")));
    RecordBatch::try_new(
        schema,
        vec![Arc::new(embedding), Arc::new(annotation), Arc::new(tags)],
    )
    .unwrap()
}

fn s3_path(uri: &str) -> Path {
    let uri = Url::parse(uri).unwrap();
    assert_eq!(uri.scheme(), "s3");
    assert_eq!(uri.host_str(), Some("test-bucket"));
    Path::from_url_path(uri.path()).unwrap()
}
