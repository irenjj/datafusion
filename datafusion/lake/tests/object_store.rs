// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: See the NOTICE file distributed with this work for additional information regarding copyright ownership
// SPDX-FileContributor: Licensed to the Apache Software Foundation (ASF) under one or more contributor license agreements

use std::collections::{BTreeMap, HashMap};
use std::fmt;
use std::num::NonZeroUsize;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Duration;

use arrow::array::{Int32Array, RecordBatch};
use arrow::compute::{concat_batches, sort_to_indices, take_record_batch};
use arrow::datatypes::{DataType, Field, Schema};
use async_trait::async_trait;
use datafusion_ai::MediaRefType;
use datafusion_catalog::{Session, TableProvider};
use datafusion_datasource::file_scan_config::FileScanConfig;
use datafusion_datasource::source::DataSourceExec;
use datafusion_execution::{TaskContext, config::SessionConfig};
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
use datafusion_physical_plan::{ExecutionPlan, collect};
use futures::stream::BoxStream;
use futures::{TryStreamExt, future::try_join_all, stream};
use lance::{BlobArrayBuilder, BlobFieldOptions, blob_field_with_options};
use lance_file::version::LanceFileVersion;
use lance_io::object_store::{
    DEFAULT_CLOUD_IO_PARALLELISM, DEFAULT_DOWNLOAD_RETRY_COUNT,
    ObjectStore as LanceObjectStore, ObjectStoreParams, ObjectStoreProvider,
    ObjectStoreRegistry,
};
use lance_table_runtime::Table;
use lance_table_runtime::write::{InsertBuilder, WriteParams};
use lance_testing::datafusion::TestSession;
use object_store::memory::InMemory;
use object_store::path::Path;
use object_store::{
    CopyMode, CopyOptions, GetOptions, GetResult, ListResult, MultipartUpload,
    ObjectMeta, ObjectStore, ObjectStoreExt, PutMode, PutMultipartOptions, PutOptions,
    PutPayload, PutResult,
};
use sha2::{Digest, Sha256};
use tokio::sync::Barrier;
use url::Url;

const URI: &str = "s3://test-bucket/lake/nested%20prefix/source.lance";
const PREFIX: &str = "s3://test-bucket/lake/nested%20prefix/";
const IDS: [i32; 7] = [42, 7, 81, 15, 9, 61, 30];

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

fn bound_stores(store: Arc<dyn ObjectStore>) -> LakeObjectStores {
    let mut stores = LakeObjectStores::default();
    stores.register(PREFIX, store).unwrap();
    stores
}

fn worker_context(runtime: LakeRuntimeConfig) -> Arc<TaskContext> {
    let mut config = SessionConfig::new();
    runtime.register(&mut config);
    Arc::new(TaskContext::new(
        None,
        "s3-worker".into(),
        config,
        Default::default(),
        Default::default(),
        Default::default(),
        Default::default(),
        Default::default(),
    ))
}

fn file_config(plan: &dyn ExecutionPlan) -> &FileScanConfig {
    plan.downcast_ref::<DataSourceExec>()
        .unwrap()
        .data_source()
        .downcast_ref()
        .unwrap()
}

fn ids(batch: &RecordBatch) -> &Int32Array {
    batch["id"].as_any().downcast_ref().unwrap()
}

fn sort_by_id(batch: &RecordBatch) -> RecordBatch {
    let order = sort_to_indices(ids(batch), None, None).unwrap();
    take_record_batch(batch, &order).unwrap()
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

fn next_manifest(table: &Table) -> Path {
    let base = Path::from_url_path(Url::parse(URI).unwrap().path()).unwrap();
    table
        .manifest_location()
        .naming_scheme
        .manifest_path(&base, table.version_id() + 1)
}

fn is_blob(path: &Path) -> bool {
    path.extension() == Some("blob")
}

async fn object_states(
    inner: &InMemory,
    prefix: &Path,
) -> BTreeMap<Path, (u64, [u8; 32])> {
    let mut states = BTreeMap::new();
    let mut objects = inner.list(Some(prefix));
    while let Some(meta) = objects.try_next().await.unwrap() {
        let bytes = inner
            .get(&meta.location)
            .await
            .unwrap()
            .bytes()
            .await
            .unwrap();
        states.insert(meta.location, (meta.size, Sha256::digest(bytes).into()));
    }
    states
}

#[derive(Debug)]
struct FixtureProvider(Arc<InMemory>);

#[async_trait]
impl ObjectStoreProvider for FixtureProvider {
    async fn new_store(
        &self,
        url: Url,
        _: &ObjectStoreParams,
    ) -> lance::Result<LanceObjectStore> {
        LanceObjectStore::try_new(
            self.0.clone(),
            url,
            None,
            None,
            false,
            true,
            DEFAULT_CLOUD_IO_PARALLELISM,
            DEFAULT_DOWNLOAD_RETRY_COUNT,
            None,
        )
    }
}

async fn write_fixture(store: Arc<InMemory>) -> Table {
    let registry = Arc::new(ObjectStoreRegistry::default());
    registry.insert("s3", Arc::new(FixtureProvider(store)));
    let session = lance::session::Session::new(0, 0, registry);
    let mut blobs = BlobArrayBuilder::new(IDS.len());
    for id in IDS {
        blobs
            .push_bytes(vec![u8::try_from(id).unwrap(); 128])
            .unwrap();
    }
    let blob = blob_field_with_options(
        "payload",
        true,
        BlobFieldOptions::default()
            .with_inline_size_threshold(1)
            .with_dedicated_size_threshold(NonZeroUsize::new(8).unwrap()),
    );
    let batch = RecordBatch::try_new(
        Arc::new(Schema::new(vec![
            Field::new("id", DataType::Int32, false),
            blob,
        ])),
        vec![
            Arc::new(Int32Array::from_iter_values(IDS)),
            blobs.finish().unwrap(),
        ],
    )
    .unwrap();
    InsertBuilder::new(URI)
        .with_params(&WriteParams {
            max_rows_per_file: 3,
            data_storage_version: Some(LanceFileVersion::V2_2),
            session: Some(Arc::new(session.as_ref().clone())),
            ..Default::default()
        })
        .execute(vec![batch])
        .await
        .unwrap()
}

#[derive(Debug, Default)]
struct ManifestProbe {
    path: Path,
    barrier: Option<Barrier>,
    reject: bool,
    writes: Mutex<Vec<(PutMode, bool)>>,
}

#[derive(Debug, Default)]
struct ObservedStore {
    inner: Arc<InMemory>,
    read_ops: AtomicUsize,
    fetched: Mutex<HashMap<Path, u64>>,
    manifest: Option<ManifestProbe>,
    reject_conditional_copy: bool,
}

impl ObservedStore {
    fn assert_no_blob_reads(&self) {
        let mut fetched = self.fetched.lock().unwrap();
        assert!(fetched.values().any(|bytes| *bytes > 0));
        let blob_bytes: u64 = fetched
            .iter()
            .filter(|(path, _)| is_blob(path))
            .map(|(_, bytes)| bytes)
            .sum();
        assert_eq!(blob_bytes, 0);
        fetched.clear();
    }
}

impl fmt::Display for ObservedStore {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("observed test S3 store")
    }
}

#[async_trait]
impl ObjectStore for ObservedStore {
    async fn get_opts(
        &self,
        path: &Path,
        options: GetOptions,
    ) -> object_store::Result<GetResult> {
        self.read_ops.fetch_add(1, Ordering::Relaxed);
        let head = options.head;
        let result = self.inner.get_opts(path, options).await?;
        if !head {
            *self
                .fetched
                .lock()
                .unwrap()
                .entry(path.clone())
                .or_default() += result.range.end - result.range.start;
        }
        Ok(result)
    }

    async fn put_opts(
        &self,
        path: &Path,
        payload: PutPayload,
        options: PutOptions,
    ) -> object_store::Result<PutResult> {
        let Some(probe) = self.manifest.as_ref().filter(|probe| probe.path == *path)
        else {
            return self.inner.put_opts(path, payload, options).await;
        };
        let mode = options.mode.clone();
        if let Some(barrier) = &probe.barrier {
            barrier.wait().await;
        }
        let result = if probe.reject && mode == PutMode::Create {
            Err(unsupported("conditional put"))
        } else {
            self.inner.put_opts(path, payload, options).await
        };
        probe.writes.lock().unwrap().push((mode, result.is_ok()));
        result
    }

    async fn put_multipart_opts(
        &self,
        path: &Path,
        options: PutMultipartOptions,
    ) -> object_store::Result<Box<dyn MultipartUpload>> {
        self.inner.put_multipart_opts(path, options).await
    }

    fn delete_stream(
        &self,
        paths: BoxStream<'static, object_store::Result<Path>>,
    ) -> BoxStream<'static, object_store::Result<Path>> {
        self.inner.delete_stream(paths)
    }

    fn list(
        &self,
        prefix: Option<&Path>,
    ) -> BoxStream<'static, object_store::Result<ObjectMeta>> {
        self.read_ops.fetch_add(1, Ordering::Relaxed);
        self.inner.list(prefix)
    }

    async fn list_with_delimiter(
        &self,
        prefix: Option<&Path>,
    ) -> object_store::Result<ListResult> {
        self.read_ops.fetch_add(1, Ordering::Relaxed);
        self.inner.list_with_delimiter(prefix).await
    }

    async fn copy_opts(
        &self,
        from: &Path,
        to: &Path,
        options: CopyOptions,
    ) -> object_store::Result<()> {
        if self.reject_conditional_copy && options.mode == CopyMode::Create {
            return Err(unsupported("conditional copy"));
        }
        self.inner.copy_opts(from, to, options).await
    }
}

fn unsupported(operation: &str) -> object_store::Error {
    object_store::Error::NotImplemented {
        operation: operation.into(),
        implementer: "observed test S3 store".into(),
    }
}
