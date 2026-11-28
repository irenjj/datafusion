// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: See the NOTICE file distributed with this work for additional information regarding copyright ownership
// SPDX-FileContributor: Licensed to the Apache Software Foundation (ASF) under one or more contributor license agreements

use std::collections::{BTreeMap, HashMap};
use std::fmt;
use std::num::NonZeroUsize;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};

use arrow::array::{Int32Array, RecordBatch};
use arrow::compute::{sort_to_indices, take_record_batch};
use arrow::datatypes::{DataType, Field, Schema};
use async_trait::async_trait;
use datafusion_datasource::file_scan_config::FileScanConfig;
use datafusion_datasource::source::DataSourceExec;
use datafusion_execution::{TaskContext, config::SessionConfig};
use datafusion_lake::LakeObjectStores;
use datafusion_lake::dist::LakeRuntimeConfig;
use datafusion_physical_plan::ExecutionPlan;
use futures::{TryStreamExt, stream::BoxStream};
use lance::{BlobArrayBuilder, BlobFieldOptions, blob_field_with_options};
use lance_file::version::LanceFileVersion;
use lance_io::object_store::{
    DEFAULT_CLOUD_IO_PARALLELISM, DEFAULT_DOWNLOAD_RETRY_COUNT,
    ObjectStore as LanceObjectStore, ObjectStoreParams, ObjectStoreProvider,
    ObjectStoreRegistry,
};
use lance_table_runtime::Table;
use lance_table_runtime::write::{InsertBuilder, WriteParams};
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

pub(super) const URI: &str = "s3://test-bucket/lake/nested%20prefix/source.lance";
pub(super) const PREFIX: &str = "s3://test-bucket/lake/nested%20prefix/";
pub(super) const IDS: [i32; 7] = [42, 7, 81, 15, 9, 61, 30];

pub(super) fn bound_stores(store: Arc<dyn ObjectStore>) -> LakeObjectStores {
    let mut stores = LakeObjectStores::default();
    stores.register(PREFIX, store).unwrap();
    stores
}

pub(super) fn worker_context(runtime: LakeRuntimeConfig) -> Arc<TaskContext> {
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

pub(super) fn file_config(plan: &dyn ExecutionPlan) -> &FileScanConfig {
    plan.downcast_ref::<DataSourceExec>()
        .unwrap()
        .data_source()
        .downcast_ref()
        .unwrap()
}

pub(super) fn ids(batch: &RecordBatch) -> &Int32Array {
    batch["id"].as_any().downcast_ref().unwrap()
}

pub(super) fn sort_by_id(batch: &RecordBatch) -> RecordBatch {
    let order = sort_to_indices(ids(batch), None, None).unwrap();
    take_record_batch(batch, &order).unwrap()
}

pub(super) fn next_manifest(table: &Table) -> Path {
    let base = Path::from_url_path(Url::parse(URI).unwrap().path()).unwrap();
    table
        .manifest_location()
        .naming_scheme
        .manifest_path(&base, table.version_id() + 1)
}

pub(super) fn is_blob(path: &Path) -> bool {
    path.extension() == Some("blob")
}

type ObjectState = (u64, String, [u8; 32]);

pub(super) async fn object_state(inner: &InMemory, path: &Path) -> ObjectState {
    let object = inner.get(path).await.unwrap();
    let size = object.meta.size;
    let etag = object.meta.e_tag.clone().unwrap();
    let bytes = object.bytes().await.unwrap();
    (size, etag, Sha256::digest(bytes).into())
}

pub(super) async fn object_states(
    inner: &InMemory,
    prefix: &Path,
) -> BTreeMap<Path, ObjectState> {
    let mut states = BTreeMap::new();
    let mut objects = inner.list(Some(prefix));
    while let Some(meta) = objects.try_next().await.unwrap() {
        let state = object_state(inner, &meta.location).await;
        states.insert(meta.location, state);
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

pub(super) async fn write_fixture(store: Arc<InMemory>) -> Table {
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
pub(super) struct ManifestProbe {
    pub(super) path: Path,
    pub(super) barrier: Option<Barrier>,
    pub(super) reject: bool,
    pub(super) writes: Mutex<Vec<(PutMode, bool)>>,
}

#[derive(Debug, Default)]
pub(super) struct ObservedStore {
    pub(super) inner: Arc<InMemory>,
    pub(super) read_ops: AtomicUsize,
    pub(super) fetched: Mutex<HashMap<Path, u64>>,
    pub(super) manifest: Option<ManifestProbe>,
    pub(super) reject_conditional_copy: bool,
}

impl ObservedStore {
    pub(super) fn assert_no_blob_reads(&self) {
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
