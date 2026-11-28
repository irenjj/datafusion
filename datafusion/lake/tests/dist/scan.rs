// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: See the NOTICE file distributed with this work for additional information regarding copyright ownership
// SPDX-FileContributor: Licensed to the Apache Software Foundation (ASF) under one or more contributor license agreements

use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};

use arrow::array::{Array, LargeBinaryArray, RecordBatch};
use arrow::compute::{SortColumn, concat_batches, lexsort_to_indices, take_record_batch};
use arrow::datatypes::{DataType, Field, Schema, SchemaRef};
use datafusion_catalog::{Session, TableProvider};
use datafusion_common::DataFusionError;
use datafusion_datasource::file_groups::FileGroup;
use datafusion_execution::{TaskContext, config::SessionConfig};
use datafusion_expr::{col, lit};
use datafusion_lake::dist::{
    LAKE_SCAN_PROTOCOL_NAME, LAKE_SCAN_PROTOCOL_VERSION, LakeRuntimeConfig, bind_split,
    decode_split, encode_scan, encode_splits, rebuild_scan,
};
use datafusion_lake::{
    LakeSplitExt, LakeTableOptions, LanceReadOnlyTable, PaimonReadOnlyTable,
};
use datafusion_physical_plan::{ExecutionPlan, collect};
use futures::future::try_join_all;
use lance::blob::{BlobDescriptorArrayBuilder, BlobRange, blob_field};
use lance_table_runtime::write::WriteMode;
use lance_testing::datafusion::TestSession;
use paimon::catalog::Identifier;
use paimon::io::{FileIOBuilder, FileRead};
use paimon::spec::{BlobDescriptor, BlobType, DataType as PaimonType};
use paimon::{Catalog, CatalogOptions, FileSystemCatalog, Options};
use serde_json::{Value, json};
use url::Url;

use super::fixtures::{
    file_config, ids, scalar_paimon_schema, scalar_rows, write_lance, write_paimon,
};
use super::media::{observed_lance, read_references};
use super::store::Reads;

const MAGIC: &[u8] = b"DFLKS1\0";
const HEADER_LEN: usize = MAGIC.len() + size_of::<u32>();
const MIME: &str = "image/jpeg";

fn worker(runtime: LakeRuntimeConfig) -> Arc<TaskContext> {
    let mut config = SessionConfig::new();
    assert!(LakeRuntimeConfig::get(&config).is_none());
    runtime.register(&mut config);
    assert!(LakeRuntimeConfig::get(&config).is_some());
    Arc::new(TaskContext::new(
        None,
        "lake-worker".into(),
        config,
        Default::default(),
        Default::default(),
        Default::default(),
        Default::default(),
        Default::default(),
    ))
}

fn with_payload(start: i32, field: Field, payload: Arc<dyn Array>) -> RecordBatch {
    let scalars = scalar_rows(start);
    let schema = scalars.schema();
    let fields = schema.fields().iter().cloned().chain([Arc::new(field)]);
    let columns = scalars.columns().iter().cloned().chain([payload]);
    RecordBatch::try_new(
        Arc::new(Schema::new(fields.collect::<Vec<_>>())),
        columns.collect(),
    )
    .unwrap()
}

fn lance_rows(start: i32, uri: &Url) -> RecordBatch {
    let mut payload =
        BlobDescriptorArrayBuilder::for_field(&blob_field("payload", true)).unwrap();
    for row in 0..6 {
        payload
            .push_external(
                uri.as_str(),
                Some(BlobRange {
                    offset: row * 2,
                    size: 2,
                }),
            )
            .unwrap();
    }
    let (mut field, array) = payload.finish().unwrap().into_parts();
    field.metadata_mut().insert("df.media.mime", MIME);
    with_payload(start, field, array)
}

fn paimon_rows(start: i32, uri: &Url) -> RecordBatch {
    let payload = LargeBinaryArray::from_iter_values(
        (0..6).map(|row| BlobDescriptor::new(uri.to_string(), row * 2, 2).serialize()),
    );
    with_payload(
        start,
        Field::new("payload", DataType::LargeBinary, true),
        Arc::new(payload),
    )
}

fn transport(plan: &dyn ExecutionPlan) -> (Vec<u8>, Vec<Vec<u8>>) {
    let config = file_config(plan);
    let payload = encode_scan(config).unwrap().unwrap();
    assert_eq!(encode_scan(config).unwrap().unwrap(), payload);
    let splits = encode_splits(config).unwrap();
    assert_eq!(splits.len(), 2);
    for (bytes, source) in splits
        .iter()
        .zip(config.file_groups.iter().flat_map(FileGroup::files))
    {
        let (_, legacy_bytes) =
            LakeSplitExt::try_encode_from_extensions(&source.extensions)
                .unwrap()
                .unwrap();
        assert_eq!(bytes, &legacy_bytes);
        let file = decode_split(bytes).unwrap();
        assert_eq!(file.object_meta.location, source.object_meta.location);
        assert_eq!(
            file.extension::<LakeSplitExt>(),
            source.extension::<LakeSplitExt>()
        );
    }
    (payload, splits)
}

fn bind(template: &dyn ExecutionPlan, splits: &[Vec<u8>]) -> Vec<Arc<dyn ExecutionPlan>> {
    let config = file_config(template);
    assert!(config.file_groups.is_empty());
    assert!(encode_splits(config).unwrap().is_empty());
    splits
        .iter()
        .map(|bytes| {
            let plan = bind_split(config, decode_split(bytes).unwrap()).unwrap();
            assert_eq!(plan.schema(), template.schema());
            assert_eq!(plan.properties().output_partitioning().partition_count(), 1);
            assert_eq!(file_config(plan.as_ref()).file_groups[0].files().len(), 1);
            plan
        })
        .collect()
}

async fn collect_splits(
    plans: Vec<Arc<dyn ExecutionPlan>>,
    ctx: Arc<TaskContext>,
) -> Vec<RecordBatch> {
    try_join_all(
        plans
            .into_iter()
            .map(|plan| collect(plan, Arc::clone(&ctx))),
    )
    .await
    .unwrap()
    .into_iter()
    .flatten()
    .collect()
}

fn sorted(schema: &SchemaRef, batches: &[RecordBatch]) -> RecordBatch {
    let batch = concat_batches(schema, batches).unwrap();
    let id = SortColumn {
        values: Arc::clone(&batch["id"]),
        options: None,
    };
    take_record_batch(&batch, &lexsort_to_indices(&[id], None).unwrap()).unwrap()
}

fn assert_zero_payload_bytes(reads: &Reads, batches: &[RecordBatch], uri: &Url) {
    let log = reads.take();
    assert!(!log.is_empty());
    let references = read_references(batches);
    assert_eq!(references.len(), 4);
    for reference in references.into_iter().map(Option::unwrap) {
        assert_eq!(reference.uri, uri.as_str());
        assert_eq!(reference.mime.as_deref(), Some(MIME));
        assert_eq!(reference.byte_len, Some(2));
    }
    let bytes: u64 = log
        .iter()
        .filter(|read| read.path.filename() == Some("payload.blob"))
        .filter_map(|read| read.range.as_ref())
        .map(|range| range.end - range.start)
        .sum();
    assert_eq!(bytes, 0, "{log:?}");
}

fn blob_projection(table: &dyn TableProvider) -> [usize; 3] {
    let schema = table.schema();
    ["payload", "id", "part"].map(|name| schema.index_of(name).unwrap())
}

#[tokio::test]
async fn lance_scan_replays_pinned_fragments_with_local_storage() {
    let directory = tempfile::tempdir().unwrap();
    let uri = Url::from_file_path(directory.path().join("fixture.lance")).unwrap();
    let payload_path = directory.path().join("payload.blob");
    std::fs::write(&payload_path, [7; 32]).unwrap();
    let media_uri = Url::from_file_path(&payload_path).unwrap();
    write_lance(uri.as_str(), lance_rows(0, &media_uri), WriteMode::Create).await;
    let table =
        LanceReadOnlyTable::open(uri.as_str(), None, &LakeTableOptions::default())
            .await
            .unwrap();
    let mut filters = vec![
        col("id").gt_eq(lit(1)),
        col("label")
            .is_null()
            .or(col("label").in_list(vec![lit("a"), lit("c")], false)),
    ];
    let coordinator = TestSession::default();
    let local = table
        .scan(&coordinator, Some(&blob_projection(&table)), &filters, None)
        .await
        .unwrap();
    let schema = local.schema();
    assert_eq!(schema.field(2).name(), "part");
    let (payload, splits) = transport(local.as_ref());
    assert!(!String::from_utf8_lossy(&payload).contains("payload.blob"));
    let expected = collect(local, coordinator.task_ctx()).await.unwrap();
    assert_eq!(ids(&expected), [1, 3, 4, 5]);

    write_lance(uri.as_str(), lance_rows(6, &media_uri), WriteMode::Append).await;
    std::fs::remove_file(payload_path).unwrap();
    let count = Arc::new(AtomicUsize::new(0));
    let reads = Reads::new();
    let worker = worker(LakeRuntimeConfig::default().with_lance_builder({
        let count = Arc::clone(&count);
        let reads = reads.clone();
        move |uri| {
            count.fetch_add(1, Ordering::SeqCst);
            Ok(observed_lance(uri, &reads))
        }
    }));
    let template = rebuild_scan(&payload, Arc::clone(&schema), &worker).unwrap();
    let bound = bind(template.as_ref(), &splits);
    assert_eq!(count.load(Ordering::SeqCst), 0);
    assert!(reads.take().is_empty());
    let actual = collect_splits(bound, worker).await;
    assert_eq!(count.load(Ordering::SeqCst), 1);
    assert_eq!(sorted(&schema, &expected), sorted(&schema, &actual));
    assert_zero_payload_bytes(&reads, &actual, &media_uri);

    let limited = table
        .scan(&coordinator, None, &filters, Some(1))
        .await
        .unwrap();
    assert!(
        encode_scan(file_config(limited.as_ref()))
            .unwrap_err()
            .to_string()
            .contains("pushed fetch")
    );
    filters.push(col("label").eq(lit("a".repeat(1024 * 1024))));
    let oversized = table
        .scan(&coordinator, None, &filters, None)
        .await
        .unwrap();
    assert!(
        encode_scan(file_config(oversized.as_ref()))
            .unwrap_err()
            .to_string()
            .contains("byte limit")
    );
}

#[tokio::test]
async fn paimon_scan_replays_pinned_partitions_with_residual_and_blobs() {
    let directory = tempfile::tempdir().unwrap();
    let payload_path = directory.path().join("payload.blob");
    std::fs::write(&payload_path, [7; 32]).unwrap();
    let media_uri = Url::from_file_path(&payload_path).unwrap();
    let mut options = Options::new();
    let warehouse = Url::from_file_path(directory.path()).unwrap();
    options.set(CatalogOptions::WAREHOUSE, warehouse.as_str());
    let catalog = FileSystemCatalog::new(options).unwrap();
    catalog
        .create_database("test", true, Default::default())
        .await
        .unwrap();
    let name = Identifier::new("test", "fixture");
    let schema = scalar_paimon_schema()
        .column("payload", PaimonType::Blob(BlobType::new()))
        .option("row-tracking.enabled", "true")
        .option("data-evolution.enabled", "true")
        .option("blob-descriptor-field", "payload")
        .option("fields.payload.df.media.mime", MIME)
        .build()
        .unwrap();
    catalog.create_table(&name, schema, false).await.unwrap();
    let native = catalog.get_table(&name).await.unwrap();
    assert!(native.location().starts_with("file://"));
    Box::pin(write_paimon(&native, &paimon_rows(0, &media_uri))).await;
    let table =
        PaimonReadOnlyTable::open(native.clone(), None, &LakeTableOptions::default())
            .await
            .unwrap();
    let coordinator = TestSession::default();
    let filters = [
        col("id").not_eq(lit(2)),
        col("label").is_null().or(col("label").eq(lit("a"))),
    ];
    let local = table
        .scan(&coordinator, Some(&blob_projection(&table)), &filters, None)
        .await
        .unwrap();
    let schema = local.schema();
    let (payload, splits) = transport(local.as_ref());
    assert!(!String::from_utf8_lossy(&payload).contains("payload.blob"));
    let expected = collect(local, coordinator.task_ctx()).await.unwrap();
    assert_eq!(ids(&expected), [0, 1, 4, 5]);

    let reads = Reads::new();
    let io = FileIOBuilder::new("file")
        .build()
        .unwrap()
        .with_file_block_cache(Arc::new(reads.clone()), 4096, "*")
        .unwrap();
    let reader = io
        .new_input(media_uri.as_str())
        .unwrap()
        .reader()
        .await
        .unwrap();
    assert_eq!(reader.read(0..2).await.unwrap().as_ref(), &[7, 7]);
    assert!(
        reads
            .take()
            .iter()
            .any(|read| read.path.filename() == Some("payload.blob"))
    );
    Box::pin(write_paimon(&native, &paimon_rows(6, &media_uri))).await;
    std::fs::remove_file(payload_path).unwrap();
    let count = Arc::new(AtomicUsize::new(0));
    let worker = worker(LakeRuntimeConfig::default().with_paimon_file_io({
        let count = Arc::clone(&count);
        move |_| {
            count.fetch_add(1, Ordering::SeqCst);
            Ok(io.clone())
        }
    }));
    let template = rebuild_scan(&payload, Arc::clone(&schema), &worker).unwrap();
    let bound = bind(template.as_ref(), &splits);
    assert_eq!(count.load(Ordering::SeqCst), 0);
    assert!(reads.take().is_empty());
    let actual = collect_splits(bound, Arc::clone(&worker)).await;
    assert_eq!(count.load(Ordering::SeqCst), 1);
    assert_eq!(sorted(&schema, &expected), sorted(&schema, &actual));
    assert_zero_payload_bytes(&reads, &actual, &media_uri);

    for (field, value) in [
        ("table_uri", json!("file:///other/table.paimon")),
        ("snapshot_id", json!(table.snapshot_id() + 1)),
    ] {
        let changed = change_template(&payload, field, value);
        let template = rebuild_scan(&changed, Arc::clone(&schema), &worker).unwrap();
        let error = bind_split(
            file_config(template.as_ref()),
            decode_split(&splits[0]).unwrap(),
        )
        .unwrap_err();
        assert_plan_error(&error, "does not match the pinned table scan");
    }
    assert_eq!(count.load(Ordering::SeqCst), 1);

    let wrong_schema =
        change_template(&payload, "schema_id", json!(native.schema().id() + 1));
    let template = rebuild_scan(&wrong_schema, schema, &worker).unwrap();
    let mut bound = bind(template.as_ref(), &splits);
    let error = collect(bound.remove(0), worker).await.unwrap_err();
    assert!(
        error
            .to_string()
            .contains("does not match the pinned snapshot and schema"),
        "{error}"
    );
}

fn change_template(payload: &[u8], field: &str, value: Value) -> Vec<u8> {
    let mut template: Value = serde_json::from_slice(&payload[HEADER_LEN..]).unwrap();
    template[field] = value;
    let mut bytes = payload[..HEADER_LEN].to_vec();
    serde_json::to_writer(&mut bytes, &template).unwrap();
    bytes
}

fn assert_plan_error(error: &DataFusionError, needle: &str) {
    assert!(
        matches!(error, DataFusionError::Plan(message) if message.contains(needle)),
        "{error}"
    );
}

#[tokio::test]
async fn lake_scan_transport_rejects_invalid_headers_and_split_assignments() {
    let directory = tempfile::tempdir().unwrap();
    let uri = Url::from_file_path(directory.path().join("fixture.lance")).unwrap();
    write_lance(uri.as_str(), scalar_rows(0), WriteMode::Create).await;
    let table =
        LanceReadOnlyTable::open(uri.as_str(), None, &LakeTableOptions::default())
            .await
            .unwrap();
    let local = table
        .scan(&TestSession::default(), Some(&[0]), &[], None)
        .await
        .unwrap();
    let schema = local.schema();
    let (payload, splits) = transport(local.as_ref());
    let worker = TaskContext::default();
    assert_eq!(LAKE_SCAN_PROTOCOL_NAME, "df.lake.scan");
    assert!(payload.starts_with(MAGIC));
    assert_eq!(
        &payload[MAGIC.len()..HEADER_LEN],
        &LAKE_SCAN_PROTOCOL_VERSION.to_le_bytes()
    );

    let mut magic = payload.clone();
    magic[0] ^= 1;
    let mut version = payload.clone();
    version[MAGIC.len()] += 1;
    let mut oversized = payload.clone();
    oversized.resize(1024 * 1024 + 1, b' ');
    for (invalid, expected) in [
        (magic, "magic"),
        (version, "Unsupported lake scan version"),
        (oversized, "byte limit"),
        (
            change_template(
                &payload,
                "dataset_uri",
                json!("s3://bucket/path?token=example"),
            ),
            "without user info",
        ),
        (
            change_template(&payload, "dataset_version", json!(0)),
            "Invalid Lance dataset version",
        ),
        (
            change_template(&payload, "splits", json!([])),
            "Invalid lake scan template",
        ),
    ] {
        assert_plan_error(
            &rebuild_scan(&invalid, Arc::clone(&schema), &worker).unwrap_err(),
            expected,
        );
    }
    for (field, value) in [
        ("dataset_uri", json!("file:///other/table.lance")),
        ("dataset_version", json!(table.dataset_version() + 1)),
    ] {
        let changed = change_template(&payload, field, value);
        let template = rebuild_scan(&changed, Arc::clone(&schema), &worker).unwrap();
        let error = bind_split(
            file_config(template.as_ref()),
            decode_split(&splits[0]).unwrap(),
        )
        .unwrap_err();
        assert_plan_error(&error, "does not match the pinned dataset snapshot");
    }

    let split: Value = serde_json::from_slice(&splits[0]).unwrap();
    for (path, value, expected) in [
        (
            "/version",
            json!(2),
            "Unsupported lake split payload version",
        ),
        ("/split_id", json!("0".repeat(64)), "SHA-256"),
        ("/payload/fragment_id", json!(99), "SHA-256"),
    ] {
        let mut invalid = split.clone();
        *invalid.pointer_mut(path).unwrap() = value;
        let bytes = serde_json::to_vec(&invalid).unwrap();
        assert_plan_error(&decode_split(&bytes).unwrap_err(), expected);
    }
    let template = rebuild_scan(&payload, schema, &worker).unwrap();
    let file = decode_split(&splits[0]).unwrap();
    let mut invalid = file.extension::<LakeSplitExt>().unwrap().clone();
    invalid.split_id = "tampered".into();
    let error = bind_split(file_config(template.as_ref()), file.with_extension(invalid))
        .unwrap_err();
    assert_plan_error(&error, "SHA-256");
}
