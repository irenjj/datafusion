// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: See the NOTICE file distributed with this work for additional information regarding copyright ownership
// SPDX-FileContributor: Licensed to the Apache Software Foundation (ASF) under one or more contributor license agreements

use std::collections::HashSet;
use std::fs;
use std::num::NonZeroUsize;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::SystemTime;

use arrow::array::cast::AsArray;
use arrow::array::{
    Array, ArrayRef, FixedSizeListArray, Float32Array, Int32Array, RecordBatch,
    StringArray, StructArray,
};
use arrow::compute::{concat_batches, sort_to_indices, take_record_batch};
use arrow::datatypes::{DataType, Field, Float32Type, Int32Type, Schema};
use datafusion_ai::MediaRefType;
use datafusion_catalog::{Session, TableProvider};
use datafusion_datasource::file_scan_config::FileScanConfig;
use datafusion_datasource::source::DataSourceExec;
use datafusion_execution::{TaskContext, config::SessionConfig};
use datafusion_lake::backfill::{
    BackfillReceipt, BackfillTarget, finalize_backfill, prepare_backfill,
    write_backfill_unit,
};
use datafusion_lake::dist::{
    LakeRuntimeConfig, bind_split, decode_split, encode_scan, encode_splits, rebuild_scan,
};
use datafusion_lake::{
    LakeSplitExt, LakeSplitPayload, LakeTableOptions, LanceReadOnlyTable,
};
use datafusion_physical_plan::{ExecutionPlan, collect};
use futures::{TryFutureExt, future::try_join_all, stream};
use lance::dataset::builder::DatasetBuilder;
use lance::{BlobArrayBuilder, BlobFieldOptions, blob_field_with_options};
use lance_file::version::LanceFileVersion;
use lance_io::object_store::ObjectStoreParams;
use lance_table_runtime::Table;
use lance_table_runtime::write::{InsertBuilder, WriteParams};
use lance_testing::datafusion::TestSession;
use object_store::path::Path as ObjectPath;
use sha2::{Digest, Sha256};
use url::Url;

#[path = "readonly_scan/store.rs"]
mod store;

use store::Reads;

const IDS: [i32; 8] = [42, 7, 81, 15, 9, 61, 30, 4];

#[tokio::test]
async fn distributed_lance_scan_feeds_snapshot_backfill() {
    let directory = tempfile::tempdir().unwrap();
    let dataset_path = directory.path().join("source.lance");
    let uri = Url::from_file_path(&dataset_path).unwrap();
    let external_path = directory.path().join("external.blob");
    fs::write(&external_path, b"external payload").unwrap();
    let external_uri = Url::from_file_path(&external_path).unwrap();
    let baseline = write_fixture(&uri, &external_uri).await;
    assert_eq!(baseline.manifest().fragments.len(), 3);
    let version = baseline.version_id();

    let mut original_paths = Vec::new();
    list_files(&dataset_path.join("data"), &mut original_paths);
    assert_eq!(original_paths.len(), 7);
    original_paths.push(external_path);
    let payload_paths = original_paths
        .iter()
        .filter(|path| path.extension().is_some_and(|ext| ext == "blob"))
        .map(|path| ObjectPath::from_filesystem_path(path).unwrap())
        .collect::<HashSet<_>>();
    assert_eq!(payload_paths.len(), 5);
    let original_files = original_paths
        .into_iter()
        .map(|path| {
            let state = FileState::read(&path);
            (path, state)
        })
        .collect::<Vec<_>>();

    let reads = Reads::new();
    let original = observed_table(&uri, &reads).await;
    assert_eq!(original.dataset_version(), version);
    let coordinator = TestSession::default();
    let direct = original.scan(&coordinator, None, &[], None).await.unwrap();
    let schema = direct.schema();
    let scan = encode_scan(file_config(direct.as_ref())).unwrap().unwrap();
    let splits = encode_splits(file_config(direct.as_ref())).unwrap();
    assert_eq!(splits.len(), baseline.manifest().fragments.len());
    let direct_batches = collect(direct, coordinator.task_ctx()).await.unwrap();
    let direct_rows = sorted_by_id(&concat_batches(&schema, &direct_batches).unwrap());
    assert_zero_payload_bytes(&reads, &payload_paths);

    let opens = Arc::new(AtomicUsize::new(0));
    let mut worker_config = SessionConfig::new();
    LakeRuntimeConfig::default()
        .with_lance_builder({
            let opens = Arc::clone(&opens);
            let reads = reads.clone();
            move |uri| {
                opens.fetch_add(1, Ordering::SeqCst);
                Ok(DatasetBuilder::from_uri(uri)
                    .with_store_params(observed_params(&reads)))
            }
        })
        .register(&mut worker_config);
    let worker = Arc::new(TaskContext::new(
        None,
        "phase2-worker".into(),
        worker_config,
        Default::default(),
        Default::default(),
        Default::default(),
        Default::default(),
        Default::default(),
    ));
    let template = rebuild_scan(&scan, Arc::clone(&schema), &worker).unwrap();
    let config = file_config(template.as_ref());
    assert!(config.file_groups.is_empty());
    let bound = splits
        .iter()
        .map(|split| {
            let file = decode_split(split).unwrap();
            let LakeSplitPayload::Lance {
                fragment_id,
                row_ranges: None,
                deletion_file_rev: None,
                ..
            } = file.extension::<LakeSplitExt>().unwrap().payload
            else {
                panic!("backfill needs every physical row of a Lance fragment");
            };
            (fragment_id, bind_split(config, file).unwrap())
        })
        .collect::<Vec<_>>();
    assert_eq!(opens.load(Ordering::SeqCst), 0);
    assert!(reads.take().is_empty());

    let replay = try_join_all(bound.into_iter().map(|(fragment_id, plan)| {
        collect(plan, Arc::clone(&worker)).map_ok(move |batches| (fragment_id, batches))
    }))
    .await
    .unwrap();
    assert_eq!(opens.load(Ordering::SeqCst), 1);
    assert_zero_payload_bytes(&reads, &payload_paths);
    let rows =
        concat_batches(&schema, replay.iter().flat_map(|(_, batches)| batches)).unwrap();
    assert_eq!(
        rows["id"].as_primitive::<Int32Type>().values().as_ref(),
        &IDS
    );
    assert_eq!(sorted_by_id(&rows), direct_rows);
    assert_payload_refs(&rows, &payload_paths, &external_uri);

    let columns = derived_columns(&rows, "winner");
    let runtime = ObjectStoreParams::default();
    let (plan, units) = prepare_backfill(
        BackfillTarget::Lance(uri.as_str()),
        &columns.schema(),
        &runtime,
    )
    .await
    .unwrap();
    assert_eq!(plan.dataset_version(), version);
    assert_eq!(units.len(), replay.len());
    let mut offset = 0;
    let unit_batches = units
        .iter()
        .zip(&replay)
        .map(|(unit, (fragment_id, batches))| {
            assert_eq!(unit.fragment_id, *fragment_id);
            let row_count = usize::try_from(unit.row_count).unwrap();
            assert_eq!(
                row_count,
                batches.iter().map(RecordBatch::num_rows).sum::<usize>()
            );
            let batch = columns.slice(offset, row_count);
            offset += row_count;
            batch
        })
        .collect::<Vec<_>>();
    assert_eq!(offset, rows.num_rows());

    let first_batch = &unit_batches[0];
    let write = |unit_index: usize, attempt: &'static str, batch| {
        write_backfill_unit(
            &plan,
            &units[unit_index],
            attempt,
            stream::iter([Ok(batch)]),
            &runtime,
        )
    };
    let losing_batch = derived_columns(&rows.slice(0, first_batch.num_rows()), "loser");
    let (loser, winner) = tokio::join!(
        write(0, "attempt-0", losing_batch),
        write(0, "attempt-1", first_batch.clone()),
    );
    let loser = loser.unwrap();
    let loser_path = artifact_path(&loser);
    let loser_state = FileState::read(&loser_path);
    assert!(write(0, "attempt-0", first_batch.clone()).await.is_err());
    let mut winners = vec![winner.unwrap()];
    for (index, batch) in unit_batches.into_iter().enumerate().skip(1) {
        winners.push(write(index, "attempt-1", batch).await.unwrap());
    }
    assert!(
        winners
            .iter()
            .all(|winner| winner.artifact_uri != loser.artifact_uri)
    );
    assert_eq!(
        Table::open(uri.as_str()).await.unwrap().version_id(),
        version
    );

    assert_eq!(
        finalize_backfill(&plan, &winners, &runtime).await.unwrap(),
        version + 1
    );
    let updated = observed_table(&uri, &reads).await;
    assert_eq!(updated.dataset_version(), version + 1);
    let updated_scan = updated.scan(&coordinator, None, &[], None).await.unwrap();
    let updated_schema = updated_scan.schema();
    let updated_batches = collect(updated_scan, coordinator.task_ctx()).await.unwrap();
    let updated_rows =
        sorted_by_id(&concat_batches(&updated_schema, &updated_batches).unwrap());
    assert_eq!(updated_rows.project(&[0, 1, 2]).unwrap(), direct_rows);
    assert_eq!(
        updated_rows.project(&[3, 4, 5]).unwrap(),
        derived_columns(&direct_rows, "winner")
    );
    assert_zero_payload_bytes(&reads, &payload_paths);

    let committed = Table::open(uri.as_str()).await.unwrap();
    let fragments = &committed.manifest().fragments;
    assert_eq!(fragments.len(), winners.len());
    for ((before, after), winner) in baseline
        .manifest()
        .fragments
        .iter()
        .zip(fragments.iter())
        .zip(&winners)
    {
        assert_eq!(before.id, after.id);
        assert_eq!(after.id, u64::from(winner.work_unit.fragment_id));
        assert_eq!(after.files.len(), before.files.len() + 1);
        assert_eq!(&after.files[..before.files.len()], &before.files);
        assert_eq!(
            after.files.last().unwrap().path.as_str(),
            artifact_path(winner).file_name().unwrap()
        );
        assert!(
            after
                .files
                .iter()
                .all(|file| file.path.as_str() != loser_path.file_name().unwrap())
        );
    }
    assert_eq!(FileState::read(&loser_path), loser_state);
    for (path, state) in original_files {
        assert_eq!(FileState::read(&path), state, "{}", path.display());
    }
}

async fn write_fixture(uri: &Url, external_uri: &Url) -> Table {
    let mut payloads = BlobArrayBuilder::new(IDS.len());
    for (row, id) in IDS.into_iter().enumerate() {
        if row.is_multiple_of(2) {
            payloads.push_uri(external_uri.as_str()).unwrap();
        } else {
            payloads
                .push_bytes(vec![u8::try_from(id).unwrap(); 16 + row])
                .unwrap();
        }
    }
    let payload = blob_field_with_options(
        "payload",
        true,
        BlobFieldOptions::default()
            .with_inline_size_threshold(1)
            .with_dedicated_size_threshold(NonZeroUsize::new(8).unwrap()),
    );
    let batch = RecordBatch::try_new(
        Arc::new(Schema::new(vec![
            Field::new("id", DataType::Int32, false),
            Field::new("source_label", DataType::Utf8, true),
            payload,
        ])),
        vec![
            Arc::new(Int32Array::from_iter_values(IDS)),
            Arc::new(StringArray::from_iter(
                IDS.map(|id| (id % 2 == 0).then(|| format!("source-{id}"))),
            )),
            payloads.finish().unwrap(),
        ],
    )
    .unwrap();
    InsertBuilder::new(uri.as_str())
        .with_params(&WriteParams {
            max_rows_per_file: 3,
            data_storage_version: Some(LanceFileVersion::V2_2),
            allow_external_blob_outside_bases: true,
            ..Default::default()
        })
        .execute(vec![batch])
        .await
        .unwrap()
}

fn derived_columns(rows: &RecordBatch, attempt: &str) -> RecordBatch {
    let ids = rows["id"].as_primitive::<Int32Type>().values();
    let scores = ids
        .iter()
        .map(|id| f32::from(i16::try_from(*id).unwrap()) * 0.25);
    let embedding = FixedSizeListArray::from_iter_primitive::<Float32Type, _, _>(
        scores
            .clone()
            .map(|score| Some([Some(score), Some(-score), Some(score + 0.5)])),
        3,
    );
    let labels = rows["source_label"].as_string::<i32>();
    let annotation = StructArray::from(vec![
        (
            Arc::new(Field::new("score", DataType::Float32, false)),
            Arc::new(Float32Array::from_iter_values(scores)) as ArrayRef,
        ),
        (
            Arc::new(Field::new("label", DataType::Utf8, true)),
            Arc::new(StringArray::from_iter(
                labels
                    .iter()
                    .map(|label| label.map(|label| format!("derived-{label}"))),
            )) as ArrayRef,
        ),
    ]);
    RecordBatch::try_new(
        Arc::new(Schema::new(vec![
            Field::new("embedding", embedding.data_type().clone(), true),
            Field::new("annotation", annotation.data_type().clone(), true),
            Field::new("tag", DataType::Utf8, true),
        ])),
        vec![
            Arc::new(embedding),
            Arc::new(annotation),
            Arc::new(StringArray::from_iter_values(
                ids.iter().map(|id| format!("{attempt}-{id}")),
            )),
        ],
    )
    .unwrap()
}

fn file_config(plan: &dyn ExecutionPlan) -> &FileScanConfig {
    plan.downcast_ref::<DataSourceExec>()
        .unwrap()
        .data_source()
        .downcast_ref()
        .unwrap()
}

fn sorted_by_id(rows: &RecordBatch) -> RecordBatch {
    let indices = sort_to_indices(rows["id"].as_ref(), None, None).unwrap();
    take_record_batch(rows, &indices).unwrap()
}

fn observed_params(reads: &Reads) -> ObjectStoreParams {
    ObjectStoreParams {
        object_store_wrapper: Some(Arc::new(reads.clone())),
        ..Default::default()
    }
}

async fn observed_table(uri: &Url, reads: &Reads) -> LanceReadOnlyTable {
    LanceReadOnlyTable::open_with_builder(
        DatasetBuilder::from_uri(uri.as_str()).with_store_params(observed_params(reads)),
        &LakeTableOptions::default(),
    )
    .await
    .unwrap()
}

fn assert_payload_refs(rows: &RecordBatch, paths: &HashSet<ObjectPath>, external: &Url) {
    rows.schema()
        .field_with_name("payload")
        .unwrap()
        .try_extension_type::<MediaRefType>()
        .unwrap();
    let payloads = rows["payload"].as_struct();
    assert_eq!(payloads.null_count(), 0);
    let uris = payloads.column_by_name("uri").unwrap().as_string::<i32>();
    let observed_paths = uris
        .iter()
        .enumerate()
        .map(|(row, uri)| {
            let uri = Url::parse(uri.unwrap()).unwrap();
            assert_eq!(&uri == external, row.is_multiple_of(2));
            ObjectPath::from_filesystem_path(uri.to_file_path().unwrap()).unwrap()
        })
        .collect::<HashSet<_>>();
    assert_eq!(&observed_paths, paths);
}

fn assert_zero_payload_bytes(reads: &Reads, payload_paths: &HashSet<ObjectPath>) {
    let requests = reads.take();
    assert!(
        requests
            .iter()
            .any(|read| read.path.extension() == Some("lance") && read.range.is_some())
    );
    let payload_bytes = requests
        .iter()
        .filter(|read| payload_paths.contains(&read.path))
        .filter_map(|read| read.range.as_ref())
        .map(|range| range.end - range.start)
        .sum::<u64>();
    assert_eq!(payload_bytes, 0, "{requests:?}");
}

fn artifact_path(receipt: &BackfillReceipt) -> PathBuf {
    Url::parse(&receipt.artifact_uri)
        .unwrap()
        .to_file_path()
        .unwrap()
}

fn list_files(directory: &Path, files: &mut Vec<PathBuf>) {
    for entry in fs::read_dir(directory).unwrap() {
        let entry = entry.unwrap();
        if entry.file_type().unwrap().is_dir() {
            list_files(&entry.path(), files);
        } else {
            files.push(entry.path());
        }
    }
}

#[derive(Debug, PartialEq, Eq)]
struct FileState {
    bytes: Vec<u8>,
    modified: SystemTime,
    sha256: [u8; 32],
}

impl FileState {
    fn read(path: &Path) -> Self {
        let bytes = fs::read(path).unwrap();
        Self {
            modified: fs::metadata(path).unwrap().modified().unwrap(),
            sha256: Sha256::digest(&bytes).into(),
            bytes,
        }
    }
}
