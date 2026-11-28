// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: See the NOTICE file distributed with this work for additional information regarding copyright ownership
// SPDX-FileContributor: Licensed to the Apache Software Foundation (ASF) under one or more contributor license agreements

use std::collections::{BTreeMap, HashMap, HashSet};
use std::fs;
use std::num::NonZeroUsize;
use std::ops::Range;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::SystemTime;

use arrow::array::cast::AsArray;
use arrow::array::{
    Array, ArrayRef, BooleanArray, FixedSizeListArray, Int32Array, RecordBatch,
    StringArray, StructArray, new_null_array,
};
use arrow::datatypes::{
    DataType, Field, Float32Type, Int32Type, Metadata, Schema, UInt64Type,
};
use datafusion_ai::{AiErrorType, MediaRefType};
use datafusion_catalog::{Session, TableProvider};
use datafusion_common::DataFusionError;
use datafusion_lake::backfill::{
    BackfillPlan, BackfillReceipt, BackfillTarget, BackfillWorkUnit, finalize_backfill,
    prepare_backfill, write_backfill_unit,
};
use datafusion_lake::{LakeTableOptions, LanceReadOnlyTable};
use datafusion_physical_plan::collect;
use futures::{future::try_join_all, stream};
use lance::dataset::builder::DatasetBuilder;
use lance::dataset::transaction::{Operation, Transaction};
use lance::{BlobArrayBuilder, BlobFieldOptions, blob_field_with_options};
use lance_core::utils::deletion::DeletionVector;
use lance_file::version::LanceFileVersion;
use lance_io::object_store::{ObjectStore, ObjectStoreParams};
use lance_table::io::deletion::write_deletion_file;
use lance_table_runtime::write::{InsertBuilder, WriteMode, WriteParams};
use lance_table_runtime::{CommitBuilder, Table};
use lance_testing::datafusion::TestSession;
use object_store::path::Path as ObjectPath;
use serde::{Serialize, de::DeserializeOwned};
use sha2::{Digest, Sha256};
use url::Url;

#[path = "backfill/paimon.rs"]
mod paimon;
#[path = "readonly_scan/store.rs"]
mod store;

use store::Reads;

#[tokio::test]
async fn lance_backfill_attaches_only_selected_attempts_without_rewriting() {
    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("source with spaces.lance");
    let uri = path.to_str().unwrap();
    let baseline =
        delete_offset_one(&write_lance(uri, 0..8, WriteMode::Create).await, uri).await;
    let old_files = file_states(&path.join("data"));
    let blob_paths: HashSet<_> = old_files
        .keys()
        .filter(|path| path.extension().is_some_and(|ext| ext == "blob"))
        .map(|path| ObjectPath::from_filesystem_path(path).unwrap())
        .collect();
    assert_eq!(blob_paths.len(), 8);
    assert_eq!(old_files.len(), 11);

    let reads = Reads::new();
    let runtime = ObjectStoreParams {
        object_store_wrapper: Some(Arc::new(reads.clone())),
        ..Default::default()
    };
    let options = LakeTableOptions::default();
    let original = LanceReadOnlyTable::open_with_builder(
        DatasetBuilder::from_uri(uri).with_store_params(runtime.clone()),
        &options,
    )
    .await
    .unwrap();
    let original_rows = scan(&original).await;
    let original_rows = rows_by_id(&original_rows);
    assert_eq!(
        original_rows.keys().copied().collect::<Vec<_>>(),
        [0, 2, 3, 4, 5, 6, 7]
    );

    let columns = feature_values(0..1, "winner").schema();
    let (plan, units) = prepare_backfill(BackfillTarget::Lance(uri), &columns, &runtime)
        .await
        .unwrap();
    assert_eq!(plan.dataset_version(), baseline.version_id());
    assert_eq!(
        plan.dataset_uri(),
        Url::from_file_path(&path).unwrap().as_str()
    );
    assert_eq!(plan.new_columns(), columns.as_ref());
    assert_eq!(
        units.iter().map(|unit| unit.row_count).collect::<Vec<_>>(),
        [3, 3, 2]
    );
    let mut reordered = columns.metadata().iter().collect::<Vec<_>>();
    reordered.sort_unstable_by(|left, right| right.0.cmp(left.0));
    let reordered = Schema::new_with_metadata(
        columns.fields().clone(),
        reordered
            .into_iter()
            .map(|(k, v)| (k.clone(), v.clone()))
            .collect::<Metadata>(),
    );
    assert_eq!(
        plan.plan_fingerprint,
        prepare_backfill(BackfillTarget::Lance(uri), &reordered, &runtime)
            .await
            .unwrap()
            .0
            .plan_fingerprint
    );
    let plan = wire_roundtrip(&plan);
    let units = wire_roundtrip(&units);

    let loser = write_backfill_unit(
        &plan,
        &units[0],
        "attempt-0",
        stream::iter([Ok(feature_values(0..3, "loser"))]),
        &runtime,
    )
    .await
    .unwrap();
    let loser_before = file_state(&artifact_path(&loser));
    let same_attempt = write_backfill_unit(
        &plan,
        &units[0],
        "attempt-0",
        stream::iter([Ok(feature_values(0..3, "other"))]),
        &runtime,
    )
    .await;
    assert!(same_attempt.is_err());
    let mut winners = write_winners(&plan, &units, &runtime).await;
    winners.reverse();
    assert!(
        winners
            .iter()
            .all(|winner| winner.artifact_uri != loser.artifact_uri)
    );
    let before_commit = Table::open(uri).await.unwrap();
    assert_eq!(before_commit.version_id(), baseline.version_id());
    assert_eq!(before_commit.schema(), baseline.schema());
    for (path, state) in &old_files {
        assert_eq!(file_state(path), *state, "{}", path.display());
    }
    assert_eq!(
        finalize_backfill(&plan, &winners, &runtime).await.unwrap(),
        baseline.version_id() + 1
    );
    let updated = Table::open(uri).await.unwrap();
    assert_eq!(
        updated.manifest().fragments.len(),
        baseline.manifest().fragments.len()
    );
    for (before, after) in baseline
        .manifest()
        .fragments
        .iter()
        .zip(updated.manifest().fragments.iter())
    {
        let winner = winners
            .iter()
            .find(|r| u64::from(r.work_unit.fragment_id) == before.id)
            .unwrap();
        assert_eq!(before.id, after.id);
        assert_eq!(before.physical_rows, after.physical_rows);
        assert_eq!(before.deletion_file, after.deletion_file);
        assert_eq!(after.files.len(), before.files.len() + 1);
        assert_eq!(&after.files[..before.files.len()], &before.files);
        let artifact = after.files.last().unwrap();
        assert_eq!(
            artifact_path(winner).file_name().unwrap(),
            artifact.path.as_str()
        );
        let written_columns = Schema::from(&artifact.schema(updated.schema()));
        assert_eq!(written_columns.fields(), columns.fields());
    }
    for (path, state) in old_files {
        assert_eq!(file_state(&path), state, "{}", path.display());
    }
    assert_eq!(file_state(&artifact_path(&loser)), loser_before);

    reads.take();
    let snapshot = LanceReadOnlyTable::open_with_builder(
        DatasetBuilder::from_uri(uri).with_store_params(runtime),
        &options,
    )
    .await
    .unwrap();
    assert_eq!(snapshot.dataset_version(), baseline.version_id() + 1);
    let scanned = scan(&snapshot).await;
    let scanned_by_id = rows_by_id(&scanned);
    assert_eq!(
        scanned_by_id.keys().collect::<Vec<_>>(),
        original_rows.keys().collect::<Vec<_>>()
    );
    for (id, row) in scanned_by_id {
        let original = &original_rows[&id];
        assert_eq!(row["payload"].to_data(), original["payload"].to_data());
        let media = row["payload"].as_struct();
        assert_eq!(
            media.column(2).as_primitive::<UInt64Type>().value(0),
            blob_len(id) as u64
        );
        let expected = feature_values(id..id + 1, "winner");
        for (field, values) in expected.schema().fields().iter().zip(expected.columns()) {
            assert_eq!(
                row[field.name()].to_data(),
                values.to_data(),
                "row {id}, {}",
                field.name()
            );
        }
    }
    snapshot
        .schema()
        .field_with_name("payload")
        .unwrap()
        .try_extension_type::<MediaRefType>()
        .unwrap();
    let reads = reads.take();
    assert!(reads.iter().any(|read| read.range.is_some()));
    assert!(
        reads
            .iter()
            .all(|read| read.range.is_none() || !blob_paths.contains(&read.path)),
        "{reads:?}"
    );
}

#[tokio::test]
async fn lance_prepare_rejects_invalid_columns_and_credentialed_locations() {
    let directory = tempfile::tempdir().unwrap();
    let uri = directory.path().to_str().unwrap();
    let baseline = write_lance(uri, 0..8, WriteMode::Create).await;
    let runtime = ObjectStoreParams::default();
    let error_field = AiErrorType::new(DataType::Utf8)
        .field("result", true)
        .unwrap();
    for (columns, expected) in [
        (Schema::empty(), "nonempty"),
        (
            Schema::new(vec![Field::new("id", DataType::Int32, true)]),
            "already exists",
        ),
        (
            Schema::new(vec![Field::new("tag", DataType::Utf8, true); 2]),
            "duplicate",
        ),
        (
            Schema::new(vec![Field::new(
                "tag",
                DataType::ListView(Arc::new(Field::new("item", DataType::Utf8, true))),
                true,
            )]),
            "Unsupported",
        ),
        (
            Schema::new(vec![Field::new_struct(
                "annotation",
                vec![error_field],
                true,
            )]),
            "df.ai_error",
        ),
    ] {
        let error = prepare_backfill(BackfillTarget::Lance(uri), &columns, &runtime)
            .await
            .unwrap_err();
        assert!(error.to_string().contains(expected), "{error}");
    }
    let columns = feature_values(0..1, "winner").schema();
    let error = prepare_backfill(
        BackfillTarget::Lance("s3://example/data?token=example-secret"),
        &columns,
        &runtime,
    )
    .await
    .unwrap_err();
    assert!(error.to_string().contains("credential-free"));
    assert!(!error.to_string().contains("example-secret"));
    assert_eq!(
        Table::open(uri).await.unwrap().version_id(),
        baseline.version_id()
    );
}

#[tokio::test]
async fn lance_rejects_bad_batches_and_receipt_sets_before_publication() {
    let directory = tempfile::tempdir().unwrap();
    let path = directory.path();
    let uri = path.to_str().unwrap();
    let baseline = write_lance(uri, 0..8, WriteMode::Create).await;
    let runtime = ObjectStoreParams::default();
    let batch = feature_values(0..3, "winner");
    let (plan, units) =
        prepare_backfill(BackfillTarget::Lance(uri), &batch.schema(), &runtime)
            .await
            .unwrap();
    let data_before = file_states(&path.join("data"));
    let error_field = AiErrorType::new(DataType::Utf8).field("tag", true).unwrap();
    let ai_error = RecordBatch::try_new(
        Arc::new(Schema::new(vec![error_field.clone()])),
        vec![new_null_array(error_field.data_type(), 2)],
    )
    .unwrap();
    let missing_metadata = RecordBatch::try_new(
        Arc::new(Schema::new(batch.schema().fields().clone())),
        batch.columns().to_vec(),
    )
    .unwrap();
    for (batches, expected) in [
        (vec![batch.slice(0, 2)], "Expected 3 backfill rows"),
        (
            vec![batch.clone(), batch.slice(0, 1)],
            "exceeds 3 physical rows",
        ),
        (vec![missing_metadata], "batch schema"),
        (vec![batch.slice(0, 1), ai_error], "batch schema"),
    ] {
        let error = write_backfill_unit(
            &plan,
            &units[0],
            "invalid",
            stream::iter(batches.into_iter().map(Ok)),
            &runtime,
        )
        .await
        .unwrap_err();
        assert!(
            matches!(error, DataFusionError::Execution(ref message) if message.contains(expected)),
            "{error}"
        );
    }
    let wrong_unit = BackfillWorkUnit {
        row_count: 2,
        ..units[0]
    };
    let error = write_backfill_unit(
        &plan,
        &wrong_unit,
        "invalid",
        stream::iter([Ok(batch.clone())]),
        &runtime,
    )
    .await
    .unwrap_err();
    assert!(error.to_string().contains("pinned fragment"), "{error}");
    let mut bad_plan = plan.clone();
    bad_plan.plan_fingerprint.replace_range(..1, "!");
    let error = write_backfill_unit(
        &bad_plan,
        &units[0],
        "invalid",
        stream::iter([Ok(batch)]),
        &runtime,
    )
    .await
    .unwrap_err();
    assert!(error.to_string().contains("fingerprint"), "{error}");
    assert_eq!(file_states(&path.join("data")), data_before);

    let receipts = write_winners(&plan, &units, &runtime).await;
    let mut duplicate = receipts.clone();
    duplicate[2] = receipts[0].clone();
    let mut wrong_fingerprint = receipts.clone();
    wrong_fingerprint[0].plan_fingerprint = bad_plan.plan_fingerprint.clone();
    let mut wrong_rows = receipts.clone();
    wrong_rows[0].row_count += 1;
    let mut wrong_total = wrong_rows.clone();
    wrong_total[0].work_unit.row_count += 1;
    let mut wrong_fragment = receipts.clone();
    wrong_fragment[0].work_unit.fragment_id = 999;
    let mut wrong_artifact = receipts.clone();
    wrong_artifact[0].artifact_uri = receipts[1].artifact_uri.clone();
    for (invalid, expected) in [
        (&receipts[..2], "Expected 3 backfill receipts"),
        (&duplicate, "Duplicate"),
        (&wrong_fingerprint, "fingerprint"),
        (&wrong_rows, "does not match its work unit"),
        (&wrong_total, "Expected 8 backfill rows"),
        (&wrong_fragment, "Missing backfill receipt"),
        (&wrong_artifact, "Invalid backfill data file"),
    ] {
        let error = finalize_backfill(&plan, invalid, &runtime)
            .await
            .unwrap_err();
        assert!(error.to_string().contains(expected), "{error}");
    }
    let missing_file = artifact_path(&receipts[0]);
    let offline = missing_file.with_extension("offline");
    fs::rename(&missing_file, &offline).unwrap();
    assert!(finalize_backfill(&plan, &receipts, &runtime).await.is_err());
    fs::rename(offline, missing_file).unwrap();
    let latest = Table::open(uri).await.unwrap();
    assert_eq!(latest.version_id(), baseline.version_id());
    assert_eq!(latest.schema(), baseline.schema());
}

#[tokio::test]
async fn lance_finalize_does_not_rebase_on_an_intervening_append() {
    let directory = tempfile::tempdir().unwrap();
    let uri = directory.path().to_str().unwrap();
    write_lance(uri, 0..8, WriteMode::Create).await;
    let runtime = ObjectStoreParams::default();
    let (plan, units) = prepare_backfill(
        BackfillTarget::Lance(uri),
        &feature_values(0..1, "winner").schema(),
        &runtime,
    )
    .await
    .unwrap();
    let receipts = write_winners(&plan, &units, &runtime).await;
    let appended = write_lance(uri, 8..11, WriteMode::Append).await;
    let error = finalize_backfill(&plan, &receipts, &runtime)
        .await
        .unwrap_err();
    assert!(error.to_string().contains("stale"), "{error}");
    let latest = Table::open(uri).await.unwrap();
    assert_eq!(latest.version_id(), appended.version_id());
    assert_eq!(latest.schema(), appended.schema());
}

#[tokio::test]
async fn lance_only_one_finalize_can_publish_at_the_pinned_version() {
    let directory = tempfile::tempdir().unwrap();
    let uri = directory.path().to_str().unwrap();
    write_lance(uri, 0..8, WriteMode::Create).await;
    let runtime = ObjectStoreParams::default();
    let (plan, units) = prepare_backfill(
        BackfillTarget::Lance(uri),
        &feature_values(0..1, "winner").schema(),
        &runtime,
    )
    .await
    .unwrap();
    let receipts = write_winners(&plan, &units, &runtime).await;
    let (left, right) = tokio::join!(
        finalize_backfill(&plan, &receipts, &runtime),
        finalize_backfill(&plan, &receipts, &runtime)
    );
    let expected = plan.dataset_version() + 1;
    assert_eq!(
        [left, right].into_iter().flatten().collect::<Vec<_>>(),
        [expected]
    );
    assert_eq!(Table::open(uri).await.unwrap().version_id(), expected);
}

#[tokio::test]
async fn lance_backfills_other_existing_file_formats() {
    let runtime = ObjectStoreParams::default();
    let options = LakeTableOptions::default();
    let expected = feature_values(0..2, "winner");
    for format in [
        LanceFileVersion::Legacy,
        LanceFileVersion::V2_0,
        LanceFileVersion::V2_1,
        LanceFileVersion::V2_3,
    ] {
        let directory = tempfile::tempdir().unwrap();
        let uri = directory.path().to_str().unwrap();
        let batch = RecordBatch::try_new(
            Arc::new(Schema::new(vec![Field::new("id", DataType::Int32, false)])),
            vec![Arc::new(Int32Array::from(vec![0, 1]))],
        )
        .unwrap();
        let baseline = InsertBuilder::new(uri)
            .with_params(&WriteParams::with_storage_version(format))
            .execute(vec![batch])
            .await
            .unwrap();
        let prepared =
            prepare_backfill(BackfillTarget::Lance(uri), &expected.schema(), &runtime)
                .await;
        if format == LanceFileVersion::Legacy {
            assert!(matches!(
                prepared,
                Err(DataFusionError::NotImplemented(message)) if message.contains("Legacy V1")
            ));
            assert_eq!(
                Table::open(uri).await.unwrap().version_id(),
                baseline.version_id()
            );
            continue;
        }
        let (plan, units) = prepared.unwrap();
        let receipts = write_winners(&plan, &units, &runtime).await;
        assert_eq!(
            finalize_backfill(&plan, &receipts, &runtime).await.unwrap(),
            baseline.version_id() + 1,
            "{format:?}"
        );
        let snapshot = LanceReadOnlyTable::open(uri, None, &options).await.unwrap();
        let scanned = rows_by_id(&scan(&snapshot).await);
        assert_eq!(scanned.keys().copied().collect::<Vec<_>>(), [0, 1]);
        for (id, columns) in scanned {
            let expected = expected.slice(usize::try_from(id).unwrap(), 1);
            for (field, values) in
                expected.schema().fields().iter().zip(expected.columns())
            {
                assert_eq!(
                    columns[field.name()].to_data(),
                    values.to_data(),
                    "{format:?}, row {id}, {}",
                    field.name(),
                );
            }
        }
    }
}

async fn write_lance(uri: &str, ids: Range<i32>, mode: WriteMode) -> Table {
    let field = blob_field_with_options(
        "payload",
        true,
        BlobFieldOptions::default()
            .with_inline_size_threshold(1)
            .with_dedicated_size_threshold(NonZeroUsize::new(8).unwrap()),
    );
    let mut blob = BlobArrayBuilder::new(ids.len());
    for id in ids.clone() {
        blob.push_bytes(vec![u8::try_from(id).unwrap(); blob_len(id)])
            .unwrap();
    }
    let batch = RecordBatch::try_new(
        Arc::new(Schema::new(vec![
            Field::new("id", DataType::Int32, false),
            field,
        ])),
        vec![
            Arc::new(Int32Array::from_iter_values(ids)),
            blob.finish().unwrap(),
        ],
    )
    .unwrap();
    InsertBuilder::new(uri)
        .with_params(&WriteParams {
            max_rows_per_file: 3,
            mode,
            data_storage_version: Some(LanceFileVersion::V2_2),
            ..Default::default()
        })
        .execute(vec![batch])
        .await
        .unwrap()
}

async fn delete_offset_one(table: &Table, uri: &str) -> Table {
    let (store, root) = ObjectStore::from_uri(uri).await.unwrap();
    let mut first = table.manifest().fragments[0].clone();
    first.deletion_file = write_deletion_file(
        &root,
        first.id,
        table.version_id(),
        &DeletionVector::from_iter([1]),
        &store,
        0,
    )
    .await
    .unwrap();
    CommitBuilder::new(table)
        .execute(Transaction::new(
            table.version_id(),
            Operation::Delete {
                updated_fragments: vec![first],
                deleted_fragment_ids: vec![],
                predicate: "id = 1".into(),
            },
            None,
        ))
        .await
        .unwrap()
}

fn blob_len(id: i32) -> usize {
    12 + usize::try_from(id).unwrap()
}

async fn write_winners(
    plan: &BackfillPlan,
    units: &[BackfillWorkUnit],
    runtime: &ObjectStoreParams,
) -> Vec<BackfillReceipt> {
    try_join_all(units.iter().map(|unit| async move {
        let start = i32::try_from(unit.fragment_id).unwrap() * 3;
        let end = start + i32::try_from(unit.row_count).unwrap();
        let batch = feature_values(start..end, "winner");
        write_backfill_unit(
            plan,
            unit,
            "attempt-1",
            stream::iter([
                Ok(batch.slice(0, 1)),
                Ok(batch.slice(1, batch.num_rows() - 1)),
            ]),
            runtime,
        )
        .await
        .map(|receipt| wire_roundtrip(&receipt))
    }))
    .await
    .unwrap()
}

fn feature_values(ids: Range<i32>, label: &str) -> RecordBatch {
    let vector = FixedSizeListArray::from_iter_primitive::<Float32Type, _, _>(
        ids.clone().map(|id| {
            let value = f32::from(i16::try_from(id).unwrap());
            Some([Some(value), Some(value + 0.5)])
        }),
        2,
    );
    let details = StructArray::from(vec![
        (
            Arc::new(Field::new("caption", DataType::Utf8, true)),
            Arc::new(StringArray::from_iter_values(
                ids.clone().map(|id| format!("{label}:{id}")),
            )) as ArrayRef,
        ),
        (
            Arc::new(Field::new("score", DataType::Int32, true)),
            Arc::new(Int32Array::from_iter(
                ids.clone().map(|id| (id % 2 == 0).then_some(id * 10)),
            )) as ArrayRef,
        ),
    ]);
    let annotation = StructArray::from(vec![
        (
            Arc::new(Field::new("details", details.data_type().clone(), true)),
            Arc::new(details) as ArrayRef,
        ),
        (
            Arc::new(Field::new("accepted", DataType::Boolean, false)),
            Arc::new(BooleanArray::from_iter(
                ids.clone().map(|id| Some(id % 2 == 0)),
            )) as ArrayRef,
        ),
    ]);
    let tag = StringArray::from_iter(
        ids.map(|id| (id % 2 == 1).then(|| format!("{label}-{id}"))),
    );
    RecordBatch::try_new(
        Arc::new(Schema::new_with_metadata(
            vec![
                Field::new("embedding", vector.data_type().clone(), true),
                Field::new("annotation", annotation.data_type().clone(), true),
                Field::new("tag", DataType::Utf8, true),
            ],
            HashMap::from([
                ("generator".into(), "fixture".into()),
                ("format".into(), "v1".into()),
            ]),
        )),
        vec![Arc::new(vector), Arc::new(annotation), Arc::new(tag)],
    )
    .unwrap()
}

async fn scan(table: &dyn TableProvider) -> Vec<RecordBatch> {
    let session = TestSession::default();
    let plan = table.scan(&session, None, &[], None).await.unwrap();
    collect(plan, session.task_ctx()).await.unwrap()
}

fn rows_by_id(batches: &[RecordBatch]) -> BTreeMap<i32, HashMap<String, ArrayRef>> {
    let mut result = BTreeMap::new();
    for batch in batches {
        for (row, id) in batch["id"]
            .as_primitive::<Int32Type>()
            .values()
            .iter()
            .enumerate()
        {
            let values = batch
                .schema()
                .fields()
                .iter()
                .zip(batch.columns())
                .map(|(field, values)| (field.name().clone(), values.slice(row, 1)))
                .collect();
            assert!(result.insert(*id, values).is_none(), "duplicate row {id}");
        }
    }
    result
}

fn artifact_path(receipt: &BackfillReceipt) -> PathBuf {
    Url::parse(&receipt.artifact_uri)
        .unwrap()
        .to_file_path()
        .unwrap()
}

#[derive(Debug, PartialEq, Eq)]
struct FileState {
    size: u64,
    modified: SystemTime,
    sha256: String,
}

fn file_state(path: &Path) -> FileState {
    let meta = fs::metadata(path).unwrap();
    FileState {
        size: meta.len(),
        modified: meta.modified().unwrap(),
        sha256: hex::encode(Sha256::digest(fs::read(path).unwrap())),
    }
}

fn file_states(dir: &Path) -> BTreeMap<PathBuf, FileState> {
    let mut files = BTreeMap::new();
    for entry in fs::read_dir(dir).unwrap() {
        let path = entry.unwrap().path();
        if path.is_dir() {
            files.extend(file_states(&path));
        } else {
            files.insert(path.clone(), file_state(&path));
        }
    }
    files
}

fn wire_roundtrip<T: Serialize + DeserializeOwned>(value: &T) -> T {
    serde_json::from_slice(&serde_json::to_vec(value).unwrap()).unwrap()
}
