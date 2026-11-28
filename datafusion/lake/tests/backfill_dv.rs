// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: See the NOTICE file distributed with this work for additional information regarding copyright ownership
// SPDX-FileContributor: Licensed to the Apache Software Foundation (ASF) under one or more contributor license agreements

use std::collections::{BTreeMap, HashMap};
use std::fs;
use std::future::ready;
use std::num::NonZeroUsize;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::SystemTime;

use arrow::array::AsArray;
use arrow::array::{
    Array, FixedSizeListArray, Float32Array, Int32Array, RecordBatch, StringArray,
    StructArray,
};
use arrow::datatypes::{
    DataType, Field, Float32Type, Int32Type, Schema, SchemaRef, UInt64Type,
};
use datafusion_catalog::TableProvider;
use datafusion_common::{DataFusionError, Result};
use datafusion_datasource::file_scan_config::FileScanConfig;
use datafusion_datasource::source::DataSourceExec;
use datafusion_execution::SendableRecordBatchStream;
use datafusion_lake::backfill::{
    BackfillPlan, BackfillReceipt, BackfillTarget, bridge_backfill_stage,
    finalize_backfill, prepare_backfill, write_backfill_unit,
};
use datafusion_lake::dist::encode_splits;
use datafusion_lake::{LakeTableOptions, LanceReadOnlyTable};
use datafusion_physical_plan::stream::RecordBatchStreamAdapter;
use futures::{TryStreamExt, future::BoxFuture, stream};
use lance::dataset::Dataset;
use lance::dataset::transaction::{Operation, Transaction};
use lance::table::format::Fragment;
use lance::{BlobArrayBuilder, BlobFieldOptions, blob_field_with_options};
use lance_file::version::LanceFileVersion;
use lance_io::object_store::ObjectStoreParams;
use lance_table_runtime::write::{InsertBuilder, WriteParams};
use lance_table_runtime::{CommitBuilder, FileFragment, FragReadConfig, Table};
use lance_testing::datafusion::TestSession;
use paimon::arrow::{
    arrow_fields_to_paimon, build_read_arrow_schema, build_target_arrow_schema,
};
use paimon::catalog::Identifier;
use paimon::spec::{Schema as PaimonSchema, SchemaBuilder};
use paimon::{Catalog, CatalogOptions, DataSplit, FileSystemCatalog, Options};
use sha2::{Digest, Sha256};
use url::Url;

const IDS: [[i32; 6]; 3] = [
    [51, 3, 71, 11, 9, 81],
    [5, 12, 50, 2, 61, 45],
    [6, 17, 22, 34, 16, 14],
];
const DELETED: [i32; 7] = [51, 71, 11, 81, 50, 61, 45];
const SPLIT_IDS: [u64; 3] = [50, 9, 101];

#[tokio::test]
async fn parquet_winners_backfill_deleted_fragments_in_physical_order() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("source with space.lance");
    let uri = path.to_str().unwrap();
    let source = write_source(uri).await;
    let old_splits = source_splits(uri).await;
    let baseline = delete_rows(&source).await;
    let splits = source_splits(uri).await;
    assert_eq!(splits.len(), IDS.len());
    assert!(baseline.manifest().fragments[0].deletion_file.is_some());
    assert!(baseline.manifest().fragments[1].deletion_file.is_some());
    assert!(baseline.manifest().fragments[2].deletion_file.is_none());
    let mut original_files = file_states(&path.join("data"));
    let deletion_files = file_states(&path.join("_deletions"));
    assert_eq!(deletion_files.len(), 2);
    original_files.extend(deletion_files);
    assert_eq!(
        original_files
            .keys()
            .filter(|path| path.extension().unwrap() == "blob")
            .count(),
        IDS.len() * IDS[0].len(),
    );

    let stage_schema = fixture_stage_schema();
    let writer_schema = build_target_arrow_schema(stage_schema.fields()).unwrap();
    let output_schema = build_read_arrow_schema(stage_schema.fields()).unwrap();
    let runtime = ObjectStoreParams::default();
    let (plan, units) =
        prepare_backfill(BackfillTarget::Lance(uri), &output_schema, &runtime)
            .await
            .unwrap();
    let plan = BackfillPlan::from_json_bytes(&plan.to_json_bytes().unwrap()).unwrap();
    assert_eq!(
        units.iter().map(|unit| unit.row_count).collect::<Vec<_>>(),
        [6, 6, 6]
    );

    let stage_catalog = stage_catalog(&dir.path().join("stage")).await;
    let mut selected = Vec::new();
    for (index, (split_id, ids)) in SPLIT_IDS.into_iter().zip(IDS).enumerate() {
        let ids = ids
            .into_iter()
            .filter(|id| !DELETED.contains(id))
            .collect::<Vec<_>>();
        let batch = features(&ids, Arc::clone(&writer_schema), "winner");
        let file = write_parquet_winner(
            &stage_catalog,
            &format!("winner_{index}"),
            &stage_schema,
            &batch,
        )
        .await;
        selected.push((split_id, file));
    }
    selected.reverse();

    let bind = |mapping: &[Vec<u8>], winners| {
        bridge_backfill_stage(
            &plan,
            SPLIT_IDS.into_iter().zip(mapping.iter().map(Vec::as_slice)),
            winners,
            open_parquet,
        )
    };
    let invalid = |mapping: &[Vec<u8>], winners, needle: &str| {
        let err = bind(mapping, winners).err().unwrap();
        assert!(err.to_string().contains(needle), "{err}");
    };
    invalid(&splits[1..], selected.clone(), "Missing stage splits");
    invalid(&splits, selected[1..].to_vec(), "Missing winning Parquet");
    let mut duplicate = selected.clone();
    duplicate[1] = selected[0].clone();
    invalid(&splits, duplicate, "Duplicate winning Parquet");
    let mut unknown = selected.clone();
    unknown[1].0 = 1000;
    invalid(&splits, unknown, "Unknown backfill stage split");
    let mut duplicate_fragment = splits.clone();
    duplicate_fragment[1] = splits[0].clone();
    invalid(
        &duplicate_fragment,
        selected.clone(),
        "Duplicate backfill split",
    );
    invalid(
        &old_splits,
        selected.clone(),
        "does not match the pinned Lance fragment",
    );

    let expected = features(&[5, 12, 2], Arc::clone(&writer_schema), "winner");
    for (label, batch, expected_error) in [
        ("short", expected.slice(0, 2), "Too few logical rows"),
        (
            "long",
            features(&[5, 12, 2, 2], Arc::clone(&writer_schema), "winner"),
            "Too many logical rows",
        ),
    ] {
        let mut bad_winners = selected.clone();
        bad_winners[1].1 =
            write_parquet_winner(&stage_catalog, label, &stage_schema, &batch).await;
        let (unit, batches) = bind(&splits, bad_winners).unwrap().remove(1);
        let err = write_backfill_unit(&plan, &unit, label, batches, &runtime)
            .await
            .unwrap_err();
        assert!(err.to_string().contains(expected_error), "{err}");
    }
    let wrong_schema = PaimonSchema::builder()
        .column(
            "tag",
            paimon::spec::DataType::Int(paimon::spec::IntType::new()),
        )
        .option("file.format", "parquet")
        .build()
        .unwrap();
    let wrong_batch = RecordBatch::try_new(
        build_target_arrow_schema(wrong_schema.fields()).unwrap(),
        vec![Arc::new(Int32Array::from(vec![3, 9]))],
    )
    .unwrap();
    let mut bad_winners = selected.clone();
    bad_winners[2].1 =
        write_parquet_winner(&stage_catalog, "wrong_schema", &wrong_schema, &wrong_batch)
            .await;
    let (unit, batches) = bind(&splits, bad_winners).unwrap().remove(0);
    let err = write_backfill_unit(&plan, &unit, "wrong-schema", batches, &runtime)
        .await
        .unwrap_err();
    assert!(err.to_string().contains("Parquet schema differs"), "{err}");
    assert_eq!(
        Table::open(uri).await.unwrap().version_id(),
        baseline.version_id()
    );
    assert_eq!(
        file_states(&path.join("data")).len() + 2,
        original_files.len()
    );

    let losing_batch = features(&[3, 9], Arc::clone(&writer_schema), "loser");
    let mut alternative = selected.clone();
    alternative[2].1 =
        write_parquet_winner(&stage_catalog, "loser", &stage_schema, &losing_batch).await;
    let (losing_unit, losing_stream) = bind(&splits, alternative).unwrap().remove(0);
    let mut work = bind(&splits, selected).unwrap();
    let (winning_unit, winning_stream) = work.remove(0);
    assert_eq!(losing_unit, winning_unit);
    let (loser, winner) = tokio::join!(
        write_backfill_unit(&plan, &losing_unit, "attempt-0", losing_stream, &runtime),
        write_backfill_unit(&plan, &winning_unit, "attempt-1", winning_stream, &runtime),
    );
    let loser = loser.unwrap();
    let loser_path = artifact_path(&loser);
    let loser_state = file_state(&loser_path);
    let mut winners = vec![winner.unwrap()];
    for (unit, batches) in work {
        winners.push(
            write_backfill_unit(&plan, &unit, "attempt-1", batches, &runtime)
                .await
                .unwrap(),
        );
    }
    let winners = winners
        .iter()
        .map(|winner| {
            BackfillReceipt::from_json_bytes(&winner.to_json_bytes(&plan).unwrap(), &plan)
                .unwrap()
        })
        .collect::<Vec<_>>();
    assert_eq!(
        finalize_backfill(&plan, &winners, &runtime).await.unwrap(),
        baseline.version_id() + 1
    );

    let updated = Arc::new(Table::open(uri).await.unwrap());
    for ((before, after), winner) in baseline
        .manifest()
        .fragments
        .iter()
        .zip(updated.manifest().fragments.iter())
        .zip(&winners)
    {
        assert_eq!(after.id, before.id);
        assert_eq!(after.physical_rows, before.physical_rows);
        assert_eq!(after.deletion_file, before.deletion_file);
        assert_eq!(after.files.len(), before.files.len() + 1);
        assert_eq!(&after.files[..before.files.len()], &before.files);
        let file = after.files.last().unwrap();
        assert_eq!(
            artifact_path(winner).file_name().unwrap(),
            file.path.as_str()
        );
        assert_ne!(loser_path.file_name().unwrap(), file.path.as_str());

        let mut physical_only = Fragment::new(after.id);
        physical_only.files.push(file.clone());
        let physical_only = FileFragment::new(Arc::clone(&updated), physical_only);
        let ids = IDS[usize::try_from(after.id).unwrap()];
        let offsets = (0..u32::try_from(ids.len()).unwrap()).collect::<Vec<_>>();
        let physical = physical_only
            .take_rows(
                &offsets,
                &file.schema(updated.schema()),
                FragReadConfig::default(),
            )
            .await
            .unwrap();
        assert_eq!(physical.num_rows(), ids.len());
        let expected = features(&ids, Arc::clone(&output_schema), "winner");
        for (row, id) in ids.into_iter().enumerate() {
            if DELETED.contains(&id) {
                for column in physical.columns() {
                    assert!(
                        column.is_null(row),
                        "expected NULL at fragment {}, row {row}",
                        after.id
                    );
                }
            } else {
                assert_feature_values(&physical.slice(row, 1), &expected.slice(row, 1));
            }
        }
    }

    let updated = Arc::new(Dataset::open(uri).await.unwrap());
    let batch = updated
        .scan()
        .project(&["id", "old_label", "embedding", "annotation", "tag"])
        .unwrap()
        .scan_in_order(true)
        .with_row_address()
        .try_into_batch()
        .await
        .unwrap();
    let visible = IDS
        .into_iter()
        .flatten()
        .filter(|id| !DELETED.contains(id))
        .collect::<Vec<_>>();
    assert_eq!(
        batch["id"].as_primitive::<Int32Type>().values().as_ref(),
        visible
    );
    let expected = features(&visible, output_schema, "winner");
    assert_feature_values(&batch, &expected);
    let addresses = batch["_rowaddr"].as_primitive::<UInt64Type>().values();
    let expected_addresses = IDS
        .into_iter()
        .enumerate()
        .flat_map(|(fragment, ids)| {
            ids.into_iter().enumerate().filter_map(move |(offset, id)| {
                (!DELETED.contains(&id))
                    .then_some(((fragment as u64) << 32) | offset as u64)
            })
        })
        .collect::<Vec<_>>();
    assert_eq!(addresses.as_ref(), expected_addresses);
    let blobs = updated
        .take_blobs_by_addresses(addresses, "payload")
        .await
        .unwrap();
    assert_eq!(blobs.len(), visible.len());
    for (row, (id, blob)) in visible.into_iter().zip(blobs).enumerate() {
        assert_eq!(
            batch["old_label"].as_string::<i32>().value(row),
            format!("old-{id}")
        );
        assert_eq!(
            blob.unwrap().read().await.unwrap().as_ref(),
            &blob_bytes(id)
        );
    }
    assert_eq!(updated.count_deleted_rows().await.unwrap(), DELETED.len());
    for (path, state) in original_files {
        assert_eq!(file_state(&path), state, "{}", path.display());
    }
    assert_eq!(file_state(&loser_path), loser_state);
}

#[tokio::test]
async fn stage_bridge_shares_live_buffers_and_pads_non_nullable_fields() {
    let dir = tempfile::tempdir().unwrap();
    let uri = dir.path().to_str().unwrap();
    delete_rows(&write_source(uri).await).await;
    let splits = source_splits(uri).await;
    let schema = Arc::new(Schema::new(vec![
        Field::new(
            "embedding",
            DataType::FixedSizeList(
                Arc::new(Field::new("item", DataType::Float32, false)),
                2,
            ),
            false,
        ),
        Field::new(
            "annotation",
            DataType::Struct(
                vec![
                    Field::new("score", DataType::Int32, false),
                    Field::new("caption", DataType::Utf8, true),
                ]
                .into(),
            ),
            false,
        ),
        Field::new("tag", DataType::Utf8, true),
    ]));
    let (plan, _) = prepare_backfill(
        BackfillTarget::Lance(uri),
        &schema,
        &ObjectStoreParams::default(),
    )
    .await
    .unwrap();
    let logical = IDS.map(|ids| {
        let ids = ids
            .into_iter()
            .filter(|id| !DELETED.contains(id))
            .collect::<Vec<_>>();
        features(&ids, Arc::clone(&schema), "winner")
    });
    let mut work = bridge_backfill_stage(
        &plan,
        SPLIT_IDS.into_iter().zip(splits.iter().map(Vec::as_slice)),
        SPLIT_IDS.into_iter().zip(logical.iter().cloned()),
        |batch: RecordBatch| {
            let stream = stream::iter([Ok(batch.clone())]);
            ready(Ok(
                Box::pin(RecordBatchStreamAdapter::new(batch.schema(), stream))
                    as SendableRecordBatchStream,
            ))
        },
    )
    .unwrap();
    let padded: Vec<_> = work.remove(0).1.try_collect().await.unwrap();
    let zeros = padded[0]["embedding"]
        .as_fixed_size_list()
        .values()
        .as_primitive::<Float32Type>();
    assert_eq!(zeros.values().as_ref(), &[0.0, 0.0]);
    let partial = padded[0]["annotation"].as_struct();
    assert_eq!(partial.null_count(), 0);
    assert_eq!(partial.column(0).as_primitive::<Int32Type>().value(0), 0);
    assert!(partial.column(1).is_null(0));
    assert!(padded[0]["tag"].is_null(0));
    let split_live = padded[1]["embedding"].as_fixed_size_list();
    let supplied = logical[0]["embedding"].as_fixed_size_list();
    let buffer = |array: &FixedSizeListArray| {
        array
            .values()
            .as_primitive::<Float32Type>()
            .values()
            .as_ptr()
    };
    assert_eq!(buffer(split_live), buffer(supplied));
    let untouched: Vec<_> = work.pop().unwrap().1.try_collect().await.unwrap();
    assert!(Arc::ptr_eq(untouched[0].column(0), logical[2].column(0)));
}

async fn source_splits(uri: &str) -> Vec<Vec<u8>> {
    let table = LanceReadOnlyTable::open(uri, None, &LakeTableOptions::default())
        .await
        .unwrap();
    let scan = table
        .scan(&TestSession::default(), None, &[], None)
        .await
        .unwrap();
    let config = scan
        .downcast_ref::<DataSourceExec>()
        .unwrap()
        .data_source()
        .downcast_ref::<FileScanConfig>()
        .unwrap();
    encode_splits(config).unwrap()
}

async fn write_source(uri: &str) -> Table {
    let mut payloads = BlobArrayBuilder::new(IDS.len() * IDS[0].len());
    for id in IDS.into_iter().flatten() {
        payloads.push_bytes(blob_bytes(id)).unwrap();
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
            Field::new("old_label", DataType::Utf8, false),
            payload,
        ])),
        vec![
            Arc::new(Int32Array::from_iter_values(IDS.into_iter().flatten())),
            Arc::new(StringArray::from_iter_values(
                IDS.into_iter().flatten().map(|id| format!("old-{id}")),
            )),
            payloads.finish().unwrap(),
        ],
    )
    .unwrap();
    InsertBuilder::new(uri)
        .with_params(&WriteParams {
            max_rows_per_file: IDS[0].len(),
            data_storage_version: Some(LanceFileVersion::V2_2),
            ..Default::default()
        })
        .execute(vec![batch])
        .await
        .unwrap()
}

async fn delete_rows(source: &Table) -> Table {
    let mut updated_fragments = Vec::new();
    for (fragment, ids) in source.get_fragments().into_iter().zip(IDS) {
        let offsets = (0..ids.len() as u32)
            .filter(|offset| DELETED.contains(&ids[*offset as usize]))
            .collect::<Vec<_>>();
        if !offsets.is_empty() {
            let deleted = fragment.extend_deletions(offsets).await.unwrap().unwrap();
            updated_fragments.push(deleted.into());
        }
    }
    CommitBuilder::new(source)
        .execute(Transaction::new(
            source.version_id(),
            Operation::Delete {
                updated_fragments,
                deleted_fragment_ids: Vec::new(),
                predicate: "id IN (51, 71, 11, 81, 50, 61, 45)".into(),
            },
            None,
        ))
        .await
        .unwrap()
}

fn blob_bytes(id: i32) -> Vec<u8> {
    vec![u8::try_from(id).unwrap(); usize::try_from(id).unwrap() + 12]
}

fn fixture_stage_schema() -> PaimonSchema {
    let fields = [
        Field::new(
            "embedding",
            DataType::FixedSizeList(
                Arc::new(Field::new("item", DataType::Float32, true)),
                2,
            ),
            true,
        ),
        Field::new(
            "annotation",
            DataType::Struct(
                vec![
                    Field::new("score", DataType::Int32, true),
                    Field::new("caption", DataType::Utf8, true),
                ]
                .into(),
            ),
            true,
        ),
        Field::new("tag", DataType::Utf8, true),
    ];
    arrow_fields_to_paimon(&fields)
        .unwrap()
        .into_iter()
        .fold(SchemaBuilder::new(), SchemaBuilder::field)
        .option("file.format", "parquet")
        .option("read.batch-size", "2")
        .build()
        .unwrap()
}

fn features(ids: &[i32], schema: SchemaRef, attempt: &str) -> RecordBatch {
    let DataType::FixedSizeList(item, size) = schema.field(0).data_type() else {
        panic!("vector field expected")
    };
    let DataType::Struct(fields) = schema.field(1).data_type() else {
        panic!("struct field expected")
    };
    let values = Float32Array::from_iter_values(ids.iter().flat_map(|id| {
        let id = f32::from(i16::try_from(*id).unwrap());
        [id, -id]
    }));
    let embedding =
        FixedSizeListArray::try_new(Arc::clone(item), *size, Arc::new(values), None)
            .unwrap();
    let annotation = StructArray::try_new(
        fields.clone(),
        vec![
            Arc::new(Int32Array::from_iter_values(ids.iter().map(|id| id * 10))),
            Arc::new(StringArray::from_iter_values(
                ids.iter().map(|id| format!("{attempt}-annotation-{id}")),
            )),
        ],
        None,
    )
    .unwrap();
    let tag =
        StringArray::from_iter_values(ids.iter().map(|id| format!("{attempt}-tag-{id}")));
    RecordBatch::try_new(
        schema,
        vec![Arc::new(embedding), Arc::new(annotation), Arc::new(tag)],
    )
    .unwrap()
}

fn assert_feature_values(actual: &RecordBatch, expected: &RecordBatch) {
    assert_eq!(actual.num_rows(), expected.num_rows());
    let actual_embedding = actual["embedding"].as_fixed_size_list();
    let expected_embedding = expected["embedding"].as_fixed_size_list();
    assert_eq!(
        actual_embedding.null_count(),
        expected_embedding.null_count()
    );
    assert_eq!(
        actual_embedding.value_length(),
        expected_embedding.value_length()
    );
    assert_eq!(
        actual_embedding.values().to_data(),
        expected_embedding.values().to_data()
    );
    let actual_annotation = actual["annotation"].as_struct();
    let expected_annotation = expected["annotation"].as_struct();
    assert_eq!(
        actual_annotation.null_count(),
        expected_annotation.null_count()
    );
    for name in ["score", "caption"] {
        assert_eq!(
            actual_annotation.column_by_name(name).unwrap().to_data(),
            expected_annotation.column_by_name(name).unwrap().to_data(),
        );
    }
    assert_eq!(actual["tag"].to_data(), expected["tag"].to_data());
}

async fn stage_catalog(path: &Path) -> FileSystemCatalog {
    let mut options = Options::new();
    options.set(CatalogOptions::WAREHOUSE, path.to_str().unwrap());
    let catalog = FileSystemCatalog::new(options).unwrap();
    catalog
        .create_database("stage", true, HashMap::new())
        .await
        .unwrap();
    catalog
}

#[derive(Clone)]
struct ParquetWinner {
    table: paimon::Table,
    split: DataSplit,
}

fn write_parquet_winner<'a>(
    catalog: &'a FileSystemCatalog,
    name: &'a str,
    schema: &'a PaimonSchema,
    batch: &'a RecordBatch,
) -> BoxFuture<'a, ParquetWinner> {
    Box::pin(async move {
        let id = Identifier::new("stage", name);
        catalog
            .create_table(&id, schema.clone(), false)
            .await
            .unwrap();
        let table = catalog.get_table(&id).await.unwrap();
        let builder = table.new_write_builder();
        let mut writer = builder.new_write().unwrap();
        writer.write_arrow_batch(batch).await.unwrap();
        let messages = writer.prepare_commit().await.unwrap();
        Box::pin(builder.new_commit().commit(messages))
            .await
            .unwrap();
        let plan = table.new_read_builder().new_scan().plan().await.unwrap();
        let [split] = plan.splits() else {
            panic!("expected one stage split")
        };
        let [file] = split.data_files() else {
            panic!("expected one winning Parquet file")
        };
        assert!(file.file_name.ends_with(".parquet"));
        let split = split.clone();
        ParquetWinner { table, split }
    })
}

fn open_parquet(
    winner: ParquetWinner,
) -> impl Future<Output = Result<SendableRecordBatchStream>> + Send {
    let external = |error| DataFusionError::External(Box::new(error));
    ready((|| {
        let builder = winner.table.new_read_builder();
        let read = builder.new_read().map_err(external)?;
        let schema = build_read_arrow_schema(read.read_type()).map_err(external)?;
        let stream = read
            .to_arrow(&[winner.split])
            .map_err(external)?
            .map_err(external);
        Ok(Box::pin(RecordBatchStreamAdapter::new(schema, stream))
            as SendableRecordBatchStream)
    })())
}

fn artifact_path(receipt: &BackfillReceipt) -> PathBuf {
    Url::parse(&receipt.artifact_uri)
        .unwrap()
        .to_file_path()
        .unwrap()
}

fn file_state(path: &Path) -> (SystemTime, [u8; 32]) {
    (
        fs::metadata(path).unwrap().modified().unwrap(),
        Sha256::digest(fs::read(path).unwrap()).into(),
    )
}

fn file_states(path: &Path) -> BTreeMap<PathBuf, (SystemTime, [u8; 32])> {
    let mut states = BTreeMap::new();
    for entry in fs::read_dir(path).unwrap() {
        let path = entry.unwrap().path();
        if path.is_dir() {
            states.extend(file_states(&path));
        } else {
            let state = file_state(&path);
            states.insert(path, state);
        }
    }
    states
}
