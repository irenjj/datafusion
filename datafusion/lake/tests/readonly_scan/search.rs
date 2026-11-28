// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: See the NOTICE file distributed with this work for additional information regarding copyright ownership
// SPDX-FileContributor: Licensed to the Apache Software Foundation (ASF) under one or more contributor license agreements

use std::sync::Arc;

use arrow::array::{
    Array, FixedSizeListArray, Float32Array, Int32Array, LargeBinaryArray, RecordBatch,
    StringArray,
};
use arrow::datatypes::{DataType, Field, Float32Type, Schema};
use bytes::Bytes;
use datafusion_catalog::{Session, TableFunctionArgs, TableFunctionImpl, TableProvider};
use datafusion_common::{DataFusionError, Result, ScalarValue};
use datafusion_expr::{Expr, TableProviderFilterPushDown, col, lit};
use datafusion_lake::search::{RankedRow, rrf_fuse};
use datafusion_lake::{
    LakeSearchCatalog, LakeTableOptions, LanceReadOnlyTable, PaimonReadOnlyTable,
    PinnedLakeTable, RequiredIndexSpec, hybrid_search_function, vector_search_function,
};
use datafusion_physical_plan::collect;
use lance::Dataset;
use lance::blob::{BlobDescriptorArrayBuilder, BlobRange, blob_field};
use lance_file::version::LanceFileVersion;
use lance_index::scalar::FullTextSearchQuery;
use lance_table_runtime::write::{InsertBuilder, WriteMode, WriteParams};
use lance_testing::datafusion::TestSession;
use paimon::io::{FileIOBuilder, FileRead};
use paimon::spec::{
    BlobDescriptor, BlobType, DataType as PaimonType, FloatType, IntType, SchemaBuilder,
    VarCharType, VectorType,
};
use paimon::{Table, spec::Schema as PaimonSchema};
use tempfile::TempDir;
use url::Url;
use uuid::Uuid;

use super::fixtures::{new_paimon, scan, write_paimon};
use super::index::{build_text_index, build_vector_index, publish_indices};
use super::media::{object_read_ranges, observed_lance, read_references};
use super::store::Reads;

const TEXTS: [&str; 5] = [
    "rock clay dust moss",
    "clay rock dust moss",
    "quartz quartz quartz quartz",
    "quartz quartz quartz rock",
    "quartz clay dust moss",
];

fn batch(
    start: i32,
    vectors: &[Vec<f32>],
    labels: &[&str],
    payload: Option<(Field, Arc<dyn Array>)>,
) -> RecordBatch {
    let dim = i32::try_from(vectors[0].len()).unwrap();
    let vectors = FixedSizeListArray::from_iter_primitive::<Float32Type, _, _>(
        vectors
            .iter()
            .map(|row| Some(row.iter().copied().map(Some))),
        dim,
    );
    let mut fields = vec![
        Field::new("id", DataType::Int32, false),
        Field::new("vector", vectors.data_type().clone(), true),
        Field::new("label", DataType::Utf8, true),
    ];
    let mut columns = vec![
        Arc::new(Int32Array::from_iter_values(
            start..start + i32::try_from(vectors.len()).unwrap(),
        )) as Arc<dyn Array>,
        Arc::new(vectors),
        Arc::new(StringArray::from(labels.to_vec())),
    ];
    if let Some((field, array)) = payload {
        fields.push(field);
        columns.push(array);
    }
    RecordBatch::try_new(Arc::new(Schema::new(fields)), columns).unwrap()
}

fn mixed_batch(payload: Option<(Field, Arc<dyn Array>)>) -> RecordBatch {
    batch(
        1,
        &[
            vec![1., 0.],
            vec![2., 0.],
            vec![3., 0.],
            vec![4., 0.],
            vec![5., 0.],
        ],
        &TEXTS,
        payload,
    )
}

fn required(dim: u32, text: bool) -> LakeTableOptions {
    let mut required_indices = vec![RequiredIndexSpec::Vector {
        column: "vector".into(),
        dimension: dim,
    }];
    if text {
        required_indices.push(RequiredIndexSpec::FullText {
            column: "label".into(),
        });
    }
    LakeTableOptions { required_indices }
}

async fn write_lance_batch(
    uri: &str,
    mode: WriteMode,
    batch: RecordBatch,
) -> lance_table_runtime::Table {
    InsertBuilder::new(uri)
        .with_params(
            &WriteParams {
                max_rows_per_file: 128,
                mode,
                data_storage_version: Some(LanceFileVersion::V2_2),
                ..Default::default()
            }
            .with_allow_external_blob_outside_bases(true),
        )
        .execute(vec![batch])
        .await
        .unwrap()
}

async fn index_lance(
    table: &lance_table_runtime::Table,
    batch: &RecordBatch,
    text: bool,
) {
    let vectors = batch["vector"]
        .as_any()
        .downcast_ref::<FixedSizeListArray>()
        .unwrap()
        .clone();
    let mut indices = vec![
        build_vector_index(
            table,
            vectors,
            "embedding",
            Uuid::from_u128(2),
            table.manifest().fragments.len(),
        )
        .await,
    ];
    if text {
        indices.push(build_text_index(table, Arc::clone(&batch["label"])).await);
    }
    publish_indices(table, indices).await;
}

fn paimon_schema(dim: u32) -> SchemaBuilder {
    PaimonSchema::builder()
        .column("id", PaimonType::Int(IntType::new()))
        .column(
            "vector",
            PaimonType::Vector(
                VectorType::new(dim, PaimonType::Float(FloatType::new())).unwrap(),
            ),
        )
        .column("label", PaimonType::VarChar(VarCharType::string_type()))
        .option("file.format", "parquet")
        .option("row-tracking.enabled", "true")
        .option("data-evolution.enabled", "true")
        .option("global-index.enabled", "true")
        .option("ivf-flat.nlist", "1")
        .option("ivf-flat.distance.metric", "l2")
}

async fn index_paimon(table: &Table, text: bool) {
    for (column, index_type) in [("vector", "ivf-flat"), ("label", "full-text")]
        .into_iter()
        .take(if text { 2 } else { 1 })
    {
        let mut build = table.new_global_index_build_builder();
        assert!(
            Box::pin(
                build
                    .with_index_column(column)
                    .with_index_type(index_type)
                    .execute()
            )
            .await
            .unwrap()
                > 0
        );
    }
}

fn catalog(table: PinnedLakeTable) -> Arc<LakeSearchCatalog> {
    let catalog = Arc::new(LakeSearchCatalog::default());
    catalog.register("fixture", table).unwrap();
    catalog
}

async fn lance_catalog(
    dir: &TempDir,
    batch: RecordBatch,
    dim: u32,
    text: bool,
) -> Arc<LakeSearchCatalog> {
    let uri = dir.path().to_str().unwrap();
    let fixture = write_lance_batch(uri, WriteMode::Create, batch.clone()).await;
    index_lance(&fixture, &batch, text).await;
    let pinned = LanceReadOnlyTable::open(uri, None, &required(dim, text))
        .await
        .unwrap();
    catalog(pinned.into())
}

async fn paimon_catalog(
    dir: &TempDir,
    batch: RecordBatch,
    dim: u32,
    text: bool,
) -> (Table, Arc<LakeSearchCatalog>) {
    let table = new_paimon(dir.path(), paimon_schema(dim).build().unwrap()).await;
    Box::pin(write_paimon(&table, &batch)).await;
    Box::pin(index_paimon(&table, text)).await;
    let pinned = PaimonReadOnlyTable::open(table.clone(), None, &required(dim, text))
        .await
        .unwrap();
    (table, catalog(pinned.into()))
}

fn provider(
    function: &dyn TableFunctionImpl,
    args: &[Expr],
) -> Result<Arc<dyn TableProvider>> {
    function.call_with_args(TableFunctionArgs::new(args, &TestSession::default()))
}

fn vector_args(json: &str, k: i64) -> Vec<Expr> {
    vec![lit("fixture"), lit("vector"), lit(json), lit(k)]
}

fn hybrid_args(k: i64) -> Vec<Expr> {
    vec![
        lit("fixture"),
        lit("vector"),
        lit("[0,0]"),
        lit("label"),
        lit("quartz"),
        lit(k),
    ]
}

fn ordered_rows(batches: &[RecordBatch]) -> Vec<(i32, f32)> {
    batches
        .iter()
        .flat_map(|batch| {
            let ids = batch["id"].as_any().downcast_ref::<Int32Array>().unwrap();
            let scores = batch["_score"]
                .as_any()
                .downcast_ref::<Float32Array>()
                .unwrap();
            assert_eq!(scores.null_count(), 0);
            ids.values()
                .iter()
                .copied()
                .zip(scores.values().iter().copied())
        })
        .collect()
}

fn rank_score(weight: f64, rrf_k: u64, rank: usize) -> f64 {
    weight / (rrf_k as f64 + rank as f64)
}

fn assert_rows(actual: Vec<(i32, f32)>, expected: &[(i32, f64)]) {
    assert_eq!(actual.len(), expected.len());
    for ((id, score), (expected_id, expected_score)) in actual.into_iter().zip(expected) {
        assert_eq!(id, *expected_id);
        assert_eq!(
            score.to_bits(),
            (*expected_score as f32).to_bits(),
            "id {id}"
        );
    }
}

#[test]
fn rrf_is_weighted_one_based_deterministic_and_breaks_ties_by_key() {
    let route = |keys: &[u64]| {
        keys.iter()
            .map(|&row_key| RankedRow {
                row_key,
                row_addr: None,
            })
            .collect::<Vec<_>>()
    };
    let first = route(&[9, 1, 3]);
    let second = route(&[1, 9, 5]);
    let equal_routes = [(1.0, first.as_slice()), (1.0, second.as_slice())];
    let rows = rrf_fuse(60, 4, &equal_routes).unwrap();
    assert_eq!(
        rows.iter().map(|r| r.row_key).collect::<Vec<_>>(),
        [1, 9, 3, 5]
    );
    assert_eq!(rows, rrf_fuse(60, 4, &equal_routes).unwrap());
    assert_eq!(rows[0].score, 1.0 / 61.0 + 1.0 / 62.0);

    let weighted = rrf_fuse(2, 1, &[(2.0, &first), (1.0, &second)]).unwrap();
    assert_eq!(weighted[0].row_key, 9);
    assert_eq!(weighted[0].score, 2.0 / 3.0 + 1.0 / 4.0);
    let single = rrf_fuse(60, 3, &[(1.0, &first)]).unwrap();
    assert_eq!(single[0].row_key, 9);
    assert_eq!(single[0].score, 1.0 / 61.0);
    assert!(rrf_fuse(0, 3, &equal_routes).is_err());
    assert!(rrf_fuse(60, 3, &[(1.0, &[])]).unwrap().is_empty());
}

#[tokio::test]
async fn fails_closed_on_invalid_arguments_and_filters() {
    let dir = tempfile::tempdir().unwrap();
    let catalog = lance_catalog(&dir, mixed_batch(None), 2, true).await;
    let vector = vector_search_function(Arc::clone(&catalog));
    let hybrid = hybrid_search_function(catalog);
    for args in [
        vector_args("[0,0]", 0),
        vector_args("[0,0]", 10_001),
        vector_args("[0]", 1),
        vector_args("[NaN,0]", 1),
        vector_args("[Infinity,0]", 1),
        vector_args("[1e40,0]", 1),
        vector_args("[null,0]", 1),
        vec![lit("fixture"), lit("missing"), lit("[0,0]"), lit(1)],
        vec![lit("fixture"), lit("label"), lit("[0,0]"), lit(1)],
        vec![lit("fixture"), lit("vector"), lit("[0,0]"), col("id")],
        vec![lit("fixture"), lit("vector"), lit("[0,0]"), lit(1) + lit(1)],
        vec![col("id"), lit("vector"), lit("[0,0]"), lit(1)],
    ] {
        assert!(
            matches!(
                provider(vector.as_ref(), &args),
                Err(DataFusionError::Plan(_))
            ),
            "{args:?}"
        );
    }
    let mut unknown_text = hybrid_args(3);
    unknown_text[3] = lit("missing");
    assert!(matches!(
        provider(hybrid.as_ref(), &unknown_text),
        Err(DataFusionError::Plan(_))
    ));
    for value in [lit(0), lit(-1), lit(1.5), lit(ScalarValue::Int64(None))] {
        let mut args = hybrid_args(3);
        args.push(value);
        assert!(matches!(
            provider(hybrid.as_ref(), &args),
            Err(DataFusionError::Plan(_))
        ));
    }
    for weight in [0.0, -1.0, f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
        for weights in [[weight, 1.0], [1.0, weight]] {
            let mut args = hybrid_args(3);
            args.extend([lit(60), lit(weights[0]), lit(weights[1])]);
            assert!(
                matches!(
                    provider(hybrid.as_ref(), &args),
                    Err(DataFusionError::Plan(_))
                ),
                "{weights:?}"
            );
        }
    }
    let mut args = hybrid_args(3);
    args.extend([lit(60), lit(1.0), lit(1.0), lit("label LIKE '%quartz%'")]);
    assert!(matches!(
        provider(hybrid.as_ref(), &args),
        Err(DataFusionError::NotImplemented(_))
    ));
    args[9] = lit("id > 1; SELECT 1");
    assert!(provider(hybrid.as_ref(), &args).is_err());
}

fn l2_oracle() -> (Vec<Vec<f32>>, Vec<f32>, Vec<i32>) {
    let vectors: Vec<Vec<f32>> = (0..1000_i32)
        .map(|id| {
            (0..32)
                .map(|axis| {
                    ((id * (979 - 2 * axis) + axis * 17) % 1009 - 500) as f32 / 16.0
                })
                .collect()
        })
        .collect();
    let query: Vec<f32> = (0..32).map(|axis| axis as f32 * 0.37 - 9.2).collect();
    let mut distances: Vec<_> = vectors
        .iter()
        .enumerate()
        .map(|(id, vec)| {
            let squared_distance: f64 = vec
                .iter()
                .zip(&query)
                .map(|(&value, &key)| (f64::from(value) - f64::from(key)).powi(2))
                .sum();
            (id as i32, squared_distance)
        })
        .collect();
    distances.sort_unstable_by(|a, b| a.1.total_cmp(&b.1).then(a.0.cmp(&b.0)));
    let top = distances.into_iter().take(24).map(|(id, _)| id).collect();
    (vectors, query, top)
}

async fn check_vector(catalog: Arc<LakeSearchCatalog>, query: &[f32], top: &[i32]) {
    let function = vector_search_function(catalog);
    let args = vector_args(&serde_json::to_string(query).unwrap(), top.len() as i64);
    let table = provider(function.as_ref(), &args).unwrap();
    let schema = table.schema();
    assert_eq!(schema.field(3).name(), "_score");
    assert_eq!(schema.field(3).data_type(), &DataType::Float32);
    assert!(!schema.field(3).is_nullable());
    assert_eq!(
        table
            .supports_filters_pushdown(&[&col("id").eq(lit(1))])
            .unwrap(),
        [TableProviderFilterPushDown::Unsupported]
    );
    let expected = top
        .iter()
        .enumerate()
        .map(|(idx, &id)| (id, rank_score(1.0, 60, idx + 1)))
        .collect::<Vec<_>>();
    let rows = scan(table.as_ref(), Some(&[3, 0]), &[], None).await;
    assert_eq!(rows[0].schema().field(0).name(), "_score");
    assert_rows(ordered_rows(&rows), &expected);
    let rows = scan(table.as_ref(), Some(&[]), &[], None).await;
    assert_eq!(
        rows.iter().map(RecordBatch::num_rows).sum::<usize>(),
        expected.len()
    );
    assert!(rows.iter().all(|b| b.num_columns() == 0));
}

#[tokio::test]
async fn lance_vector_matches_exact_l2_oracle() {
    let dir = tempfile::tempdir().unwrap();
    let uri = dir.path().to_str().unwrap();
    let (vectors, query, top) = l2_oracle();
    let batch = batch(0, &vectors, &["other"; 1000], None);
    let fixture = write_lance_batch(uri, WriteMode::Create, batch.clone()).await;
    let vectors = batch["vector"]
        .as_any()
        .downcast_ref::<FixedSizeListArray>()
        .unwrap();
    let partial = build_vector_index(
        &fixture,
        vectors.slice(0, 128),
        "a_partial",
        Uuid::from_u128(4),
        1,
    )
    .await;
    let complete = build_vector_index(
        &fixture,
        vectors.clone(),
        "embedding",
        Uuid::from_u128(2),
        fixture.manifest().fragments.len(),
    )
    .await;
    publish_indices(&fixture, vec![partial, complete]).await;
    assert!(top.iter().any(|&id| id >= 128));
    let pinned = LanceReadOnlyTable::open(uri, None, &required(32, false))
        .await
        .unwrap();
    check_vector(catalog(pinned.into()), &query, &top).await;
}

#[tokio::test]
async fn paimon_vector_matches_exact_l2_oracle() {
    let dir = tempfile::tempdir().unwrap();
    let (vectors, query, top) = l2_oracle();
    let batch = batch(0, &vectors, &["other"; 1000], None);
    let (_, catalog) = Box::pin(paimon_catalog(&dir, batch, 32, false)).await;
    check_vector(catalog, &query, &top).await;
}

async fn check_hybrid(catalog: Arc<LakeSearchCatalog>) {
    let function = hybrid_search_function(catalog);
    let v = |rank| rank_score(1.0, 60, rank);
    let mut args = hybrid_args(3);
    let table = provider(function.as_ref(), &args).unwrap();
    let rows = scan(table.as_ref(), Some(&[0, 3]), &[], None).await;
    assert_rows(
        ordered_rows(&rows),
        &[(3, v(3) + v(1)), (1, v(1)), (2, v(2))],
    );

    args.extend([lit(60), lit(1.0), lit(2.0)]);
    let table = provider(function.as_ref(), &args).unwrap();
    let rows = scan(table.as_ref(), Some(&[0, 3]), &[], None).await;
    assert_rows(
        ordered_rows(&rows),
        &[(3, v(3) + 2.0 * v(1)), (4, 2.0 * v(2)), (5, 2.0 * v(3))],
    );

    args[5] = lit(2);
    args[6] = lit(4);
    args[8] = lit(1.0);
    args.push(lit("id IN (2, 4, 5) AND NOT (id = 4 OR label IS NULL)"));
    let table = provider(function.as_ref(), &args).unwrap();
    let rows = scan(table.as_ref(), Some(&[0, 3]), &[], None).await;
    assert_rows(
        ordered_rows(&rows),
        &[(5, 1.0 / 6.0 + 1.0 / 5.0), (2, 1.0 / 5.0)],
    );

    args[9] = lit("id < 0");
    let table = provider(function.as_ref(), &args).unwrap();
    assert!(
        scan(table.as_ref(), Some(&[0, 3]), &[], None)
            .await
            .is_empty()
    );
}

#[tokio::test]
async fn lance_hybrid_and_bm25_oracle() {
    let dir = tempfile::tempdir().unwrap();
    let catalog = lance_catalog(&dir, mixed_batch(None), 2, true).await;
    let dataset = Dataset::open(dir.path().to_str().unwrap()).await.unwrap();
    let mut scanner = dataset.scan();
    scanner.project(&["id"]).unwrap();
    scanner
        .full_text_search(
            FullTextSearchQuery::new("quartz".into())
                .with_column("label".into())
                .unwrap(),
        )
        .unwrap();
    let scores = ordered_rows(&[scanner.try_into_batch().await.unwrap()]);
    let idf = ((5.0_f64 - 3.0 + 0.5) / (3.0 + 0.5)).ln_1p();
    let bm25 = [4.0, 3.0, 1.0].map(|tf| idf * 2.2 * tf / (tf + 1.2));
    assert_eq!(scores.len(), 3);
    for ((id, actual), (expected_id, score)) in scores.into_iter().zip((3..=5).zip(bm25))
    {
        assert_eq!(id, expected_id);
        assert!(
            (f64::from(actual) - score).abs() < 1e-6,
            "{actual} != {score}"
        );
    }
    check_hybrid(catalog).await;
}

#[tokio::test]
async fn paimon_hybrid_and_bm25_rank_oracle() {
    let dir = tempfile::tempdir().unwrap();
    let (table, catalog) =
        Box::pin(paimon_catalog(&dir, mixed_batch(None), 2, true)).await;
    let result = Box::pin(
        table
            .new_full_text_search_builder()
            .with_text_column("label")
            .with_query_text("quartz")
            .with_limit(3)
            .execute(),
    )
    .await
    .unwrap();
    let hits = result.row_ids().unwrap();
    // Equal-length documents make BM25 strictly increasing in term frequency.
    assert_eq!(hits.row_ids, [2, 3, 4]);
    assert!(hits.scores.windows(2).all(|pair| pair[0] > pair[1]));
    check_hybrid(catalog).await;
}

#[tokio::test]
async fn lance_search_pins_snapshot_and_requires_open_time_index_validation() {
    let dir = tempfile::tempdir().unwrap();
    let uri = dir.path().to_str().unwrap();
    let batch = mixed_batch(None);
    let fixture = write_lance_batch(uri, WriteMode::Create, batch.clone()).await;
    let missing = LanceReadOnlyTable::open(uri, None, &required(2, false)).await;
    assert!(
        matches!(missing, Err(DataFusionError::Plan(message)) if message.contains("missing"))
    );
    let unvalidated = LanceReadOnlyTable::open(uri, None, &LakeTableOptions::default())
        .await
        .unwrap();
    let function = vector_search_function(catalog(unvalidated.into()));
    assert!(matches!(
        provider(function.as_ref(), &vector_args("[0,0]", 3)),
        Err(DataFusionError::Plan(message)) if message.contains("index")
    ));

    index_lance(&fixture, &batch, false).await;
    let missing = LanceReadOnlyTable::open(uri, None, &required(2, true)).await;
    assert!(
        matches!(missing, Err(DataFusionError::Plan(message)) if message.contains("missing"))
    );
    let pinned = LanceReadOnlyTable::open(uri, None, &required(2, false))
        .await
        .unwrap();
    let catalog = catalog(pinned.into());
    let function = hybrid_search_function(Arc::clone(&catalog));
    assert!(matches!(provider(function.as_ref(), &hybrid_args(3)),
        Err(DataFusionError::Plan(message)) if message.contains("FullText")
    ));
    let new_row = self::batch(99, &[vec![0., 0.]], &["quartz"], None);
    write_lance_batch(uri, WriteMode::Append, new_row).await;
    let latest = LanceReadOnlyTable::open(uri, None, &LakeTableOptions::default())
        .await
        .unwrap();
    assert_eq!(
        super::fixtures::ids(&scan(&latest, Some(&[0]), &[], None).await),
        [1, 2, 3, 4, 5, 99]
    );
    check_vector(catalog, &[0., 0.], &[1, 2, 3, 4, 5]).await;
}

#[tokio::test]
async fn paimon_search_pins_snapshot_and_requires_open_time_index_validation() {
    let dir = tempfile::tempdir().unwrap();
    let table = new_paimon(dir.path(), paimon_schema(2).build().unwrap()).await;
    Box::pin(write_paimon(&table, &mixed_batch(None))).await;
    let missing =
        PaimonReadOnlyTable::open(table.clone(), None, &required(2, false)).await;
    assert!(matches!(missing, Err(DataFusionError::Plan(_))));
    let unvalidated =
        PaimonReadOnlyTable::open(table.clone(), None, &LakeTableOptions::default())
            .await
            .unwrap();
    let function = vector_search_function(catalog(unvalidated.into()));
    assert!(matches!(
        provider(function.as_ref(), &vector_args("[0,0]", 3)),
        Err(DataFusionError::Plan(message)) if message.contains("index")
    ));

    Box::pin(index_paimon(&table, false)).await;
    let missing =
        PaimonReadOnlyTable::open(table.clone(), None, &required(2, true)).await;
    assert!(matches!(missing, Err(DataFusionError::Plan(_))));
    let pinned = PaimonReadOnlyTable::open(table.clone(), None, &required(2, false))
        .await
        .unwrap();
    let catalog = catalog(pinned.into());
    let function = hybrid_search_function(Arc::clone(&catalog));
    assert!(matches!(provider(function.as_ref(), &hybrid_args(3)),
        Err(DataFusionError::Plan(message)) if message.contains("FullText")
    ));
    let new_row = batch(99, &[vec![0., 0.]], &["quartz"], None);
    Box::pin(write_paimon(&table, &new_row)).await;
    let latest = PaimonReadOnlyTable::open(table, None, &LakeTableOptions::default())
        .await
        .unwrap();
    assert_eq!(
        super::fixtures::ids(&scan(&latest, Some(&[0]), &[], None).await),
        [1, 2, 3, 4, 5, 99]
    );
    check_vector(catalog, &[0., 0.], &[1, 2, 3, 4, 5]).await;
}

fn assert_payload_unread(reads: &Reads, batches: &[RecordBatch]) {
    let read_log = reads.take();
    assert!(!read_log.is_empty());
    let refs = read_references(batches);
    assert_eq!(refs.len(), 3);
    for reference in refs.into_iter().map(Option::unwrap) {
        let start = reference.byte_offset.unwrap();
        let end = start + reference.byte_len.unwrap();
        let bytes: u64 = object_read_ranges(&read_log, &reference)
            .map(|range| range.end.min(end).saturating_sub(range.start.max(start)))
            .sum();
        assert_eq!(bytes, 0, "{reference:?}, {read_log:?}");
    }
}

async fn projected_media(catalog: Arc<LakeSearchCatalog>) -> Vec<RecordBatch> {
    let function = hybrid_search_function(catalog);
    let table = provider(function.as_ref(), &hybrid_args(3)).unwrap();
    let session = TestSession::default();
    let plan = table
        .scan(&session, Some(&[3, 4]), &[], None)
        .await
        .unwrap();
    collect(plan, session.task_ctx()).await.unwrap()
}

#[tokio::test]
async fn lance_late_read_does_not_fetch_blob_payloads() {
    let dir = tempfile::tempdir().unwrap();
    let media = dir.path().join("payload.blob");
    std::fs::write(&media, [7; 16]).unwrap();
    let media = Url::from_file_path(media).unwrap();
    let path = dir.path().join("fixture.lance");
    let uri = path.to_str().unwrap();
    let mut values =
        BlobDescriptorArrayBuilder::for_field(&blob_field("payload", true)).unwrap();
    for row in 0..5 {
        values
            .push_external(
                media.as_str(),
                Some(BlobRange {
                    offset: 2 * row,
                    size: 2,
                }),
            )
            .unwrap();
    }
    let batch = mixed_batch(Some(values.finish().unwrap().into_parts()));
    let fixture = write_lance_batch(uri, WriteMode::Create, batch.clone()).await;
    index_lance(&fixture, &batch, true).await;
    let reads = Reads::new();
    let table = LanceReadOnlyTable::open_with_builder(
        observed_lance(uri, &reads),
        &required(2, true),
    )
    .await
    .unwrap();
    let batches = projected_media(catalog(table.into())).await;
    assert_payload_unread(&reads, &batches);
}

#[tokio::test]
async fn paimon_late_read_does_not_fetch_blob_payloads() {
    let dir = tempfile::tempdir().unwrap();
    let media_path = dir.path().join("payload.blob");
    std::fs::write(&media_path, [7; 16]).unwrap();
    let media = Url::from_file_path(&media_path).unwrap();
    let schema = paimon_schema(2)
        .column("payload", PaimonType::Blob(BlobType::new()))
        .option("blob-descriptor-field", "payload")
        .build()
        .unwrap();
    let table = new_paimon(dir.path(), schema).await;
    let payload = LargeBinaryArray::from_iter_values(
        (0..5).map(|row| BlobDescriptor::new(media.to_string(), row * 2, 2).serialize()),
    );
    let batch = mixed_batch(Some((
        Field::new("payload", DataType::LargeBinary, true),
        Arc::new(payload),
    )));
    Box::pin(write_paimon(&table, &batch)).await;
    Box::pin(index_paimon(&table, true)).await;

    let reads = Reads::new();
    let io = FileIOBuilder::new("file")
        .build()
        .unwrap()
        .with_file_block_cache(Arc::new(reads.clone()), 4096, "*")
        .unwrap();
    let reader = io
        .new_input(media.as_str())
        .unwrap()
        .reader()
        .await
        .unwrap();
    assert_eq!(
        reader.read(0..2).await.unwrap(),
        Bytes::from_static(&[7, 7])
    );
    assert!(reads.take().iter().any(|read| {
        read.path.filename() == Some("payload.blob")
            && read.range.as_ref().is_some_and(|r| r.end > r.start)
    }));
    std::fs::remove_file(media_path).unwrap();

    let table = Table::new(
        io,
        table.identifier().clone(),
        table.location().into(),
        table.schema().clone(),
        None,
    );
    let pinned = PaimonReadOnlyTable::open(table, None, &required(2, true))
        .await
        .unwrap();
    let batches = projected_media(catalog(pinned.into())).await;
    assert_payload_unread(&reads, &batches);
}
