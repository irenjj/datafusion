// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: See the NOTICE file distributed with this work for additional information regarding copyright ownership
// SPDX-FileContributor: Licensed to the Apache Software Foundation (ASF) under one or more contributor license agreements

use std::collections::HashMap;
use std::path::Path;
use std::sync::Arc;

use arrow::array::{Array, FixedSizeListArray, Int32Array, RecordBatch, StringArray};
use arrow::datatypes::{Field, Float32Type, Schema};
use datafusion_catalog::{Session, TableProvider};
use datafusion_datasource::file_scan_config::FileScanConfig;
use datafusion_datasource::source::DataSourceExec;
use datafusion_expr::Expr;
use datafusion_physical_plan::{ExecutionPlan, collect};
use lance_file::version::LanceFileVersion;
use lance_table_runtime::write::{InsertBuilder, WriteMode, WriteParams};
use lance_testing::datafusion::TestSession;
use paimon::catalog::Identifier;
use paimon::spec::{
    DataType, FloatType, IntType, Schema as PaimonSchema, VarCharType, VectorType,
};
use paimon::{Catalog, CatalogOptions, FileSystemCatalog, Options, Table};

pub(super) fn scalar_rows(start: i32) -> RecordBatch {
    let labels =
        StringArray::from(vec![Some("a"), None, Some("b"), Some("c"), None, Some("a")]);
    let parts = StringArray::from(vec!["p0", "p0", "p0", "p1", "p1", "p1"]);
    let vectors = FixedSizeListArray::from_iter_primitive::<Float32Type, _, _>(
        [Some([Some(1.0), Some(2.0)]); 6],
        2,
    );
    RecordBatch::try_new(
        Arc::new(Schema::new(vec![
            Field::new("id", arrow::datatypes::DataType::Int32, false),
            Field::new("label", arrow::datatypes::DataType::Utf8, true),
            Field::new("part", arrow::datatypes::DataType::Utf8, false),
            Field::new("vector", vectors.data_type().clone(), true),
        ])),
        vec![
            Arc::new(Int32Array::from_iter_values(start..start + 6)),
            Arc::new(labels),
            Arc::new(parts),
            Arc::new(vectors),
        ],
    )
    .unwrap()
}

pub(super) fn scalar_paimon_schema() -> paimon::spec::SchemaBuilder {
    PaimonSchema::builder()
        .column("id", DataType::Int(IntType::new()))
        .column("label", DataType::VarChar(VarCharType::string_type()))
        .column("part", DataType::VarChar(VarCharType::string_type()))
        .column(
            "vector",
            DataType::Vector(
                VectorType::new(2, DataType::Float(FloatType::new())).unwrap(),
            ),
        )
        .partition_keys(["part"])
        .option("file.format", "parquet")
}

pub(super) async fn new_paimon(dir: &Path, schema: PaimonSchema) -> Table {
    let mut options = Options::new();
    options.set(CatalogOptions::WAREHOUSE, dir.to_str().unwrap());
    let catalog = FileSystemCatalog::new(options).unwrap();
    catalog
        .create_database("test", true, HashMap::new())
        .await
        .unwrap();
    let table = Identifier::new("test", "fixture");
    catalog.create_table(&table, schema, false).await.unwrap();
    catalog.get_table(&table).await.unwrap()
}

pub(super) async fn write_paimon(table: &Table, batch: &RecordBatch) {
    let mut writer = table.new_write_builder().new_write().unwrap();
    writer.write_arrow_batch(batch).await.unwrap();
    let messages = writer.prepare_commit().await.unwrap();
    let commit = table.new_write_builder().new_commit();
    Box::pin(commit.commit(messages)).await.unwrap();
}

pub(super) async fn write_lance(
    uri: &str,
    batch: RecordBatch,
    mode: WriteMode,
) -> lance_table_runtime::Table {
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

pub(super) fn file_config(plan: &dyn ExecutionPlan) -> &FileScanConfig {
    let plan = plan.downcast_ref::<DataSourceExec>().unwrap();
    plan.data_source().downcast_ref().unwrap()
}

pub(super) async fn scan(
    table: &dyn TableProvider,
    projection: Option<&[usize]>,
    filters: &[Expr],
    limit: Option<usize>,
) -> Vec<RecordBatch> {
    let session = TestSession::default();
    let plan = table
        .scan(&session, projection, filters, limit)
        .await
        .unwrap();
    collect(plan, session.task_ctx()).await.unwrap()
}

pub(super) fn ids(batches: &[RecordBatch]) -> Vec<i32> {
    let mut rows: Vec<_> = batches
        .iter()
        .flat_map(|batch| {
            batch["id"]
                .as_any()
                .downcast_ref::<Int32Array>()
                .unwrap()
                .values()
                .iter()
                .copied()
        })
        .collect();
    rows.sort_unstable();
    rows
}
