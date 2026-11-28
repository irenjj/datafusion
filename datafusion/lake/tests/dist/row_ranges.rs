// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: See the NOTICE file distributed with this work for additional information regarding copyright ownership
// SPDX-FileContributor: Licensed to the Apache Software Foundation (ASF) under one or more contributor license agreements

use std::collections::BTreeMap;
use std::sync::Arc;

use arrow::array::{Int32Array, RecordBatch};
use arrow::datatypes::{DataType, Field, Schema};
use datafusion_catalog::{Session, TableProvider};
use datafusion_datasource::file_groups::FileGroup;
use datafusion_datasource::file_scan_config::FileScanConfig;
use datafusion_datasource::source::DataSourceExec;
use datafusion_execution::TaskContext;
use datafusion_physical_plan::{ExecutionPlan, collect};
use lance_core::utils::deletion::DeletionVector;
use lance_file::version::LanceFileVersion;
use lance_io::object_store::ObjectStore;
use lance_table::feature_flags::apply_feature_flags;
use lance_table::format::Manifest;
use lance_table::io::commit::write_manifest_file_to_path;
use lance_table::io::deletion::write_deletion_file;
use lance_table_runtime::write::{InsertBuilder, WriteParams};
use lance_testing::datafusion::TestSession;
use paimon::catalog::Identifier;
use paimon::spec::{DataType as PaimonType, IntType, Schema as PaimonSchema};
use paimon::{Catalog, CatalogOptions, FileSystemCatalog, Options, RowRange};
use url::Url;

use super::{bind_split, decode_split, encode_scan, encode_splits, rebuild_scan};
use crate::{
    InclusiveRowRange, LakeSplitExt, LakeSplitPayload, LakeTableOptions,
    LanceReadOnlyTable, PaimonReadOnlyTable,
};

fn rows() -> RecordBatch {
    let schema = Schema::new(vec![Field::new("id", DataType::Int32, false)]);
    RecordBatch::try_new(
        Arc::new(schema),
        vec![Arc::new(Int32Array::from_iter_values(0..6))],
    )
    .unwrap()
}

fn ids(batches: &[RecordBatch]) -> Vec<i32> {
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

fn config(plan: &dyn ExecutionPlan) -> &FileScanConfig {
    plan.downcast_ref::<DataSourceExec>()
        .unwrap()
        .data_source()
        .downcast_ref()
        .unwrap()
}

async fn assert_range_replay(plan: Arc<dyn ExecutionPlan>, expected: &[i32]) {
    let coordinator = TestSession::default();
    let source = config(plan.as_ref());
    let splits = encode_splits(source).unwrap();
    assert!(!splits.is_empty());
    let payload = encode_scan(source).unwrap().unwrap();
    let worker = Arc::new(TaskContext::default());
    let template = rebuild_scan(&payload, plan.schema(), &worker).unwrap();
    let template = config(template.as_ref());
    let mut replayed = vec![];
    for (bytes, file) in splits
        .iter()
        .zip(source.file_groups.iter().flat_map(FileGroup::files))
    {
        let transported = decode_split(bytes).unwrap();
        let split = transported.extension::<LakeSplitExt>().unwrap();
        assert_eq!(Some(split), file.extension::<LakeSplitExt>());
        let ranges = match &split.payload {
            LakeSplitPayload::Lance { row_ranges, .. }
            | LakeSplitPayload::Paimon { row_ranges, .. } => row_ranges,
        };
        assert!(ranges.as_ref().is_some_and(|ranges| !ranges.is_empty()));
        let bound = bind_split(template, transported).unwrap();
        replayed.extend(collect(bound, Arc::clone(&worker)).await.unwrap());
    }
    let local = collect(plan, coordinator.task_ctx()).await.unwrap();
    assert_eq!(ids(&local), expected);
    assert_eq!(ids(&replayed), expected);
}

#[tokio::test]
async fn lance_row_ranges_replay_with_pinned_deletions() {
    let directory = tempfile::tempdir().unwrap();
    let uri = Url::from_file_path(directory.path().join("ranges.lance")).unwrap();
    let uri = uri.as_str();
    let native = InsertBuilder::new(uri)
        .with_params(&WriteParams {
            max_rows_per_file: 3,
            data_storage_version: Some(LanceFileVersion::V2_2),
            ..Default::default()
        })
        .execute(vec![rows()])
        .await
        .unwrap();
    let (store, base) = ObjectStore::from_uri(uri).await.unwrap();
    let mut fragments = native.manifest().fragments.as_ref().clone();
    assert_eq!(fragments.len(), 2);
    let first = fragments[0].id;
    let second = fragments[1].id;
    let deletion_version = native.manifest().version;
    fragments[0].deletion_file = write_deletion_file(
        &base,
        first,
        deletion_version,
        &DeletionVector::from_iter([1]),
        &store,
        0,
    )
    .await
    .unwrap();
    let mut manifest = Manifest::new_from_previous(
        native.manifest(),
        native.schema().clone(),
        Arc::new(fragments),
    );
    apply_feature_flags(&mut manifest, false, false).unwrap();
    let path = native
        .manifest_location()
        .naming_scheme
        .manifest_path(&base, manifest.version);
    write_manifest_file_to_path(&store, &mut manifest, None, &path, None)
        .await
        .unwrap();
    let table = LanceReadOnlyTable::open(
        uri,
        Some(manifest.version),
        &LakeTableOptions::default(),
    )
    .await
    .unwrap();
    let plan = table
        .scan_row_ranges(
            table.schema(),
            BTreeMap::from([
                (
                    u32::try_from(first).unwrap(),
                    vec![InclusiveRowRange { from: 1, to: 2 }],
                ),
                (
                    u32::try_from(second).unwrap(),
                    vec![InclusiveRowRange { from: 0, to: 1 }],
                ),
            ]),
        )
        .unwrap();
    let split = config(plan.as_ref()).file_groups[0].files()[0]
        .extension::<LakeSplitExt>()
        .unwrap();
    assert!(
        matches!(split.payload, LakeSplitPayload::Lance { deletion_file_rev: Some(rev), .. } if rev == deletion_version)
    );
    assert_range_replay(plan, &[2, 3, 4]).await;
}

#[tokio::test]
async fn paimon_row_ranges_replay_with_pinned_data_split() {
    let directory = tempfile::tempdir().unwrap();
    let warehouse = Url::from_file_path(directory.path()).unwrap();
    let mut options = Options::new();
    options.set(CatalogOptions::WAREHOUSE, warehouse.as_str());
    let catalog = FileSystemCatalog::new(options).unwrap();
    catalog
        .create_database("test", true, Default::default())
        .await
        .unwrap();
    let name = Identifier::new("test", "ranges");
    let schema = PaimonSchema::builder()
        .column("id", PaimonType::Int(IntType::new()))
        .option("file.format", "parquet")
        .option("row-tracking.enabled", "true")
        .option("data-evolution.enabled", "true")
        .build()
        .unwrap();
    catalog.create_table(&name, schema, false).await.unwrap();
    let native = catalog.get_table(&name).await.unwrap();
    let mut writer = native.new_write_builder().new_write().unwrap();
    writer.write_arrow_batch(&rows()).await.unwrap();
    let messages = writer.prepare_commit().await.unwrap();
    Box::pin(native.new_write_builder().new_commit().commit(messages))
        .await
        .unwrap();
    let files = native.new_read_builder().new_scan().plan().await.unwrap();
    assert_eq!(files.splits().len(), 1);
    let first_id = files.splits()[0].data_files()[0].first_row_id.unwrap();
    let table = PaimonReadOnlyTable::open(native, None, &LakeTableOptions::default())
        .await
        .unwrap();
    let plan = table
        .scan_row_ranges(
            table.schema(),
            vec![
                RowRange::new(first_id + 1, first_id + 2),
                RowRange::new(first_id + 4, first_id + 4),
            ],
        )
        .await
        .unwrap();
    assert_range_replay(plan, &[1, 2, 4]).await;
}
