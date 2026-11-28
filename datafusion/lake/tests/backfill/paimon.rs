// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: See the NOTICE file distributed with this work for additional information regarding copyright ownership
// SPDX-FileContributor: Licensed to the Apache Software Foundation (ASF) under one or more contributor license agreements

use std::collections::HashMap;
use std::fs;
use std::ops::Range;
use std::path::Path;
use std::sync::Arc;

use arrow::array::{Int32Array, LargeBinaryArray, RecordBatch};
use arrow::datatypes::{DataType, Field, Schema};
use bytes::Bytes;
use datafusion_ai::MediaRefType;
use datafusion_catalog::TableProvider;
use datafusion_common::DataFusionError;
use datafusion_lake::backfill::{BackfillTarget, prepare_backfill};
use datafusion_lake::{LakeTableOptions, PaimonReadOnlyTable};
use lance_io::object_store::ObjectStoreParams;
use paimon::arrow::arrow_field_to_paimon_type;
use paimon::catalog::Identifier;
use paimon::spec::{
    BlobDescriptor, BlobType, DataType as PaimonType, IntType, Schema as PaimonSchema,
    SchemaChange,
};
use paimon::{Catalog, CatalogOptions, FileSystemCatalog, Options, Table};
use url::Url;

use super::{feature_values, file_states, rows_by_id, scan};

#[tokio::test]
async fn paimon_prepare_rejects_plain_and_primary_key_tables() {
    let directory = tempfile::tempdir().unwrap();
    let catalog = catalog(directory.path()).await;
    let columns = feature_values(0..1, "winner").schema();
    for (name, primary_key, expected) in [
        ("plain", false, "data-evolution.enabled=true"),
        ("keyed", true, "without primary keys"),
    ] {
        let identifier = Identifier::new("test", name);
        let mut schema = PaimonSchema::builder()
            .column("id", PaimonType::Int(IntType::new()))
            .option("file.format", "parquet");
        if primary_key {
            schema = schema.primary_key(["id"]).option("bucket", "1");
        }
        catalog
            .create_table(&identifier, schema.build().unwrap(), false)
            .await
            .unwrap();
        let table = catalog.get_table(&identifier).await.unwrap();
        let batch = RecordBatch::try_new(
            Arc::new(Schema::new(vec![Field::new("id", DataType::Int32, false)])),
            vec![Arc::new(Int32Array::from(vec![1]))],
        )
        .unwrap();
        write(&table, &batch).await;
        let before = file_states(directory.path());
        let error = prepare_backfill(
            BackfillTarget::Paimon(&table),
            &columns,
            &ObjectStoreParams::default(),
        )
        .await
        .unwrap_err();
        assert!(
            matches!(error, DataFusionError::Plan(ref message) if message.contains(expected)),
            "{error}"
        );
        assert_eq!(file_states(directory.path()), before);
    }
}

#[tokio::test]
async fn paimon_prepare_refuses_nonatomic_publication_before_any_writes() {
    let directory = tempfile::tempdir().unwrap();
    let payload = directory.path().join("payload.bin");
    fs::write(&payload, b"012345").unwrap();
    let payload_uri = Url::from_file_path(payload).unwrap();
    let catalog = catalog(directory.path()).await;
    let identifier = Identifier::new("test", "features");
    let schema = PaimonSchema::builder()
        .column("id", PaimonType::Int(IntType::new()))
        .column("payload", PaimonType::Blob(BlobType::new()))
        .option("file.format", "parquet")
        .option("row-tracking.enabled", "true")
        .option("data-evolution.enabled", "true")
        .option("blob-descriptor-field", "payload")
        .build()
        .unwrap();
    catalog
        .create_table(&identifier, schema, false)
        .await
        .unwrap();
    let table = catalog.get_table(&identifier).await.unwrap();
    for ids in [0..2, 2..6] {
        write(&table, &blob_rows(ids, payload_uri.as_str())).await;
    }
    let baseline = table
        .snapshot_manager()
        .get_latest_snapshot_id()
        .await
        .unwrap()
        .unwrap();
    assert_eq!(
        table.snapshot_manager().list_all_ids().await.unwrap().len(),
        2
    );
    let before = file_states(directory.path());
    let columns = feature_values(0..1, "winner").schema();
    let result = prepare_backfill(
        BackfillTarget::Paimon(&table),
        &columns,
        &ObjectStoreParams::default(),
    )
    .await;
    assert!(
        matches!(
            result,
            Err(DataFusionError::NotImplemented(ref message))
                if message.contains(&format!("snapshot {baseline}"))
                    && message.contains("atomic schema-and-snapshot")
                    && message.contains("commit_if_latest_snapshot")
        ),
        "{result:?}"
    );
    assert_eq!(file_states(directory.path()), before);
    let current = catalog.get_table(&identifier).await.unwrap();
    assert_eq!(current.schema().id(), table.schema().id());
    let view = PaimonReadOnlyTable::open(current, None, &LakeTableOptions::default())
        .await
        .unwrap();
    assert_eq!(view.snapshot_id(), baseline);
    let schema = view.schema();
    assert_eq!(schema.fields().len(), 2);
    schema
        .field_with_name("payload")
        .unwrap()
        .try_extension_type::<MediaRefType>()
        .unwrap();
    assert_eq!(
        rows_by_id(&scan(&view).await)
            .into_keys()
            .collect::<Vec<_>>(),
        [0, 1, 2, 3, 4, 5]
    );

    // A schema file at its final path is independently visible to the catalog.
    let changes = columns
        .fields()
        .iter()
        .map(|column| {
            SchemaChange::add_column(
                column.name().clone(),
                arrow_field_to_paimon_type(column).unwrap(),
            )
        })
        .collect();
    let draft = table.schema().apply_changes(changes).unwrap();
    let schema_path = table.schema_manager().schema_path(draft.id());
    table
        .file_io()
        .new_output(&schema_path)
        .unwrap()
        .write(Bytes::from(serde_json::to_vec(&draft).unwrap()))
        .await
        .unwrap();
    let catalog_view = catalog.get_table(&identifier).await.unwrap();
    assert_eq!(catalog_view.schema().id(), draft.id());
    assert_eq!(
        catalog_view
            .snapshot_manager()
            .get_latest_snapshot_id()
            .await
            .unwrap(),
        Some(baseline)
    );
}

async fn catalog(path: &Path) -> FileSystemCatalog {
    let mut options = Options::new();
    options.set(CatalogOptions::WAREHOUSE, path.to_str().unwrap());
    let catalog = FileSystemCatalog::new(options).unwrap();
    catalog
        .create_database("test", true, HashMap::new())
        .await
        .unwrap();
    catalog
}

async fn write(table: &Table, batch: &RecordBatch) {
    let builder = table.new_write_builder();
    let mut writer = builder.new_write().unwrap();
    writer.write_arrow_batch(batch).await.unwrap();
    let prepared = writer.prepare_commit().await.unwrap();
    Box::pin(builder.new_commit().commit(prepared))
        .await
        .unwrap();
}

fn blob_rows(ids: Range<i32>, uri: &str) -> RecordBatch {
    let descriptors = LargeBinaryArray::from_iter_values(
        ids.clone()
            .map(|id| BlobDescriptor::new(uri.to_owned(), i64::from(id), 1).serialize()),
    );
    RecordBatch::try_new(
        Arc::new(Schema::new(vec![
            Field::new("id", DataType::Int32, false),
            Field::new("payload", DataType::LargeBinary, true),
        ])),
        vec![
            Arc::new(Int32Array::from_iter_values(ids)),
            Arc::new(descriptors),
        ],
    )
    .unwrap()
}
