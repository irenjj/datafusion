// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: See the NOTICE file distributed with this work for additional information regarding copyright ownership
// SPDX-FileContributor: Licensed to the Apache Software Foundation (ASF) under one or more contributor license agreements

use std::collections::HashMap;
use std::future::ready;
use std::path::Path;

use arrow::array::RecordBatch;
use arrow::datatypes::Field;
use datafusion_common::{DataFusionError, Result};
use datafusion_execution::SendableRecordBatchStream;
use datafusion_physical_plan::stream::RecordBatchStreamAdapter;
use futures::{TryStreamExt, future::BoxFuture};
use paimon::arrow::{arrow_fields_to_paimon, build_read_arrow_schema};
use paimon::catalog::Identifier;
use paimon::spec::{Schema as PaimonSchema, SchemaBuilder};
use paimon::{Catalog, CatalogOptions, DataSplit, FileSystemCatalog, Options};

pub(super) fn parquet_schema(fields: &[Field]) -> PaimonSchema {
    arrow_fields_to_paimon(fields)
        .unwrap()
        .into_iter()
        .fold(SchemaBuilder::new(), SchemaBuilder::field)
        .option("file.format", "parquet")
        .option("read.batch-size", "2")
        .build()
        .unwrap()
}

pub(super) async fn stage_catalog(path: &Path) -> FileSystemCatalog {
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
pub(super) struct ParquetWinner {
    table: paimon::Table,
    split: DataSplit,
}

pub(super) fn write_parquet_winner<'a>(
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

pub(super) fn open_parquet(
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
