// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: See the NOTICE file distributed with this work for additional information regarding copyright ownership
// SPDX-FileContributor: Licensed to the Apache Software Foundation (ASF) under one or more contributor license agreements

use std::sync::Arc;

use arrow::array::{RecordBatch, RecordBatchOptions};
use arrow::datatypes::SchemaRef;
use datafusion_common::{Result, tree_node::TreeNodeRecursion};
use datafusion_datasource::file::FileSource;
use datafusion_datasource::file_groups::FileGroup;
use datafusion_datasource::file_scan_config::{FileScanConfig, FileScanConfigBuilder};
use datafusion_datasource::file_stream::{FileOpenFuture, FileOpener};
use datafusion_datasource::source::DataSourceExec;
use datafusion_datasource::{PartitionedFile, TableSchema};
use datafusion_execution::object_store::ObjectStoreUrl;
use datafusion_physical_expr::{PhysicalExpr, projection::ProjectionExprs};
use datafusion_physical_plan::ExecutionPlan;
use datafusion_physical_plan::metrics::ExecutionPlanMetricsSet;
use futures::lock::Mutex;
use object_store::ObjectStore;

use crate::dist::LakeRuntimeConfig;
use crate::lance_table::LanceScan;
use crate::paimon_table::PaimonScan;
use crate::split::{LakeSplitExt, LakeSplitPayload};

#[derive(Clone)]
pub(crate) enum LakeScan {
    Lance(LanceScan),
    Paimon(PaimonScan),
}

impl LakeScan {
    pub(crate) fn validate_split(&self, payload: &LakeSplitPayload) -> Result<()> {
        match self {
            Self::Lance(scan) => scan.validate_split(payload),
            Self::Paimon(scan) => scan.validate_split(payload),
        }
    }

    fn is_worker(&self) -> bool {
        match self {
            Self::Lance(scan) => scan.is_worker(),
            Self::Paimon(scan) => scan.is_worker(),
        }
    }
}

pub(crate) enum ScanHandle<T> {
    Local(Arc<T>),
    Worker {
        runtime: Arc<LakeRuntimeConfig>,
        opened: Mutex<Option<Arc<T>>>,
    },
}

impl<T> ScanHandle<T> {
    pub(crate) fn worker(runtime: Arc<LakeRuntimeConfig>) -> Self {
        Self::Worker {
            runtime,
            opened: Mutex::new(None),
        }
    }

    pub(crate) fn local(&self) -> Option<&T> {
        match self {
            Self::Local(opened) => Some(opened),
            Self::Worker { .. } => None,
        }
    }

    pub(crate) async fn get_or_open<Fut: Future<Output = Result<T>>>(
        &self,
        open: impl FnOnce(Arc<LakeRuntimeConfig>) -> Fut,
    ) -> Result<Arc<T>> {
        match self {
            Self::Local(opened) => Ok(Arc::clone(opened)),
            Self::Worker { runtime, opened } => {
                let mut opened = opened.lock().await;
                if let Some(handle) = opened.as_ref() {
                    return Ok(Arc::clone(handle));
                }
                let handle = Arc::new(open(Arc::clone(runtime)).await?);
                *opened = Some(Arc::clone(&handle));
                Ok(handle)
            }
        }
    }
}

#[derive(Clone)]
pub(crate) struct LakeSource {
    pub(crate) engine: LakeScan,
    table_schema: TableSchema,
    projection: Option<ProjectionExprs>,
    output_schema: SchemaRef,
    metrics: ExecutionPlanMetricsSet,
    batch_size: usize,
}

pub(crate) fn scan_plan(
    engine: LakeScan,
    schema: SchemaRef,
    projection: Option<&[usize]>,
    files: Vec<PartitionedFile>,
    limit: Option<usize>,
) -> Result<Arc<dyn ExecutionPlan>> {
    // FileScanConfig applies its fetch per partition. A pushed table limit
    // needs one ordered group to stop opening splits once the limit is met.
    let groups = if limit.is_some() || files.is_empty() {
        vec![FileGroup::new(files)]
    } else {
        files
            .into_iter()
            .map(|file| FileGroup::new(vec![file]))
            .collect()
    };
    LakeSource::plan(engine, schema, projection, groups, limit)
}

pub(crate) fn projected_schema(
    schema: &SchemaRef,
    indices: Option<&[usize]>,
) -> Result<SchemaRef> {
    match indices {
        Some(indices) => Ok(Arc::new(schema.project(indices)?)),
        None => Ok(Arc::clone(schema)),
    }
}

impl LakeSource {
    pub(crate) fn plan(
        engine: LakeScan,
        schema: SchemaRef,
        projection: Option<&[usize]>,
        groups: Vec<FileGroup>,
        limit: Option<usize>,
    ) -> Result<Arc<dyn ExecutionPlan>> {
        let output_schema = projected_schema(&schema, projection)?;
        let projection =
            projection.map(|indices| ProjectionExprs::from_indices(indices, &schema));
        let source = Self {
            engine,
            table_schema: schema.into(),
            projection,
            output_schema,
            metrics: ExecutionPlanMetricsSet::new(),
            batch_size: 8192,
        };
        // The file opener reads through a pinned engine handle. The registry
        // lookup required by FileScanConfig does not select the lake's store.
        let config = FileScanConfigBuilder::new(
            ObjectStoreUrl::local_filesystem(),
            Arc::new(source),
        )
        .with_file_groups(groups)
        .with_limit(limit)
        .with_preserve_order(limit.is_some())
        .build();
        Ok(DataSourceExec::from_data_source(config))
    }

    pub(crate) fn with_fresh_metrics(&self) -> Self {
        Self {
            metrics: ExecutionPlanMetricsSet::new(),
            ..self.clone()
        }
    }
}

impl FileSource for LakeSource {
    fn create_file_opener(
        &self,
        _store: Arc<dyn ObjectStore>,
        config: &FileScanConfig,
        _partition: usize,
    ) -> Result<Arc<dyn FileOpener>> {
        Ok(Arc::new(LakeOpener {
            source: self.clone(),
            limit: config.limit,
        }))
    }

    fn table_schema(&self) -> &TableSchema {
        &self.table_schema
    }

    fn with_batch_size(&self, batch_size: usize) -> Arc<dyn FileSource> {
        Arc::new(Self {
            batch_size,
            ..self.clone()
        })
    }

    fn projection(&self) -> Option<&ProjectionExprs> {
        self.projection.as_ref()
    }

    fn metrics(&self) -> &ExecutionPlanMetricsSet {
        &self.metrics
    }

    fn file_type(&self) -> &str {
        match self.engine {
            LakeScan::Lance(_) => "lance",
            LakeScan::Paimon(_) => "paimon",
        }
    }

    fn apply_expressions(
        &self,
        f: &mut dyn FnMut(&Arc<dyn PhysicalExpr>) -> Result<TreeNodeRecursion>,
    ) -> Result<TreeNodeRecursion> {
        let row_filter = match &self.engine {
            LakeScan::Lance(_) => None,
            LakeScan::Paimon(scan) => scan.row_filter(),
        };
        datafusion_physical_plan::apply_expression_roots(
            row_filter.into_iter().chain(
                self.projection
                    .iter()
                    .flat_map(ProjectionExprs::iter)
                    .map(|expr| &expr.expr),
            ),
            f,
        )
    }

    fn supports_repartitioning(&self) -> bool {
        false
    }
}

struct LakeOpener {
    source: LakeSource,
    limit: Option<usize>,
}

impl FileOpener for LakeOpener {
    fn open(&self, file: PartitionedFile) -> Result<FileOpenFuture> {
        let split = LakeSplitExt::for_read(&file)?;
        if self.source.engine.is_worker() {
            split.validate_identity()?;
        }
        self.source.engine.validate_split(&split.payload)?;
        let payload = split.payload.clone();
        let source = self.source.clone();
        let limit = self.limit;
        Ok(Box::pin(async move {
            match source.engine {
                LakeScan::Lance(scan) => {
                    scan.open(payload, source.output_schema, source.batch_size, limit)
                        .await
                }
                LakeScan::Paimon(scan) => {
                    scan.open(payload, source.output_schema, limit).await
                }
            }
        }))
    }
}

pub(crate) fn make_batch(
    schema: SchemaRef,
    columns: Vec<arrow::array::ArrayRef>,
    rows: usize,
) -> Result<RecordBatch> {
    Ok(RecordBatch::try_new_with_options(
        schema,
        columns,
        &RecordBatchOptions::new().with_row_count(Some(rows)),
    )?)
}
