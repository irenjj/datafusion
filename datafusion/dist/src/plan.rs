// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: See the NOTICE file distributed with this work for additional information regarding copyright ownership
// SPDX-FileContributor: Licensed to the Apache Software Foundation (ASF) under one or more contributor license agreements

use std::sync::Arc;

use datafusion::arrow::{datatypes::SchemaRef, record_batch::RecordBatch};
use datafusion::common::{Result, Statistics, plan_datafusion_err, plan_err};
use datafusion::datasource::{
    listing::PartitionedFile,
    physical_plan::{
        FileGroup, FileScanConfig, FileScanConfigBuilder, FileSource, ParquetSource,
        parquet::{CachedParquetFileReaderFactory, DefaultParquetFileReaderFactory},
    },
    source::DataSourceExec,
};
use datafusion::execution::{TaskContext, context::SessionContext};
use datafusion::llm::LlmExec;
use datafusion::physical_expr::{Distribution, LexOrdering, Partitioning};
// DataFusion still decodes this operator in existing physical plans.
#[expect(deprecated)]
use datafusion::physical_plan::coalesce_batches::CoalesceBatchesExec;
use datafusion::physical_plan::{
    ChildrenPropertiesMode, ExecutionPlan, ReplaceChildrenOptions, filter::FilterExec,
    projection::ProjectionExec, repartition::RepartitionExec,
};
use datafusion_proto::bytes::{
    physical_plan_from_bytes_with_extension_codec,
    physical_plan_to_bytes_with_extension_codec,
};
use datafusion_proto::physical_plan::{
    DefaultPhysicalProtoConverter, PhysicalPlanDecodeContext,
    from_proto::parse_physical_sort_exprs, to_proto::serialize_physical_sort_exprs,
};
use datafusion_proto::protobuf;
use prost::Message;

use crate::codec::LlmPhysicalCodec;

/// One file or byte range, with IDs scoped to its stage.
#[derive(Clone, Debug)]
pub struct Split {
    pub split_id: u64,
    pub attempt_id: u32,
    /// Protobuf envelope containing a DataFusion PartitionedFile and supplemental metadata.
    pub file_proto: Vec<u8>,
}

/// An optimized map-only stage with an empty Parquet scan.
#[derive(Clone, Debug)]
pub struct StagePlan {
    pub stage_id: u64,
    /// DataFusion physical plan protobuf produced by the local coordinator.
    pub plan_bytes: Vec<u8>,
    pub output_schema: SchemaRef,
}

/// A validated stage template decoded with one worker's runtime and functions.
#[derive(Debug)]
pub struct PreparedStage {
    stage_id: u64,
    template: Arc<dyn ExecutionPlan>,
    pub(crate) task_ctx: Arc<TaskContext>,
}

/// Counters collected from one split attempt.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct SplitMetrics {
    pub output_rows: usize,
    pub submitted_prompts: usize,
    pub row_groups_pruned_statistics: usize,
    pub row_groups_matched_statistics: usize,
}

/// Batches and counters from a completed split attempt.
#[derive(Debug)]
pub struct SplitOutput {
    pub split_id: u64,
    pub attempt_id: u32,
    pub batches: Vec<RecordBatch>,
    pub metrics: SplitMetrics,
}

#[derive(Clone, PartialEq, Message)]
struct SplitFileEnvelope {
    #[prost(message, optional, tag = "1")]
    file: Option<protobuf::PartitionedFile>,
    #[prost(string, optional, tag = "2")]
    e_tag: Option<String>,
    #[prost(string, optional, tag = "3")]
    version: Option<String>,
    #[prost(uint64, optional, tag = "4")]
    metadata_size_hint: Option<u64>,
    #[prost(message, repeated, tag = "5")]
    ordering: Vec<protobuf::PhysicalSortExprNode>,
    #[prost(message, optional, tag = "6")]
    table_reference: Option<protobuf::TableReference>,
}

impl SplitFileEnvelope {
    fn encode_file(file: &PartitionedFile) -> Result<Vec<u8>> {
        if !file.extensions.is_empty() {
            return plan_err!(
                "Map-only distribution cannot encode PartitionedFile.extensions"
            );
        }
        let ordering = match &file.ordering {
            Some(ordering) => serialize_physical_sort_exprs(
                ordering.iter().cloned(),
                &LlmPhysicalCodec,
                &DefaultPhysicalProtoConverter {},
            )?,
            None => vec![],
        };
        Ok(Self {
            file: Some(file.try_into()?),
            e_tag: file.object_meta.e_tag.clone(),
            version: file.object_meta.version.clone(),
            metadata_size_hint: file.metadata_size_hint.map(|size| size as u64),
            ordering,
            table_reference: file.table_reference.clone().map(Into::into),
        }
        .encode_to_vec())
    }

    fn decode_file(
        bytes: &[u8],
        scan: &FileScanConfig,
        task_ctx: &TaskContext,
    ) -> Result<PartitionedFile> {
        let envelope = Self::decode(bytes)
            .map_err(|error| plan_datafusion_err!("Invalid split protobuf: {error}"))?;
        let Some(file_proto) = envelope.file else {
            return plan_err!("Split protobuf is missing its file");
        };
        if file_proto.path.is_empty() {
            return plan_err!("Split protobuf is missing its file path");
        }
        let mut file = PartitionedFile::try_from(&file_proto)?;
        file.object_meta.e_tag = envelope.e_tag;
        file.object_meta.version = envelope.version;
        file.metadata_size_hint = envelope
            .metadata_size_hint
            .map(usize::try_from)
            .transpose()
            .map_err(|error| {
                plan_datafusion_err!("Invalid split metadata size hint: {error}")
            })?;
        if !envelope.ordering.is_empty() {
            let decode_ctx = PhysicalPlanDecodeContext::new(task_ctx, &LlmPhysicalCodec);
            file.ordering = LexOrdering::new(parse_physical_sort_exprs(
                &envelope.ordering,
                &decode_ctx,
                scan.file_source().table_schema().table_schema(),
                &DefaultPhysicalProtoConverter {},
            )?);
        }
        file.table_reference = envelope
            .table_reference
            .map(TryInto::try_into)
            .transpose()?;
        Ok(file)
    }
}

/// Extract the planned files/ranges and serialize a template with no file groups.
pub fn split_plan_for_distribution(
    plan: Arc<dyn ExecutionPlan>,
    stage_id: u64,
) -> Result<(StagePlan, Vec<Split>)> {
    let output_schema = plan.schema();
    let mut splits = Vec::new();
    let template = rewrite_map_plan(plan, true, &mut |scan, _| {
        for file in scan.file_groups.iter().flat_map(FileGroup::files) {
            splits.push(Split {
                split_id: splits.len() as u64,
                attempt_id: 0,
                file_proto: SplitFileEnvelope::encode_file(file)?,
            });
        }
        let config = FileScanConfigBuilder::from(scan.clone())
            .with_file_groups(vec![])
            .with_statistics(unknown_statistics(scan))
            .build();
        Ok(Some(DataSourceExec::from_data_source(config)))
    })?;
    let plan_bytes =
        physical_plan_to_bytes_with_extension_codec(template, &LlmPhysicalCodec)?;
    Ok((
        StagePlan {
            stage_id,
            plan_bytes: plan_bytes.to_vec(),
            output_schema,
        },
        splits,
    ))
}

/// Decode a coordinator-produced template in the worker and bind one file or range.
pub fn bind_split_to_plan(
    stage: &StagePlan,
    split: &Split,
    worker_ctx: &SessionContext,
) -> Result<Arc<dyn ExecutionPlan>> {
    bind_split_to_template(&decode_stage_template(stage, worker_ctx)?, split)
}

/// Decode and validate a coordinator-produced template once in the receiving session.
pub fn decode_stage_template(
    stage: &StagePlan,
    worker_ctx: &SessionContext,
) -> Result<PreparedStage> {
    let task_ctx = worker_ctx.task_ctx();
    let template = physical_plan_from_bytes_with_extension_codec(
        &stage.plan_bytes,
        &task_ctx,
        &LlmPhysicalCodec,
    )?;
    let template = rewrite_map_plan(template, false, &mut |scan, _| {
        if !scan.file_groups.is_empty() {
            return plan_err!("Map-only stage template must have an empty Parquet scan");
        }
        Ok(None)
    })?;
    if template.schema() != stage.output_schema {
        return plan_err!(
            "Template output schema differs from stage {}",
            stage.stage_id
        );
    }
    Ok(PreparedStage {
        stage_id: stage.stage_id,
        template,
        task_ctx,
    })
}

/// Create an independent single-partition execution plan from a prepared template.
pub fn bind_split_to_template(
    stage: &PreparedStage,
    split: &Split,
) -> Result<Arc<dyn ExecutionPlan>> {
    let plan =
        rewrite_map_plan(Arc::clone(&stage.template), false, &mut |scan, parquet| {
            let mut file =
                SplitFileEnvelope::decode_file(&split.file_proto, scan, &stage.task_ctx)?;
            if file.partition_values.len() != scan.table_partition_cols().len()
                || file
                    .partition_values
                    .iter()
                    .zip(scan.table_partition_cols())
                    .any(|(value, field)| &value.data_type() != field.data_type())
            {
                return plan_err!(
                    "Split Hive partition values do not match the stage partition columns"
                );
            }
            if let Some(statistics) = &file.statistics
                && statistics.column_statistics.len()
                    != scan
                        .file_source()
                        .table_schema()
                        .table_schema()
                        .fields()
                        .len()
            {
                return plan_err!("Split statistics do not match the stage table schema");
            }
            if let Some(range) = &file.range
                && (range.start < 0
                    || range.start >= range.end
                    || range.end as u64 > file.object_meta.size)
            {
                return plan_err!("Split byte range is outside the file or empty");
            }
            // File-level counts and bounds may describe the whole file even when
            // DataFusion's scan repartitioner has assigned a range of it.
            if file.range.is_some() {
                file.statistics = None;
            }
            let statistics = file
                .statistics
                .as_deref()
                .cloned()
                .unwrap_or_else(|| unknown_statistics(scan));
            let config = FileScanConfigBuilder::from(scan.clone())
                .with_source(fresh_parquet_source(parquet))
                .with_file_groups(vec![FileGroup::new(vec![file])])
                .with_statistics(statistics)
                .build();
            Ok(Some(DataSourceExec::from_data_source(config)))
        })?;
    if plan.schema() != stage.template.schema() {
        return plan_err!(
            "Bound plan output schema differs from stage {}",
            stage.stage_id
        );
    }
    if plan.properties().output_partitioning().partition_count() != 1 {
        return plan_err!(
            "Bound plan for stage {} must have one partition",
            stage.stage_id
        );
    }
    Ok(plan)
}

#[expect(deprecated)]
fn rewrite_map_plan(
    plan: Arc<dyn ExecutionPlan>,
    strip_round_robin: bool,
    replace_scan: &mut impl FnMut(
        &FileScanConfig,
        &ParquetSource,
    ) -> Result<Option<Arc<dyn ExecutionPlan>>>,
) -> Result<Arc<dyn ExecutionPlan>> {
    let name = plan.name();
    if plan.required_input_ordering().iter().any(Option::is_some)
        || plan
            .required_input_distribution()
            .iter()
            .any(|d| !matches!(d, Distribution::UnspecifiedDistribution))
    {
        return plan_err!(
            "Map-only distribution cannot satisfy {name} input ordering or distribution"
        );
    }

    if let Some(source) = plan.downcast_ref::<DataSourceExec>() {
        let Some((scan, parquet)) = source.downcast_to_file_source::<ParquetSource>()
        else {
            return plan_err!("Map-only distribution requires a Parquet FileScanConfig");
        };
        if scan.limit.is_some() {
            return plan_err!(
                "Map-only distribution does not support FileScanConfig.limit"
            );
        }
        if scan.preserve_order
            || !scan.output_ordering.is_empty()
            || scan.output_partitioning.is_some()
            || scan.expr_adapter_factory.is_some()
        {
            return plan_err!(
                "Map-only distribution does not support ordered, explicitly partitioned, or custom-adapter Parquet scans"
            );
        }
        if parquet.uses_encryption() {
            return plan_err!(
                "Map-only distribution does not support Parquet encryption settings"
            );
        }
        if parquet
            .parquet_file_reader_factory()
            .is_some_and(|factory| {
                !factory.is::<CachedParquetFileReaderFactory>()
                    && !factory.is::<DefaultParquetFileReaderFactory>()
            })
        {
            return plan_err!(
                "Map-only distribution does not support custom Parquet reader factories"
            );
        }
        return Ok(replace_scan(scan, parquet)?.unwrap_or(plan));
    }

    if plan.fetch().is_some() {
        return plan_err!(
            "Map-only distribution does not support {name} with a fetch limit"
        );
    }
    let strip = if let Some(repartition) = plan.downcast_ref::<RepartitionExec>() {
        if !strip_round_robin
            || !matches!(repartition.partitioning(), Partitioning::RoundRobinBatch(_))
            || repartition.preserve_order()
        {
            return plan_err!(
                "Map-only distribution does not support this {name} partitioning"
            );
        }
        true
    } else if plan.is::<ProjectionExec>()
        || plan.is::<FilterExec>()
        || plan.is::<CoalesceBatchesExec>()
        || plan.is::<LlmExec>()
    {
        false
    } else {
        return plan_err!("Unsupported map-only operator {name}");
    };
    let [child] = plan.children()[..] else {
        return plan_err!("Map-only operator {name} requires exactly one input");
    };
    let rewritten = rewrite_map_plan(Arc::clone(child), strip_round_robin, replace_scan)?;
    if strip {
        Ok(rewritten)
    } else if Arc::ptr_eq(child, &rewritten) {
        Ok(plan)
    } else {
        plan.replace_children(
            vec![rewritten],
            ReplaceChildrenOptions::new(ChildrenPropertiesMode::Recompute),
        )
    }
}

fn fresh_parquet_source(template: &ParquetSource) -> Arc<dyn FileSource> {
    Arc::new(template.with_fresh_metrics())
}

fn unknown_statistics(scan: &FileScanConfig) -> Statistics {
    Statistics::new_unknown(scan.file_source().table_schema().table_schema())
}
