// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: See the NOTICE file distributed with this work for additional information regarding copyright ownership
// SPDX-FileContributor: Licensed to the Apache Software Foundation (ASF) under one or more contributor license agreements

use std::collections::{BTreeMap, BTreeSet};
use std::fmt::Debug;
use std::sync::Arc;

use datafusion::arrow::{datatypes::SchemaRef, record_batch::RecordBatch};
use datafusion::common::extensions::Extensions as FileExtensions;
use datafusion::common::{
    DataFusionError, Result, Statistics, plan_datafusion_err, plan_err,
};
use datafusion::datasource::{
    listing::PartitionedFile,
    physical_plan::{
        FileGroup, FileScanConfig, FileScanConfigBuilder, FileSource, ParquetSource,
        parquet::{CachedParquetFileReaderFactory, DefaultParquetFileReaderFactory},
    },
    source::DataSourceExec,
};
use datafusion::execution::{TaskContext, context::SessionContext};
use datafusion::physical_expr::{Distribution, LexOrdering, Partitioning};
use datafusion::physical_plan::{
    ChildrenPropertiesMode, ExecutionPlan, ReplaceChildrenOptions,
    repartition::RepartitionExec,
};
use datafusion_proto::bytes::{
    physical_plan_from_bytes_with_proto_converter,
    physical_plan_to_bytes_with_proto_converter,
};
use datafusion_proto::physical_plan::{
    DefaultPhysicalProtoConverter, PhysicalPlanDecodeContext,
    from_proto::parse_physical_sort_exprs, to_proto::serialize_physical_sort_exprs,
};
use datafusion_proto::protobuf;
use prost::Message;

use crate::codec::{
    DistExtensionRegistry, DistPhysicalCodec, MapPlanProtoConverter, is_builtin,
};
use crate::fingerprint::{plan_fingerprint, split_fingerprint};

/// Operator and file extension codecs shared by a coordinator and its workers.
#[derive(Clone, Debug, Default)]
pub struct DistCodecs {
    /// Registry for physical extension operators.
    pub physical: DistExtensionRegistry,
    /// Registry for typed [`PartitionedFile::extensions`].
    pub file_extensions: FileExtensionCodecRegistry,
}

impl DistCodecs {
    /// Set the registry for physical extension operators.
    pub fn with_physical_codecs(mut self, physical: DistExtensionRegistry) -> Self {
        self.physical = physical;
        self
    }

    /// Set the registry for typed [`PartitionedFile::extensions`].
    pub fn with_extension_codecs(
        mut self,
        file_extensions: FileExtensionCodecRegistry,
    ) -> Self {
        self.file_extensions = file_extensions;
        self
    }
}

/// One file or byte range, with IDs scoped to its stage.
#[derive(Clone, Debug)]
pub struct Split {
    /// Split identifier within the stage.
    pub split_id: u64,
    /// Execution attempt identifier within the split.
    pub attempt_id: u32,
    /// Protobuf envelope containing a DataFusion PartitionedFile and supplemental metadata.
    pub file_proto: Vec<u8>,
    /// SHA-256 fingerprint of the stage fingerprint and file protobuf.
    pub fingerprint: String,
}

/// An optimized map-only stage with an empty Parquet scan.
#[derive(Clone, Debug)]
pub struct StagePlan {
    /// Coordinator-assigned stage identifier.
    pub stage_id: u64,
    /// DataFusion physical plan protobuf produced by the local coordinator.
    pub plan_bytes: Vec<u8>,
    /// Schema of batches returned by the stage.
    pub output_schema: SchemaRef,
    /// SHA-256 fingerprint of the physical plan protobuf and its output schema.
    pub plan_fingerprint: String,
}

/// A validated stage template for one worker's runtime and functions.
#[derive(Debug)]
pub struct PreparedStage {
    stage_id: u64,
    plan_bytes: Vec<u8>,
    plan_fingerprint: String,
    output_schema: SchemaRef,
    pub(crate) task_ctx: Arc<TaskContext>,
    codecs: DistCodecs,
}

/// Counters collected from one split attempt.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct SplitMetrics {
    /// Rows returned by the stage root.
    pub output_rows: usize,
    /// Prompt submissions reported by all operators.
    pub submitted_prompts: usize,
    /// Parquet row groups eliminated by statistics.
    pub row_groups_pruned_statistics: usize,
    /// Parquet row groups matched by statistics.
    pub row_groups_matched_statistics: usize,
}

/// Batches and counters from a completed split attempt.
#[derive(Debug)]
pub struct SplitOutput {
    /// Split identifier within the stage.
    pub split_id: u64,
    /// Execution attempt identifier within the split.
    pub attempt_id: u32,
    /// Batches returned by the attempt.
    pub batches: Vec<RecordBatch>,
    /// Counters for this attempt.
    pub metrics: SplitMetrics,
}

/// Serializes one concrete type stored in [`PartitionedFile::extensions`].
///
/// Use a stable wire name and a distinct name for each registered Rust type.
/// Encodings must be deterministic for the same value and version.
pub trait FileExtensionCodec: Debug + Send + Sync {
    /// Return the stable, nonblank wire name of this extension.
    fn extension_name(&self) -> &str;

    /// Encode this codec's extension as a version and payload, or return
    /// `Ok(None)` when its type is absent.
    fn try_encode(&self, extensions: &FileExtensions) -> Result<Option<(u32, Vec<u8>)>>;

    /// Decode a versioned payload and insert exactly one value of this codec's type.
    ///
    /// Return an error for an unsupported version or invalid payload.
    fn try_decode(
        &self,
        version: u32,
        payload: &[u8],
        extensions: &mut FileExtensions,
    ) -> Result<()>;
}

/// Codecs indexed by wire name for deterministic split encoding.
///
/// Clones share the registered codecs. Registering on a clone leaves the
/// original registry's entries intact.
#[derive(Clone, Debug, Default)]
pub struct FileExtensionCodecRegistry {
    codecs: Arc<BTreeMap<String, Arc<dyn FileExtensionCodec>>>,
}

impl FileExtensionCodecRegistry {
    /// Register a codec. Blank names and duplicate names are plan errors.
    pub fn register(&mut self, codec: Arc<dyn FileExtensionCodec>) -> Result<()> {
        let name = codec.extension_name();
        if name.trim().is_empty() {
            return plan_err!("File extension codec name must not be empty");
        }
        if self.codecs.contains_key(name) {
            return plan_err!("File extension codec '{name}' is already registered");
        }
        Arc::make_mut(&mut self.codecs).insert(name.to_owned(), codec);
        Ok(())
    }

    fn encode_extensions(
        &self,
        extensions: &FileExtensions,
    ) -> Result<Vec<SplitFileExtensionEntry>> {
        let mut entries = Vec::with_capacity(extensions.len());
        for (name, codec) in self.codecs.iter() {
            let encoded = codec.try_encode(extensions).map_err(|error| {
                plan_datafusion_err!(
                    "Cannot encode PartitionedFile.extensions entry '{name}': {error}"
                )
            })?;
            if let Some((version, payload)) = encoded {
                entries.push(SplitFileExtensionEntry {
                    extension_name: name.clone(),
                    version,
                    payload,
                });
            }
        }
        let encoded_count = entries.len();
        if encoded_count != extensions.len() {
            return plan_err!(
                "Map-only distribution cannot encode PartitionedFile.extensions: encoded {encoded_count} of {}",
                extensions.len()
            );
        }
        Ok(entries)
    }

    fn decode_extensions(
        &self,
        entries: &[SplitFileExtensionEntry],
        extensions: &mut FileExtensions,
    ) -> Result<()> {
        let mut seen = BTreeSet::new();
        for entry in entries {
            let name = entry.extension_name.as_str();
            if name.trim().is_empty() {
                return plan_err!("Split file extension name must not be empty");
            }
            if !seen.insert(name) {
                return plan_err!("Duplicate split file extension '{name}'");
            }
            let codec = self.codecs.get(name).ok_or_else(|| {
                plan_datafusion_err!("Unregistered split file extension '{name}'")
            })?;
            // Isolate each decoder so it cannot change a previously decoded value.
            let mut decoded = FileExtensions::new();
            codec
                .try_decode(entry.version, &entry.payload, &mut decoded)
                .map_err(|error| {
                    plan_datafusion_err!(
                        "Cannot decode split file extension '{name}': {error}"
                    )
                })?;
            if decoded.len() != 1 {
                return plan_err!(
                    "Split file extension '{name}' must decode exactly one value"
                );
            }
            let previous_len = extensions.len();
            extensions.merge(&decoded);
            if extensions.len() != previous_len + 1 {
                return plan_err!(
                    "Split file extension '{name}' repeats a decoded Rust type"
                );
            }
        }
        Ok(())
    }
}

#[derive(Clone, PartialEq, Message)]
struct SplitFileExtensionEntry {
    #[prost(string, tag = "1")]
    extension_name: String,
    #[prost(uint32, tag = "2")]
    version: u32,
    #[prost(bytes = "vec", tag = "3")]
    payload: Vec<u8>,
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
    #[prost(message, repeated, tag = "7")]
    extensions: Vec<SplitFileExtensionEntry>,
}

impl SplitFileEnvelope {
    fn encode_file(
        file: &PartitionedFile,
        physical_codec: &DistPhysicalCodec,
        extension_codecs: &FileExtensionCodecRegistry,
    ) -> Result<Vec<u8>> {
        let extensions = extension_codecs.encode_extensions(&file.extensions)?;
        let ordering = match &file.ordering {
            Some(ordering) => serialize_physical_sort_exprs(
                ordering.iter().cloned(),
                physical_codec,
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
            extensions,
        }
        .encode_to_vec())
    }

    fn decode_file(
        bytes: &[u8],
        scan: &FileScanConfig,
        task_ctx: &TaskContext,
        physical_codec: &DistPhysicalCodec,
        extension_codecs: &FileExtensionCodecRegistry,
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
            let decode_ctx = PhysicalPlanDecodeContext::new(task_ctx, physical_codec);
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
        extension_codecs.decode_extensions(&envelope.extensions, &mut file.extensions)?;
        Ok(file)
    }
}

/// Extract the planned files/ranges and serialize a template with no file groups.
pub fn split_plan_for_distribution(
    plan: Arc<dyn ExecutionPlan>,
    stage_id: u64,
) -> Result<(StagePlan, Vec<Split>)> {
    split_plan_for_distribution_with_codecs(plan, stage_id, &DistCodecs::default())
}

/// Extract the files and ranges using the supplied file extension codecs.
pub fn split_plan_for_distribution_with_extensions(
    plan: Arc<dyn ExecutionPlan>,
    stage_id: u64,
    extension_codecs: &FileExtensionCodecRegistry,
) -> Result<(StagePlan, Vec<Split>)> {
    split_plan_for_distribution_with_codecs(
        plan,
        stage_id,
        &DistCodecs::default().with_extension_codecs(extension_codecs.clone()),
    )
}

/// Extract the files and ranges and encode a template using the supplied [`DistCodecs`].
pub fn split_plan_for_distribution_with_codecs(
    plan: Arc<dyn ExecutionPlan>,
    stage_id: u64,
    codecs: &DistCodecs,
) -> Result<(StagePlan, Vec<Split>)> {
    let output_schema = plan.schema();
    let codec = DistPhysicalCodec::new(codecs.physical.clone());
    let mut files = Vec::new();
    let template = rewrite_map_plan(plan, true, &codecs.physical, &mut |scan| {
        for file in scan.file_groups.iter().flat_map(FileGroup::files) {
            files.push(
                SplitFileEnvelope::encode_file(file, &codec, &codecs.file_extensions)
                    .map_err(as_plan_error)?,
            );
        }
        let config = FileScanConfigBuilder::from(scan.clone())
            .with_file_groups(vec![])
            .with_statistics(unknown_statistics(scan))
            .build();
        Ok(Some(DataSourceExec::from_data_source(config)))
    })?;
    if template.schema() != output_schema {
        return plan_err!("Map-only stage {stage_id} changed its output schema");
    }
    let plan_bytes = physical_plan_to_bytes_with_proto_converter(
        template,
        &codec,
        &MapPlanProtoConverter,
    )
    .map_err(as_plan_error)?
    .to_vec();
    let plan_fingerprint = plan_fingerprint(&plan_bytes, output_schema.as_ref())?;
    let splits = files
        .into_iter()
        .enumerate()
        .map(|(index, file_proto)| Split {
            split_id: index as u64,
            attempt_id: 0,
            fingerprint: split_fingerprint(&plan_fingerprint, &file_proto),
            file_proto,
        })
        .collect();
    Ok((
        StagePlan {
            stage_id,
            plan_bytes,
            output_schema,
            plan_fingerprint,
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
    bind_split_to_plan_with_codecs(stage, split, worker_ctx, &DistCodecs::default())
}

/// Decode the stage and bind one split using the supplied file extension codecs.
pub fn bind_split_to_plan_with_extensions(
    stage: &StagePlan,
    split: &Split,
    worker_ctx: &SessionContext,
    extension_codecs: &FileExtensionCodecRegistry,
) -> Result<Arc<dyn ExecutionPlan>> {
    bind_split_to_plan_with_codecs(
        stage,
        split,
        worker_ctx,
        &DistCodecs::default().with_extension_codecs(extension_codecs.clone()),
    )
}

/// Decode the stage and bind one split using the supplied [`DistCodecs`].
pub fn bind_split_to_plan_with_codecs(
    stage: &StagePlan,
    split: &Split,
    worker_ctx: &SessionContext,
    codecs: &DistCodecs,
) -> Result<Arc<dyn ExecutionPlan>> {
    let prepared = decode_stage_template_with_codecs(stage, worker_ctx, codecs)?;
    bind_split_to_template(&prepared, split)
}

/// Decode and validate a coordinator-produced template once in the receiving session.
pub fn decode_stage_template(
    stage: &StagePlan,
    worker_ctx: &SessionContext,
) -> Result<PreparedStage> {
    decode_stage_template_with_codecs(stage, worker_ctx, &DistCodecs::default())
}

/// Decode and validate a stage and retain the supplied file extension codecs.
pub fn decode_stage_template_with_extensions(
    stage: &StagePlan,
    worker_ctx: &SessionContext,
    extension_codecs: &FileExtensionCodecRegistry,
) -> Result<PreparedStage> {
    decode_stage_template_with_codecs(
        stage,
        worker_ctx,
        &DistCodecs::default().with_extension_codecs(extension_codecs.clone()),
    )
}

/// Decode and validate a stage in the receiving session using the supplied [`DistCodecs`].
pub fn decode_stage_template_with_codecs(
    stage: &StagePlan,
    worker_ctx: &SessionContext,
    codecs: &DistCodecs,
) -> Result<PreparedStage> {
    if plan_fingerprint(&stage.plan_bytes, stage.output_schema.as_ref())?
        != stage.plan_fingerprint
    {
        return plan_err!("Plan fingerprint does not match stage {}", stage.stage_id);
    }
    let task_ctx = worker_ctx.task_ctx();
    let codec = DistPhysicalCodec::new(codecs.physical.clone());
    let template = decode_plan(&stage.plan_bytes, &task_ctx, &codec)?;
    let template = rewrite_map_plan(template, false, &codecs.physical, &mut |scan| {
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
        plan_bytes: stage.plan_bytes.clone(),
        plan_fingerprint: stage.plan_fingerprint.clone(),
        output_schema: Arc::clone(&stage.output_schema),
        task_ctx,
        codecs: codecs.clone(),
    })
}

/// Create an independent single-partition plan using the prepared stage's codecs.
pub fn bind_split_to_template(
    stage: &PreparedStage,
    split: &Split,
) -> Result<Arc<dyn ExecutionPlan>> {
    bind_split_to_template_with_extensions(stage, split, &stage.codecs.file_extensions)
}

/// Bind a split to a prepared template using the supplied file extension
/// codecs for this binding.
pub fn bind_split_to_template_with_extensions(
    stage: &PreparedStage,
    split: &Split,
    extension_codecs: &FileExtensionCodecRegistry,
) -> Result<Arc<dyn ExecutionPlan>> {
    if split_fingerprint(&stage.plan_fingerprint, &split.file_proto) != split.fingerprint
    {
        return plan_err!(
            "Split {} fingerprint does not match stage {}",
            split.split_id,
            stage.stage_id
        );
    }
    let codec = DistPhysicalCodec::new(stage.codecs.physical.clone());
    // Deserializing for each attempt gives the scan and all operators fresh metrics.
    let template = decode_plan(&stage.plan_bytes, &stage.task_ctx, &codec)?;
    let plan = rewrite_map_plan(template, false, &stage.codecs.physical, &mut |scan| {
        let mut file = SplitFileEnvelope::decode_file(
            &split.file_proto,
            scan,
            &stage.task_ctx,
            &codec,
            extension_codecs,
        )
        .map_err(as_plan_error)?;
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
            .with_file_groups(vec![FileGroup::new(vec![file])])
            .with_statistics(statistics)
            .build();
        Ok(Some(DataSourceExec::from_data_source(config)))
    })?;
    if plan.schema() != stage.output_schema {
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

fn rewrite_map_plan(
    plan: Arc<dyn ExecutionPlan>,
    strip_round_robin: bool,
    registry: &DistExtensionRegistry,
    replace_scan: &mut impl FnMut(&FileScanConfig) -> Result<Option<Arc<dyn ExecutionPlan>>>,
) -> Result<Arc<dyn ExecutionPlan>> {
    let name = plan.name();
    let distribution = plan.input_distribution_requirements();
    if plan.required_input_ordering().iter().any(Option::is_some)
        || distribution.is_co_partitioned()
        || distribution
            .per_child_distributions()
            .any(|d| !matches!(d, Distribution::UnspecifiedDistribution))
    {
        return plan_err!(
            "Map-only distribution cannot satisfy {name} input ordering or distribution"
        );
    }

    if let Some(source) = plan.downcast_ref::<DataSourceExec>()
        && plan.downcast_delegate().is_none()
    {
        let Some((scan, parquet)) = source.downcast_to_file_source::<ParquetSource>()
        else {
            return plan_err!("Map-only distribution requires a Parquet FileScanConfig");
        };
        if scan.limit.is_some() {
            return plan_err!(
                "Map-only distribution does not support FileScanConfig.limit"
            );
        }
        if !parquet.table_schema().virtual_columns().is_empty() {
            return plan_err!(
                "Map-only distribution does not support Parquet scans with virtual columns"
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
        if parquet.table_parquet_options().crypto != Default::default() {
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
        return Ok(replace_scan(scan)?.unwrap_or(plan));
    }

    if plan.fetch().is_some() {
        return plan_err!(
            "Map-only distribution does not support {name} with a fetch limit"
        );
    }
    let strip = if let Some(repartition) = plan.downcast_ref::<RepartitionExec>()
        && plan.downcast_delegate().is_none()
    {
        if !strip_round_robin
            || !matches!(repartition.partitioning(), Partitioning::RoundRobinBatch(_))
            || repartition.preserve_order()
        {
            return plan_err!(
                "Map-only distribution does not support this {name} partitioning"
            );
        }
        true
    } else if is_builtin(plan.as_ref())
        || registry
            .policy_for(plan.as_ref())?
            .is_some_and(|policy| policy.map_only)
    {
        false
    } else {
        return plan_err!("Unsupported map-only operator {name}");
    };
    let [child] = plan.children()[..] else {
        return plan_err!("Map-only operator {name} requires exactly one input");
    };
    let rewritten =
        rewrite_map_plan(Arc::clone(child), strip_round_robin, registry, replace_scan)?;
    if strip {
        Ok(rewritten)
    } else if Arc::ptr_eq(child, &rewritten) {
        Ok(plan)
    } else {
        plan.replace_children(
            vec![rewritten],
            ReplaceChildrenOptions::new(ChildrenPropertiesMode::Recompute),
        )
        .map_err(as_plan_error)
    }
}

fn decode_plan(
    bytes: &[u8],
    task_ctx: &TaskContext,
    codec: &DistPhysicalCodec,
) -> Result<Arc<dyn ExecutionPlan>> {
    physical_plan_from_bytes_with_proto_converter(
        bytes,
        task_ctx,
        codec,
        &MapPlanProtoConverter,
    )
    .map_err(as_plan_error)
}

fn as_plan_error(error: DataFusionError) -> DataFusionError {
    if matches!(error.find_root(), DataFusionError::Plan(_)) {
        error
    } else {
        plan_datafusion_err!("{error}")
    }
}

fn unknown_statistics(scan: &FileScanConfig) -> Statistics {
    Statistics::new_unknown(scan.file_source().table_schema().table_schema())
}
