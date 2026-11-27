// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: See the NOTICE file distributed with this work for additional information regarding copyright ownership
// SPDX-FileContributor: Licensed to the Apache Software Foundation (ASF) under one or more contributor license agreements

use std::collections::{HashMap, HashSet};
use std::fmt::{self, Formatter};
use std::fs::File;
use std::path::Path;
use std::sync::Arc;

use datafusion::arrow::{
    array::{ArrayRef, Int64Array, LargeStringArray, StringArray, StringViewArray},
    datatypes::{DataType, Field, Schema, SchemaBuilder},
    json::reader::ValueIter,
    record_batch::RecordBatch,
};
use datafusion::common::tree_node::TreeNodeRecursion;
use datafusion::common::{
    DataFusionError, Result, ScalarValue, plan_datafusion_err, plan_err,
};
use datafusion::datasource::{
    listing::PartitionedFile,
    physical_plan::{
        FileGroup, FileScanConfig, FileScanConfigBuilder, FileSource, ParquetSource,
    },
    source::DataSourceExec,
};
use datafusion::execution::TaskContext;
use datafusion::logical_expr::Operator;
use datafusion::parquet::{arrow::ArrowWriter, file::metadata::ParquetMetaData};
use datafusion::physical_expr::expressions::{BinaryExpr, Column, LikeExpr, Literal};
use datafusion::physical_expr::{EquivalenceProperties, Partitioning, PhysicalExpr};
use datafusion::physical_plan::{
    ChildrenPropertiesMode, DisplayAs, DisplayFormatType, ExecutionPlan, PlanProperties,
    ReplaceChildrenOptions, SendableRecordBatchStream,
    filter::FilterExec,
    metrics::{ExecutionPlanMetricsSet, MetricBuilder, MetricsSet},
    projection::ProjectionExec,
    stream::RecordBatchStreamAdapter,
};
use datafusion::prelude::{ParquetReadOptions, SessionConfig, SessionContext, col, lit};
use datafusion_dist::{
    DistCodecs, DistExtensionRegistry, ExtensionOperatorPolicy, LocalWorker,
    NamedPhysicalExtensionCodec, Split, SplitOutput, StagePlan,
    split_plan_for_distribution,
};

use datafusion_proto::physical_plan::{
    PhysicalExtensionCodec, PhysicalPlanDecodeContext, PhysicalProtoConverterExtension,
};
use datafusion_proto::protobuf;
use futures::StreamExt;
use prost::Message;

pub(super) fn coordinator() -> SessionContext {
    context(8192, 1, true, false)
}

pub(super) fn context(
    batch_size: usize,
    target_partitions: usize,
    pushdown_filters: bool,
    natural_ranges: bool,
) -> SessionContext {
    let mut config = SessionConfig::new()
        .with_batch_size(batch_size)
        .with_target_partitions(target_partitions);
    config.options_mut().execution.parquet.pushdown_filters = pushdown_filters;
    config.options_mut().optimizer.repartition_file_min_size = 1;
    config.options_mut().optimizer.repartition_file_scans = natural_ranges;
    SessionContext::new_with_config(config)
}

pub(super) fn write_parquet(
    path: &Path,
    groups: &[(i64, i64)],
) -> Result<ParquetMetaData> {
    let groups = groups.iter().map(|&(id, filter)| {
        vec![
            Row {
                id,
                filter,
                prompt: Some("a prompt"),
            },
            Row {
                id: id + 1,
                filter: filter + 1,
                prompt: None,
            },
        ]
    });
    write_row_groups(path, &DataType::Utf8, groups)
}

#[derive(Clone, Copy)]
pub(super) struct Row {
    pub(super) id: i64,
    pub(super) filter: i64,
    pub(super) prompt: Option<&'static str>,
}

pub(super) fn write_row_groups(
    path: &Path,
    string_type: &DataType,
    groups: impl IntoIterator<Item = Vec<Row>>,
) -> Result<ParquetMetaData> {
    let schema = Arc::new(Schema::new(vec![
        Field::new("id", DataType::Int64, false),
        Field::new("filter_only_col", DataType::Int64, false),
        Field::new("unused_wide_col", DataType::Utf8, false),
        Field::new("prompt_col", string_type.clone(), true),
    ]));
    let mut writer =
        ArrowWriter::try_new(File::create(path)?, Arc::clone(&schema), None)?;
    let unused = "unused data".repeat(128);
    for group in groups {
        let prompts = group.iter().map(|row| row.prompt).collect::<Vec<_>>();
        let prompts: ArrayRef = match string_type {
            DataType::Utf8 => Arc::new(StringArray::from(prompts)),
            DataType::LargeUtf8 => Arc::new(LargeStringArray::from(prompts)),
            DataType::Utf8View => Arc::new(StringViewArray::from(prompts)),
            _ => unreachable!("unsupported test prompt type {string_type}"),
        };
        let batch = RecordBatch::try_new(
            Arc::clone(&schema),
            vec![
                Arc::new(Int64Array::from_iter_values(group.iter().map(|row| row.id))),
                Arc::new(Int64Array::from_iter_values(
                    group.iter().map(|row| row.filter),
                )),
                Arc::new(StringArray::from(vec![unused.as_str(); group.len()])),
                prompts,
            ],
        )?;
        writer.write(&batch)?;
        writer.flush()?;
    }
    Ok(writer.close()?)
}

pub(super) fn hive_options() -> ParquetReadOptions<'static> {
    ParquetReadOptions::default()
        .table_partition_cols(vec![("part_col".into(), DataType::Utf8)])
}

pub(super) async fn register_parquet(
    ctx: &SessionContext,
    path: &Path,
    hive: bool,
) -> Result<()> {
    let options = if hive {
        hive_options()
    } else {
        ParquetReadOptions::default()
    };
    ctx.register_parquet("t", path.to_str().unwrap(), options)
        .await
}

pub(super) async fn scan_plan(ctx: &SessionContext) -> Result<Arc<dyn ExecutionPlan>> {
    ctx.table("t").await?.create_physical_plan().await
}

pub(super) async fn map_input(
    ctx: &SessionContext,
    columns: &[&str],
    min_filter: Option<i64>,
) -> Result<Arc<dyn ExecutionPlan>> {
    let mut df = ctx.table("t").await?;
    if let Some(min) = min_filter {
        df = df.filter(col("filter_only_col").gt(lit(min)))?;
    }
    df.select_columns(columns)?.create_physical_plan().await
}

pub(super) async fn filtered_plan(
    ctx: &SessionContext,
    response_filter: Option<&str>,
) -> Result<Arc<dyn ExecutionPlan>> {
    let input = map_input(ctx, &["id", "prompt_col"], Some(10)).await?;
    response_plan(input, &["id"], false, response_filter)
}

pub(super) fn response_plan(
    input: Arc<dyn ExecutionPlan>,
    columns: &[&str],
    nested: bool,
    response_filter: Option<&str>,
) -> Result<Arc<dyn ExecutionPlan>> {
    let prompt = column_expr("prompt_col", input.as_ref())?;
    let mut plan: Arc<dyn ExecutionPlan> = if nested {
        let inner: Arc<dyn ExecutionPlan> = Arc::new(MockAiExec::new(
            input,
            prompt,
            "inner-m".into(),
            r#"{"temperature":0}"#.into(),
            "inner".into(),
        ));
        let prompt = Arc::new(BinaryExpr::new(
            Arc::new(Literal::new(ScalarValue::from("outer:"))),
            Operator::StringConcat,
            column_expr("inner", inner.as_ref())?,
        ));
        Arc::new(MockAiExec::new(
            inner,
            prompt,
            "outer-m".into(),
            "{}".into(),
            "resp".into(),
        ))
    } else {
        Arc::new(MockAiExec::new(
            input,
            prompt,
            "mock-m".into(),
            r#"{"temperature":0}"#.into(),
            "resp".into(),
        ))
    };
    if let Some(pattern) = response_filter {
        let predicate = Arc::new(LikeExpr::new(
            false,
            false,
            column_expr("resp", plan.as_ref())?,
            Arc::new(Literal::new(ScalarValue::from(pattern))),
        ));
        plan = Arc::new(FilterExec::try_new(predicate, plan)?);
    }
    project(plan, columns.iter().copied().chain(["resp"]))
}

pub(super) fn column_expr(
    name: &str,
    plan: &dyn ExecutionPlan,
) -> Result<Arc<dyn PhysicalExpr>> {
    Ok(Arc::new(Column::new_with_schema(
        name,
        plan.schema().as_ref(),
    )?))
}

pub(super) fn project<'a>(
    plan: Arc<dyn ExecutionPlan>,
    columns: impl IntoIterator<Item = &'a str>,
) -> Result<Arc<dyn ExecutionPlan>> {
    let exprs = columns
        .into_iter()
        .map(|name| Ok((column_expr(name, plan.as_ref())?, name.to_owned())))
        .collect::<Result<Vec<_>>>()?;
    Ok(Arc::new(ProjectionExec::try_new(exprs, plan)?))
}

pub(super) fn ai_codecs() -> Result<DistCodecs> {
    let mut registry = DistExtensionRegistry::default();
    registry.register(Arc::new(MockAiCodec::new("test.mock_ai", true)))?;
    Ok(DistCodecs::default().with_physical_codecs(registry))
}

pub(super) fn scan_node(plan: &dyn ExecutionPlan) -> (&FileScanConfig, &ParquetSource) {
    if let Some(source) = plan.downcast_ref::<DataSourceExec>() {
        return source.downcast_to_file_source::<ParquetSource>().unwrap();
    }
    let [child] = plan.children()[..] else {
        panic!("expected unary map-only plan at {}", plan.name());
    };
    scan_node(child.as_ref())
}

pub(super) fn rewrite_files(
    plan: Arc<dyn ExecutionPlan>,
    update: &impl Fn(&PartitionedFile) -> Vec<PartitionedFile>,
) -> Result<Arc<dyn ExecutionPlan>> {
    if let Some(exec) = plan.downcast_ref::<DataSourceExec>() {
        let (scan, _) = exec.downcast_to_file_source::<ParquetSource>().unwrap();
        let groups = scan
            .file_groups
            .iter()
            .flat_map(FileGroup::files)
            .flat_map(update)
            .map(|file| FileGroup::new(vec![file]))
            .collect();
        return Ok(DataSourceExec::from_data_source(
            FileScanConfigBuilder::from(scan.clone())
                .with_file_groups(groups)
                .build(),
        ));
    }
    let [child] = plan.children()[..] else {
        panic!("expected unary map-only plan at {}", plan.name());
    };
    let child = rewrite_files(Arc::clone(child), update)?;
    plan.replace_children(
        vec![child],
        ReplaceChildrenOptions::new(ChildrenPropertiesMode::Recompute),
    )
}

pub(super) fn assert_independent(coordinator: &SessionContext, workers: &[LocalWorker]) {
    let contexts = std::iter::once(coordinator)
        .chain(workers.iter().map(LocalWorker::context))
        .collect::<Vec<_>>();
    for (index, a) in contexts.iter().enumerate() {
        for b in &contexts[index + 1..] {
            assert_ne!(a.session_id(), b.session_id());
            assert!(!Arc::ptr_eq(&a.runtime_env(), &b.runtime_env()));
        }
    }
    for worker in workers {
        assert!(!worker.context().table_exist("t").unwrap());
    }
}

pub(super) fn nodes(
    plan: &dyn ExecutionPlan,
) -> impl Iterator<Item = &dyn ExecutionPlan> {
    std::iter::successors(Some(plan), |node| {
        let mut children = node.children().into_iter();
        let child = children.next()?;
        assert!(
            children.next().is_none(),
            "expected unary map-only plan at {}",
            node.name()
        );
        Some(child.as_ref())
    })
}

pub(super) fn submitted(plan: &dyn ExecutionPlan) -> usize {
    nodes(plan)
        .filter(|node| node.is::<MockAiExec>())
        .map(|node| metric(node, "submitted_prompts"))
        .sum()
}

pub(super) fn metric(plan: &dyn ExecutionPlan, name: &str) -> usize {
    plan.metrics()
        .and_then(|metrics| metrics.sum_by_name(name))
        .map_or(0, |value| value.as_usize())
}

pub(super) fn assert_pruned_projection(plan: &dyn ExecutionPlan) {
    let (scan, parquet) = scan_node(plan);
    let unused = scan.file_schema().index_of("unused_wide_col").unwrap();
    let indices = parquet.projection().unwrap().column_indices();
    assert!(
        !indices.contains(&unused),
        "scan read projection: {indices:?}"
    );
    if let Some(predicate) = parquet.filter() {
        assert!(!predicate.to_string().contains("unused_wide_col"));
    }
    assert_eq!(plan.properties().output_partitioning().partition_count(), 1);
}

pub(super) fn assert_attempts(expected: &[Split], outputs: &[SplitOutput]) {
    let expected_ids: HashSet<_> = expected
        .iter()
        .map(|s| (s.split_id, s.attempt_id))
        .collect();
    let actual_ids: HashSet<_> =
        outputs.iter().map(|s| (s.split_id, s.attempt_id)).collect();
    assert_eq!(
        expected_ids.len(),
        expected.len(),
        "duplicate split IDs in fixture"
    );
    assert_eq!(
        actual_ids.len(),
        outputs.len(),
        "duplicate split execution outputs"
    );
    assert_eq!(actual_ids, expected_ids);
}

pub(super) fn rows(batches: &[RecordBatch]) -> Result<HashMap<Vec<ScalarValue>, usize>> {
    // Arrow memory accounting does not affect ScalarValue hashing.
    #[allow(clippy::allow_attributes, clippy::mutable_key_type)]
    let mut counts = HashMap::new();
    for batch in batches {
        for row in 0..batch.num_rows() {
            let values = batch
                .columns()
                .iter()
                .map(|array| ScalarValue::try_from_array(array, row))
                .collect::<Result<Vec<_>>>()?;
            *counts.entry(values).or_default() += 1;
        }
    }
    Ok(counts)
}

pub(super) fn assert_matches_single_node(
    stage: &StagePlan,
    baseline_schema: &Schema,
    baseline: &[RecordBatch],
    outputs: &[SplitOutput],
) -> Result<()> {
    let batches: Vec<_> = outputs
        .iter()
        .flat_map(|output| output.batches.iter().cloned())
        .collect();
    assert_eq!(stage.output_schema.as_ref(), baseline_schema);
    assert!(
        baseline
            .iter()
            .all(|b| b.schema().as_ref() == baseline_schema)
    );
    for batch in &batches {
        assert_eq!(batch.schema(), stage.output_schema);
    }
    assert_eq!(rows(&batches)?, rows(baseline)?);
    for output in outputs {
        assert_eq!(
            output.metrics.output_rows,
            output
                .batches
                .iter()
                .map(RecordBatch::num_rows)
                .sum::<usize>()
        );
    }
    Ok(())
}

pub(super) fn rejected(plan: Arc<dyn ExecutionPlan>, operator: &str) {
    let error = split_plan_for_distribution(plan, 9).unwrap_err();
    assert_plan_error(&error, operator);
}

pub(super) fn assert_plan_error(error: &DataFusionError, expected: &str) {
    assert!(
        matches!(error.find_root(), DataFusionError::Plan(_)),
        "{error}"
    );
    assert!(error.to_string().contains(expected), "{error}");
}

pub(super) type MockAiExec = AiExec<false>;
pub(super) type SpeculatableExec = AiExec<true>;
pub(super) type MockAiCodec = AiCodec<false>;
pub(super) type SpeculatableCodec = AiCodec<true>;

#[derive(Debug)]
pub(super) struct AiExec<const SPECULATION: bool> {
    input: Arc<dyn ExecutionPlan>,
    prompt: Arc<dyn PhysicalExpr>,
    pub(super) model: String,
    pub(super) options_json: String,
    pub(super) output_col: String,
    properties: Arc<PlanProperties>,
    metrics: ExecutionPlanMetricsSet,
}

impl<const SPECULATION: bool> AiExec<SPECULATION> {
    pub(super) fn new(
        input: Arc<dyn ExecutionPlan>,
        prompt: Arc<dyn PhysicalExpr>,
        model: String,
        options_json: String,
        output_col: String,
    ) -> Self {
        let mut schema = SchemaBuilder::from(input.schema().as_ref());
        schema.push(Field::new(&output_col, DataType::Utf8, true));
        let properties = Arc::new(PlanProperties::new(
            EquivalenceProperties::new(Arc::new(schema.finish())),
            Partitioning::UnknownPartitioning(
                input.properties().output_partitioning().partition_count(),
            ),
            input.properties().emission_type,
            input.properties().boundedness,
        ));
        Self {
            input,
            prompt,
            model,
            options_json,
            output_col,
            properties,
            metrics: ExecutionPlanMetricsSet::new(),
        }
    }
}

impl<const SPECULATION: bool> DisplayAs for AiExec<SPECULATION> {
    fn fmt_as(&self, _format: DisplayFormatType, f: &mut Formatter<'_>) -> fmt::Result {
        f.write_str(self.name())
    }
}

impl<const SPECULATION: bool> ExecutionPlan for AiExec<SPECULATION> {
    fn name(&self) -> &str {
        if SPECULATION {
            "SpeculatableExec"
        } else {
            "MockAiExec"
        }
    }

    fn properties(&self) -> &Arc<PlanProperties> {
        &self.properties
    }

    fn children(&self) -> Vec<&Arc<dyn ExecutionPlan>> {
        vec![&self.input]
    }

    fn replace_children(
        self: Arc<Self>,
        children: Vec<Arc<dyn ExecutionPlan>>,
        _options: ReplaceChildrenOptions,
    ) -> Result<Arc<dyn ExecutionPlan>> {
        let [input] = children.as_slice() else {
            return plan_err!("{} requires exactly one input", self.name());
        };
        Ok(Arc::new(Self::new(
            Arc::clone(input),
            Arc::clone(&self.prompt),
            self.model.clone(),
            self.options_json.clone(),
            self.output_col.clone(),
        )))
    }

    fn with_new_children(
        self: Arc<Self>,
        children: Vec<Arc<dyn ExecutionPlan>>,
    ) -> Result<Arc<dyn ExecutionPlan>> {
        self.replace_children(
            children,
            ReplaceChildrenOptions::new(ChildrenPropertiesMode::Recompute),
        )
    }

    fn apply_expressions(
        &self,
        f: &mut dyn FnMut(&Arc<dyn PhysicalExpr>) -> Result<TreeNodeRecursion>,
    ) -> Result<TreeNodeRecursion> {
        f(&self.prompt)
    }

    fn execute(
        &self,
        partition: usize,
        context: Arc<TaskContext>,
    ) -> Result<SendableRecordBatchStream> {
        let input = self.input.execute(partition, context)?;
        let submitted =
            MetricBuilder::new(&self.metrics).counter("submitted_prompts", partition);
        let prefix = format!("mock[{}]: ", self.model);
        let prompt = Arc::clone(&self.prompt);
        let schema = self.schema();
        let output_schema = Arc::clone(&schema);
        let stream = input.map(move |batch| {
            let batch = batch?;
            let prompts = prompt.evaluate(&batch)?.into_array(batch.num_rows())?;
            let mut responses = Vec::with_capacity(batch.num_rows());
            for row in 0..batch.num_rows() {
                let text = match ScalarValue::try_from_array(&prompts, row)? {
                    ScalarValue::Utf8(text)
                    | ScalarValue::LargeUtf8(text)
                    | ScalarValue::Utf8View(text) => text,
                    value => {
                        return plan_err!(
                            "Mock AI expected a string, got {}",
                            value.data_type()
                        );
                    }
                };
                responses.push(text.map(|text| {
                    submitted.add(1);
                    format!("{prefix}{text}")
                }));
            }
            let mut columns = batch.columns().to_vec();
            columns.push(Arc::new(StringArray::from(responses)));
            Ok(RecordBatch::try_new(Arc::clone(&output_schema), columns)?)
        });
        Ok(Box::pin(RecordBatchStreamAdapter::new(schema, stream)))
    }

    fn metrics(&self) -> Option<MetricsSet> {
        Some(self.metrics.clone_inner())
    }
}

#[derive(Clone, PartialEq, Message)]
pub(super) struct MockAiNode {
    #[prost(message, optional, tag = "1")]
    pub(super) prompt: Option<protobuf::PhysicalExprNode>,
    #[prost(string, tag = "2")]
    pub(super) model: String,
    #[prost(string, tag = "3")]
    pub(super) options_json: String,
    #[prost(string, tag = "4")]
    pub(super) output_col: String,
}

#[derive(Debug)]
pub(super) struct AiCodec<const SPECULATION: bool> {
    name: &'static str,
    map_only: bool,
}

impl<const SPECULATION: bool> AiCodec<SPECULATION> {
    pub(super) fn new(name: &'static str, map_only: bool) -> Self {
        Self { name, map_only }
    }
}

impl<const SPECULATION: bool> NamedPhysicalExtensionCodec for AiCodec<SPECULATION> {
    fn extension_name(&self) -> &str {
        self.name
    }

    fn operator_policy(
        &self,
        node: &dyn ExecutionPlan,
    ) -> Option<ExtensionOperatorPolicy> {
        node.is::<AiExec<SPECULATION>>()
            .then_some(ExtensionOperatorPolicy {
                map_only: self.map_only,
                speculation_allowed: SPECULATION,
            })
    }

    fn try_encode(
        &self,
        node: &Arc<dyn ExecutionPlan>,
        converter: &dyn PhysicalProtoConverterExtension,
        root_codec: &dyn PhysicalExtensionCodec,
    ) -> Result<Option<(u32, Vec<u8>)>> {
        let Some(exec) = node.downcast_ref::<AiExec<SPECULATION>>() else {
            return Ok(None);
        };
        let options_json =
            canonical_options(&exec.model, &exec.options_json, &exec.output_col)?;
        let node = MockAiNode {
            prompt: Some(converter.physical_expr_to_proto(&exec.prompt, root_codec)?),
            model: exec.model.clone(),
            options_json,
            output_col: exec.output_col.clone(),
        };
        Ok(Some((1, node.encode_to_vec())))
    }

    fn try_decode(
        &self,
        version: u32,
        payload: &[u8],
        inputs: &[Arc<dyn ExecutionPlan>],
        ctx: &TaskContext,
        converter: &dyn PhysicalProtoConverterExtension,
        root_codec: &dyn PhysicalExtensionCodec,
    ) -> Result<Arc<dyn ExecutionPlan>> {
        if version != 1 {
            return plan_err!("Unsupported mock AI schema version {version}");
        }
        let [input] = inputs else {
            return plan_err!("Mock AI requires exactly one input");
        };
        let node = MockAiNode::decode(payload)
            .map_err(|error| plan_datafusion_err!("Invalid mock AI protobuf: {error}"))?;
        let options_json =
            canonical_options(&node.model, &node.options_json, &node.output_col)?;
        let prompt = node.prompt.as_ref().ok_or_else(|| {
            plan_datafusion_err!("Mock AI protobuf is missing its prompt")
        })?;
        let prompt = converter.proto_to_physical_expr(
            prompt,
            input.schema().as_ref(),
            &PhysicalPlanDecodeContext::new(ctx, root_codec),
        )?;
        Ok(Arc::new(AiExec::<SPECULATION>::new(
            Arc::clone(input),
            prompt,
            node.model,
            options_json,
            node.output_col,
        )))
    }
}

fn canonical_options(model: &str, options: &str, output_col: &str) -> Result<String> {
    if model.trim().is_empty() || output_col.trim().is_empty() {
        return plan_err!("Mock AI requires a model and output column");
    }
    // ValueIter reads one line. Tabs preserve whitespace semantics inside and outside strings.
    let json = options.replace(['\r', '\n'], "\t");
    let mut value = ValueIter::new(json.as_bytes(), None)
        .next()
        .ok_or_else(|| plan_datafusion_err!("Missing mock AI options_json"))?
        .map_err(|error| plan_datafusion_err!("Invalid mock AI options_json: {error}"))?;
    if !value.is_object() {
        return plan_err!("Mock AI options_json must be an object");
    }
    // Inspect the original keys, including those in objects overwritten by duplicate keys.
    let mut bytes = json.bytes().enumerate();
    while let Some((start, _)) = bytes.find(|&(_, byte)| byte == b'"') {
        let end = loop {
            let (index, byte) = bytes.next().expect("validated JSON string");
            match byte {
                b'"' => break index,
                b'\\' => {
                    bytes.next();
                }
                _ => {}
            }
        };
        if json[end + 1..].trim_start().starts_with(':') {
            let key = ValueIter::new(&json.as_bytes()[start..=end], None)
                .next()
                .expect("validated JSON key")?;
            if matches!(
                key.as_str().unwrap().to_ascii_lowercase().as_str(),
                "api_key" | "token" | "secret" | "password" | "credentials"
            ) {
                return plan_err!(
                    "Mock AI options_json contains a sensitive credential key"
                );
            }
        }
    }
    value.sort_all_objects();
    Ok(value.to_string())
}
