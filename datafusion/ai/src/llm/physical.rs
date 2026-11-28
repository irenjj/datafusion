// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: See the NOTICE file distributed with this work for additional information regarding copyright ownership
// SPDX-FileContributor: Licensed to the Apache Software Foundation (ASF) under one or more contributor license agreements

use std::fmt;
use std::sync::Arc;

use arrow::datatypes::{FieldRef, Schema, SchemaRef};
use datafusion_common::stats::{ColumnStatistics, Precision};
use datafusion_common::tree_node::{Transformed, TreeNode, TreeNodeRecursion};
use datafusion_common::{
    Result, Statistics, internal_err, plan_datafusion_err, plan_err,
};
use datafusion_execution::config::SessionConfig;
use datafusion_execution::{SendableRecordBatchStream, TaskContext};
use datafusion_expr::physical_planning_context::PhysicalPlanningContext;
use datafusion_expr::{LogicalPlan, UserDefinedLogicalNode};
use datafusion_physical_expr::expressions::Column;
use datafusion_physical_expr::{
    EquivalenceProperties, Partitioning, PhysicalExpr, create_physical_expr,
};
use datafusion_physical_plan::execution_plan::CardinalityEffect;
use datafusion_physical_plan::metrics::{ExecutionPlanMetricsSet, MetricsSet};
use datafusion_physical_plan::{
    ChildStats, ChildrenPropertiesMode, DisplayAs, DisplayFormatType, ExecutionPlan,
    LimitBehavior, PlanProperties, ReplaceChildrenOptions, StatisticsArgs,
    apply_expression_roots,
};
use datafusion_session::Session;

use super::logical::{LlmNode, validate_media_field};
use super::stream::{Dispatch, LlmMetrics, LlmStream};
use super::{LlmOptions, LlmProviderRegistry, OnError, TaskKind};
use crate::{
    CheckpointOutputColumn, CheckpointableTarget, MediaResolver, ObjectStoreMediaResolver,
};

/// Plans an [`LlmNode`] with the coordinator session's registry, or the built-in
/// mock registry when the session has no registry extension.
///
/// # Public Only for Internal Use:
///
/// This is not a public API and is for internal use only; see [API policy] for details.
/// [API policy]: https://datafusion.apache.org/contributor-guide/api-health.html#internal-public-apis
#[doc(hidden)]
pub fn plan_llm(
    node: &dyn UserDefinedLogicalNode,
    logical_inputs: &[&LogicalPlan],
    physical_inputs: &[Arc<dyn ExecutionPlan>],
    session: &dyn Session,
    planning_context: &PhysicalPlanningContext,
) -> Result<Option<Arc<dyn ExecutionPlan>>> {
    let Some(node) = node.as_any().downcast_ref::<LlmNode>() else {
        return Ok(None);
    };
    let ([logical_input], [input]) = (logical_inputs, physical_inputs) else {
        return internal_err!("LlmProject requires one input");
    };
    // Bind qualified column names before the conversion to Arrow drops table qualifiers.
    let prompt = create_physical_expr(
        node.prompt_expr(),
        logical_input.schema(),
        session.execution_props(),
        planning_context,
    )?;
    let registry = session
        .config()
        .get_extension::<LlmProviderRegistry>()
        .unwrap_or_default();
    let media = node
        .media_expr()
        .map(|expr| {
            create_physical_expr(
                expr,
                logical_input.schema(),
                session.execution_props(),
                planning_context,
            )
            .map(MediaPlan::new)
        })
        .transpose()?;
    let media = bind_session_media(media, node.task_kind(), session.config())?;
    Ok(Some(Arc::new(LlmExec::try_new_task_with_registry(
        Arc::clone(input),
        prompt,
        media,
        node.task_kind(),
        node.model().to_owned(),
        node.options_json(),
        node.output_col().to_owned(),
        &registry,
    )?)))
}

/// The physical media expression and resolver for vision inference.
#[derive(Clone)]
pub struct MediaPlan {
    pub(super) expr: Arc<dyn PhysicalExpr>,
    pub(super) resolver: Option<Arc<dyn MediaResolver>>,
}

impl fmt::Debug for MediaPlan {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("MediaPlan").finish_non_exhaustive()
    }
}

impl MediaPlan {
    /// Bind the media expression to the input schema. The task-context constructor
    /// binds the resolver from the session configuration.
    pub fn new(expr: Arc<dyn PhysicalExpr>) -> Self {
        Self {
            expr,
            resolver: None,
        }
    }

    /// Attach an explicit resolver when building a plan with a provider registry.
    #[must_use]
    pub fn with_resolver(mut self, resolver: Arc<dyn MediaResolver>) -> Self {
        self.resolver = Some(resolver);
        self
    }

    /// Returns the media expression bound to the input schema.
    pub fn expr(&self) -> &Arc<dyn PhysicalExpr> {
        &self.expr
    }
}

fn bind_session_media(
    media: Option<MediaPlan>,
    kind: TaskKind,
    config: &SessionConfig,
) -> Result<Option<MediaPlan>> {
    if kind != TaskKind::Vision {
        return Ok(media);
    }
    let media = media
        .ok_or_else(|| plan_datafusion_err!("Vision requires a media expression"))?;
    let resolver = config
        .get_extension::<ObjectStoreMediaResolver>()
        .map(|resolver| resolver as Arc<dyn MediaResolver>)
        .or_else(|| {
            config
                .get_extension::<Arc<dyn MediaResolver>>()
                .map(|resolver| Arc::clone(resolver.as_ref()))
        })
        .ok_or_else(|| {
            plan_datafusion_err!(
                "Vision requires a media resolver extension in SessionConfig"
            )
        })?;
    Ok(Some(media.with_resolver(resolver)))
}

/// Physical plan that evaluates prompts and appends asynchronous LLM results.
#[derive(Debug)]
pub struct LlmExec {
    input: Arc<dyn ExecutionPlan>,
    prompt: Arc<dyn PhysicalExpr>,
    media: Option<MediaPlan>,
    dispatch: Arc<Dispatch>,
    canonical_options_json: String,
    output_col: String,
    properties: Arc<PlanProperties>,
    metrics: ExecutionPlanMetricsSet,
}

impl LlmExec {
    /// Returns the input plan.
    pub fn input(&self) -> &Arc<dyn ExecutionPlan> {
        &self.input
    }

    /// Returns the prompt expression bound to the input schema.
    pub fn prompt(&self) -> &Arc<dyn PhysicalExpr> {
        &self.prompt
    }

    /// Returns the model identifier.
    pub fn model(&self) -> &str {
        &self.dispatch.model
    }

    /// Returns the validated canonical JSON options, retaining response property order.
    pub fn options_json(&self) -> &str {
        &self.canonical_options_json
    }

    /// Returns the parsed inference options.
    pub fn options(&self) -> &LlmOptions {
        &self.dispatch.options
    }

    /// Returns the name of the appended result column.
    pub fn output_col(&self) -> &str {
        &self.output_col
    }

    /// Returns the task performed by this operator.
    pub fn task_kind(&self) -> TaskKind {
        self.dispatch.kind
    }

    /// Returns the bound media plan for vision inference.
    pub fn media(&self) -> Option<&MediaPlan> {
        self.media.as_ref()
    }

    /// Builds an operator with a prompt bound to the input schema. Each operator
    /// has its own request limit and shares the selected provider's limit.
    ///
    /// # Errors
    /// Returns a planning error for invalid identifiers or options, an unknown
    /// provider, or an unsupported request batch size.
    pub fn try_new_with_registry(
        input: Arc<dyn ExecutionPlan>,
        prompt: Arc<dyn PhysicalExpr>,
        model: String,
        options_json: &str,
        output_col: String,
        registry: &LlmProviderRegistry,
    ) -> Result<Self> {
        Self::try_new_task_with_registry(
            input,
            prompt,
            None,
            TaskKind::Chat,
            model,
            options_json,
            output_col,
            registry,
        )
    }

    /// Builds an operator for the given task. Vision requires a media expression and a resolver.
    ///
    /// # Errors
    /// Returns a planning error for incompatible media, options, or provider settings.
    #[expect(clippy::too_many_arguments)]
    pub fn try_new_task_with_registry(
        input: Arc<dyn ExecutionPlan>,
        prompt: Arc<dyn PhysicalExpr>,
        media: Option<MediaPlan>,
        kind: TaskKind,
        model: String,
        options_json: &str,
        output_col: String,
        registry: &LlmProviderRegistry,
    ) -> Result<Self> {
        if model.trim().is_empty() {
            return plan_err!("LlmExec model must be non-empty");
        }
        if output_col.trim().is_empty() {
            return plan_err!("LlmExec output column must be non-empty");
        }
        match (kind, &media) {
            (TaskKind::Vision, Some(media)) if media.resolver.is_some() => {
                validate_media_field(
                    media.expr.return_field(input.schema().as_ref())?.as_ref(),
                )?;
            }
            (TaskKind::Vision, _) => {
                return plan_err!("Vision requires a media expression and resolver");
            }
            (_, Some(_)) => return plan_err!("Only vision inference accepts media"),
            (_, None) => {}
        }
        let (options, canonical_options_json, schema) = LlmOptions::parse(options_json)?;
        let output_field = options.output_field(kind, &output_col, schema.as_ref())?;
        let provider_name = options
            .provider
            .as_deref()
            .unwrap_or_else(|| registry.default_name())
            .to_owned();
        let provider = registry.resolve(Some(&provider_name))?;
        let dispatch =
            Dispatch::try_new(model, options, schema, kind, provider, provider_name)?;
        Ok(Self::new(
            input,
            prompt,
            media,
            Arc::new(dispatch),
            canonical_options_json,
            output_col,
            output_field,
        ))
    }

    /// Builds an operator using the registry in the task's session config.
    ///
    /// # Errors
    /// Returns a planning error if the context has no [`LlmProviderRegistry`]
    /// extension or if [`Self::try_new_with_registry`] rejects the operator.
    pub fn try_new_from_task_context(
        input: Arc<dyn ExecutionPlan>,
        prompt: Arc<dyn PhysicalExpr>,
        model: String,
        options_json: &str,
        output_col: String,
        ctx: &TaskContext,
    ) -> Result<Self> {
        let registry = ctx
            .session_config()
            .get_extension::<LlmProviderRegistry>()
            .ok_or_else(|| {
                plan_datafusion_err!(
                    "LlmExec worker decoding requires a LlmProviderRegistry extension in the task's SessionConfig"
                )
            })?;
        Self::try_new_with_registry(
            input,
            prompt,
            model,
            options_json,
            output_col,
            &registry,
        )
    }

    /// Builds an operator from the worker's session extensions. For a custom
    /// media resolver, register an `Arc<dyn MediaResolver>` as an extension value.
    ///
    /// # Errors
    /// Returns a planning error for missing extensions or invalid operator settings.
    #[expect(clippy::too_many_arguments)]
    pub fn try_new_task_from_task_context(
        input: Arc<dyn ExecutionPlan>,
        prompt: Arc<dyn PhysicalExpr>,
        media: Option<MediaPlan>,
        kind: TaskKind,
        model: String,
        options_json: &str,
        output_col: String,
        ctx: &TaskContext,
    ) -> Result<Self> {
        if kind == TaskKind::Chat && media.is_none() {
            return Self::try_new_from_task_context(
                input,
                prompt,
                model,
                options_json,
                output_col,
                ctx,
            );
        }
        let config = ctx.session_config();
        let media = bind_session_media(media, kind, config)?;
        let registry = config.get_extension::<LlmProviderRegistry>().ok_or_else(|| {
            plan_datafusion_err!("LlmExec worker decoding requires a LlmProviderRegistry extension in the task's SessionConfig")
        })?;
        Self::try_new_task_with_registry(
            input,
            prompt,
            media,
            kind,
            model,
            options_json,
            output_col,
            &registry,
        )
    }

    fn new(
        input: Arc<dyn ExecutionPlan>,
        prompt: Arc<dyn PhysicalExpr>,
        media: Option<MediaPlan>,
        dispatch: Arc<Dispatch>,
        canonical_options_json: String,
        output_col: String,
        output_field: FieldRef,
    ) -> Self {
        let schema = input.schema();
        let fields = schema
            .fields()
            .iter()
            .cloned()
            .chain(std::iter::once(output_field))
            .collect::<Vec<_>>();
        let output_schema =
            Arc::new(Schema::new_with_metadata(fields, schema.metadata().clone()));
        let child = input.properties();
        let properties = Arc::new(PlanProperties::new(
            EquivalenceProperties::new(output_schema),
            Partitioning::UnknownPartitioning(
                child.output_partitioning().partition_count(),
            ),
            child.emission_type,
            child.boundedness,
        ));
        Self {
            input,
            prompt,
            media,
            dispatch,
            canonical_options_json,
            output_col,
            properties,
            metrics: ExecutionPlanMetricsSet::new(),
        }
    }
}

impl CheckpointableTarget for LlmExec {
    fn passthrough_column_count(&self) -> usize {
        self.input.schema().fields().len()
    }

    fn ai_output_columns(&self) -> Vec<CheckpointOutputColumn> {
        vec![CheckpointOutputColumn {
            index: self.passthrough_column_count(),
            name: self.output_col.clone(),
            is_error_envelope: self.dispatch.options.on_error == OnError::Envelope,
        }]
    }

    fn preserves_input_order(&self) -> bool {
        false
    }

    fn preserves_row_cardinality(&self) -> bool {
        true
    }
}

impl DisplayAs for LlmExec {
    fn fmt_as(
        &self,
        _kind: DisplayFormatType,
        f: &mut fmt::Formatter<'_>,
    ) -> fmt::Result {
        write!(
            f,
            "LlmExec: model={}, output={}, prompt={}",
            self.dispatch.model, self.output_col, self.prompt
        )
    }
}

impl ExecutionPlan for LlmExec {
    fn name(&self) -> &str {
        "LlmExec"
    }

    fn properties(&self) -> &Arc<PlanProperties> {
        &self.properties
    }

    fn children(&self) -> Vec<&Arc<dyn ExecutionPlan>> {
        vec![&self.input]
    }

    fn apply_expressions(
        &self,
        f: &mut dyn FnMut(&Arc<dyn PhysicalExpr>) -> Result<TreeNodeRecursion>,
    ) -> Result<TreeNodeRecursion> {
        apply_expression_roots(
            std::iter::once(&self.prompt).chain(self.media.iter().map(MediaPlan::expr)),
            f,
        )
    }

    fn replace_children(
        self: Arc<Self>,
        children: Vec<Arc<dyn ExecutionPlan>>,
        _options: ReplaceChildrenOptions,
    ) -> Result<Arc<dyn ExecutionPlan>> {
        let Ok([input]): Result<[_; 1], _> = children.try_into() else {
            return internal_err!("LlmExec requires one input");
        };
        let prompt = rebind_prompt(
            Arc::clone(&self.prompt),
            &self.input.schema(),
            &input.schema(),
        )?;
        let media = self
            .media
            .clone()
            .map(|mut media| {
                media.expr =
                    rebind_prompt(media.expr, &self.input.schema(), &input.schema())?;
                Ok::<_, datafusion_common::DataFusionError>(media)
            })
            .transpose()?;
        Ok(Arc::new(Self::new(
            input,
            prompt,
            media,
            Arc::clone(&self.dispatch),
            self.canonical_options_json.clone(),
            self.output_col.clone(),
            Arc::clone(self.schema().fields().last().unwrap()),
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

    fn maintains_input_order(&self) -> Vec<bool> {
        vec![false]
    }

    fn benefits_from_input_partitioning(&self) -> Vec<bool> {
        vec![false]
    }

    fn limit_behavior(&self) -> LimitBehavior {
        LimitBehavior::MoveToInput
    }

    fn cardinality_effect(&self) -> CardinalityEffect {
        CardinalityEffect::Equal
    }

    fn child_stats_requests(&self, partition: Option<usize>) -> Vec<ChildStats> {
        vec![ChildStats::At(partition)]
    }

    fn statistics_from_inputs(
        &self,
        input_stats: &[Arc<Statistics>],
        _args: &StatisticsArgs,
    ) -> Result<Arc<Statistics>> {
        let mut stats = input_stats[0].as_ref().clone();
        stats.total_byte_size = Precision::Absent;
        stats
            .column_statistics
            .push(ColumnStatistics::new_unknown());
        Ok(Arc::new(stats))
    }

    fn metrics(&self) -> Option<MetricsSet> {
        Some(self.metrics.clone_inner())
    }

    fn execute(
        &self,
        partition: usize,
        context: Arc<TaskContext>,
    ) -> Result<SendableRecordBatchStream> {
        let input = self.input.execute(partition, context)?;
        Ok(Box::pin(LlmStream::new(
            self.schema(),
            input,
            Arc::clone(&self.prompt),
            self.media.clone(),
            Arc::clone(&self.dispatch),
            LlmMetrics::new(&self.metrics, partition, &self.dispatch.provider_name),
        )))
    }
}

fn rebind_prompt(
    prompt: Arc<dyn PhysicalExpr>,
    old: &SchemaRef,
    new: &SchemaRef,
) -> Result<Arc<dyn PhysicalExpr>> {
    if Arc::ptr_eq(old, new) {
        return Ok(prompt);
    }

    Ok(prompt
        .transform_up(|expr| {
            let Some(column) = expr.downcast_ref::<Column>() else {
                return Ok(Transformed::no(expr));
            };
            let Some(source) = old.fields().get(column.index()) else {
                return internal_err!("LlmExec prompt column '{}' is outside the input schema", column.name());
            };
            // Projections typically reuse Arrow fields. Prefer their identity so
            // two same-named fields from a join can still be told apart.
            let same_field = |field: &FieldRef| Arc::ptr_eq(field, source);
            let equal_field = |field: &FieldRef| field == source;
            let index = unique_field_index(old, same_field)
                .and_then(|_| unique_field_index(new, same_field))
                .or_else(|| {
                    unique_field_index(old, equal_field)
                        .and_then(|_| unique_field_index(new, equal_field))
                });
            let Some(index) = index else {
                return plan_err!(
                    "LlmExec cannot rebind prompt column '{}' to a unique field in the replacement input",
                    column.name()
                );
            };
            if index == column.index() {
                Ok(Transformed::no(expr))
            } else {
                Ok(Transformed::yes(Arc::new(Column::new(source.name(), index)) as _))
            }
        })?
        .data)
}

fn unique_field_index(
    schema: &Schema,
    predicate: impl Fn(&FieldRef) -> bool,
) -> Option<usize> {
    let mut matches = schema
        .fields()
        .iter()
        .enumerate()
        .filter(|(_, field)| predicate(field));
    let (index, _) = matches.next()?;
    matches.next().is_none().then_some(index)
}
