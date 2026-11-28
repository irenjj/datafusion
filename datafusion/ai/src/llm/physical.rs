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

use super::logical::{LlmNode, llm_field};
use super::stream::{Dispatch, LlmMetrics, LlmStream};
use super::{LlmOptions, LlmProviderRegistry};
use crate::{CheckpointOutputColumn, CheckpointableTarget};

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
    Ok(Some(Arc::new(LlmExec::try_new_with_registry(
        Arc::clone(input),
        prompt,
        node.model().to_owned(),
        node.options_json(),
        node.output_col().to_owned(),
        &registry,
    )?)))
}

/// Physical plan that evaluates prompts and appends asynchronous LLM results.
#[derive(Debug)]
pub struct LlmExec {
    input: Arc<dyn ExecutionPlan>,
    prompt: Arc<dyn PhysicalExpr>,
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

    /// Returns the validated, recursively sorted JSON options.
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
        if model.trim().is_empty() {
            return plan_err!("LlmExec model must be non-empty");
        }
        if output_col.trim().is_empty() {
            return plan_err!("LlmExec output column must be non-empty");
        }
        let (options, canonical_options_json) = LlmOptions::parse(options_json)?;
        let provider = registry.resolve(options.provider.as_deref())?;
        let dispatch = Dispatch::try_new(model, options, provider)?;
        Ok(Self::new(
            input,
            prompt,
            Arc::new(dispatch),
            canonical_options_json,
            output_col,
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

    fn new(
        input: Arc<dyn ExecutionPlan>,
        prompt: Arc<dyn PhysicalExpr>,
        dispatch: Arc<Dispatch>,
        canonical_options_json: String,
        output_col: String,
    ) -> Self {
        let schema = input.schema();
        let fields = schema
            .fields()
            .iter()
            .cloned()
            .chain(std::iter::once(llm_field(&output_col)))
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
            is_error_envelope: false,
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
        apply_expression_roots([&self.prompt], f)
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
        Ok(Arc::new(Self::new(
            input,
            prompt,
            Arc::clone(&self.dispatch),
            self.canonical_options_json.clone(),
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
            Arc::clone(&self.dispatch),
            LlmMetrics::new(&self.metrics, partition),
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
