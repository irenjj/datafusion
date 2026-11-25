// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: See the NOTICE file distributed with this work for additional information regarding copyright ownership
// SPDX-FileContributor: Licensed to the Apache Software Foundation (ASF) under one or more contributor license agreements

use std::fmt;
use std::sync::Arc;

use super::stream::{Dispatch, LlmMetrics, LlmStream};
use super::{LlmOptions, LlmProviderRegistry};
use crate::arrow::datatypes::{FieldRef, Schema, SchemaRef};
use crate::common::stats::{ColumnStatistics, Precision};
use crate::common::tree_node::{Transformed, TreeNode, TreeNodeRecursion};
use crate::common::{Result, Statistics, internal_err, plan_err};
use crate::execution::session_state::SessionState;
use crate::execution::{SendableRecordBatchStream, TaskContext};
use crate::logical_expr::physical_planning_context::PhysicalPlanningContext;
use crate::logical_expr::{LogicalPlan, UserDefinedLogicalNode};
use crate::physical_expr::expressions::Column;
use crate::physical_expr::{EquivalenceProperties, Partitioning, PhysicalExpr};
use crate::physical_plan::execution_plan::CardinalityEffect;
use crate::physical_plan::metrics::{ExecutionPlanMetricsSet, MetricsSet};
use crate::physical_plan::{
    ChildStats, ChildrenPropertiesMode, DisplayAs, DisplayFormatType, ExecutionPlan,
    LimitBehavior, PlanProperties, ReplaceChildrenOptions, StatisticsArgs,
    apply_expression_roots,
};
use crate::physical_planner::PhysicalPlanner;
use datafusion_session::Session;

use super::logical::{LlmNode, llm_field};

pub(crate) fn plan_llm(
    planner: &dyn PhysicalPlanner,
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
    let prompt = planner.create_physical_expr(
        &node.prompt_expr,
        logical_input.schema(),
        session,
        planning_context,
    )?;
    let (options, _) = LlmOptions::parse(&node.options_json)?;
    let fallback;
    let registry = if let Some(state) = session.as_any().downcast_ref::<SessionState>() {
        state.llm_provider_registry()
    } else {
        fallback = LlmProviderRegistry::default();
        &fallback
    };
    let provider = registry.resolve(options.provider.as_deref())?;
    let dispatch = Dispatch::try_new(node.model.clone(), options, provider)?;
    Ok(Some(Arc::new(LlmExec::new(
        Arc::clone(input),
        prompt,
        Arc::new(dispatch),
        node.output_col.clone(),
    ))))
}

#[derive(Debug)]
struct LlmExec {
    input: Arc<dyn ExecutionPlan>,
    prompt: Arc<dyn PhysicalExpr>,
    dispatch: Arc<Dispatch>,
    output_col: String,
    properties: Arc<PlanProperties>,
    metrics: ExecutionPlanMetricsSet,
}

impl LlmExec {
    fn new(
        input: Arc<dyn ExecutionPlan>,
        prompt: Arc<dyn PhysicalExpr>,
        dispatch: Arc<Dispatch>,
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
            output_col,
            properties,
            metrics: ExecutionPlanMetricsSet::new(),
        }
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
