// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: See the NOTICE file distributed with this work for additional information regarding copyright ownership
// SPDX-FileContributor: Licensed to the Apache Software Foundation (ASF) under one or more contributor license agreements

use std::fmt;
use std::sync::Arc;

use async_trait::async_trait;
use datafusion::arrow::array::{Array, StringArray, StringBuilder};
use datafusion::arrow::datatypes::{FieldRef, Schema, SchemaRef};
use datafusion::arrow::record_batch::RecordBatch;
use datafusion::common::stats::{ColumnStatistics, Precision};
use datafusion::common::tree_node::{Transformed, TreeNode};
use datafusion::common::{Result, Statistics, internal_err, plan_err};
use datafusion::execution::context::QueryPlanner;
use datafusion::execution::{SendableRecordBatchStream, SessionState, TaskContext};
use datafusion::logical_expr::{LogicalPlan, UserDefinedLogicalNode};
use datafusion::physical_expr::expressions::Column;
use datafusion::physical_expr::{EquivalenceProperties, Partitioning, PhysicalExpr};
use datafusion::physical_plan::execution_plan::CardinalityEffect;
use datafusion::physical_plan::metrics::{
    ExecutionPlanMetricsSet, MetricBuilder, MetricsSet,
};
use datafusion::physical_plan::stream::RecordBatchStreamAdapter;
use datafusion::physical_plan::{
    DisplayAs, DisplayFormatType, ExecutionPlan, PlanProperties,
};
use datafusion::physical_planner::{
    DefaultPhysicalPlanner, ExtensionPlanner, PhysicalPlanner,
};
use futures::StreamExt;

use crate::logical::{VllmNode, vllm_field};

#[derive(Debug)]
pub(crate) struct VllmQueryPlanner;

#[async_trait]
impl QueryPlanner for VllmQueryPlanner {
    async fn create_physical_plan(
        &self,
        plan: &LogicalPlan,
        state: &SessionState,
    ) -> Result<Arc<dyn ExecutionPlan>> {
        DefaultPhysicalPlanner::with_extension_planners(vec![Arc::new(VllmPlanner)])
            .create_physical_plan(plan, state)
            .await
    }
}

#[derive(Debug)]
pub struct VllmPlanner;

#[async_trait]
impl ExtensionPlanner for VllmPlanner {
    async fn plan_extension(
        &self,
        planner: &dyn PhysicalPlanner,
        node: &dyn UserDefinedLogicalNode,
        logical_inputs: &[&LogicalPlan],
        physical_inputs: &[Arc<dyn ExecutionPlan>],
        state: &SessionState,
    ) -> Result<Option<Arc<dyn ExecutionPlan>>> {
        let Some(node) = node.as_any().downcast_ref::<VllmNode>() else {
            return Ok(None);
        };
        let ([logical_input], [input]) = (logical_inputs, physical_inputs) else {
            return internal_err!("VllmProject requires one input");
        };
        // Bind by qualified column name against the optimized logical schema first.
        // Arrow schemas no longer carry table qualifiers.
        let prompt = planner.create_physical_expr(
            &node.prompt_expr,
            logical_input.schema(),
            state,
        )?;
        Ok(Some(Arc::new(VllmExec::new(
            Arc::clone(input),
            prompt,
            node.model.clone(),
            node.options_json.clone(),
            node.output_col.clone(),
        ))))
    }
}

#[derive(Debug)]
pub struct VllmExec {
    input: Arc<dyn ExecutionPlan>,
    prompt: Arc<dyn PhysicalExpr>,
    model: String,
    options_json: String,
    output_col: String,
    properties: Arc<PlanProperties>,
    metrics: ExecutionPlanMetricsSet,
}

impl VllmExec {
    fn new(
        input: Arc<dyn ExecutionPlan>,
        prompt: Arc<dyn PhysicalExpr>,
        model: String,
        options_json: String,
        output_col: String,
    ) -> Self {
        let schema = input.schema();
        let fields = schema
            .fields()
            .iter()
            .cloned()
            .chain(std::iter::once(vllm_field(&output_col)))
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
            model,
            options_json,
            output_col,
            properties,
            metrics: ExecutionPlanMetricsSet::new(),
        }
    }
}

impl DisplayAs for VllmExec {
    fn fmt_as(
        &self,
        _kind: DisplayFormatType,
        f: &mut fmt::Formatter<'_>,
    ) -> fmt::Result {
        write!(
            f,
            "VllmExec: model={}, output={}, prompt={}",
            self.model, self.output_col, self.prompt
        )
    }
}

impl ExecutionPlan for VllmExec {
    fn name(&self) -> &str {
        "VllmExec"
    }

    fn properties(&self) -> &Arc<PlanProperties> {
        &self.properties
    }

    fn children(&self) -> Vec<&Arc<dyn ExecutionPlan>> {
        vec![&self.input]
    }

    fn with_new_children(
        self: Arc<Self>,
        children: Vec<Arc<dyn ExecutionPlan>>,
    ) -> Result<Arc<dyn ExecutionPlan>> {
        let Ok([input]): Result<[_; 1], _> = children.try_into() else {
            return internal_err!("VllmExec requires one input");
        };
        let prompt = rebind_prompt(
            Arc::clone(&self.prompt),
            &self.input.schema(),
            &input.schema(),
        )?;
        Ok(Arc::new(Self::new(
            input,
            prompt,
            self.model.clone(),
            self.options_json.clone(),
            self.output_col.clone(),
        )))
    }

    fn maintains_input_order(&self) -> Vec<bool> {
        vec![false]
    }

    fn benefits_from_input_partitioning(&self) -> Vec<bool> {
        vec![false]
    }

    fn cardinality_effect(&self) -> CardinalityEffect {
        CardinalityEffect::Equal
    }

    fn partition_statistics(&self, partition: Option<usize>) -> Result<Arc<Statistics>> {
        let mut stats = Arc::unwrap_or_clone(self.input.partition_statistics(partition)?);
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
        let submitted =
            MetricBuilder::new(&self.metrics).counter("submitted_prompts", partition);
        let model = self.model.clone();
        let prompt = Arc::clone(&self.prompt);
        let schema = self.schema();
        let output_schema = Arc::clone(&schema);
        let stream = input.map(move |batch| {
            let batch = batch?;
            let values = prompt.evaluate(&batch)?.into_array(batch.num_rows())?;
            let Some(prompts) = values.as_any().downcast_ref::<StringArray>() else {
                return internal_err!(
                    "VllmExec expected Utf8 prompts, got {}",
                    values.data_type()
                );
            };
            let mut responses = StringBuilder::new();
            for value in prompts.iter() {
                match value {
                    Some(value) => {
                        submitted.add(1);
                        responses.append_value(mock_infer(&model, value));
                    }
                    None => responses.append_null(),
                }
            }
            let mut columns = batch.columns().to_vec();
            columns.push(Arc::new(responses.finish()));
            Ok(RecordBatch::try_new(Arc::clone(&output_schema), columns)?)
        });
        Ok(Box::pin(RecordBatchStreamAdapter::new(schema, stream)))
    }
}

fn mock_infer(model: &str, prompt: &str) -> String {
    format!("mock[{model}]: {prompt}")
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
                return internal_err!("VllmExec prompt column '{}' is outside the input schema", column.name());
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
                    "VllmExec cannot rebind prompt column '{}' to a unique field in the replacement input",
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
