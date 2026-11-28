// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: See the NOTICE file distributed with this work for additional information regarding copyright ownership
// SPDX-FileContributor: Licensed to the Apache Software Foundation (ASF) under one or more contributor license agreements

use std::collections::HashMap;
use std::fmt::{self, Formatter};
use std::sync::Arc;

use arrow::array::{
    Array, ArrayRef, Float32Array, Int64Array, RecordBatch, UInt32Array, UInt64Array,
};
use arrow::compute::{concat_batches, take, take_record_batch};
use arrow::datatypes::{DataType, SchemaRef};
use datafusion_common::tree_node::TreeNodeRecursion;
use datafusion_common::{
    Result, exec_datafusion_err, exec_err, internal_datafusion_err, internal_err,
};
use datafusion_execution::{SendableRecordBatchStream, TaskContext};
use datafusion_physical_expr::{EquivalenceProperties, PhysicalExpr};
use datafusion_physical_plan::execution_plan::{
    Boundedness, ChildrenPropertiesMode, EmissionType, ReplaceChildrenOptions,
};
use datafusion_physical_plan::stream::RecordBatchStreamAdapter;
use datafusion_physical_plan::{
    DisplayAs, DisplayFormatType, ExecutionPlan, Partitioning, PlanProperties,
    execute_stream,
};
use futures::{TryStreamExt, stream};

use crate::scan::make_batch;

use super::FusedRow;

#[derive(Debug, Clone)]
pub(super) struct FusedScoreExec {
    input: Arc<dyn ExecutionPlan>,
    output: SchemaRef,
    projection: Arc<[Option<usize>]>,
    key_column: usize,
    fused: Arc<[FusedRow]>,
    properties: Arc<PlanProperties>,
}

impl FusedScoreExec {
    pub(super) fn try_new(
        input: Arc<dyn ExecutionPlan>,
        output: SchemaRef,
        projection: Vec<Option<usize>>,
        key_column: &str,
        fused: Vec<FusedRow>,
    ) -> Result<Self> {
        let schema = input.schema();
        let key_column = schema.index_of(key_column)?;
        if !matches!(
            schema.field(key_column).data_type(),
            DataType::UInt64 | DataType::Int64
        ) {
            return internal_err!("Invalid type for lake search row key");
        }
        let properties = Arc::new(PlanProperties::new(
            EquivalenceProperties::new(Arc::clone(&output)),
            Partitioning::UnknownPartitioning(1),
            EmissionType::Final,
            Boundedness::Bounded,
        ));
        Ok(Self {
            input,
            output,
            projection: projection.into(),
            key_column,
            fused: fused.into(),
            properties,
        })
    }

    async fn reorder(
        &self,
        mut stream: SendableRecordBatchStream,
    ) -> Result<RecordBatch> {
        let mut positions = vec![None; self.fused.len()];
        let ranks: HashMap<_, _> = self
            .fused
            .iter()
            .enumerate()
            .map(|(rank, row)| (row.row_key, rank))
            .collect();
        let schema = stream.schema();
        let mut batches = Vec::new();
        let mut matched = 0u32;
        let materialize_columns = self.projection.iter().any(Option::is_some);
        while let Some(batch) = stream.try_next().await? {
            let keys = batch.column(self.key_column);
            let mut selected = Vec::new();
            for index in 0..batch.num_rows() {
                let key = read_row_key(keys.as_ref(), index)?;
                let Some(&rank) = ranks.get(&key) else {
                    continue;
                };
                if positions[rank].replace(matched).is_some() {
                    return exec_err!("Duplicate fused row key {key} in lake late read");
                }
                matched += 1;
                if materialize_columns {
                    let index = u32::try_from(index).map_err(|_| {
                        exec_datafusion_err!("Lake late read batch is too large")
                    })?;
                    selected.push(index);
                }
            }
            if !selected.is_empty() {
                batches.push(take_record_batch(&batch, &UInt32Array::from(selected))?);
            }
        }
        let permutation = positions
            .into_iter()
            .zip(self.fused.iter())
            .map(|(pos, hit)| {
                pos.ok_or_else(|| {
                    exec_datafusion_err!(
                        "Missing fused row key {} in lake late read",
                        hit.row_key
                    )
                })
            })
            .collect::<Result<UInt32Array>>()?;
        let selected = concat_batches(&schema, &batches)?;
        let scores: ArrayRef = Arc::new(Float32Array::from_iter_values(
            self.fused.iter().map(|hit| hit.score as f32),
        ));
        let columns = self
            .projection
            .iter()
            .map(|index| match index {
                Some(index) => {
                    Ok(take(selected.column(*index).as_ref(), &permutation, None)?)
                }
                None => Ok(Arc::clone(&scores)),
            })
            .collect::<Result<Vec<_>>>()?;
        make_batch(Arc::clone(&self.output), columns, self.fused.len())
    }
}

fn read_row_key(column: &dyn Array, index: usize) -> Result<u64> {
    if column.is_null(index) {
        return exec_err!("Null fused row key in lake late read");
    }
    if let Some(keys) = column.as_any().downcast_ref::<UInt64Array>() {
        return Ok(keys.value(index));
    }
    if let Some(keys) = column.as_any().downcast_ref::<Int64Array>() {
        return u64::try_from(keys.value(index)).map_err(|_| {
            exec_datafusion_err!("Negative fused row key in Paimon late read")
        });
    }
    internal_err!("Invalid type for lake search row key")
}

impl DisplayAs for FusedScoreExec {
    fn fmt_as(&self, _: DisplayFormatType, f: &mut Formatter<'_>) -> fmt::Result {
        write!(f, "FusedScoreExec: rows={}", self.fused.len())
    }
}

impl ExecutionPlan for FusedScoreExec {
    fn name(&self) -> &'static str {
        "FusedScoreExec"
    }

    fn properties(&self) -> &Arc<PlanProperties> {
        &self.properties
    }

    fn children(&self) -> Vec<&Arc<dyn ExecutionPlan>> {
        vec![&self.input]
    }

    fn maintains_input_order(&self) -> Vec<bool> {
        vec![false]
    }

    fn apply_expressions(
        &self,
        _: &mut dyn FnMut(&Arc<dyn PhysicalExpr>) -> Result<TreeNodeRecursion>,
    ) -> Result<TreeNodeRecursion> {
        Ok(TreeNodeRecursion::Continue)
    }

    fn replace_children(
        self: Arc<Self>,
        children: Vec<Arc<dyn ExecutionPlan>>,
        _: ReplaceChildrenOptions,
    ) -> Result<Arc<dyn ExecutionPlan>> {
        let [input] = <[_; 1]>::try_from(children)
            .map_err(|_| internal_datafusion_err!("FusedScoreExec requires one input"))?;
        if input.schema() != self.input.schema() {
            return internal_err!("FusedScoreExec input schema changed");
        }
        Ok(Arc::new(Self {
            input,
            ..self.as_ref().clone()
        }))
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

    fn execute(
        &self,
        partition: usize,
        context: Arc<TaskContext>,
    ) -> Result<SendableRecordBatchStream> {
        if partition != 0 {
            return internal_err!("FusedScoreExec only has partition 0");
        }
        let this = self.clone();
        let stream = stream::once(async move {
            let source = execute_stream(Arc::clone(&this.input), context)?;
            this.reorder(source).await
        });
        Ok(Box::pin(RecordBatchStreamAdapter::new(
            Arc::clone(&self.output),
            stream,
        )))
    }
}

#[cfg(test)]
#[path = "../../tests/search_contract/late_read.rs"]
mod tests;
