// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: See the NOTICE file distributed with this work for additional information regarding copyright ownership
// SPDX-FileContributor: Licensed to the Apache Software Foundation (ASF) under one or more contributor license agreements

use std::sync::Arc;

use datafusion::arrow::array::{
    Array, ArrayRef, Int64Array, LargeStringArray, StringArray, StringViewArray,
};
use datafusion::arrow::datatypes::{DataType, Field, Schema};
use datafusion::arrow::record_batch::RecordBatch;
use datafusion::common::{DataFusionError, Result};
use datafusion::physical_plan::{ExecutionPlan, collect};
use datafusion::prelude::{SessionConfig, SessionContext};

pub(super) fn test_context(batch_size: usize, partitions: usize) -> SessionContext {
    SessionContext::new_with_config(
        SessionConfig::new()
            .with_batch_size(batch_size)
            .with_target_partitions(partitions),
    )
}

pub(super) fn context() -> Result<SessionContext> {
    let ctx = SessionContext::new();
    register_text(&ctx, "t", &[Some("a"), Some("b"), Some("c"), Some("d")])?;
    Ok(ctx)
}

pub(super) fn register_text(
    ctx: &SessionContext,
    name: &str,
    prompts: &[Option<&str>],
) -> Result<()> {
    let ids = std::iter::once(1).chain(11..).take(prompts.len());
    let schema = Arc::new(Schema::new(vec![
        Field::new("id", DataType::Int64, false),
        Field::new("text", DataType::Utf8, true),
    ]));
    ctx.register_batch(
        name,
        RecordBatch::try_new(
            schema,
            vec![
                Arc::new(Int64Array::from_iter_values(ids)),
                Arc::new(StringArray::from(prompts.to_vec())),
            ],
        )?,
    )?;
    Ok(())
}

pub(super) fn strings(batches: &[RecordBatch], index: usize) -> Vec<Option<&str>> {
    batches
        .iter()
        .flat_map(|batch| {
            batch
                .column(index)
                .as_any()
                .downcast_ref::<StringArray>()
                .unwrap()
                .iter()
        })
        .collect()
}

pub(super) fn exec_nodes(plan: &dyn ExecutionPlan) -> Vec<&dyn ExecutionPlan> {
    let mut result = Vec::new();
    if plan.name() == "LlmExec" {
        result.push(plan);
    }
    for child in plan.children() {
        result.extend(exec_nodes(child.as_ref()));
    }
    result
}

pub(super) fn submitted(plan: &dyn ExecutionPlan) -> usize {
    exec_nodes(plan)
        .into_iter()
        .filter_map(|exec| exec.metrics().unwrap().sum_by_name("submitted_prompts"))
        .map(|value| value.as_usize())
        .sum()
}

pub(super) async fn run(
    ctx: &SessionContext,
    sql: &str,
) -> Result<(Arc<dyn ExecutionPlan>, Vec<RecordBatch>)> {
    let plan = ctx.sql(sql).await?.create_physical_plan().await?;
    let batches = collect(Arc::clone(&plan), ctx.task_ctx()).await?;
    Ok((plan, batches))
}

fn explain_text(array: &ArrayRef, index: usize) -> &str {
    if let Some(array) = array.as_any().downcast_ref::<StringArray>() {
        array.value(index)
    } else {
        array
            .as_any()
            .downcast_ref::<StringViewArray>()
            .unwrap()
            .value(index)
    }
}

pub(super) async fn explain(
    ctx: &SessionContext,
    sql: &str,
    label: &str,
) -> Result<(String, String)> {
    let batches = ctx.sql(&format!("EXPLAIN {sql}")).await?.collect().await?;
    let mut logical = String::new();
    let mut physical = String::new();
    for batch in batches {
        for row in 0..batch.num_rows() {
            let dest = match explain_text(batch.column(0), row) {
                "logical_plan" => &mut logical,
                "physical_plan" => &mut physical,
                _ => continue,
            };
            explain_text(batch.column(1), row).clone_into(dest);
        }
    }
    assert!(!logical.is_empty() && !physical.is_empty());
    println!("\n{label}\n{logical}\n{physical}");
    Ok((logical, physical))
}

pub(super) fn assert_below(plan: &str, child: &str, parent: &str) {
    let indent = |line: &str| line.len() - line.trim_start().len();
    let mut lines = plan.lines();
    let parent = lines
        .find(|line| line.contains(parent))
        .unwrap_or_else(|| panic!("{plan}"));
    assert!(
        lines
            .take_while(|line| indent(line) > indent(parent))
            .any(|line| line.contains(child)),
        "{plan}"
    );
}

pub(super) async fn assert_plan_error(
    ctx: &SessionContext,
    sql: &str,
    messages: &[&str],
) {
    let error = match ctx.sql(sql).await {
        Ok(df) => df.into_optimized_plan().unwrap_err(),
        Err(error) => error,
    };
    assert!(
        matches!(error.find_root(), DataFusionError::Plan(_)),
        "{sql}\n{error}"
    );
    assert!(
        messages
            .iter()
            .any(|message| error.to_string().contains(message)),
        "{sql}\nExpected a planning error mentioning {messages:?}\n{error}"
    );
}

pub(super) fn string_array(data_type: &DataType, values: Vec<Option<&str>>) -> ArrayRef {
    match data_type {
        DataType::Utf8 => Arc::new(StringArray::from(values)),
        DataType::LargeUtf8 => Arc::new(LargeStringArray::from(values)),
        DataType::Utf8View => Arc::new(StringViewArray::from(values)),
        _ => panic!("Expected a string type, got {data_type}"),
    }
}

pub(super) type AnswerRow = (i64, Option<String>);

pub(super) fn int_values(batches: &[RecordBatch]) -> impl Iterator<Item = i64> + '_ {
    batches.iter().flat_map(|batch| {
        batch
            .column(0)
            .as_any()
            .downcast_ref::<Int64Array>()
            .unwrap()
            .values()
            .iter()
            .copied()
    })
}

pub(super) fn answer_rows(batches: &[RecordBatch]) -> Vec<AnswerRow> {
    int_values(batches)
        .zip(
            strings(batches, 1)
                .into_iter()
                .map(|answer| answer.map(str::to_owned)),
        )
        .collect()
}

pub(super) fn assert_rows(
    batches: &[RecordBatch],
    mut expected: Vec<AnswerRow>,
    ordered: bool,
) {
    let mut actual = answer_rows(batches);
    if !ordered {
        // Sorting the whole vector keeps duplicate rows visible.
        actual.sort_unstable();
        expected.sort_unstable();
    }
    assert_eq!(actual, expected);
}
