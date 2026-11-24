// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: See the NOTICE file distributed with this work for additional information regarding copyright ownership
// SPDX-FileContributor: Licensed to the Apache Software Foundation (ASF) under one or more contributor license agreements

use std::sync::Arc;

use datafusion::arrow::array::{
    Array, ArrayRef, Int64Array, StringArray, StringViewArray,
};
use datafusion::arrow::datatypes::{DataType, Field, Schema};
use datafusion::arrow::record_batch::RecordBatch;
use datafusion::common::stats::ColumnStatistics;
use datafusion::common::tree_node::TreeNodeRecursion;
use datafusion::common::{DataFusionError, Result};
use datafusion::datasource::MemTable;
use datafusion::logical_expr::LogicalPlan;
use datafusion::optimizer::OptimizerContext;
use datafusion::physical_expr::expressions::Column;
use datafusion::physical_plan::execution_plan::CardinalityEffect;
use datafusion::physical_plan::projection::{ProjectionExec, ProjectionExpr};
use datafusion::physical_plan::{
    ChildrenPropertiesMode, ExecutionPlan, ReplaceChildrenOptions, StatisticsArgs,
    StatisticsContext, collect,
};
use datafusion::prelude::SessionContext;

fn context() -> Result<SessionContext> {
    let ctx = SessionContext::new();
    register_text(&ctx, "t", &[Some("a"), Some("b"), Some("c"), Some("d")])?;
    Ok(ctx)
}

fn register_text(
    ctx: &SessionContext,
    name: &str,
    prompts: &[Option<&str>],
) -> Result<()> {
    let ids = [1, 11, 12, 13];
    let schema = Arc::new(Schema::new(vec![
        Field::new("id", DataType::Int64, false),
        Field::new("text", DataType::Utf8, true),
    ]));
    ctx.register_batch(
        name,
        RecordBatch::try_new(
            schema,
            vec![
                Arc::new(Int64Array::from(ids[..prompts.len()].to_vec())),
                Arc::new(StringArray::from(prompts.to_vec())),
            ],
        )?,
    )?;
    Ok(())
}

fn register_wide(ctx: &SessionContext) -> Result<()> {
    let schema = Arc::new(Schema::new(vec![
        Field::new("c0", DataType::Int64, false),
        Field::new("c1", DataType::Utf8, false),
        Field::new("unused_a", DataType::Utf8, false),
        Field::new("unused_b", DataType::Int64, false),
        Field::new("prompt_col", DataType::Utf8, false),
    ]));
    ctx.register_batch(
        "wide_t",
        RecordBatch::try_new(
            schema,
            vec![
                Arc::new(Int64Array::from(vec![1, 2])),
                Arc::new(StringArray::from(vec!["keep-1", "keep-2"])),
                Arc::new(StringArray::from(vec!["wrong-1", "wrong-2"])),
                Arc::new(Int64Array::from(vec![3, 4])),
                Arc::new(StringArray::from(vec!["right-1", "right-2"])),
            ],
        )?,
    )?;
    Ok(())
}

fn strings(batches: &[RecordBatch], index: usize) -> Vec<Option<&str>> {
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

fn exec_nodes(plan: &dyn ExecutionPlan) -> Vec<&dyn ExecutionPlan> {
    let mut result = Vec::new();
    if plan.name() == "VllmExec" {
        result.push(plan);
    }
    for child in plan.children() {
        result.extend(exec_nodes(child.as_ref()));
    }
    result
}

fn submitted(plan: &dyn ExecutionPlan) -> usize {
    exec_nodes(plan)
        .into_iter()
        .map(|exec| {
            exec.metrics()
                .unwrap()
                .sum_by_name("submitted_prompts")
                .unwrap()
                .as_usize()
        })
        .sum()
}

async fn run(
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

async fn explain(
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

fn assert_below(plan: &str, child: &str, parent: &str) {
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

#[tokio::test]
async fn basic_projection_and_constant_prompt() -> Result<()> {
    let ctx = context()?;
    let (plan, batches) = run(
        &ctx,
        "SELECT id, vllm(text, 'qwen3', '{\"temperature\":0.2}') AS resp FROM t",
    )
    .await?;
    assert_eq!(batches[0].schema().field(1).name(), "resp");
    assert_eq!(
        batches[0]
            .column(0)
            .as_any()
            .downcast_ref::<Int64Array>()
            .unwrap()
            .values(),
        &[1, 11, 12, 13]
    );
    assert_eq!(
        strings(&batches, 1),
        [
            Some("mock[qwen3]: a"),
            Some("mock[qwen3]: b"),
            Some("mock[qwen3]: c"),
            Some("mock[qwen3]: d")
        ]
    );
    assert_eq!(submitted(plan.as_ref()), 4);

    let (plan, batches) = run(&ctx, "SELECT vllm('hello', 'qwen3', '{}') FROM t").await?;
    assert_eq!(strings(&batches, 0), vec![Some("mock[qwen3]: hello"); 4]);
    assert_eq!(submitted(plan.as_ref()), 4);
    Ok(())
}

#[tokio::test]
async fn foldable_model_and_options_are_allowed() -> Result<()> {
    let ctx = context()?;
    let (_, batches) = run(
        &ctx,
        "SELECT vllm(text, concat('qwen', '-3'), concat('{', '}')) FROM t LIMIT 1",
    )
    .await?;
    assert_eq!(strings(&batches, 0), [Some("mock[qwen-3]: a")]);
    Ok(())
}

#[tokio::test]
async fn predicate_pushdown_keeps_output_filter_above_inference() -> Result<()> {
    let ctx = context()?;
    let sql = "SELECT id, vllm(text, 'qwen3', '{}') AS resp FROM t WHERE id > 10";
    let (logical, physical) = explain(&ctx, sql, "Input predicate").await?;
    assert_below(&logical, "Filter:", "VllmProject:");
    assert_below(&physical, "FilterExec:", "VllmExec:");
    let (plan, batches) = run(&ctx, sql).await?;
    assert_eq!(strings(&batches, 1).len(), 3);
    assert_eq!(submitted(plan.as_ref()), 3);

    let sql = "SELECT id, resp FROM (SELECT id, vllm(text, 'qwen3', '{}') AS resp FROM t) WHERE resp LIKE 'mock%'";
    let (logical, physical) = explain(&ctx, sql, "Output predicate").await?;
    assert_below(&logical, "VllmProject:", "Filter:");
    assert_below(&physical, "VllmExec:", "FilterExec:");

    let sql = "SELECT id, resp FROM (SELECT id, vllm(text, 'qwen3', '{}') AS resp FROM t) WHERE id > 10 AND resp LIKE 'mock%'";
    let (logical, physical) = explain(&ctx, sql, "Mixed predicate").await?;
    assert_below(&logical, "Filter: t.id >", "VllmProject:");
    assert_below(&logical, "VllmProject:", "LIKE Utf8");
    assert_below(&physical, "FilterExec: id@0 >", "VllmExec:");
    assert_below(&physical, "VllmExec:", "LIKE mock%");
    let (plan, batches) = run(&ctx, sql).await?;
    assert_eq!(strings(&batches, 1).len(), 3);
    assert_eq!(submitted(plan.as_ref()), 3);
    Ok(())
}

#[tokio::test]
async fn limit_pushdown_submits_only_two_rows() -> Result<()> {
    let ctx = context()?;
    let sql = "SELECT id, vllm(text, 'qwen3', '{}') FROM t LIMIT 2";
    let (logical, physical) = explain(&ctx, sql, "Limit").await?;
    assert_below(&logical, "fetch=2", "VllmProject:");
    assert_below(&physical, "fetch=2", "VllmExec:");
    let (plan, batches) = run(&ctx, sql).await?;
    assert_eq!(
        strings(&batches, 1),
        [Some("mock[qwen3]: a"), Some("mock[qwen3]: b")]
    );
    assert_eq!(submitted(plan.as_ref()), 2);
    Ok(())
}

#[tokio::test]
async fn limit_pushdown_caps_inference_across_partitions() -> Result<()> {
    let ctx = SessionContext::new();
    let schema = Arc::new(Schema::new(vec![Field::new("text", DataType::Utf8, false)]));
    let make_batch = |values| {
        RecordBatch::try_new(
            Arc::clone(&schema),
            vec![Arc::new(StringArray::from(values))],
        )
    };
    let partitions = vec![
        vec![make_batch(vec!["a", "b", "c"])?],
        vec![make_batch(vec!["d", "e", "f"])?],
    ];
    ctx.register_table(
        "partitioned",
        Arc::new(MemTable::try_new(schema, partitions)?),
    )?;

    let sql = "SELECT vllm(text, 'm', '{}') FROM partitioned LIMIT 2";
    let (plan, batches) = run(&ctx, sql).await?;
    assert_eq!(strings(&batches, 0).len(), 2);
    assert_eq!(submitted(plan.as_ref()), 2);
    Ok(())
}

#[tokio::test]
async fn column_pruning_keeps_prompt_dependency() -> Result<()> {
    let ctx = context()?;
    register_wide(&ctx)?;
    let sql = "SELECT c1, vllm(prompt_col, 'qwen3', '{}') AS resp FROM wide_t";
    let (logical, physical) = explain(&ctx, sql, "Wide table").await?;
    assert_below(&logical, "projection=[c1, prompt_col]", "VllmProject:");
    assert!(physical.contains("prompt=prompt_col@1"), "{physical}");
    assert!(!logical.contains("unused_") && !physical.contains("unused_"));
    let (plan, batches) = run(&ctx, sql).await?;
    assert_eq!(strings(&batches, 0), [Some("keep-1"), Some("keep-2")]);
    assert_eq!(
        strings(&batches, 1),
        [Some("mock[qwen3]: right-1"), Some("mock[qwen3]: right-2")]
    );
    let nodes = exec_nodes(plan.as_ref());
    let exec = nodes[0];
    let child = exec.children()[0];
    assert_eq!(
        child
            .schema()
            .fields()
            .iter()
            .map(|f| f.name().as_str())
            .collect::<Vec<_>>(),
        ["c1", "prompt_col"]
    );
    let statistics = StatisticsContext::new();
    let args = StatisticsArgs::new();
    let stats = statistics.compute(exec, &args)?;
    let input_stats = statistics.compute(child.as_ref(), &args)?;
    assert_eq!(stats.num_rows, input_stats.num_rows);
    assert_eq!(&stats.column_statistics[..2], input_stats.column_statistics);
    assert_eq!(stats.column_statistics[2], ColumnStatistics::new_unknown());
    assert!(matches!(
        exec.cardinality_effect(),
        CardinalityEffect::Equal
    ));
    assert_eq!(exec.maintains_input_order(), [false]);
    assert_eq!(exec.benefits_from_input_partitioning(), [false]);
    Ok(())
}

#[tokio::test]
async fn replacing_children_rebinds_a_reordered_and_pruned_prompt() -> Result<()> {
    let ctx = context()?;
    register_wide(&ctx)?;
    let plan = ctx
        .sql("SELECT *, vllm(concat('prompt:', prompt_col), 'qwen3', '{}') AS resp FROM wide_t")
        .await?
        .create_physical_plan()
        .await?;
    let exec = Arc::clone(plan.children()[0]);
    assert_eq!(exec.name(), "VllmExec");
    let child = Arc::clone(exec.children()[0]);
    assert_eq!(child.schema().index_of("prompt_col")?, 4);

    let reordered = Arc::new(ProjectionExec::try_new(
        [
            ProjectionExpr {
                expr: Arc::new(Column::new("prompt_col", 4)),
                alias: "prompt_col".into(),
            },
            ProjectionExpr {
                expr: Arc::new(Column::new("unused_a", 2)),
                alias: "unused_a".into(),
            },
        ],
        Arc::clone(&child),
    )?);
    let options = ReplaceChildrenOptions::new(ChildrenPropertiesMode::Recompute);
    let rebound = Arc::clone(&exec).replace_children(vec![reordered], options)?;
    assert_eq!(rebound.schema().index_of("prompt_col")?, 0);
    let batches = collect(Arc::clone(&rebound), ctx.task_ctx()).await?;
    assert_eq!(strings(&batches, 1), [Some("wrong-1"), Some("wrong-2")]);
    assert_eq!(
        strings(&batches, 2),
        [
            Some("mock[qwen3]: prompt:right-1"),
            Some("mock[qwen3]: prompt:right-2")
        ]
    );
    assert_eq!(submitted(rebound.as_ref()), 2);

    let no_prompt = Arc::new(ProjectionExec::try_new(
        [ProjectionExpr {
            expr: Arc::new(Column::new("unused_a", 2)),
            alias: "unused_a".into(),
        }],
        child,
    )?);
    let error = exec.replace_children(vec![no_prompt], options).unwrap_err();
    assert!(error.to_string().contains("cannot rebind prompt column"));
    Ok(())
}

#[tokio::test]
async fn nested_calls_are_extracted_inside_out() -> Result<()> {
    let ctx = context()?;
    let sql = "SELECT vllm(concat('outer:', vllm(text, 'm1', '{}')), 'm2', '{}') FROM t LIMIT 1";
    let (logical, physical) = explain(&ctx, sql, "Nested projections").await?;
    assert_eq!(logical.matches("VllmProject:").count(), 2);
    assert_below(&logical, "VllmProject: model=m1", "VllmProject: model=m2");
    assert_eq!(physical.matches("VllmExec:").count(), 2);
    assert_below(&physical, "VllmExec: model=m1", "VllmExec: model=m2");
    let (plan, batches) = run(&ctx, sql).await?;
    assert_eq!(strings(&batches, 0), [Some("mock[m2]: outer:mock[m1]: a")]);
    assert_eq!(submitted(plan.as_ref()), 2);
    Ok(())
}

#[tokio::test]
async fn canonical_options_deduplicate_calls_and_rule_is_idempotent() -> Result<()> {
    let ctx = context()?;
    let sql = "SELECT vllm(text, 'm1', '{\"b\":1,\"a\":2}'), vllm(text, 'm1', '{\"a\":2,\"b\":1}') FROM t LIMIT 1";
    let (logical, physical) = explain(&ctx, sql, "Common subexpression").await?;
    assert_eq!(logical.matches("VllmProject:").count(), 1);
    assert_eq!(physical.matches("VllmExec:").count(), 1);
    let (plan, batches) = run(&ctx, sql).await?;
    assert_eq!(strings(&batches, 0), [Some("mock[m1]: a")]);
    assert_eq!(strings(&batches, 1), strings(&batches, 0));
    assert_eq!(submitted(plan.as_ref()), 1);

    let df = ctx.sql(sql).await?;
    let optimized = df.into_optimized_plan()?;
    let state = ctx.state();
    let rule = state
        .optimizer()
        .rules
        .iter()
        .find(|rule| rule.name() == "extract_vllm_project")
        .unwrap();
    optimized.apply_with_subqueries(|node: &LogicalPlan| {
        let rewritten = rule.rewrite(node.clone(), &OptimizerContext::new())?;
        assert!(!rewritten.transformed);
        assert_eq!(rewritten.data, *node);
        Ok(TreeNodeRecursion::Continue)
    })?;
    Ok(())
}

#[tokio::test]
async fn null_prompts_never_submit_inference() -> Result<()> {
    let ctx = context()?;
    register_text(&ctx, "mixed", &[None, Some("present"), None])?;
    register_text(&ctx, "all_null", &[None, None])?;
    for (table, expected, count) in [
        ("mixed", vec![None, Some("mock[m]: present"), None], 1),
        ("all_null", vec![None, None], 0),
    ] {
        let (plan, batches) = run(
            &ctx,
            &format!("SELECT vllm(text, 'm', '{{}}') FROM {table}"),
        )
        .await?;
        assert_eq!(strings(&batches, 0), expected);
        assert_eq!(submitted(plan.as_ref()), count);
    }
    let (plan, batches) = run(&ctx, "SELECT vllm(NULL, 'm', '{}') FROM t").await?;
    assert_eq!(strings(&batches, 0), [None; 4]);
    assert_eq!(submitted(plan.as_ref()), 0);
    Ok(())
}

async fn assert_plan_error(ctx: &SessionContext, sql: &str, message: &str) {
    let error = match ctx.sql(sql).await {
        Ok(df) => df.into_optimized_plan().unwrap_err(),
        Err(error) => error,
    };
    assert!(
        matches!(error.find_root(), DataFusionError::Plan(_)),
        "{sql}\n{error}"
    );
    assert!(error.to_string().contains(message), "{sql}\n{error}");
}

#[tokio::test]
async fn forbidden_short_circuits_fail_before_simplification() -> Result<()> {
    let ctx = context()?;
    for expr in [
        "CASE WHEN id > 0 THEN vllm(text, 'm1', '{}') ELSE 'skip' END",
        "CASE WHEN false THEN vllm(text, 'm1', '{}') ELSE 'skip' END",
        "COALESCE(vllm(text, 'm1', '{}'), '')",
        "COALESCE('skip', vllm(text, 'm1', '{}'))",
        "false AND vllm(text, 'm1', '{}') = 'x'",
        "true OR vllm(text, 'm1', '{}') = 'x'",
    ] {
        assert_plan_error(&ctx, &format!("SELECT {expr} FROM t"), "short-circuit").await;
    }
    Ok(())
}

#[tokio::test]
async fn forbidden_non_projection_calls_fail_in_planning() -> Result<()> {
    let ctx = context()?;
    for sql in [
        "SELECT id FROM t WHERE vllm(text, 'm1', '{}') = 'x'",
        "SELECT id FROM t WHERE false AND vllm(text, 'm1', '{}') = 'x'",
        "SELECT t.id FROM t JOIN t AS s ON vllm(t.text, 'm1', '{}') = s.text",
        "SELECT count(vllm(text, 'm1', '{}')) FROM t",
        "SELECT id FROM t ORDER BY vllm(text, 'm1', '{}')",
    ] {
        let error = match ctx.sql(sql).await {
            Ok(df) => df.into_optimized_plan().unwrap_err(),
            Err(error) => error,
        };
        assert!(
            matches!(error.find_root(), DataFusionError::Plan(_)),
            "{sql}\n{error}"
        );
        let error = error.to_string();
        assert!(
            error.contains("can only be used in a projection")
                || error.contains("short-circuit"),
            "{sql}\n{error}"
        );
    }
    Ok(())
}

#[tokio::test]
async fn invalid_configuration_fails_in_planning() -> Result<()> {
    let ctx = context()?;
    for (sql, message) in [
        ("SELECT vllm(text, text, '{}') FROM t", "model must be"),
        (
            "SELECT vllm(text, 'm1', text) FROM t",
            "options_json must be",
        ),
        ("SELECT vllm(text, NULL, '{}') FROM t", "model must be"),
        (
            "SELECT vllm(text, 'm1', NULL) FROM t",
            "options_json must be",
        ),
        ("SELECT vllm(text, '   ', '{}') FROM t", "model must be"),
        ("SELECT vllm(text, 'm1', '') FROM t", "options_json must be"),
        (
            "SELECT vllm(text, 'm1', '{invalid}') FROM t",
            "invalid vllm() options_json",
        ),
        (
            "SELECT vllm(text, 'm1', '{\"api_key\":\"example\"}') FROM t",
            "sensitive credential key",
        ),
        (
            "SELECT vllm(text, 'm1', '{\"nested\":[{\"Token\":\"example\"}]}') FROM t",
            "sensitive credential key",
        ),
        (
            "SELECT vllm(text, 'm1', '{\"config\":{\"secret\":\"example\"},\"config\":{}}') FROM t",
            "sensitive credential key",
        ),
    ] {
        assert_plan_error(&ctx, sql, message).await;
    }
    Ok(())
}
