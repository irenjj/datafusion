// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: See the NOTICE file distributed with this work for additional information regarding copyright ownership
// SPDX-FileContributor: Licensed to the Apache Software Foundation (ASF) under one or more contributor license agreements

use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};

use datafusion::arrow::array::{Array, Int64Array, StringArray};
use datafusion::arrow::datatypes::{DataType, Field, Schema};
use datafusion::arrow::record_batch::RecordBatch;
use datafusion::common::stats::ColumnStatistics;
use datafusion::common::tree_node::TreeNodeRecursion;
use datafusion::common::{Result, ScalarValue};
use datafusion::datasource::MemTable;
use datafusion::logical_expr::{ColumnarValue, LogicalPlan, Volatility, create_udf};
use datafusion::optimizer::OptimizerContext;
use datafusion::physical_expr::expressions::Column;
use datafusion::physical_plan::execution_plan::CardinalityEffect;
use datafusion::physical_plan::projection::{ProjectionExec, ProjectionExpr};
use datafusion::physical_plan::{
    ChildrenPropertiesMode, ReplaceChildrenOptions, StatisticsArgs, StatisticsContext,
    collect,
};
use datafusion::prelude::SessionContext;

use super::harness::{
    assert_below, assert_plan_error, assert_rows, context, exec_nodes, explain,
    int_values, register_text, run, strings, submitted, test_context,
};

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
        assert_plan_error(&ctx, &format!("SELECT {expr} FROM t"), &["short-circuit"])
            .await;
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
        assert_plan_error(
            &ctx,
            sql,
            &["can only be used in a projection", "short-circuit"],
        )
        .await;
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
        assert_plan_error(&ctx, sql, &[message]).await;
    }
    Ok(())
}

fn register_seq(ctx: &SessionContext) -> Arc<AtomicUsize> {
    let calls = Arc::new(AtomicUsize::new(0));
    let seq = Arc::clone(&calls);
    ctx.register_udf(create_udf(
        "seq",
        vec![],
        DataType::Utf8,
        Volatility::Volatile,
        Arc::new(move |_| {
            let n = seq.fetch_add(1, Ordering::Relaxed);
            Ok(ColumnarValue::Scalar(ScalarValue::Utf8(Some(
                n.to_string(),
            ))))
        }),
    ));
    calls
}

#[tokio::test]
async fn volatile_prompt_occurrences_are_evaluated_independently() -> Result<()> {
    for (prompt, prefix) in [("seq()", ""), ("concat('p:', seq())", "p:")] {
        let ctx = context()?;
        let calls = register_seq(&ctx);

        let (plan, batches) = run(
            &ctx,
            &format!("SELECT vllm({prompt}, 'm', '{{}}') AS a, vllm({prompt}, 'm', '{{}}') AS b FROM t LIMIT 1"),
        )
        .await?;
        assert_eq!(exec_nodes(plan.as_ref()).len(), 2);
        assert_eq!(submitted(plan.as_ref()), 2);
        assert_eq!(calls.load(Ordering::Relaxed), 2);
        let first = format!("mock[m]: {prefix}0");
        let second = format!("mock[m]: {prefix}1");
        let mut answers = [strings(&batches, 0)[0], strings(&batches, 1)[0]];
        answers.sort_unstable();
        assert_eq!(answers, [Some(first.as_str()), Some(second.as_str())]);
    }
    Ok(())
}

#[tokio::test]
async fn unused_inference_is_pruned_with_its_prompt_dependencies() -> Result<()> {
    let ctx = context()?;
    let calls = register_seq(&ctx);
    let ids = [1, 11, 12, 13];
    for (sql, expected_plan, expected_rows) in [
        (
            "SELECT id FROM (SELECT id, vllm(text, 'm', '{}') AS r FROM t) q",
            "projection=[id]",
            ids.as_slice(),
        ),
        (
            "SELECT id FROM (SELECT id, vllm(vllm(text, 'm1', '{}'), 'm2', '{}') AS r FROM t) q WHERE id > 0",
            "projection=[id]",
            ids.as_slice(),
        ),
        (
            "SELECT COUNT(*) FROM (SELECT id, vllm(text, 'm', '{}') AS r FROM t) q",
            "Aggregate:",
            &[4],
        ),
        (
            "SELECT id FROM (SELECT id, vllm(concat('p:', seq()), 'm', '{}') AS r FROM t) q",
            "projection=[id]",
            ids.as_slice(),
        ),
    ] {
        let (logical, physical) = explain(&ctx, sql, "Unused inference").await?;
        assert!(!logical.contains("VllmProject:"), "{logical}");
        assert!(!physical.contains("VllmExec:"), "{physical}");
        assert!(logical.contains(expected_plan), "{logical}");

        let (plan, batches) = run(&ctx, sql).await?;
        assert!(exec_nodes(plan.as_ref()).is_empty());
        assert_eq!(submitted(plan.as_ref()), 0);
        let mut rows = int_values(&batches).collect::<Vec<_>>();
        rows.sort_unstable();
        assert_eq!(rows, expected_rows);
    }
    assert_eq!(calls.load(Ordering::Relaxed), 0, "unused volatile prompt");
    Ok(())
}

#[tokio::test]
async fn inference_used_by_filter_is_retained_and_limit_waits_for_match() -> Result<()> {
    let ctx = test_context(8192, 1);
    register_text(&ctx, "t", &[Some("a"), Some("b"), Some("c"), Some("d")])?;
    let from = "FROM (SELECT id, vllm(text, 'm', '{}') AS r FROM t) q WHERE r = 'mock[m]: d' LIMIT 1";
    let sql = format!("SELECT id, r {from}");
    let (logical, physical) = explain(&ctx, &sql, "Filter before limit").await?;
    assert_below(&logical, "Filter:", "Limit:");
    assert_below(&logical, "VllmProject:", "Filter:");
    assert_below(&physical, "VllmExec:", "FilterExec:");
    let (plan, batches) = run(&ctx, &sql).await?;
    assert_rows(&batches, vec![(13, Some("mock[m]: d".into()))], true);
    assert_eq!(submitted(plan.as_ref()), 4);

    let (plan, batches) = run(&ctx, &format!("SELECT id {from}")).await?;
    assert_eq!(
        batches[0]
            .column(0)
            .as_any()
            .downcast_ref::<Int64Array>()
            .unwrap()
            .values(),
        &[13],
    );
    assert_eq!(submitted(plan.as_ref()), 4);
    Ok(())
}

#[tokio::test]
async fn large_utf8_prompts_keep_row_alignment_and_skip_nulls() -> Result<()> {
    let ctx = SessionContext::new();
    ctx.register_batch("typed", typed_prompt_batch(DataType::LargeUtf8)?)?;
    assert_typed_prompts(&ctx, DataType::LargeUtf8).await
}

#[cfg(feature = "parquet")]
#[tokio::test]
async fn parquet_scan_reads_utf8_view_prompts_and_skips_nulls() -> Result<()> {
    use std::fs::File;

    use datafusion::parquet::arrow::ArrowWriter;
    use datafusion::prelude::ParquetReadOptions;

    let directory = tempfile::tempdir()?;
    let path = directory.path().join("prompts.parquet");
    let batch = typed_prompt_batch(DataType::Utf8)?;
    let mut writer = ArrowWriter::try_new(File::create(&path)?, batch.schema(), None)?;
    writer.write(&batch)?;
    writer.close()?;

    let ctx = SessionContext::new();
    ctx.register_parquet(
        "typed",
        path.to_str().unwrap(),
        ParquetReadOptions::default(),
    )
    .await?;
    assert_typed_prompts(&ctx, DataType::Utf8View).await
}

fn typed_prompt_batch(data_type: DataType) -> Result<RecordBatch> {
    let prompts = super::harness::string_array(
        &data_type,
        vec![
            Some("same"),
            None,
            Some("longer than twelve bytes"),
            Some("same"),
        ],
    );
    Ok(RecordBatch::try_new(
        Arc::new(Schema::new(vec![
            Field::new("id", DataType::Int64, false),
            Field::new("text", data_type, true),
        ])),
        vec![Arc::new(Int64Array::from(vec![9, 2, 7, 4])), prompts],
    )?)
}

async fn assert_typed_prompts(ctx: &SessionContext, scan_type: DataType) -> Result<()> {
    let (scan, batches) = run(ctx, "SELECT text FROM typed").await?;
    assert_eq!(scan.schema().field(0).data_type(), &scan_type);
    assert!(!batches.is_empty());
    for batch in batches {
        assert_eq!(batch.column(0).data_type(), &scan_type);
    }
    let (plan, batches) =
        run(ctx, "SELECT id, vllm(text, 'm', '{}') AS r FROM typed").await?;
    assert_rows(
        &batches,
        vec![
            (9, Some("mock[m]: same".into())),
            (2, None),
            (7, Some("mock[m]: longer than twelve bytes".into())),
            (4, Some("mock[m]: same".into())),
        ],
        false,
    );
    assert_eq!(submitted(plan.as_ref()), 3);
    Ok(())
}
