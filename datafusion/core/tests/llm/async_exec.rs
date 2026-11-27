// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: See the NOTICE file distributed with this work for additional information regarding copyright ownership
// SPDX-FileContributor: Licensed to the Apache Software Foundation (ASF) under one or more contributor license agreements

use std::any::Any;
use std::collections::HashMap;
use std::sync::Arc;
use std::time::Duration;

use async_trait::async_trait;
use datafusion::arrow::array::{Int64Array, StringArray};
use datafusion::arrow::datatypes::{DataType, Field, Schema};
use datafusion::arrow::record_batch::RecordBatch;
use datafusion::catalog::CatalogProviderList;
use datafusion::common::config::TableOptions;
use datafusion::common::{DFSchema, Result};
use datafusion::datasource::MemTable;
use datafusion::execution::runtime_env::RuntimeEnv;
use datafusion::execution::{SessionState, SessionStateBuilder, TaskContext};
use datafusion::llm::{
    LlmBackendError, LlmExec, LlmNode, LlmProviderRegistry, LlmResponse, LlmUsage,
    MockLlmBackend, plan_llm,
};
use datafusion::logical_expr::execution_props::ExecutionProps;
use datafusion::logical_expr::physical_planning_context::PhysicalPlanningContext;
use datafusion::logical_expr::registry::ExtensionTypeRegistryRef;
use datafusion::logical_expr::{
    AggregateUDF, Expr, HigherOrderUDF, LogicalPlan, ScalarUDF, WindowUDF,
};
use datafusion::physical_expr::PhysicalExpr;
use datafusion::physical_plan::{
    ExecutionPlan, collect, execute_stream, execute_stream_partitioned,
};
use datafusion::prelude::{SessionConfig, SessionContext};
use datafusion_session::Session;
use futures::stream::select_all;
use futures::{StreamExt, TryStreamExt, poll};

use super::harness::{
    answer_rows, assert_plan_error, assert_rows, context, exec_nodes, metric,
    register_text, run, strings, submitted, test_context,
};
use super::scripted_backend::{InferenceProbe, ScriptedBackend};

async fn plan(ctx: &SessionContext, sql: &str) -> Result<Arc<dyn ExecutionPlan>> {
    ctx.sql(sql).await?.create_physical_plan().await
}

async fn wait_for(mut ready: impl FnMut() -> bool) {
    for _ in 0..100 {
        if ready() {
            return;
        }
        tokio::task::yield_now().await;
    }
    panic!("driver did not reach the expected state");
}

fn register_script(
    ctx: &SessionContext,
    limit: usize,
) -> Result<(LlmProviderRegistry, InferenceProbe)> {
    let (backend, probe) = ScriptedBackend::new();
    let registry = ctx.state().llm_provider_registry().clone();
    registry.register_with_limit("script", backend, limit)?;
    Ok((registry, probe))
}

fn rate_limited(seconds: u64) -> LlmBackendError {
    LlmBackendError::RateLimited {
        retry_after: Some(Duration::from_secs(seconds)),
        message: "try again".into(),
    }
}

#[tokio::test]
async fn shared_limits_and_out_of_order_completion_preserve_rows() -> Result<()> {
    // Exercise each semaphore independently with the other limit set to four.
    for (provider_limit, inflight_limit) in [(2, 4), (4, 2)] {
        let ctx = test_context(8, 4);
        let (registry, mut probe) = register_script(&ctx, provider_limit)?;
        let task_ctx = ctx.task_ctx();
        let task_registry = task_ctx
            .session_config()
            .get_extension::<LlmProviderRegistry>()
            .unwrap();
        let schema = Arc::new(Schema::new(vec![
            Field::new("id", DataType::Int64, false),
            Field::new("text", DataType::Utf8, false),
        ]));
        let partitions = (0..4i64)
            .map(|i| {
                Ok(vec![RecordBatch::try_new(
                    Arc::clone(&schema),
                    vec![
                        Arc::new(Int64Array::from(vec![i])),
                        Arc::new(StringArray::from(vec![i.to_string()])),
                    ],
                )?])
            })
            .collect::<Result<Vec<_>>>()?;
        ctx.register_table("t", Arc::new(MemTable::try_new(schema, partitions)?))?;
        let plan = plan(
            &ctx,
            &format!(
                r#"SELECT id, llm(text, 'm', '{{"provider":"script","inflight_limit":{inflight_limit}}}') FROM t"#
            ),
        ).await?;

        let streams = execute_stream_partitioned(Arc::clone(&plan), task_ctx)?;
        assert_eq!(streams.len(), 4);
        let mut output = select_all(streams);
        assert!(poll!(output.next()).is_pending());

        let first = probe.next_attempt().await;
        let second = probe.next_attempt().await;
        tokio::task::yield_now().await;
        assert_eq!(probe.counts().in_flight, 2);
        assert_eq!(probe.counts().peak_in_flight, 2);
        assert_eq!(
            task_registry.available_permits("script"),
            Some(provider_limit - 2)
        );
        probe.assert_no_new_attempt();

        let second_id = second.prompts[0].parse::<i64>().unwrap();
        second.succeed("answer:", None);
        let second_batch = output.next().await.unwrap()?;
        assert_eq!(
            answer_rows(std::slice::from_ref(&second_batch)),
            vec![(second_id, Some(format!("answer:{second_id}")))]
        );
        first.succeed("answer:", None);
        probe.next_attempt().await.succeed("answer:", None);
        probe.next_attempt().await.succeed("answer:", None);
        let mut batches = vec![second_batch];
        batches.extend(output.try_collect::<Vec<_>>().await?);
        assert_rows(
            &batches,
            (0..4)
                .map(|id| (id, Some(format!("answer:{id}"))))
                .collect(),
            false,
        );
        assert_eq!(submitted(plan.as_ref()), 4);
        assert_eq!(probe.counts().peak_in_flight, 2);
        assert_eq!(probe.counts().in_flight, 0);
        assert_eq!(registry.available_permits("script"), Some(provider_limit));
    }
    Ok(())
}

#[tokio::test]
async fn slow_consumer_bounds_prefetch_and_nulls_skip_the_backend() -> Result<()> {
    let ctx = test_context(16, 1);
    register_text(
        &ctx,
        "t",
        &[None, Some("a"), Some("b"), Some("c"), Some("d"), Some("e")],
    )?;
    let (registry, mut probe) = register_script(&ctx, 1)?;
    let plan = plan(
        &ctx,
        r#"SELECT llm(text, 'm', '{"provider":"script","inflight_limit":1}') FROM t"#,
    )
    .await?;
    let mut stream = execute_stream(Arc::clone(&plan), ctx.task_ctx())?;
    probe.assert_no_new_attempt();
    let first = stream.next().await.unwrap()?;
    assert_eq!(strings(&[first], 0), [None]);
    for prompt in ["a", "b"] {
        let attempt = probe.next_attempt().await;
        assert_eq!(attempt.prompts, [prompt]);
        attempt.succeed("", None);
    }
    wait_for(|| probe.counts().in_flight == 0).await;
    tokio::task::yield_now().await;
    probe.assert_no_new_attempt();
    assert_eq!(registry.available_permits("script"), Some(1));
    drop(stream);
    assert_eq!(submitted(plan.as_ref()), 2);
    Ok(())
}

#[tokio::test]
async fn drop_cancels_permit_waiters_and_active_requests_across_queries() -> Result<()> {
    let ctx = test_context(8, 1);
    let (registry, mut probe) = register_script(&ctx, 1)?;
    let sql = r#"SELECT llm('one', 'm', '{"provider":"script","inflight_limit":1}')"#;
    let active_plan = plan(&ctx, sql).await?;
    let waiting_plan = plan(&ctx, sql).await?;
    let mut active = execute_stream(Arc::clone(&active_plan), ctx.task_ctx())?;
    assert!(poll!(active.next()).is_pending());
    let mut pending = probe.next_attempt().await;

    let mut waiting = execute_stream(Arc::clone(&waiting_plan), ctx.task_ctx())?;
    assert!(poll!(waiting.next()).is_pending());
    tokio::task::yield_now().await;
    assert_eq!(registry.available_permits("script"), Some(0));
    probe.assert_no_new_attempt();
    drop(waiting);
    wait_for(|| metric(waiting_plan.as_ref(), "cancelled_requests") == 1).await;
    assert_eq!(submitted(waiting_plan.as_ref()), 0);
    assert_eq!(probe.counts().in_flight, 1);

    drop(active);
    pending.reply.closed().await;
    wait_for(|| metric(active_plan.as_ref(), "cancelled_requests") == 1).await;
    assert_eq!(probe.counts().cancelled_attempts, 1);
    assert_eq!(probe.counts().in_flight, 0);
    assert_eq!(registry.available_permits("script"), Some(1));
    probe.assert_no_new_attempt();

    // Re-executing the same plan also proves its operator-level slot was returned.
    for plan in [active_plan, waiting_plan] {
        let mut stream = execute_stream(plan, ctx.task_ctx())?;
        assert!(poll!(stream.next()).is_pending());
        probe.next_attempt().await.succeed("", None);
        assert_eq!(
            strings(&stream.try_collect::<Vec<_>>().await?, 0),
            [Some("one")]
        );
    }
    Ok(())
}

#[tokio::test(start_paused = true)]
async fn drop_during_retry_backoff_releases_both_limits() -> Result<()> {
    let ctx = test_context(8, 1);
    let (registry, mut probe) = register_script(&ctx, 1)?;
    let plan = plan(
        &ctx,
        r#"SELECT llm('one', 'm', '{"provider":"script","inflight_limit":1}')"#,
    )
    .await?;
    let mut stream = execute_stream(Arc::clone(&plan), ctx.task_ctx())?;
    assert!(poll!(stream.next()).is_pending());
    probe
        .next_attempt()
        .await
        .reply
        .send(Err(rate_limited(8)))
        .unwrap();
    wait_for(|| probe.counts().in_flight == 0).await;
    assert_eq!(registry.available_permits("script"), Some(0));
    drop(stream);
    wait_for(|| metric(plan.as_ref(), "cancelled_requests") == 1).await;
    assert_eq!(registry.available_permits("script"), Some(1));
    tokio::time::advance(Duration::from_secs(60)).await;
    probe.assert_no_new_attempt();

    let mut stream = execute_stream(Arc::clone(&plan), ctx.task_ctx())?;
    assert!(poll!(stream.next()).is_pending());
    probe.next_attempt().await.succeed("", None);
    assert_eq!(
        strings(&stream.try_collect::<Vec<_>>().await?, 0),
        [Some("one")]
    );
    Ok(())
}

#[tokio::test]
async fn sql_limit_cancels_requests_after_two_matching_outputs() -> Result<()> {
    let ctx = test_context(1, 1);
    register_text(
        &ctx,
        "t",
        &[Some("a"), Some("b"), Some("c"), Some("d"), Some("e")],
    )?;
    let (registry, mut probe) = register_script(&ctx, 3)?;
    let plan = plan(&ctx, r#"
        SELECT id, r FROM (
            SELECT id, llm(text, 'm', '{"provider":"script","inflight_limit":3}') AS r FROM t
        ) q WHERE r LIKE 'match:%' LIMIT 2
    "#).await?;
    let mut stream = execute_stream(Arc::clone(&plan), ctx.task_ctx())?;
    assert!(poll!(stream.next()).is_pending());
    let first = probe.next_attempt().await;
    let second = probe.next_attempt().await;
    let mut third = probe.next_attempt().await;
    first.succeed("match:", None);
    second.succeed("match:", None);
    let batches = stream.try_collect::<Vec<_>>().await?;
    assert_eq!(answer_rows(&batches).len(), 2);
    third.reply.closed().await;
    wait_for(|| registry.available_permits("script") == Some(3)).await;
    let attempts = probe.counts().backend_attempts;
    assert!(metric(plan.as_ref(), "cancelled_requests") > 0);
    assert_eq!(probe.counts().in_flight, 0);
    tokio::task::yield_now().await;
    assert_eq!(probe.counts().backend_attempts, attempts);
    Ok(())
}

#[tokio::test(start_paused = true)]
async fn retry_after_preserves_permits_and_counts_attempts_and_usage() -> Result<()> {
    let ctx = test_context(8, 1);
    let (registry, mut probe) = register_script(&ctx, 1)?;
    let plan = plan(&ctx, r#"SELECT llm('one', 'm', '{"provider":"script"}')"#).await?;
    let mut stream = execute_stream(Arc::clone(&plan), ctx.task_ctx())?;
    assert!(poll!(stream.next()).is_pending());
    probe
        .next_attempt()
        .await
        .reply
        .send(Err(rate_limited(8)))
        .unwrap();
    wait_for(|| probe.counts().in_flight == 0).await;

    tokio::time::advance(Duration::from_secs(7)).await;
    tokio::task::yield_now().await;
    probe.assert_no_new_attempt();
    assert_eq!(registry.available_permits("script"), Some(0));
    tokio::time::advance(Duration::from_secs(1)).await;
    let retry = probe.next_attempt().await;
    assert_eq!(retry.prompts, ["one"]);
    retry.succeed(
        "answer:",
        Some(LlmUsage {
            input_tokens: 4,
            output_tokens: 9,
        }),
    );
    let batches = stream.try_collect::<Vec<_>>().await?;
    assert_eq!(strings(&batches, 0), [Some("answer:one")]);
    for (name, expected) in [
        ("submitted_prompts", 1),
        ("backend_attempts", 2),
        ("retries", 1),
        ("input_tokens", 4),
        ("output_tokens", 9),
        ("failed_rows", 0),
        ("cancelled_requests", 0),
    ] {
        assert_eq!(metric(plan.as_ref(), name), expected, "{name}");
    }
    assert_eq!(registry.available_permits("script"), Some(1));
    Ok(())
}

#[tokio::test(start_paused = true)]
async fn long_retry_after_is_capped_at_sixty_seconds() -> Result<()> {
    let ctx = test_context(8, 1);
    let (registry, mut probe) = register_script(&ctx, 1)?;
    let plan = plan(&ctx, r#"SELECT llm('one', 'm', '{"provider":"script"}')"#).await?;
    let mut stream = execute_stream(Arc::clone(&plan), ctx.task_ctx())?;
    assert!(poll!(stream.next()).is_pending());
    probe
        .next_attempt()
        .await
        .reply
        .send(Err(rate_limited(3600)))
        .unwrap();
    wait_for(|| probe.counts().in_flight == 0).await;

    tokio::time::advance(Duration::from_secs(59)).await;
    tokio::task::yield_now().await;
    probe.assert_no_new_attempt();
    assert_eq!(registry.available_permits("script"), Some(0));
    tokio::time::advance(Duration::from_secs(1)).await;
    let retry = tokio::time::timeout(Duration::from_millis(10), probe.next_attempt())
        .await
        .expect("retry should start at the 60-second cap");
    retry.succeed("answer:", None);
    let batches = stream.try_collect::<Vec<_>>().await?;
    assert_eq!(strings(&batches, 0), [Some("answer:one")]);
    assert_eq!(metric(plan.as_ref(), "backend_attempts"), 2);
    assert_eq!(metric(plan.as_ref(), "retries"), 1);
    assert_eq!(registry.available_permits("script"), Some(1));
    Ok(())
}

#[tokio::test(start_paused = true)]
async fn permanent_and_exhausted_failures_follow_on_error() -> Result<()> {
    for (transient, on_error, attempts) in [
        (false, "raise", 1),
        (false, "log", 1),
        (false, "ignore", 1),
        (true, "raise", 3),
        (true, "ignore", 3),
    ] {
        let ctx = test_context(8, 1);
        register_text(&ctx, "t", &[Some("one"), Some("two")])?;
        let (registry, mut probe) = register_script(&ctx, 1)?;
        let plan = plan(&ctx, &format!(
            r#"SELECT llm(text, 'm', '{{"provider":"script","batch_size":2,"on_error":"{on_error}","max_retries":2}}') FROM t"#
        )).await?;
        let mut stream = execute_stream(Arc::clone(&plan), ctx.task_ctx())?;
        assert!(poll!(stream.next()).is_pending());
        for _ in 0..attempts {
            let attempt = probe.next_attempt().await;
            assert_eq!(attempt.prompts, ["one", "two"]);
            let error = if transient {
                LlmBackendError::Transient {
                    status: Some(500),
                    message: "temporary error".into(),
                }
            } else {
                LlmBackendError::Permanent {
                    status: Some(400),
                    message: "permanent error".into(),
                }
            };
            attempt.reply.send(Err(error)).unwrap();
        }
        let result = stream.try_collect::<Vec<_>>().await;
        if on_error == "raise" {
            assert!(
                result
                    .unwrap_err()
                    .to_string()
                    .contains("llm backend failed")
            );
        } else {
            assert_eq!(strings(&result?, 0), [None, None]);
        }
        assert_eq!(submitted(plan.as_ref()), 2);
        assert_eq!(metric(plan.as_ref(), "backend_attempts"), attempts);
        assert_eq!(metric(plan.as_ref(), "retries"), attempts - 1);
        assert_eq!(metric(plan.as_ref(), "failed_rows"), 2);
        assert_eq!(registry.available_permits("script"), Some(1));
        probe.assert_no_new_attempt();
    }
    Ok(())
}

#[tokio::test]
async fn response_with_wrong_cardinality_is_permanent() -> Result<()> {
    let ctx = test_context(8, 1);
    let (_, mut probe) = register_script(&ctx, 1)?;
    let plan = plan(&ctx, r#"SELECT llm('one', 'm', '{"provider":"script"}')"#).await?;
    let mut stream = execute_stream(Arc::clone(&plan), ctx.task_ctx())?;
    assert!(poll!(stream.next()).is_pending());
    probe
        .next_attempt()
        .await
        .reply
        .send(Ok(LlmResponse {
            outputs: vec![],
            usage: None,
        }))
        .unwrap();
    let error = stream.next().await.unwrap().unwrap_err();
    assert!(
        error
            .to_string()
            .contains("returned 0 outputs for 1 prompts"),
        "{error}"
    );
    assert_eq!(metric(plan.as_ref(), "backend_attempts"), 1);
    assert_eq!(metric(plan.as_ref(), "failed_rows"), 1);
    probe.assert_no_new_attempt();
    Ok(())
}

#[tokio::test]
async fn deduplicate_false_and_unknown_provider() -> Result<()> {
    let ctx = context()?;
    let sql = r#"SELECT llm(text, 'm', '{"deduplicate":false}') AS first, llm(text, 'm', '{"deduplicate":false}') AS second FROM t LIMIT 1"#;
    let (plan, _) = run(&ctx, sql).await?;
    assert_eq!(exec_nodes(plan.as_ref()).len(), 2);
    assert_eq!(submitted(plan.as_ref()), 2);
    assert_plan_error(
        &ctx,
        r#"SELECT llm(text, 'm', '{"provider":"nonexistent"}') FROM t"#,
        &["Unknown llm provider 'nonexistent'"],
    )
    .await;
    Ok(())
}

#[tokio::test]
async fn builder_registry_can_replace_the_default_backend() -> Result<()> {
    let registry = LlmProviderRegistry::default();
    let (backend, mut probe) = ScriptedBackend::new();
    registry.register("default", backend)?;
    let state = SessionStateBuilder::new_from_existing(SessionContext::new().state())
        .with_llm_provider_registry(registry)
        .build();
    let ctx = SessionContext::new_with_state(state);
    let plan = plan(&ctx, "SELECT llm('one', 'm', '{}')").await?;
    let mut stream = execute_stream(plan, ctx.task_ctx())?;
    assert!(poll!(stream.next()).is_pending());
    probe.next_attempt().await.succeed("default:", None);
    assert_eq!(
        strings(&stream.try_collect::<Vec<_>>().await?, 0),
        [Some("default:one")]
    );
    Ok(())
}

#[tokio::test]
async fn provider_batch_limit_is_checked_before_inference() -> Result<()> {
    let ctx = SessionContext::new();
    let (backend, mut probe) = ScriptedBackend::with_max_batch_size(Some(1));
    ctx.state()
        .llm_provider_registry()
        .register("script", backend)?;
    let error = plan(
        &ctx,
        r#"SELECT llm('one', 'm', '{"provider":"script","batch_size":2}')"#,
    )
    .await
    .unwrap_err();
    assert!(
        error
            .to_string()
            .contains("accepts at most 1 prompt(s) per request")
    );
    probe.assert_no_new_attempt();
    Ok(())
}

#[tokio::test]
async fn builder_registry_precedence_is_shared_with_planning_and_task_context()
-> Result<()> {
    let explicit = LlmProviderRegistry::default();
    let configured = LlmProviderRegistry::default();
    let inherited = LlmProviderRegistry::default();
    let providers = ["explicit", "configured", "inherited"];
    for (name, registry) in
        providers
            .into_iter()
            .zip([&explicit, &configured, &inherited])
    {
        registry.register_with_limit(name, Arc::new(MockLlmBackend), 1)?;
    }
    let source = SessionStateBuilder::new_with_default_features()
        .with_llm_provider_registry(inherited)
        .build();
    let config = SessionConfig::new().with_extension(Arc::new(configured));
    let states = [
        (
            SessionStateBuilder::new_from_existing(source.clone())
                .with_config(config.clone())
                .with_llm_provider_registry(explicit)
                .build(),
            Some("explicit"),
        ),
        (
            SessionStateBuilder::new_from_existing(source.clone())
                .with_config(config)
                .build(),
            Some("configured"),
        ),
        (
            SessionStateBuilder::new_from_existing(source)
                .with_config(SessionConfig::new())
                .build(),
            Some("inherited"),
        ),
        (
            SessionStateBuilder::new_with_default_features().build(),
            None,
        ),
    ];
    for (state, selected) in states {
        let task_ctx = state.task_ctx();
        let task_registry = task_ctx
            .session_config()
            .get_extension::<LlmProviderRegistry>()
            .unwrap();
        for provider in providers {
            let expected = (selected == Some(provider)).then_some(1);
            assert_eq!(
                state.llm_provider_registry().available_permits(provider),
                expected
            );
            assert_eq!(task_registry.available_permits(provider), expected);
        }
        let sql = selected.map_or_else(
            || "SELECT llm('one', 'm', '{}')".to_owned(),
            |name| format!(r#"SELECT llm('one', 'm', '{{"provider":"{name}"}}')"#),
        );
        let (_, batches) = run(&SessionContext::new_with_state(state), &sql).await?;
        assert_eq!(strings(&batches, 0), [Some("mock[m]: one")]);
    }
    Ok(())
}

#[tokio::test]
async fn late_registration_is_visible_to_coordinator_and_existing_worker_contexts()
-> Result<()> {
    let ctx = SessionContext::new();
    let state = ctx.state();
    let task_ctx = ctx.task_ctx();
    let (backend, mut probe) = ScriptedBackend::new();
    state.llm_provider_registry().register("late", backend)?;
    let plan = plan(&ctx, r#"SELECT llm('one', 'm', '{"provider":"late"}')"#).await?;
    let nodes = exec_nodes(plan.as_ref());
    let llm = nodes[0].downcast_ref::<LlmExec>().unwrap();
    let worker = LlmExec::try_new_from_task_context(
        Arc::clone(llm.input()),
        Arc::clone(llm.prompt()),
        llm.model().to_owned(),
        llm.options_json(),
        llm.output_col().to_owned(),
        &task_ctx,
    )?;
    for mut stream in [
        execute_stream(Arc::clone(&plan), Arc::clone(&task_ctx))?,
        worker.execute(0, task_ctx)?,
    ] {
        assert!(poll!(stream.next()).is_pending());
        probe.next_attempt().await.succeed("late:", None);
        let result_col = stream.schema().fields().len() - 1;
        assert_eq!(
            strings(&stream.try_collect::<Vec<_>>().await?, result_col),
            [Some("late:one")]
        );
    }
    Ok(())
}

#[tokio::test]
async fn coordinator_plans_with_a_custom_session_without_a_registry() -> Result<()> {
    let ctx = context()?;
    let optimized = ctx
        .sql("SELECT llm(t.text, 'm', '{}') FROM t")
        .await?
        .into_optimized_plan()?;
    let LogicalPlan::Projection(projection) = optimized else {
        panic!("expected projection over LlmNode: {optimized:?}");
    };
    let LogicalPlan::Extension(extension) = projection.input.as_ref() else {
        panic!(
            "expected LlmNode under the projection: {:?}",
            projection.input
        );
    };
    let node = extension.node.as_any().downcast_ref::<LlmNode>().unwrap();
    assert_eq!(node.model(), "m");
    assert_eq!(node.options_json(), "{}");
    assert!(
        matches!(node.prompt_expr(), Expr::Column(column) if column.relation.is_some())
    );
    let state = ctx.state();
    let physical_input = state.create_physical_plan(node.input()).await?;
    let result_col = physical_input.schema().fields().len();
    let session: &dyn Session = &PlanningSession {
        state,
        config: SessionConfig::new(),
    };
    assert!(
        session
            .config()
            .get_extension::<LlmProviderRegistry>()
            .is_none()
    );
    let exec = plan_llm(
        extension.node.as_ref(),
        &[node.input()],
        &[physical_input],
        session,
        &PhysicalPlanningContext::default(),
    )?
    .unwrap();
    assert_eq!(exec.schema().field(result_col).name(), node.output_col());
    let batches = collect(exec, session.task_ctx()).await?;
    assert_eq!(
        strings(&batches, result_col),
        [
            Some("mock[m]: a"),
            Some("mock[m]: b"),
            Some("mock[m]: c"),
            Some("mock[m]: d")
        ]
    );
    Ok(())
}

struct PlanningSession {
    state: SessionState,
    config: SessionConfig,
}

#[async_trait]
impl Session for PlanningSession {
    fn session_id(&self) -> &str {
        self.state.session_id()
    }

    fn config(&self) -> &SessionConfig {
        &self.config
    }

    fn catalog_list(&self) -> Arc<dyn CatalogProviderList> {
        Arc::clone(self.state.catalog_list())
    }

    async fn create_physical_plan(
        &self,
        plan: &LogicalPlan,
    ) -> Result<Arc<dyn ExecutionPlan>> {
        self.state.create_physical_plan(plan).await
    }

    fn create_physical_expr(
        &self,
        expr: Expr,
        schema: &DFSchema,
    ) -> Result<Arc<dyn PhysicalExpr>> {
        Session::create_physical_expr(&self.state, expr, schema)
    }

    fn scalar_functions(&self) -> &HashMap<String, Arc<ScalarUDF>> {
        self.state.scalar_functions()
    }

    fn higher_order_functions(&self) -> &HashMap<String, Arc<HigherOrderUDF>> {
        self.state.higher_order_functions()
    }

    fn aggregate_functions(&self) -> &HashMap<String, Arc<AggregateUDF>> {
        self.state.aggregate_functions()
    }

    fn window_functions(&self) -> &HashMap<String, Arc<WindowUDF>> {
        self.state.window_functions()
    }

    fn extension_type_registry(&self) -> &ExtensionTypeRegistryRef {
        self.state.extension_type_registry()
    }

    fn runtime_env(&self) -> &Arc<RuntimeEnv> {
        self.state.runtime_env()
    }

    fn execution_props(&self) -> &ExecutionProps {
        self.state.execution_props()
    }

    fn as_any(&self) -> &dyn Any {
        self
    }

    fn table_options(&self) -> &TableOptions {
        self.state.table_options()
    }

    fn table_options_mut(&mut self) -> &mut TableOptions {
        self.state.table_options_mut()
    }

    fn task_ctx(&self) -> Arc<TaskContext> {
        Arc::new(TaskContext::from(self as &dyn Session))
    }
}
