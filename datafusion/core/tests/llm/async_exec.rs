// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: See the NOTICE file distributed with this work for additional information regarding copyright ownership
// SPDX-FileContributor: Licensed to the Apache Software Foundation (ASF) under one or more contributor license agreements

use std::sync::Arc;
use std::time::Duration;

use datafusion::arrow::array::{Int64Array, StringArray};
use datafusion::arrow::datatypes::{DataType, Field, Schema};
use datafusion::arrow::record_batch::RecordBatch;
use datafusion::common::Result;
use datafusion::datasource::MemTable;
use datafusion::execution::SessionStateBuilder;
use datafusion::llm::{LlmBackendError, LlmProviderRegistry, LlmResponse, LlmUsage};
use datafusion::physical_plan::{
    ExecutionPlan, execute_stream, execute_stream_partitioned,
};
use datafusion::prelude::SessionContext;
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
    let registry = ctx.llm_provider_registry();
    registry.register_with_limit("script", backend, limit)?;
    Ok((registry, probe))
}

fn rate_limited() -> LlmBackendError {
    LlmBackendError::RateLimited {
        retry_after: Some(Duration::from_secs(8)),
        message: "try again".into(),
    }
}

#[tokio::test]
async fn shared_limits_and_out_of_order_completion_preserve_rows() -> Result<()> {
    // Exercise each semaphore independently with the other limit set to four.
    for (provider_limit, inflight_limit) in [(2, 4), (4, 2)] {
        let ctx = test_context(8, 4);
        let (registry, mut probe) = register_script(&ctx, provider_limit)?;
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

        let streams = execute_stream_partitioned(Arc::clone(&plan), ctx.task_ctx())?;
        assert_eq!(streams.len(), 4);
        let mut output = select_all(streams);
        assert!(poll!(output.next()).is_pending());

        let first = probe.next_attempt().await;
        let second = probe.next_attempt().await;
        tokio::task::yield_now().await;
        assert_eq!(probe.counts().in_flight, 2);
        assert_eq!(probe.counts().peak_in_flight, 2);
        assert_eq!(
            registry.available_permits("script"),
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
        .send(Err(rate_limited()))
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
        .send(Err(rate_limited()))
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
