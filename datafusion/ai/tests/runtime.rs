// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: See the NOTICE file distributed with this work for additional information regarding copyright ownership
// SPDX-FileContributor: Licensed to the Apache Software Foundation (ASF) under one or more contributor license agreements

use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::Duration;

use arrow::array::{Int32Array, StringArray};
use arrow::datatypes::{DataType, Field, Schema};
use arrow::record_batch::RecordBatch;
use async_trait::async_trait;
use datafusion_ai::llm::{
    CancellationToken, LlmBackend, LlmBackendError, LlmExec, LlmProviderRegistry,
    LlmRequest, LlmResponse, MockLlmBackend, OnError,
};
use datafusion_ai::{
    CheckpointOutputColumn, CheckpointableTarget, validate_checkpoint_target,
};
use datafusion_common::{DataFusionError, Result};
use datafusion_execution::TaskContext;
use datafusion_execution::config::SessionConfig;
use datafusion_physical_expr::expressions::Column;
use datafusion_physical_plan::ExecutionPlan;
use datafusion_physical_plan::test::TestMemoryExec;
use futures::{StreamExt, TryStreamExt, poll};
use tokio::sync::{mpsc, oneshot};
use tokio::time::{Instant, advance, timeout};

const TEST_TIMEOUT: Duration = Duration::from_secs(5);

fn input(prompts: &[Option<&str>]) -> Result<Arc<dyn ExecutionPlan>> {
    let schema = Arc::new(Schema::new(vec![
        Field::new("id", DataType::Int32, false),
        Field::new("text", DataType::Utf8, true),
    ]));
    let batch = RecordBatch::try_new(
        Arc::clone(&schema),
        vec![
            Arc::new(Int32Array::from_iter_values(
                0..i32::try_from(prompts.len()).unwrap(),
            )),
            Arc::new(StringArray::from(prompts.to_vec())),
        ],
    )?;
    Ok(TestMemoryExec::try_new_exec(&[vec![batch]], schema, None)?)
}

fn task_context(registry: &LlmProviderRegistry) -> Arc<TaskContext> {
    Arc::new(TaskContext::default().with_session_config(
        SessionConfig::new().with_extension(Arc::new(registry.clone())),
    ))
}

fn decode(
    input: Arc<dyn ExecutionPlan>,
    options: &str,
    ctx: &TaskContext,
) -> Result<LlmExec> {
    LlmExec::try_new_from_task_context(
        input,
        Arc::new(Column::new("text", 1)),
        "test-model".into(),
        options,
        "answer".into(),
        ctx,
    )
}

fn metric(exec: &LlmExec, name: &str) -> usize {
    exec.metrics()
        .unwrap()
        .sum_by_name(name)
        .map_or(0, |value| value.as_usize())
}

fn submitted(exec: &LlmExec) -> usize {
    metric(exec, "submitted_prompts")
}

fn rows(batches: &[RecordBatch]) -> Vec<(i32, Option<&str>, Option<&str>)> {
    let mut rows = Vec::new();
    for batch in batches {
        let ids = batch
            .column(0)
            .as_any()
            .downcast_ref::<Int32Array>()
            .unwrap();
        let prompts = batch
            .column(1)
            .as_any()
            .downcast_ref::<StringArray>()
            .unwrap();
        let answers = batch
            .column(2)
            .as_any()
            .downcast_ref::<StringArray>()
            .unwrap();
        rows.extend(
            ids.values()
                .iter()
                .copied()
                .zip(prompts.iter().zip(answers.iter()))
                .map(|(id, (prompt, answer))| (id, prompt, answer)),
        );
    }
    rows.sort_unstable_by_key(|&(id, _, _)| id);
    rows
}

#[test]
fn worker_decode_requires_a_registry_even_for_default_options() -> Result<()> {
    let error =
        decode(input(&[Some("prompt")])?, "{}", &TaskContext::default()).unwrap_err();
    assert!(
        matches!(error, DataFusionError::Plan(message) if message.contains("LlmProviderRegistry extension"))
    );
    Ok(())
}

#[test]
fn worker_decode_rejects_invalid_identifiers_options_and_sensitive_keys() -> Result<()> {
    let ctx = task_context(&LlmProviderRegistry::default());
    let input = input(&[Some("prompt")])?;
    for (model, output_col, json, message) in [
        (" ", "answer", "{}", "model must be non-empty"),
        ("model", " ", "{}", "output column must be non-empty"),
        (
            "model",
            "answer",
            r#"{"provider":"missing"}"#,
            "Unknown llm provider",
        ),
        ("model", "answer", "{", "invalid llm() options_json"),
        ("model", "answer", "[]", "expected a JSON object"),
        (
            "model",
            "answer",
            r#"{"batch_size":0}"#,
            "batch_size must be",
        ),
        (
            "model",
            "answer",
            r#"{"API_KEY":"sentinel","API_KEY":null}"#,
            "sensitive credential key",
        ),
        (
            "model",
            "answer",
            r#"{"extra_body":{"nested":[{"ToKeN":"sentinel"}]},"extra_body":{}}"#,
            "sensitive credential key",
        ),
        (
            "model",
            "answer",
            r#"{"extra_body":{"nested":{"Secret":"sentinel"},"nested":null}}"#,
            "sensitive credential key",
        ),
        (
            "model",
            "answer",
            r#"{"extra_body":{"nested":[[{"pass\u0077ord":"sentinel"}]]}}"#,
            "sensitive credential key",
        ),
    ] {
        let error = LlmExec::try_new_from_task_context(
            Arc::clone(&input),
            Arc::new(Column::new("text", 1)),
            model.into(),
            json,
            output_col.into(),
            &ctx,
        )
        .unwrap_err();
        assert!(
            matches!(&error, DataFusionError::Plan(actual) if actual.contains(message) && !actual.contains("sentinel"),),
            "{error}"
        );
    }
    Ok(())
}

#[tokio::test]
async fn decoded_operators_preserve_rows_and_own_independent_metrics() -> Result<()> {
    let registry = LlmProviderRegistry::default();
    registry.register("worker", Arc::new(MockLlmBackend))?;
    let ctx = task_context(&registry);
    let input = input(&[Some("first"), None, Some("last")])?;
    let options = r#"{ "provider":"worker", "extra_body":{"z":[{"b":1,"a":2}],"a":{}}, "batch_size":2 }"#;
    let first = decode(Arc::clone(&input), options, &ctx)?;
    let second = decode(Arc::clone(&input), options, &ctx)?;
    assert!(Arc::ptr_eq(first.input(), &input));
    assert_eq!(first.model(), "test-model");
    assert_eq!(first.options().provider.as_deref(), Some("worker"));
    assert_eq!(first.options().batch_size, Some(2));
    assert_eq!(first.output_col(), "answer");
    assert_eq!(
        first.options_json(),
        r#"{"batch_size":2,"extra_body":{"a":{},"z":[{"a":2,"b":1}]},"provider":"worker"}"#,
    );
    assert_eq!(
        first.prompt().downcast_ref::<Column>(),
        Some(&Column::new("text", 1))
    );
    let expected = [
        (0, Some("first"), Some("mock[test-model]: first")),
        (1, None, None),
        (2, Some("last"), Some("mock[test-model]: last")),
    ];

    let batches = first
        .execute(0, Arc::clone(&ctx))?
        .try_collect::<Vec<_>>()
        .await?;
    assert_eq!(rows(&batches), expected);
    assert_eq!(submitted(&first), 2);
    assert_eq!(submitted(&second), 0);
    let batches = second.execute(0, ctx)?.try_collect::<Vec<_>>().await?;
    assert_eq!(rows(&batches), expected);
    assert_eq!(submitted(&second), 2);
    assert_eq!(submitted(&first), 2);
    Ok(())
}

#[test]
fn llm_exec_declares_checkpoint_output_and_row_guarantees() -> Result<()> {
    let input = input(&[Some("prompt")])?;
    let input_cols = input.schema().fields().len();
    let ctx = task_context(&LlmProviderRegistry::default());
    let exec = decode(input, "{}", &ctx)?;
    validate_checkpoint_target(&exec, exec.schema().as_ref()).unwrap();
    assert_eq!(exec.passthrough_column_count(), input_cols);
    assert_eq!(
        exec.ai_output_columns(),
        [CheckpointOutputColumn {
            index: input_cols,
            name: "answer".into(),
            is_error_envelope: false,
        }]
    );
    assert!(!exec.preserves_input_order());
    assert_eq!(exec.maintains_input_order(), [false]);
    assert!(exec.preserves_row_cardinality());
    Ok(())
}

type Reply = std::result::Result<LlmResponse, LlmBackendError>;

struct Attempt {
    prompts: Vec<String>,
    cancel: CancellationToken,
    reply: oneshot::Sender<Reply>,
}

impl Attempt {
    fn succeed(self) {
        let outputs = self
            .prompts
            .into_iter()
            .map(|prompt| Some(format!("answer:{prompt}")))
            .collect();
        self.reply
            .send(Ok(LlmResponse {
                outputs,
                usage: None,
            }))
            .unwrap();
    }
}

#[derive(Debug)]
struct ScriptedBackend(mpsc::UnboundedSender<Attempt>);

#[async_trait]
impl LlmBackend for ScriptedBackend {
    async fn generate(&self, request: LlmRequest, cancel: CancellationToken) -> Reply {
        let (reply, outcome) = oneshot::channel();
        assert!(
            self.0
                .send(Attempt {
                    prompts: request.prompts,
                    cancel: cancel.clone(),
                    reply
                })
                .is_ok()
        );
        tokio::select! {
            biased;
            () = cancel.cancelled() => Err(LlmBackendError::Cancelled),
            outcome = outcome => outcome.expect("scripted inference needs a reply"),
        }
    }
}

fn script(
    limit: usize,
) -> Result<(LlmProviderRegistry, mpsc::UnboundedReceiver<Attempt>)> {
    let (send, recv) = mpsc::unbounded_channel();
    let registry = LlmProviderRegistry::default();
    registry.register_with_limit("script", Arc::new(ScriptedBackend(send)), limit)?;
    Ok((registry, recv))
}

async fn next_attempt(attempts: &mut mpsc::UnboundedReceiver<Attempt>) -> Attempt {
    timeout(TEST_TIMEOUT, attempts.recv())
        .await
        .expect("timed out waiting for a backend attempt")
        .unwrap()
}

fn assert_no_attempt(attempts: &mut mpsc::UnboundedReceiver<Attempt>) {
    assert!(matches!(
        attempts.try_recv(),
        Err(mpsc::error::TryRecvError::Empty)
    ));
}

#[tokio::test]
async fn operator_and_provider_limits_preserve_out_of_order_rows() -> Result<()> {
    for (provider_limit, operator_limit) in [(2, 3), (3, 2)] {
        let (registry, mut attempts) = script(provider_limit)?;
        let ctx = task_context(&registry);
        let exec = decode(
            input(&[Some("a"), Some("b"), Some("c"), Some("d")])?,
            &format!(r#"{{"provider":"script","inflight_limit":{operator_limit}}}"#),
            &ctx,
        )?;
        let mut stream = exec.execute(0, ctx)?;
        assert_no_attempt(&mut attempts);
        assert!(poll!(stream.next()).is_pending());
        let first = next_attempt(&mut attempts).await;
        let second = next_attempt(&mut attempts).await;
        tokio::task::yield_now().await;
        assert_no_attempt(&mut attempts);
        assert_eq!(
            registry.available_permits("script"),
            Some(provider_limit - 2)
        );
        assert_eq!(first.prompts, ["a"]);
        assert_eq!(second.prompts, ["b"]);

        second.succeed();
        let batch = stream.next().await.unwrap()?;
        assert_eq!(
            rows(std::slice::from_ref(&batch)),
            [(1, Some("b"), Some("answer:b"))]
        );
        first.succeed();
        let (remaining, ()) = tokio::join!(stream.try_collect::<Vec<_>>(), async {
            next_attempt(&mut attempts).await.succeed();
            next_attempt(&mut attempts).await.succeed();
        });
        let mut batches = vec![batch];
        batches.extend(remaining?);
        assert_eq!(
            rows(&batches),
            [
                (0, Some("a"), Some("answer:a")),
                (1, Some("b"), Some("answer:b")),
                (2, Some("c"), Some("answer:c")),
                (3, Some("d"), Some("answer:d")),
            ]
        );
        assert_eq!(submitted(&exec), 4);
        assert_eq!(registry.available_permits("script"), Some(provider_limit));
        assert_no_attempt(&mut attempts);
    }
    Ok(())
}

#[tokio::test]
async fn completed_jobs_release_provider_permits_under_backpressure() -> Result<()> {
    let (registry, mut attempts) = script(1)?;
    let ctx = task_context(&registry);
    let exec = decode(
        input(&[Some("a"), Some("b"), Some("c")])?,
        r#"{"provider":"script","inflight_limit":1}"#,
        &ctx,
    )?;
    let mut stream = exec.execute(0, Arc::clone(&ctx))?;
    assert!(poll!(stream.next()).is_pending());

    next_attempt(&mut attempts).await.succeed();
    let second = next_attempt(&mut attempts).await;
    let completed = second.cancel.clone();
    second.succeed();
    timeout(TEST_TIMEOUT, completed.cancelled()).await.unwrap();
    assert_eq!(registry.available_permits("script"), Some(1));
    assert_no_attempt(&mut attempts);

    let mut other = exec.execute(0, ctx)?;
    assert!(poll!(other.next()).is_pending());
    let attempt = next_attempt(&mut attempts).await;
    assert_eq!(attempt.prompts, ["a"]);
    drop(other);
    drop(stream);
    Ok(())
}

#[tokio::test]
async fn dropping_stream_immediately_cancels_and_releases_both_limits() -> Result<()> {
    let (registry, mut attempts) = script(1)?;
    let ctx = task_context(&registry);
    let exec = decode(
        input(&[Some("a")])?,
        r#"{"provider":"script","inflight_limit":1}"#,
        &ctx,
    )?;
    let mut stream = exec.execute(0, Arc::clone(&ctx))?;
    assert!(poll!(stream.next()).is_pending());
    let mut pending = next_attempt(&mut attempts).await;
    assert_eq!(registry.available_permits("script"), Some(0));
    assert!(!pending.cancel.is_cancelled());
    drop(stream);
    assert!(pending.cancel.is_cancelled());
    timeout(TEST_TIMEOUT, pending.reply.closed()).await.unwrap();
    assert_eq!(registry.available_permits("script"), Some(1));
    assert_eq!(metric(&exec, "cancelled_requests"), 1);

    let mut stream = exec.execute(0, ctx)?;
    assert!(poll!(stream.next()).is_pending());
    next_attempt(&mut attempts).await.succeed();
    let batches = stream.try_collect::<Vec<_>>().await?;
    assert_eq!(rows(&batches), [(0, Some("a"), Some("answer:a"))]);
    assert_eq!(registry.available_permits("script"), Some(1));
    Ok(())
}

#[tokio::test(start_paused = true)]
async fn retry_backoff_keeps_permits_and_charges_each_prompt_once() -> Result<()> {
    let (registry, mut attempts) = script(1)?;
    let ctx = task_context(&registry);
    let exec = decode(
        input(&[Some("a")])?,
        r#"{"provider":"script","inflight_limit":1,"max_retries":2}"#,
        &ctx,
    )?;
    let mut stream = exec.execute(0, ctx)?;
    assert!(poll!(stream.next()).is_pending());
    let mut attempt = next_attempt(&mut attempts).await;
    for delay_ms in [100, 200] {
        assert_eq!(attempt.prompts, ["a"]);
        let completed = attempt.cancel;
        attempt
            .reply
            .send(Err(LlmBackendError::Transient {
                status: None,
                message: "transient test failure".into(),
            }))
            .unwrap();
        timeout(TEST_TIMEOUT, completed.cancelled()).await.unwrap();
        let backoff_start = Instant::now();
        advance(Duration::from_millis(delay_ms - 1)).await;
        tokio::task::yield_now().await;
        assert_no_attempt(&mut attempts);
        assert_eq!(registry.available_permits("script"), Some(0));
        advance(Duration::from_millis(1)).await;
        attempt = next_attempt(&mut attempts).await;
        assert!(backoff_start.elapsed() <= Duration::from_millis(delay_ms + 1));
    }
    attempt.succeed();
    let batches = stream.try_collect::<Vec<_>>().await?;
    assert_eq!(rows(&batches), [(0, Some("a"), Some("answer:a"))]);
    assert_eq!(submitted(&exec), 1);
    assert_eq!(metric(&exec, "backend_attempts"), 3);
    assert_eq!(metric(&exec, "retries"), 2);
    assert_eq!(registry.available_permits("script"), Some(1));
    Ok(())
}

struct WarningCounter(AtomicUsize);

static WARNINGS: WarningCounter = WarningCounter(AtomicUsize::new(0));

impl log::Log for WarningCounter {
    fn enabled(&self, metadata: &log::Metadata<'_>) -> bool {
        metadata.level() == log::Level::Warn
            && metadata.target() == "datafusion_ai::llm::stream"
    }

    fn log(&self, record: &log::Record<'_>) {
        if self.enabled(record.metadata()) {
            self.0.fetch_add(1, Ordering::Relaxed);
        }
    }

    fn flush(&self) {}
}

#[tokio::test]
async fn terminal_failures_raise_log_or_ignore_as_requested() -> Result<()> {
    log::set_logger(&WARNINGS).unwrap();
    log::set_max_level(log::LevelFilter::Warn);
    for (name, mode) in [
        ("raise", OnError::Raise),
        ("log", OnError::Log),
        ("ignore", OnError::Ignore),
    ] {
        let warnings_before = WARNINGS.0.load(Ordering::Relaxed);
        let (registry, mut attempts) = script(1)?;
        let ctx = task_context(&registry);
        let exec = decode(
            input(&[Some("a"), Some("b"), None])?,
            &format!(r#"{{"provider":"script","batch_size":2,"on_error":"{name}"}}"#),
            &ctx,
        )?;
        assert_eq!(exec.options().on_error, mode);
        let mut stream = exec.execute(0, ctx)?;
        assert!(poll!(stream.next()).is_pending());
        let attempt = next_attempt(&mut attempts).await;
        assert_eq!(attempt.prompts, ["a", "b"]);
        attempt
            .reply
            .send(Err(LlmBackendError::Permanent {
                status: None,
                message: "permanent test failure".into(),
            }))
            .unwrap();

        let result = stream.try_collect::<Vec<_>>().await;
        if mode == OnError::Raise {
            assert!(
                matches!(result, Err(DataFusionError::Execution(message)) if message.contains("permanent test failure"))
            );
        } else {
            assert_eq!(
                rows(&result?),
                [(0, Some("a"), None), (1, Some("b"), None), (2, None, None)]
            );
        }
        assert_eq!(
            WARNINGS.0.load(Ordering::Relaxed) - warnings_before,
            usize::from(mode == OnError::Log)
        );
        assert_eq!(metric(&exec, "failed_rows"), 2);
        assert_eq!(metric(&exec, "backend_attempts"), 1);
        assert_eq!(registry.available_permits("script"), Some(1));
        assert_no_attempt(&mut attempts);
    }
    Ok(())
}

#[tokio::test]
async fn all_null_batches_preserve_rows_without_inference() -> Result<()> {
    let (registry, mut attempts) = script(1)?;
    let ctx = task_context(&registry);
    let exec = decode(
        input(&[None, None, None])?,
        r#"{"provider":"script","batch_size":2}"#,
        &ctx,
    )?;
    let batches = exec.execute(0, ctx)?.try_collect::<Vec<_>>().await?;
    assert_eq!(
        rows(&batches),
        [(0, None, None), (1, None, None), (2, None, None)]
    );
    assert_eq!(submitted(&exec), 0);
    assert_eq!(registry.available_permits("script"), Some(1));
    assert_no_attempt(&mut attempts);
    Ok(())
}
