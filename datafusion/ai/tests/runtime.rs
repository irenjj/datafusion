// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: See the NOTICE file distributed with this work for additional information regarding copyright ownership
// SPDX-FileContributor: Licensed to the Apache Software Foundation (ASF) under one or more contributor license agreements

use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Duration;

use arrow::array::{
    Array, AsArray, FixedSizeListArray, Int32Array, ListBuilder, StringArray,
    StringBuilder, StructArray, UInt64Array,
};
use arrow::datatypes::{DataType, Field, Schema};
use arrow::record_batch::RecordBatch;
use async_trait::async_trait;
use datafusion_ai::llm::{
    CancellationToken, LlmBackend, LlmBackendError, LlmEmbeddingResponse, LlmExec,
    LlmProviderRegistry, LlmRequest, LlmResponse, LlmUsage, MediaPlan, MockLlmBackend,
    OnError, TaskKind,
};
use datafusion_ai::{
    AiErrorType, CheckpointOutputColumn, CheckpointableTarget, MediaRef, MediaRefType,
    MediaResolver, ObjectStoreMediaResolver, ResolvedMedia, validate_checkpoint_target,
};
use datafusion_common::{DataFusionError, Result};
use datafusion_execution::TaskContext;
use datafusion_execution::config::SessionConfig;
use datafusion_execution::runtime_env::RuntimeEnv;
use datafusion_physical_expr::expressions::Column;
use datafusion_physical_plan::ExecutionPlan;
use datafusion_physical_plan::test::TestMemoryExec;
use futures::{StreamExt, TryStreamExt, poll};
use object_store::{ObjectStoreExt, memory::InMemory, path::Path};
use tokio::sync::{mpsc, oneshot};
use tokio::time::{Instant, advance, timeout};

const TEST_TIMEOUT: Duration = Duration::from_secs(5);

fn input(prompts: &[Option<&str>]) -> Result<Arc<dyn ExecutionPlan>> {
    partitioned_input(&[prompts])
}

fn partitioned_input(partitions: &[&[Option<&str>]]) -> Result<Arc<dyn ExecutionPlan>> {
    let schema = Arc::new(Schema::new(vec![
        Field::new("id", DataType::Int32, false),
        Field::new("text", DataType::Utf8, true),
    ]));
    let partitions = partitions
        .iter()
        .map(|prompts| {
            Ok(vec![RecordBatch::try_new(
                Arc::clone(&schema),
                vec![
                    Arc::new(Int32Array::from_iter_values(
                        0..i32::try_from(prompts.len()).unwrap(),
                    )),
                    Arc::new(StringArray::from(prompts.to_vec())),
                ],
            )?])
        })
        .collect::<Result<Vec<_>>>()?;
    Ok(TestMemoryExec::try_new_exec(&partitions, schema, None)?)
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

#[test]
fn worker_decode_preserves_schema_order_and_canonical_bytes() -> Result<()> {
    let ctx = task_context(&LlmProviderRegistry::default());
    let input = input(&[Some("prompt")])?;
    let options = r#"{
        "response_schema":{
            "type":"object",
            "required":["z","a"],
            "properties":{
                "z":{"type":"object","properties":{"y":{"type":"integer"},"x":{"type":"boolean"}}},
                "a":{"type":"string","enum":["second","first"]}
            }
        },
        "extra_body":{"z":0,"a":1}
    }"#;
    let first = decode(Arc::clone(&input), options, &ctx)?;
    let second = decode(Arc::clone(&input), first.options_json(), &ctx)?;
    let third = decode(input, options, &ctx)?;
    let expected = r#"{"extra_body":{"a":1,"z":0},"response_schema":{"properties":{"z":{"properties":{"y":{"type":"integer"},"x":{"type":"boolean"}},"type":"object"},"a":{"enum":["second","first"],"type":"string"}},"required":["z","a"],"type":"object"}}"#;
    for exec in [&first, &second, &third] {
        assert_eq!(exec.options_json(), expected);
        assert_eq!(exec.schema(), first.schema());
        let schema = exec.schema();
        let DataType::Struct(fields) = schema.field(2).data_type() else {
            panic!("expected struct output");
        };
        assert_eq!(fields[0].name(), "z");
        assert_eq!(fields[1].name(), "a");
        let DataType::Struct(nested) = fields[0].data_type() else {
            panic!("expected a nested struct");
        };
        assert_eq!(nested[0].name(), "y");
        assert_eq!(nested[1].name(), "x");
    }
    Ok(())
}

#[test]
fn worker_decode_round_trips_schema_retry_options() -> Result<()> {
    let ctx = task_context(&LlmProviderRegistry::default());
    let input = input(&[Some("prompt")])?;
    for mode in ["strict", "json_object", "prompt_only"] {
        let options = format!(
            r#"{{"system_message":"Original\n message","schema_retries":2,"response_schema":{{"type":"object","properties":{{"z":{{"type":"integer"}},"a":{{"type":"boolean"}}}}}},"schema_mode":"{mode}"}}"#,
        );
        let expected = format!(
            r#"{{"response_schema":{{"properties":{{"z":{{"type":"integer"}},"a":{{"type":"boolean"}}}},"type":"object"}},"schema_mode":"{mode}","schema_retries":2,"system_message":"Original\n message"}}"#,
        );
        let first = decode(Arc::clone(&input), &options, &ctx)?;
        let second = decode(Arc::clone(&input), first.options_json(), &ctx)?;
        assert_eq!(first.options_json(), expected);
        assert_eq!(second.options_json().as_bytes(), expected.as_bytes());
        assert_eq!(first.schema(), second.schema());
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
    assert_eq!(first.task_kind(), TaskKind::Chat);
    assert!(first.media().is_none());
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

#[tokio::test]
async fn object_store_resolver_checks_ranges_limits_and_etags() -> Result<()> {
    let env = RuntimeEnv::default();
    let store = Arc::new(InMemory::new());
    let etag = store
        .put(
            &Path::from("asset"),
            bytes::Bytes::from_static(b"abcdef").into(),
        )
        .await?
        .e_tag;
    assert!(etag.is_some());
    env.register_object_store(&"memory://test".parse().unwrap(), store);
    let resolver = ObjectStoreMediaResolver::new(Arc::clone(&env.object_store_registry));
    let mut media = MediaRef {
        uri: "memory://test/asset".into(),
        byte_offset: None,
        byte_len: None,
        etag,
        mime: Some("image/png".into()),
    };
    assert!(resolver.resolve(&media, None).await.is_err());
    assert!(
        resolver
            .resolve(&media, Some(5))
            .await
            .unwrap_err()
            .to_string()
            .contains("max_bytes")
    );
    let full = resolver.resolve(&media, Some(6)).await?;
    assert_eq!(full.bytes, "abcdef");
    assert_eq!(full.mime, media.mime);
    assert_eq!(full.etag, media.etag);
    media.byte_offset = Some(2);
    assert_eq!(resolver.resolve(&media, Some(4)).await?.bytes, "cdef");
    media.byte_len = Some(2);
    assert_eq!(resolver.resolve(&media, Some(2)).await?.bytes, "cd");
    media.byte_len = Some(9);
    assert!(resolver.resolve(&media, Some(9)).await.is_err());
    media.etag = Some("wrong".into());
    media.byte_len = Some(0);
    assert!(
        resolver
            .resolve(&media, Some(0))
            .await
            .unwrap_err()
            .to_string()
            .contains("ETag")
    );
    media.etag = None;
    media.byte_offset = Some(u64::MAX);
    media.uri = "memory://unregistered/missing".into();
    assert!(resolver.resolve(&media, None).await?.bytes.is_empty());

    media.byte_len = None;
    media.uri = "memory://unregistered/asset?sig=private-marker".into();
    let error = resolver
        .resolve(&media, Some(1))
        .await
        .unwrap_err()
        .to_string();
    assert!(
        !error.contains("://") && !error.contains("private-marker"),
        "{error}"
    );
    Ok(())
}

#[test]
fn vision_worker_decode_requires_a_media_resolver() -> Result<()> {
    let ctx = task_context(&LlmProviderRegistry::default());
    let plan = || MediaPlan::new(Arc::new(Column::new("text", 1)));
    let decode_vision = |ctx| {
        LlmExec::try_new_task_from_task_context(
            input(&[Some("prompt")])?,
            Arc::new(Column::new("text", 1)),
            Some(plan()),
            TaskKind::Vision,
            "model".into(),
            "{}",
            "answer".into(),
            ctx,
        )
    };
    let error = decode_vision(&ctx).unwrap_err();
    assert!(
        matches!(error, DataFusionError::Plan(message) if message.contains("resolver"))
    );
    let env = RuntimeEnv::default();
    let resolver = ObjectStoreMediaResolver::new(Arc::clone(&env.object_store_registry));
    let ctx = TaskContext::default().with_session_config(
        ctx.session_config()
            .clone()
            .with_extension(Arc::new(resolver)),
    );
    let exec = decode_vision(&ctx)?;
    assert_eq!(exec.task_kind(), TaskKind::Vision);
    assert!(exec.media().is_some());
    Ok(())
}

#[tokio::test]
async fn usage_metrics_follow_explicit_and_default_provider_names() -> Result<()> {
    #[derive(Debug)]
    struct ReportingBackend(LlmUsage);

    #[async_trait]
    impl LlmBackend for ReportingBackend {
        async fn generate(
            &self,
            request: LlmRequest,
            cancel: CancellationToken,
        ) -> Reply {
            let mut response = MockLlmBackend.generate(request, cancel).await?;
            response.usage = Some(self.0);
            Ok(response)
        }

        async fn embed(
            &self,
            request: LlmRequest,
            cancel: CancellationToken,
        ) -> std::result::Result<LlmEmbeddingResponse, LlmBackendError> {
            let mut response = MockLlmBackend.embed(request, cancel).await?;
            response.usage = Some(self.0);
            Ok(response)
        }
    }

    let registry = LlmProviderRegistry::empty_with_default("gpt-gateway");
    for (name, input_tokens, output_tokens) in
        [("qwen-openai", 11, 7), ("gpt-gateway", 19, 0)]
    {
        registry.register(
            name,
            Arc::new(ReportingBackend(LlmUsage {
                input_tokens,
                output_tokens,
            })),
        )?;
    }
    let ctx = task_context(&registry);
    let chat = Arc::new(decode(
        input(&[Some("prompt")])?,
        r#"{"provider":"qwen-openai"}"#,
        &ctx,
    )?);
    let embedding = LlmExec::try_new_task_from_task_context(
        Arc::clone(&chat) as Arc<dyn ExecutionPlan>,
        Arc::new(Column::new("text", 1)),
        None,
        TaskKind::Embed,
        "embedding-model".into(),
        r#"{"dimension":2}"#,
        "embedding".into(),
        &ctx,
    )?;
    embedding.execute(0, ctx)?.try_collect::<Vec<_>>().await?;
    for (name, chat_usage, embed_usage) in [
        ("input_tokens", 11, 19),
        ("output_tokens", 7, 0),
        ("llm_requests", 1, 1),
    ] {
        let qwen = format!("{name}::qwen-openai");
        let gpt = format!("{name}::gpt-gateway");
        let usage = |exec: &LlmExec, name: &str| {
            exec.metrics()
                .unwrap()
                .sum_by_name(name)
                .unwrap()
                .as_usize()
        };
        assert_eq!(usage(&chat, &qwen), chat_usage);
        assert_eq!(usage(&embedding, &gpt), embed_usage);
        assert_eq!(
            metric(&chat, name) + metric(&embedding, name),
            chat_usage + embed_usage
        );
        assert_eq!(metric(&chat, &gpt), 0);
        assert_eq!(metric(&embedding, &qwen), 0);
    }
    Ok(())
}

#[test]
fn provider_names_reserve_metric_delimiter() -> Result<()> {
    let registry = LlmProviderRegistry::default();
    registry.register("test.mock_ai-2", Arc::new(MockLlmBackend))?;
    for name in ["qwen::openai", "qwen/openai", "qwen openai", "qwen💡"] {
        for error in [
            registry
                .register(name, Arc::new(MockLlmBackend))
                .unwrap_err(),
            registry
                .register_with_limit(name, Arc::new(MockLlmBackend), 1)
                .unwrap_err(),
        ] {
            assert!(
                matches!(error, DataFusionError::Plan(message) if message.contains("provider name"))
            );
        }
    }
    Ok(())
}

#[tokio::test]
async fn embedding_envelope_is_checkpointable_and_round_trips_fixed_size_lists()
-> Result<()> {
    use arrow::ipc::{reader::StreamReader, writer::StreamWriter};
    use std::io::Cursor;

    #[derive(Debug)]
    struct EmbeddingBackend;

    #[async_trait]
    impl LlmBackend for EmbeddingBackend {
        async fn generate(&self, _: LlmRequest, _: CancellationToken) -> Reply {
            unreachable!("embedding execution uses embed")
        }

        async fn embed(
            &self,
            request: LlmRequest,
            cancel: CancellationToken,
        ) -> std::result::Result<LlmEmbeddingResponse, LlmBackendError> {
            assert_eq!(request.prompts.len(), 2);
            let mut response = MockLlmBackend.embed(request, cancel).await?;
            response.usage = Some(LlmUsage {
                input_tokens: 19,
                output_tokens: 0,
            });
            Ok(response)
        }
    }

    let registry = LlmProviderRegistry::default();
    registry.register("embedding-test", Arc::new(EmbeddingBackend))?;
    let ctx = task_context(&registry);
    let exec = LlmExec::try_new_task_from_task_context(
        partitioned_input(&[
            &[Some("prompt"), Some("another")],
            &[Some("third"), Some("fourth")],
        ])?,
        Arc::new(Column::new("text", 1)),
        None,
        TaskKind::Embed,
        "model".into(),
        r#"{"provider":"embedding-test","dimension":2,"batch_size":2,"on_error":"envelope"}"#,
        "answer".into(),
        &ctx,
    )?;
    assert_eq!(exec.task_kind(), TaskKind::Embed);
    assert!(exec.media().is_none());
    let schema = exec.schema();
    validate_checkpoint_target(&exec, &schema)?;
    assert!(exec.ai_output_columns()[0].is_error_envelope);
    schema.field(2).try_extension_type::<AiErrorType>()?;
    let batch = exec
        .execute(0, Arc::clone(&ctx))?
        .try_collect::<Vec<_>>()
        .await?
        .remove(0);
    assert_eq!(submitted(&exec), 2);
    assert_eq!(metric(&exec, "input_tokens"), 19);
    assert_eq!(metric(&exec, "output_tokens"), 0);
    assert_eq!(metric(&exec, "llm_requests"), 1);
    assert_eq!(metric(&exec, "llm_retries"), 0);
    exec.execute(1, ctx)?.try_collect::<Vec<_>>().await?;
    let metrics = exec.metrics().unwrap();
    for partition in 0..2 {
        let partition = metrics.for_partition(partition);
        assert_eq!(
            partition.sum_by_name("input_tokens").unwrap().as_usize(),
            19
        );
        assert_eq!(partition.sum_by_name("llm_requests").unwrap().as_usize(), 1);
    }
    assert_eq!(metric(&exec, "input_tokens"), 38);
    assert_eq!(metric(&exec, "llm_requests"), 2);
    let envelope = batch
        .column(2)
        .as_any()
        .downcast_ref::<StructArray>()
        .unwrap();
    let values = envelope
        .column(0)
        .as_any()
        .downcast_ref::<FixedSizeListArray>()
        .unwrap();
    assert_eq!(values.value_length(), 2);
    assert!(!values.is_null(0));
    assert!(envelope.columns()[1..].iter().all(|field| field.is_null(0)));
    let mut writer = StreamWriter::try_new(Vec::new(), &schema)?;
    writer.write(&batch)?;
    let read_back = StreamReader::try_new(Cursor::new(writer.into_inner()?), None)?;
    assert_eq!(
        read_back.collect::<std::result::Result<Vec<_>, _>>()?,
        [batch]
    );
    Ok(())
}

#[tokio::test]
async fn vision_envelopes_keep_struct_references_and_failures_aligned() -> Result<()> {
    #[derive(Debug, Default)]
    struct Resolver(AtomicUsize);

    #[async_trait]
    impl MediaResolver for Resolver {
        async fn resolve(
            &self,
            media: &MediaRef,
            max_bytes: Option<u64>,
        ) -> Result<ResolvedMedia> {
            self.0.fetch_add(1, Ordering::Relaxed);
            assert_eq!(max_bytes, Some(8 << 20));
            match media.uri.as_str() {
                "memory://test/first" => {
                    assert_eq!((media.byte_offset, media.byte_len), (Some(1), Some(3)));
                    assert_eq!(media.etag.as_deref(), Some("opaque"));
                    Ok(ResolvedMedia {
                        bytes: bytes::Bytes::from_static(b"img"),
                        mime: None,
                        etag: media.etag.clone(),
                    })
                }
                "memory://test/missing" => datafusion_common::exec_err!("object was not found"),
                "memory://test/busy" => Err(std::io::Error::new(
                    std::io::ErrorKind::TimedOut,
                    "timed out\nwhile reading https://storage.invalid/signed?sig=private-marker",
                ).into()),
                uri => panic!("unexpected media read for {uri}"),
            }
        }
    }

    let media_field = MediaRefType::field("media", false)?;
    let DataType::Struct(fields) = media_field.data_type() else {
        unreachable!()
    };
    let references = StructArray::try_new(
        fields.clone(),
        vec![
            Arc::new(StringArray::from(vec![
                "memory://test/first",
                "memory://test/missing",
                "memory://test/busy",
                "memory://test/unused",
            ])),
            Arc::new(UInt64Array::from(vec![Some(1), None, None, None])),
            Arc::new(UInt64Array::from(vec![Some(3), None, None, None])),
            Arc::new(StringArray::from(vec![Some("opaque"), None, None, None])),
            Arc::new(StringArray::from(vec![Some("image/png"), None, None, None])),
        ],
        None,
    )?;
    let schema = Arc::new(Schema::new(vec![
        Field::new("text", DataType::Utf8, true),
        media_field,
    ]));
    let batch = RecordBatch::try_new(
        schema.clone(),
        vec![
            Arc::new(StringArray::from(vec![
                Some("a"),
                Some("b"),
                Some("c"),
                None,
            ])),
            Arc::new(references),
        ],
    )?;
    let (registry, mut attempts) = script(1)?;
    let resolver = Arc::new(Resolver::default());
    let ctx = Arc::new(
        TaskContext::default().with_session_config(
            SessionConfig::new()
                .with_extension(Arc::new(registry))
                .with_extension(Arc::new(resolver.clone() as Arc<dyn MediaResolver>)),
        ),
    );
    let input = TestMemoryExec::try_new_exec(&[vec![batch]], schema, None)?;
    let exec = LlmExec::try_new_task_from_task_context(
        input,
        Arc::new(Column::new("text", 0)),
        Some(MediaPlan::new(Arc::new(Column::new("media", 1)))),
        TaskKind::Vision,
        "model".into(),
        r#"{"provider":"script","batch_size":3,"on_error":"envelope"}"#,
        "answer".into(),
        &ctx,
    )?;
    let output = exec.execute(0, ctx)?.try_collect::<Vec<_>>();
    let (output, ()) = tokio::join!(output, async {
        let attempt = next_attempt(&mut attempts).await;
        assert_eq!(attempt.prompts, ["a"]);
        assert_eq!(
            attempt.media[0].as_ref().unwrap()[0].mime.as_deref(),
            Some("image/png")
        );
        attempt.succeed();
    });
    let output = output?;
    let envelope = output
        .iter()
        .find(|batch| batch.num_rows() == 3)
        .unwrap()
        .column(2)
        .as_struct();
    assert_eq!(
        envelope
            .column(0)
            .as_string::<i32>()
            .iter()
            .collect::<Vec<_>>(),
        [Some("answer:a"), None, None]
    );
    assert_eq!(
        envelope
            .column(1)
            .as_string::<i32>()
            .iter()
            .collect::<Vec<_>>(),
        [None, Some("PERMANENT_HTTP"), Some("TRANSIENT_HTTP")]
    );
    assert_eq!(
        envelope.column(3).as_boolean().iter().collect::<Vec<_>>(),
        [None, Some(false), Some(true)]
    );
    for message in envelope.column(2).as_string::<i32>().iter().flatten() {
        assert!(message.starts_with("media: "));
        assert!(
            !message.contains(['\n', '\r'])
                && !message.contains("://")
                && !message.contains("private-marker")
        );
    }
    assert_eq!(resolver.0.load(Ordering::Relaxed), 3);
    assert_eq!(submitted(&exec), 1);
    assert_eq!(metric(&exec, "failed_rows"), 2);
    Ok(())
}

#[tokio::test]
async fn vision_resize_isolates_mixed_formats_and_honors_fallback() -> Result<()> {
    static PNG: &[u8] = include_bytes!("fixtures/vision.png");
    static JPEG: &[u8] = include_bytes!("fixtures/vision.jpeg");

    #[derive(Debug)]
    struct Resolver;

    #[async_trait]
    impl MediaResolver for Resolver {
        async fn resolve(
            &self,
            media: &MediaRef,
            _: Option<u64>,
        ) -> Result<ResolvedMedia> {
            let (bytes, mime) = match media.uri.as_str() {
                "memory://test/png" => (PNG, "application/octet-stream"),
                "memory://test/jpeg" => (JPEG, "image/pjpeg"),
                "memory://test/broken" => (&PNG[..32], "image/png"),
                uri => panic!("unexpected input media {uri}"),
            };
            Ok(ResolvedMedia {
                bytes: bytes::Bytes::from_static(bytes),
                mime: Some(mime.into()),
                etag: None,
            })
        }
    }

    for (options, strict) in [
        (
            r#"{"provider":"script","batch_size":3,"on_error":"envelope","image_max_edge":33}"#,
            false,
        ),
        (
            r#"{"provider":"script","batch_size":3,"on_error":"envelope","image_max_edge":16,"image_resize_fallback":"error"}"#,
            true,
        ),
    ] {
        let batch = RecordBatch::try_from_iter([
            (
                "text",
                Arc::new(StringArray::from(vec!["png", "jpeg", "broken"]))
                    as arrow::array::ArrayRef,
            ),
            (
                "media",
                Arc::new(StringArray::from(vec![
                    "memory://test/png",
                    "memory://test/jpeg",
                    "memory://test/broken",
                ])),
            ),
        ])?;
        let schema = batch.schema();
        let input = TestMemoryExec::try_new_exec(&[vec![batch]], schema, None)?;
        let (registry, mut attempts) = script(1)?;
        let ctx = Arc::new(
            TaskContext::default().with_session_config(
                SessionConfig::new()
                    .with_extension(Arc::new(registry))
                    .with_extension(Arc::new(
                        Arc::new(Resolver) as Arc<dyn MediaResolver>
                    )),
            ),
        );
        let exec = LlmExec::try_new_task_from_task_context(
            input,
            Arc::new(Column::new("text", 0)),
            Some(MediaPlan::new(Arc::new(Column::new("media", 1)))),
            TaskKind::Vision,
            "vision".into(),
            options,
            "answer".into(),
            &ctx,
        )?;
        let collect = exec.execute(0, ctx)?.try_collect::<Vec<_>>();
        let (batches, ()) = tokio::join!(collect, async {
            let attempt = next_attempt(&mut attempts).await;
            let png = &attempt.media[0].as_ref().unwrap()[0];
            if strict {
                assert_eq!(attempt.prompts, ["png"]);
                assert_eq!(png.mime.as_deref(), Some("image/png"));
                assert_eq!(&png.bytes[16..24], &[0, 0, 0, 16, 0, 0, 0, 8]);
            } else {
                assert_eq!(attempt.prompts, ["png", "jpeg"]);
                assert_eq!(png.bytes.as_ptr(), PNG.as_ptr());
                assert_eq!(png.mime.as_deref(), Some("application/octet-stream"));
                let jpeg = &attempt.media[1].as_ref().unwrap()[0];
                assert_eq!(jpeg.bytes.as_ptr(), JPEG.as_ptr());
                assert_eq!(jpeg.mime.as_deref(), Some("image/pjpeg"));
            }
            attempt.succeed();
        });
        let batches = batches?;
        let envelope = batches
            .iter()
            .find(|batch| batch.num_rows() == 3)
            .unwrap()
            .column(2)
            .as_struct();
        assert_eq!(
            envelope
                .column(0)
                .as_string::<i32>()
                .iter()
                .collect::<Vec<_>>(),
            [Some("answer:png"), (!strict).then_some("answer:jpeg"), None,]
        );
        assert_eq!(
            envelope
                .column(1)
                .as_string::<i32>()
                .iter()
                .collect::<Vec<_>>(),
            [
                None,
                strict.then_some("PERMANENT_HTTP"),
                Some("PERMANENT_HTTP"),
            ]
        );
        assert_eq!(
            envelope.column(3).as_boolean().iter().collect::<Vec<_>>(),
            [None, strict.then_some(false), Some(false),]
        );
        let messages = envelope.column(2).as_string::<i32>();
        assert!(messages.value(2).contains("PNG IHDR"));
        if strict {
            assert!(messages.value(1).contains("unsupported for JPEG"));
        }
        for message in messages.iter().flatten() {
            assert!(message.starts_with("media: "));
            assert!(!message.contains(['\n', '\r']) && !message.contains("://"));
        }
        assert_eq!(submitted(&exec), if strict { 1 } else { 2 });
        assert_no_attempt(&mut attempts);
    }
    Ok(())
}

#[tokio::test]
async fn vision_list_errors_are_row_local_and_byte_budget_stops_reads() -> Result<()> {
    #[derive(Debug, Default)]
    struct Resolver(Mutex<Vec<(String, Option<u64>)>>);

    #[async_trait]
    impl MediaResolver for Resolver {
        async fn resolve(
            &self,
            media: &MediaRef,
            max_bytes: Option<u64>,
        ) -> Result<ResolvedMedia> {
            self.0.lock().unwrap().push((media.uri.clone(), max_bytes));
            let payload: &[u8] = match media.uri.as_str() {
                "memory://test/a" | "memory://test/b" => b"12",
                "memory://test/big-first" | "memory://test/big-second" => b"123",
                "memory://test/per-image-overflow" => b"1234",
                uri => panic!("unexpected media read for {uri}"),
            };
            // Exercise the executor's budget checks even if a resolver returns too much.
            Ok(ResolvedMedia {
                bytes: bytes::Bytes::from_static(payload),
                mime: Some("image/png".into()),
                etag: None,
            })
        }
    }

    let mut media = ListBuilder::new(StringBuilder::new());
    let unread = Some("memory://test/unread");
    for images in [
        &[Some("memory://test/a"), Some("memory://test/b")][..],
        &[],
        &[unread, None],
        &[Some(" \t")],
        &[Some("invalid//uri")],
        &[unread; 6],
        &[
            Some("memory://test/big-first"),
            Some("memory://test/big-second"),
            unread,
            unread,
            unread,
        ],
        &[Some("memory://test/per-image-overflow")],
        &[Some("memory://test/a")],
    ] {
        media.append_value(images.iter().copied());
    }
    let media = media.finish();
    let schema = Arc::new(Schema::new(vec![
        Field::new("text", DataType::Utf8, false),
        Field::new("media", media.data_type().clone(), false),
    ]));
    let batch = RecordBatch::try_new(
        Arc::clone(&schema),
        vec![
            Arc::new(StringArray::from(vec![
                "first",
                "empty",
                "null_item",
                "blank_uri",
                "invalid_uri",
                "item_budget",
                "row_budget",
                "one_image_budget",
                "last",
            ])),
            Arc::new(media),
        ],
    )?;
    let (registry, mut attempts) = script(1)?;
    let resolver = Arc::new(Resolver::default());
    let exec = LlmExec::try_new_task_with_registry(
        TestMemoryExec::try_new_exec(&[vec![batch]], schema, None)?,
        Arc::new(Column::new("text", 0)),
        Some(
            MediaPlan::new(Arc::new(Column::new("media", 1)))
                .with_resolver(resolver.clone()),
        ),
        TaskKind::Vision,
        "model".into(),
        r#"{"provider":"script","batch_size":9,"on_error":"envelope","max_media_items":5,"max_media_bytes":3,"max_row_media_bytes":4}"#,
        "answer".into(),
        &registry,
    )?;
    let output = exec
        .execute(0, task_context(&registry))?
        .try_collect::<Vec<_>>();
    let (output, ()) = tokio::join!(output, async {
        let attempt = next_attempt(&mut attempts).await;
        assert_eq!(attempt.prompts, ["first", "last"]);
        attempt.succeed();
    });
    let output = output?;
    assert_eq!(output.len(), 1);
    let envelope = output[0].column(2).as_struct();
    let values = envelope.column(0).as_string::<i32>();
    let codes = envelope.column(1).as_string::<i32>();
    let messages = envelope.column(2).as_string::<i32>();
    let retryable = envelope.column(3).as_boolean();
    for (row, detail) in [
        "empty",
        "non-NULL media reference",
        "blank",
        "invalid media URI",
        "max_media_items",
        "max_row_media_bytes",
        "max_bytes",
    ]
    .into_iter()
    .enumerate()
    {
        let row = row + 1;
        assert!(values.is_null(row));
        assert_eq!(codes.value(row), "PERMANENT_HTTP");
        assert!(
            messages.value(row).contains(detail),
            "{}",
            messages.value(row)
        );
        assert!(retryable.is_valid(row));
        assert!(!retryable.value(row));
    }
    for (row, value) in [(0, "answer:first"), (8, "answer:last")] {
        assert_eq!(values.value(row), value);
        assert!(
            envelope.columns()[1..]
                .iter()
                .all(|field| field.is_null(row))
        );
    }
    let reads = resolver.0.lock().unwrap();
    assert_eq!(
        reads
            .iter()
            .map(|(uri, cap)| (uri.as_str(), *cap))
            .collect::<Vec<_>>(),
        [
            ("memory://test/a", Some(3)),
            ("memory://test/b", Some(2)),
            ("memory://test/big-first", Some(3)),
            ("memory://test/big-second", Some(1)),
            ("memory://test/per-image-overflow", Some(3)),
            ("memory://test/a", Some(3)),
        ]
    );
    assert_eq!(submitted(&exec), 2);
    assert_eq!(metric(&exec, "failed_rows"), 7);
    assert_eq!(metric(&exec, "llm_retries"), 0);
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
    media: Vec<Option<Vec<ResolvedMedia>>>,
    cancel: CancellationToken,
    reply: oneshot::Sender<Reply>,
}

impl Attempt {
    fn reply_with(self, outputs: &[Option<&str>], usage: Option<LlmUsage>) {
        self.reply
            .send(Ok(LlmResponse {
                outputs: outputs
                    .iter()
                    .map(|output| output.map(str::to_owned))
                    .collect(),
                usage,
            }))
            .unwrap();
    }

    fn succeed(self) {
        self.succeed_with_usage(None);
    }

    fn succeed_with_usage(self, usage: Option<LlmUsage>) {
        let outputs = self
            .prompts
            .into_iter()
            .map(|prompt| Some(format!("answer:{prompt}")))
            .collect();
        self.reply.send(Ok(LlmResponse { outputs, usage })).unwrap();
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
                    media: request.media,
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

#[tokio::test]
async fn structured_batch_isolates_a_malformed_middle_row() -> Result<()> {
    let (registry, mut attempts) = script(1)?;
    let ctx = task_context(&registry);
    let options = r#"{
        "provider":"script","batch_size":3,"on_error":"envelope",
        "response_schema":{"type":"object","properties":{
            "id":{"type":"integer"},
            "items":{"type":"array","items":{"type":["object","null"],"properties":{
                "passed":{"type":"boolean"},
                "score":{"type":"number"}
            },"required":["passed"]}}
        },"required":["id"]}
    }"#;
    let exec = decode(input(&[Some("a"), Some("b"), Some("c")])?, options, &ctx)?;
    let schema = exec.schema();
    validate_checkpoint_target(&exec, &schema)?;
    schema.field(2).try_extension_type::<AiErrorType>()?;
    let collect = exec.execute(0, ctx)?.try_collect::<Vec<_>>();
    let (result, ()) = tokio::join!(collect, async {
        let attempt = next_attempt(&mut attempts).await;
        assert_eq!(attempt.prompts, ["a", "b", "c"]);
        attempt
            .reply
            .send(Ok(LlmResponse {
                outputs: [
                    Some(r#"{"items":[{"passed":true,"score":1.25},null],"id":1}"#),
                    Some(r#"{"id":2,"items":[{"passed":false}"#),
                    Some(r#"{"id":3}"#),
                ]
                .map(|output| output.map(str::to_owned))
                .to_vec(),
                usage: Some(LlmUsage {
                    input_tokens: 10,
                    output_tokens: 20,
                }),
            }))
            .unwrap();
    });
    let batches = result?;
    assert_eq!(batches.len(), 1);
    let envelope = batches[0].column(2).as_struct();
    let value = envelope.column(0).as_struct();
    assert_eq!(
        value
            .column(0)
            .as_primitive::<arrow::datatypes::Int64Type>()
            .iter()
            .collect::<Vec<_>>(),
        [Some(1), None, Some(3)]
    );
    assert!(value.is_null(1));
    let items = value.column(1).as_list::<i32>();
    assert_eq!(items.null_count(), 2);
    let first_items = items.value(0);
    let first_items = first_items.as_struct();
    assert!(first_items.column(0).as_boolean().value(0));
    assert_eq!(
        first_items
            .column(1)
            .as_primitive::<arrow::datatypes::Float64Type>()
            .value(0),
        1.25
    );
    assert!(first_items.is_null(1));
    assert_eq!(
        envelope
            .column(1)
            .as_string::<i32>()
            .iter()
            .collect::<Vec<_>>(),
        [None, Some("SCHEMA_MISMATCH"), None]
    );
    assert_eq!(
        envelope.column(3).as_boolean().iter().collect::<Vec<_>>(),
        [None, Some(false), None]
    );
    let messages = envelope.column(2).as_string::<i32>();
    assert!(messages.value(1).contains("JSON"));
    assert!(messages.is_null(0) && messages.is_null(2));
    assert_eq!(metric(&exec, "failed_rows"), 1);
    assert_eq!(metric(&exec, "backend_attempts"), 1);
    assert_eq!(metric(&exec, "input_tokens"), 10);
    assert_eq!(metric(&exec, "output_tokens"), 20);
    assert_no_attempt(&mut attempts);
    Ok(())
}

#[tokio::test]
async fn schema_retries_isolate_rows_and_count_all_provider_usage() -> Result<()> {
    let (registry, mut attempts) = script(1)?;
    let ctx = task_context(&registry);
    let exec = decode(
        input(&[
            Some("first"),
            Some("recover"),
            Some("third"),
            Some("exhaust"),
        ])?,
        r#"{"provider":"script","batch_size":4,"on_error":"envelope",
            "schema_mode":"json_object","schema_retries":2,
            "response_schema":{"type":"object","properties":{
                "count":{"type":"integer"}
            },"required":["count"]}}"#,
        &ctx,
    )?;
    let collect = exec.execute(0, ctx)?.try_collect::<Vec<_>>();
    let (batches, ()) = tokio::join!(collect, async {
        let initial = next_attempt(&mut attempts).await;
        assert_eq!(initial.prompts, ["first", "recover", "third", "exhaust"]);
        initial.reply_with(
            &[
                Some(r#"{"count":1}"#),
                Some(r#"{"count":"2"}"#),
                Some(r#"{"count":3}"#),
                Some("{"),
            ],
            Some(LlmUsage {
                input_tokens: 10,
                output_tokens: 20,
            }),
        );
        let mut recovered = 0;
        let mut exhausted = 0;
        for _ in 0..3 {
            let retry = next_attempt(&mut attempts).await;
            assert_eq!(retry.prompts.len(), 1);
            match retry.prompts[0].as_str() {
                "recover" => {
                    recovered += 1;
                    retry.reply_with(
                        &[Some("```json\n{\"count\":2}\n```")],
                        Some(LlmUsage {
                            input_tokens: 2,
                            output_tokens: 3,
                        }),
                    );
                }
                "exhaust" => {
                    exhausted += 1;
                    let output = if exhausted == 1 {
                        "{}"
                    } else {
                        r#"{"count":false}"#
                    };
                    retry.reply_with(
                        &[Some(output)],
                        Some(LlmUsage {
                            input_tokens: 1,
                            output_tokens: 2,
                        }),
                    );
                }
                prompt => panic!("unexpected retry for {prompt}"),
            }
        }
        assert_eq!((recovered, exhausted), (1, 2));
    });
    let batches = batches?;
    assert_eq!(batches.len(), 1);
    let batch = &batches[0];
    assert_eq!(
        batch
            .column(0)
            .as_primitive::<arrow::datatypes::Int32Type>()
            .values(),
        &[0, 1, 2, 3]
    );
    batch
        .schema()
        .field(2)
        .try_extension_type::<AiErrorType>()?;
    let envelope = batch.column(2).as_struct();
    let values = envelope.column(0).as_struct();
    assert!(values.is_null(3));
    assert_eq!(
        values
            .column(0)
            .as_primitive::<arrow::datatypes::Int64Type>()
            .iter()
            .collect::<Vec<_>>(),
        [Some(1), Some(2), Some(3), None]
    );
    assert_eq!(
        envelope
            .column(1)
            .as_string::<i32>()
            .iter()
            .collect::<Vec<_>>(),
        [None, None, None, Some("SCHEMA_MISMATCH")]
    );
    assert_eq!(
        envelope.column(3).as_boolean().iter().collect::<Vec<_>>(),
        [None, None, None, Some(false)]
    );
    for (name, expected) in [
        ("llm_requests", 4),
        ("llm_retries", 3),
        ("submitted_prompts", 4),
        ("failed_rows", 1),
        ("input_tokens", 14),
        ("output_tokens", 27),
        ("llm_requests::script", 4),
        ("input_tokens::script", 14),
        ("output_tokens::script", 27),
    ] {
        assert_eq!(metric(&exec, name), expected, "{name}");
    }
    assert_no_attempt(&mut attempts);
    Ok(())
}

#[tokio::test]
async fn schema_retry_observes_stream_and_backend_cancellation() -> Result<()> {
    for drop_stream in [true, false] {
        let (registry, mut attempts) = script(1)?;
        let ctx = task_context(&registry);
        let exec = decode(
            input(&[Some("a")])?,
            r#"{"provider":"script","on_error":"envelope","schema_retries":3,
                "response_schema":{"type":"object","properties":{
                    "count":{"type":"integer"}
                },"required":["count"]}}"#,
            &ctx,
        )?;
        let mut stream = exec.execute(0, ctx)?;
        assert!(poll!(stream.next()).is_pending());
        next_attempt(&mut attempts)
            .await
            .reply_with(&[Some("{}")], None);
        let mut retry = next_attempt(&mut attempts).await;
        assert_eq!(retry.prompts, ["a"]);
        assert_eq!(registry.available_permits("script"), Some(0));
        assert!(!retry.cancel.is_cancelled());
        if drop_stream {
            drop(stream);
            assert!(retry.cancel.is_cancelled());
            timeout(TEST_TIMEOUT, retry.reply.closed()).await.unwrap();
        } else {
            retry.reply.send(Err(LlmBackendError::Cancelled)).unwrap();
            let batch = timeout(TEST_TIMEOUT, stream.next())
                .await
                .unwrap()
                .unwrap()?;
            let envelope = batch.column(2).as_struct();
            assert_eq!(envelope.column(1).as_string::<i32>().value(0), "CANCELLED");
        }
        assert_eq!(registry.available_permits("script"), Some(1));
        assert_eq!(metric(&exec, "cancelled_requests"), 1);
        assert_eq!(metric(&exec, "llm_requests"), 2);
        assert_eq!(metric(&exec, "llm_retries"), 1);
        assert_no_attempt(&mut attempts);
    }
    Ok(())
}

#[tokio::test]
async fn structured_types_are_checked_without_coercion() -> Result<()> {
    let outputs = [
        (Some(r#"{"id":"123"}"#), "64-bit integer"),
        (Some(r#"{"id":1.5}"#), "64-bit integer"),
        (Some(r#"{"id":9223372036854775808}"#), "64-bit integer"),
        (
            Some(r#"{"id":{"$serde_json::private::Number":"42"}}"#),
            "64-bit integer",
        ),
        (Some(r#"{"id":null}"#), "null"),
        (Some(r#"{"state":"ok"}"#), "missing"),
        (Some(r#"{"id":1,"unknown":123}"#), "undeclared"),
        (Some(r#"{"id":1,"state":true}"#), "string"),
        (Some(r#"{"id":1,"state":"secret-canary"}"#), "enum"),
        (Some(r#"{"id":1,"state":"ok","state":"ok"}"#), "duplicate"),
        (None, "missing"),
    ];
    let (registry, mut attempts) = script(1)?;
    let ctx = task_context(&registry);
    let exec = decode(
        input(&vec![Some("p"); outputs.len()])?,
        r#"{
            "provider":"script", "batch_size":16, "on_error":"envelope",
            "response_schema":{"type":"object","properties":{
                "id":{"type":"integer"},
                "state":{"type":"string","enum":["ok","fail"]}
            },"required":["id"]}
        }"#,
        &ctx,
    )?;
    let collect = exec.execute(0, ctx)?.try_collect::<Vec<_>>();
    let (batches, ()) = tokio::join!(collect, async {
        let attempt = next_attempt(&mut attempts).await;
        assert_eq!(attempt.prompts.len(), outputs.len());
        attempt
            .reply
            .send(Ok(LlmResponse {
                outputs: outputs
                    .iter()
                    .map(|(output, _)| output.map(str::to_owned))
                    .collect(),
                usage: None,
            }))
            .unwrap();
    });
    let batches = batches?;
    let envelope = batches[0].column(2).as_struct();
    assert_eq!(envelope.column(0).null_count(), outputs.len());
    for (row, (_, expected)) in outputs.iter().enumerate() {
        assert_eq!(
            envelope.column(1).as_string::<i32>().value(row),
            "SCHEMA_MISMATCH"
        );
        let message = envelope.column(2).as_string::<i32>().value(row);
        assert!(message.contains(expected), "{message}");
        assert!(!message.contains("secret-canary"), "{message}");
        assert!(!envelope.column(3).as_boolean().value(row));
    }
    assert_eq!(metric(&exec, "failed_rows"), outputs.len());
    assert_eq!(metric(&exec, "backend_attempts"), 1);
    Ok(())
}

#[tokio::test]
async fn structured_failures_support_ignore_log_and_raise() -> Result<()> {
    for on_error in ["ignore", "log", "raise"] {
        let (registry, mut attempts) = script(1)?;
        let ctx = task_context(&registry);
        let exec = decode(
            input(&[Some("a"), Some("b"), None])?,
            &format!(
                r#"{{"provider":"script","batch_size":2,"on_error":"{on_error}",
                "response_schema":{{"type":"object","properties":{{"qty":{{"type":"integer"}}}},"required":["qty"]}}}}"#,
            ),
            &ctx,
        )?;
        let collect = exec.execute(0, ctx)?.try_collect::<Vec<_>>();
        let (result, ()) = tokio::join!(collect, async {
            next_attempt(&mut attempts)
                .await
                .reply
                .send(Ok(LlmResponse {
                    outputs: vec![Some(r#"{"qty":10}"#.into()), Some("[]".into())],
                    usage: None,
                }))
                .unwrap();
        });
        if on_error != "raise" {
            let batches = result?;
            let batch = batches.iter().find(|batch| batch.num_rows() == 2).unwrap();
            let value = batch.column(2).as_struct();
            assert!(value.is_valid(0));
            assert!(value.is_null(1));
            assert!(
                batches
                    .iter()
                    .find(|batch| batch.num_rows() == 1)
                    .unwrap()
                    .column(2)
                    .is_null(0)
            );
        } else {
            assert!(
                matches!(result, Err(DataFusionError::Execution(message)) if message.contains("expected an object"))
            );
        }
        assert_eq!(metric(&exec, "failed_rows"), 1);
    }
    Ok(())
}

#[tokio::test]
async fn structured_vision_failures_use_slice_row_indices() -> Result<()> {
    let env = RuntimeEnv::default();
    let store = Arc::new(InMemory::new());
    let url = url::Url::parse("memory://structured-vision").unwrap();
    store
        .put(
            &Path::from("present"),
            bytes::Bytes::from_static(b"image").into(),
        )
        .await?;
    store
        .put(
            &Path::from("retry"),
            bytes::Bytes::from_static(b"retry image").into(),
        )
        .await?;
    env.register_object_store(&url, store);
    let resolver = ObjectStoreMediaResolver::new(Arc::clone(&env.object_store_registry));

    let (registry, mut attempts) = script(1)?;
    let ctx = task_context(&registry);
    let present = "memory://structured-vision/present";
    let mut media = ListBuilder::new(StringBuilder::new());
    media.append_value([Some("not a URI"), Some(present)]);
    media.append_value([Some("memory://structured-vision/retry"), Some(present)]);
    for _ in 0..2 {
        media.append_value([Some(present); 2]);
    }
    media.append_null();
    let batch = RecordBatch::try_from_iter([
        (
            "text",
            Arc::new(StringArray::from(vec![
                "bad media",
                "repairable JSON",
                "bad JSON",
                "valid",
                "skipped",
            ])) as arrow::array::ArrayRef,
        ),
        ("media", Arc::new(media.finish())),
    ])?;
    let schema = batch.schema();
    let input = TestMemoryExec::try_new_exec(&[vec![batch]], schema, None)?;
    let exec = LlmExec::try_new_task_with_registry(
        input,
        Arc::new(Column::new("text", 0)),
        Some(
            MediaPlan::new(Arc::new(Column::new("media", 1)))
                .with_resolver(Arc::new(resolver)),
        ),
        TaskKind::Vision,
        "vision".into(),
        r#"{"provider":"script","batch_size":5,"on_error":"envelope","schema_retries":1,
            "response_schema":{"type":"object","properties":{
                "visible":{"type":"boolean"}
            },"required":["visible"]}}"#,
        "answer".into(),
        &registry,
    )?;
    let collect = exec.execute(0, ctx)?.try_collect::<Vec<_>>();
    let (batches, ()) = tokio::join!(collect, async {
        let attempt = next_attempt(&mut attempts).await;
        assert_eq!(attempt.prompts, ["repairable JSON", "bad JSON", "valid"]);
        assert!(
            attempt
                .media
                .iter()
                .all(|slot| slot.as_ref().unwrap().len() == 2)
        );
        attempt.reply_with(
            &[
                Some("invalid JSON"),
                Some("{}"),
                Some(r#"{"visible":true}"#),
            ],
            None,
        );
        for _ in 0..2 {
            let retry = next_attempt(&mut attempts).await;
            assert_eq!(retry.prompts.len(), 1);
            let images = retry.media[0].as_ref().unwrap();
            assert_eq!(images.len(), 2);
            assert_eq!(images[1].bytes.as_ref(), b"image");
            let output = match retry.prompts[0].as_str() {
                "repairable JSON" => {
                    assert_eq!(images[0].bytes.as_ref(), b"retry image");
                    r#"{"visible":false}"#
                }
                "bad JSON" => {
                    assert_eq!(images[0].bytes.as_ref(), b"image");
                    "{}"
                }
                prompt => panic!("unexpected retry for {prompt}"),
            };
            retry.reply_with(&[Some(output)], None);
        }
    });
    let batches = batches?;
    let populated = batches.iter().find(|batch| batch.num_rows() == 4).unwrap();
    let envelope = populated.column(2).as_struct();
    assert_eq!(
        envelope
            .column(1)
            .as_string::<i32>()
            .iter()
            .collect::<Vec<_>>(),
        [Some("PERMANENT_HTTP"), None, Some("SCHEMA_MISMATCH"), None],
    );
    let value = envelope.column(0).as_struct();
    assert!(value.is_null(0) && value.is_null(2));
    assert!(value.is_valid(1) && !value.column(0).as_boolean().value(1));
    assert!(value.is_valid(3) && value.column(0).as_boolean().value(3));
    let skipped = batches.iter().find(|batch| batch.num_rows() == 1).unwrap();
    assert!(
        skipped
            .column(2)
            .as_struct()
            .columns()
            .iter()
            .all(|child| child.is_null(0))
    );
    assert_eq!(metric(&exec, "failed_rows"), 2);
    assert_no_attempt(&mut attempts);
    Ok(())
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
    attempt.succeed_with_usage(Some(LlmUsage {
        input_tokens: 11,
        output_tokens: 7,
    }));
    let batches = stream.try_collect::<Vec<_>>().await?;
    assert_eq!(rows(&batches), [(0, Some("a"), Some("answer:a"))]);
    assert_eq!(submitted(&exec), 1);
    assert_eq!(metric(&exec, "backend_attempts"), 3);
    assert_eq!(metric(&exec, "retries"), 2);
    assert_eq!(metric(&exec, "llm_requests"), 3);
    assert_eq!(metric(&exec, "llm_retries"), 2);
    assert_eq!(metric(&exec, "input_tokens"), 11);
    assert_eq!(metric(&exec, "output_tokens"), 7);
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
        if self.enabled(record.metadata())
            && record.args().to_string().contains("permanent test failure")
        {
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
async fn envelope_preserves_terminal_error_codes_and_retryability() -> Result<()> {
    let errors = [
        (
            LlmBackendError::RateLimited {
                retry_after: None,
                message: "throttled".into(),
            },
            "RATE_LIMITED",
            true,
        ),
        (
            LlmBackendError::Transient {
                status: Some(503),
                message: "unavailable".into(),
            },
            "TRANSIENT_HTTP",
            true,
        ),
        (
            LlmBackendError::Permanent {
                status: Some(403),
                message: "forbidden".into(),
            },
            "PERMANENT_HTTP",
            false,
        ),
        (
            LlmBackendError::InvalidResponse {
                status: Some(200),
                message: "invalid response".into(),
            },
            "INVALID_RESPONSE",
            false,
        ),
        (
            LlmBackendError::SchemaMismatch {
                message: "wrong dimension".into(),
            },
            "SCHEMA_MISMATCH",
            false,
        ),
        (LlmBackendError::Cancelled, "CANCELLED", false),
    ];
    for (error, code, retryable) in errors {
        let (registry, mut attempts) = script(1)?;
        let ctx = task_context(&registry);
        let exec = decode(
            input(&[Some("a"), Some("b")])?,
            r#"{"provider":"script","max_retries":0,"batch_size":2,"on_error":"envelope"}"#,
            &ctx,
        )?;
        let message = error.to_string();
        let collect = exec.execute(0, ctx)?.try_collect::<Vec<_>>();
        let (batches, ()) = tokio::join!(collect, async {
            next_attempt(&mut attempts)
                .await
                .reply
                .send(Err(error))
                .unwrap();
        });
        let batches = batches?;
        assert_eq!(batches.len(), 1);
        let envelope = batches[0].column(2).as_struct();
        assert_eq!(envelope.len(), 2);
        assert_eq!(envelope.column(0).null_count(), 2);
        for row in 0..2 {
            assert_eq!(envelope.column(1).as_string::<i32>().value(row), code);
            assert_eq!(envelope.column(2).as_string::<i32>().value(row), message);
            assert_eq!(envelope.column(3).as_boolean().value(row), retryable);
        }
        assert_eq!(metric(&exec, "failed_rows"), 2);
        assert_eq!(metric(&exec, "backend_attempts"), 1);
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
