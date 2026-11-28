// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: See the NOTICE file distributed with this work for additional information regarding copyright ownership
// SPDX-FileContributor: Licensed to the Apache Software Foundation (ASF) under one or more contributor license agreements

use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};

use async_trait::async_trait;
use datafusion::arrow::array::{
    Array, ArrayRef, AsArray, FixedSizeListArray, LargeListBuilder, ListArray,
    ListBuilder, StringArray, StringBuilder, StringViewBuilder, StructArray,
    new_empty_array, new_null_array,
};
use datafusion::arrow::datatypes::{DataType, Field, Metadata};
use datafusion::arrow::record_batch::RecordBatch;
use datafusion::common::{DataFusionError, Result, ScalarValue};
use datafusion::physical_plan::collect;
use datafusion::prelude::{SessionConfig, SessionContext, col, lit};
use datafusion_ai::{MediaRef, MediaRefType, MediaResolver, ResolvedMedia};

use super::harness;

const IMAGE_A: &str = "memory://test/a";
const IMAGE_B: &str = "memory://test/b";

#[derive(Debug, Default)]
struct CountingResolver(AtomicUsize);

#[async_trait]
impl MediaResolver for CountingResolver {
    async fn resolve(&self, _: &MediaRef, _: Option<u64>) -> Result<ResolvedMedia> {
        self.0.fetch_add(1, Ordering::Relaxed);
        Ok(ResolvedMedia {
            bytes: b"image".to_vec().into(),
            mime: None,
            etag: None,
        })
    }
}

fn vision_context() -> (SessionContext, Arc<CountingResolver>) {
    let resolver = Arc::new(CountingResolver::default());
    let ctx = SessionContext::new_with_config(
        SessionConfig::new()
            .with_target_partitions(1)
            .with_extension(Arc::new(resolver.clone() as Arc<dyn MediaResolver>)),
    );
    (ctx, resolver)
}

fn uri_list<'a>(images: impl IntoIterator<Item = &'a str>) -> ListArray {
    let mut builder = ListBuilder::new(StringBuilder::new());
    builder.append_value(images.into_iter().map(Some));
    builder.finish()
}

#[tokio::test]
async fn vision_accepts_all_three_list_types() -> Result<()> {
    let (ctx, resolver) = vision_context();
    let mut large = LargeListBuilder::new(StringViewBuilder::new());
    large.append_value([Some(IMAGE_A), Some(IMAGE_B)]);

    let item = Arc::new(MediaRefType::field("image", true)?);
    let DataType::Struct(fields) = item.data_type() else {
        unreachable!()
    };
    let references = StructArray::try_new(
        fields.clone(),
        std::iter::once(Arc::new(StringArray::from(vec![IMAGE_A, IMAGE_B])) as ArrayRef)
            .chain(fields[1..].iter().map(|f| new_null_array(f.data_type(), 2)))
            .collect(),
        None,
    )?;
    let fixed = FixedSizeListArray::try_new(item, 2, Arc::new(references), None)?;
    ctx.register_batch(
        "t",
        RecordBatch::try_from_iter([
            (
                "prompt",
                Arc::new(StringArray::from(vec!["prompt"])) as ArrayRef,
            ),
            ("regular", Arc::new(uri_list([IMAGE_A, IMAGE_B]))),
            ("large", Arc::new(large.finish())),
            ("fixed", Arc::new(fixed)),
        ])?,
    )?;

    let (plan, batches) = harness::run(
        &ctx,
        "SELECT vlm(prompt, regular, 'm', '{}'), vlm(prompt, large, 'm', '{}'), \
         vlm(prompt, fixed, 'm', '{}') FROM t",
    )
    .await?;
    assert_eq!(harness::exec_nodes(plan.as_ref()).len(), 3);
    for column in 0..3 {
        assert_eq!(
            harness::strings(&batches, column),
            [Some("mock[m]: prompt")]
        );
    }
    assert_eq!(resolver.0.load(Ordering::Relaxed), 6);
    Ok(())
}

#[tokio::test]
async fn vision_rejects_unsupported_media_at_planning() -> Result<()> {
    let ctx = harness::test_context(8, 1);
    let string = Arc::new(Field::new_list_field(DataType::Utf8, true));
    let unannotated_ref =
        MediaRefType::field("image", true)?.with_metadata(Metadata::default());
    for (name, kind) in [
        (
            "nested",
            DataType::List(Arc::new(Field::new_list_field(
                DataType::List(string),
                true,
            ))),
        ),
        (
            "integers",
            DataType::List(Arc::new(Field::new_list_field(DataType::Int32, true))),
        ),
        ("raw_struct", unannotated_ref.data_type().clone()),
        (
            "raw_struct_items",
            DataType::List(Arc::new(unannotated_ref)),
        ),
    ] {
        ctx.register_batch(
            name,
            RecordBatch::try_from_iter([("media", new_empty_array(&kind))])?,
        )?;
        let sql = format!("SELECT vlm('prompt', media, 'm', '{{}}') FROM {name}");
        let error = match ctx.sql(&sql).await {
            Err(error) => error,
            Ok(df) => df.into_optimized_plan().unwrap_err(),
        };
        assert!(
            matches!(error.find_root(), DataFusionError::Plan(_)),
            "{error}"
        );
        let message = error.to_string();
        assert!(
            message
                .contains("vlm() media must be a string URI or a df.media_ref v1 field"),
            "{message}"
        );
        assert!(
            message.contains("List, LargeList, or FixedSizeList"),
            "{message}"
        );
        assert!(message.contains(&format!("got {kind}")), "{message}");
    }
    Ok(())
}

#[tokio::test]
async fn vision_validates_list_budget_options_at_planning() -> Result<()> {
    let (ctx, _) = vision_context();
    ctx.register_batch(
        "t",
        RecordBatch::try_from_iter([(
            "media",
            Arc::new(uri_list([IMAGE_A])) as ArrayRef,
        )])?,
    )?;
    for (options, message) in [
        (
            r#"{"max_media_items":0}"#,
            "max_media_items must be between 1 and 64",
        ),
        (
            r#"{"max_media_items":65}"#,
            "max_media_items must be between 1 and 64",
        ),
        (
            r#"{"max_row_media_bytes":134217729}"#,
            "max_row_media_bytes must be at most 134217728 bytes",
        ),
    ] {
        let sql = format!("SELECT vlm('prompt', media, 'm', '{options}') FROM t");
        harness::assert_plan_error(&ctx, &sql, &[message]).await;
    }
    ctx.sql(r#"SELECT vlm('prompt', media, 'm', '{"max_media_items":64,"max_row_media_bytes":134217728}') FROM t"#)
        .await?
        .create_physical_plan()
        .await?;
    Ok(())
}

#[tokio::test]
async fn vision_validates_image_resize_options_at_planning() -> Result<()> {
    let (ctx, resolver) = vision_context();
    for (sql, message) in [
        (
            r#"SELECT llm('prompt', 'm', '{"image_max_edge":16}')"#,
            "image_max_edge is only supported by vlm()",
        ),
        (
            r#"SELECT embed('prompt', 'm', '{"dimension":4,"image_max_edge":16}')"#,
            "image_max_edge is only supported by vlm()",
        ),
    ] {
        harness::assert_plan_error(&ctx, sql, &[message]).await;
    }

    for (options, message) in [
        (
            r#"{"image_max_edge":15}"#,
            "image_max_edge must be between 16 and 8192",
        ),
        (
            r#"{"image_max_edge":8193}"#,
            "image_max_edge must be between 16 and 8192",
        ),
        (
            r#"{"image_resize_fallback":"passthrough"}"#,
            "image_resize_fallback requires image_max_edge",
        ),
        (
            r#"{"image_max_edge":16,"image_resize_fallback":"ignore"}"#,
            "unknown variant",
        ),
    ] {
        let sql = format!("SELECT vlm('prompt', '{IMAGE_A}', 'm', '{options}')");
        harness::assert_plan_error(&ctx, &sql, &[message]).await;
    }

    for (edge, fallback) in [(16, "error"), (8192, "passthrough")] {
        let sql = format!(
            r#"SELECT vlm('prompt', '{IMAGE_A}', 'm', '{{"image_max_edge":{edge},"image_resize_fallback":"{fallback}"}}')"#
        );
        ctx.sql(&sql).await?.create_physical_plan().await?;
    }
    assert_eq!(resolver.0.load(Ordering::Relaxed), 0);
    Ok(())
}

#[tokio::test]
async fn vision_list_nulls_limits_and_unused_output_never_resolve_media() -> Result<()> {
    let (ctx, resolver) = vision_context();
    let mut lists = ListBuilder::new(StringBuilder::new());
    lists.append_value([Some(IMAGE_A)]);
    lists.append_value([Some(IMAGE_B)]);
    let lists = Arc::new(lists.finish()) as ArrayRef;
    ctx.register_batch(
        "t",
        RecordBatch::try_from_iter([
            (
                "prompt",
                Arc::new(StringArray::from(vec!["first", "second"])) as ArrayRef,
            ),
            ("images", Arc::clone(&lists)),
            ("null_images", new_null_array(lists.data_type(), 2)),
        ])?,
    )?;

    for (sql, rows) in [
        (
            "SELECT prompt FROM (SELECT prompt, vlm(prompt, images, 'm', '{}') AS response FROM t)",
            2,
        ),
        ("SELECT vlm(prompt, images, 'm', '{}') FROM t LIMIT 0", 0),
    ] {
        let (plan, batches) = harness::run(&ctx, sql).await?;
        assert_eq!(
            batches.iter().map(RecordBatch::num_rows).sum::<usize>(),
            rows
        );
        assert_eq!(harness::metric(plan.as_ref(), "llm_requests"), 0, "{sql}");
        assert_eq!(resolver.0.load(Ordering::Relaxed), 0, "{sql}");
    }
    for args in ["NULL, images", "prompt, null_images"] {
        let sql = format!(
            "SELECT vlm({args}, 'm', '{{}}'), vlm({args}, 'm', '{{\"on_error\":\"envelope\"}}') FROM t"
        );
        let (plan, batches) = harness::run(&ctx, &sql).await?;
        assert_eq!(harness::strings(&batches, 0), [None, None]);
        for batch in batches {
            let envelope = batch.column(1).as_struct();
            assert_eq!(envelope.num_columns(), 4);
            assert!(
                envelope
                    .columns()
                    .iter()
                    .all(|field| field.null_count() == batch.num_rows())
            );
        }
        assert_eq!(harness::metric(plan.as_ref(), "llm_requests"), 0, "{sql}");
        assert_eq!(resolver.0.load(Ordering::Relaxed), 0, "{sql}");
    }

    let (plan, batches) =
        harness::run(&ctx, "SELECT vlm(prompt, images, 'm', '{}') FROM t LIMIT 1")
            .await?;
    assert_eq!(harness::strings(&batches, 0), [Some("mock[m]: first")]);
    assert_eq!(harness::metric(plan.as_ref(), "llm_requests"), 1);
    assert_eq!(resolver.0.load(Ordering::Relaxed), 1);
    Ok(())
}

#[tokio::test]
async fn vision_deduplication_compares_ordered_list_contents() -> Result<()> {
    let (ctx, resolver) = vision_context();
    harness::register_text(&ctx, "t", &[Some("prompt")])?;
    let input = ctx.table("t").await?;
    let vlm = input.registry().udf("vlm")?;
    let call = |images, alias| {
        vlm.call(vec![
            col("text"),
            lit(ScalarValue::List(Arc::new(uri_list(images)))),
            lit("m"),
            lit(r#"{"deduplicate":true}"#),
        ])
        .alias(alias)
    };
    let plan = input
        .select(vec![
            call([IMAGE_A, IMAGE_B], "ab"),
            call([IMAGE_B, IMAGE_A], "ba"),
            call([IMAGE_A, IMAGE_B], "ab_again"),
        ])?
        .create_physical_plan()
        .await?;
    let batches = collect(Arc::clone(&plan), ctx.task_ctx()).await?;
    assert_eq!(harness::exec_nodes(plan.as_ref()).len(), 2);
    assert_eq!(harness::metric(plan.as_ref(), "llm_requests"), 2);
    assert_eq!(resolver.0.load(Ordering::Relaxed), 4);
    for column in 0..3 {
        assert_eq!(
            harness::strings(&batches, column),
            [Some("mock[m]: prompt")]
        );
    }
    Ok(())
}

#[cfg(all(feature = "ai-http", feature = "parquet"))]
#[tokio::test]
async fn parquet_sql_vision_sends_three_images_in_list_order() -> Result<()> {
    use std::fs::File;
    use std::time::Duration;

    use datafusion::datasource::listing::ListingTableUrl;
    use datafusion::execution::runtime_env::RuntimeEnv;
    use datafusion::llm::{OpenAiHttpBackend, OpenAiHttpConfig};
    use datafusion::parquet::arrow::ArrowWriter;
    use datafusion::prelude::ParquetReadOptions;
    use datafusion_ai::ObjectStoreMediaResolver;
    use serde_json::json;
    use tokio::net::TcpListener;

    let directory = tempfile::tempdir()?;
    let path = directory.path().join("t.parquet");
    let mut images = Vec::new();
    for (name, content) in [("a.png", b"1"), ("b.png", b"2"), ("c.png", b"3")] {
        let image_path = directory.path().join(name);
        std::fs::write(&image_path, content)?;
        images.push(ListingTableUrl::parse(image_path.to_str().unwrap())?);
    }
    let batch = RecordBatch::try_from_iter([
        (
            "prompt",
            Arc::new(StringArray::from(vec!["ordered images"])) as ArrayRef,
        ),
        (
            "img_list",
            Arc::new(uri_list(images.iter().map(ListingTableUrl::as_str))),
        ),
    ])?;
    let mut writer = ArrowWriter::try_new(File::create(&path)?, batch.schema(), None)?;
    writer.write(&batch)?;
    writer.close()?;

    let listener = TcpListener::bind("127.0.0.1:0").await?;
    let key = "loopback-vision-list-test-key";
    let backend = OpenAiHttpBackend::try_new(OpenAiHttpConfig {
        base_url: format!("http://{}/v1", listener.local_addr()?),
        api_key: Some(key.into()),
        api_key_env: None,
        ca_cert_pem: None,
    })?;
    let runtime = Arc::new(RuntimeEnv::default());
    let resolver =
        ObjectStoreMediaResolver::new(Arc::clone(&runtime.object_store_registry));
    let ctx = SessionContext::new_with_config_rt(
        SessionConfig::new().with_extension(Arc::new(resolver)),
        runtime,
    );
    ctx.state()
        .llm_provider_registry()
        .register("http", Arc::new(backend))?;
    ctx.register_parquet("t", path.to_str().unwrap(), ParquetReadOptions::default())
        .await?;

    let serve = async {
        let (endpoint, body, socket) = super::read_http_request(&listener, key).await;
        assert_eq!(endpoint, "/v1/chat/completions");
        assert_eq!(
            body["messages"][0]["content"],
            json!([
                {"type": "text", "text": "ordered images"},
                {"type": "image_url", "image_url": {"url": "data:application/octet-stream;base64,MQ=="}},
                {"type": "image_url", "image_url": {"url": "data:application/octet-stream;base64,Mg=="}},
                {"type": "image_url", "image_url": {"url": "data:application/octet-stream;base64,Mw=="}},
            ])
        );
        super::write_http_response(
            socket,
            r#"{"choices":[{"message":{"content":"three images"}}]}"#,
        )
        .await;
    };
    let query = harness::run(
        &ctx,
        r#"SELECT vlm(prompt, img_list, 'm', '{"provider":"http"}') FROM t"#,
    );
    let (result, ()) = tokio::time::timeout(Duration::from_secs(8), async {
        tokio::join!(query, serve)
    })
    .await
    .unwrap();
    let (plan, batches) = result?;
    assert_eq!(harness::metric(plan.as_ref(), "llm_requests"), 1);
    assert_eq!(harness::strings(&batches, 0), [Some("three images")]);
    Ok(())
}

#[cfg(all(feature = "ai-http", feature = "parquet"))]
#[tokio::test]
async fn parquet_sql_isolates_multi_image_and_schema_errors() -> Result<()> {
    use std::fs::File;
    use std::time::Duration;

    use datafusion::arrow::datatypes::Int64Type;
    use datafusion::datasource::listing::ListingTableUrl;
    use datafusion::execution::runtime_env::RuntimeEnv;
    use datafusion::llm::{OpenAiHttpBackend, OpenAiHttpConfig};
    use datafusion::parquet::arrow::ArrowWriter;
    use datafusion::prelude::ParquetReadOptions;
    use datafusion_ai::{AiErrorType, ObjectStoreMediaResolver};
    use serde_json::{Value, json};
    use tokio::net::TcpListener;

    let directory = tempfile::tempdir()?;
    let mut uris = Vec::new();
    for (name, contents) in [("a.png", b"a"), ("b.png", b"b")] {
        let path = directory.path().join(name);
        std::fs::write(&path, contents)?;
        uris.push(ListingTableUrl::parse(path.to_str().unwrap())?);
    }

    let prompts = ["a-first", "b-bad-image", "c-bad-schema", "d-last"];
    let mut images = ListBuilder::new(StringBuilder::new());
    for prompt in prompts {
        let second = if prompt == "b-bad-image" {
            "not a URI"
        } else {
            uris[1].as_str()
        };
        images.append_value([Some(uris[0].as_str()), Some(second)]);
    }
    let batch = RecordBatch::try_from_iter([
        (
            "prompt",
            Arc::new(StringArray::from(prompts.to_vec())) as ArrayRef,
        ),
        ("images", Arc::new(images.finish())),
    ])?;
    let path = directory.path().join("t.parquet");
    let mut writer = ArrowWriter::try_new(File::create(&path)?, batch.schema(), None)?;
    writer.write(&batch)?;
    writer.close()?;

    let listener = TcpListener::bind("127.0.0.1:0").await?;
    let key = "loopback-vision-schema-test-key";
    let backend = OpenAiHttpBackend::try_new(OpenAiHttpConfig {
        base_url: format!("http://{}/v1", listener.local_addr()?),
        api_key: Some(key.into()),
        api_key_env: None,
        ca_cert_pem: None,
    })?;
    let runtime = Arc::new(RuntimeEnv::default());
    let resolver =
        ObjectStoreMediaResolver::new(Arc::clone(&runtime.object_store_registry));
    let ctx = SessionContext::new_with_config_rt(
        SessionConfig::new()
            .with_target_partitions(1)
            .with_extension(Arc::new(resolver)),
        runtime,
    );
    ctx.state()
        .llm_provider_registry()
        .register("http", Arc::new(backend))?;
    ctx.register_parquet("t", path.to_str().unwrap(), ParquetReadOptions::default())
        .await?;

    let serve = async {
        for _ in 0..7 {
            let (endpoint, body, socket) = super::read_http_request(&listener, key).await;
            assert_eq!(endpoint, "/v1/chat/completions");
            let content = &body["messages"][0]["content"];
            let answer = if let Some(prompt) = content.as_str() {
                let format = &body["response_format"];
                assert_eq!(format["type"], "json_schema");
                assert_eq!(format["json_schema"]["strict"], true);
                let count = match prompt {
                    "a-first" => json!(10),
                    "b-bad-image" => json!(20),
                    "c-bad-schema" => json!("30"),
                    "d-last" => json!(40),
                    _ => panic!("unexpected chat prompt {prompt}"),
                };
                json!({"z_label": prompt, "a_count": count}).to_string()
            } else {
                let prompt = content[0]["text"].as_str().unwrap();
                assert!(matches!(prompt, "a-first" | "c-bad-schema" | "d-last"));
                assert_eq!(
                    content,
                    &json!([
                        {"type":"text","text":prompt},
                        {"type":"image_url","image_url":{"url":"data:application/octet-stream;base64,YQ=="}},
                        {"type":"image_url","image_url":{"url":"data:application/octet-stream;base64,Yg=="}}
                    ])
                );
                assert_eq!(body["response_format"], Value::Null);
                format!("vision {prompt}")
            };
            let response = json!({"choices":[{"message":{"content":answer}}]});
            super::write_http_response(socket, &response.to_string()).await;
        }
        Ok(())
    };
    let query = harness::run(
        &ctx,
        r#"SELECT
            llm(prompt, 'm', '{"provider":"http","on_error":"envelope","response_schema":{"type":"object","properties":{"z_label":{"type":"string"},"a_count":{"type":"integer"}},"required":["z_label","a_count"]}}') AS chat,
            vlm(prompt, images, 'm', '{"provider":"http","on_error":"envelope"}') AS vision
        FROM t ORDER BY prompt"#,
    );
    let ((plan, batches), ()) = tokio::time::timeout(Duration::from_secs(8), async {
        tokio::try_join!(query, serve)
    })
    .await
    .expect("SQL and loopback server must finish")?;
    assert_eq!(harness::metric(plan.as_ref(), "llm_requests"), 7);
    let schema = plan.schema();
    schema.field(0).try_extension_type::<AiErrorType>()?;
    schema.field(1).try_extension_type::<AiErrorType>()?;

    let rows: Vec<_> = batches
        .iter()
        .flat_map(|batch| {
            let chat = batch.column(0).as_struct();
            let vision = batch.column(1).as_struct();
            let value = chat.column(0).as_struct();
            assert_eq!(
                value.fields()[0].as_ref(),
                &Field::new("z_label", DataType::Utf8, false)
            );
            assert_eq!(
                value.fields()[1].as_ref(),
                &Field::new("a_count", DataType::Int64, false)
            );
            let labels = value.column(0).as_string::<i32>();
            let counts = value.column(1).as_primitive::<Int64Type>();
            let chat_errors = chat.column(1).as_string::<i32>();
            let chat_retryable = chat.column(3).as_boolean();
            let vision_answers = vision.column(0).as_string::<i32>();
            let vision_errors = vision.column(1).as_string::<i32>();
            (0..batch.num_rows()).map(move |row| {
                (
                    value
                        .is_valid(row)
                        .then(|| (labels.value(row), counts.value(row))),
                    chat_errors.is_valid(row).then(|| chat_errors.value(row)),
                    chat_retryable
                        .is_valid(row)
                        .then(|| chat_retryable.value(row)),
                    vision_answers
                        .is_valid(row)
                        .then(|| vision_answers.value(row)),
                    vision_errors
                        .is_valid(row)
                        .then(|| vision_errors.value(row)),
                )
            })
        })
        .collect();
    assert_eq!(
        rows,
        [
            (
                Some(("a-first", 10)),
                None,
                None,
                Some("vision a-first"),
                None
            ),
            (
                Some(("b-bad-image", 20)),
                None,
                None,
                None,
                Some("PERMANENT_HTTP")
            ),
            (
                None,
                Some("SCHEMA_MISMATCH"),
                Some(false),
                Some("vision c-bad-schema"),
                None
            ),
            (
                Some(("d-last", 40)),
                None,
                None,
                Some("vision d-last"),
                None
            ),
        ]
    );
    Ok(())
}

#[cfg(all(
    feature = "ai-http",
    feature = "parquet",
    feature = "encoding_expressions"
))]
#[tokio::test]
async fn parquet_sql_vision_downsamples_images_and_retries_schema_rows() -> Result<()> {
    use std::fs::File;
    use std::time::Duration;

    use datafusion::arrow::datatypes::Int64Type;
    use datafusion::datasource::listing::ListingTableUrl;
    use datafusion::execution::runtime_env::RuntimeEnv;
    use datafusion::functions::encoding::expr_fn::decode;
    use datafusion::llm::{OpenAiHttpBackend, OpenAiHttpConfig};
    use datafusion::parquet::arrow::ArrowWriter;
    use datafusion::prelude::ParquetReadOptions;
    use datafusion_ai::{AiErrorType, ObjectStoreMediaResolver};
    use serde_json::json;
    use tokio::net::TcpListener;

    const PNG: &[u8] = include_bytes!("fixtures/downsample-96x48.png");
    const JPEG: &[u8] = include_bytes!("../../../ai/tests/fixtures/vision.jpeg");

    let directory = tempfile::tempdir()?;
    let save_image = |name, bytes| -> Result<ListingTableUrl> {
        let path = directory.path().join(name);
        std::fs::write(&path, bytes)?;
        ListingTableUrl::parse(path.to_str().unwrap())
    };
    let png = save_image("large.png", PNG)?;
    let jpeg = save_image("original.jpg", JPEG)?;
    let prompts = ["a-png", "b-bad-uri", "c-recover", "d-exhaust", "e-jpeg"];
    let mut images = ListBuilder::new(StringBuilder::new());
    images.append_value([Some(png.as_str())]);
    images.append_value([Some("not a URI")]);
    images.append_value([Some(png.as_str()), Some(jpeg.as_str())]);
    images.append_value([Some(png.as_str())]);
    images.append_value([Some(jpeg.as_str())]);
    let batch = RecordBatch::try_from_iter([
        (
            "prompt",
            Arc::new(StringArray::from(prompts.to_vec())) as ArrayRef,
        ),
        ("images", Arc::new(images.finish())),
    ])?;
    let path = directory.path().join("t.parquet");
    let mut writer = ArrowWriter::try_new(File::create(&path)?, batch.schema(), None)?;
    writer.write(&batch)?;
    writer.close()?;

    let listener = TcpListener::bind("127.0.0.1:0").await?;
    let key = "loopback-vision-resize-schema-test-key";
    let backend = OpenAiHttpBackend::try_new(OpenAiHttpConfig {
        base_url: format!("http://{}/v1", listener.local_addr()?),
        api_key: Some(key.into()),
        api_key_env: None,
        ca_cert_pem: None,
    })?;
    let runtime = Arc::new(RuntimeEnv::default());
    let resolver =
        ObjectStoreMediaResolver::new(Arc::clone(&runtime.object_store_registry));
    let ctx = SessionContext::new_with_config_rt(
        SessionConfig::new()
            .with_target_partitions(1)
            .with_extension(Arc::new(resolver)),
        runtime,
    );
    ctx.state()
        .llm_provider_registry()
        .register("http", Arc::new(backend))?;
    ctx.register_parquet("t", path.to_str().unwrap(), ParquetReadOptions::default())
        .await?;

    let serve = async {
        let mut attempts = [0; 5];
        for _ in 0..6 {
            let (endpoint, body, socket) = super::read_http_request(&listener, key).await;
            assert_eq!(endpoint, "/v1/chat/completions");
            assert_eq!(body["response_format"], json!({"type":"json_object"}));
            let messages = body["messages"].as_array().unwrap();
            assert_eq!(messages.len(), 2);
            assert_eq!(messages[0]["role"], "system");
            assert!(
                messages[0]["content"]
                    .as_str()
                    .unwrap()
                    .contains("JSON Schema")
            );
            assert_eq!(messages[1]["role"], "user");
            let content = messages[1]["content"].as_array().unwrap();
            assert_eq!(content[0]["type"], "text");
            let prompt = content[0]["text"].as_str().unwrap();
            let row = prompts
                .iter()
                .position(|candidate| *candidate == prompt)
                .unwrap();
            attempts[row] += 1;
            let mime_types: &[_] = match prompt {
                "a-png" | "d-exhaust" => &["data:image/png"],
                "c-recover" => &["data:image/png", "data:image/jpeg"],
                "e-jpeg" => &["data:image/jpeg"],
                _ => panic!("unexpected inference request for {prompt}"),
            };
            assert_eq!(content.len(), 1 + mime_types.len());
            for (image, mime_type) in content[1..].iter().zip(mime_types) {
                assert_eq!(image["type"], "image_url");
                let url = image["image_url"]["url"].as_str().unwrap();
                let (prefix, encoded) = url.split_once(";base64,").unwrap();
                assert_eq!(prefix, *mime_type);
                let decoded = ctx
                    .read_empty()?
                    .select(vec![decode(lit(encoded), lit("base64"))])?
                    .collect()
                    .await?;
                let bytes = decoded[0].column(0).as_binary::<i32>().value(0);
                if prefix == "data:image/png" {
                    assert!(bytes.starts_with(b"\x89PNG\r\n\x1a\n\x00\x00\x00\x0dIHDR"));
                    let width = u32::from_be_bytes(bytes[16..20].try_into().unwrap());
                    let height = u32::from_be_bytes(bytes[20..24].try_into().unwrap());
                    assert_eq!((width, height), (64, 32));
                } else {
                    assert_eq!(bytes, JPEG);
                }
            }
            let answer = match prompt {
                "a-png" => r#"{"count":1}"#,
                "c-recover" if attempts[row] == 2 => "```json\n{\"count\":3}\n```",
                "c-recover" | "d-exhaust" => r#"{"count":"invalid"}"#,
                "e-jpeg" => r#"{"count":5}"#,
                _ => unreachable!(),
            };
            let response = json!({"choices":[{"message":{"content":answer}}]});
            super::write_http_response(socket, &response.to_string()).await;
        }
        assert_eq!(attempts, [1, 0, 2, 2, 1]);
        Ok(())
    };
    let query = harness::run(
        &ctx,
        r#"SELECT vlm(prompt, images, 'm', '{
            "provider":"http", "on_error":"envelope", "max_retries":0,
            "response_schema":{"type":"object","properties":{"count":{"type":"integer"}},"required":["count"]},
            "schema_mode":"json_object", "schema_retries":1,
            "image_max_edge":64, "image_resize_fallback":"passthrough"
        }') AS answer FROM t ORDER BY prompt"#,
    );
    let ((plan, batches), ()) = tokio::time::timeout(Duration::from_secs(10), async {
        tokio::try_join!(query, serve)
    })
    .await
    .expect("SQL and loopback server must finish")?;
    assert_eq!(harness::metric(plan.as_ref(), "llm_requests"), 6);
    assert_eq!(harness::metric(plan.as_ref(), "llm_retries"), 2);
    plan.schema().field(0).try_extension_type::<AiErrorType>()?;

    let rows: Vec<_> = batches
        .iter()
        .flat_map(|batch| {
            let envelope = batch.column(0).as_struct();
            let value = envelope.column(0).as_struct();
            assert_eq!(
                value.fields()[0].as_ref(),
                &Field::new("count", DataType::Int64, false)
            );
            let count = value.column(0).as_primitive::<Int64Type>();
            let error = envelope.column(1).as_string::<i32>();
            let retryable = envelope.column(3).as_boolean();
            (0..batch.num_rows()).map(move |row| {
                (
                    value.is_valid(row).then(|| count.value(row)),
                    error.is_valid(row).then(|| error.value(row)),
                    retryable.is_valid(row).then(|| retryable.value(row)),
                )
            })
        })
        .collect();
    assert_eq!(
        rows,
        [
            (Some(1), None, None),
            (None, Some("PERMANENT_HTTP"), Some(false)),
            (Some(3), None, None),
            (None, Some("SCHEMA_MISMATCH"), Some(false)),
            (Some(5), None, None),
        ]
    );
    Ok(())
}
