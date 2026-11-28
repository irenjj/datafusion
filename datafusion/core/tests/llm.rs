// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: See the NOTICE file distributed with this work for additional information regarding copyright ownership
// SPDX-FileContributor: Licensed to the Apache Software Foundation (ASF) under one or more contributor license agreements

#[path = "llm/async_exec.rs"]
mod async_exec;
#[path = "llm/contract.rs"]
mod contract;
#[path = "llm/fuzz.rs"]
mod fuzz;
#[path = "llm/harness.rs"]
mod harness;
#[path = "llm/scripted_backend.rs"]
mod scripted_backend;

#[cfg(feature = "ai-http")]
#[tokio::test]
async fn loopback_http_sql_reaches_the_registered_backend()
-> datafusion::common::Result<()> {
    use std::sync::Arc;
    use std::time::Duration;

    use datafusion::llm::{OpenAiHttpBackend, OpenAiHttpConfig};
    use tokio::net::TcpListener;

    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let backend = OpenAiHttpBackend::try_new(OpenAiHttpConfig {
        base_url: format!("http://{}/v1", listener.local_addr().unwrap()),
        api_key: Some("loopback-sql-test-key".into()),
        api_key_env: None,
        ca_cert_pem: None,
    })?;
    let ctx = harness::test_context(8, 1);
    ctx.state().llm_provider_registry().register_with_limit(
        "http",
        Arc::new(backend),
        2,
    )?;
    harness::register_text(&ctx, "t", &[Some("sql prompt")])?;

    let serve = async {
        let (endpoint, body, socket) =
            read_http_request(&listener, "loopback-sql-test-key").await;
        assert_eq!(endpoint, "/v1/chat/completions");
        assert_eq!(body["model"], "test-model");
        assert_eq!(body["messages"][0]["content"], "sql prompt");
        assert_eq!(body["stream"], false);
        assert_eq!(body["n"], 1);
        write_http_response(
            socket,
            r#"{"choices":[{"message":{"content":"via HTTP"}}]}"#,
        )
        .await;
    };
    let run = harness::run(
        &ctx,
        r#"SELECT llm(text, 'test-model', '{"provider":"http"}') AS answer FROM t"#,
    );
    let (query, ()) =
        tokio::time::timeout(Duration::from_secs(5), async { tokio::join!(run, serve) })
            .await
            .unwrap();
    let (plan, batches) = query?;
    assert_eq!(harness::exec_nodes(plan.as_ref()).len(), 1);
    assert_eq!(harness::strings(&batches, 0), [Some("via HTTP")]);
    Ok(())
}

#[tokio::test]
async fn unused_limited_and_null_vision_prompts_never_resolve_media()
-> datafusion::common::Result<()> {
    use std::sync::Arc;
    use std::sync::atomic::{AtomicUsize, Ordering};

    use async_trait::async_trait;
    use datafusion::common::Result;
    use datafusion::prelude::{SessionConfig, SessionContext};
    use datafusion_ai::{MediaRef, MediaResolver, ResolvedMedia};

    #[derive(Debug, Default)]
    struct CountingResolver(AtomicUsize);

    #[async_trait]
    impl MediaResolver for CountingResolver {
        async fn resolve(&self, _: &MediaRef, _: Option<u64>) -> Result<ResolvedMedia> {
            self.0.fetch_add(1, Ordering::Relaxed);
            datafusion::common::exec_err!("unexpected media read")
        }
    }

    let resolver = Arc::new(CountingResolver::default());
    let ctx = SessionContext::new_with_config(
        SessionConfig::new()
            .with_extension(Arc::new(resolver.clone() as Arc<dyn MediaResolver>)),
    );
    harness::register_text(&ctx, "t", &[Some("memory://test/asset")])?;
    for (sql, expected_rows) in [
        (
            "SELECT id FROM (SELECT id, vlm(text, text, 'm', '{}') AS response FROM t)",
            1,
        ),
        ("SELECT vlm(text, text, 'm', '{}') FROM t LIMIT 0", 0),
        ("SELECT vlm(NULL, text, 'm', '{}') FROM t", 1),
    ] {
        let (_, batches) = harness::run(&ctx, sql).await?;
        assert_eq!(
            batches.iter().map(|batch| batch.num_rows()).sum::<usize>(),
            expected_rows,
            "{sql}"
        );
        assert_eq!(resolver.0.load(Ordering::Relaxed), 0, "{sql}");
    }
    Ok(())
}

#[cfg(all(feature = "ai-http", feature = "parquet"))]
#[tokio::test]
async fn parquet_sql_embeds_text_and_resolves_vision_media_over_http()
-> datafusion::common::Result<()> {
    use std::fs::File;
    use std::sync::Arc;
    use std::time::Duration;

    use datafusion::arrow::array::{
        Array, FixedSizeListArray, Float32Array, Int32Array, StringArray,
    };
    use datafusion::arrow::datatypes::{DataType, Field, Schema};
    use datafusion::arrow::record_batch::RecordBatch;
    use datafusion::datasource::listing::ListingTableUrl;
    use datafusion::execution::runtime_env::RuntimeEnv;
    use datafusion::llm::{OpenAiHttpBackend, OpenAiHttpConfig};
    use datafusion::parquet::arrow::ArrowWriter;
    use datafusion::prelude::{ParquetReadOptions, SessionConfig, SessionContext};
    use datafusion_ai::ObjectStoreMediaResolver;
    use serde_json::json;
    use tokio::net::TcpListener;

    let directory = tempfile::tempdir()?;
    let path = directory.path().join("t.parquet");
    let image = |filename: &str, content: &[u8]| -> datafusion::common::Result<String> {
        let path = directory.path().join(filename);
        std::fs::write(&path, content)?;
        Ok(ListingTableUrl::parse(path.to_str().unwrap())?
            .as_str()
            .to_owned())
    };
    let batch = RecordBatch::try_new(
        Arc::new(Schema::new(vec![
            Field::new("id", DataType::Int32, false),
            Field::new("text", DataType::Utf8, true),
            Field::new("img_uri", DataType::Utf8, false),
        ])),
        vec![
            Arc::new(Int32Array::from(vec![1, 2, 3, 4])),
            Arc::new(StringArray::from(vec![
                Some("a"),
                Some("b"),
                None,
                Some("d"),
            ])),
            Arc::new(StringArray::from(vec![
                image("a.png", b"a")?,
                image("b.png", b"b")?,
                "file:///unused".into(),
                image("d.png", b"d")?,
            ])),
        ],
    )?;
    let mut writer = ArrowWriter::try_new(File::create(&path)?, batch.schema(), None)?;
    writer.write(&batch)?;
    writer.close()?;

    let listener = TcpListener::bind("127.0.0.1:0").await?;
    let backend = OpenAiHttpBackend::try_new(OpenAiHttpConfig {
        base_url: format!("http://{}/v1", listener.local_addr()?),
        api_key: Some("loopback-parquet-test-key".into()),
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
        for _ in 0..5 {
            let (endpoint, body, socket) =
                read_http_request(&listener, "loopback-parquet-test-key").await;
            let response = match endpoint.as_str() {
                "/v1/embeddings" => {
                    let inputs = body["input"].as_array().unwrap();
                    let data = inputs.iter().enumerate().rev().map(|(index, input)| {
                        let value = input.as_str().unwrap().as_bytes()[0] - b'a';
                        json!({"index": index, "embedding": [value, 1]})
                    }).collect::<Vec<_>>();
                    json!({"data": data})
                }
                "/v1/chat/completions" => {
                    let content = body["messages"][0]["content"].as_array().unwrap();
                    assert_eq!(content.len(), 2);
                    let text = content[0]["text"].as_str().unwrap();
                    let data = match text {
                        "a" => "YQ==",
                        "b" => "Yg==",
                        "d" => "ZA==",
                        _ => panic!("unexpected prompt {text}"),
                    };
                    assert_eq!(content[1]["image_url"]["url"], format!("data:application/octet-stream;base64,{data}"));
                    json!({"choices": [{"message": {"content": format!("vision {text}")}}]})
                }
                _ => panic!("unexpected endpoint {endpoint}"),
            }.to_string();
            write_http_response(socket, &response).await;
        }
    };
    let query = async {
        let (_, embed) = harness::run(&ctx, r#"SELECT embed(text, 'm', concat('{"provider":"http",', '"dimension":2}')) AS v FROM t ORDER BY id"#).await?;
        let (_, vision) = harness::run(&ctx, r#"SELECT vlm(text, img_uri, 'm', '{"provider":"http"}') AS v FROM t ORDER BY id"#).await?;
        Ok::<_, datafusion::common::DataFusionError>((embed, vision))
    };
    let (result, ()) = tokio::time::timeout(Duration::from_secs(8), async {
        tokio::join!(query, serve)
    })
    .await
    .unwrap();
    let (embed, vision) = result?;
    let mut vectors = Vec::new();
    for batch in &embed {
        let lists = batch
            .column(0)
            .as_any()
            .downcast_ref::<FixedSizeListArray>()
            .unwrap();
        for row in 0..lists.len() {
            vectors.push(lists.is_valid(row).then(|| {
                lists
                    .value(row)
                    .as_any()
                    .downcast_ref::<Float32Array>()
                    .unwrap()
                    .values()
                    .to_vec()
            }));
        }
    }
    assert_eq!(
        vectors,
        [
            Some(vec![0.0, 1.0]),
            Some(vec![1.0, 1.0]),
            None,
            Some(vec![3.0, 1.0])
        ]
    );
    assert_eq!(
        harness::strings(&vision, 0),
        [Some("vision a"), Some("vision b"), None, Some("vision d")]
    );
    Ok(())
}

#[cfg(feature = "ai-http")]
async fn read_http_request(
    listener: &tokio::net::TcpListener,
    key: &str,
) -> (String, serde_json::Value, tokio::net::TcpStream) {
    use tokio::io::{AsyncBufReadExt, AsyncReadExt, BufReader};

    let (socket, _) = listener.accept().await.unwrap();
    let mut reader = BufReader::new(socket);
    let mut line = String::new();
    reader.read_line(&mut line).await.unwrap();
    let endpoint = line
        .strip_prefix("POST ")
        .unwrap()
        .strip_suffix(" HTTP/1.1\r\n")
        .unwrap()
        .to_owned();
    let mut length = None;
    let mut authorization = None;
    loop {
        line.clear();
        assert_ne!(reader.read_line(&mut line).await.unwrap(), 0);
        if line == "\r\n" {
            break;
        }
        let (name, value) = line.split_once(':').unwrap();
        match name.to_ascii_lowercase().as_str() {
            "content-length" => length = Some(value.trim().parse::<usize>().unwrap()),
            "authorization" => authorization = Some(value.trim().to_owned()),
            _ => {}
        }
    }
    assert_eq!(authorization.unwrap(), format!("Bearer {key}"));
    let mut bytes = vec![0; length.unwrap()];
    reader.read_exact(&mut bytes).await.unwrap();
    (
        endpoint,
        serde_json::from_slice(&bytes).unwrap(),
        reader.into_inner(),
    )
}

#[cfg(feature = "ai-http")]
async fn write_http_response(mut socket: tokio::net::TcpStream, response: &str) {
    use tokio::io::AsyncWriteExt;

    let wire = format!(
        "HTTP/1.1 200 OK\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{response}",
        response.len(),
    );
    socket.write_all(wire.as_bytes()).await.unwrap();
    socket.shutdown().await.unwrap();
}
