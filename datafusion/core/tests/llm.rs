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
    use tokio::io::{AsyncBufReadExt, AsyncReadExt, AsyncWriteExt, BufReader};
    use tokio::net::TcpListener;

    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let backend = OpenAiHttpBackend::try_new(OpenAiHttpConfig {
        base_url: format!("http://{}/v1", listener.local_addr().unwrap()),
        api_key: Some("loopback-sql-test-key".into()),
        api_key_env: None,
    })?;
    let ctx = harness::test_context(8, 1);
    ctx.state().llm_provider_registry().register_with_limit(
        "http",
        Arc::new(backend),
        2,
    )?;
    harness::register_text(&ctx, "t", &[Some("sql prompt")])?;

    let serve = async {
        let (socket, _) = listener.accept().await.unwrap();
        let mut reader = BufReader::new(socket);
        let mut line = String::new();
        reader.read_line(&mut line).await.unwrap();
        assert_eq!(line, "POST /v1/chat/completions HTTP/1.1\r\n");
        let mut length = None;
        let mut bearer = false;
        loop {
            line.clear();
            assert_ne!(reader.read_line(&mut line).await.unwrap(), 0);
            if line == "\r\n" {
                break;
            }
            let (name, value) = line.split_once(':').unwrap();
            match name.to_ascii_lowercase().as_str() {
                "content-length" => length = Some(value.trim().parse::<usize>().unwrap()),
                "authorization" => {
                    bearer = value.trim() == "Bearer loopback-sql-test-key"
                }
                _ => {}
            }
        }
        assert!(bearer);
        let mut bytes = vec![0; length.unwrap()];
        reader.read_exact(&mut bytes).await.unwrap();
        let body: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
        assert_eq!(body["model"], "test-model");
        assert_eq!(body["messages"][0]["content"], "sql prompt");
        assert_eq!(body["stream"], false);
        assert_eq!(body["n"], 1);
        let body = r#"{"choices":[{"message":{"content":"via HTTP"}}]}"#;
        let wire = format!(
            "HTTP/1.1 200 OK\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}",
            body.len()
        );
        let mut socket = reader.into_inner();
        socket.write_all(wire.as_bytes()).await.unwrap();
        socket.shutdown().await.unwrap();
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
