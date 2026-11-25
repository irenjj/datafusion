// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: See the NOTICE file distributed with this work for additional information regarding copyright ownership
// SPDX-FileContributor: Licensed to the Apache Software Foundation (ASF) under one or more contributor license agreements

use std::collections::HashMap;
use std::sync::Arc;
use std::time::Duration;

use datafusion::common::{DataFusionError, Result};
use datafusion::llm::{OpenAiHttpBackend, OpenAiHttpConfig};
use datafusion::physical_plan::{ExecutionPlan, collect};
use datafusion::prelude::SessionContext;
use datafusion_common_runtime::SpawnedTask;
use serde_json::{Value, json};
use tokio::io::{self, AsyncBufReadExt, AsyncReadExt, AsyncWriteExt, BufReader};
use tokio::net::{TcpListener, TcpStream};

use super::harness::{metric, run, strings, test_context};

const SUCCESS: &str = r#"{"choices":[{"message":{"content":"generated"}}],"usage":{"prompt_tokens":17,"completion_tokens":4}}"#;

struct Capture {
    request_line: String,
    headers: HashMap<String, String>,
    body: Value,
}

enum Reply {
    Status(&'static str, &'static str, String),
    Hang,
}

impl Reply {
    fn status(
        status: &'static str,
        headers: &'static str,
        body: impl Into<String>,
    ) -> Self {
        Self::Status(status, headers, body.into())
    }

    fn success() -> Self {
        Self::status("200 OK", "", SUCCESS)
    }
}

struct MockServer {
    base_url: String,
    task: SpawnedTask<io::Result<Vec<Capture>>>,
}

impl MockServer {
    async fn start(replies: Vec<Reply>) -> io::Result<Self> {
        let listener = TcpListener::bind("127.0.0.1:0").await?;
        let base_url = format!("http://{}/v1/", listener.local_addr()?);
        let task = SpawnedTask::spawn(async move {
            let mut captured = Vec::new();
            let mut held_open = Vec::new();
            for reply in replies {
                let (socket, _) = listener.accept().await?;
                let (request, mut socket) = read_request(socket).await?;
                captured.push(request);
                match reply {
                    Reply::Status(status, headers, body) => {
                        let response = format!(
                            "HTTP/1.1 {status}\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n{headers}\r\n{body}",
                            body.len()
                        );
                        socket.write_all(response.as_bytes()).await?;
                        socket.shutdown().await?;
                    }
                    Reply::Hang => held_open.push(socket),
                }
            }
            Ok(captured)
        });
        Ok(Self { base_url, task })
    }

    async fn captures(self) -> Vec<Capture> {
        tokio::time::timeout(Duration::from_secs(5), self.task.join_unwind())
            .await
            .unwrap()
            .unwrap()
            .unwrap()
    }

    fn context(&self, key: Option<&str>) -> Result<SessionContext> {
        let ctx = test_context(8, 1);
        let backend = OpenAiHttpBackend::try_new(OpenAiHttpConfig {
            base_url: self.base_url.clone(),
            api_key: key.map(str::to_owned),
            api_key_env: None,
        })?;
        ctx.register_llm_provider("http", Arc::new(backend))?;
        Ok(ctx)
    }
}

async fn read_request(socket: TcpStream) -> io::Result<(Capture, TcpStream)> {
    let mut reader = BufReader::new(socket);
    let mut request_line = String::new();
    reader.read_line(&mut request_line).await?;
    let mut headers = HashMap::new();
    loop {
        let mut line = String::new();
        if reader.read_line(&mut line).await? == 0 {
            return Err(io::ErrorKind::UnexpectedEof.into());
        }
        if line == "\r\n" {
            break;
        }
        let (key, value) = line.split_once(':').unwrap();
        headers.insert(key.to_ascii_lowercase(), value.trim().to_owned());
    }
    let length = headers["content-length"].parse().unwrap();
    let mut bytes = vec![0; length];
    reader.read_exact(&mut bytes).await?;
    let capture = Capture {
        request_line: request_line.trim_end().to_owned(),
        headers,
        body: serde_json::from_slice(&bytes).unwrap(),
    };
    Ok((capture, reader.into_inner()))
}

fn assert_attempts(plan: &dyn ExecutionPlan, attempts: usize) {
    assert_eq!(metric(plan, "submitted_prompts"), 1);
    assert_eq!(metric(plan, "backend_attempts"), attempts);
    assert_eq!(metric(plan, "retries"), attempts - 1);
}

#[tokio::test]
async fn chat_request_and_token_usage() -> Result<()> {
    let server = MockServer::start(vec![Reply::success()]).await?;
    let ctx = server.context(Some("runtime-test-key"))?;
    let (plan, batches) = run(
        &ctx,
        r#"
        SELECT llm('question', 'served-model', '{
            "provider":"http", "system_message":"instructions", "temperature":0.2,
            "top_p":0.9, "max_tokens":20, "stop":["done"],
            "extra_body":{"top_k":4,"chat_template_kwargs":{"enable_thinking":false},
                          "stream":true,"n":2,"model":"other"}
        }')
    "#,
    )
    .await?;
    assert_eq!(strings(&batches, 0), [Some("generated")]);
    assert_attempts(plan.as_ref(), 1);
    assert_eq!(metric(plan.as_ref(), "input_tokens"), 17);
    assert_eq!(metric(plan.as_ref(), "output_tokens"), 4);

    let captures = server.captures().await;
    let [request] = captures.as_slice() else {
        panic!("expected one HTTP request")
    };
    assert_eq!(request.request_line, "POST /v1/chat/completions HTTP/1.1");
    assert_eq!(request.headers["authorization"], "Bearer runtime-test-key");
    assert_eq!(
        request.body,
        json!({
            "model":"served-model", "stream":false, "n":1,
            "messages":[
                {"role":"system","content":"instructions"},
                {"role":"user","content":"question"}
            ],
            "temperature":0.2, "top_p":0.9, "max_tokens":20, "stop":["done"],
            "top_k":4, "chat_template_kwargs":{"enable_thinking":false}
        })
    );
    Ok(())
}

#[tokio::test]
async fn rate_limited_http_request_retries() -> Result<()> {
    let server = MockServer::start(vec![
        Reply::status("429 Too Many Requests", "Retry-After: 0\r\n", "{}"),
        Reply::success(),
    ])
    .await?;
    let ctx = server.context(None)?;
    let (plan, batches) = run(
        &ctx,
        r#"SELECT llm('question', 'm', '{"provider":"http","max_retries":1}')"#,
    )
    .await?;
    assert_eq!(strings(&batches, 0), [Some("generated")]);
    assert_attempts(plan.as_ref(), 2);
    assert_eq!(server.captures().await.len(), 2);
    Ok(())
}

#[tokio::test]
async fn server_errors_exhaust_retries_and_return_null() -> Result<()> {
    let server = MockServer::start(vec![
        Reply::status("500 Internal Server Error", "", "{}"),
        Reply::status("500 Internal Server Error", "", "{}"),
    ])
    .await?;
    let ctx = server.context(None)?;
    let (plan, batches) = run(&ctx, r#"SELECT llm('question', 'm', '{"provider":"http","on_error":"ignore","max_retries":1}')"#).await?;
    assert_eq!(strings(&batches, 0), [None]);
    assert_attempts(plan.as_ref(), 2);
    assert_eq!(metric(plan.as_ref(), "failed_rows"), 1);
    assert_eq!(server.captures().await.len(), 2);
    Ok(())
}

#[tokio::test]
async fn bad_request_is_permanent_and_hides_response_body() -> Result<()> {
    let oversized = format!(
        r#"{{"padding":"{}","error":{{"message":"private-server-body"}}}}"#,
        "x".repeat(8 * 1024)
    );
    for body in ["private-server-body".to_owned(), oversized] {
        let server =
            MockServer::start(vec![Reply::status("400 Bad Request", "", body)]).await?;
        let ctx = server.context(None)?;
        let plan = ctx
            .sql(r#"SELECT llm('question', 'm', '{"provider":"http","max_retries":2}')"#)
            .await?
            .create_physical_plan()
            .await?;
        let error = collect(Arc::clone(&plan), ctx.task_ctx())
            .await
            .unwrap_err();
        assert!(matches!(
            error.find_root(),
            DataFusionError::Execution(message)
                if message == "llm backend failed: HTTP 400: chat completions request rejected"
        ));
        assert_attempts(plan.as_ref(), 1);
        assert_eq!(server.captures().await.len(), 1);
    }
    Ok(())
}

#[tokio::test]
async fn bad_request_reports_sanitized_bounded_reason_without_echoed_credentials()
-> Result<()> {
    let key = "runtime-test-key";
    let reason = format!(
        "model 'bad' not found\n\r\tcredential: Bearer {key}\u{0}\u{202e} {} should be truncated",
        "界".repeat(200)
    );
    let server = MockServer::start(vec![Reply::status(
        "400 Bad Request",
        "",
        json!({"error": {"message": reason}}).to_string(),
    )])
    .await?;
    let ctx = server.context(Some(key))?;
    let plan = ctx
        .sql(r#"SELECT llm('question', 'm', '{"provider":"http","max_retries":2}')"#)
        .await?
        .create_physical_plan()
        .await?;
    let error = collect(Arc::clone(&plan), ctx.task_ctx())
        .await
        .unwrap_err();
    let DataFusionError::Execution(message) = error.find_root() else {
        panic!("expected execution error");
    };
    let detail = message
        .strip_prefix("llm backend failed: HTTP 400: chat completions request rejected: ")
        .expect("expected the server reason in the error");
    let suffix = detail
        .strip_prefix("model 'bad' not found credential: Bearer [REDACTED] ")
        .expect("expected a single line with the key redacted");
    assert!((254..=256).contains(&detail.len()));
    assert!(suffix.chars().all(|ch| ch == '界'));
    assert!(!message.contains(key));
    assert_attempts(plan.as_ref(), 1);
    assert_eq!(server.captures().await.len(), 1);
    Ok(())
}

#[tokio::test]
async fn hanging_server_times_out_then_retries() -> Result<()> {
    let server = MockServer::start(vec![Reply::Hang, Reply::success()]).await?;
    let ctx = server.context(None)?;
    let (plan, batches) = run(&ctx, r#"SELECT llm('question', 'm', '{"provider":"http","timeout_ms":300,"max_retries":1}')"#).await?;
    assert_eq!(strings(&batches, 0), [Some("generated")]);
    assert_attempts(plan.as_ref(), 2);
    let captures = server.captures().await;
    assert_eq!(captures.len(), 2);
    assert_eq!(captures[0].body, captures[1].body);
    Ok(())
}

#[tokio::test]
async fn chat_api_rejects_multiple_prompts_during_planning() -> Result<()> {
    let server = MockServer::start(vec![]).await?;
    let ctx = server.context(None)?;
    let error = ctx
        .sql(r#"SELECT llm('question', 'm', '{"provider":"http","batch_size":2}')"#)
        .await?
        .create_physical_plan()
        .await
        .unwrap_err();
    assert!(
        matches!(error.find_root(), DataFusionError::Plan(_)),
        "{error}"
    );
    assert!(error.to_string().contains("at most 1 prompt"), "{error}");
    assert!(server.captures().await.is_empty());
    Ok(())
}
