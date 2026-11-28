// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: See the NOTICE file distributed with this work for additional information regarding copyright ownership
// SPDX-FileContributor: Licensed to the Apache Software Foundation (ASF) under one or more contributor license agreements

use std::collections::HashMap;
use std::sync::Arc;
use std::time::{Duration, SystemTime};

use arrow::array::StringArray;
use arrow::datatypes::{DataType, Field, Schema};
use arrow::record_batch::RecordBatch;
use datafusion_ai::ResolvedMedia;
use datafusion_ai::llm::{
    CancellationToken, LlmBackend, LlmBackendError, LlmExec, LlmOptions,
    LlmProviderRegistry, LlmRequest, LlmResponse, OpenAiHttpBackend, OpenAiHttpConfig,
    TaskKind,
};
use datafusion_common::{DataFusionError, Result};
use datafusion_execution::TaskContext;
use datafusion_execution::config::SessionConfig;
use datafusion_physical_expr::expressions::Column;
use datafusion_physical_plan::ExecutionPlan;
use datafusion_physical_plan::empty::EmptyExec;
use datafusion_physical_plan::test::TestMemoryExec;
use futures::TryStreamExt;
use serde_json::{Value, json};
use tokio::io::{AsyncBufReadExt, AsyncReadExt, AsyncWriteExt, BufReader};
use tokio::net::{TcpListener, TcpStream};
use tokio::sync::oneshot;
use tokio::time::{Instant, sleep, timeout};

const KEY: &str = "local-test-key";
const TEST_TIMEOUT: Duration = Duration::from_secs(5);
const SUCCESS: &str = r#"{"choices":[{"message":{"content":"generated"}}],"usage":{"prompt_tokens":17,"completion_tokens":4}}"#;

struct Capture {
    request_line: String,
    headers: HashMap<String, String>,
    body: Value,
    raw_body: Vec<u8>,
}

impl Capture {
    fn assert_request(&self, key: Option<&str>, expected: &Value) {
        assert_eq!(
            self.request_line,
            "POST /provider/v1/chat/completions HTTP/1.1"
        );
        assert_eq!(self.headers["content-type"], "application/json");
        let authorization = key.map(|key| format!("Bearer {key}"));
        assert_eq!(self.headers.get("authorization"), authorization.as_ref());
        assert_eq!(&self.body, expected);
    }
}

async fn read_request(listener: &TcpListener) -> (Capture, TcpStream) {
    let (socket, _) = listener.accept().await.unwrap();
    let mut reader = BufReader::new(socket);
    let mut request_line = String::new();
    reader.read_line(&mut request_line).await.unwrap();
    let mut headers = HashMap::new();
    loop {
        let mut line = String::new();
        assert_ne!(reader.read_line(&mut line).await.unwrap(), 0);
        if line == "\r\n" {
            break;
        }
        let (key, value) = line.split_once(':').unwrap();
        headers.insert(key.to_ascii_lowercase(), value.trim().to_owned());
    }

    let length = headers["content-length"].parse().unwrap();
    let mut body = vec![0; length];
    reader.read_exact(&mut body).await.unwrap();
    let capture = Capture {
        request_line: request_line.trim_end().to_owned(),
        headers,
        body: serde_json::from_slice(&body).unwrap(),
        raw_body: body,
    };
    (capture, reader.into_inner())
}

fn reply(status: &str, headers: &str, body: &str) -> String {
    format!(
        "HTTP/1.1 {status}\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n{headers}\r\n{body}",
        body.len()
    )
}

fn success() -> String {
    reply("200 OK", "", SUCCESS)
}

fn partial_reply(status: &str) -> String {
    format!(
        "HTTP/1.1 {status}\r\nContent-Type: application/json\r\nContent-Length: 1024\r\nConnection: close\r\n\r\n{{"
    )
}

fn config(base_url: String, key: Option<&str>) -> OpenAiHttpConfig {
    OpenAiHttpConfig {
        base_url,
        api_key: key.map(str::to_owned),
        api_key_env: None,
        ca_cert_pem: None,
    }
}

async fn bind(key: Option<&str>) -> (TcpListener, OpenAiHttpBackend) {
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let base_url = format!("http://{}/provider/v1/", listener.local_addr().unwrap());
    let backend = OpenAiHttpBackend::try_new(config(base_url, key)).unwrap();
    (listener, backend)
}

fn request(options: LlmOptions) -> LlmRequest {
    LlmRequest {
        model: "test-model".into(),
        prompts: vec!["prompt".into()],
        media: vec![None],
        options: Arc::new(options),
    }
}

#[tokio::test]
async fn embeddings_reorder_a_full_batch_and_reject_schema_mismatches() {
    let prompts: Vec<_> = (0..32).map(|index| format!("prompt {index}")).collect();
    let data: Vec<_> = (0..32)
        .rev()
        .map(|index| json!({"index": index, "embedding": [index, 0]}))
        .collect();
    let mut wrong_dim = data.clone();
    wrong_dim[0]["embedding"] = json!([31]);
    let mut duplicate = data.clone();
    duplicate[0]["index"] = json!(30);
    let mut out_of_range = data.clone();
    out_of_range[0]["index"] = json!(32);
    for (entries, valid) in [
        (&data[..], true),
        (&data[1..], false),
        (&wrong_dim[..], false),
        (&duplicate[..], false),
        (&out_of_range[..], false),
    ] {
        let (listener, backend) = bind(Some(KEY)).await;
        assert_eq!(backend.max_batch_size_for(TaskKind::Embed), Some(32));
        let options = LlmOptions {
            dimension: Some(2),
            ..Default::default()
        };
        let request = LlmRequest {
            prompts: prompts.clone(),
            media: vec![None; prompts.len()],
            ..request(options)
        };
        let (result, ()) = timeout(TEST_TIMEOUT, async {
            tokio::join!(backend.embed(request, CancellationToken::new()), async {
                let (capture, mut socket) = read_request(&listener).await;
                assert_eq!(
                    capture.request_line,
                    "POST /provider/v1/embeddings HTTP/1.1"
                );
                assert_eq!(capture.headers["authorization"], format!("Bearer {KEY}"));
                assert_eq!(
                    capture.body,
                    json!({"model": "test-model", "input": prompts})
                );
                let body = json!({"data": entries, "usage": {}}).to_string();
                socket
                    .write_all(reply("200 OK", "", &body).as_bytes())
                    .await
                    .unwrap();
            },)
        })
        .await
        .unwrap();
        if valid {
            let response = result.unwrap();
            assert_eq!(
                response.outputs,
                (0..32)
                    .map(|index| vec![index as f32, 0.0])
                    .collect::<Vec<_>>()
            );
            let usage = response.usage.unwrap();
            assert_eq!((usage.input_tokens, usage.output_tokens), (0, 0));
        } else {
            assert!(
                matches!(result, Err(LlmBackendError::SchemaMismatch { .. })),
                "{result:?}"
            );
        }
    }
}

#[tokio::test]
async fn vision_posts_text_and_ordered_inline_images_with_authorization() {
    let (listener, backend) = bind(Some(KEY)).await;
    let image = |bytes, mime: Option<&str>| ResolvedMedia {
        bytes: bytes::Bytes::from_static(bytes),
        mime: mime.map(str::to_owned),
        etag: None,
    };
    for (images, urls) in [
        (
            vec![image(b"\x00\xff", Some("image/png"))],
            vec!["data:image/png;base64,AP8="],
        ),
        (
            vec![image(b"\x00\xff", None)],
            vec!["data:application/octet-stream;base64,AP8="],
        ),
        (
            vec![
                image(b"a", Some("image/png")),
                image(b"b", None),
                image(b"c", Some("image/jpeg")),
            ],
            vec![
                "data:image/png;base64,YQ==",
                "data:application/octet-stream;base64,Yg==",
                "data:image/jpeg;base64,Yw==",
            ],
        ),
    ] {
        let request = LlmRequest {
            media: vec![Some(images)],
            ..request(LlmOptions::default())
        };
        let blocks = std::iter::once(json!({"type": "text", "text": "prompt"}))
            .chain(
                urls.into_iter()
                    .map(|url| json!({"type": "image_url", "image_url": {"url": url}})),
            )
            .collect::<Vec<_>>();
        let (result, ()) = timeout(TEST_TIMEOUT, async {
            tokio::join!(backend.generate(request, CancellationToken::new()), async {
                let (capture, mut socket) = read_request(&listener).await;
                capture.assert_request(
                    Some(KEY),
                    &json!({
                        "model": "test-model",
                        "messages": [{"role": "user", "content": blocks}],
                        "stream": false,
                        "n": 1,
                    }),
                );
                socket.write_all(success().as_bytes()).await.unwrap();
            },)
        })
        .await
        .unwrap();
        assert_eq!(result.unwrap().outputs, [Some("generated".to_owned())]);
    }
}

#[test]
fn invalid_custom_ca_pem_is_a_planning_error() {
    let error = OpenAiHttpBackend::try_new(OpenAiHttpConfig {
        ca_cert_pem: Some(b"private-marker".to_vec()),
        ..config("https://localhost/v1".into(), None)
    })
    .unwrap_err();
    assert!(
        matches!(error, DataFusionError::Plan(message) if message.contains("ca_cert_pem") && !message.contains("private-marker"))
    );
}

#[tokio::test]
#[ignore = "requires Python 3 and the OpenSSL CLI to serve loopback TLS"]
async fn custom_ca_controls_trust_of_the_loopback_server() {
    use std::io::{BufRead, Read};
    use std::process::{Command, Stdio};

    let mut server = Command::new("python3")
        .arg("-c")
        .arg(include_str!("tls_loopback.py"))
        .stdout(Stdio::piped())
        .spawn()
        .unwrap();
    let mut output = std::io::BufReader::new(server.stdout.take().unwrap());
    let mut header = String::new();
    output.read_line(&mut header).unwrap();
    let (port, pem_len) = header.trim_end().split_once(' ').unwrap();
    let mut pem = vec![0; pem_len.parse().unwrap()];
    output.read_exact(&mut pem).unwrap();
    let config = || config(format!("https://127.0.0.1:{port}/v1"), None);
    let untrusted = OpenAiHttpBackend::try_new(config()).unwrap();
    let trusted = OpenAiHttpBackend::try_new(OpenAiHttpConfig {
        ca_cert_pem: Some(pem),
        ..config()
    })
    .unwrap();
    let result = timeout(TEST_TIMEOUT, async {
        let untrusted = untrusted
            .generate(request(LlmOptions::default()), CancellationToken::new())
            .await;
        let trusted = trusted
            .generate(request(LlmOptions::default()), CancellationToken::new())
            .await;
        (untrusted, trusted)
    })
    .await;
    assert!(server.wait().unwrap().success());
    let (untrusted, trusted) = result.unwrap();
    assert!(
        matches!(untrusted, Err(LlmBackendError::Transient { .. })),
        "{untrusted:?}"
    );
    assert_eq!(trusted.unwrap().outputs, [Some("trusted".to_owned())]);
}

fn basic_body(prompt: &str) -> Value {
    json!({
        "model": "test-model",
        "messages": [{"role": "user", "content": prompt}],
        "stream": false,
        "n": 1
    })
}

async fn exchange(
    options: LlmOptions,
    key: Option<&str>,
    reply: &str,
    hold_open: bool,
) -> (std::result::Result<LlmResponse, LlmBackendError>, Capture) {
    let (listener, backend) = bind(key).await;
    let (finished, done) = oneshot::channel();
    timeout(TEST_TIMEOUT, async {
        tokio::join!(
            async {
                let result = backend
                    .generate(request(options), CancellationToken::new())
                    .await;
                if hold_open {
                    finished.send(()).unwrap();
                }
                result
            },
            async {
                let (capture, mut socket) = read_request(&listener).await;
                socket.write_all(reply.as_bytes()).await.unwrap();
                if hold_open {
                    done.await.unwrap();
                } else {
                    socket.shutdown().await.unwrap();
                }
                capture
            },
        )
    })
    .await
    .expect("client and loopback server must finish")
}

async fn attempt(reply: String) -> std::result::Result<LlmResponse, LlmBackendError> {
    let (result, capture) =
        exchange(LlmOptions::default(), Some(KEY), &reply, false).await;
    capture.assert_request(Some(KEY), &basic_body("prompt"));
    result
}

#[tokio::test]
async fn posts_single_prompt_and_maps_usage() {
    let (result, capture) =
        exchange(LlmOptions::default(), None, &success(), false).await;
    capture.assert_request(None, &basic_body("prompt"));
    let legacy_body = json!({
        "messages": [{"role": "user", "content": "prompt"}],
        "model": "test-model",
        "stream": false,
        "n": 1
    });
    assert_eq!(capture.raw_body, legacy_body.to_string().as_bytes());
    let response = result.unwrap();
    assert_eq!(response.outputs, [Some("generated".into())]);
    let usage = response.usage.unwrap();
    assert_eq!(usage.input_tokens, 17);
    assert_eq!(usage.output_tokens, 4);
}

#[tokio::test]
async fn posts_normalized_response_schema() {
    let options = serde_json::from_str(
        r#"{
            "response_schema": {
                "type":"object",
                "properties":{
                    "records":{"type":"array","items":{
                        "type":"object",
                        "properties":{
                            "score":{"type":"number"},
                            "token":{"type":"string","enum":["a","b"]}
                        },
                        "required":["score"]
                    }}
                }
            }
        }"#,
    )
    .unwrap();
    let (result, capture) = exchange(options, Some(KEY), &success(), false).await;
    assert!(result.is_ok());
    let mut expected = basic_body("prompt");
    expected["response_format"] = json!({
        "type":"json_schema",
        "json_schema":{
            "name":"df_response",
            "strict":true,
            "schema":{
                "type":"object",
                "additionalProperties":false,
                "required":["records"],
                "properties":{
                    "records":{"type":["array","null"],"items":{
                        "type":"object",
                        "additionalProperties":false,
                        "required":["score","token"],
                        "properties":{
                            "score":{"type":"number"},
                            "token":{"type":["string","null"],"enum":["a","b",null]}
                        }
                    }}
                }
            }
        }
    });
    capture.assert_request(Some(KEY), &expected);
}

#[tokio::test]
async fn rejects_conflicting_response_format_before_http() -> Result<()> {
    let (listener, backend) = bind(None).await;
    let registry = LlmProviderRegistry::default();
    registry.register("http", Arc::new(backend))?;
    let schema = Arc::new(Schema::new(vec![Field::new("text", DataType::Utf8, false)]));
    let error = LlmExec::try_new_with_registry(
        Arc::new(EmptyExec::new(schema)),
        Arc::new(Column::new("text", 0)),
        "test-model".into(),
        r#"{
            "provider":"http",
            "response_schema":{"type":"object","properties":{"text":{"type":"string"}}},
            "extra_body":{"response_format":{"type":"json_object"}}
        }"#,
        "answer".into(),
        &registry,
    )
    .unwrap_err();
    assert!(
        matches!(error, DataFusionError::Plan(message) if message.contains("response_schema") && message.contains("response_format"))
    );
    assert!(
        timeout(Duration::from_millis(20), listener.accept())
            .await
            .is_err()
    );
    Ok(())
}

#[tokio::test]
async fn defaults_missing_usage_fields_to_zero() {
    for (usage, input_tokens, output_tokens) in [
        (json!({"prompt_tokens": 3}), 3, 0),
        (json!({"completion_tokens": 2}), 0, 2),
        (json!({}), 0, 0),
    ] {
        let body = json!({
            "choices": [{"message": {"content": "generated"}}],
            "usage": usage,
        });
        let response = attempt(reply("200 OK", "", &body.to_string()))
            .await
            .unwrap();
        assert_eq!(response.outputs, [Some("generated".into())]);
        let usage = response.usage.unwrap();
        assert_eq!(usage.input_tokens, input_tokens);
        assert_eq!(usage.output_tokens, output_tokens);
    }
}

#[tokio::test]
async fn sends_system_message_and_generation_options() {
    let options = LlmOptions {
        system_message: Some("Be concise.".into()),
        temperature: Some(0.25),
        max_tokens: Some(12),
        top_p: Some(0.75),
        stop: Some(vec!["stop".into(), "end".into()]),
        ..Default::default()
    };
    let (result, capture) = exchange(options, Some(KEY), &success(), false).await;
    assert_eq!(result.unwrap().outputs, [Some("generated".into())]);
    let mut body = basic_body("prompt");
    body["messages"] = json!([
        {"role": "system", "content": "Be concise."},
        {"role": "user", "content": "prompt"}
    ]);
    body["temperature"] = json!(0.25);
    body["max_tokens"] = json!(12);
    body["top_p"] = json!(0.75);
    body["stop"] = json!(["stop", "end"]);
    capture.assert_request(Some(KEY), &body);
    let legacy_body = json!({
        "messages": [
            {"role": "system", "content": "Be concise."},
            {"role": "user", "content": "prompt"}
        ],
        "model": "test-model",
        "stream": false,
        "n": 1,
        "temperature": 0.25,
        "max_tokens": 12,
        "top_p": 0.75,
        "stop": ["stop", "end"]
    });
    assert_eq!(capture.raw_body, legacy_body.to_string().as_bytes());
}

#[tokio::test]
async fn expands_extra_body_with_fixed_fields_taking_precedence() {
    let options = LlmOptions {
        temperature: Some(0.25),
        extra_body: Some(
            json!({
                "vendor": {"seed": 15},
                "model": "overridden",
                "messages": [],
                "stream": true,
                "n": 8,
                "temperature": 2
            })
            .as_object()
            .unwrap()
            .clone(),
        ),
        ..Default::default()
    };
    let (result, capture) = exchange(options, Some(KEY), &success(), false).await;
    assert_eq!(result.unwrap().outputs, [Some("generated".into())]);
    let mut body = basic_body("prompt");
    body["vendor"] = json!({"seed": 15});
    body["temperature"] = json!(0.25);
    capture.assert_request(Some(KEY), &body);
}

#[tokio::test]
async fn rejects_nested_credential_keys_before_sending() {
    let (listener, backend) = bind(Some(KEY)).await;
    let options = LlmOptions {
        extra_body: Some(
            json!({"messages": [{"nested": {"API_KEY": "should-not-be-sent"}}]})
                .as_object()
                .unwrap()
                .clone(),
        ),
        ..Default::default()
    };
    let result = timeout(TEST_TIMEOUT, async {
        tokio::select! {
            biased;
            _ = listener.accept() => panic!("credential validation must precede HTTP"),
            result = backend.generate(request(options), CancellationToken::new()) => result,
        }
    })
    .await
    .unwrap();
    assert!(matches!(
        result,
        Err(LlmBackendError::Permanent { status: None, message })
            if message.contains("sensitive credential key")
    ));
}

#[tokio::test]
async fn parses_retry_after_seconds() {
    let error = attempt(reply("429 Too Many Requests", "Retry-After: 3\r\n", ""))
        .await
        .unwrap_err();
    assert!(matches!(
        error,
        LlmBackendError::RateLimited { retry_after: Some(delay), message }
            if delay == Duration::from_secs(3) && message == "HTTP 429"
    ));
}

#[tokio::test]
async fn parses_future_and_expired_retry_after_dates() {
    let target = (SystemTime::now() + Duration::from_secs(180))
        .duration_since(SystemTime::UNIX_EPOCH)
        .unwrap()
        .as_secs();
    let target = SystemTime::UNIX_EPOCH + Duration::from_secs(target);
    let header = format!("Retry-After: {}\r\n", httpdate::fmt_http_date(target));
    let before = SystemTime::now();
    let error = attempt(reply("429 Too Many Requests", &header, ""))
        .await
        .unwrap_err();
    let after = SystemTime::now();
    let LlmBackendError::RateLimited {
        retry_after: Some(delay),
        ..
    } = error
    else {
        panic!("expected a rate limit with a delay, got {error:?}");
    };
    assert!(
        (target.duration_since(after).unwrap()..=target.duration_since(before).unwrap())
            .contains(&delay)
    );

    assert!(matches!(
        attempt(reply(
            "429 Too Many Requests",
            "Retry-After: Sun, 06 Nov 1994 08:49:37 GMT\r\n",
            ""
        ))
        .await,
        Err(LlmBackendError::RateLimited {
            retry_after: Some(Duration::ZERO),
            ..
        })
    ));
}

#[tokio::test]
async fn preserves_missing_retry_after() {
    assert!(matches!(
        attempt(reply("429 Too Many Requests", "", "")).await,
        Err(LlmBackendError::RateLimited {
            retry_after: None,
            ..
        })
    ));
}

#[tokio::test]
async fn reports_503_with_single_line_utf8_detail() {
    let detail = format!("device\r\n\t\u{202e} busy {}", "界".repeat(100));
    let body = json!({"error": {"message": detail}}).to_string();
    let error = attempt(reply("503 Service Unavailable", "", &body))
        .await
        .unwrap_err();
    let LlmBackendError::Transient {
        status: Some(503),
        message,
    } = error
    else {
        panic!("expected a transient 503, got {error:?}");
    };
    let detail = message
        .strip_prefix("chat completions request failed: ")
        .unwrap();
    assert_eq!(detail, format!("device busy {}", "界".repeat(81)));
    assert!(detail.len() <= 256);
}

#[tokio::test]
async fn classifies_408_as_transient() {
    assert!(matches!(
        attempt(reply("408 Request Timeout", "", "")).await,
        Err(LlmBackendError::Transient {
            status: Some(408),
            ..
        })
    ));
}

#[tokio::test]
async fn classifies_401_as_permanent() {
    let error = attempt(reply(
        "401 Unauthorized",
        "",
        r#"{"message":"permission denied"}"#,
    ))
    .await
    .unwrap_err();
    assert!(!error.to_string().contains(KEY));
    assert!(matches!(
        error,
        LlmBackendError::Permanent { status: Some(401), message }
            if message == "chat completions request rejected: permission denied"
    ));
}

#[tokio::test]
async fn redacts_echoed_credentials_and_normalized_forms() {
    let key = "local  echo-key";
    let body = json!({
        "error": {
            "message": format!("provider echoed Bearer {key}\nlocal \u{202e} echo-key")
        }
    })
    .to_string();
    let (result, capture) = exchange(
        LlmOptions::default(),
        Some(key),
        &reply("401 Unauthorized", "", &body),
        false,
    )
    .await;
    capture.assert_request(Some(key), &basic_body("prompt"));
    let error = result.unwrap_err();
    let message = error.to_string();
    assert!(!message.contains(key));
    assert!(!message.contains("local echo-key"));
    assert!(matches!(
        error,
        LlmBackendError::Permanent { status: Some(401), message }
            if message == "chat completions request rejected: provider echoed Bearer [REDACTED] [REDACTED]"
    ));

    let key = "[REDACTED]";
    let (result, capture) = exchange(
        LlmOptions::default(),
        Some(key),
        &reply("401 Unauthorized", "", r#"{"message":"echo [REDACTED]"}"#),
        false,
    )
    .await;
    capture.assert_request(Some(key), &basic_body("prompt"));
    assert!(matches!(
        result,
        Err(LlmBackendError::Permanent { status: Some(401), message })
            if message == "chat completions request rejected"
    ));
}

#[tokio::test]
async fn omits_urls_in_provider_diagnostics() {
    let error = attempt(reply(
        "400 Bad Request",
        "",
        r#"{"error":{"message":"failed at http://localhost/v1/chat/completions?tenant=private"}}"#,
    ))
    .await
    .unwrap_err();
    assert!(matches!(
        error,
        LlmBackendError::Permanent { status: Some(400), message }
            if message == "chat completions request rejected"
    ));
}

#[tokio::test]
async fn falls_back_on_oversized_invalid_and_stalled_error_bodies() {
    let oversized = format!(r#"{{"message":"busy"}}{}"#, " ".repeat(8192));
    for body in [&oversized, "<html>unavailable</html>"] {
        let error = attempt(reply("503 Service Unavailable", "", body))
            .await
            .unwrap_err();
        assert!(matches!(
            error,
            LlmBackendError::Transient { status: Some(503), message }
                if message == "chat completions request failed"
        ));
    }

    let started = Instant::now();
    let (result, capture) = timeout(
        Duration::from_secs(1),
        exchange(
            LlmOptions::default(),
            Some(KEY),
            &partial_reply("503 Service Unavailable"),
            true,
        ),
    )
    .await
    .expect("error body read must finish while the provider holds the socket open");
    assert!(started.elapsed() >= Duration::from_millis(250));
    capture.assert_request(Some(KEY), &basic_body("prompt"));
    assert!(matches!(
        result,
        Err(LlmBackendError::Transient { status: Some(503), message })
            if message == "chat completions request failed"
    ));
}

#[tokio::test]
async fn cancellation_interrupts_a_live_request() {
    let (listener, backend) = bind(Some(KEY)).await;
    let cancel = CancellationToken::new();
    let hold = async {
        let (capture, _socket) = read_request(&listener).await;
        capture.assert_request(Some(KEY), &basic_body("prompt"));
        cancel.cancel();
        sleep(Duration::from_secs(5)).await;
    };

    let result = timeout(Duration::from_secs(1), async {
        tokio::select! {
            _ = hold => panic!("provider should still be waiting"),
            result = backend.generate(request(LlmOptions::default()), cancel.clone()) => result,
        }
    })
    .await
    .unwrap();
    assert!(cancel.is_cancelled());
    assert!(matches!(result, Err(LlmBackendError::Cancelled)));
}

#[tokio::test]
async fn classifies_a_truncated_response_as_a_transport_error() {
    assert!(matches!(
        attempt(partial_reply("200 OK")).await,
        Err(LlmBackendError::Transient { status: None, .. })
    ));
}

#[tokio::test]
async fn rejects_an_invalid_json_response() {
    assert!(matches!(
        attempt(reply("200 OK", "", "{oops")).await,
        Err(LlmBackendError::Permanent { status: Some(200), message })
            if message == "invalid chat completions response"
    ));
}

#[tokio::test]
async fn rejects_missing_or_empty_choices() {
    for body in [r#"{"choices":[]}"#, "{}"] {
        assert!(matches!(
            attempt(reply("200 OK", "", body)).await,
            Err(LlmBackendError::Permanent { message, .. })
                if message.contains("no choices")
        ));
    }
}

#[tokio::test]
async fn rejects_more_than_one_choice() {
    assert!(matches!(
        attempt(reply(
            "200 OK",
            "",
            r#"{"choices":[{"message":{"content":"one"}},{"message":{"content":"two"}}]}"#
        ))
        .await,
        Err(LlmBackendError::Permanent { message, .. })
            if message.contains("exactly one choice")
    ));
}

#[tokio::test]
async fn maps_null_content_to_null_output() {
    let response = attempt(reply(
        "200 OK",
        "",
        r#"{"choices":[{"message":{"content":null}}]}"#,
    ))
    .await
    .unwrap();
    assert_eq!(response.outputs, [None]);
    assert!(response.usage.is_none());
}

#[tokio::test]
async fn rejects_redirects() {
    assert!(matches!(
        attempt(reply("307 Temporary Redirect", "Location: /other\r\n", "")).await,
        Err(LlmBackendError::Permanent { status: Some(307), message })
            if message == "chat completions request rejected"
    ));
}

#[test]
fn rejects_invalid_base_urls() {
    for base in [
        "ftp://localhost/v1",
        "https:localhost/v1",
        "http://",
        "http://user:private-marker@localhost/v1",
        "http://localhost/v1#private-marker",
        "http://localhost/v1?api_key=private-marker",
        "http://localhost/v1?%61pi%5FKey=private-marker",
        "http://localhost/v1?access-token=private-marker",
    ] {
        let error = OpenAiHttpBackend::try_new(config(base.into(), None)).unwrap_err();
        assert!(!error.to_string().contains("private-marker"));
        assert!(matches!(error, DataFusionError::Plan(_)), "{base}");
    }
}

#[test]
fn rejects_conflicting_empty_invalid_and_missing_credentials() {
    let base = "http://localhost/v1";
    let missing = format!("DATAFUSION_HTTP_TEST_MISSING_KEY_{}", std::process::id());
    assert!(std::env::var_os(&missing).is_none());
    for (key, env, expected) in [
        (Some(KEY), Some(missing.as_str()), "Set only one"),
        (Some(" \t"), None, "must be non-empty"),
        (Some("broken\nkey"), None, "Invalid llm HTTP API key"),
        (None, Some(missing.as_str()), "Could not read"),
    ] {
        let error = OpenAiHttpBackend::try_new(OpenAiHttpConfig {
            api_key_env: env.map(str::to_owned),
            ..config(base.into(), key)
        })
        .unwrap_err();
        assert!(matches!(
            error,
            DataFusionError::Plan(message) if message.contains(expected)
        ));
    }
}

#[test]
fn debug_omits_configuration_fields() {
    let config = config(
        "http://localhost/v1?tenant=private-marker".into(),
        Some(KEY),
    );
    assert_eq!(format!("{config:?}"), "OpenAiHttpConfig { .. }");
    let backend = OpenAiHttpBackend::try_new(config).unwrap();
    assert_eq!(format!("{backend:?}"), "OpenAiHttpBackend { .. }");
}

#[tokio::test]
async fn registered_backend_executes_three_rows_after_a_rate_limit() -> Result<()> {
    let (listener, backend) = bind(Some(KEY)).await;
    let registry = LlmProviderRegistry::default();
    registry.register_with_limit("http", Arc::new(backend), 2)?;
    let ctx = Arc::new(
        TaskContext::default()
            .with_session_config(SessionConfig::new().with_extension(Arc::new(registry))),
    );

    let prompts = ["one", "two", "three"];
    let schema = Arc::new(Schema::new(vec![Field::new("text", DataType::Utf8, false)]));
    let batch = RecordBatch::try_new(
        Arc::clone(&schema),
        vec![Arc::new(StringArray::from(Vec::from(prompts)))],
    )?;
    let input = TestMemoryExec::try_new_exec(&[vec![batch]], schema, None)?;
    let exec = LlmExec::try_new_from_task_context(
        input,
        Arc::new(Column::new("text", 0)),
        "test-model".into(),
        r#"{"provider":"http","inflight_limit":1,"max_retries":1}"#,
        "answer".into(),
        &ctx,
    )?;

    let serve = async {
        for (attempt, prompt) in ["one", "one", "two", "three"].into_iter().enumerate() {
            let (capture, mut socket) = read_request(&listener).await;
            capture.assert_request(Some(KEY), &basic_body(prompt));
            let response = if attempt == 0 {
                reply("429 Too Many Requests", "Retry-After: 0\r\n", "")
            } else {
                let body = json!({
                    "choices": [{"message": {"content": format!("answer: {prompt}")}}]
                });
                reply("200 OK", "", &body.to_string())
            };
            socket.write_all(response.as_bytes()).await.unwrap();
            socket.shutdown().await.unwrap();
        }
    };

    let output = exec.execute(0, ctx)?;
    let (batches, ()) = timeout(TEST_TIMEOUT, async {
        tokio::join!(output.try_collect::<Vec<_>>(), serve)
    })
    .await
    .unwrap();
    let batches = batches?;
    let actual = batches
        .iter()
        .flat_map(|batch| {
            let text = batch
                .column(0)
                .as_any()
                .downcast_ref::<StringArray>()
                .unwrap();
            let answer = batch
                .column(1)
                .as_any()
                .downcast_ref::<StringArray>()
                .unwrap();
            text.iter().zip(answer.iter())
        })
        .collect::<Vec<_>>();
    assert_eq!(
        actual,
        [
            (Some("one"), Some("answer: one")),
            (Some("two"), Some("answer: two")),
            (Some("three"), Some("answer: three")),
        ]
    );
    Ok(())
}
