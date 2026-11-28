// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: See the NOTICE file distributed with this work for additional information regarding copyright ownership
// SPDX-FileContributor: Licensed to the Apache Software Foundation (ASF) under one or more contributor license agreements

use std::fmt;
use std::time::{Duration, SystemTime};

use async_trait::async_trait;
use datafusion_common::{Result, plan_datafusion_err, plan_err};
use reqwest::header::{AUTHORIZATION, HeaderValue, RETRY_AFTER};
use reqwest::{Client, Response, StatusCode, Url};
use serde::de::DeserializeOwned;
use serde::{Deserialize, Serialize};
use serde_json::{Map, Value, json};

use super::options::is_sensitive_key;
use super::{
    CancellationToken, LlmBackend, LlmBackendError, LlmRequest, LlmResponse, LlmUsage,
};

/// Runtime configuration for an OpenAI-compatible HTTP endpoint.
pub struct OpenAiHttpConfig {
    /// Provider root including any API path prefix, e.g. "http://host:8000/v1".
    pub base_url: String,
    /// Bearer token. Mutually exclusive with [Self::api_key_env].
    pub api_key: Option<String>,
    /// Environment variable read once at construction. Mutually exclusive with
    /// [Self::api_key].
    pub api_key_env: Option<String>,
}

impl fmt::Debug for OpenAiHttpConfig {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("OpenAiHttpConfig").finish_non_exhaustive()
    }
}

/// A backend for OpenAI-compatible HTTP chat completions.
///
/// Register it in the [`LlmProviderRegistry`](super::LlmProviderRegistry) and select
/// that provider in the options for `llm()`.
pub struct OpenAiHttpBackend {
    transport: Transport,
}

impl fmt::Debug for OpenAiHttpBackend {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("OpenAiHttpBackend").finish_non_exhaustive()
    }
}

impl OpenAiHttpBackend {
    /// Builds the backend and validates its provider root and credentials.
    ///
    /// ```
    /// use std::sync::Arc;
    /// use datafusion_ai::llm::{LlmProviderRegistry, OpenAiHttpBackend, OpenAiHttpConfig};
    ///
    /// let backend = OpenAiHttpBackend::try_new(OpenAiHttpConfig {
    ///     base_url: "http://localhost:8000/v1".into(),
    ///     api_key: None,
    ///     api_key_env: None,
    /// })?;
    /// LlmProviderRegistry::default().register("local", Arc::new(backend))?;
    /// # Ok::<(), datafusion_common::DataFusionError>(())
    /// ```
    pub fn try_new(config: OpenAiHttpConfig) -> Result<Self> {
        Ok(Self {
            transport: Transport::try_new(config)?,
        })
    }
}

#[async_trait]
impl LlmBackend for OpenAiHttpBackend {
    fn max_batch_size(&self) -> Option<usize> {
        Some(1)
    }

    async fn generate(
        &self,
        request: LlmRequest,
        cancel: CancellationToken,
    ) -> std::result::Result<LlmResponse, LlmBackendError> {
        if cancel.is_cancelled() {
            return Err(LlmBackendError::Cancelled);
        }
        let body = build_body(request)?;
        let reply: ChatResponse = self
            .transport
            .post_json(&["chat", "completions"], &body, &cancel)
            .await?;
        reply.into_response()
    }
}

fn build_body(
    request: LlmRequest,
) -> std::result::Result<Map<String, Value>, LlmBackendError> {
    let [prompt] = request.prompts.as_slice() else {
        return Err(permanent(
            None,
            "chat completions require exactly one prompt per request",
        ));
    };
    let options = &request.options;
    let mut body = options.extra_body.clone().unwrap_or_default();
    if contains_credential_key(&body) {
        return Err(permanent(
            None,
            "extra_body contains a sensitive credential key",
        ));
    }

    let mut messages = Vec::new();
    if let Some(system) = &options.system_message {
        messages.push(json!({"role": "system", "content": system}));
    }
    messages.push(json!({"role": "user", "content": prompt}));
    body.insert("messages".into(), Value::Array(messages));
    body.insert("model".into(), Value::String(request.model));
    body.insert("stream".into(), Value::Bool(false));
    body.insert("n".into(), json!(1));
    for (key, value) in [
        ("temperature", json!(options.temperature)),
        ("max_tokens", json!(options.max_tokens)),
        ("top_p", json!(options.top_p)),
        ("stop", json!(options.stop)),
    ] {
        if !value.is_null() {
            body.insert(key.into(), value);
        }
    }
    Ok(body)
}

fn contains_credential_key(object: &Map<String, Value>) -> bool {
    fn nested(value: &Value) -> bool {
        match value {
            Value::Object(object) => contains_credential_key(object),
            Value::Array(values) => values.iter().any(nested),
            _ => false,
        }
    }
    object
        .iter()
        .any(|(key, value)| is_sensitive_key(key) || nested(value))
}

#[derive(Deserialize)]
struct ChatResponse {
    #[serde(default)]
    choices: Vec<ChatChoice>,
    usage: Option<ChatUsage>,
}

impl ChatResponse {
    fn into_response(self) -> std::result::Result<LlmResponse, LlmBackendError> {
        let mut choices = self.choices.into_iter();
        let choice = choices
            .next()
            .ok_or_else(|| permanent(None, "chat completions response has no choices"))?;
        if choices.len() != 0 {
            return Err(permanent(
                None,
                "chat completions response must contain exactly one choice",
            ));
        }
        Ok(LlmResponse {
            outputs: vec![choice.message.content],
            usage: self.usage.map(|usage| LlmUsage {
                input_tokens: usage.prompt_tokens,
                output_tokens: usage.completion_tokens,
            }),
        })
    }
}

#[derive(Deserialize)]
struct ChatChoice {
    message: ChatMessage,
}

#[derive(Deserialize)]
struct ChatMessage {
    content: Option<String>,
}

#[derive(Deserialize)]
struct ChatUsage {
    #[serde(default)]
    prompt_tokens: u64,
    #[serde(default)]
    completion_tokens: u64,
}

struct Transport {
    client: Client,
    base: Url,
    authorization: Option<HeaderValue>,
}

impl Transport {
    fn try_new(config: OpenAiHttpConfig) -> Result<Self> {
        let base = parse_base_url(&config.base_url)?;
        let key = match (config.api_key, config.api_key_env) {
            (Some(_), Some(_)) => {
                return plan_err!(
                    "Set only one of api_key and api_key_env for the llm HTTP backend"
                );
            }
            (Some(key), None) => Some(key),
            (None, Some(env)) => Some(std::env::var(env).map_err(|_| {
                plan_datafusion_err!(
                    "Could not read the configured llm HTTP API key environment variable"
                )
            })?),
            (None, None) => None,
        };
        let authorization = key.as_deref().map(bearer_header).transpose()?;

        // A redirect can send credentials or inference data to a different endpoint.
        let client = Client::builder()
            .use_rustls_tls()
            .redirect(reqwest::redirect::Policy::none())
            .retry(reqwest::retry::never())
            .build()
            .map_err(|_| plan_datafusion_err!("Could not build the llm HTTP client"))?;
        Ok(Self {
            client,
            base,
            authorization,
        })
    }

    async fn post_json<B: Serialize, R: DeserializeOwned>(
        &self,
        path: &[&str],
        body: &B,
        cancel: &CancellationToken,
    ) -> std::result::Result<R, LlmBackendError> {
        tokio::select! {
            biased;
            _ = cancel.cancelled() => Err(LlmBackendError::Cancelled),
            result = async {
                let mut endpoint = self.base.clone();
                endpoint
                    .path_segments_mut()
                    .map_err(|_| permanent(None, "invalid llm HTTP endpoint path"))?
                    .pop_if_empty()
                    .extend(path.iter().copied());

                let mut request = self.client.post(endpoint).json(body);
                if let Some(authorization) = &self.authorization {
                    request = request.header(AUTHORIZATION, authorization.clone());
                }
                let response = request.send().await.map_err(|error| transport_error(&error, path))?;
                let status = response.status();
                if !status.is_success() {
                    return Err(response_error(response, self.authorization.as_ref(), path).await);
                }
                let bytes = response.bytes().await.map_err(|error| transport_error(&error, path))?;
                serde_json::from_slice(&bytes).map_err(|_| {
                    permanent(Some(status.as_u16()), &format!("invalid {} response", path.join(" ")))
                })
            } => result,
        }
    }
}

fn parse_base_url(base_url: &str) -> Result<Url> {
    if !base_url.split_once("://").is_some_and(|(scheme, _)| {
        scheme.eq_ignore_ascii_case("http") || scheme.eq_ignore_ascii_case("https")
    }) {
        return plan_err!("llm HTTP base_url must begin with http:// or https://");
    }
    let base = Url::parse(base_url)
        .map_err(|_| plan_datafusion_err!("Invalid llm HTTP base_url"))?;
    if !matches!(base.scheme(), "http" | "https")
        || base.host_str().is_none()
        || !base.username().is_empty()
        || base.password().is_some()
        || base.fragment().is_some()
        || base.query_pairs().any(|(key, _)| sensitive_query_key(&key))
    {
        return plan_err!(
            "llm HTTP base_url must be an HTTP(S) URL without credentials or a fragment"
        );
    }
    Ok(base)
}

fn sensitive_query_key(key: &str) -> bool {
    let key = key.to_ascii_lowercase().replace('-', "_");
    is_sensitive_key(&key)
        || key.contains("api_key")
        || key.contains("apikey")
        || key.contains("token")
        || key.contains("secret")
        || key.contains("password")
        || key.contains("signature")
        || key.contains("credential")
        || matches!(key.as_str(), "key" | "authorization" | "auth" | "sig")
}

fn bearer_header(key: &str) -> Result<HeaderValue> {
    if key.trim().is_empty() {
        return plan_err!("llm HTTP API key must be non-empty");
    }
    let mut value = HeaderValue::from_str(&format!("Bearer {key}"))
        .map_err(|_| plan_datafusion_err!("Invalid llm HTTP API key header value"))?;
    value.set_sensitive(true);
    Ok(value)
}

const MAX_ERROR_BODY_BYTES: usize = 8 * 1024;
const MAX_ERROR_DETAIL_BYTES: usize = 256;

async fn response_error(
    response: Response,
    authorization: Option<&HeaderValue>,
    path: &[&str],
) -> LlmBackendError {
    let status = response.status();
    let retry_after = response
        .headers()
        .get(RETRY_AFTER)
        .and_then(|value| value.to_str().ok())
        .and_then(|value| retry_after(value, SystemTime::now()));
    let transient = status == StatusCode::REQUEST_TIMEOUT || status.is_server_error();
    let mut message = if status == StatusCode::TOO_MANY_REQUESTS {
        "HTTP 429".into()
    } else if transient {
        format!("{} request failed", path.join(" "))
    } else {
        format!("{} request rejected", path.join(" "))
    };

    // We can classify the status immediately, even when the body stalls.
    if let Ok(Some(detail)) = tokio::time::timeout(
        Duration::from_millis(250),
        read_error_detail(response, authorization),
    )
    .await
    {
        message.push_str(": ");
        message.push_str(&detail);
    }
    if status == StatusCode::TOO_MANY_REQUESTS {
        LlmBackendError::RateLimited {
            retry_after,
            message,
        }
    } else if transient {
        LlmBackendError::Transient {
            status: Some(status.as_u16()),
            message,
        }
    } else {
        LlmBackendError::Permanent {
            status: Some(status.as_u16()),
            message,
        }
    }
}

async fn read_error_detail(
    mut response: Response,
    authorization: Option<&HeaderValue>,
) -> Option<String> {
    if response
        .content_length()
        .is_some_and(|length| length > MAX_ERROR_BODY_BYTES as u64)
    {
        return None;
    }
    let mut body = Vec::new();
    while let Some(chunk) = response.chunk().await.ok()? {
        if chunk.len() > MAX_ERROR_BODY_BYTES - body.len() {
            return None;
        }
        body.extend_from_slice(&chunk);
    }

    let body: Value = serde_json::from_slice(&body).ok()?;
    let message = body["error"]["message"]
        .as_str()
        .or_else(|| body["message"].as_str())?;
    let mut detail = single_line(message);
    if detail.contains("://") {
        return None;
    }
    if let Some(authorization) = authorization {
        let key = std::str::from_utf8(authorization.as_bytes())
            .ok()?
            .strip_prefix("Bearer ")?;
        let normalized_key = single_line(key);
        if normalized_key.is_empty() {
            return None;
        }
        detail = detail
            .replace(key, "[REDACTED]")
            .replace(&normalized_key, "[REDACTED]");
        if detail.contains(key) || detail.contains(&normalized_key) {
            return None;
        }
    }
    detail.truncate(detail.floor_char_boundary(MAX_ERROR_DETAIL_BYTES));
    let detail = detail.trim_end();
    (!detail.is_empty()).then(|| detail.to_owned())
}

fn single_line(value: &str) -> String {
    value
        .split(|ch: char| {
            ch.is_control() || ch.is_whitespace()
                || matches!(ch, '\u{061c}' | '\u{200b}'..='\u{200f}' | '\u{202a}'..='\u{202e}' | '\u{2060}'..='\u{206f}' | '\u{feff}')
        })
        .filter(|part| !part.is_empty())
        .collect::<Vec<_>>()
        .join(" ")
}

fn retry_after(value: &str, now: SystemTime) -> Option<Duration> {
    let value = value.trim();
    if let Ok(seconds) = value.parse::<u64>() {
        return Some(Duration::from_secs(seconds));
    }
    let time = httpdate::parse_http_date(value).ok()?;
    Some(time.duration_since(now).unwrap_or_default())
}

fn transport_error(error: &reqwest::Error, path: &[&str]) -> LlmBackendError {
    let operation = path.join(" ");
    if error.is_builder() {
        return permanent(None, &format!("could not construct {operation} request"));
    }
    LlmBackendError::Transient {
        status: None,
        message: if error.is_timeout() {
            format!("{operation} request timed out")
        } else {
            format!("{operation} transport failed")
        },
    }
}

fn permanent(status: Option<u16>, message: &str) -> LlmBackendError {
    LlmBackendError::Permanent {
        status,
        message: message.into(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_retry_after_seconds_and_http_dates() {
        let now = SystemTime::UNIX_EPOCH + Duration::from_secs(1_771_000_000);
        let later = httpdate::fmt_http_date(now + Duration::from_secs(120));
        assert_eq!(retry_after("120", now), Some(Duration::from_secs(120)));
        assert_eq!(retry_after(&later, now), Some(Duration::from_secs(120)));
        assert_eq!(
            retry_after(&later, now + Duration::from_secs(180)),
            Some(Duration::ZERO)
        );
        assert_eq!(retry_after("invalid", now), None);
    }
}
