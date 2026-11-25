// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: See the NOTICE file distributed with this work for additional information regarding copyright ownership
// SPDX-FileContributor: Licensed to the Apache Software Foundation (ASF) under one or more contributor license agreements

use std::fmt;
use std::time::{Duration, SystemTime};

use async_trait::async_trait;
use reqwest::header::{AUTHORIZATION, HeaderValue, RETRY_AFTER};
use reqwest::{Client, StatusCode, Url};
use serde::Deserialize;
use serde_json::{Value, json};

use crate::common::{Result, plan_datafusion_err, plan_err};

use super::options::is_sensitive_key;
use super::{
    CancellationToken, LlmBackend, LlmBackendError, LlmRequest, LlmResponse, LlmUsage,
};

/// Runtime configuration for an OpenAI-compatible chat completions endpoint.
pub struct OpenAiHttpConfig {
    /// Provider root, including any API path prefix such as /v1.
    pub base_url: String,
    /// Optional API key injected by the application.
    pub api_key: Option<String>,
    /// Optional environment variable from which to load the key at construction.
    pub api_key_env: Option<String>,
}

impl fmt::Debug for OpenAiHttpConfig {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("OpenAiHttpConfig").finish_non_exhaustive()
    }
}

/// A backend for OpenAI-compatible HTTP chat completions.
pub struct OpenAiHttpBackend {
    client: Client,
    endpoint: Url,
    authorization: Option<HeaderValue>,
}

impl fmt::Debug for OpenAiHttpBackend {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("OpenAiHttpBackend").finish_non_exhaustive()
    }
}

impl OpenAiHttpBackend {
    /// Builds a shared HTTP client and validates its endpoint and credentials.
    pub fn try_new(config: OpenAiHttpConfig) -> Result<Self> {
        if !config
            .base_url
            .split_once("://")
            .is_some_and(|(scheme, _)| {
                scheme.eq_ignore_ascii_case("http")
                    || scheme.eq_ignore_ascii_case("https")
            })
        {
            return plan_err!("llm HTTP base_url must begin with http:// or https://");
        }
        let mut endpoint = Url::parse(&config.base_url)
            .map_err(|_| plan_datafusion_err!("Invalid llm HTTP base_url"))?;
        if !matches!(endpoint.scheme(), "http" | "https")
            || endpoint.host_str().is_none()
            || !endpoint.username().is_empty()
            || endpoint.password().is_some()
            || endpoint.fragment().is_some()
            || endpoint
                .query_pairs()
                .any(|(key, _)| sensitive_query_key(&key))
        {
            return plan_err!(
                "llm HTTP base_url must be an HTTP(S) URL without credentials or a fragment"
            );
        }
        endpoint
            .path_segments_mut()
            .map_err(|_| plan_datafusion_err!("Invalid llm HTTP base_url path"))?
            .pop_if_empty()
            .extend(["chat", "completions"]);

        let api_key = match (config.api_key, config.api_key_env) {
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
        let authorization = api_key
            .map(|key| {
                if key.trim().is_empty() {
                    return plan_err!("llm HTTP API key must be non-empty");
                }
                let mut value =
                    HeaderValue::from_str(&format!("Bearer {key}")).map_err(|_| {
                        plan_datafusion_err!("Invalid llm HTTP API key header value")
                    })?;
                value.set_sensitive(true);
                Ok(value)
            })
            .transpose()?;

        // One generate call maps to one HTTP attempt. Redirects could also
        // change the endpoint or forward request data outside its origin.
        let client = Client::builder()
            .redirect(reqwest::redirect::Policy::none())
            .retry(reqwest::retry::never())
            .build()
            .map_err(|_| plan_datafusion_err!("Could not build the llm HTTP client"))?;
        Ok(Self {
            client,
            endpoint,
            authorization,
        })
    }

    async fn complete(
        &self,
        request: LlmRequest,
    ) -> std::result::Result<LlmResponse, LlmBackendError> {
        let [prompt] = request.prompts.as_slice() else {
            return Err(permanent(
                None,
                "chat completions require exactly one prompt per request",
            ));
        };
        let options = &request.options;
        let mut messages = Vec::new();
        if let Some(system) = &options.system_message {
            messages.push(json!({"role": "system", "content": system}));
        }
        messages.push(json!({"role": "user", "content": prompt}));
        let mut body = options.extra_body.clone().unwrap_or_default();
        if contains_credential_key(&body) {
            return Err(permanent(
                None,
                "extra_body contains a sensitive credential key",
            ));
        }
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

        let mut http = self.client.post(self.endpoint.clone()).json(&body);
        if let Some(authorization) = &self.authorization {
            http = http.header(AUTHORIZATION, authorization.clone());
        }
        let response = http.send().await.map_err(|error| transport_error(&error))?;
        let status = response.status();
        if status == StatusCode::TOO_MANY_REQUESTS {
            return Err(LlmBackendError::RateLimited {
                retry_after: response
                    .headers()
                    .get(RETRY_AFTER)
                    .and_then(|value| value.to_str().ok())
                    .and_then(|value| retry_after(value, SystemTime::now())),
                message: "HTTP 429".into(),
            });
        }
        if status == StatusCode::REQUEST_TIMEOUT || status.is_server_error() {
            return Err(LlmBackendError::Transient {
                status: Some(status.as_u16()),
                message: "chat completions request failed".into(),
            });
        }
        if !status.is_success() {
            return Err(permanent(
                Some(status.as_u16()),
                "chat completions request rejected",
            ));
        }

        let bytes = response
            .bytes()
            .await
            .map_err(|error| transport_error(&error))?;
        let reply: ChatResponse = serde_json::from_slice(&bytes).map_err(|_| {
            permanent(Some(status.as_u16()), "invalid chat completions response")
        })?;
        let choice = reply.choices.into_iter().next().ok_or_else(|| {
            permanent(
                Some(status.as_u16()),
                "chat completions response has no choices",
            )
        })?;
        Ok(LlmResponse {
            outputs: vec![choice.message.content],
            usage: reply.usage.map(|usage| LlmUsage {
                input_tokens: usage.prompt_tokens,
                output_tokens: usage.completion_tokens,
            }),
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
        tokio::select! {
            biased;
            _ = cancel.cancelled() => Err(LlmBackendError::Cancelled),
            response = self.complete(request) => response,
        }
    }
}

fn contains_credential_key(object: &serde_json::Map<String, Value>) -> bool {
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

fn retry_after(value: &str, now: SystemTime) -> Option<Duration> {
    let value = value.trim();
    if let Ok(seconds) = value.parse() {
        return Some(Duration::from_secs(seconds));
    }
    let time = httpdate::parse_http_date(value).ok()?;
    Some(time.duration_since(now).unwrap_or_default())
}

fn transport_error(error: &reqwest::Error) -> LlmBackendError {
    if error.is_builder() {
        return permanent(None, "could not construct chat completions request");
    }
    LlmBackendError::Transient {
        status: None,
        message: if error.is_timeout() {
            "chat completions request timed out"
        } else {
            "chat completions transport failed"
        }
        .into(),
    }
}

fn permanent(status: Option<u16>, message: &str) -> LlmBackendError {
    LlmBackendError::Permanent {
        status,
        message: message.into(),
    }
}

#[derive(Deserialize)]
struct ChatResponse {
    choices: Vec<ChatChoice>,
    usage: Option<ChatUsage>,
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
    prompt_tokens: u64,
    completion_tokens: u64,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_retry_after_seconds_and_http_dates() {
        let now = SystemTime::UNIX_EPOCH + Duration::from_secs(1_771_000_000);
        let later = httpdate::fmt_http_date(now + Duration::from_secs(12));
        assert_eq!(retry_after("5", now), Some(Duration::from_secs(5)));
        assert_eq!(retry_after(&later, now), Some(Duration::from_secs(12)));
        assert_eq!(
            retry_after(&later, now + Duration::from_secs(30)),
            Some(Duration::ZERO)
        );
        assert_eq!(retry_after("invalid", now), None);
    }

    #[test]
    fn rejects_credentials_in_urls_and_redacts_debug() {
        for url in [
            "file:///v1",
            "https:localhost/v1",
            "https://user:private-marker@localhost/v1",
            "https://localhost/v1?%61pi%5FKey=private-marker",
            "https://localhost/v1?access-token=private-marker",
        ] {
            let config = OpenAiHttpConfig {
                base_url: url.into(),
                api_key: None,
                api_key_env: None,
            };
            let error = OpenAiHttpBackend::try_new(config).unwrap_err();
            assert!(!error.to_string().contains("private-marker"), "{error}");
        }

        let config = OpenAiHttpConfig {
            base_url: "https://localhost/v1".into(),
            api_key: Some("private-marker".into()),
            api_key_env: None,
        };
        assert!(!format!("{config:?}").contains("private-marker"));
        let backend = OpenAiHttpBackend::try_new(config).unwrap();
        assert!(!format!("{backend:?}").contains("private-marker"));
    }
}
