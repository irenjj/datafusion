// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: See the NOTICE file distributed with this work for additional information regarding copyright ownership
// SPDX-FileContributor: Licensed to the Apache Software Foundation (ASF) under one or more contributor license agreements

use std::fmt;

use datafusion_common::{DataFusionError, Result, plan_err};
use serde::de::{self, Deserialize, Deserializer, MapAccess, SeqAccess, Visitor};
use serde_json::{Map, Value};

/// Determines how an inference slice handles a terminal backend error.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum OnError {
    /// Propagate the error and stop the stream.
    #[default]
    Raise,
    /// Log a warning and output NULL for the affected rows.
    Log,
    /// Output NULL for the affected rows.
    Ignore,
}

/// SQL execution controls and provider generation options.
#[derive(Debug, Clone, Default, serde::Deserialize)]
#[serde(deny_unknown_fields)]
pub struct LlmOptions {
    /// Registered backend name. Defaults to the session's default provider.
    pub provider: Option<String>,
    /// Number of non-NULL prompts per request. Defaults to the backend's preference.
    pub batch_size: Option<usize>,
    /// Maximum concurrent requests per operator across partitions. Defaults to 64.
    pub inflight_limit: Option<usize>,
    /// Timeout per attempt in milliseconds. Defaults to 30,000.
    pub timeout_ms: Option<u64>,
    /// Maximum retries after the initial attempt. Defaults to 3.
    pub max_retries: Option<usize>,
    /// Action when a request fails permanently or exhausts retries.
    #[serde(default)]
    pub on_error: OnError,
    /// Reuse identical calls in a SELECT for non-volatile prompts. Defaults to true.
    pub deduplicate: Option<bool>,
    /// Optional system message.
    pub system_message: Option<String>,
    /// Nonnegative sampling temperature.
    pub temperature: Option<f64>,
    /// Positive generation token limit.
    pub max_tokens: Option<u32>,
    /// Nucleus sampling probability in (0, 1].
    pub top_p: Option<f64>,
    /// Nonempty list of nonempty stop strings.
    pub stop: Option<Vec<String>>,
    /// Provider-specific body fields. Credential keys are forbidden at any depth.
    pub extra_body: Option<Map<String, Value>>,
}

impl LlmOptions {
    pub(super) fn parse(json: &str) -> Result<(Self, String)> {
        // Check every key before materializing the object, including duplicate members.
        let Options(mut value) =
            serde_json::from_str(json).map_err(|error| invalid_options(&error))?;
        if !value.is_object() {
            return plan_err!("invalid llm() options_json: expected a JSON object");
        }
        value.sort_all_objects();
        let canonical = value.to_string();
        let options: Self =
            serde_json::from_value(value).map_err(|error| invalid_options(&error))?;
        if options
            .provider
            .as_ref()
            .is_some_and(|name| name.trim().is_empty())
        {
            return invalid_value("provider", "a non-empty name");
        }
        if options.batch_size == Some(0) {
            return invalid_value("batch_size", "at least 1");
        }
        if let Some(limit) = options.inflight_limit {
            validate_inflight_limit(limit)?;
        }
        if options.timeout_ms == Some(0) {
            return invalid_value("timeout_ms", "at least 1");
        }
        if options.max_tokens == Some(0) {
            return invalid_value("max_tokens", "at least 1");
        }
        if options
            .temperature
            .is_some_and(|t| !t.is_finite() || t < 0.0)
        {
            return invalid_value("temperature", "finite and at least 0");
        }
        if options.top_p.is_some_and(|p| !(0.0 < p && p <= 1.0)) {
            return invalid_value("top_p", "in (0, 1]");
        }
        if options
            .stop
            .as_ref()
            .is_some_and(|stop| stop.is_empty() || stop.iter().any(String::is_empty))
        {
            return invalid_value("stop", "a non-empty list of non-empty strings");
        }
        Ok((options, canonical))
    }
}

pub(super) fn validate_inflight_limit(limit: usize) -> Result<()> {
    if !(1..=tokio::sync::Semaphore::MAX_PERMITS).contains(&limit) {
        return invalid_value("inflight_limit", "between 1 and Semaphore::MAX_PERMITS");
    }
    Ok(())
}

fn invalid_value<T>(field: &str, expected: &str) -> Result<T> {
    plan_err!("invalid llm() options_json: {field} must be {expected}")
}

fn invalid_options(error: &serde_json::Error) -> DataFusionError {
    DataFusionError::Plan(format!("invalid llm() options_json: {error}"))
}

pub(super) fn is_sensitive_key(key: &str) -> bool {
    matches!(
        key.to_ascii_lowercase().as_str(),
        "api_key" | "token" | "secret" | "password"
    )
}

struct Options(Value);

impl<'de> Deserialize<'de> for Options {
    fn deserialize<D: Deserializer<'de>>(
        deserializer: D,
    ) -> std::result::Result<Self, D::Error> {
        deserializer.deserialize_any(OptionsVisitor).map(Self)
    }
}

struct OptionsVisitor;

impl<'de> Visitor<'de> for OptionsVisitor {
    type Value = Value;

    fn expecting(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("a JSON value")
    }

    fn visit_bool<E: de::Error>(self, value: bool) -> std::result::Result<Value, E> {
        Ok(Value::from(value))
    }

    fn visit_i64<E: de::Error>(self, value: i64) -> std::result::Result<Value, E> {
        Ok(Value::from(value))
    }

    fn visit_u64<E: de::Error>(self, value: u64) -> std::result::Result<Value, E> {
        Ok(Value::from(value))
    }

    fn visit_f64<E: de::Error>(self, value: f64) -> std::result::Result<Value, E> {
        Ok(Value::from(value))
    }

    fn visit_str<E: de::Error>(self, value: &str) -> std::result::Result<Value, E> {
        Ok(Value::from(value))
    }

    fn visit_unit<E: de::Error>(self) -> std::result::Result<Value, E> {
        Ok(Value::Null)
    }

    fn visit_seq<A: SeqAccess<'de>>(
        self,
        mut seq: A,
    ) -> std::result::Result<Value, A::Error> {
        let mut values = Vec::new();
        while let Some(Options(value)) = seq.next_element()? {
            values.push(value);
        }
        Ok(Value::Array(values))
    }

    fn visit_map<A: MapAccess<'de>>(
        self,
        mut map: A,
    ) -> std::result::Result<Value, A::Error> {
        let mut values = Map::new();
        // Check each key as it is read, including members a duplicate key might overwrite.
        while let Some(key) = map.next_key::<String>()? {
            if is_sensitive_key(&key) {
                return Err(de::Error::custom("contains a sensitive credential key"));
            }
            let Options(value) = map.next_value()?;
            values.insert(key, value);
        }
        Ok(Value::Object(values))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rejects_invalid_options() {
        for (json, message) in [
            ("[]", "JSON object"),
            (r#"{"unknown":1}"#, "unknown field"),
            (r#"{"provider":" "}"#, "provider"),
            (r#"{"batch_size":0}"#, "batch_size"),
            (r#"{"inflight_limit":0}"#, "inflight_limit"),
            (r#"{"timeout_ms":0}"#, "timeout_ms"),
            (r#"{"max_retries":-1}"#, "usize"),
            (r#"{"on_error":"continue"}"#, "unknown variant"),
            (r#"{"deduplicate":0}"#, "boolean"),
            (r#"{"max_tokens":0}"#, "max_tokens"),
            (r#"{"temperature":-0.1}"#, "temperature"),
            (r#"{"top_p":0.0}"#, "top_p"),
            (r#"{"top_p":1.1}"#, "top_p"),
            (r#"{"stop":[]}"#, "stop"),
            (r#"{"stop":[""]}"#, "stop"),
            (r#"{"extra_body":[]}"#, "map"),
            (
                r#"{"extra_body":{"nested":[{"PassWord":"value"}]},"extra_body":{}}"#,
                "sensitive credential key",
            ),
        ] {
            let error = LlmOptions::parse(json).unwrap_err();
            assert!(matches!(error, DataFusionError::Plan(_)));
            assert!(error.to_string().contains(message), "{error} for {json}");
        }
        let limit = tokio::sync::Semaphore::MAX_PERMITS + 1;
        let invalid = format!(r#"{{"inflight_limit":{limit}}}"#);
        assert!(LlmOptions::parse(&invalid).is_err());
    }

    #[test]
    fn parses_typed_options_and_recursively_sorts_json() {
        let (options, canonical) = LlmOptions::parse(
            r#"{"top_p":1,"extra_body":{"z":[{"b":1,"a":2}],"a":{}}, "on_error":"ignore","max_retries":0}"#,
        )
        .unwrap();
        assert_eq!(options.on_error, OnError::Ignore);
        assert_eq!(options.max_retries, Some(0));
        assert_eq!(options.top_p, Some(1.0));
        assert_eq!(
            canonical,
            r#"{"extra_body":{"a":{},"z":[{"a":2,"b":1}]},"max_retries":0,"on_error":"ignore","top_p":1}"#
        );
    }
}
