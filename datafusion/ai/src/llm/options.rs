// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: See the NOTICE file distributed with this work for additional information regarding copyright ownership
// SPDX-FileContributor: Licensed to the Apache Software Foundation (ASF) under one or more contributor license agreements

use std::fmt;
use std::sync::Arc;

use arrow::datatypes::{DataType, Field, FieldRef};
use datafusion_common::{DataFusionError, Result, plan_err};
use serde::de::{
    self, Deserialize, DeserializeSeed, Deserializer, MapAccess, SeqAccess, Visitor,
};
use serde_json::{Map, Value};

use super::TaskKind;
use super::schema::{ResponseSchema, SchemaInput};
use crate::AiErrorType;

const MAX_DIMENSION: u32 = 1 << 20;
const DEFAULT_MEDIA_BYTES: u64 = 8 << 20;
const MAX_MEDIA_BYTES: u64 = 32 << 20;
const DEFAULT_MEDIA_ITEMS: u32 = 16;
const MAX_MEDIA_ITEMS: u32 = 64;
const DEFAULT_ROW_MEDIA_BYTES: u64 = 32 << 20;
const MAX_ROW_MEDIA_BYTES: u64 = 128 << 20;

/// Determines how inference handles terminal request and row errors.
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
    /// Emit the result and per-row diagnostics in a `df.ai_error` extension.
    Envelope,
}

/// Controls how the HTTP backend asks a provider to honor a response schema.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SchemaMode {
    /// Use the provider's strict JSON Schema response format.
    #[default]
    Strict,
    /// Request a JSON object and include the schema in the system message.
    JsonObject,
    /// Include the schema in the system message alone.
    PromptOnly,
}

/// Determines how vision handles recognized image formats it cannot resize.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ImageResizeFallback {
    /// Send the original image bytes to the provider.
    #[default]
    Passthrough,
    /// Report an error for the row using `on_error`.
    Error,
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
    /// Action when a request or an individual output row fails.
    #[serde(default)]
    pub on_error: OnError,
    /// Output dimension required for `embed()`.
    pub dimension: Option<u32>,
    /// Include the requested embedding dimension in the HTTP body. Defaults to false.
    pub send_dimensions: Option<bool>,
    /// Maximum bytes to read per media reference. Defaults to 8 MiB.
    pub max_media_bytes: Option<u64>,
    /// Maximum images per vision row, between 1 and 64. Defaults to 16.
    pub max_media_items: Option<u32>,
    /// Maximum total image bytes per vision row. Defaults to 32 MiB and caps at 128 MiB.
    pub max_row_media_bytes: Option<u64>,
    /// Maximum edge in pixels when downsampling PNGs for `vlm()`. Between 16 and 8192.
    pub image_max_edge: Option<u32>,
    /// Fallback for formats that cannot be resized. Requires `image_max_edge` when set.
    #[serde(default)]
    pub image_resize_fallback: ImageResizeFallback,
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
    /// A JSON Schema object for typed `llm()` and `vlm()` completion values.
    pub response_schema: Option<Value>,
    /// Provider response format for structured output. Defaults to [`SchemaMode::Strict`].
    #[serde(default)]
    pub schema_mode: SchemaMode,
    /// Additional requests per row whose completion fails schema validation (0 to 3).
    /// Defaults to zero.
    pub schema_retries: Option<u32>,
    /// Provider-specific body fields. Credential keys are forbidden at any depth.
    pub extra_body: Option<Map<String, Value>>,
}

impl LlmOptions {
    pub(super) fn parse(json: &str) -> Result<(Self, String, Option<ResponseSchema>)> {
        // Scan even overwritten members before letting serde_json decode values.
        let _: CheckedOptions =
            serde_json::from_str(json).map_err(|error| invalid_options(&error))?;
        let mut value: Value =
            serde_json::from_str(json).map_err(|error| invalid_options(&error))?;
        if !value.is_object() {
            return plan_err!("invalid llm() options_json: expected a JSON object");
        }
        if value.get("max_retries").is_some_and(|value| {
            value.is_number()
                && value
                    .as_u64()
                    .and_then(|number| usize::try_from(number).ok())
                    .is_none()
        }) {
            return invalid_value(
                "max_retries",
                "a nonnegative integer that fits in usize",
            );
        }
        value.sort_all_objects();
        let schema = value
            .get("response_schema")
            .map(|_| SchemaInput::from_options(json))
            .transpose()?;
        let canonical = match &schema {
            Some(schema) => schema.canonical_options(value.as_object().unwrap()),
            None => value.to_string(),
        };
        let has_image_resize_fallback = value.get("image_resize_fallback").is_some();
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
            .max_media_bytes
            .is_some_and(|limit| limit > MAX_MEDIA_BYTES)
        {
            return invalid_value("max_media_bytes", "at most 33554432 bytes");
        }
        if options
            .max_media_items
            .is_some_and(|limit| !(1..=MAX_MEDIA_ITEMS).contains(&limit))
        {
            return invalid_value(
                "max_media_items",
                &format!("between 1 and {MAX_MEDIA_ITEMS}"),
            );
        }
        if options
            .max_row_media_bytes
            .is_some_and(|limit| limit > MAX_ROW_MEDIA_BYTES)
        {
            return invalid_value(
                "max_row_media_bytes",
                &format!("at most {MAX_ROW_MEDIA_BYTES} bytes"),
            );
        }
        if options
            .image_max_edge
            .is_some_and(|edge| !(16..=8192).contains(&edge))
        {
            return invalid_value("image_max_edge", "between 16 and 8192");
        }
        if has_image_resize_fallback && options.image_max_edge.is_none() {
            return plan_err!("image_resize_fallback requires image_max_edge");
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
        if options.schema_mode != SchemaMode::Strict && schema.is_none() {
            return plan_err!(
                "schema_mode requires response_schema for non-strict modes"
            );
        }
        let schema_retries = options.schema_retries.unwrap_or_default();
        if schema_retries > 3 {
            return invalid_value("schema_retries", "between 0 and 3");
        }
        if schema_retries != 0 && schema.is_none() {
            return plan_err!("nonzero schema_retries requires response_schema");
        }
        if schema.is_some()
            && options
                .extra_body
                .as_ref()
                .is_some_and(|body| body.contains_key("response_format"))
        {
            return plan_err!(
                "response_schema conflicts with extra_body.response_format"
            );
        }
        let schema = schema.as_ref().map(SchemaInput::compile).transpose()?;
        Ok((options, canonical, schema))
    }

    pub(super) fn embedding_dimension(&self) -> Result<i32> {
        match self.dimension {
            Some(dim) if (1..=MAX_DIMENSION).contains(&dim) => Ok(dim as i32),
            _ => plan_err!(
                "embed() requires a dimension between 1 and {MAX_DIMENSION} in options_json"
            ),
        }
    }

    pub(super) fn media_byte_limit(&self) -> u64 {
        self.max_media_bytes.unwrap_or(DEFAULT_MEDIA_BYTES)
    }

    pub(super) fn media_item_limit(&self) -> usize {
        self.max_media_items.unwrap_or(DEFAULT_MEDIA_ITEMS) as usize
    }

    pub(super) fn row_media_byte_limit(&self) -> u64 {
        self.max_row_media_bytes.unwrap_or(DEFAULT_ROW_MEDIA_BYTES)
    }

    pub(super) fn output_field(
        &self,
        kind: TaskKind,
        name: &str,
        schema: Option<&ResponseSchema>,
    ) -> Result<FieldRef> {
        if kind != TaskKind::Embed && self.send_dimensions.is_some() {
            return plan_err!("send_dimensions is only supported by embed()");
        }
        if self.image_max_edge.is_some() && kind != TaskKind::Vision {
            return plan_err!("image_max_edge is only supported by vlm()");
        }
        let data_type = match (kind, schema) {
            (TaskKind::Embed, Some(_)) => {
                return plan_err!("embed() does not support response_schema");
            }
            (TaskKind::Embed, None) => DataType::FixedSizeList(
                Arc::new(Field::new_list_field(DataType::Float32, false)),
                self.embedding_dimension()?,
            ),
            (_, Some(schema)) => schema.data_type().clone(),
            (_, None) => DataType::Utf8,
        };
        let field = if self.on_error == OnError::Envelope {
            AiErrorType::new(data_type).field(name, false)?
        } else {
            Field::new(name, data_type, true)
        };
        Ok(Arc::new(field))
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

struct CheckedOptions;

impl<'de> Deserialize<'de> for CheckedOptions {
    fn deserialize<D: Deserializer<'de>>(
        deserializer: D,
    ) -> std::result::Result<Self, D::Error> {
        OptionsVisitor::Root
            .deserialize(deserializer)
            .map(|()| Self)
    }
}

#[derive(Clone, Copy)]
enum OptionsVisitor {
    Root,
    Schema,
    PropertyNames,
    Other,
}

impl<'de> DeserializeSeed<'de> for OptionsVisitor {
    type Value = ();

    fn deserialize<D: Deserializer<'de>>(
        self,
        deserializer: D,
    ) -> std::result::Result<(), D::Error> {
        deserializer.deserialize_any(self)
    }
}

impl<'de> Visitor<'de> for OptionsVisitor {
    type Value = ();

    fn expecting(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("a JSON value")
    }

    fn visit_bool<E: de::Error>(self, _value: bool) -> std::result::Result<(), E> {
        Ok(())
    }

    fn visit_i64<E: de::Error>(self, _value: i64) -> std::result::Result<(), E> {
        Ok(())
    }

    fn visit_u64<E: de::Error>(self, _value: u64) -> std::result::Result<(), E> {
        Ok(())
    }

    fn visit_f64<E: de::Error>(self, _value: f64) -> std::result::Result<(), E> {
        Ok(())
    }

    fn visit_str<E: de::Error>(self, _value: &str) -> std::result::Result<(), E> {
        Ok(())
    }

    fn visit_unit<E: de::Error>(self) -> std::result::Result<(), E> {
        Ok(())
    }

    fn visit_seq<A: SeqAccess<'de>>(
        self,
        mut seq: A,
    ) -> std::result::Result<(), A::Error> {
        while seq.next_element_seed(Self::Other)?.is_some() {}
        Ok(())
    }

    fn visit_map<A: MapAccess<'de>>(
        self,
        mut map: A,
    ) -> std::result::Result<(), A::Error> {
        // Check each key as it is read, including members a duplicate key might overwrite.
        while let Some(key) = map.next_key::<String>()? {
            if !matches!(self, Self::PropertyNames) && is_sensitive_key(&key) {
                return Err(de::Error::custom("contains a sensitive credential key"));
            }
            let child = match (self, key.as_str()) {
                (Self::Root, "response_schema")
                | (Self::Schema, "items")
                | (Self::PropertyNames, _) => Self::Schema,
                (Self::Schema, "properties") => Self::PropertyNames,
                _ => Self::Other,
            };
            map.next_value_seed(child)?;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const SCHEMA: &str = r#"{"type":"object","properties":{"count":{"type":"integer"}}}"#;

    #[test]
    fn schema_mode_requires_schema_except_for_strict() -> Result<()> {
        assert_eq!(LlmOptions::parse("{}")?.0.schema_mode, SchemaMode::Strict);
        assert!(LlmOptions::parse(r#"{"schema_mode":"strict"}"#).is_ok());
        for (name, mode) in [
            ("strict", SchemaMode::Strict),
            ("json_object", SchemaMode::JsonObject),
            ("prompt_only", SchemaMode::PromptOnly),
        ] {
            let options =
                format!(r#"{{"schema_mode":"{name}","response_schema":{SCHEMA}}}"#);
            assert_eq!(LlmOptions::parse(&options)?.0.schema_mode, mode);
        }
        for name in ["json_object", "prompt_only", "invalid"] {
            let options = format!(r#"{{"schema_mode":"{name}"}}"#);
            assert!(matches!(
                LlmOptions::parse(&options),
                Err(DataFusionError::Plan(_))
            ));
        }
        Ok(())
    }

    #[test]
    fn schema_retries_are_bounded_and_require_schema_when_nonzero() -> Result<()> {
        assert_eq!(LlmOptions::parse("{}")?.0.schema_retries, None);
        assert!(LlmOptions::parse(r#"{"schema_retries":0}"#).is_ok());
        for retries in [0, 3] {
            let options =
                format!(r#"{{"schema_retries":{retries},"response_schema":{SCHEMA}}}"#);
            assert_eq!(LlmOptions::parse(&options)?.0.schema_retries, Some(retries));
        }
        for options in [
            r#"{"schema_retries":1}"#.to_owned(),
            format!(r#"{{"schema_retries":4,"response_schema":{SCHEMA}}}"#),
        ] {
            assert!(matches!(
                LlmOptions::parse(&options),
                Err(DataFusionError::Plan(message)) if message.contains("schema_retries")
            ));
        }
        Ok(())
    }

    #[test]
    fn send_dimensions_is_embed_only() -> Result<()> {
        for value in [true, false] {
            let json = format!(r#"{{"dimension":2,"send_dimensions":{value}}}"#);
            let (options, canonical, _) = LlmOptions::parse(&json)?;
            assert_eq!(options.send_dimensions, Some(value));
            assert_eq!(canonical, json);
            options.output_field(TaskKind::Embed, "answer", None)?;
            for kind in [TaskKind::Chat, TaskKind::Vision] {
                assert!(matches!(
                    options.output_field(kind, "answer", None),
                    Err(DataFusionError::Plan(message)) if message.contains("send_dimensions")
                ));
            }
        }
        Ok(())
    }

    #[test]
    fn rejects_invalid_options() {
        for (json, message) in [
            ("[]", "JSON object"),
            (r#"{"unknown":1}"#, "unknown field"),
            (r#"{"provider":" "}"#, "provider"),
            (r#"{"batch_size":0}"#, "batch_size"),
            (r#"{"inflight_limit":0}"#, "inflight_limit"),
            (r#"{"timeout_ms":0}"#, "timeout_ms"),
            (r#"{"max_media_bytes":33554433}"#, "max_media_bytes"),
            (
                r#"{"max_retries":-1}"#,
                "max_retries must be a nonnegative integer",
            ),
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
        let (options, canonical, _) = LlmOptions::parse(
            r#"{"top_p":0.75,"extra_body":{"z":[{"b":0.125,"a":2}],"a":{}}, "on_error":"ignore","max_retries":0,"temperature":0.25}"#,
        )
        .unwrap();
        assert_eq!(options.on_error, OnError::Ignore);
        assert_eq!(options.max_retries, Some(0));
        assert_eq!(options.top_p, Some(0.75));
        assert_eq!(options.temperature, Some(0.25));
        assert_eq!(
            canonical,
            r#"{"extra_body":{"a":{},"z":[{"a":2,"b":0.125}]},"max_retries":0,"on_error":"ignore","temperature":0.25,"top_p":0.75}"#
        );
    }

    #[test]
    fn credential_scan_exempts_only_schema_property_names() {
        let accepted =
            r#"{"response_schema":{"properties":{"token":{"type":"string"}}}}"#;
        serde_json::from_str::<CheckedOptions>(accepted).unwrap();
        let valid = r#"{"response_schema":{"type":"object","properties":{
            "token":{"type":"string"},
            "rows":{"type":"array","items":{"type":"object","properties":{
                "password":{"type":"integer"},"api_key":{"type":"boolean"}
            }}}
        }}}"#;
        assert!(LlmOptions::parse(valid).is_ok());
        for json in [
            r#"{"extra_body":{"token":"x"}}"#,
            r#"{"response_schema":{"properties":{"token":{"default":"x"}}},"extra_body":{"api_key":"x"}}"#,
            r#"{"extra_body":{"response_schema":{"properties":{"token":{"type":"string"}}}}}"#,
            r#"{"response_schema":{"properties":{"token":{"secret":"x"}}}}"#,
            r#"{"response_schema":{"properties":{"token":{"default":{"password":"x"}}}}}"#,
        ] {
            assert!(
                matches!(LlmOptions::parse(json), Err(DataFusionError::Plan(message)) if message.contains("sensitive credential key")),
                "{json}"
            );
        }
    }
}
