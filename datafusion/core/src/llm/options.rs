// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: See the NOTICE file distributed with this work for additional information regarding copyright ownership
// SPDX-FileContributor: Licensed to the Apache Software Foundation (ASF) under one or more contributor license agreements

use std::fmt;

use crate::common::{DataFusionError, Result};
use serde::de::{self, Deserialize, Deserializer, MapAccess, SeqAccess, Visitor};
use serde_json::{Map, Value};

/// Canonicalize LLM options and reject credential keys.
pub fn canonical_options(options: &str) -> Result<String> {
    let Options(mut value) = serde_json::from_str(options).map_err(|error| {
        DataFusionError::Plan(format!("invalid llm() options_json: {error}"))
    })?;
    value.sort_all_objects();
    Ok(value.to_string())
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
            if matches!(
                key.to_ascii_lowercase().as_str(),
                "api_key" | "token" | "secret" | "password"
            ) {
                return Err(de::Error::custom("contains a sensitive credential key"));
            }
            let Options(value) = map.next_value()?;
            values.insert(key, value);
        }
        Ok(Value::Object(values))
    }
}
