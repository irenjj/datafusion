// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: See the NOTICE file distributed with this work for additional information regarding copyright ownership
// SPDX-FileContributor: Licensed to the Apache Software Foundation (ASF) under one or more contributor license agreements

use std::cell::Cell;
use std::collections::{BTreeMap, HashSet};
use std::fmt;
use std::sync::Arc;

use arrow::array::{
    ArrayRef, BooleanArray, Float64Array, Int64Array, ListArray, StringArray, StructArray,
};
use arrow::buffer::{NullBuffer, OffsetBuffer, ScalarBuffer};
use arrow::datatypes::{DataType, Field};
use datafusion_common::{
    DataFusionError, Result, exec_datafusion_err, internal_err, plan_err,
};
use serde::de::{self, DeserializeSeed, MapAccess, SeqAccess, Visitor};
use serde::ser::SerializeMap;
use serde::{Deserialize, Deserializer, Serialize, Serializer};
use serde_json::{Map, Number, Value};

use super::LlmBackendError;

const MAX_DEPTH: usize = 16;
const MAX_PROPERTIES: usize = 1024;

#[derive(Deserialize)]
struct OptionsWithSchema {
    response_schema: SchemaInput,
}

#[derive(Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub(super) struct SchemaInput {
    #[serde(
        rename = "additionalProperties",
        default,
        deserialize_with = "present",
        skip_serializing_if = "Option::is_none"
    )]
    additional_properties: Option<bool>,
    #[serde(
        rename = "enum",
        default,
        deserialize_with = "present",
        skip_serializing_if = "Option::is_none"
    )]
    choices: Option<Vec<String>>,
    #[serde(
        default,
        deserialize_with = "present",
        skip_serializing_if = "Option::is_none"
    )]
    items: Option<Box<Self>>,
    #[serde(
        default,
        deserialize_with = "present",
        skip_serializing_if = "Option::is_none"
    )]
    properties: Option<Properties>,
    #[serde(
        default,
        deserialize_with = "present",
        skip_serializing_if = "Option::is_none"
    )]
    required: Option<Vec<String>>,
    #[serde(rename = "type")]
    data_type: Value,
}

fn present<'de, D: Deserializer<'de>, T: Deserialize<'de>>(
    deserializer: D,
) -> std::result::Result<Option<T>, D::Error> {
    T::deserialize(deserializer).map(Some)
}

// Property maps carry Arrow field order in the worker codec. Keep that ordering
// local to schemas, independent of the serde_json feature graph.
struct Properties(Vec<(String, SchemaInput)>);

impl<'de> Deserialize<'de> for Properties {
    fn deserialize<D: Deserializer<'de>>(
        deserializer: D,
    ) -> std::result::Result<Self, D::Error> {
        struct PropertyVisitor;

        impl<'de> Visitor<'de> for PropertyVisitor {
            type Value = Properties;

            fn expecting(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
                f.write_str("a response_schema properties object")
            }

            fn visit_map<A: MapAccess<'de>>(
                self,
                mut map: A,
            ) -> std::result::Result<Properties, A::Error> {
                let mut fields = Vec::new();
                let mut names = HashSet::new();
                while let Some(name) = map.next_key::<String>()? {
                    if fields.len() == MAX_PROPERTIES {
                        return Err(de::Error::custom(format!(
                            "properties exceeds the limit of {MAX_PROPERTIES}"
                        )));
                    }
                    if !names.insert(name.clone()) {
                        return Err(de::Error::custom("duplicate name in properties"));
                    }
                    fields.push((name, map.next_value()?));
                }
                Ok(Properties(fields))
            }
        }

        deserializer.deserialize_map(PropertyVisitor)
    }
}

impl Serialize for Properties {
    fn serialize<S: Serializer>(
        &self,
        serializer: S,
    ) -> std::result::Result<S::Ok, S::Error> {
        serializer.collect_map(self.0.iter().map(|(name, schema)| (name, schema)))
    }
}

struct CanonicalOptions<'a> {
    sorted: &'a Map<String, Value>,
    schema: &'a SchemaInput,
}

impl Serialize for CanonicalOptions<'_> {
    fn serialize<S: Serializer>(
        &self,
        serializer: S,
    ) -> std::result::Result<S::Ok, S::Error> {
        let mut map = serializer.serialize_map(Some(self.sorted.len()))?;
        for (key, value) in self.sorted {
            if key == "response_schema" {
                map.serialize_entry(key, self.schema)?;
            } else {
                map.serialize_entry(key, value)?;
            }
        }
        map.end()
    }
}

impl SchemaInput {
    pub(super) fn from_options(json: &str) -> Result<Self> {
        serde_json::from_str::<OptionsWithSchema>(json)
            .map(|options| options.response_schema)
            .map_err(|error| schema_parse_error(&error))
    }

    pub(super) fn canonical_options(&self, sorted: &Map<String, Value>) -> String {
        serde_json::to_string(&CanonicalOptions {
            sorted,
            schema: self,
        })
        .expect("validated LLM options can be serialized")
    }

    pub(super) fn compile(&self) -> Result<ResponseSchema> {
        if self.data_type.as_str() != Some("object") {
            return plan_err!("response_schema root type must be object");
        }
        let mut remaining_properties = MAX_PROPERTIES;
        self.compile_node(false, 1, &mut remaining_properties)
    }

    fn compile_node(
        &self,
        optional: bool,
        depth: usize,
        remaining_properties: &mut usize,
    ) -> Result<ResponseSchema> {
        if depth > MAX_DEPTH {
            return plan_err!("response_schema exceeds the depth limit of {MAX_DEPTH}");
        }
        let (name, explicit_null) = match &self.data_type {
            Value::String(name) => (name.as_str(), false),
            Value::Array(types) => match types.as_slice() {
                [first, second] if first.as_str() == Some("null") => {
                    (second.as_str().unwrap_or_default(), true)
                }
                [first, second] if second.as_str() == Some("null") => {
                    (first.as_str().unwrap_or_default(), true)
                }
                _ => {
                    return plan_err!(
                        "response_schema type arrays must contain one type and null"
                    );
                }
            },
            _ => {
                return plan_err!(
                    "response_schema type must be a string or nullable type array"
                );
            }
        };
        for (keyword, present, expected_type) in [
            (
                "additionalProperties",
                self.additional_properties.is_some(),
                "object",
            ),
            ("properties", self.properties.is_some(), "object"),
            ("required", self.required.is_some(), "object"),
            ("items", self.items.is_some(), "array"),
            ("enum", self.choices.is_some(), "string"),
        ] {
            if present && name != expected_type {
                return plan_err!(
                    "response_schema {keyword} requires type {expected_type}"
                );
            }
        }
        let (data_type, shape) = match name {
            "string" => {
                let choices = self
                    .choices
                    .as_ref()
                    .map(|choices| choices.iter().cloned().collect::<HashSet<_>>());
                if let Some((names, original)) =
                    choices.as_ref().zip(self.choices.as_ref())
                    && (names.is_empty() || names.len() != original.len())
                {
                    return plan_err!(
                        "response_schema enum must contain distinct strings"
                    );
                }
                (DataType::Utf8, Shape::String(choices))
            }
            "boolean" => (DataType::Boolean, Shape::Boolean),
            "integer" => (DataType::Int64, Shape::Integer),
            "number" => (DataType::Float64, Shape::Number),
            "object" => self.compile_object(depth, remaining_properties)?,
            "array" => {
                let Some(items) = &self.items else {
                    return plan_err!("response_schema array requires an items schema");
                };
                let item = items.compile_node(false, depth + 1, remaining_properties)?;
                let field = Field::new_list_field(item.data_type.clone(), item.nullable);
                (DataType::List(Arc::new(field)), Shape::List(Box::new(item)))
            }
            _ => return plan_err!("response_schema has unsupported type {name:?}"),
        };
        Ok(ResponseSchema {
            data_type,
            nullable: optional || explicit_null,
            shape,
        })
    }

    fn compile_object(
        &self,
        depth: usize,
        remaining_properties: &mut usize,
    ) -> Result<(DataType, Shape)> {
        if self.additional_properties == Some(true) {
            return plan_err!("response_schema additionalProperties must be false");
        }
        // Parquet has no representation for an object without leaf fields.
        let Some(properties) =
            self.properties.as_ref().filter(|props| !props.0.is_empty())
        else {
            return plan_err!("response_schema object requires non-empty properties");
        };
        *remaining_properties = remaining_properties
            .checked_sub(properties.0.len())
            .ok_or_else(|| DataFusionError::Plan(format!(
                "response_schema exceeds the limit of {MAX_PROPERTIES} total properties"
            )))?;
        let required = self.required.as_deref().unwrap_or_default();
        let mut pending: HashSet<_> = required.iter().map(String::as_str).collect();
        if pending.len() != required.len() {
            return plan_err!("response_schema required contains a duplicate name");
        }
        let mut fields = Vec::with_capacity(properties.0.len());
        let mut children = BTreeMap::new();
        for (name, schema) in &properties.0 {
            let optional = !pending.remove(name.as_str());
            let child = schema.compile_node(optional, depth + 1, remaining_properties)?;
            fields.push(Field::new(name, child.data_type.clone(), child.nullable));
            children.insert(name.clone(), child);
        }
        if !pending.is_empty() {
            return plan_err!("response_schema required contains an undeclared property");
        }
        Ok((DataType::Struct(fields.into()), Shape::Object(children)))
    }
}

fn schema_parse_error(error: &serde_json::Error) -> DataFusionError {
    DataFusionError::Plan(format!("invalid llm() response_schema: {error}"))
}

#[derive(Debug)]
pub(super) struct ResponseSchema {
    data_type: DataType,
    nullable: bool,
    shape: Shape,
}

#[derive(Debug)]
enum Shape {
    String(Option<HashSet<String>>),
    Boolean,
    Integer,
    Number,
    Object(BTreeMap<String, ResponseSchema>),
    List(Box<ResponseSchema>),
}

impl ResponseSchema {
    pub(super) fn data_type(&self) -> &DataType {
        &self.data_type
    }

    pub(super) fn decode_array(
        &self,
        outputs: Vec<Option<String>>,
        output_rows: impl Iterator<Item = usize>,
    ) -> Result<(ArrayRef, Vec<(usize, LlmBackendError)>)> {
        let mut values = Vec::with_capacity(outputs.len());
        let mut errors = Vec::new();
        for (row, output) in output_rows.zip(outputs) {
            let value = match output.as_deref().map(|text| self.decode(text)) {
                Some(Ok(value)) => Some(value),
                result => {
                    let message = result
                        .and_then(std::result::Result::err)
                        .unwrap_or("structured response is missing JSON content");
                    errors.push((
                        row,
                        LlmBackendError::SchemaMismatch {
                            message: message.into(),
                        },
                    ));
                    None
                }
            };
            values.push(value);
        }
        let values: Vec<_> = values.iter().map(Option::as_ref).collect();
        Ok((build_array(self.data_type(), &values)?, errors))
    }

    fn decode(&self, text: &str) -> std::result::Result<Value, &'static str> {
        let mismatch = Cell::new(None);
        let mut json = serde_json::Deserializer::from_str(text);
        let value = DecodeValue {
            schema: self,
            mismatch: &mismatch,
        }
        .deserialize(&mut json);
        value
            .and_then(|value| json.end().map(|()| value))
            .map_err(|error| {
                if error.is_data() {
                    mismatch
                        .get()
                        .unwrap_or("structured response does not match response_schema")
                } else {
                    "structured response is not valid JSON"
                }
            })
    }
}

#[derive(Clone, Copy)]
struct DecodeValue<'a> {
    schema: &'a ResponseSchema,
    mismatch: &'a Cell<Option<&'static str>>,
}

impl DecodeValue<'_> {
    fn fail<T, E: de::Error>(self, message: &'static str) -> std::result::Result<T, E> {
        self.mismatch.set(Some(message));
        Err(E::custom(message))
    }
}

impl<'de> DeserializeSeed<'de> for DecodeValue<'_> {
    type Value = Value;

    fn deserialize<D: Deserializer<'de>>(
        self,
        deserializer: D,
    ) -> std::result::Result<Value, D::Error> {
        deserializer.deserialize_option(self)
    }
}

impl<'de> Visitor<'de> for DecodeValue<'_> {
    type Value = Value;

    fn expecting(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("a value that matches response_schema")
    }

    fn visit_none<E: de::Error>(self) -> std::result::Result<Value, E> {
        if self.schema.nullable {
            Ok(Value::Null)
        } else {
            self.fail("structured response contains null for a non-nullable field")
        }
    }

    fn visit_some<D: Deserializer<'de>>(
        self,
        deserializer: D,
    ) -> std::result::Result<Value, D::Error> {
        let (result, expected) = match &self.schema.shape {
            Shape::String(choices) => (
                String::deserialize(deserializer).and_then(|text| {
                    if choices
                        .as_ref()
                        .is_some_and(|choices| !choices.contains(&text))
                    {
                        return self.fail(
                            "structured response string is outside the declared enum",
                        );
                    }
                    Ok(Value::String(text))
                }),
                "expected a string",
            ),
            Shape::Boolean => (
                bool::deserialize(deserializer).map(Value::Bool),
                "expected a boolean",
            ),
            Shape::Integer => (
                i64::deserialize(deserializer).map(Value::from),
                "expected a signed 64-bit integer",
            ),
            Shape::Number => (
                f64::deserialize(deserializer).and_then(|value| {
                    Number::from_f64(value)
                        .map(Value::Number)
                        .ok_or_else(|| de::Error::custom("number must be finite"))
                }),
                "expected a finite number",
            ),
            Shape::Object(_) => {
                (deserializer.deserialize_map(self), "expected an object")
            }
            Shape::List(_) => (deserializer.deserialize_seq(self), "expected an array"),
        };
        if result.is_err() && self.mismatch.get().is_none() {
            self.mismatch.set(Some(expected));
        }
        result
    }

    fn visit_map<A: MapAccess<'de>>(
        self,
        mut map: A,
    ) -> std::result::Result<Value, A::Error> {
        let Shape::Object(properties) = &self.schema.shape else {
            unreachable!()
        };
        let mut values = Map::new();
        while let Some(name) = map.next_key::<String>()? {
            let Some(schema) = properties.get(&name) else {
                return self.fail("structured response contains an undeclared property");
            };
            if values.contains_key(&name) {
                return self.fail("structured response contains a duplicate property");
            }
            values.insert(name, map.next_value_seed(Self { schema, ..self })?);
        }
        if properties
            .iter()
            .any(|(name, schema)| !schema.nullable && !values.contains_key(name))
        {
            return self.fail("structured response is missing a non-nullable property");
        }
        Ok(Value::Object(values))
    }

    fn visit_seq<A: SeqAccess<'de>>(
        self,
        mut seq: A,
    ) -> std::result::Result<Value, A::Error> {
        let Shape::List(schema) = &self.schema.shape else {
            unreachable!()
        };
        let mut values = Vec::new();
        while let Some(value) = seq.next_element_seed(Self { schema, ..self })? {
            values.push(value);
        }
        Ok(Value::Array(values))
    }
}

fn build_array(data_type: &DataType, values: &[Option<&Value>]) -> Result<ArrayRef> {
    let array: ArrayRef = match data_type {
        DataType::Utf8 => Arc::new(StringArray::from_iter(
            values.iter().map(|value| value.and_then(Value::as_str)),
        )),
        DataType::Boolean => Arc::new(BooleanArray::from_iter(
            values.iter().map(|value| value.and_then(Value::as_bool)),
        )),
        DataType::Int64 => Arc::new(Int64Array::from_iter(
            values.iter().map(|value| value.and_then(Value::as_i64)),
        )),
        DataType::Float64 => Arc::new(Float64Array::from_iter(
            values.iter().map(|value| value.and_then(Value::as_f64)),
        )),
        DataType::Struct(fields) => {
            let children = fields
                .iter()
                .map(|field| {
                    let children: Vec<_> = values
                        .iter()
                        .map(|value| {
                            value
                                .and_then(|object| object.get(field.name()))
                                .filter(|value| !value.is_null())
                        })
                        .collect();
                    build_array(field.data_type(), &children)
                })
                .collect::<Result<Vec<_>>>()?;
            Arc::new(StructArray::try_new(
                fields.clone(),
                children,
                validity(values),
            )?)
        }
        DataType::List(field) => {
            let mut elements = Vec::new();
            let mut offsets = Vec::with_capacity(values.len() + 1);
            offsets.push(0);
            for value in values {
                if let Some(items) = value.and_then(Value::as_array) {
                    elements.extend(
                        items.iter().map(|item| (!item.is_null()).then_some(item)),
                    );
                }
                offsets.push(i32::try_from(elements.len()).map_err(|_| {
                    exec_datafusion_err!(
                        "structured output exceeds Arrow List offset limits"
                    )
                })?);
            }
            Arc::new(ListArray::try_new(
                Arc::clone(field),
                OffsetBuffer::new(ScalarBuffer::from(offsets)),
                build_array(field.data_type(), &elements)?,
                validity(values),
            )?)
        }
        other => return internal_err!("unexpected response schema type {other}"),
    };
    Ok(array)
}

fn validity(values: &[Option<&Value>]) -> Option<NullBuffer> {
    values
        .iter()
        .any(Option::is_none)
        .then(|| NullBuffer::from_iter(values.iter().map(Option::is_some)))
}

#[cfg(feature = "http")]
pub(super) fn openai_response_format(schema: &Value) -> Result<Value> {
    use serde_json::json;

    let schema: SchemaInput =
        SchemaInput::deserialize(schema).map_err(|error| schema_parse_error(&error))?;
    schema.compile()?;
    Ok(json!({
        "type": "json_schema",
        "json_schema": {
            "name": "df_response",
            "strict": true,
            "schema": schema.into_strict_json(false)
        }
    }))
}

#[cfg(feature = "http")]
impl SchemaInput {
    fn into_strict_json(mut self, optional: bool) -> Value {
        let mut object = Map::new();
        if let Some(properties) = self.properties {
            let required: HashSet<_> = self
                .required
                .take()
                .unwrap_or_default()
                .into_iter()
                .collect();
            let mut keys = Vec::with_capacity(properties.0.len());
            let mut children = Map::new();
            for (name, schema) in properties.0 {
                let child = schema.into_strict_json(!required.contains(&name));
                keys.push(Value::String(name.clone()));
                children.insert(name, child);
            }
            object.insert("properties".into(), Value::Object(children));
            object.insert("required".into(), Value::Array(keys));
            object.insert("additionalProperties".into(), Value::Bool(false));
        }
        if let Some(items) = self.items {
            object.insert("items".into(), items.into_strict_json(false));
        }
        if optional && !self.data_type.is_array() {
            self.data_type =
                Value::Array(vec![self.data_type, Value::String("null".into())]);
        }
        if let Some(choices) = self.choices {
            let mut choices: Vec<_> = choices.into_iter().map(Value::String).collect();
            if self.data_type.is_array() {
                choices.push(Value::Null);
            }
            object.insert("enum".into(), Value::Array(choices));
        }
        object.insert("type".into(), self.data_type);
        Value::Object(object)
    }
}
