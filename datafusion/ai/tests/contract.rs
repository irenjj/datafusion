// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: See the NOTICE file distributed with this work for additional information regarding copyright ownership
// SPDX-FileContributor: Licensed to the Apache Software Foundation (ASF) under one or more contributor license agreements

use std::io::Cursor;

use arrow::datatypes::{DataType, Field, Schema};
use arrow::ipc::{reader::StreamReader, writer::StreamWriter};
use arrow_schema::extension::{EXTENSION_TYPE_METADATA_KEY, EXTENSION_TYPE_NAME_KEY};
use datafusion_ai::{AiError, AiErrorType, ErrorCode, MediaRef, MediaRefType};
use proptest::prelude::*;
use serde_json::{Value, json};

const ERROR_CODES: [(ErrorCode, &str); 7] = [
    (ErrorCode::RateLimited, "RATE_LIMITED"),
    (ErrorCode::TransientHttp, "TRANSIENT_HTTP"),
    (ErrorCode::Timeout, "TIMEOUT"),
    (ErrorCode::PermanentHttp, "PERMANENT_HTTP"),
    (ErrorCode::InvalidResponse, "INVALID_RESPONSE"),
    (ErrorCode::SchemaMismatch, "SCHEMA_MISMATCH"),
    (ErrorCode::Cancelled, "CANCELLED"),
];

proptest! {
    #[test]
    fn ai_error_json_round_trip(
        value in proptest::option::of(any::<String>()),
        (error_code, wire_code) in proptest::sample::select(ERROR_CODES.to_vec()),
        error_message in any::<String>(),
        retryable in any::<bool>(),
    ) {
        let envelope = AiError { value, error_code, error_message, retryable };
        let wire = serde_json::to_string(&envelope).unwrap();
        prop_assert_eq!(serde_json::from_str::<Value>(&wire).unwrap(), json!({
            "value": envelope.value,
            "error_code": wire_code,
            "error_message": envelope.error_message,
            "retryable": retryable,
        }));
        prop_assert_eq!(serde_json::from_str::<AiError<String>>(&wire).unwrap(), envelope);
    }

    #[test]
    fn media_ref_json_round_trip(
        uri in any::<String>(),
        byte_offset in proptest::option::of(any::<u64>()),
        byte_len in proptest::option::of(any::<u64>()),
        etag in proptest::option::of(any::<String>()),
        mime in proptest::option::of(any::<String>()),
    ) {
        let reference = MediaRef { uri, byte_offset, byte_len, etag, mime };
        let wire = serde_json::to_string(&reference).unwrap();
        prop_assert_eq!(serde_json::from_str::<Value>(&wire).unwrap(), json!({
            "uri": reference.uri,
            "byte_offset": byte_offset,
            "byte_len": byte_len,
            "etag": reference.etag,
            "mime": reference.mime,
        }));
        prop_assert_eq!(serde_json::from_str::<MediaRef>(&wire).unwrap(), reference);
    }

    #[test]
    fn arrow_ipc_preserves_extension_contracts(
        name in "[a-z_]{1,20}",
        nullable in any::<bool>(),
        value_type in proptest::sample::select(vec![
            DataType::Utf8, DataType::LargeUtf8, DataType::UInt64, DataType::Boolean,
            DataType::List(Field::new("item", DataType::Utf8, true).into()),
        ]),
    ) {
        let error_type = AiErrorType::new(value_type.clone());
        let schema = Schema::new(vec![
            error_type.field(&name, nullable).unwrap(),
            MediaRefType::field("media", nullable).unwrap(),
        ]);
        let writer = StreamWriter::try_new(Vec::new(), &schema).unwrap();
        let reader = StreamReader::try_new(Cursor::new(writer.into_inner().unwrap()), None).unwrap();
        let decoded = reader.schema();
        prop_assert_eq!(decoded.as_ref(), &schema);

        let error = decoded.field(0);
        prop_assert_eq!(error.extension_type_name(), Some("df.ai_error"));
        prop_assert_eq!(error.extension_type_metadata(), Some(r#"{"version":1}"#));
        prop_assert_eq!(error.try_extension_type::<AiErrorType>().unwrap(), error_type);
        prop_assert_eq!(error.data_type(), &DataType::Struct(vec![
            Field::new("value", value_type, true),
            Field::new("error_code", DataType::Utf8, false),
            Field::new("error_message", DataType::Utf8, false),
            Field::new("retryable", DataType::Boolean, false),
        ].into()));

        let media = decoded.field(1);
        prop_assert_eq!(media.extension_type_name(), Some("df.media_ref"));
        prop_assert_eq!(media.extension_type_metadata(), Some(r#"{"version":1}"#));
        prop_assert_eq!(media.try_extension_type::<MediaRefType>().unwrap(), MediaRefType);
        prop_assert_eq!(media.data_type(), &DataType::Struct(vec![
            Field::new("uri", DataType::Utf8, false),
            Field::new("byte_offset", DataType::UInt64, true),
            Field::new("byte_len", DataType::UInt64, true),
            Field::new("etag", DataType::Utf8, true),
            Field::new("mime", DataType::Utf8, true),
        ].into()));
    }
}

#[test]
fn error_code_strings_are_stable() {
    for (code, wire_code) in ERROR_CODES {
        let wire = json!(wire_code);
        assert_eq!(serde_json::to_value(code).unwrap(), wire);
        assert_eq!(serde_json::from_value::<ErrorCode>(wire).unwrap(), code);
    }
    assert!(serde_json::from_value::<ErrorCode>(json!("UNKNOWN")).is_err());
}

#[test]
fn arrow_rejects_unrecognized_metadata_and_storage() {
    let error = AiErrorType::new(DataType::Utf8)
        .field("error", true)
        .unwrap();
    assert!(error.try_extension_type::<MediaRefType>().is_err());

    for metadata in [
        None,
        Some("broken"),
        Some("[1]"),
        Some(r#"{"version":2}"#),
        Some(r#"{"version":1,"version":1}"#),
    ] {
        let mut field = error.clone();
        match metadata {
            Some(metadata) => {
                field
                    .metadata_mut()
                    .insert(EXTENSION_TYPE_METADATA_KEY, metadata);
            }
            None => {
                field.metadata_mut().remove(EXTENSION_TYPE_METADATA_KEY);
            }
        }
        assert!(field.try_extension_type::<AiErrorType>().is_err());
    }

    let malformed =
        Field::new("error", DataType::Utf8, true).with_metadata(error.metadata().clone());
    assert!(malformed.try_extension_type::<AiErrorType>().is_err());

    let mut malformed = malformed;
    malformed
        .metadata_mut()
        .insert(EXTENSION_TYPE_NAME_KEY, "df.media_ref");
    assert!(malformed.try_extension_type::<MediaRefType>().is_err());

    let DataType::Struct(fields) = error.data_type() else {
        panic!("expected struct storage");
    };
    let truncated =
        Field::new("error", DataType::Struct(fields[..3].to_vec().into()), true)
            .with_metadata(error.metadata().clone());
    assert!(truncated.try_extension_type::<AiErrorType>().is_err());
}
