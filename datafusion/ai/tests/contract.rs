// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: See the NOTICE file distributed with this work for additional information regarding copyright ownership
// SPDX-FileContributor: Licensed to the Apache Software Foundation (ASF) under one or more contributor license agreements

use std::fmt::Debug;
use std::future::Future;
use std::io::Cursor;
use std::pin::pin;
use std::sync::Arc;
use std::task::{Context, Poll, Waker};

use arrow::array::{
    BooleanArray, FixedSizeListArray, Float32Array, RecordBatch, StringArray, StructArray,
};
use arrow::buffer::NullBuffer;
use arrow::datatypes::{DataType, Field, Schema};
use arrow::ipc::{reader::StreamReader, writer::StreamWriter};
use arrow_schema::extension::{EXTENSION_TYPE_METADATA_KEY, EXTENSION_TYPE_NAME_KEY};
use async_trait::async_trait;
use bytes::Bytes;
use datafusion_ai::{
    AiError, AiErrorType, CheckpointOutputColumn, CheckpointableTarget, ErrorCode,
    MediaByteRange, MediaRef, MediaRefType, MediaResolver, ResolvedMedia,
    validate_checkpoint_target,
};
use datafusion_common::{DataFusionError, Result, plan_err};
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
        let envelope = AiError {
            value,
            error_code: Some(error_code),
            error_message: Some(error_message),
            retryable: Some(retryable),
        };
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
            Field::new("error_code", DataType::Utf8, true),
            Field::new("error_message", DataType::Utf8, true),
            Field::new("retryable", DataType::Boolean, true),
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
fn ai_error_success_and_failure_round_trip() {
    let success = AiError {
        value: Some("result".to_owned()),
        error_code: None,
        error_message: None,
        retryable: None,
    };
    let failure = AiError {
        value: None,
        error_code: Some(ErrorCode::Timeout),
        error_message: Some("request timed out".to_owned()),
        retryable: Some(false),
    };
    for (envelope, expected) in [
        (
            &success,
            json!({
                "value": "result",
                "error_code": null,
                "error_message": null,
                "retryable": null,
            }),
        ),
        (
            &failure,
            json!({
                "value": null,
                "error_code": "TIMEOUT",
                "error_message": "request timed out",
                "retryable": false,
            }),
        ),
    ] {
        envelope.validate().unwrap();
        let wire = serde_json::to_value(envelope).unwrap();
        assert_eq!(wire, expected);
        assert_eq!(
            serde_json::from_value::<AiError<String>>(wire).unwrap(),
            *envelope
        );
    }

    let field = AiErrorType::new(DataType::Utf8)
        .field("result", false)
        .unwrap();
    let DataType::Struct(fields) = field.data_type() else {
        panic!("expected struct storage");
    };
    let column = StructArray::try_new(
        fields.clone(),
        vec![
            Arc::new(StringArray::from(vec![Some("result"), None])),
            Arc::new(StringArray::from(vec![None, Some("TIMEOUT")])),
            Arc::new(StringArray::from(vec![None, Some("request timed out")])),
            Arc::new(BooleanArray::from(vec![None, Some(false)])),
        ],
        None,
    )
    .unwrap();
    let schema = Arc::new(Schema::new(vec![field]));
    let batch = RecordBatch::try_new(schema.clone(), vec![Arc::new(column)]).unwrap();
    let mut writer = StreamWriter::try_new(Vec::new(), &schema).unwrap();
    writer.write(&batch).unwrap();
    let mut reader =
        StreamReader::try_new(Cursor::new(writer.into_inner().unwrap()), None).unwrap();
    assert_eq!(reader.next().unwrap().unwrap(), batch);
}

#[test]
fn ai_error_validate_rejects_partial_error_fields() {
    let partial: AiError<String> = serde_json::from_value(json!({
        "value": null,
        "error_code": "TIMEOUT",
        "error_message": null,
        "retryable": false,
    }))
    .unwrap();

    assert!(matches!(partial.validate(), Err(DataFusionError::Plan(_))));
}

#[test]
fn ai_error_with_fixed_size_list_values_round_trips_through_ipc() -> Result<()> {
    let item = Arc::new(Field::new_list_field(DataType::Float32, false));
    let field = AiErrorType::new(DataType::FixedSizeList(Arc::clone(&item), 2))
        .field("embedding", false)?;
    let DataType::Struct(fields) = field.data_type() else {
        unreachable!()
    };
    let values = FixedSizeListArray::try_new(
        item,
        2,
        Arc::new(Float32Array::from(vec![1.0, 2.0, 0.0, 0.0])),
        Some(NullBuffer::from(vec![true, false])),
    )?;
    let envelopes = StructArray::try_new(
        fields.clone(),
        vec![
            Arc::new(values),
            Arc::new(StringArray::from(vec![None, Some("SCHEMA_MISMATCH")])),
            Arc::new(StringArray::from(vec![None, Some("wrong dimension")])),
            Arc::new(BooleanArray::from(vec![None, Some(false)])),
        ],
        None,
    )?;
    let schema = Arc::new(Schema::new(vec![field]));
    let batch = RecordBatch::try_new(Arc::clone(&schema), vec![Arc::new(envelopes)])?;
    let mut writer = StreamWriter::try_new(Vec::new(), &schema)?;
    writer.write(&batch)?;
    let reader = StreamReader::try_new(Cursor::new(writer.into_inner()?), None)?;
    assert_eq!(reader.collect::<std::result::Result<Vec<_>, _>>()?, [batch]);
    Ok(())
}

#[test]
fn ai_error_arrow_fields_are_nullable() {
    for nullable in [false, true] {
        let field = AiErrorType::new(DataType::Utf8)
            .field("result", nullable)
            .unwrap();
        assert_eq!(field.is_nullable(), nullable);
        let DataType::Struct(fields) = field.data_type() else {
            panic!("expected struct storage");
        };
        for name in ["value", "error_code", "error_message", "retryable"] {
            assert!(fields.find(name).unwrap().1.is_nullable(), "{name}");
        }
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

fn assert_plan_error<T: Debug>(result: Result<T>, expected: &str) {
    let error = result.unwrap_err();
    let DataFusionError::Plan(message) = error else {
        panic!("expected a planning error, got {error}");
    };
    assert!(message.contains(expected), "{message}");
}

fn media_ref(byte_offset: Option<u64>, byte_len: Option<u64>) -> MediaRef {
    MediaRef {
        uri: "memory://media".to_owned(),
        byte_offset,
        byte_len,
        etag: None,
        mime: None,
    }
}

#[test]
fn media_ref_rejects_invalid_fields() {
    for blank in ["", " \t\n\u{2003}"] {
        let media = MediaRef {
            uri: blank.to_owned(),
            ..media_ref(None, None)
        };
        assert_plan_error(media.validate(), "URI");
        assert_plan_error(media.byte_range(), "URI");

        let media = MediaRef {
            mime: Some(blank.to_owned()),
            ..media_ref(None, None)
        };
        assert_plan_error(media.validate(), "MIME type");
        assert_plan_error(media.byte_range(), "MIME type");
    }
    for (offset, len) in [(u64::MAX, 1), (1, u64::MAX)] {
        let media = media_ref(Some(offset), Some(len));
        assert_plan_error(media.validate(), "overflows");
        assert_plan_error(media.byte_range(), "overflows");
    }
}

#[test]
fn media_ref_normalizes_ranges() {
    use MediaByteRange::{Bounded, FromOffset, Full};

    for (offset, len, expected) in [
        (None, None, Full),
        (Some(5), None, FromOffset(5)),
        (Some(u64::MAX), None, FromOffset(u64::MAX)),
        (None, Some(0), Bounded(0..0)),
        (Some(0), Some(0), Bounded(0..0)),
        (Some(u64::MAX), Some(0), Bounded(u64::MAX..u64::MAX)),
        (None, Some(4), Bounded(0..4)),
        (Some(2), Some(4), Bounded(2..6)),
        (Some(1), Some(u64::MAX - 1), Bounded(1..u64::MAX)),
    ] {
        let media = media_ref(offset, len);
        media.validate().unwrap();
        assert_eq!(media.byte_range().unwrap(), expected);
    }
}

#[test]
fn media_ref_accepts_opaque_etags() {
    for etag in ["", "\"abc\"", "W/\"weak\"", " 影片🦀\0\n"] {
        let media = MediaRef {
            etag: Some(etag.to_owned()),
            ..media_ref(None, None)
        };
        media.validate().unwrap();
        assert_eq!(media.byte_range().unwrap(), MediaByteRange::Full);
    }
}

#[derive(Debug)]
struct InMemoryMediaResolver(ResolvedMedia);

#[async_trait]
impl MediaResolver for InMemoryMediaResolver {
    async fn resolve(
        &self,
        media: &MediaRef,
        max_bytes: Option<u64>,
    ) -> Result<ResolvedMedia> {
        let object_len = self.0.bytes.len() as u64;
        let range = match media.byte_range()? {
            MediaByteRange::Full => 0..object_len,
            MediaByteRange::FromOffset(offset) => offset..object_len,
            MediaByteRange::Bounded(range) => range,
        };
        if media.etag.is_some() && media.etag != self.0.etag {
            return plan_err!("media ETag mismatch");
        }
        if range.start > range.end {
            return plan_err!("media range starts past EOF");
        }
        let len = range.end - range.start;
        if max_bytes.is_some_and(|limit| len > limit) {
            return plan_err!("media range exceeds max_bytes");
        }
        let bytes = if len == 0 {
            Bytes::new()
        } else {
            if range.end > object_len {
                return plan_err!("media range ends past EOF");
            }
            self.0.bytes.slice(range.start as usize..range.end as usize)
        };
        Ok(ResolvedMedia {
            bytes,
            mime: self.0.mime.clone(),
            etag: self.0.etag.clone(),
        })
    }
}

const OBJECT_ETAG: &str = "W/\"影片\"";

fn in_memory_resolver() -> Arc<dyn MediaResolver> {
    Arc::new(InMemoryMediaResolver(ResolvedMedia {
        bytes: Bytes::from_static(b"abcdef"),
        mime: Some("video/test".to_owned()),
        etag: Some(OBJECT_ETAG.to_owned()),
    }))
}

fn poll_ready<F: Future>(future: F) -> F::Output {
    let mut future = pin!(future);
    let Poll::Ready(output) = future
        .as_mut()
        .poll(&mut Context::from_waker(Waker::noop()))
    else {
        panic!("in-memory resolver must be ready on the first poll");
    };
    output
}

#[test]
fn media_resolver_reads_ranges_and_metadata() {
    let resolver = in_memory_resolver();
    for (offset, len, expected) in [
        (None, None, b"abcdef".as_slice()),
        (Some(2), None, b"cdef"),
        (Some(1), Some(3), b"bcd"),
        (Some(u64::MAX), Some(0), b""),
    ] {
        let media = media_ref(offset, len);
        let resolved = poll_ready(resolver.resolve(&media, None)).unwrap();
        assert_eq!(resolved.bytes.as_ref(), expected);
        assert_eq!(resolved.mime.as_deref(), Some("video/test"));
        assert_eq!(resolved.etag.as_deref(), Some(OBJECT_ETAG));
    }
}

#[test]
fn media_resolver_enforces_max_bytes() {
    let resolver = in_memory_resolver();
    let media = media_ref(None, None);
    assert_eq!(
        poll_ready(resolver.resolve(&media, Some(6))).unwrap().bytes,
        Bytes::from_static(b"abcdef")
    );
    assert_plan_error(poll_ready(resolver.resolve(&media, Some(5))), "max_bytes");

    let media = media_ref(Some(3), Some(0));
    assert!(
        poll_ready(resolver.resolve(&media, Some(0)))
            .unwrap()
            .bytes
            .is_empty()
    );
}

#[test]
fn media_resolver_validates_reference_and_etag() {
    let resolver = in_memory_resolver();
    let mut media = MediaRef {
        uri: String::new(),
        ..media_ref(None, None)
    };
    assert_plan_error(poll_ready(resolver.resolve(&media, None)), "URI");
    let overflow = media_ref(Some(u64::MAX), Some(1));
    assert_plan_error(poll_ready(resolver.resolve(&overflow, None)), "overflows");

    media = media_ref(None, None);
    media.etag = Some(OBJECT_ETAG.to_owned());
    assert_eq!(
        poll_ready(resolver.resolve(&media, None))
            .unwrap()
            .etag
            .as_deref(),
        Some(OBJECT_ETAG)
    );
    media.etag = Some("\"影片\"".to_owned());
    assert_plan_error(poll_ready(resolver.resolve(&media, None)), "ETag mismatch");
}

#[derive(Debug)]
struct TestCheckpointTarget {
    passthrough_count: usize,
    columns: Vec<CheckpointOutputColumn>,
}

impl CheckpointableTarget for TestCheckpointTarget {
    fn passthrough_column_count(&self) -> usize {
        self.passthrough_count
    }

    fn ai_output_columns(&self) -> Vec<CheckpointOutputColumn> {
        self.columns.clone()
    }

    fn preserves_input_order(&self) -> bool {
        true
    }
}

fn checkpoint_column(
    index: usize,
    name: &str,
    is_error_envelope: bool,
) -> CheckpointOutputColumn {
    CheckpointOutputColumn {
        index,
        name: name.to_owned(),
        is_error_envelope,
    }
}

#[test]
fn checkpoint_target_accepts_valid_columns() {
    let target = TestCheckpointTarget {
        passthrough_count: 0,
        columns: vec![checkpoint_column(0, "score", false)],
    };
    let score = Field::new("score", DataType::Float64, true);
    validate_checkpoint_target(&target, &Schema::new(vec![score.clone()])).unwrap();
    assert!(target.preserves_row_cardinality());

    let target = TestCheckpointTarget {
        passthrough_count: 2,
        columns: vec![
            checkpoint_column(3, "score", false),
            checkpoint_column(4, "result", true),
        ],
    };
    let schema = Schema::new(vec![
        Field::new("id", DataType::UInt64, false),
        Field::new("uri", DataType::Utf8, false),
        Field::new("group", DataType::Utf8, false),
        score,
        AiErrorType::new(DataType::Utf8)
            .field("result", true)
            .unwrap(),
    ]);
    validate_checkpoint_target(&target, &schema).unwrap();
}

#[test]
fn checkpoint_target_rejects_invalid_column_declarations() {
    let schema = Schema::new(vec![
        Field::new("id", DataType::UInt64, false),
        Field::new("score", DataType::Float64, true),
        Field::new("label", DataType::Utf8, true),
    ]);
    let score = checkpoint_column(1, "score", false);
    for (passthrough_count, columns, expected) in [
        (4, vec![score.clone()], "exceeds output schema"),
        (1, vec![], "at least one AI output"),
        (1, vec![checkpoint_column(0, "id", false)], "overlaps"),
        (
            1,
            vec![checkpoint_column(3, "result", false)],
            "out of bounds",
        ),
        (1, vec![score.clone(), score.clone()], "strictly increasing"),
        (
            1,
            vec![checkpoint_column(2, "label", false), score],
            "strictly increasing",
        ),
        (
            1,
            vec![checkpoint_column(1, "label", false)],
            "declares name",
        ),
    ] {
        let target = TestCheckpointTarget {
            passthrough_count,
            columns,
        };
        assert_plan_error(validate_checkpoint_target(&target, &schema), expected);
    }
}

#[test]
fn checkpoint_target_rejects_invalid_error_envelopes() {
    let envelope = AiErrorType::new(DataType::Utf8)
        .field("result", true)
        .unwrap();
    let mut wrong_version = envelope.clone();
    wrong_version
        .metadata_mut()
        .insert(EXTENSION_TYPE_METADATA_KEY, r#"{"version":2}"#);
    let target = TestCheckpointTarget {
        passthrough_count: 0,
        columns: vec![checkpoint_column(0, "result", true)],
    };
    for field in [
        Field::new("result", DataType::Utf8, true),
        Field::new("result", envelope.data_type().clone(), true),
        MediaRefType::field("result", true).unwrap(),
        wrong_version,
    ] {
        assert_plan_error(
            validate_checkpoint_target(&target, &Schema::new(vec![field])),
            "must use the df.ai_error extension",
        );
    }
}

proptest! {
    #[test]
    fn media_ref_range_matches_wide_arithmetic(
        byte_offset in proptest::option::of(any::<u64>()),
        byte_len in proptest::option::of(any::<u64>()),
        etag in proptest::option::of(any::<String>()),
    ) {
        let media = MediaRef { etag, ..media_ref(byte_offset, byte_len) };
        let start = byte_offset.unwrap_or(0);
        let expected = match byte_len {
            Some(len) => u64::try_from(u128::from(start) + u128::from(len))
                .map(|end| MediaByteRange::Bounded(start..end)),
            None => Ok(byte_offset.map_or(MediaByteRange::Full, MediaByteRange::FromOffset)),
        };
        match expected {
            Ok(range) => {
                prop_assert!(media.validate().is_ok());
                prop_assert_eq!(media.byte_range().unwrap(), range);
            }
            Err(_) => {
                prop_assert!(matches!(media.validate(), Err(DataFusionError::Plan(_))));
                prop_assert!(matches!(media.byte_range(), Err(DataFusionError::Plan(_))));
            }
        }
    }

    #[test]
    fn checkpoint_target_accepts_generated_schema(
        passthrough_count in 0usize..8,
        outputs in proptest::collection::vec(
            (
                "[a-z_]{1,12}",
                proptest::sample::select(vec![DataType::Utf8, DataType::UInt64, DataType::Boolean]),
                any::<bool>(),
                any::<bool>(),
            ),
            1..8,
        ),
    ) {
        let mut fields: Vec<_> = (0..passthrough_count)
            .map(|index| Field::new(format!("input_{index}"), DataType::Utf8, false))
            .collect();
        let columns = outputs.into_iter().map(|(name, data_type, nullable, is_error_envelope)| {
            let index = fields.len();
            let field = if is_error_envelope {
                AiErrorType::new(data_type).field(&name, nullable).unwrap()
            } else {
                Field::new(&name, data_type, nullable)
            };
            fields.push(field);
            CheckpointOutputColumn { index, name, is_error_envelope }
        }).collect();
        let target = TestCheckpointTarget { passthrough_count, columns };
        prop_assert!(validate_checkpoint_target(&target, &Schema::new(fields)).is_ok());
    }
}
