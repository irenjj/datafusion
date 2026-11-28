// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: See the NOTICE file distributed with this work for additional information regarding copyright ownership
// SPDX-FileContributor: Licensed to the Apache Software Foundation (ASF) under one or more contributor license agreements

//! Cross-crate AI wire contracts used by DataFusion.

use arrow::datatypes::{DataType, Field};
use arrow_schema::{ArrowError, extension::ExtensionType};
use datafusion_common::{Result, plan_err};
use serde::{Deserialize, Serialize};

mod checkpoint;
#[cfg(feature = "runtime")]
pub mod llm;
mod media;
#[cfg(feature = "runtime")]
mod sanitize;

pub use checkpoint::{
    CheckpointOutputColumn, CheckpointableTarget, validate_checkpoint_target,
};
#[cfg(feature = "runtime")]
pub use media::ObjectStoreMediaResolver;
pub use media::{MediaByteRange, MediaResolver, ResolvedMedia};

/// Stable wire codes for inference failures.
///
/// # Public Only for Internal Use:
///
/// This is not a public API and is for internal use only; see [API policy] for details.
/// [API policy]: https://datafusion.apache.org/contributor-guide/api-health.html#internal-public-apis
#[doc(hidden)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "SCREAMING_SNAKE_CASE")]
pub enum ErrorCode {
    /// The provider throttled the request.
    RateLimited,
    /// The HTTP request failed with a transient error.
    TransientHttp,
    /// The request exceeded its deadline.
    Timeout,
    /// The HTTP request failed with a permanent error.
    PermanentHttp,
    /// The response could not be decoded.
    InvalidResponse,
    /// The response did not match the expected schema.
    SchemaMismatch,
    /// The request was cancelled.
    Cancelled,
}

/// A success or failure row, stored in Arrow as [`AiErrorType`].
///
/// In the same column, `error_code IS NOT NULL` identifies a failure row, and
/// all three error fields are NULL for a success row.
///
/// # Public Only for Internal Use:
///
/// This is not a public API and is for internal use only; see [API policy] for details.
/// [API policy]: https://datafusion.apache.org/contributor-guide/api-health.html#internal-public-apis
#[doc(hidden)]
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct AiError<T> {
    /// Value associated with the row, if available.
    pub value: Option<T>,
    /// Stable failure classification.
    pub error_code: Option<ErrorCode>,
    /// Diagnostic for the failed row.
    pub error_message: Option<String>,
    /// Whether the failure can be retried.
    pub retryable: Option<bool>,
}

impl<T> AiError<T> {
    /// Validate that the error fields are either all present or all absent before writing.
    ///
    /// # Errors
    /// Returns a planning error if only some error fields are present.
    ///
    /// # Public Only for Internal Use:
    /// This is not a public API and is for internal use only; see [API policy] for details.
    /// [API policy]: https://datafusion.apache.org/contributor-guide/api-health.html#internal-public-apis
    #[doc(hidden)]
    pub fn validate(&self) -> Result<()> {
        match (&self.error_code, &self.error_message, &self.retryable) {
            (Some(_), Some(_), Some(_)) | (None, None, None) => Ok(()),
            _ => plan_err!(
                "df.ai_error fields error_code, error_message, and retryable must all be present or all absent"
            ),
        }
    }
}

/// A reference to media bytes, stored in Arrow as [`MediaRefType`].
///
/// # Public Only for Internal Use:
///
/// This is not a public API and is for internal use only; see [API policy] for details.
/// [API policy]: https://datafusion.apache.org/contributor-guide/api-health.html#internal-public-apis
#[doc(hidden)]
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct MediaRef {
    /// URI of the object containing the bytes.
    pub uri: String,
    /// Start of the range in bytes, when specified.
    pub byte_offset: Option<u64>,
    /// Length of the range in bytes, when specified.
    pub byte_len: Option<u64>,
    /// Object ETag, when available.
    pub etag: Option<String>,
    /// Media type, when known.
    pub mime: Option<String>,
}

/// Arrow extension for an [`AiError`] with a caller-selected value type.
///
/// # Public Only for Internal Use:
///
/// This is not a public API and is for internal use only; see [API policy] for details.
/// [API policy]: https://datafusion.apache.org/contributor-guide/api-health.html#internal-public-apis
#[doc(hidden)]
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AiErrorType {
    value_type: DataType,
}

impl AiErrorType {
    /// Create an extension for values of the given Arrow type.
    #[must_use]
    pub fn new(value_type: DataType) -> Self {
        Self { value_type }
    }

    /// Build a field carrying the extension metadata.
    ///
    /// # Errors
    ///
    /// Returns an error if Arrow rejects the storage type.
    pub fn field(&self, name: &str, nullable: bool) -> Result<Field> {
        let mut field = Field::new(name, self.storage_type(), nullable);
        field.try_with_extension_type(self.clone())?;
        Ok(field)
    }

    fn storage_type(&self) -> DataType {
        DataType::Struct(
            vec![
                Field::new("value", self.value_type.clone(), true),
                Field::new("error_code", DataType::Utf8, true),
                Field::new("error_message", DataType::Utf8, true),
                Field::new("retryable", DataType::Boolean, true),
            ]
            .into(),
        )
    }
}

impl ExtensionType for AiErrorType {
    const NAME: &'static str = "df.ai_error";
    type Metadata = u8;

    fn metadata(&self) -> &Self::Metadata {
        &EXTENSION_VERSION
    }

    fn serialize_metadata(&self) -> Option<String> {
        Some(serialize_version())
    }

    fn deserialize_metadata(
        metadata: Option<&str>,
    ) -> Result<Self::Metadata, ArrowError> {
        deserialize_version(metadata)
    }

    fn supports_data_type(&self, data_type: &DataType) -> Result<(), ArrowError> {
        validate_storage(Self::NAME, data_type, &self.storage_type())
    }

    fn try_new(
        data_type: &DataType,
        version: Self::Metadata,
    ) -> Result<Self, ArrowError> {
        validate_version(version)?;
        if let DataType::Struct(fields) = data_type
            && let Some(value) = fields.first()
        {
            let extension = Self::new(value.data_type().clone());
            extension.supports_data_type(data_type)?;
            return Ok(extension);
        }
        Err(ArrowError::InvalidArgumentError(format!(
            "{} requires a struct with a value field, got {data_type}",
            Self::NAME
        )))
    }
}

/// Arrow extension for a [`MediaRef`].
///
/// # Public Only for Internal Use:
///
/// This is not a public API and is for internal use only; see [API policy] for details.
/// [API policy]: https://datafusion.apache.org/contributor-guide/api-health.html#internal-public-apis
#[doc(hidden)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MediaRefType;

impl MediaRefType {
    /// Build a field carrying the extension metadata.
    ///
    /// # Errors
    ///
    /// Returns an error if Arrow rejects the storage type.
    pub fn field(name: &str, nullable: bool) -> Result<Field> {
        let mut field = Field::new(name, Self::storage_type(), nullable);
        field.try_with_extension_type(Self)?;
        Ok(field)
    }

    fn storage_type() -> DataType {
        DataType::Struct(
            vec![
                Field::new("uri", DataType::Utf8, false),
                Field::new("byte_offset", DataType::UInt64, true),
                Field::new("byte_len", DataType::UInt64, true),
                Field::new("etag", DataType::Utf8, true),
                Field::new("mime", DataType::Utf8, true),
            ]
            .into(),
        )
    }
}

impl ExtensionType for MediaRefType {
    const NAME: &'static str = "df.media_ref";
    type Metadata = u8;

    fn metadata(&self) -> &Self::Metadata {
        &EXTENSION_VERSION
    }

    fn serialize_metadata(&self) -> Option<String> {
        Some(serialize_version())
    }

    fn deserialize_metadata(
        metadata: Option<&str>,
    ) -> Result<Self::Metadata, ArrowError> {
        deserialize_version(metadata)
    }

    fn supports_data_type(&self, data_type: &DataType) -> Result<(), ArrowError> {
        validate_storage(Self::NAME, data_type, &Self::storage_type())
    }

    fn try_new(
        data_type: &DataType,
        version: Self::Metadata,
    ) -> Result<Self, ArrowError> {
        validate_version(version)?;
        Self.supports_data_type(data_type)?;
        Ok(Self)
    }
}

const EXTENSION_VERSION: u8 = 1;

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct ExtensionMetadata {
    version: u8,
}

fn serialize_version() -> String {
    serde_json::json!({"version": EXTENSION_VERSION}).to_string()
}

fn deserialize_version(metadata: Option<&str>) -> Result<u8, ArrowError> {
    let metadata = metadata.ok_or_else(|| {
        ArrowError::InvalidArgumentError("missing AI extension metadata".to_owned())
    })?;
    if !metadata.trim_start().starts_with('{') {
        return Err(ArrowError::InvalidArgumentError(
            "AI extension metadata must be a JSON object".to_owned(),
        ));
    }
    let metadata: ExtensionMetadata =
        serde_json::from_str(metadata).map_err(|error| {
            ArrowError::InvalidArgumentError(format!(
                "invalid AI extension metadata: {error}"
            ))
        })?;
    validate_version(metadata.version)?;
    Ok(metadata.version)
}

fn validate_version(version: u8) -> Result<(), ArrowError> {
    if version == EXTENSION_VERSION {
        Ok(())
    } else {
        Err(ArrowError::InvalidArgumentError(format!(
            "unsupported AI extension metadata version {version}"
        )))
    }
}

fn validate_storage(
    name: &str,
    actual: &DataType,
    expected: &DataType,
) -> Result<(), ArrowError> {
    if actual == expected {
        Ok(())
    } else {
        Err(ArrowError::InvalidArgumentError(format!(
            "{name} requires {expected}, got {actual}"
        )))
    }
}
