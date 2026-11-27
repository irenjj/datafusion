// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: See the NOTICE file distributed with this work for additional information regarding copyright ownership
// SPDX-FileContributor: Licensed to the Apache Software Foundation (ASF) under one or more contributor license agreements

use std::{fmt::Debug, ops::Range};

use async_trait::async_trait;
use bytes::Bytes;
use datafusion_common::{Result, plan_datafusion_err, plan_err};

use crate::MediaRef;

/// Resolved byte range semantics derived from a validated [`MediaRef`].
///
/// # Public Only for Internal Use:
///
/// This is not a public API and is for internal use only; see [API policy] for details.
/// [API policy]: https://datafusion.apache.org/contributor-guide/api-health.html#internal-public-apis
#[doc(hidden)]
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum MediaByteRange {
    /// Read the entire object (`byte_offset: None, byte_len: None`).
    Full,
    /// Read from `offset` to EOF (`byte_offset: Some(offset), byte_len: None`).
    FromOffset(u64),
    /// Read the half-open range `[start, end)`, with `start == end` for zero length.
    Bounded(Range<u64>),
}

impl MediaRef {
    /// Validate the URI, byte range, and optional MIME type.
    ///
    /// # Errors
    /// Returns a planning error for a blank URI, a blank MIME type, or
    /// a bounded range whose end exceeds `u64::MAX`.
    ///
    /// # Public Only for Internal Use:
    ///
    /// This is not a public API and is for internal use only; see [API policy] for details.
    /// [API policy]: https://datafusion.apache.org/contributor-guide/api-health.html#internal-public-apis
    #[doc(hidden)]
    pub fn validate(&self) -> Result<()> {
        if self.uri.trim().is_empty() {
            return plan_err!("df.media_ref URI must not be blank");
        }
        if let Some(len) = self.byte_len {
            checked_range_end(self.byte_offset.unwrap_or(0), len)?;
        }
        if self
            .mime
            .as_ref()
            .is_some_and(|mime| mime.trim().is_empty())
        {
            return plan_err!("df.media_ref MIME type must not be blank");
        }
        Ok(())
    }

    /// Validate this reference and resolve its byte range.
    ///
    /// # Errors
    /// Returns the planning errors of [`MediaRef::validate`].
    ///
    /// # Public Only for Internal Use:
    ///
    /// This is not a public API and is for internal use only; see [API policy] for details.
    /// [API policy]: https://datafusion.apache.org/contributor-guide/api-health.html#internal-public-apis
    #[doc(hidden)]
    pub fn byte_range(&self) -> Result<MediaByteRange> {
        self.validate()?;
        match (self.byte_offset, self.byte_len) {
            (None, None) => Ok(MediaByteRange::Full),
            (Some(offset), None) => Ok(MediaByteRange::FromOffset(offset)),
            (offset, Some(len)) => {
                let start = offset.unwrap_or(0);
                let end = checked_range_end(start, len)?;
                Ok(MediaByteRange::Bounded(start..end))
            }
        }
    }
}

fn checked_range_end(start: u64, len: u64) -> Result<u64> {
    start.checked_add(len).ok_or_else(|| {
        plan_datafusion_err!(
            "df.media_ref byte range overflows u64 (offset {start}, length {len})"
        )
    })
}

/// Bytes and metadata obtained by resolving a [`MediaRef`].
///
/// # Public Only for Internal Use:
///
/// This is not a public API and is for internal use only; see [API policy] for details.
/// [API policy]: https://datafusion.apache.org/contributor-guide/api-health.html#internal-public-apis
#[doc(hidden)]
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ResolvedMedia {
    /// Resolved raw bytes, limited by the requested `max_bytes`.
    pub bytes: Bytes,
    /// Observed or declared MIME type, if known.
    pub mime: Option<String>,
    /// Observed opaque ETag from the backing store, if available.
    pub etag: Option<String>,
}

/// Asynchronously resolve a [`MediaRef`] to bytes.
///
/// # Public Only for Internal Use:
///
/// This is not a public API and is for internal use only; see [API policy] for details.
/// [API policy]: https://datafusion.apache.org/contributor-guide/api-health.html#internal-public-apis
#[doc(hidden)]
#[async_trait]
pub trait MediaResolver: Debug + Send + Sync {
    /// Resolve `media` into raw bytes.
    ///
    /// Implementations must:
    ///
    /// 1. Validate `media` via [`MediaRef::validate`].
    /// 2. Enforce `max_bytes`, when set, without allocating unbounded memory.
    /// 3. Compare a supplied `media.etag` by opaque equality when the backing
    ///    store supports ETag verification.
    /// 4. Support `byte_len == Some(0)` without failing on empty slices.
    ///
    /// # Errors
    /// Returns an error for invalid references, read failures, ETag mismatches
    /// when verification is supported, or payloads larger than `max_bytes`.
    async fn resolve(
        &self,
        media: &MediaRef,
        max_bytes: Option<u64>,
    ) -> Result<ResolvedMedia>;
}
