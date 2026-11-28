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

#[cfg(feature = "runtime")]
mod object_store_resolver {
    use std::fmt;
    use std::sync::Arc;

    use bytes::BytesMut;
    use datafusion_common::{DataFusionError, exec_datafusion_err, exec_err};
    use datafusion_execution::object_store::ObjectStoreRegistry;
    use futures::TryStreamExt;
    use object_store::path::Path;
    use object_store::{Attribute, GetOptions, GetRange, ObjectStoreExt};
    use url::Url;

    use super::*;
    use crate::sanitize::sanitized_detail;

    /// Resolves references with a DataFusion runtime's object store registry.
    ///
    /// # Public Only for Internal Use:
    /// This is not a public API and is for internal use only; see [API policy] for details.
    /// [API policy]: https://datafusion.apache.org/contributor-guide/api-health.html#internal-public-apis
    #[doc(hidden)]
    #[derive(Debug)]
    pub struct ObjectStoreMediaResolver {
        stores: Arc<dyn ObjectStoreRegistry>,
    }

    impl ObjectStoreMediaResolver {
        /// Use the same object stores as the session that reads the media references.
        ///
        /// ```
        /// use std::sync::Arc;
        /// use datafusion_ai::ObjectStoreMediaResolver;
        /// use datafusion_execution::{config::SessionConfig, runtime_env::RuntimeEnv};
        ///
        /// let env = RuntimeEnv::default();
        /// let resolver = ObjectStoreMediaResolver::new(Arc::clone(&env.object_store_registry));
        /// let config = SessionConfig::new().with_extension(Arc::new(resolver));
        /// ```
        pub fn new(stores: Arc<dyn ObjectStoreRegistry>) -> Self {
            Self { stores }
        }
    }

    #[async_trait]
    impl MediaResolver for ObjectStoreMediaResolver {
        async fn resolve(
            &self,
            media: &MediaRef,
            max_bytes: Option<u64>,
        ) -> Result<ResolvedMedia> {
            let range = media.byte_range()?;
            let uri = Url::parse(&media.uri)
                .map_err(|_| exec_datafusion_err!("invalid object store media URI"))?;
            if uri.cannot_be_a_base()
                || !uri.username().is_empty()
                || uri.password().is_some()
                || uri.query().is_some()
                || uri.fragment().is_some()
            {
                return exec_err!("invalid object store media URI");
            }
            let resolved = |bytes, mime: Option<String>, etag| ResolvedMedia {
                bytes,
                mime: mime.or_else(|| media.mime.clone()),
                etag,
            };
            if media.byte_len == Some(0) && media.etag.is_none() {
                return Ok(resolved(Bytes::new(), None, None));
            }
            if max_bytes.is_none() && !matches!(range, MediaByteRange::Bounded(_)) {
                return exec_err!("unbounded media reads require max_bytes");
            }
            let store = self.stores.get_store(&uri).map_err(|error| {
                let message = sanitized_detail(&error.strip_backtrace())
                    .unwrap_or_else(|| "could not resolve the media object store".into());
                if is_transient_media_error(&error) {
                    DataFusionError::External(Box::new(TransientMediaError(message)))
                } else {
                    exec_datafusion_err!("{message}")
                }
            })?;
            let path = Path::from_url_path(uri.path())
                .map_err(|_| exec_datafusion_err!("invalid media object path"))?;
            let meta = store
                .head(&path)
                .await
                .map_err(|error| store_error(&error))?;
            check_etag(media.etag.as_deref(), meta.e_tag.as_deref())?;
            if media.byte_len == Some(0) {
                return Ok(resolved(Bytes::new(), None, meta.e_tag));
            }
            let range = match range {
                MediaByteRange::Full => 0..meta.size,
                MediaByteRange::FromOffset(start) => start..meta.size,
                MediaByteRange::Bounded(range) => range,
            };
            if range.start > range.end || range.end > meta.size {
                return exec_err!(
                    "media range is outside the object length of {} bytes",
                    meta.size
                );
            }
            let len = range.end - range.start;
            if let Some(max) = max_bytes
                && len > max
            {
                return exec_err!("media range of {len} bytes exceeds max_bytes {max}");
            }
            let len = usize::try_from(len).map_err(|_| {
                exec_datafusion_err!("media range does not fit in memory")
            })?;
            if len == 0 {
                return Ok(resolved(Bytes::new(), None, meta.e_tag));
            }

            let response = store
                .get_opts(
                    &path,
                    GetOptions {
                        range: Some(GetRange::Bounded(range.clone())),
                        if_match: meta.e_tag.clone(),
                        version: meta.version,
                        ..Default::default()
                    },
                )
                .await
                .map_err(|error| store_error(&error))?;
            let etag = response.meta.e_tag.clone();
            check_etag(meta.e_tag.as_deref(), etag.as_deref())?;
            if response.range != range {
                return exec_err!("media store returned an unexpected byte range");
            }
            let mime = response
                .attributes
                .get(&Attribute::ContentType)
                .map(|mime| mime.as_ref().to_owned());
            let bytes = response
                .into_stream()
                .map_err(|error| store_error(&error))
                .try_fold(BytesMut::new(), |mut bytes, chunk| async move {
                    if chunk.len() > len - bytes.len() {
                        return exec_err!(
                            "media store exceeded the requested {len} bytes"
                        );
                    }
                    bytes.extend_from_slice(&chunk);
                    Ok(bytes)
                })
                .await?;
            if bytes.len() != len {
                return exec_err!(
                    "media store returned {} bytes, expected {len}",
                    bytes.len()
                );
            }
            Ok(resolved(bytes.freeze(), mime, etag))
        }
    }

    fn check_etag(expected: Option<&str>, observed: Option<&str>) -> Result<()> {
        if expected.is_some() && expected != observed {
            return exec_err!("media ETag mismatch");
        }
        Ok(())
    }

    #[derive(Debug)]
    struct TransientMediaError(String);

    impl fmt::Display for TransientMediaError {
        fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
            f.write_str(&self.0)
        }
    }

    impl std::error::Error for TransientMediaError {}

    fn transient_store_error(error: &object_store::Error) -> bool {
        matches!(
            error,
            object_store::Error::Generic { .. } | object_store::Error::JoinError { .. }
        )
    }

    pub(in crate::media) fn is_transient_media_error(error: &DataFusionError) -> bool {
        match error.find_root() {
            DataFusionError::ObjectStore(error) => transient_store_error(error),
            DataFusionError::External(error) => {
                error.is::<TransientMediaError>()
                    || error
                        .downcast_ref::<object_store::Error>()
                        .is_some_and(transient_store_error)
            }
            DataFusionError::IoError(error) => matches!(
                error.kind(),
                std::io::ErrorKind::TimedOut
                    | std::io::ErrorKind::Interrupted
                    | std::io::ErrorKind::WouldBlock
                    | std::io::ErrorKind::ConnectionAborted
                    | std::io::ErrorKind::ConnectionReset
                    | std::io::ErrorKind::ConnectionRefused
            ),
            _ => false,
        }
    }

    fn store_error(error: &object_store::Error) -> DataFusionError {
        let fallback = match error {
            object_store::Error::NotFound { .. } => "media object not found",
            object_store::Error::Precondition { .. } => "media ETag precondition failed",
            _ => "media object store read failed",
        };
        let message =
            sanitized_detail(&error.to_string()).unwrap_or_else(|| fallback.into());
        if transient_store_error(error) {
            DataFusionError::External(Box::new(TransientMediaError(message)))
        } else {
            exec_datafusion_err!("{message}")
        }
    }
}

#[cfg(feature = "runtime")]
pub use object_store_resolver::ObjectStoreMediaResolver;
#[cfg(feature = "runtime")]
pub(super) fn is_transient_media_error(
    error: &datafusion_common::DataFusionError,
) -> bool {
    object_store_resolver::is_transient_media_error(error)
}
