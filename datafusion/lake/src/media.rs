// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: See the NOTICE file distributed with this work for additional information regarding copyright ownership
// SPDX-FileContributor: Licensed to the Apache Software Foundation (ASF) under one or more contributor license agreements

use std::ops::Range;
use std::sync::Arc;

use arrow::array::{ArrayRef, StringBuilder, StructArray, UInt64Builder};
use arrow::buffer::NullBuffer;
use arrow::datatypes::DataType;
use async_trait::async_trait;
use bytes::{Bytes, BytesMut};
use datafusion_ai::{MediaByteRange, MediaRef, MediaResolver, ResolvedMedia};
use datafusion_common::{
    DataFusionError, Result, exec_datafusion_err, exec_err, plan_datafusion_err,
};
use datafusion_execution::object_store::{
    DefaultObjectStoreRegistry, ObjectStoreRegistry,
};
use futures::TryStreamExt;
use object_store::path::Path;
use object_store::{GetOptions, GetRange, ObjectStoreExt};
use paimon::io::{FileIO, FileRead};
use url::Url;

/// Resolves lake media through the runtime's object stores or a scoped Paimon FileIO.
///
/// # Public Only for Internal Use:
///
/// This is not a public API and is for internal use only; see [API policy] for details.
/// [API policy]: https://datafusion.apache.org/contributor-guide/api-health.html#internal-public-apis
#[doc(hidden)]
#[derive(Debug)]
pub struct LakeMediaResolver {
    stores: Arc<dyn ObjectStoreRegistry>,
    paimon: Vec<(Url, FileIO)>,
}

impl Default for LakeMediaResolver {
    fn default() -> Self {
        Self::new(Arc::new(DefaultObjectStoreRegistry::new()))
    }
}

impl LakeMediaResolver {
    /// Use a registry with the same store bindings as the references being resolved.
    /// A Lance store can be registered through its underlying `ObjectStore`.
    #[doc(hidden)]
    pub fn new(stores: Arc<dyn ObjectStoreRegistry>) -> Self {
        Self {
            stores,
            paimon: Vec::new(),
        }
    }

    /// Bind a Paimon FileIO to a URI subtree, including its authority.
    /// The most specific matching binding takes precedence over the registry.
    #[must_use]
    #[doc(hidden)]
    pub fn with_paimon_file_io(mut self, root: Url, file_io: FileIO) -> Self {
        self.paimon.push((root, file_io));
        self
    }

    async fn resolve_paimon(
        file_io: &FileIO,
        media: &MediaRef,
        range: MediaByteRange,
        max_bytes: Option<u64>,
    ) -> Result<Bytes> {
        // FileIO exposes size but no opaque ETag.
        if media.etag.is_some() {
            return exec_err!("Paimon FileIO cannot verify the media ETag");
        }
        let input = file_io.new_input(&media.uri).map_err(external)?;
        let range = match range {
            MediaByteRange::Bounded(range) => range,
            other => eof_range(other, input.metadata().await.map_err(external)?.size)?,
        };
        let len = checked_len(&range, max_bytes)?;
        if len == 0 {
            return Ok(Bytes::new());
        }
        let reader = input.reader().await.map_err(external)?;
        let bytes = reader.read(range).await.map_err(external)?;
        if bytes.len() != len {
            return exec_err!(
                "Paimon media read returned {} bytes, expected {len}",
                bytes.len()
            );
        }
        Ok(bytes)
    }
}

#[async_trait]
impl MediaResolver for LakeMediaResolver {
    async fn resolve(
        &self,
        media: &MediaRef,
        max_bytes: Option<u64>,
    ) -> Result<ResolvedMedia> {
        let range = media.byte_range()?;
        let uri = media_url(&media.uri)?;
        let resolved = |bytes, etag| ResolvedMedia {
            bytes,
            mime: media.mime.clone(),
            etag,
        };
        let empty = match &range {
            MediaByteRange::Bounded(bounded) => checked_len(bounded, max_bytes)? == 0,
            _ => false,
        };
        if empty && media.etag.is_none() {
            return Ok(resolved(Bytes::new(), None));
        }
        if let Some((_, io)) = self
            .paimon
            .iter()
            .filter(|(root, _)| under_root(&uri, root))
            .max_by_key(|(root, _)| root.path().len())
        {
            let bytes = Self::resolve_paimon(io, media, range, max_bytes).await?;
            return Ok(resolved(bytes, None));
        }

        let store = self.stores.get_store(&uri)?;
        let path = Path::from_url_path(uri.path()).map_err(external)?;
        let mut if_match = media.etag.clone();
        let range = match range {
            MediaByteRange::Bounded(range) if !empty => range,
            other => {
                let meta = store.head(&path).await?;
                check_etag(media.etag.as_deref(), meta.e_tag.as_deref())?;
                if_match = meta.e_tag;
                eof_range(other, meta.size)?
            }
        };
        let len = checked_len(&range, max_bytes)?;
        if len == 0 {
            return Ok(resolved(Bytes::new(), if_match));
        }
        let response = store
            .get_opts(
                &path,
                GetOptions {
                    if_match: if_match.clone(),
                    range: Some(GetRange::Bounded(range.clone())),
                    ..Default::default()
                },
            )
            .await
            .map_err(|err| match err {
                object_store::Error::Precondition { .. } => {
                    exec_datafusion_err!("Media ETag precondition failed: {err}")
                }
                other => external(other),
            })?;
        let etag = response.meta.e_tag.clone();
        check_etag(if_match.as_deref(), etag.as_deref())?;
        if response.range != range {
            return exec_err!(
                "Media store returned range {:?}, expected {range:?}",
                response.range
            );
        }
        let bytes = response
            .into_stream()
            .map_err(DataFusionError::from)
            .try_fold(BytesMut::new(), |mut bytes, chunk| async move {
                if chunk.len() > len - bytes.len() {
                    return exec_err!("Media store exceeded the requested {len} bytes");
                }
                bytes.extend_from_slice(&chunk);
                Ok(bytes)
            })
            .await?;
        if bytes.len() != len {
            return exec_err!(
                "Media store returned {} bytes, expected {len}",
                bytes.len()
            );
        }
        Ok(resolved(bytes.freeze(), etag))
    }
}

fn under_root(uri: &Url, root: &Url) -> bool {
    uri[..url::Position::BeforePath] == root[..url::Position::BeforePath]
        && uri
            .path()
            .strip_prefix(root.path().trim_end_matches('/'))
            .is_some_and(|tail| tail.is_empty() || tail.starts_with('/'))
}

fn checked_len(range: &Range<u64>, max_bytes: Option<u64>) -> Result<usize> {
    let len = range.end - range.start;
    if max_bytes.is_some_and(|limit| len > limit) {
        return exec_err!("Media range of {len} bytes exceeds max_bytes {max_bytes:?}");
    }
    usize::try_from(len).map_err(|_| {
        exec_datafusion_err!("Media range of {len} bytes does not fit in memory")
    })
}

fn eof_range(range: MediaByteRange, size: u64) -> Result<Range<u64>> {
    let start = match range {
        MediaByteRange::Full => 0,
        MediaByteRange::FromOffset(start) => start,
        MediaByteRange::Bounded(range) => return Ok(range),
    };
    if start > size {
        return exec_err!("Media offset {start} is beyond object length {size}");
    }
    Ok(start..size)
}

fn check_etag(expected: Option<&str>, actual: Option<&str>) -> Result<()> {
    if expected.is_some() && expected != actual {
        return exec_err!("Media ETag mismatch: expected {expected:?}, got {actual:?}");
    }
    Ok(())
}

pub(crate) fn media_url(uri: &str) -> Result<Url> {
    if std::path::Path::new(uri).is_absolute() {
        return Url::from_file_path(uri)
            .map_err(|()| plan_datafusion_err!("Invalid media file path {uri:?}"));
    }
    let url = Url::parse(uri)
        .map_err(|e| plan_datafusion_err!("Invalid media URI {uri:?}: {e}"))?;
    if url.cannot_be_a_base()
        || !url.username().is_empty()
        || url.password().is_some()
        || url.query().is_some()
        || url.fragment().is_some()
    {
        return Err(plan_datafusion_err!("Invalid media object URI {uri:?}"));
    }
    Ok(url)
}

pub(crate) fn external(
    error: impl std::error::Error + Send + Sync + 'static,
) -> DataFusionError {
    DataFusionError::External(Box::new(error))
}

pub(crate) fn media_array(
    data_type: &DataType,
    values: impl IntoIterator<Item = Result<Option<MediaRef>>>,
) -> Result<ArrayRef> {
    let DataType::Struct(fields) = data_type else {
        return exec_err!("Media projection requires a struct field");
    };
    let mut uri = StringBuilder::new();
    let mut offset = UInt64Builder::new();
    let mut len = UInt64Builder::new();
    let mut etag = StringBuilder::new();
    let mut mime = StringBuilder::new();
    let mut validity = Vec::new();
    for value in values {
        let value = value?;
        validity.push(value.is_some());
        if let Some(value) = value {
            value.validate()?;
            uri.append_value(value.uri);
            offset.append_option(value.byte_offset);
            len.append_option(value.byte_len);
            etag.append_option(value.etag);
            mime.append_option(value.mime);
        } else {
            uri.append_null();
            offset.append_null();
            len.append_null();
            etag.append_null();
            mime.append_null();
        }
    }
    let array = StructArray::try_new(
        fields.clone(),
        vec![
            Arc::new(uri.finish()),
            Arc::new(offset.finish()),
            Arc::new(len.finish()),
            Arc::new(etag.finish()),
            Arc::new(mime.finish()),
        ],
        Some(NullBuffer::from(validity)),
    )?;
    Ok(Arc::new(array))
}
