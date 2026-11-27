// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: See the NOTICE file distributed with this work for additional information regarding copyright ownership
// SPDX-FileContributor: Licensed to the Apache Software Foundation (ASF) under one or more contributor license agreements

use std::fmt::{self, Display};
use std::ops::Range;
use std::sync::{Arc, Mutex};

use async_trait::async_trait;
use futures::stream::BoxStream;
use lance_io::object_store::WrappingObjectStore;
use object_store::list::PaginatedListStore;
use object_store::path::Path;
use object_store::{
    CopyOptions, GetOptions, GetResult, ListResult, MultipartUpload, ObjectMeta,
    ObjectStore, PutMultipartOptions, PutOptions, PutPayload, PutResult, Result,
};

#[derive(Debug, Clone)]
pub(super) struct Reads(Arc<Mutex<Vec<Read>>>);

#[derive(Debug)]
pub(super) struct Read {
    pub(super) path: Path,
    // None indicates a metadata request.
    pub(super) range: Option<Range<u64>>,
}

impl Reads {
    pub(super) fn new() -> Self {
        Self(Arc::default())
    }

    pub(super) fn take(&self) -> Vec<Read> {
        std::mem::take(&mut *self.0.lock().unwrap())
    }
}

impl WrappingObjectStore for Reads {
    fn wrap_paginated(
        &self,
        _: &str,
        _: Arc<dyn PaginatedListStore>,
    ) -> Option<Arc<dyn PaginatedListStore>> {
        None
    }

    fn wrap(
        &self,
        _store_prefix: &str,
        original: Arc<dyn ObjectStore>,
    ) -> Arc<dyn ObjectStore> {
        Arc::new(ObservedStore {
            inner: original,
            reads: self.clone(),
        })
    }
}

#[derive(Debug)]
struct ObservedStore {
    inner: Arc<dyn ObjectStore>,
    reads: Reads,
}

impl Display for ObservedStore {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(formatter, "ObservedStore({})", self.inner)
    }
}

#[async_trait]
impl ObjectStore for ObservedStore {
    async fn get_opts(&self, location: &Path, options: GetOptions) -> Result<GetResult> {
        let head = options.head;
        let result = self.inner.get_opts(location, options).await?;
        self.reads.0.lock().unwrap().push(Read {
            path: location.clone(),
            range: (!head).then(|| result.range.clone()),
        });
        Ok(result)
    }

    async fn put_opts(
        &self,
        path: &Path,
        payload: PutPayload,
        options: PutOptions,
    ) -> Result<PutResult> {
        self.inner.put_opts(path, payload, options).await
    }

    async fn put_multipart_opts(
        &self,
        path: &Path,
        options: PutMultipartOptions,
    ) -> Result<Box<dyn MultipartUpload>> {
        self.inner.put_multipart_opts(path, options).await
    }

    fn delete_stream(
        &self,
        locations: BoxStream<'static, Result<Path>>,
    ) -> BoxStream<'static, Result<Path>> {
        self.inner.delete_stream(locations)
    }

    fn list(&self, prefix: Option<&Path>) -> BoxStream<'static, Result<ObjectMeta>> {
        self.inner.list(prefix)
    }

    async fn list_with_delimiter(&self, prefix: Option<&Path>) -> Result<ListResult> {
        self.inner.list_with_delimiter(prefix).await
    }

    async fn copy_opts(
        &self,
        from: &Path,
        to: &Path,
        options: CopyOptions,
    ) -> Result<()> {
        self.inner.copy_opts(from, to, options).await
    }
}
