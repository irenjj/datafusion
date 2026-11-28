// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: See the NOTICE file distributed with this work for additional information regarding copyright ownership
// SPDX-FileContributor: Licensed to the Apache Software Foundation (ASF) under one or more contributor license agreements

use std::pin::Pin;
use std::sync::Arc;
use std::task::{Context, Poll};
use std::time::Duration;

use arrow::array::{
    Array, ArrayRef, AsArray, BooleanBuilder, FixedSizeListArray, Float32Array,
    LargeStringArray, StringArray, StringBuilder, StringViewArray, StructArray,
    UInt64Array, new_null_array,
};
use arrow::compute::take;
use arrow::datatypes::{DataType, SchemaRef, UInt64Type};
use arrow::record_batch::RecordBatch;
use futures::stream::{FuturesUnordered, Stream, StreamExt, once};
use tokio::sync::{Semaphore, mpsc};

use datafusion_common::{Result, exec_err, internal_err, plan_err};
use datafusion_execution::SendableRecordBatchStream;
use datafusion_physical_expr::PhysicalExpr;
use datafusion_physical_plan::RecordBatchStream;
use datafusion_physical_plan::metrics::{Count, ExecutionPlanMetricsSet, MetricBuilder};
use datafusion_physical_plan::stream::{
    RecordBatchReceiverStream, RecordBatchStreamAdapter,
};

use super::backend::RegisteredProvider;
use super::{
    CancellationToken, LlmBackendError, LlmEmbeddingResponse, LlmOptions, LlmRequest,
    LlmResponse, LlmUsage, MediaPlan, OnError, TaskKind,
};
use crate::media::is_transient_media_error;
use crate::sanitize::sanitized_detail;
use crate::{MediaRef, ResolvedMedia};

// Bound time spent holding concurrency slots at the provider's request.
const MAX_RETRY_AFTER: Duration = Duration::from_secs(60);

#[derive(Debug)]
pub(super) struct Dispatch {
    pub(super) model: String,
    pub(super) options: Arc<LlmOptions>,
    pub(super) kind: TaskKind,
    dimension: Option<i32>,
    provider: RegisteredProvider,
    slots: Arc<Semaphore>,
    inflight_limit: usize,
    batch_size: usize,
}

impl Dispatch {
    pub(super) fn try_new(
        model: String,
        options: LlmOptions,
        kind: TaskKind,
        provider: RegisteredProvider,
    ) -> Result<Self> {
        let batch_size = options
            .batch_size
            .unwrap_or_else(|| provider.backend.default_batch_size_for(kind));
        if batch_size == 0 {
            return plan_err!("llm provider default batch_size must be at least 1");
        }
        if let Some(max) = provider.backend.max_batch_size_for(kind)
            && batch_size > max
        {
            return plan_err!(
                "llm provider accepts at most {max} prompt(s) per request, but batch_size is {batch_size}"
            );
        }
        let inflight_limit = options.inflight_limit.unwrap_or(64);
        Ok(Self {
            model,
            kind,
            dimension: (kind == TaskKind::Embed)
                .then(|| options.embedding_dimension())
                .transpose()?,
            options: Arc::new(options),
            provider,
            slots: Arc::new(Semaphore::new(inflight_limit)),
            inflight_limit,
            batch_size,
        })
    }

    async fn generate(
        &self,
        prompts: Vec<String>,
        media: Vec<Option<ResolvedMedia>>,
        metrics: &LlmMetrics,
        cancel: &CancellationToken,
    ) -> std::result::Result<Outputs, LlmBackendError> {
        // Both limits count requests. One request reserves one slot from each
        // semaphore, in the same order, across all of its attempts and backoff.
        let _slot = self
            .slots
            .acquire()
            .await
            .expect("operator semaphore stays open");
        let _provider_slot = match &self.provider.limit {
            Some(limit) => Some(
                limit
                    .acquire()
                    .await
                    .expect("provider semaphore stays open"),
            ),
            None => None,
        };
        let request = LlmRequest {
            model: self.model.clone(),
            prompts,
            media,
            options: Arc::clone(&self.options),
        };
        let timeout = Duration::from_millis(self.options.timeout_ms.unwrap_or(30_000));
        let max_retries = self.options.max_retries.unwrap_or(3);
        let mut delay = Duration::from_millis(100);
        for attempt in 0..=max_retries {
            if cancel.is_cancelled() {
                return Err(LlmBackendError::Cancelled);
            }
            if attempt == 0 {
                metrics.submitted.add(request.prompts.len());
            } else {
                metrics.retries.add(1);
            }
            metrics.attempts.add(1);
            let attempt_cancel = cancel.child_token();
            let response = {
                let _cancel_on_drop = attempt_cancel.clone().drop_guard();
                tokio::time::timeout(
                    timeout,
                    self.attempt(request.clone(), attempt_cancel),
                )
                .await
                .unwrap_or_else(|_| {
                    Err(LlmBackendError::Transient {
                        status: None,
                        message: "inference attempt timed out".into(),
                    })
                })
            };
            let error = match response {
                Ok(response) => return Ok(response),
                Err(error) => error,
            };
            let retry_after = match &error {
                LlmBackendError::RateLimited { retry_after, .. } => {
                    retry_after.unwrap_or(delay).min(MAX_RETRY_AFTER)
                }
                LlmBackendError::Transient { .. } => delay,
                _ => return Err(error),
            };
            if attempt == max_retries {
                return Err(error);
            }
            tokio::time::sleep(retry_after).await;
            delay = (delay * 2).min(Duration::from_secs(5));
        }
        unreachable!("the attempt loop returns on success or exhaustion")
    }

    async fn attempt(
        &self,
        request: LlmRequest,
        cancel: CancellationToken,
    ) -> std::result::Result<Outputs, LlmBackendError> {
        let count = request.prompts.len();
        let backend = &self.provider.backend;
        if let Some(dimension) = self.dimension {
            let response = backend.embed(request, cancel).await?;
            response.validate(count, dimension as usize)?;
            Ok(Outputs::Embeddings(response))
        } else {
            let response = backend.generate(request, cancel).await?;
            if response.outputs.len() != count {
                return Err(LlmBackendError::InvalidResponse {
                    status: None,
                    message: format!(
                        "backend returned {} outputs for {count} prompts",
                        response.outputs.len()
                    ),
                });
            }
            Ok(Outputs::Completions(response))
        }
    }
}

enum Outputs {
    Completions(LlmResponse),
    Embeddings(LlmEmbeddingResponse),
}

impl Outputs {
    fn into_array(self, value_type: &DataType) -> Result<(ArrayRef, Option<LlmUsage>)> {
        match self {
            Self::Completions(response) => Ok((
                Arc::new(StringArray::from(response.outputs)),
                response.usage,
            )),
            Self::Embeddings(response) => {
                let DataType::FixedSizeList(item, dimension) = value_type else {
                    return internal_err!(
                        "LlmExec expected a fixed-size list for embeddings"
                    );
                };
                let values = Float32Array::from_iter_values(
                    response.outputs.into_iter().flatten(),
                );
                let list = FixedSizeListArray::try_new(
                    Arc::clone(item),
                    *dimension,
                    Arc::new(values),
                    None,
                )?;
                Ok((Arc::new(list), response.usage))
            }
        }
    }
}

pub(super) struct LlmMetrics {
    submitted: Count,
    attempts: Count,
    retries: Count,
    failed_rows: Count,
    cancelled_requests: Count,
    input_tokens: Count,
    output_tokens: Count,
}

impl LlmMetrics {
    pub(super) fn new(metrics: &ExecutionPlanMetricsSet, partition: usize) -> Self {
        let counter = |name| MetricBuilder::new(metrics).counter(name, partition);
        Self {
            submitted: counter("submitted_prompts"),
            attempts: counter("backend_attempts"),
            retries: counter("retries"),
            failed_rows: counter("failed_rows"),
            cancelled_requests: counter("cancelled_requests"),
            input_tokens: counter("input_tokens"),
            output_tokens: counter("output_tokens"),
        }
    }
}

struct TrackCancellation<'a> {
    counter: &'a Count,
    pending: bool,
}

impl Drop for TrackCancellation<'_> {
    fn drop(&mut self) {
        if self.pending {
            self.counter.add(1);
        }
    }
}

pub(super) struct LlmStream {
    inner: SendableRecordBatchStream,
    cancel: CancellationToken,
}

impl LlmStream {
    pub(super) fn new(
        schema: SchemaRef,
        input: SendableRecordBatchStream,
        prompt: Arc<dyn PhysicalExpr>,
        media: Option<MediaPlan>,
        dispatch: Arc<Dispatch>,
        metrics: LlmMetrics,
    ) -> Self {
        let cancel = CancellationToken::new();
        let driver = Driver {
            schema: Arc::clone(&schema),
            prompt,
            media,
            dispatch,
            metrics,
            cancel: cancel.clone(),
        };
        let inner = once(async move {
            // A partition starts only when it is first polled.
            let mut builder =
                RecordBatchReceiverStream::builder(Arc::clone(&driver.schema), 1);
            let tx = builder.tx();
            builder.spawn(async move {
                if let Err(error) = driver.run(input, &tx).await {
                    let _ = tx.send(Err(error)).await;
                }
                Ok(())
            });
            builder.build()
        })
        .flatten();
        Self {
            inner: Box::pin(RecordBatchStreamAdapter::new(schema, inner)),
            cancel,
        }
    }
}

impl Stream for LlmStream {
    type Item = Result<RecordBatch>;

    fn poll_next(
        mut self: Pin<&mut Self>,
        cx: &mut Context<'_>,
    ) -> Poll<Option<Self::Item>> {
        self.inner.as_mut().poll_next(cx)
    }
}

impl RecordBatchStream for LlmStream {
    fn schema(&self) -> SchemaRef {
        self.inner.schema()
    }
}

impl Drop for LlmStream {
    fn drop(&mut self) {
        self.cancel.cancel();
        // Dropping the receiver stream aborts the driver and drops its futures.
    }
}

struct Driver {
    schema: SchemaRef,
    prompt: Arc<dyn PhysicalExpr>,
    media: Option<MediaPlan>,
    dispatch: Arc<Dispatch>,
    metrics: LlmMetrics,
    cancel: CancellationToken,
}

impl Driver {
    async fn run(
        &self,
        mut input: SendableRecordBatchStream,
        tx: &mpsc::Sender<Result<RecordBatch>>,
    ) -> Result<()> {
        let mut jobs = FuturesUnordered::new();
        let mut current = None;
        let mut eof = false;
        while !eof || !jobs.is_empty() {
            let output = tokio::select! {
                biased;
                _ = self.cancel.cancelled() => break,
                Some(output) = jobs.next(), if !jobs.is_empty() => output?,
                next = self.read_slice(&mut input, &mut current),
                    if !eof && jobs.len() < self.dispatch.inflight_limit =>
                {
                    let Some(slice) = next? else {
                        eof = true;
                        continue;
                    };
                    if !slice.prompts.is_empty() {
                        jobs.push(self.next_slice(slice));
                        continue;
                    }
                    let nulls = new_null_array(self.value_type(), slice.rows.num_rows());
                    self.append_output(&slice.rows, nulls, None, &[])?
                }
            };
            // A completed job has released both slots before awaiting output capacity.
            if tx.send(Ok(output)).await.is_err() {
                break;
            }
        }
        Ok(())
    }

    async fn read_slice(
        &self,
        input: &mut SendableRecordBatchStream,
        current: &mut Option<PromptBatch>,
    ) -> Result<Option<PromptSlice>> {
        while current.is_none() {
            let Some(rows) = input.next().await.transpose()? else {
                return Ok(None);
            };
            if rows.num_rows() != 0 {
                let values = self.prompt.evaluate(&rows)?.into_array(rows.num_rows())?;
                let media = if values.null_count() == rows.num_rows() {
                    None
                } else {
                    self.media
                        .as_ref()
                        .map(|media| {
                            media.expr.evaluate(&rows)?.into_array(rows.num_rows())
                        })
                        .transpose()?
                };
                *current = Some(PromptBatch {
                    rows,
                    values,
                    media,
                    offset: 0,
                });
            }
        }
        let batch = current.as_mut().unwrap();
        let slice = batch.next_slice(self.dispatch.batch_size)?;
        if batch.offset == batch.rows.num_rows() {
            *current = None;
        }
        Ok(Some(slice))
    }

    fn next_slice(
        &self,
        slice: PromptSlice,
    ) -> impl Future<Output = Result<RecordBatch>> + '_ {
        let mut tracking = TrackCancellation {
            counter: &self.metrics.cancelled_requests,
            pending: true,
        };
        async move {
            let PromptSlice {
                rows,
                prompts,
                media,
            } = slice;
            let mut media_errors = Vec::new();
            let mut indices = Vec::new();
            // Media reads belong to the queued job so polling another completed
            // slice cannot interrupt a read after advancing the input cursor.
            let (prompts, resolved_media) = if let Some(media) = media {
                let mut ready = Vec::with_capacity(prompts.len());
                let mut resolved = Vec::with_capacity(prompts.len());
                for (row, prompt) in prompts.into_iter().enumerate() {
                    match self.resolve_media(media.as_ref(), row).await {
                        Ok(value) => {
                            indices.push(Some(ready.len() as u64));
                            ready.push(prompt);
                            resolved.push(Some(value));
                        }
                        Err(error) => {
                            if let Err(error) = self.handle_failure(&error, 1) {
                                tracking.pending = false;
                                drop(tracking);
                                return Err(error);
                            }
                            indices.push(None);
                            media_errors.push((row, error));
                        }
                    }
                }
                (ready, resolved)
            } else {
                let resolved = vec![None; prompts.len()];
                (prompts, resolved)
            };
            let count = prompts.len();
            if count == 0 {
                tracking.pending = false;
                drop(tracking);
                let nulls = new_null_array(self.value_type(), rows.num_rows());
                return self.append_output(&rows, nulls, None, &media_errors);
            }
            let response = self
                .dispatch
                .generate(prompts, resolved_media, &self.metrics, &self.cancel)
                .await;
            tracking.pending = matches!(response, Err(LlmBackendError::Cancelled));
            drop(tracking);
            match response {
                Ok(response) => {
                    let (mut output, usage) = response.into_array(self.value_type())?;
                    if let Some(usage) = usage {
                        self.metrics.input_tokens.add(
                            usize::try_from(usage.input_tokens).unwrap_or(usize::MAX),
                        );
                        self.metrics.output_tokens.add(
                            usize::try_from(usage.output_tokens).unwrap_or(usize::MAX),
                        );
                    }
                    if !media_errors.is_empty() {
                        output =
                            take(output.as_ref(), &UInt64Array::from(indices), None)?;
                    }
                    self.append_output(&rows, output, None, &media_errors)
                }
                Err(error) => {
                    if matches!(error, LlmBackendError::Cancelled)
                        && self.dispatch.options.on_error != OnError::Envelope
                    {
                        return exec_err!("llm request cancelled");
                    }
                    self.handle_failure(&error, count)?;
                    let nulls = new_null_array(self.value_type(), rows.num_rows());
                    self.append_output(&rows, nulls, Some(&error), &media_errors)
                }
            }
        }
    }

    async fn resolve_media(
        &self,
        values: &dyn Array,
        row: usize,
    ) -> std::result::Result<ResolvedMedia, LlmBackendError> {
        let result = async {
            let media = media_ref_at(values, row)?;
            media.validate()?;
            let resolver = self
                .media
                .as_ref()
                .and_then(|media| media.resolver.as_ref())
                .expect("vision media plan has a resolver");
            let limit = self.dispatch.options.media_byte_limit();
            let mut resolved = resolver.resolve(&media, Some(limit)).await?;
            if resolved.bytes.len() as u64 > limit {
                return exec_err!("resolved media exceeds max_bytes {limit}");
            }
            resolved.mime = resolved.mime.or(media.mime);
            Ok(resolved)
        }
        .await;
        result.map_err(|error| {
            let detail = sanitized_detail(&error.strip_backtrace())
                .unwrap_or_else(|| "could not resolve the media reference".into());
            let message = format!("media: {detail}");
            if is_transient_media_error(&error) {
                LlmBackendError::Transient {
                    status: None,
                    message,
                }
            } else {
                LlmBackendError::Permanent {
                    status: None,
                    message,
                }
            }
        })
    }

    fn handle_failure(&self, error: &LlmBackendError, rows: usize) -> Result<()> {
        self.metrics.failed_rows.add(rows);
        match self.dispatch.options.on_error {
            OnError::Raise => exec_err!("llm backend failed: {error}"),
            OnError::Log => {
                log::warn!("llm backend failed for {rows} rows: {error}");
                Ok(())
            }
            OnError::Ignore | OnError::Envelope => Ok(()),
        }
    }

    fn value_type(&self) -> &DataType {
        let output = self.schema.fields().last().unwrap().data_type();
        if let DataType::Struct(fields) = output {
            return fields[0].data_type();
        }
        output
    }

    fn append_output(
        &self,
        rows: &RecordBatch,
        mut output: ArrayRef,
        batch_error: Option<&LlmBackendError>,
        media_errors: &[(usize, LlmBackendError)],
    ) -> Result<RecordBatch> {
        if self.dispatch.options.on_error == OnError::Envelope {
            let DataType::Struct(fields) =
                self.schema.fields().last().unwrap().data_type()
            else {
                return internal_err!("LlmExec envelope output must be a struct");
            };
            let mut code = StringBuilder::new();
            let mut message = StringBuilder::new();
            let mut retryable = BooleanBuilder::new();
            let mut media_errors = media_errors.iter().peekable();
            for row in 0..rows.num_rows() {
                let error = if media_errors.peek().is_some_and(|(index, _)| *index == row)
                {
                    media_errors.next().map(|(_, error)| error)
                } else {
                    batch_error
                };
                if let Some(error) = error {
                    let (kind, retry) = error.classification();
                    let wire_code = serde_json::to_value(kind)
                        .expect("AI error codes serialize as strings");
                    code.append_value(wire_code.as_str().unwrap());
                    message.append_value(error.to_string());
                    retryable.append_value(retry);
                } else {
                    code.append_null();
                    message.append_null();
                    retryable.append_null();
                }
            }
            output = Arc::new(StructArray::try_new(
                fields.clone(),
                vec![
                    output,
                    Arc::new(code.finish()),
                    Arc::new(message.finish()),
                    Arc::new(retryable.finish()),
                ],
                None,
            )?);
        }
        let mut columns = rows.columns().to_vec();
        columns.push(output);
        Ok(RecordBatch::try_new(Arc::clone(&self.schema), columns)?)
    }
}

fn media_ref_at(values: &dyn Array, row: usize) -> Result<MediaRef> {
    if values.is_null(row) {
        return exec_err!("expected a non-NULL media reference");
    }
    if let Some(value) = values.as_struct_opt() {
        let uri = value.column(0).as_string::<i32>();
        let offset = value.column(1).as_primitive::<UInt64Type>();
        let len = value.column(2).as_primitive::<UInt64Type>();
        let etag = value.column(3).as_string::<i32>();
        let mime = value.column(4).as_string::<i32>();
        if uri.is_null(row) {
            return exec_err!("expected a non-NULL media URI");
        }
        return Ok(MediaRef {
            uri: uri.value(row).to_owned(),
            byte_offset: offset.is_valid(row).then(|| offset.value(row)),
            byte_len: len.is_valid(row).then(|| len.value(row)),
            etag: etag.is_valid(row).then(|| etag.value(row).to_owned()),
            mime: mime.is_valid(row).then(|| mime.value(row).to_owned()),
        });
    }
    let uri = match values.data_type() {
        DataType::Utf8 => values.as_string::<i32>().value(row),
        DataType::LargeUtf8 => values.as_string::<i64>().value(row),
        DataType::Utf8View => values.as_string_view().value(row),
        _ => return exec_err!("unsupported media reference type"),
    };
    Ok(MediaRef {
        uri: uri.to_owned(),
        byte_offset: None,
        byte_len: None,
        etag: None,
        mime: None,
    })
}

struct PromptBatch {
    rows: RecordBatch,
    values: ArrayRef,
    media: Option<ArrayRef>,
    offset: usize,
}

struct PromptSlice {
    rows: RecordBatch,
    prompts: Vec<String>,
    media: Option<ArrayRef>,
}

impl PromptBatch {
    fn next_slice(&mut self, batch_size: usize) -> Result<PromptSlice> {
        let start = self.offset;
        let end = start + (self.rows.num_rows() - start).min(batch_size);
        let is_null = self.values.is_null(start);
        let end = (start + 1..end)
            .find(|&row| self.values.is_null(row) != is_null)
            .unwrap_or(end);
        let indices = start..end;
        let prompts = if is_null {
            Vec::new()
        } else {
            match self.values.data_type() {
                DataType::Utf8 => {
                    let array =
                        self.values.as_any().downcast_ref::<StringArray>().unwrap();
                    indices.map(|row| array.value(row).to_owned()).collect()
                }
                DataType::LargeUtf8 => {
                    let array = self
                        .values
                        .as_any()
                        .downcast_ref::<LargeStringArray>()
                        .unwrap();
                    indices.map(|row| array.value(row).to_owned()).collect()
                }
                DataType::Utf8View => {
                    let array = self
                        .values
                        .as_any()
                        .downcast_ref::<StringViewArray>()
                        .unwrap();
                    indices.map(|row| array.value(row).to_owned()).collect()
                }
                kind => {
                    return internal_err!("LlmExec expected a string prompt, got {kind}");
                }
            }
        };
        self.offset = end;
        let len = end - start;
        let media = if is_null {
            None
        } else {
            self.media.as_ref().map(|media| media.slice(start, len))
        };
        Ok(PromptSlice {
            rows: self.rows.slice(start, len),
            prompts,
            media,
        })
    }
}
