// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: See the NOTICE file distributed with this work for additional information regarding copyright ownership
// SPDX-FileContributor: Licensed to the Apache Software Foundation (ASF) under one or more contributor license agreements

use std::pin::Pin;
use std::sync::Arc;
use std::task::{Context, Poll};
use std::time::Duration;

use arrow::array::{Array, ArrayRef, LargeStringArray, StringArray, StringViewArray};
use arrow::datatypes::{DataType, SchemaRef};
use arrow::record_batch::RecordBatch;
use futures::stream::{FuturesUnordered, Stream, StreamExt, once};
use tokio::sync::{Semaphore, mpsc};

use crate::common::{Result, exec_err, internal_err, plan_err};
use crate::execution::SendableRecordBatchStream;
use crate::physical_expr::PhysicalExpr;
use crate::physical_plan::RecordBatchStream;
use crate::physical_plan::metrics::{Count, ExecutionPlanMetricsSet, MetricBuilder};
use crate::physical_plan::stream::{RecordBatchReceiverStream, RecordBatchStreamAdapter};

use super::backend::RegisteredProvider;
use super::{
    CancellationToken, LlmBackendError, LlmOptions, LlmRequest, LlmResponse, OnError,
};

// Bound time spent holding concurrency slots at the provider's request.
const MAX_RETRY_AFTER: Duration = Duration::from_secs(60);

#[derive(Debug)]
pub(super) struct Dispatch {
    pub(super) model: String,
    options: Arc<LlmOptions>,
    provider: RegisteredProvider,
    slots: Arc<Semaphore>,
    inflight_limit: usize,
    batch_size: usize,
}

impl Dispatch {
    pub(super) fn try_new(
        model: String,
        options: LlmOptions,
        provider: RegisteredProvider,
    ) -> Result<Self> {
        let batch_size = options
            .batch_size
            .unwrap_or_else(|| provider.backend.default_batch_size());
        if batch_size == 0 {
            return plan_err!("llm provider default batch_size must be at least 1");
        }
        if let Some(max) = provider.backend.max_batch_size()
            && batch_size > max
        {
            return plan_err!(
                "llm provider accepts at most {max} prompt(s) per request, but batch_size is {batch_size}"
            );
        }
        let inflight_limit = options.inflight_limit.unwrap_or(64);
        Ok(Self {
            model,
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
        metrics: &LlmMetrics,
        cancel: &CancellationToken,
    ) -> std::result::Result<LlmResponse, LlmBackendError> {
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
                    self.provider
                        .backend
                        .generate(request.clone(), attempt_cancel),
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
                Ok(response) if response.outputs.len() == request.prompts.len() => {
                    return Ok(response);
                }
                Ok(response) => {
                    return Err(LlmBackendError::Permanent {
                        status: None,
                        message: format!(
                            "backend returned {} outputs for {} prompts",
                            response.outputs.len(),
                            request.prompts.len()
                        ),
                    });
                }
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
        dispatch: Arc<Dispatch>,
        metrics: LlmMetrics,
    ) -> Self {
        let cancel = CancellationToken::new();
        let driver = Driver {
            schema: Arc::clone(&schema),
            prompt,
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
                next = self.next_slice(&mut input, &mut current),
                    if !eof && jobs.len() < self.dispatch.inflight_limit =>
                {
                    let Some((rows, prompts)) = next? else {
                        eof = true;
                        continue;
                    };
                    if !prompts.is_empty() {
                        jobs.push(self.enqueue(rows, prompts));
                        continue;
                    }
                    let nulls = StringArray::new_null(rows.num_rows());
                    self.append_output(&rows, nulls)?
                }
            };
            // A completed job has released both slots before awaiting output capacity.
            if tx.send(Ok(output)).await.is_err() {
                break;
            }
        }
        Ok(())
    }

    async fn next_slice(
        &self,
        input: &mut SendableRecordBatchStream,
        current: &mut Option<PromptBatch>,
    ) -> Result<Option<(RecordBatch, Vec<String>)>> {
        while current.is_none() {
            let Some(rows) = input.next().await.transpose()? else {
                return Ok(None);
            };
            if rows.num_rows() != 0 {
                let values = self.prompt.evaluate(&rows)?.into_array(rows.num_rows())?;
                *current = Some(PromptBatch {
                    rows,
                    values,
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

    fn enqueue(
        &self,
        rows: RecordBatch,
        prompts: Vec<String>,
    ) -> impl Future<Output = Result<RecordBatch>> + '_ {
        let mut tracking = TrackCancellation {
            counter: &self.metrics.cancelled_requests,
            pending: true,
        };
        async move {
            let response = self
                .dispatch
                .generate(prompts, &self.metrics, &self.cancel)
                .await;
            if matches!(response, Err(LlmBackendError::Cancelled)) {
                return exec_err!("llm request cancelled");
            }
            tracking.pending = false;
            drop(tracking);
            let output = match response {
                Ok(response) => {
                    if let Some(usage) = response.usage {
                        self.metrics.input_tokens.add(
                            usize::try_from(usage.input_tokens).unwrap_or(usize::MAX),
                        );
                        self.metrics.output_tokens.add(
                            usize::try_from(usage.output_tokens).unwrap_or(usize::MAX),
                        );
                    }
                    StringArray::from(response.outputs)
                }
                Err(error) => {
                    self.metrics.failed_rows.add(rows.num_rows());
                    match self.dispatch.options.on_error {
                        OnError::Raise => {
                            return exec_err!("llm backend failed: {error}");
                        }
                        OnError::Log => log::warn!(
                            "llm backend failed for {} rows: {error}",
                            rows.num_rows()
                        ),
                        OnError::Ignore => {}
                    }
                    StringArray::new_null(rows.num_rows())
                }
            };
            self.append_output(&rows, output)
        }
    }

    fn append_output(
        &self,
        rows: &RecordBatch,
        output: StringArray,
    ) -> Result<RecordBatch> {
        let mut columns = rows.columns().to_vec();
        columns.push(Arc::new(output));
        Ok(RecordBatch::try_new(Arc::clone(&self.schema), columns)?)
    }
}

struct PromptBatch {
    rows: RecordBatch,
    values: ArrayRef,
    offset: usize,
}

impl PromptBatch {
    fn next_slice(&mut self, batch_size: usize) -> Result<(RecordBatch, Vec<String>)> {
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
        Ok((self.rows.slice(start, end - start), prompts))
    }
}
