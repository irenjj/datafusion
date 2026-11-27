// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: See the NOTICE file distributed with this work for additional information regarding copyright ownership
// SPDX-FileContributor: Licensed to the Apache Software Foundation (ASF) under one or more contributor license agreements

use std::collections::HashMap;
use std::fmt::{self, Debug, Display};
use std::sync::Arc;
use std::time::Duration;

use async_trait::async_trait;
use parking_lot::RwLock;
use tokio::sync::Semaphore;

use super::options::{LlmOptions, validate_inflight_limit};
use datafusion_common::{Result, plan_datafusion_err, plan_err};

pub use tokio_util::sync::CancellationToken;

/// A request containing a batch of non-NULL prompts for one model.
#[derive(Debug, Clone)]
pub struct LlmRequest {
    /// Provider model identifier.
    pub model: String,
    /// Prompts in input row order.
    pub prompts: Vec<String>,
    /// Validated query and generation options.
    pub options: Arc<LlmOptions>,
}

/// Token usage reported by the provider for a successful request.
#[derive(Debug, Clone, Copy)]
pub struct LlmUsage {
    /// Prompt tokens.
    pub input_tokens: u64,
    /// Generated tokens.
    pub output_tokens: u64,
}

/// A response to one inference request.
#[derive(Debug)]
pub struct LlmResponse {
    /// One output per prompt, in the same order. Missing output becomes SQL NULL.
    pub outputs: Vec<Option<String>>,
    /// Usage, when the provider reports it.
    pub usage: Option<LlmUsage>,
}

/// Failure classification for a single backend attempt. Messages may be shown
/// in query errors or logs and should be safe for those audiences.
#[derive(Debug)]
pub enum LlmBackendError {
    /// The provider has throttled the request.
    RateLimited {
        /// Delay requested by the provider.
        retry_after: Option<Duration>,
        /// Diagnostic message.
        message: String,
    },
    /// A failure that may succeed on retry.
    Transient {
        /// HTTP status, if applicable.
        status: Option<u16>,
        /// Diagnostic message.
        message: String,
    },
    /// A failure that will not be retried.
    Permanent {
        /// HTTP status, if applicable.
        status: Option<u16>,
        /// Diagnostic message.
        message: String,
    },
    /// The attempt was cancelled.
    Cancelled,
}

impl Display for LlmBackendError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::RateLimited { message, .. } => write!(f, "rate limited: {message}"),
            Self::Transient { status, message } | Self::Permanent { status, message } => {
                if let Some(status) = status {
                    write!(f, "HTTP {status}: ")?;
                }
                f.write_str(message)
            }
            Self::Cancelled => f.write_str("request cancelled"),
        }
    }
}

impl std::error::Error for LlmBackendError {}

/// An asynchronous inference provider. The executor may drop the generation
/// future during cancellation or timeout. Backends that spawn independent work
/// should also observe the cancellation token.
#[async_trait]
pub trait LlmBackend: Debug + Send + Sync {
    /// Number of prompts per request when SQL does not specify a batch size.
    fn default_batch_size(&self) -> usize {
        1
    }

    /// Maximum number of prompts the backend accepts in a request, if bounded.
    fn max_batch_size(&self) -> Option<usize> {
        None
    }

    /// Runs one attempt. The executor owns the timeout and retry policy.
    async fn generate(
        &self,
        request: LlmRequest,
        cancel: CancellationToken,
    ) -> std::result::Result<LlmResponse, LlmBackendError>;
}

/// The built-in deterministic backend. It performs no network requests.
#[derive(Debug, Default)]
pub struct MockLlmBackend;

#[async_trait]
impl LlmBackend for MockLlmBackend {
    fn default_batch_size(&self) -> usize {
        1024
    }

    async fn generate(
        &self,
        request: LlmRequest,
        cancel: CancellationToken,
    ) -> std::result::Result<LlmResponse, LlmBackendError> {
        if cancel.is_cancelled() {
            return Err(LlmBackendError::Cancelled);
        }
        let outputs = request
            .prompts
            .iter()
            .map(|prompt| Some(format!("mock[{}]: {prompt}", request.model)))
            .collect();
        Ok(LlmResponse {
            outputs,
            usage: None,
        })
    }
}

#[derive(Clone)]
pub(super) struct RegisteredProvider {
    pub(super) backend: Arc<dyn LlmBackend>,
    pub(super) limit: Option<Arc<Semaphore>>,
}

impl Debug for RegisteredProvider {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("RegisteredProvider").finish_non_exhaustive()
    }
}

/// Session-level registry of named LLM backends.
///
/// Clones share registrations and provider concurrency limits. The reserved
/// name "default" starts as [`MockLlmBackend`] and can be replaced.
///
/// ```
/// use std::sync::Arc;
/// use datafusion_ai::llm::{LlmProviderRegistry, MockLlmBackend};
///
/// let registry = LlmProviderRegistry::default();
/// let shared = registry.clone();
/// registry.register_with_limit("offline", Arc::new(MockLlmBackend), 2)?;
/// assert_eq!(shared.available_permits("offline"), Some(2));
/// # Ok::<(), datafusion_common::DataFusionError>(())
/// ```
#[derive(Debug, Clone)]
pub struct LlmProviderRegistry {
    providers: Arc<RwLock<HashMap<String, RegisteredProvider>>>,
}

impl Default for LlmProviderRegistry {
    fn default() -> Self {
        Self {
            providers: Arc::new(RwLock::new(HashMap::from([(
                "default".into(),
                RegisteredProvider {
                    backend: Arc::new(MockLlmBackend),
                    limit: None,
                },
            )]))),
        }
    }
}

impl LlmProviderRegistry {
    /// Registers or replaces a backend without a provider-wide concurrency cap.
    pub fn register(
        &self,
        name: impl Into<String>,
        backend: Arc<dyn LlmBackend>,
    ) -> Result<()> {
        self.insert(name.into(), backend, None)
    }

    /// Registers or replaces a backend with a cap on concurrent requests shared
    /// across queries and partitions. A request holds its slot through retries.
    pub fn register_with_limit(
        &self,
        name: impl Into<String>,
        backend: Arc<dyn LlmBackend>,
        inflight_limit: usize,
    ) -> Result<()> {
        validate_inflight_limit(inflight_limit)?;
        self.insert(
            name.into(),
            backend,
            Some(Arc::new(Semaphore::new(inflight_limit))),
        )
    }

    /// Returns the number of available request slots for a capped provider.
    /// Returns None for an unknown or uncapped provider.
    pub fn available_permits(&self, name: &str) -> Option<usize> {
        self.providers
            .read()
            .get(name)?
            .limit
            .as_ref()
            .map(|limit| limit.available_permits())
    }

    fn insert(
        &self,
        name: String,
        backend: Arc<dyn LlmBackend>,
        limit: Option<Arc<Semaphore>>,
    ) -> Result<()> {
        if name.trim().is_empty() {
            return plan_err!("llm provider name must be non-empty");
        }
        self.providers
            .write()
            .insert(name, RegisteredProvider { backend, limit });
        Ok(())
    }

    pub(super) fn resolve(&self, name: Option<&str>) -> Result<RegisteredProvider> {
        let name = name.unwrap_or("default");
        self.providers
            .read()
            .get(name)
            .cloned()
            .ok_or_else(|| plan_datafusion_err!("Unknown llm provider '{name}'"))
    }
}
