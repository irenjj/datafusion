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

use super::TaskKind;
use super::options::{LlmOptions, validate_inflight_limit};
use crate::{ErrorCode, ResolvedMedia};
use datafusion_common::{Result, plan_datafusion_err, plan_err};

pub use tokio_util::sync::CancellationToken;

/// A request containing a batch of non-NULL prompts for one model.
#[derive(Debug, Clone)]
pub struct LlmRequest {
    /// Provider model identifier.
    pub model: String,
    /// Prompts in input row order.
    pub prompts: Vec<String>,
    /// One media slot per prompt. Vision slots contain images in input order.
    /// Text-only prompts have a `None` slot.
    pub media: Vec<Option<Vec<ResolvedMedia>>>,
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

/// Embeddings returned in the same order as the prompts.
#[derive(Debug)]
pub struct LlmEmbeddingResponse {
    /// One vector of the requested dimension per prompt.
    pub outputs: Vec<Vec<f32>>,
    /// Usage, when the provider reports it.
    pub usage: Option<LlmUsage>,
}

impl LlmEmbeddingResponse {
    pub(super) fn validate(
        &self,
        prompts: usize,
        dimension: usize,
    ) -> std::result::Result<(), LlmBackendError> {
        if self.outputs.len() != prompts {
            return Err(LlmBackendError::SchemaMismatch {
                message: format!(
                    "backend returned {} embeddings for {prompts} prompts",
                    self.outputs.len()
                ),
            });
        }
        if let Some(size) = self
            .outputs
            .iter()
            .map(Vec::len)
            .find(|&len| len != dimension)
        {
            return Err(LlmBackendError::SchemaMismatch {
                message: format!(
                    "backend returned an embedding of dimension {size}, expected {dimension}"
                ),
            });
        }
        Ok(())
    }
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
    /// The provider's response could not be decoded.
    InvalidResponse {
        /// HTTP status, if applicable.
        status: Option<u16>,
        /// Diagnostic message.
        message: String,
    },
    /// The response did not match the expected number or shape of outputs.
    SchemaMismatch {
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
            Self::Transient { status, message }
            | Self::Permanent { status, message }
            | Self::InvalidResponse { status, message } => {
                if let Some(status) = status {
                    write!(f, "HTTP {status}: ")?;
                }
                f.write_str(message)
            }
            Self::SchemaMismatch { message } => f.write_str(message),
            Self::Cancelled => f.write_str("request cancelled"),
        }
    }
}

impl std::error::Error for LlmBackendError {}

impl LlmBackendError {
    pub(super) fn classification(&self) -> (ErrorCode, bool) {
        match self {
            Self::RateLimited { .. } => (ErrorCode::RateLimited, true),
            Self::Transient { .. } => (ErrorCode::TransientHttp, true),
            Self::Permanent { .. } => (ErrorCode::PermanentHttp, false),
            Self::InvalidResponse { .. } => (ErrorCode::InvalidResponse, false),
            Self::SchemaMismatch { .. } => (ErrorCode::SchemaMismatch, false),
            Self::Cancelled => (ErrorCode::Cancelled, false),
        }
    }
}

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

    /// Preferred batch size for a specific task when SQL does not specify one.
    fn default_batch_size_for(&self, _kind: TaskKind) -> usize {
        self.default_batch_size()
    }

    /// Maximum batch size for a specific task, if bounded.
    fn max_batch_size_for(&self, _kind: TaskKind) -> Option<usize> {
        self.max_batch_size()
    }

    /// Runs one completion attempt. The executor owns the timeout and retry policy.
    async fn generate(
        &self,
        request: LlmRequest,
        cancel: CancellationToken,
    ) -> std::result::Result<LlmResponse, LlmBackendError>;

    /// Runs one embedding attempt. Text-only backends may use the default error.
    async fn embed(
        &self,
        _request: LlmRequest,
        _cancel: CancellationToken,
    ) -> std::result::Result<LlmEmbeddingResponse, LlmBackendError> {
        Err(LlmBackendError::Permanent {
            status: None,
            message: "provider does not support embeddings".into(),
        })
    }
}

/// The built-in deterministic backend. Completions echo the prompt for both
/// text and vision inputs, independent of the supplied images.
#[derive(Debug, Default)]
pub struct MockLlmBackend;

#[async_trait]
impl LlmBackend for MockLlmBackend {
    fn default_batch_size(&self) -> usize {
        1024
    }

    fn default_batch_size_for(&self, kind: TaskKind) -> usize {
        self.max_batch_size_for(kind)
            .unwrap_or_else(|| self.default_batch_size())
    }

    fn max_batch_size_for(&self, kind: TaskKind) -> Option<usize> {
        match kind {
            TaskKind::Chat => None,
            TaskKind::Embed => Some(32),
            TaskKind::Vision => Some(1),
        }
    }

    async fn generate(
        &self,
        request: LlmRequest,
        cancel: CancellationToken,
    ) -> std::result::Result<LlmResponse, LlmBackendError> {
        if cancel.is_cancelled() {
            return Err(LlmBackendError::Cancelled);
        }
        let LlmRequest { model, prompts, .. } = request;
        let outputs = prompts
            .into_iter()
            .map(|prompt| Some(format!("mock[{model}]: {prompt}")))
            .collect();
        Ok(LlmResponse {
            outputs,
            usage: None,
        })
    }

    async fn embed(
        &self,
        request: LlmRequest,
        cancel: CancellationToken,
    ) -> std::result::Result<LlmEmbeddingResponse, LlmBackendError> {
        if cancel.is_cancelled() {
            return Err(LlmBackendError::Cancelled);
        }
        let dimension = request.options.embedding_dimension().map_err(|error| {
            LlmBackendError::SchemaMismatch {
                message: error.to_string(),
            }
        })? as usize;
        let outputs = request
            .prompts
            .iter()
            .map(|prompt| vec![prompt.len() as f32; dimension])
            .collect();
        Ok(LlmEmbeddingResponse {
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
    default_provider: Arc<str>,
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
            default_provider: Arc::from("default"),
        }
    }
}

impl LlmProviderRegistry {
    /// Creates an empty registry and selects the name used for unnamed requests.
    ///
    /// # Public Only for Internal Use:
    ///
    /// This is not a public API and is for internal use only; see [API policy] for details.
    ///
    /// [API policy]: https://datafusion.apache.org/contributor-guide/api-health.html#internal-public-apis
    #[doc(hidden)]
    pub fn empty_with_default(name: impl Into<Arc<str>>) -> Self {
        Self {
            providers: Arc::default(),
            default_provider: name.into(),
        }
    }

    /// Registers or replaces a backend without a provider-wide concurrency cap.
    /// Names must be nonempty and contain only ASCII letters, digits, `_`, `.`, or `-`.
    pub fn register(
        &self,
        name: impl Into<String>,
        backend: Arc<dyn LlmBackend>,
    ) -> Result<()> {
        self.insert(name.into(), backend, None)
    }

    /// Registers or replaces a backend with a cap on concurrent requests shared
    /// across queries and partitions. A request holds its slot through retries.
    /// The name must follow the same rules as [`Self::register`].
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
        if !name.bytes().all(|byte| {
            byte.is_ascii_alphanumeric() || matches!(byte, b'_' | b'.' | b'-')
        }) {
            return plan_err!(
                "llm provider name may contain only ASCII letters, digits, _, ., or -"
            );
        }
        self.providers
            .write()
            .insert(name, RegisteredProvider { backend, limit });
        Ok(())
    }

    pub(super) fn default_name(&self) -> &str {
        &self.default_provider
    }

    pub(super) fn resolve(&self, name: Option<&str>) -> Result<RegisteredProvider> {
        let name = name.unwrap_or_else(|| self.default_name());
        self.providers
            .read()
            .get(name)
            .cloned()
            .ok_or_else(|| plan_datafusion_err!("Unknown llm provider '{name}'"))
    }
}
