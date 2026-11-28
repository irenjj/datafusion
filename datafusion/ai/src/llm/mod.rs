// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: See the NOTICE file distributed with this work for additional information regarding copyright ownership
// SPDX-FileContributor: Licensed to the Apache Software Foundation (ASF) under one or more contributor license agreements

//! LLM query planning and asynchronous inference with pluggable providers.

mod backend;
#[cfg(feature = "http")]
mod http;
mod image;
mod logical;
mod options;
mod physical;
mod schema;
mod stream;

pub use backend::{
    CancellationToken, LlmBackend, LlmBackendError, LlmEmbeddingResponse,
    LlmProviderRegistry, LlmRequest, LlmResponse, LlmUsage, MockLlmBackend,
};
#[cfg(feature = "http")]
pub use http::{OpenAiHttpBackend, OpenAiHttpConfig};
pub use logical::LlmNode;
pub use options::{ImageResizeFallback, LlmOptions, OnError, SchemaMode};
pub use physical::{LlmExec, MediaPlan, plan_llm};

use std::sync::Arc;

use datafusion_expr::ScalarUDF;
use datafusion_expr::planner::ExprPlanner;
use datafusion_optimizer::simplify_expressions::SimplifyExpressions;
use datafusion_optimizer::{Analyzer, Optimizer, OptimizerRule};

/// The inference task performed by an [`LlmExec`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum TaskKind {
    /// Text completion.
    Chat,
    /// Text embedding.
    Embed,
    /// Text completion with one or more ordered images per prompt.
    Vision,
}

impl TaskKind {
    fn function(self) -> &'static str {
        match self {
            Self::Chat => "llm",
            Self::Embed => "embed",
            Self::Vision => "vlm",
        }
    }
}

/// Creates the SQL marker for calls extracted into an [`LlmNode`].
///
/// # Public Only for Internal Use:
///
/// This is not a public API and is for internal use only; see [API policy] for details.
/// [API policy]: https://datafusion.apache.org/contributor-guide/api-health.html#internal-public-apis
#[doc(hidden)]
pub fn marker_udf() -> Arc<ScalarUDF> {
    marker(TaskKind::Chat)
}

/// Creates the SQL markers for embedding and vision.
///
/// # Public Only for Internal Use:
/// This is not a public API and is for internal use only; see [API policy] for details.
/// [API policy]: https://datafusion.apache.org/contributor-guide/api-health.html#internal-public-apis
#[doc(hidden)]
pub fn multimodal_marker_udfs() -> [Arc<ScalarUDF>; 2] {
    [TaskKind::Embed, TaskKind::Vision].map(marker)
}

fn marker(kind: TaskKind) -> Arc<ScalarUDF> {
    Arc::new(ScalarUDF::from(logical::LlmMarkerUdf::new(kind)))
}

/// Creates the SQL planner for marker options that determine return types.
///
/// # Public Only for Internal Use:
/// This is not a public API and is for internal use only; see [API policy] for details.
/// [API policy]: https://datafusion.apache.org/contributor-guide/api-health.html#internal-public-apis
#[doc(hidden)]
pub fn marker_expr_planner() -> Arc<dyn ExprPlanner> {
    Arc::new(logical::MarkerExprPlanner)
}

/// Creates the default analyzer with SQL LLM usage validation.
///
/// # Public Only for Internal Use:
///
/// This is not a public API and is for internal use only; see [API policy] for details.
/// [API policy]: https://datafusion.apache.org/contributor-guide/api-health.html#internal-public-apis
#[doc(hidden)]
pub fn analyzer() -> Analyzer {
    let mut analyzer = Analyzer::new();
    analyzer.rules.push(Arc::new(logical::ValidateLlm));
    analyzer
}

/// Creates the default LLM-aware optimizer or rebinds the LLM rule in the
/// supplied optimizer to the session's [`LlmProviderRegistry`].
///
/// # Public Only for Internal Use:
///
/// This is not a public API and is for internal use only; see [API policy] for details.
/// [API policy]: https://datafusion.apache.org/contributor-guide/api-health.html#internal-public-apis
#[doc(hidden)]
pub fn optimizer(
    optimizer: Option<Optimizer>,
    registry: &LlmProviderRegistry,
) -> Optimizer {
    let rule = Arc::new(logical::LlmProjectRule::new(registry.clone()));
    if let Some(mut optimizer) = optimizer {
        // A builder derived from another session may replace the registry.
        if let Some(previous) =
            optimizer.rules.iter_mut().find(|r| r.name() == rule.name())
        {
            *previous = rule;
        }
        return optimizer;
    }
    let mut optimizer = Optimizer::new();
    let simplify = optimizer
        .rules
        .iter()
        .position(|rule| rule.name() == SimplifyExpressions::new().name())
        .expect("Default optimizer includes SimplifyExpressions");
    optimizer.rules.insert(simplify + 1, rule);
    optimizer
}
