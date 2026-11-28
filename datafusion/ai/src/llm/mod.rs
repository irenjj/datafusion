// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: See the NOTICE file distributed with this work for additional information regarding copyright ownership
// SPDX-FileContributor: Licensed to the Apache Software Foundation (ASF) under one or more contributor license agreements

//! LLM query planning and asynchronous inference with pluggable providers.

mod backend;
mod logical;
mod options;
mod physical;
mod stream;

pub use backend::{
    CancellationToken, LlmBackend, LlmBackendError, LlmProviderRegistry, LlmRequest,
    LlmResponse, LlmUsage, MockLlmBackend,
};
pub use logical::LlmNode;
pub use options::{LlmOptions, OnError};
pub use physical::{LlmExec, plan_llm};

use std::sync::Arc;

use datafusion_expr::ScalarUDF;
use datafusion_optimizer::simplify_expressions::SimplifyExpressions;
use datafusion_optimizer::{Analyzer, Optimizer, OptimizerRule};

/// Creates the SQL marker for calls extracted into an [`LlmNode`].
///
/// # Public Only for Internal Use:
///
/// This is not a public API and is for internal use only; see [API policy] for details.
/// [API policy]: https://datafusion.apache.org/contributor-guide/api-health.html#internal-public-apis
#[doc(hidden)]
pub fn marker_udf() -> Arc<ScalarUDF> {
    Arc::new(ScalarUDF::from(logical::LlmMarkerUdf::default()))
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
