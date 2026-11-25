// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: See the NOTICE file distributed with this work for additional information regarding copyright ownership
// SPDX-FileContributor: Licensed to the Apache Software Foundation (ASF) under one or more contributor license agreements

//! Native SQL planning and asynchronous execution for the opt-in `llm` feature.

mod backend;
#[cfg(feature = "llm-http")]
mod http;
mod logical;
mod options;
mod physical;
mod stream;

pub use backend::{
    CancellationToken, LlmBackend, LlmBackendError, LlmProviderRegistry, LlmRequest,
    LlmResponse, LlmUsage, MockLlmBackend,
};
#[cfg(feature = "llm-http")]
pub use http::{OpenAiHttpBackend, OpenAiHttpConfig};
pub use options::{LlmOptions, OnError};

use std::sync::Arc;

use datafusion_expr::ScalarUDF;
use datafusion_optimizer::simplify_expressions::SimplifyExpressions;
use datafusion_optimizer::{Analyzer, Optimizer, OptimizerRule};

pub(crate) use physical::plan_llm;

pub(crate) fn marker_udf() -> Arc<ScalarUDF> {
    Arc::new(ScalarUDF::from(logical::LlmMarkerUdf::default()))
}

pub(crate) fn analyzer() -> Analyzer {
    let mut analyzer = Analyzer::new();
    analyzer.rules.push(Arc::new(logical::ValidateLlm));
    analyzer
}

pub(crate) fn optimizer(
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
