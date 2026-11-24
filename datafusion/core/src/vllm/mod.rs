// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: See the NOTICE file distributed with this work for additional information regarding copyright ownership
// SPDX-FileContributor: Licensed to the Apache Software Foundation (ASF) under one or more contributor license agreements

//! Native SQL planning and mock execution for the opt-in `vllm` feature.

mod logical;
mod options;
mod physical;

use std::sync::Arc;

use datafusion_expr::ScalarUDF;
use datafusion_optimizer::simplify_expressions::SimplifyExpressions;
use datafusion_optimizer::{Analyzer, Optimizer, OptimizerRule};

pub(crate) use physical::plan_vllm;

pub(crate) fn marker_udf() -> Arc<ScalarUDF> {
    Arc::new(ScalarUDF::from(logical::VllmMarkerUdf::default()))
}

pub(crate) fn analyzer() -> Analyzer {
    let mut analyzer = Analyzer::new();
    analyzer.rules.push(Arc::new(logical::ValidateVllm));
    analyzer
}

pub(crate) fn optimizer() -> Optimizer {
    let mut optimizer = Optimizer::new();
    let simplify = optimizer
        .rules
        .iter()
        .position(|rule| rule.name() == SimplifyExpressions::new().name())
        .expect("Default optimizer includes SimplifyExpressions");
    optimizer
        .rules
        .insert(simplify + 1, Arc::new(logical::VllmProjectRule::default()));
    optimizer
}
