// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: See the NOTICE file distributed with this work for additional information regarding copyright ownership
// SPDX-FileContributor: Licensed to the Apache Software Foundation (ASF) under one or more contributor license agreements

//! DataFusion extension that lowers projection-only SQL vllm calls to a mock operator.

mod logical;
mod options;
mod physical;

use std::sync::Arc;

use datafusion::execution::{SessionStateBuilder, context::SessionContext};
use datafusion::logical_expr::ScalarUDF;
use datafusion::optimizer::{
    Optimizer, OptimizerRule, simplify_expressions::SimplifyExpressions,
};

pub use logical::{VllmMarkerUdf, VllmNode, VllmProjectRule};
pub use physical::{VllmExec, VllmPlanner};

/// Create a session with the marker UDF, logical rewrite and extension planner installed.
pub fn create_vllm_session_context() -> SessionContext {
    let mut rules = Optimizer::new().rules;
    let simplify = rules
        .iter()
        .position(|rule| rule.name() == SimplifyExpressions::new().name())
        .expect("DataFusion 54.1.0 default optimizer includes SimplifyExpressions");
    rules.insert(simplify + 1, Arc::new(VllmProjectRule::default()));

    let state = SessionStateBuilder::new()
        .with_default_features()
        .with_analyzer_rule(Arc::new(logical::ValidateVllm))
        .with_optimizer_rules(rules)
        .with_query_planner(Arc::new(physical::VllmQueryPlanner))
        .build();
    let context = SessionContext::new_with_state(state);
    context.register_udf(ScalarUDF::from(VllmMarkerUdf::default()));
    context
}
