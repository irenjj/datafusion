// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: See the NOTICE file distributed with this work for additional information regarding copyright ownership
// SPDX-FileContributor: Licensed to the Apache Software Foundation (ASF) under one or more contributor license agreements

//! Local map-only distribution of optimized physical plans.

pub mod codec;
pub mod plan;
pub mod runner;

pub use codec::LlmPhysicalCodec;
pub use plan::{
    PreparedStage, Split, SplitMetrics, SplitOutput, StagePlan, bind_split_to_plan,
    bind_split_to_template, decode_stage_template, split_plan_for_distribution,
};
pub use runner::{InMemorySplitQueue, LocalWorker};
