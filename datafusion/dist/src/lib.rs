// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: See the NOTICE file distributed with this work for additional information regarding copyright ownership
// SPDX-FileContributor: Licensed to the Apache Software Foundation (ASF) under one or more contributor license agreements

//! Map-only physical plan transport and local execution of Parquet splits.

mod codec;
mod fingerprint;
mod plan;
mod runner;

pub use codec::{
    DistExtensionRegistry, DistPhysicalCodec, ExtensionOperatorPolicy,
    NamedPhysicalExtensionCodec,
};
pub use plan::{
    DistCodecs, FileExtensionCodec, FileExtensionCodecRegistry, PreparedStage, Split,
    SplitMetrics, SplitOutput, StagePlan, bind_split_to_plan,
    bind_split_to_plan_with_codecs, bind_split_to_plan_with_extensions,
    bind_split_to_template, bind_split_to_template_with_extensions,
    decode_stage_template, decode_stage_template_with_codecs,
    decode_stage_template_with_extensions, split_plan_for_distribution,
    split_plan_for_distribution_with_codecs, split_plan_for_distribution_with_extensions,
};
pub use runner::LocalWorker;
