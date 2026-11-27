// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: See the NOTICE file distributed with this work for additional information regarding copyright ownership
// SPDX-FileContributor: Licensed to the Apache Software Foundation (ASF) under one or more contributor license agreements

use std::sync::Arc;

use datafusion::common::{Result, exec_err};
use datafusion::datasource::source::DataSourceExec;
use datafusion::execution::context::SessionContext;
use datafusion::physical_plan::ExecutionPlan;
use datafusion::physical_plan::metrics::MetricValue;
use futures::TryStreamExt;

use crate::codec::DistExtensionRegistry;
use crate::plan::{
    DistCodecs, FileExtensionCodecRegistry, PreparedStage, Split, SplitMetrics,
    SplitOutput, StagePlan, bind_split_to_template, decode_stage_template_with_codecs,
};
use crate::scheduler::{LeaseToken, LeasedSplitQueue};

/// A local worker with its own session and runtime by default.
pub struct LocalWorker {
    context: SessionContext,
    codecs: DistCodecs,
}

impl Default for LocalWorker {
    fn default() -> Self {
        Self::new()
    }
}

impl LocalWorker {
    /// Create a worker with its own session and runtime.
    pub fn new() -> Self {
        Self::with_context(SessionContext::new())
    }

    /// Use a caller-configured session and runtime for decoding and execution.
    pub fn with_context(context: SessionContext) -> Self {
        Self {
            context,
            codecs: DistCodecs::default(),
        }
    }

    /// Use the supplied [`DistCodecs`] for stage preparation and split binding.
    pub fn with_codecs(mut self, codecs: DistCodecs) -> Self {
        self.codecs = codecs;
        self
    }

    /// Use the supplied physical extension codecs when preparing stages.
    pub fn with_physical_codecs(
        mut self,
        physical_codecs: DistExtensionRegistry,
    ) -> Self {
        self.codecs.physical = physical_codecs;
        self
    }

    /// Use the supplied file extension codecs when binding splits on this worker.
    pub fn with_extension_codecs(
        mut self,
        extension_codecs: FileExtensionCodecRegistry,
    ) -> Self {
        self.codecs.file_extensions = extension_codecs;
        self
    }

    /// The session used by this worker.
    pub fn context(&self) -> &SessionContext {
        &self.context
    }

    /// Prepare a stage once to execute multiple splits in this worker's context.
    pub fn prepare_stage(&self, stage: &StagePlan) -> Result<PreparedStage> {
        decode_stage_template_with_codecs(stage, &self.context, &self.codecs)
    }

    /// Prepare the stage and execute one split.
    pub async fn execute_split(
        &self,
        stage: &StagePlan,
        split: &Split,
    ) -> Result<SplitOutput> {
        self.prepare_stage(stage)?.execute_split(split).await
    }
}

impl PreparedStage {
    /// Drain the whole single-partition stream before reporting this attempt.
    pub async fn execute_split(&self, split: &Split) -> Result<SplitOutput> {
        let plan = bind_split_to_template(self, split)?;
        self.execute_bound_split(split, plan).await
    }

    /// Verify a claimed split, start its lease, and drain its single partition.
    /// The caller publishes the output through [`LeasedSplitQueue::commit`].
    pub async fn execute_lease(
        &self,
        queue: &LeasedSplitQueue,
        split: &Split,
        token: &LeaseToken,
    ) -> Result<SplitOutput> {
        if (split.split_id, split.attempt_id) != (token.split_id, token.attempt_id) {
            return exec_err!("Split does not match lease token");
        }
        queue.verify_leased_split(split, token)?;
        let plan = bind_split_to_template(self, split)?;
        queue.validate_plan(plan.as_ref(), &self.codecs.physical)?;
        queue.start(token)?;
        self.execute_bound_split(split, plan).await
    }

    async fn execute_bound_split(
        &self,
        split: &Split,
        plan: Arc<dyn ExecutionPlan>,
    ) -> Result<SplitOutput> {
        let batches: Vec<_> = plan
            .execute(0, Arc::clone(&self.task_ctx))?
            .try_collect()
            .await?;
        let mut metrics = SplitMetrics {
            output_rows: batches.iter().map(|batch| batch.num_rows()).sum(),
            ..Default::default()
        };
        collect_metrics(plan.as_ref(), &mut metrics);
        Ok(SplitOutput {
            split_id: split.split_id,
            attempt_id: split.attempt_id,
            batches,
            metrics,
        })
    }
}

fn collect_metrics(plan: &dyn ExecutionPlan, output: &mut SplitMetrics) {
    if let Some(metrics) = plan.metrics() {
        if let Some(submitted) = metrics.sum(|metric| {
            matches!(metric.value(), MetricValue::Count { name, .. } if name == "submitted_prompts")
        }) {
            output.submitted_prompts += submitted.as_usize();
        }
        if plan.is::<DataSourceExec>()
            && let Some(MetricValue::PruningMetrics {
                pruning_metrics, ..
            }) = metrics.sum(|metric| {
                matches!(metric.value(), MetricValue::PruningMetrics { name, .. } if name == "row_groups_pruned_statistics")
            })
        {
            output.row_groups_pruned_statistics += pruning_metrics.pruned();
            output.row_groups_matched_statistics += pruning_metrics.matched();
        }
    }
    for child in plan.children() {
        collect_metrics(child.as_ref(), output);
    }
}
