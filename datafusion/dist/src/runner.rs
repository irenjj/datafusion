// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: See the NOTICE file distributed with this work for additional information regarding copyright ownership
// SPDX-FileContributor: Licensed to the Apache Software Foundation (ASF) under one or more contributor license agreements

use std::collections::VecDeque;
use std::sync::{Arc, Mutex};

use datafusion::common::{Result, exec_datafusion_err};
use datafusion::datasource::source::DataSourceExec;
use datafusion::execution::context::SessionContext;
use datafusion::llm::LlmExec;
use datafusion::physical_plan::ExecutionPlan;
use datafusion::physical_plan::metrics::MetricValue;
use futures::TryStreamExt;

use crate::plan::{
    PreparedStage, Split, SplitMetrics, SplitOutput, StagePlan, bind_split_to_template,
    decode_stage_template,
};

/// A local worker with its own session and runtime by default.
pub struct LocalWorker {
    context: SessionContext,
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
        Self { context }
    }

    /// The session used by this worker.
    pub fn context(&self) -> &SessionContext {
        &self.context
    }

    /// Prepare a stage once to execute multiple splits in this worker's context.
    pub fn prepare_stage(&self, stage: &StagePlan) -> Result<PreparedStage> {
        decode_stage_template(stage, &self.context)
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
        if plan.is::<LlmExec>()
            && let Some(submitted) = metrics.sum_by_name("submitted_prompts")
        {
            output.submitted_prompts += submitted.as_usize();
        }
        if plan.is::<DataSourceExec>()
            && let Some(MetricValue::PruningMetrics {
                pruning_metrics, ..
            }) = metrics.sum_by_name("row_groups_pruned_statistics")
        {
            output.row_groups_pruned_statistics += pruning_metrics.pruned();
            output.row_groups_matched_statistics += pruning_metrics.matched();
        }
    }
    for child in plan.children() {
        collect_metrics(child.as_ref(), output);
    }
}

/// A stage's finite work queue. Each entry is handed to one claimant.
pub struct InMemorySplitQueue {
    pending: Mutex<VecDeque<Split>>,
}

impl InMemorySplitQueue {
    /// Populate the queue in claim order.
    pub fn new(splits: impl IntoIterator<Item = Split>) -> Self {
        Self {
            pending: Mutex::new(splits.into_iter().collect()),
        }
    }

    /// Claim the next split. Workers call this again after finishing the current split.
    pub fn claim(&self) -> Result<Option<Split>> {
        let mut pending = self
            .pending
            .lock()
            .map_err(|_| exec_datafusion_err!("Split queue mutex poisoned"))?;
        Ok(pending.pop_front())
    }
}
