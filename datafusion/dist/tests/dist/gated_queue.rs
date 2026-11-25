// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: See the NOTICE file distributed with this work for additional information regarding copyright ownership
// SPDX-FileContributor: Licensed to the Apache Software Foundation (ASF) under one or more contributor license agreements

use std::collections::BTreeMap;
use std::sync::{Arc, Mutex};

use datafusion::common::Result;
use datafusion::physical_plan::collect;
use datafusion::prelude::SessionContext;
use datafusion_dist::{
    InMemorySplitQueue, LocalWorker, Split, SplitOutput, StagePlan,
    split_plan_for_distribution,
};
use tempfile::tempdir;
use tokio::sync::oneshot;

use super::harness::{
    FILTERED, assert_attempts, assert_independent, assert_matches_single_node,
    coordinator, optimized, register_parquet, write_parquet,
};

pub(super) async fn gated_drain(
    ctx: &SessionContext,
    stage: &StagePlan,
    splits: &[Split],
) -> Result<(Vec<SplitOutput>, Vec<SplitOutput>)> {
    let workers = [LocalWorker::new(), LocalWorker::new()];
    assert_independent(ctx, &workers);
    let slow_stage = workers[0].prepare_stage(stage)?;
    let fast_stage = workers[1].prepare_stage(stage)?;
    // Create the gates after preparation, so neither waiter can be stranded by a setup error.
    let queue = InMemorySplitQueue::new(splits.iter().cloned());
    let (claimed, wait_for_claim) = oneshot::channel();
    let (release, wait_for_release) = oneshot::channel();
    let (slow, fast) = tokio::try_join!(
        async {
            let split = queue.claim()?.expect("first split");
            claimed.send(()).expect("fast worker waits for claim");
            wait_for_release.await.expect("fast worker drained queue");
            let output = slow_stage.execute_split(&split).await?;
            assert!(queue.claim()?.is_none());
            Ok::<_, datafusion::common::DataFusionError>(vec![output])
        },
        async {
            wait_for_claim
                .await
                .expect("slow worker claimed first split");
            let mut outputs = Vec::new();
            while let Some(split) = queue.claim()? {
                outputs.push(fast_stage.execute_split(&split).await?);
            }
            release.send(()).expect("slow worker waits for release");
            Ok(outputs)
        },
    )?;
    Ok((slow, fast))
}

enum Status {
    Pending,
    Running,
    Committed(SplitOutput),
}

struct Entry {
    split: Split,
    status: Status,
}

impl Entry {
    fn is_running(&self, attempt_id: u32) -> bool {
        self.split.attempt_id == attempt_id && matches!(self.status, Status::Running)
    }

    fn requeue(&mut self) {
        self.split.attempt_id += 1;
        self.status = Status::Pending;
    }
}

struct LeaseProbe {
    entries: Mutex<BTreeMap<u64, Entry>>,
}

impl LeaseProbe {
    fn new(splits: &[Split]) -> Self {
        Self {
            entries: Mutex::new(
                splits
                    .iter()
                    .map(|split| {
                        (
                            split.split_id,
                            Entry {
                                split: split.clone(),
                                status: Status::Pending,
                            },
                        )
                    })
                    .collect(),
            ),
        }
    }

    fn claim(&self) -> Option<Lease<'_>> {
        let mut entries = self.entries.lock().unwrap();
        let entry = entries
            .values_mut()
            .find(|entry| matches!(entry.status, Status::Pending))?;
        entry.status = Status::Running;
        Some(Lease {
            queue: self,
            split: entry.split.clone(),
        })
    }

    fn reclaim(&self, split_id: u64, attempt_id: u32) -> bool {
        let mut entries = self.entries.lock().unwrap();
        let entry = entries.get_mut(&split_id).unwrap();
        if entry.is_running(attempt_id) {
            entry.requeue();
            true
        } else {
            false
        }
    }

    fn into_outputs(self) -> Vec<SplitOutput> {
        self.entries
            .into_inner()
            .unwrap()
            .into_values()
            .map(|entry| {
                let Status::Committed(output) = entry.status else {
                    panic!("split {} was never committed", entry.split.split_id);
                };
                output
            })
            .collect()
    }
}

struct Lease<'a> {
    queue: &'a LeaseProbe,
    split: Split,
}

impl Lease<'_> {
    fn commit(self, output: SplitOutput) -> bool {
        assert_eq!(
            (output.split_id, output.attempt_id),
            (self.split.split_id, self.split.attempt_id)
        );
        let mut entries = self.queue.entries.lock().unwrap();
        let entry = entries.get_mut(&output.split_id).unwrap();
        if entry.is_running(output.attempt_id) {
            entry.status = Status::Committed(output);
            true
        } else {
            false
        }
    }
}

impl Drop for Lease<'_> {
    fn drop(&mut self) {
        let mut entries = self.queue.entries.lock().unwrap_or_else(|e| e.into_inner());
        if let Some(entry) = entries.get_mut(&self.split.split_id)
            && entry.is_running(self.split.attempt_id)
        {
            entry.requeue();
        }
    }
}

#[tokio::test]
async fn reclaimed_lease_fences_late_completion() -> Result<()> {
    let directory = tempdir()?;
    let path = directory.path().join("data.parquet");
    write_parquet(&path, &[(0, 0), (2, 20)])?;
    let ctx = coordinator();
    register_parquet(&ctx, &path, false).await?;
    let native = optimized(&ctx, FILTERED).await?;
    let schema = native.schema();
    let baseline = collect(Arc::clone(&native), ctx.task_ctx()).await?;
    let (stage, mut splits) = split_plan_for_distribution(native, 20)?;
    assert_eq!(splits.len(), 1);
    let workers = [LocalWorker::new(), LocalWorker::new()];
    assert_independent(&ctx, &workers);
    let original_stage = workers[0].prepare_stage(&stage)?;
    let replacement_stage = workers[1].prepare_stage(&stage)?;
    let queue = LeaseProbe::new(&splits);
    let (claimed, wait_for_claim) = oneshot::channel();
    let (release, wait_for_release) = oneshot::channel();
    tokio::try_join!(
        async {
            let original = queue.claim().expect("original lease");
            assert_eq!((original.split.split_id, original.split.attempt_id), (0, 0));
            claimed.send(()).unwrap();
            wait_for_release.await.unwrap();
            let output = original_stage.execute_split(&original.split).await?;
            assert_eq!(output.metrics.output_rows, 2);
            assert!(!original.commit(output), "stale attempt must be fenced");
            Ok::<_, datafusion::common::DataFusionError>(())
        },
        async {
            wait_for_claim.await.unwrap();
            assert!(queue.claim().is_none());
            assert!(queue.reclaim(0, 0));
            let replacement = queue.claim().expect("replacement lease");
            assert_eq!(
                (replacement.split.split_id, replacement.split.attempt_id),
                (0, 1)
            );
            assert!(!queue.reclaim(0, 0));
            let output = replacement_stage.execute_split(&replacement.split).await?;
            assert!(replacement.commit(output));
            release.send(()).unwrap();
            Ok(())
        },
    )?;
    assert!(queue.claim().is_none());
    assert!(!queue.reclaim(0, 1));
    let outputs = queue.into_outputs();
    splits[0].attempt_id = 1;
    assert_attempts(&splits, &outputs);
    assert_matches_single_node(&stage, &schema, &baseline, &outputs)?;
    assert_eq!(outputs[0].metrics.submitted_prompts, 1);
    Ok(())
}
