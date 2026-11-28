// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: See the NOTICE file distributed with this work for additional information regarding copyright ownership
// SPDX-FileContributor: Licensed to the Apache Software Foundation (ASF) under one or more contributor license agreements

use std::collections::{BTreeMap, BTreeSet};
use std::ops::AsyncFnMut;
use std::sync::Arc;

use datafusion::common::{DataFusionError, Result, exec_datafusion_err, exec_err};
use datafusion_dist::{
    LeaseConfig, LeaseError, LeaseToken, LeasedSplitQueue, MonotonicClock, PreparedStage,
    Split, SplitOutput, SplitState, StageManifest, SystemClock,
};

use super::job::StageJob;

pub(super) async fn run_with_timeout(
    job: &StageJob,
    config: LeaseConfig,
) -> Result<StageManifest> {
    run(
        job,
        config,
        Arc::new(SystemClock::default()),
        async |stage, queue, split, token| {
            tokio::time::timeout(
                config.lease_ttl,
                stage.execute_lease(queue, split, token),
            )
            .await
            .map_err(|_| {
                exec_datafusion_err!("Split {} execution timed out", split.split_id)
            })?
        },
    )
    .await
}

pub(super) async fn run(
    job: &StageJob,
    config: LeaseConfig,
    clock: Arc<dyn MonotonicClock>,
    mut execute: impl AsyncFnMut(
        &PreparedStage,
        &LeasedSplitQueue,
        &Split,
        &LeaseToken,
    ) -> Result<SplitOutput>,
) -> Result<StageManifest> {
    let sink = job.sink(None)?;
    let pending = job
        .pending_splits
        .iter()
        .map(|split| job.split(split.split_id, split.attempt_id))
        .collect::<Result<Vec<_>>>()?;
    if pending.is_empty() {
        return sink.finalize_stage(&job.splits).await;
    }
    let first_attempt: BTreeMap<_, _> = pending
        .iter()
        .map(|split| (split.split_id, split.attempt_id))
        .collect();
    let queue = LeasedSplitQueue::new(pending, config, clock)?;
    let stage = job.worker()?.prepare_stage(job.stage())?;
    let mut published_during_expiry = BTreeSet::new();

    while let Some((mut split, token)) = queue.claim(0)? {
        if published_during_expiry.remove(&split.split_id) {
            continue;
        }
        let result = execute(&stage, &queue, &split, &token).await;
        if !renew(&queue, &token)? {
            continue;
        }
        let mut output = match result {
            Ok(output) => output,
            Err(error) => {
                retry(&queue, &token, error)?;
                continue;
            }
        };
        split.attempt_id = first_attempt[&split.split_id]
            .checked_add(token.attempt_id)
            .ok_or_else(|| {
                exec_datafusion_err!("Attempt IDs exhausted for split {}", split.split_id)
            })?;
        let result = sink.write_attempt(&split, &output.batches).await;
        if !renew(&queue, &token)? {
            continue;
        }
        let artifact = match result {
            Ok(artifact) => artifact,
            Err(error) => {
                retry(&queue, &token, error)?;
                continue;
            }
        };
        sink.commit_split_with_metrics(&split, artifact, output.metrics.clone())
            .await?;
        // The on-disk checkpoint is now authoritative. Keep only metrics in the queue.
        output.batches = vec![];
        match queue.commit(&token, output) {
            Ok(_) => {}
            Err(LeaseError::LeaseExpired | LeaseError::StaleAttempt) => {
                published_during_expiry.insert(split.split_id);
            }
            Err(error) => return Err(error.into()),
        }
    }
    sink.finalize_stage(&job.splits).await
}

fn renew(queue: &LeasedSplitQueue, token: &LeaseToken) -> Result<bool> {
    match queue.heartbeat(token) {
        Ok(()) => Ok(true),
        Err(LeaseError::LeaseExpired | LeaseError::StaleAttempt) => {
            if queue.state(token.split_id)? == SplitState::Failed {
                return exec_err!(
                    "Split {} exhausted attempts after lease expiry",
                    token.split_id
                );
            }
            Ok(false)
        }
        Err(error) => Err(error.into()),
    }
}

fn retry(
    queue: &LeasedSplitQueue,
    token: &LeaseToken,
    error: DataFusionError,
) -> Result<()> {
    let retryable = !matches!(error, DataFusionError::Plan(_));
    match queue.report_failure(token, retryable, error.to_string()) {
        Ok(()) | Err(LeaseError::LeaseExpired | LeaseError::StaleAttempt) => {
            if queue.state(token.split_id)? == SplitState::Failed {
                return Err(error);
            }
            Ok(())
        }
        Err(error) => Err(error.into()),
    }
}

#[cfg(test)]
mod tests {
    use std::fs::File;
    use std::time::Duration;

    use datafusion::arrow::array::Int64Array;
    use datafusion::arrow::datatypes::{DataType, Field, Schema};
    use datafusion::arrow::record_batch::RecordBatch;
    use datafusion::parquet::arrow::ArrowWriter;
    use datafusion_dist::{ManualClock, SinkOptions};
    use tempfile::tempdir;

    use super::*;
    use crate::job::PlanningOptions;

    #[tokio::test]
    async fn local_retries_expired_attempt_and_drops_late_output() -> Result<()> {
        let directory = tempdir()?;
        let input = directory.path().join("input");
        std::fs::create_dir(&input)?;
        let schema =
            Arc::new(Schema::new(vec![Field::new("id", DataType::Int64, false)]));
        for value in [1, 2] {
            let mut writer = ArrowWriter::try_new(
                File::create(input.join(format!("{value}.parquet")))?,
                Arc::clone(&schema),
                None,
            )?;
            writer.write(&RecordBatch::try_new(
                Arc::clone(&schema),
                vec![Arc::new(Int64Array::from(vec![value]))],
            )?)?;
            writer.close()?;
        }
        let output = directory.path().join("output");
        let job = StageJob::plan(PlanningOptions {
            sql: "SELECT id FROM t".into(),
            tables: vec![format!("t={}", input.display())],
            stage_tables: vec![],
            stage_dlq_tables: vec![],
            #[cfg(feature = "lake")]
            lance_tables: vec![],
            sink: SinkOptions::default(),
            output_root: Some(output.to_str().unwrap().into()),
        })
        .await?;
        let clock = Arc::new(ManualClock::default());
        let lease_ttl = Duration::from_secs(1);
        let config = LeaseConfig {
            lease_ttl,
            ..Default::default()
        };
        let mut calls = Vec::new();
        let manifest = run(
            &job,
            config,
            clock.clone(),
            async |stage, queue, split, token| {
                let output = stage.execute_lease(queue, split, token).await?;
                calls.push((split.split_id, split.attempt_id));
                if (split.split_id, split.attempt_id) == (0, 0) {
                    clock.advance(2 * lease_ttl);
                }
                Ok(output)
            },
        )
        .await?;
        assert_eq!(calls, [(0, 0), (1, 0), (0, 1)]);
        assert_eq!(manifest.output_rows, 2);
        let root = output.join("stages").join(&job.stage().plan_fingerprint);
        let first_marker = root
            .join("splits")
            .join(&job.splits[0].fingerprint)
            .join("COMMITTED.json");
        let first: serde_json::Value =
            super::super::read_json(File::open(first_marker)?)?;
        assert_eq!(first["winning_attempt_id"], 1);
        assert!(
            !root
                .join("attempts/split-0000000000000000-attempt-0000")
                .exists()
        );
        Ok(())
    }
}
