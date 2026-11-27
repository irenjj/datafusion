// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: See the NOTICE file distributed with this work for additional information regarding copyright ownership
// SPDX-FileContributor: Licensed to the Apache Software Foundation (ASF) under one or more contributor license agreements

use std::pin::pin;
use std::sync::Arc;
use std::time::Duration;

use datafusion::common::{DataFusionError, Result};
use datafusion::execution::object_store::ObjectStoreUrl;
use datafusion::object_store::{
    ObjectStore, limit::LimitStore, local::LocalFileSystem, path::Path,
};
use datafusion::physical_plan::{ExecutionPlan, collect};
use datafusion_dist::{
    CommitOutcome, LeaseConfig, LeaseError, LeasedSplitQueue, LocalWorker, ManualClock,
    Split, SplitOutput, SplitState, split_plan_for_distribution_with_codecs,
};
use futures::{TryStreamExt, poll};
use tempfile::tempdir;
use tokio::sync::oneshot;

use super::harness::{
    MockAiCodec, SpeculatableCodec, SpeculatableExec, ai_codecs, assert_attempts,
    assert_independent, assert_matches_single_node, assert_plan_error, column_expr,
    coordinator, filtered_plan, register_parquet, scan_plan, submitted, write_parquet,
};

#[tokio::test]
async fn slow_and_fast_workers_drain_leased_queue_equivalently() -> Result<()> {
    let dir = tempdir()?;
    for (file, id) in [("a.parquet", 0), ("b.parquet", 4), ("c.parquet", 8)] {
        write_parquet(&dir.path().join(file), &[(id, 0), (id + 2, 20)])?;
    }
    let ctx = coordinator();
    register_parquet(&ctx, dir.path(), false).await?;
    let native = filtered_plan(&ctx, None).await?;
    let baseline = collect(Arc::clone(&native), ctx.task_ctx()).await?;
    let expected_prompts = submitted(native.as_ref());
    assert_eq!(expected_prompts, 3);

    let codecs = ai_codecs()?;
    let (stage, splits) =
        split_plan_for_distribution_with_codecs(Arc::clone(&native), 20, &codecs)?;
    assert_eq!(splits.len(), 3);
    let workers = [
        LocalWorker::new().with_codecs(codecs.clone()),
        LocalWorker::new().with_codecs(codecs),
    ];
    assert_independent(&ctx, &workers);
    let slow_stage = workers[0].prepare_stage(&stage)?;
    let fast_stage = workers[1].prepare_stage(&stage)?;
    let queue = LeasedSplitQueue::new(
        splits.iter().cloned(),
        LeaseConfig::default(),
        Arc::new(ManualClock::default()),
    )?;
    let (claimed, wait_for_claim) = oneshot::channel();
    let (release, wait_for_release) = oneshot::channel();
    tokio::try_join!(
        async {
            let (split, token) = queue.claim(0)?.expect("slow worker claims first");
            assert_eq!(split.split_id, splits[0].split_id);
            claimed.send(split.split_id).unwrap();
            wait_for_release.await.unwrap();
            let output = slow_stage.execute_lease(&queue, &split, &token).await?;
            assert_eq!(queue.commit(&token, output)?, CommitOutcome::Committed);
            Ok::<_, DataFusionError>(())
        },
        async {
            let slow_id = wait_for_claim.await.unwrap();
            while let Some((split, token)) = queue.claim(1)? {
                let output = fast_stage.execute_lease(&queue, &split, &token).await?;
                assert_eq!(queue.commit(&token, output)?, CommitOutcome::Committed);
            }
            assert_eq!(queue.state(slow_id)?, SplitState::Leased);
            assert_eq!(queue.committed_outputs()?.len(), splits.len() - 1);
            release.send(()).unwrap();
            Ok(())
        },
    )?;

    let outputs = take_outputs(queue)?;
    assert_attempts(&splits, &outputs);
    assert_matches_single_node(&stage, native.schema().as_ref(), &baseline, &outputs)?;
    assert_eq!(
        outputs
            .iter()
            .map(|o| o.metrics.submitted_prompts)
            .sum::<usize>(),
        expected_prompts
    );
    Ok(())
}

#[tokio::test]
async fn reclaimed_lease_fences_late_completion() -> Result<()> {
    let dir = tempdir()?;
    let path = dir.path().join("data.parquet");
    write_parquet(&path, &[(0, 0), (2, 20)])?;
    let ctx = coordinator();
    register_parquet(&ctx, &path, false).await?;
    let native = filtered_plan(&ctx, None).await?;
    let baseline = collect(Arc::clone(&native), ctx.task_ctx()).await?;
    let codecs = ai_codecs()?;
    let (stage, mut splits) =
        split_plan_for_distribution_with_codecs(Arc::clone(&native), 20, &codecs)?;
    assert_eq!(splits.len(), 1);

    let workers = [
        LocalWorker::new().with_codecs(codecs.clone()),
        LocalWorker::new().with_codecs(codecs),
    ];
    assert_independent(&ctx, &workers);
    let slow_store = Arc::new(LimitStore::new(LocalFileSystem::new(), 1));
    let prefix = Path::from_filesystem_path(dir.path()).unwrap();
    // Keep the slow worker's only I/O permit until the replacement has committed.
    let mut held_request = slow_store.list(Some(&prefix));
    assert!(held_request.try_next().await?.is_some());
    workers[0]
        .context()
        .register_object_store(ObjectStoreUrl::local_filesystem().as_ref(), slow_store);
    let slow_stage = workers[0].prepare_stage(&stage)?;
    let fast_stage = workers[1].prepare_stage(&stage)?;
    let clock = Arc::new(ManualClock::default());
    let config = LeaseConfig::default();
    let queue = LeasedSplitQueue::new(splits.iter().cloned(), config, clock.clone())?;
    let (started, wait_for_start) = oneshot::channel();
    let (release, wait_for_release) = oneshot::channel();
    tokio::try_join!(
        async {
            let (split, token) = queue.claim(0)?.expect("original lease");
            assert_eq!((split.split_id, split.attempt_id), (0, 0));
            let mut execution = pin!(slow_stage.execute_lease(&queue, &split, &token));
            assert!(poll!(&mut execution).is_pending());
            assert_eq!(queue.state(split.split_id)?, SplitState::Running);
            started.send(token).unwrap();
            wait_for_release.await.unwrap();
            drop(held_request);
            let output = execution.await?;
            assert_eq!(output.metrics.output_rows, 2);
            assert_eq!(queue.commit(&token, output), Err(LeaseError::StaleAttempt));
            Ok::<_, DataFusionError>(())
        },
        async {
            let original = wait_for_start.await.unwrap();
            assert!(queue.claim(1)?.is_none());
            clock.advance(config.lease_ttl + Duration::from_nanos(1));
            let (split, token) = queue.claim(1)?.expect("replacement lease");
            assert_eq!((split.split_id, split.attempt_id), (original.split_id, 1));
            assert!(token.fencing_epoch > original.fencing_epoch);
            assert!(!split.fingerprint.is_empty());
            assert_eq!(split.fingerprint, splits[0].fingerprint);
            assert_eq!(split.file_proto, splits[0].file_proto);
            let output = fast_stage.execute_lease(&queue, &split, &token).await?;
            assert_eq!(queue.commit(&token, output)?, CommitOutcome::Committed);
            release.send(()).unwrap();
            Ok(())
        },
    )?;

    let outputs = take_outputs(queue)?;
    splits[0].attempt_id = 1;
    assert_attempts(&splits, &outputs);
    assert_matches_single_node(&stage, native.schema().as_ref(), &baseline, &outputs)?;
    assert_eq!(outputs[0].metrics.submitted_prompts, 1);
    Ok(())
}

#[tokio::test]
async fn non_speculatable_plan_rejects_speculation() -> Result<()> {
    let dir = tempdir()?;
    write_parquet(&dir.path().join("a.parquet"), &[(0, 20)])?;
    write_parquet(&dir.path().join("b.parquet"), &[(2, 20)])?;
    let ctx = coordinator();
    register_parquet(&ctx, dir.path(), false).await?;
    let scan = scan_plan(&ctx).await?;
    let nonspeculatable = filtered_plan(&ctx, None).await?;
    let speculatable: Arc<dyn ExecutionPlan> = Arc::new(SpeculatableExec::new(
        Arc::clone(&scan),
        column_expr("prompt_col", scan.as_ref())?,
        "spec-m".into(),
        "{}".into(),
        "resp".into(),
    ));

    let mut codecs = ai_codecs()?;
    codecs
        .physical
        .register(Arc::new(SpeculatableCodec::new("test.speculatable", true)))?;
    let mut ambiguous = codecs.physical.clone();
    ambiguous.register(Arc::new(MockAiCodec::new("test.ambiguous", true)))?;
    let speculating = LeaseConfig {
        speculation_allowed: true,
        ..Default::default()
    };
    let default = LeaseConfig::default();
    for config in [default, speculating] {
        let error = config
            .validate_for_plan(nonspeculatable.as_ref(), &ambiguous)
            .unwrap_err();
        assert_plan_error(&error, "Ambiguous physical extension policy");
    }

    let worker = LocalWorker::new().with_codecs(codecs.clone());
    let refusal = "Speculation is not allowed for plans containing non-speculatable extension operators";
    assert_plan_error(
        &speculating
            .validate_for_plan(nonspeculatable.as_ref(), &codecs.physical)
            .unwrap_err(),
        refusal,
    );
    let (stage, splits) = split_plan_for_distribution_with_codecs(
        Arc::clone(&nonspeculatable),
        20,
        &codecs,
    )?;
    let prepared = worker.prepare_stage(&stage)?;
    let queue =
        LeasedSplitQueue::new(splits, speculating, Arc::new(ManualClock::default()))?;
    let (split, token) = queue.claim(0)?.unwrap();
    let error = prepared
        .execute_lease(&queue, &split, &token)
        .await
        .unwrap_err();
    assert_plan_error(&error, refusal);
    assert_eq!(queue.state(split.split_id)?, SplitState::Leased);

    for (plan, config) in [
        (nonspeculatable, default),
        (speculatable, speculating),
        (scan, speculating),
    ] {
        config.validate_for_plan(plan.as_ref(), &codecs.physical)?;
        let (stage, splits) = split_plan_for_distribution_with_codecs(plan, 21, &codecs)?;
        let prepared = worker.prepare_stage(&stage)?;
        assert_eq!(splits.len(), 2);
        let queue = LeasedSplitQueue::new(
            splits.iter().cloned(),
            config,
            Arc::new(ManualClock::default()),
        )?;
        let (split, token) = queue.claim(0)?.unwrap();

        let wrong_attempt = Split {
            attempt_id: token.attempt_id + 1,
            ..split.clone()
        };
        let error = prepared
            .execute_lease(&queue, &wrong_attempt, &token)
            .await
            .unwrap_err();
        assert!(matches!(
            error,
            DataFusionError::Execution(message) if message.starts_with("Split does not match lease token")
        ));
        assert_eq!(queue.state(split.split_id)?, SplitState::Leased);

        for (file_proto, fingerprint) in [
            (splits[1].file_proto.clone(), split.fingerprint.clone()),
            (split.file_proto.clone(), splits[1].fingerprint.clone()),
            (splits[1].file_proto.clone(), splits[1].fingerprint.clone()),
        ] {
            let swapped = Split {
                file_proto,
                fingerprint,
                ..split.clone()
            };
            let error = prepared
                .execute_lease(&queue, &swapped, &token)
                .await
                .unwrap_err();
            assert!(matches!(
                error,
                DataFusionError::External(source)
                    if source.downcast_ref::<LeaseError>() == Some(&LeaseError::ConflictingCommit)
            ));
            assert_eq!(queue.state(split.split_id)?, SplitState::Leased);
        }

        let output = prepared.execute_lease(&queue, &split, &token).await?;
        assert_eq!(output.metrics.output_rows, 2);
        assert_eq!(queue.commit(&token, output)?, CommitOutcome::Committed);
    }
    Ok(())
}

fn take_outputs(queue: LeasedSplitQueue) -> Result<Vec<SplitOutput>> {
    let outputs = queue.committed_outputs()?;
    drop(queue);
    Ok(outputs
        .into_iter()
        .map(|output| Arc::try_unwrap(output).unwrap())
        .collect())
}
