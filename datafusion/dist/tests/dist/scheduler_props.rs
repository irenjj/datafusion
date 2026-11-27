// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: See the NOTICE file distributed with this work for additional information regarding copyright ownership
// SPDX-FileContributor: Licensed to the Apache Software Foundation (ASF) under one or more contributor license agreements

use std::collections::{BTreeMap, VecDeque};
use std::env;
use std::sync::Arc;
use std::time::Duration;

use datafusion::arrow::array::Int32Array;
use datafusion::arrow::record_batch::RecordBatch;
use datafusion::common::DataFusionError;
use datafusion_dist::{
    CommitOutcome, LeaseConfig, LeaseError, LeaseToken, LeasedSplitQueue, ManualClock,
    MonotonicClock, Split, SplitMetrics, SplitOutput, SplitState,
};
use proptest::prelude::*;
use proptest::test_runner::{Config, RngSeed};

fn queue(ids: &[u64], config: LeaseConfig) -> (LeasedSplitQueue, Arc<ManualClock>) {
    let clock = Arc::new(ManualClock::default());
    let splits = ids.iter().map(|&split_id| Split {
        split_id,
        attempt_id: 0,
        file_proto: vec![],
        fingerprint: String::new(),
    });
    (
        LeasedSplitQueue::new(splits, config, clock.clone()).unwrap(),
        clock,
    )
}

fn claim(queue: &LeasedSplitQueue, worker: u64) -> LeaseToken {
    let (split, token) = queue.claim(worker).unwrap().unwrap();
    assert_eq!(
        (split.split_id, split.attempt_id),
        (token.split_id, token.attempt_id)
    );
    token
}

fn output(token: LeaseToken, row: i32) -> SplitOutput {
    let batch = RecordBatch::try_from_iter([(
        "value",
        Arc::new(Int32Array::from(vec![row])) as _,
    )])
    .unwrap();
    SplitOutput {
        split_id: token.split_id,
        attempt_id: token.attempt_id,
        batches: vec![batch],
        metrics: SplitMetrics {
            output_rows: 1,
            ..Default::default()
        },
    }
}

fn assert_output(actual: &SplitOutput, expected: &SplitOutput) {
    assert_eq!(
        (actual.split_id, actual.attempt_id),
        (expected.split_id, expected.attempt_id)
    );
    assert_eq!(actual.metrics, expected.metrics);
    assert_eq!(actual.batches, expected.batches);
}

#[test]
fn lifecycle_renewal_and_idempotent_commit() {
    let config = LeaseConfig {
        lease_ttl: Duration::from_millis(5),
        ..Default::default()
    };
    let (queue, clock) = queue(&[19], config);
    assert_eq!(queue.state(19), Ok(SplitState::Queued));
    let token = claim(&queue, 4);
    assert_eq!(queue.state(19), Ok(SplitState::Leased));
    assert_eq!(
        queue.commit(&token, output(token, 8)),
        Err(LeaseError::InvalidState(SplitState::Leased))
    );
    for _ in 0..3 {
        clock.advance(config.lease_ttl);
        queue.heartbeat(&token).unwrap();
        assert_eq!(queue.state(19), Ok(SplitState::Leased));
    }
    clock.advance(config.lease_ttl);
    queue.start(&token).unwrap();
    assert_eq!(
        queue.start(&token),
        Err(LeaseError::InvalidState(SplitState::Running))
    );
    clock.advance(config.lease_ttl);
    assert_eq!(queue.state(19), Ok(SplitState::Running));
    assert_eq!(
        queue.commit(&token, output(token, 8)),
        Ok(CommitOutcome::Committed)
    );
    let published = queue.committed_outputs().unwrap();
    clock.advance(config.lease_ttl);
    assert_eq!(
        queue.commit(&token, output(token, 8)),
        Ok(CommitOutcome::AlreadyCommitted)
    );
    assert_eq!(
        queue.commit(&token, output(token, 9)),
        Err(LeaseError::ConflictingCommit)
    );
    let mut different_metrics = output(token, 8);
    different_metrics.metrics.submitted_prompts += 1;
    assert_eq!(
        queue.commit(&token, different_metrics),
        Err(LeaseError::ConflictingCommit)
    );
    for wrong_token in [
        LeaseToken {
            worker_id: 5,
            ..token
        },
        LeaseToken {
            attempt_id: 1,
            ..token
        },
        LeaseToken {
            fencing_epoch: 2,
            ..token
        },
    ] {
        assert_eq!(
            queue.commit(&wrong_token, output(token, 8)),
            Err(LeaseError::StaleAttempt)
        );
    }
    let outputs = queue.committed_outputs().unwrap();
    assert_eq!(outputs.len(), 1);
    assert!(Arc::ptr_eq(&outputs[0], &published[0]));
    assert_output(&outputs[0], &output(token, 8));
    assert!(queue.claim(8).unwrap().is_none());
    assert_eq!(
        queue
            .transitions(19)
            .unwrap()
            .iter()
            .map(|transition| (transition.from, transition.to))
            .collect::<Vec<_>>(),
        [
            (SplitState::Queued, SplitState::Leased),
            (SplitState::Leased, SplitState::Running),
            (SplitState::Running, SplitState::Committed),
        ]
    );
}

#[test]
fn hard_deadline_caps_start_and_heartbeat() {
    let config = LeaseConfig {
        lease_ttl: Duration::from_millis(5),
        max_attempt_duration: Some(Duration::from_millis(8)),
        max_attempts: 1,
        ..Default::default()
    };
    let (queue, clock) = queue(&[9], config);
    let token = claim(&queue, 4);
    clock.advance(Duration::from_millis(4));
    queue.start(&token).unwrap();
    clock.advance(Duration::from_millis(4));
    queue.heartbeat(&token).unwrap();
    assert_eq!(queue.expire_stale(), Ok(0));
    clock.advance(Duration::from_nanos(1));
    assert_eq!(
        queue.commit(&token, output(token, 8)),
        Err(LeaseError::LeaseExpired)
    );
    assert_eq!(queue.state(9), Ok(SplitState::Failed));
    assert!(queue.claim(4).unwrap().is_none());
    let audit = queue.transitions(9).unwrap();
    assert_eq!(
        (audit[2].from, audit[2].to),
        (SplitState::Running, SplitState::Expired)
    );
    assert_eq!(
        (audit[3].from, audit[3].to),
        (SplitState::Expired, SplitState::Failed)
    );
    assert_eq!(audit[2].at, audit[3].at);
}

#[test]
fn hard_deadline_shorter_than_ttl_caps_initial_lease_and_renewals() {
    let hard_limit = Duration::from_millis(2);
    let config = LeaseConfig {
        lease_ttl: Duration::from_millis(5),
        max_attempt_duration: Some(hard_limit),
        ..Default::default()
    };
    type Renewal = fn(&LeasedSplitQueue, &LeaseToken) -> Result<(), LeaseError>;
    let renewals: [Option<Renewal>; 3] = [
        None,
        Some(LeasedSplitQueue::start),
        Some(LeasedSplitQueue::heartbeat),
    ];
    for renew in renewals {
        let (queue, clock) = queue(&[2], config);
        clock.advance(Duration::from_millis(7));
        let token = claim(&queue, 0);
        clock.advance(hard_limit);
        assert_eq!(queue.expire_stale(), Ok(0));
        if let Some(renew) = renew {
            renew(&queue, &token).unwrap();
        }
        clock.advance(Duration::from_nanos(1));
        let replacement = claim(&queue, 1);
        assert_eq!(replacement.attempt_id, 1);
        assert_eq!(queue.start(&token), Err(LeaseError::StaleAttempt));
        let audit = queue.transitions(2).unwrap();
        assert_eq!(audit[audit.len() - 3].to, SplitState::Expired);
        assert_eq!(audit[audit.len() - 2].to, SplitState::Queued);
    }
}

#[test]
fn clock_and_lease_deadlines_saturate() {
    let config = LeaseConfig {
        max_attempt_duration: Some(Duration::from_secs(5)),
        ..Default::default()
    };
    let (queue, clock) = queue(&[2], config);
    clock.advance(Duration::MAX.checked_sub(Duration::from_secs(1)).unwrap());
    let token = claim(&queue, 0);
    clock.advance(Duration::from_secs(2));
    assert_eq!(clock.now(), Duration::MAX);
    assert_eq!(queue.expire_stale(), Ok(0));
    queue.start(&token).unwrap();
    queue.heartbeat(&token).unwrap();
    assert_eq!(
        queue.commit(&token, output(token, 8)),
        Ok(CommitOutcome::Committed)
    );
}

#[test]
fn every_entry_point_checks_expiry_before_changing_state() {
    let config = LeaseConfig {
        lease_ttl: Duration::from_millis(5),
        ..Default::default()
    };
    for operation in 0..4 {
        let (queue, clock) = queue(&[2], config);
        let token = claim(&queue, 0);
        if operation == 3 {
            queue.start(&token).unwrap();
        }
        clock.advance(config.lease_ttl + Duration::from_nanos(1));
        let error = match operation {
            0 => queue.start(&token),
            1 => queue.heartbeat(&token),
            2 => queue.report_failure(&token, false, "late"),
            _ => queue.commit(&token, output(token, 8)).map(|_| ()),
        };
        assert_eq!(error, Err(LeaseError::LeaseExpired));
        assert_eq!(queue.state(2), Ok(SplitState::Queued));
        assert_eq!(queue.expire_stale(), Ok(0));
        let audit = queue.transitions(2).unwrap();
        assert_eq!(audit[audit.len() - 2].to, SplitState::Expired);
        assert_eq!(
            (audit.last().unwrap().from, audit.last().unwrap().to),
            (SplitState::Expired, SplitState::Queued)
        );

        let replacement = claim(&queue, 1);
        assert_eq!(replacement.attempt_id, token.attempt_id + 1);
        assert!(replacement.fencing_epoch > token.fencing_epoch);
        assert_eq!(
            queue.commit(&token, output(token, 8)),
            Err(LeaseError::StaleAttempt)
        );
        assert_eq!(queue.heartbeat(&token), Err(LeaseError::StaleAttempt));
        assert_eq!(
            queue.report_failure(&token, false, "stale"),
            Err(LeaseError::StaleAttempt)
        );
        queue.start(&replacement).unwrap();
        assert_eq!(
            queue.commit(&replacement, output(replacement, 8)),
            Ok(CommitOutcome::Committed)
        );
    }
}

#[test]
fn failures_respect_token_and_retry_budget() {
    let config = LeaseConfig {
        max_attempts: 2,
        ..Default::default()
    };
    let (queue, _) = queue(&[2, 1], config);
    let first = claim(&queue, 0);
    let second = claim(&queue, 1);
    let wrong_worker = LeaseToken {
        worker_id: 2,
        ..first
    };
    let wrong_epoch = LeaseToken {
        fencing_epoch: second.fencing_epoch,
        ..first
    };
    let unknown = LeaseToken {
        split_id: 0,
        ..first
    };
    assert_eq!(queue.heartbeat(&wrong_worker), Err(LeaseError::WrongWorker));
    assert_eq!(
        queue.report_failure(&wrong_epoch, false, "wrong epoch"),
        Err(LeaseError::StaleAttempt)
    );
    assert_eq!(queue.start(&unknown), Err(LeaseError::UnknownSplit));
    queue.start(&first).unwrap();
    assert_eq!(
        queue.commit(&first, output(second, 1)),
        Err(LeaseError::ConflictingCommit)
    );
    assert_eq!(
        queue.commit(&wrong_worker, output(first, 1)),
        Err(LeaseError::WrongWorker)
    );
    queue.report_failure(&first, true, "transient").unwrap();
    queue.report_failure(&second, false, "permanent").unwrap();
    assert_eq!(queue.state(first.split_id), Ok(SplitState::Queued));
    assert_eq!(queue.state(second.split_id), Ok(SplitState::Failed));
    let transient = queue.transitions(first.split_id).unwrap();
    let requeued = transient.last().unwrap();
    assert_eq!((requeued.to, requeued.token), (SplitState::Queued, first));
    assert_eq!(requeued.failure_reason.as_deref(), Some("transient"));
    let retry = claim(&queue, 1);
    assert_eq!((retry.split_id, retry.attempt_id), (first.split_id, 1));
    queue.report_failure(&retry, true, "exhausted").unwrap();
    assert_eq!(queue.state(retry.split_id), Ok(SplitState::Failed));
    assert_eq!(queue.heartbeat(&retry), Err(LeaseError::AlreadyTerminal));
    assert!(queue.claim(0).unwrap().is_none());
    assert!(queue.committed_outputs().unwrap().is_empty());
    for (token, reason) in [(retry, "exhausted"), (second, "permanent")] {
        let audit = queue.transitions(token.split_id).unwrap();
        assert_eq!(
            audit.last().unwrap().failure_reason.as_deref(),
            Some(reason)
        );
    }
}

#[test]
fn sweep_preserves_deterministic_claim_order() {
    let config = LeaseConfig {
        lease_ttl: Duration::from_nanos(1),
        ..Default::default()
    };
    let (queue, clock) = queue(&[7, 5, 3, 1], config);
    assert_eq!(claim(&queue, 0).split_id, 7);
    assert_eq!(claim(&queue, 0).split_id, 5);
    assert_eq!(claim(&queue, 0).split_id, 3);
    clock.advance(Duration::from_nanos(2));
    assert_eq!(queue.expire_stale(), Ok(3));
    assert_eq!(queue.expire_stale(), Ok(0));
    for (id, attempt) in [(1, 0), (3, 1), (5, 1), (7, 1)] {
        let token = claim(&queue, 0);
        assert_eq!((token.split_id, token.attempt_id), (id, attempt));
    }
    assert!(queue.claim(0).unwrap().is_none());
}

#[test]
fn invalid_queue_configuration_is_rejected() {
    let split = Split {
        split_id: 0,
        attempt_id: 0,
        file_proto: vec![],
        fingerprint: String::new(),
    };
    let config = LeaseConfig::default();
    let clock = Arc::new(ManualClock::default());
    assert!(matches!(
        LeasedSplitQueue::new(vec![split.clone(); 2], config, clock.clone()),
        Err(DataFusionError::Plan(message)) if message.starts_with("Duplicate split ID 0")
    ));
    for (config, expected) in [
        (
            LeaseConfig {
                max_attempts: 0,
                ..config
            },
            "max_attempts must be positive",
        ),
        (
            LeaseConfig {
                lease_ttl: Duration::ZERO,
                ..config
            },
            "Lease TTL must be positive",
        ),
        (
            LeaseConfig {
                max_attempt_duration: Some(Duration::ZERO),
                ..config
            },
            "max_attempt_duration must be positive when set",
        ),
    ] {
        assert!(matches!(
            LeasedSplitQueue::new([split.clone()], config, clock.clone()),
            Err(DataFusionError::Plan(message)) if message.starts_with(expected)
        ));
    }
}

#[derive(Clone, Copy, Debug)]
enum Operation {
    Claim(u64),
    Start,
    Heartbeat,
    AdvanceClock(u64),
    ExpireStale,
    Commit(i32),
    Replay(bool),
    Fail(bool),
}

fn operations() -> impl Strategy<Value = (usize, bool, Operation)> {
    (
        0..3usize,
        any::<bool>(),
        prop_oneof![
            3 => (0..3u64).prop_map(Operation::Claim),
            3 => Just(Operation::Start),
            2 => Just(Operation::Heartbeat),
            3 => (0..12u64).prop_map(Operation::AdvanceClock),
            1 => Just(Operation::ExpireStale),
            3 => (-2..3i32).prop_map(Operation::Commit),
            2 => any::<bool>().prop_map(Operation::Replay),
            2 => any::<bool>().prop_map(Operation::Fail),
        ],
    )
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
enum Phase {
    #[default]
    Waiting,
    Active {
        deadline: u64,
        hard: u64,
        started: bool,
    },
    Published(i32),
    Failed,
}

#[derive(Debug, Default)]
struct ExpectedSplit {
    tokens: Vec<LeaseToken>,
    phase: Phase,
    expired: bool,
}

impl ExpectedSplit {
    fn expect_active(&self, token: LeaseToken) -> Result<(), LeaseError> {
        if self.tokens.last() != Some(&token) {
            return Err(LeaseError::StaleAttempt);
        }
        if self.expired {
            return Err(LeaseError::LeaseExpired);
        }
        match self.phase {
            Phase::Waiting => Err(LeaseError::StaleAttempt),
            Phase::Published(_) | Phase::Failed => Err(LeaseError::AlreadyTerminal),
            Phase::Active { .. } => Ok(()),
        }
    }
}

const IDS: [u64; 3] = [2, 0, 1];
const TTL: u64 = 5;
const ATTEMPTS: u32 = 3;

struct Model {
    now: u64,
    hard_duration: u64,
    epoch: u64,
    ready: VecDeque<u64>,
    splits: BTreeMap<u64, ExpectedSplit>,
}

impl Model {
    fn new(hard_duration: u64) -> Self {
        Self {
            now: 0,
            hard_duration,
            epoch: 0,
            ready: IDS.into(),
            splits: IDS
                .into_iter()
                .map(|id| (id, ExpectedSplit::default()))
                .collect(),
        }
    }

    fn fail(&mut self, id: u64, retryable: bool, expired: bool) {
        let split = self.splits.get_mut(&id).unwrap();
        split.expired = expired;
        split.phase = if retryable && split.tokens.len() < ATTEMPTS as usize {
            self.ready.push_back(id);
            Phase::Waiting
        } else {
            Phase::Failed
        };
    }

    fn expire(&mut self, id: u64) -> bool {
        if matches!(self.splits[&id].phase, Phase::Active { deadline, .. } if deadline < self.now)
        {
            self.fail(id, true, true);
            return true;
        }
        false
    }

    fn sweep(&mut self) -> usize {
        (0..IDS.len() as u64).filter(|&id| self.expire(id)).count()
    }

    fn claim(&mut self, queue: &LeasedSplitQueue, worker: u64) {
        self.sweep();
        let actual = queue.claim(worker).unwrap();
        let Some(id) = self.ready.pop_front() else {
            assert!(actual.is_none());
            return;
        };
        let (split, token) = actual.unwrap();
        let expected = self.splits.get_mut(&id).unwrap();
        self.epoch += 1;
        assert_eq!(
            token,
            LeaseToken {
                split_id: id,
                attempt_id: expected.tokens.len() as u32,
                worker_id: worker,
                fencing_epoch: self.epoch,
            }
        );
        assert_eq!(
            (split.split_id, split.attempt_id),
            (token.split_id, token.attempt_id)
        );
        assert_eq!(expected.phase, Phase::Waiting);
        expected.tokens.push(token);
        expected.expired = false;
        expected.phase = Phase::Active {
            deadline: self.now + TTL.min(self.hard_duration),
            hard: self.now + self.hard_duration,
            started: false,
        };
    }

    fn renew(&mut self, token: LeaseToken, start: bool) -> Result<(), LeaseError> {
        let split = self.splits.get_mut(&token.split_id).unwrap();
        split.expect_active(token)?;
        let Phase::Active {
            deadline,
            hard,
            started,
        } = &mut split.phase
        else {
            unreachable!()
        };
        if start && *started {
            return Err(LeaseError::InvalidState(SplitState::Running));
        }
        *started |= start;
        *deadline = (self.now + TTL).min(*hard);
        Ok(())
    }

    fn commit(
        &mut self,
        token: LeaseToken,
        row: i32,
    ) -> Result<CommitOutcome, LeaseError> {
        let split = self.splits.get_mut(&token.split_id).unwrap();
        if let Phase::Published(committed) = split.phase {
            return if split.tokens.last() != Some(&token) {
                Err(LeaseError::StaleAttempt)
            } else if row == committed {
                Ok(CommitOutcome::AlreadyCommitted)
            } else {
                Err(LeaseError::ConflictingCommit)
            };
        }
        split.expect_active(token)?;
        if matches!(split.phase, Phase::Active { started: false, .. }) {
            return Err(LeaseError::InvalidState(SplitState::Leased));
        }
        split.phase = Phase::Published(row);
        Ok(CommitOutcome::Committed)
    }

    fn apply(
        &mut self,
        queue: &LeasedSplitQueue,
        clock: &ManualClock,
        id: u64,
        old: bool,
        operation: Operation,
    ) {
        match operation {
            Operation::Claim(worker) => return self.claim(queue, worker),
            Operation::AdvanceClock(ms) => {
                clock.advance(Duration::from_millis(ms));
                self.now += ms;
                return;
            }
            Operation::ExpireStale => {
                assert_eq!(queue.expire_stale(), Ok(self.sweep()));
                return;
            }
            _ => {}
        }

        let split = &self.splits[&id];
        let token = if old {
            split.tokens.first()
        } else {
            split.tokens.last()
        };
        let Some(&token) = token else {
            return;
        };
        let operation = match (operation, split.phase) {
            (Operation::Replay(conflict), Phase::Published(row)) => {
                Operation::Commit(row + i32::from(conflict))
            }
            (Operation::Replay(_), _) => return,
            _ => operation,
        };
        self.expire(id);
        match operation {
            Operation::Start => assert_eq!(queue.start(&token), self.renew(token, true)),
            Operation::Heartbeat => {
                assert_eq!(queue.heartbeat(&token), self.renew(token, false))
            }
            Operation::Commit(row) => assert_eq!(
                queue.commit(&token, output(token, row)),
                self.commit(token, row)
            ),
            Operation::Fail(retryable) => {
                let expected = self.splits[&id].expect_active(token);
                assert_eq!(
                    queue.report_failure(&token, retryable, "generated failure"),
                    expected
                );
                if expected.is_ok() {
                    self.fail(id, retryable, false);
                }
            }
            _ => unreachable!(),
        }
    }

    fn assert_snapshot(&self, queue: &LeasedSplitQueue) {
        let outputs = queue.committed_outputs().unwrap();
        let mut expected_outputs = Vec::new();
        for (&id, split) in &self.splits {
            let expected_state = match split.phase {
                Phase::Waiting => SplitState::Queued,
                Phase::Active { started: false, .. } => SplitState::Leased,
                Phase::Active { started: true, .. } => SplitState::Running,
                Phase::Published(row) => {
                    expected_outputs.push(output(*split.tokens.last().unwrap(), row));
                    SplitState::Committed
                }
                Phase::Failed => SplitState::Failed,
            };
            assert_eq!(queue.state(id), Ok(expected_state));
            let mut state = SplitState::Queued;
            let mut active = None;
            let mut attempts = Vec::new();
            for transition in queue.transitions(id).unwrap() {
                assert_eq!(state, transition.from);
                assert!(!matches!(state, SplitState::Committed | SplitState::Failed));
                if transition.to == SplitState::Leased {
                    assert_eq!(state, SplitState::Queued);
                    assert!(active.replace(transition.token).is_none());
                    attempts.push(transition.token);
                } else if state != SplitState::Expired {
                    assert_eq!(active, Some(transition.token));
                    if transition.to != SplitState::Running {
                        active = None;
                    }
                }
                state = transition.to;
            }
            assert_eq!(state, expected_state);
            assert_eq!(attempts, split.tokens);
            assert_eq!(
                active.is_some(),
                matches!(split.phase, Phase::Active { .. })
            );
        }
        assert_eq!(outputs.len(), expected_outputs.len());
        for (actual, expected) in outputs.iter().zip(expected_outputs) {
            assert_output(actual, &expected);
        }
    }
}

fn proptest_config() -> Config {
    let seed = env::var("DIST_LEASE_PROPTEST_SEED")
        .ok()
        .map_or(0xc1b, |value| {
            value
                .parse()
                .expect("DIST_LEASE_PROPTEST_SEED must be a u64")
        });
    Config {
        cases: 256,
        rng_seed: RngSeed::Fixed(seed),
        failure_persistence: None,
        ..Config::default()
    }
}

proptest! {
    #![proptest_config(proptest_config())]

    #[test]
    fn random_operations_preserve_attempt_fencing(
        hard_duration in 2..16u64,
        sequence in prop::collection::vec(operations(), 1..100),
    ) {
        let config = LeaseConfig {
            lease_ttl: Duration::from_millis(TTL),
            max_attempt_duration: Some(Duration::from_millis(hard_duration)),
            max_attempts: ATTEMPTS,
            ..Default::default()
        };
        let (queue, clock) = queue(&IDS, config);
        let mut model = Model::new(hard_duration);
        for (id, old, operation) in sequence {
            model.apply(&queue, &clock, id as u64, old, operation);
            model.assert_snapshot(&queue);
        }

        // Complete eligible work, including retries left queued by the generator.
        model.apply(&queue, &clock, 0, false, Operation::ExpireStale);
        model.assert_snapshot(&queue);
        while let Some(&id) = model.ready.front() {
            model.claim(&queue, 0);
            model.assert_snapshot(&queue);
            model.apply(&queue, &clock, id, false, Operation::Start);
            model.assert_snapshot(&queue);
            model.apply(&queue, &clock, id, false, Operation::Commit(7));
            model.assert_snapshot(&queue);
            model.apply(&queue, &clock, id, false, Operation::Replay(false));
            model.assert_snapshot(&queue);
        }
    }
}
