// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: See the NOTICE file distributed with this work for additional information regarding copyright ownership
// SPDX-FileContributor: Licensed to the Apache Software Foundation (ASF) under one or more contributor license agreements

//! In-process split scheduling with lease deadlines and attempt fencing.

use std::collections::{BTreeMap, BTreeSet, VecDeque};
use std::fmt::{self, Display};
use std::sync::{Arc, Mutex, MutexGuard};
use std::time::Duration;

use datafusion::common::instant::Instant;
use datafusion::common::{DataFusionError, Result, plan_err};
use datafusion::physical_plan::ExecutionPlan;

use crate::codec::DistExtensionRegistry;
use crate::{Split, SplitOutput};

/// Time elapsed from a clock's own origin.
pub type MonotonicTime = Duration;

/// Supplies nondecreasing timestamps from a single origin.
///
/// The queue reads the clock with its state lock held. Implementations must
/// avoid calling back into the queue.
pub trait MonotonicClock: fmt::Debug + Send + Sync {
    /// Read the current timestamp.
    fn now(&self) -> MonotonicTime;
}

/// A monotonic clock with an origin set when it is created.
#[derive(Debug)]
pub struct SystemClock(Instant);

impl Default for SystemClock {
    fn default() -> Self {
        Self(Instant::now())
    }
}

impl MonotonicClock for SystemClock {
    fn now(&self) -> MonotonicTime {
        self.0.elapsed()
    }
}

/// A thread-safe clock that starts at zero and advances only on request.
#[derive(Debug, Default)]
pub struct ManualClock(Mutex<MonotonicTime>);

impl ManualClock {
    /// Move time forward, saturating at the largest representable timestamp.
    pub fn advance(&self, duration: Duration) {
        let mut now = self.0.lock().unwrap();
        *now = now.saturating_add(duration);
    }
}

impl MonotonicClock for ManualClock {
    fn now(&self) -> MonotonicTime {
        *self.0.lock().unwrap()
    }
}

/// Lease and retry policy for a queue.
#[derive(Clone, Copy, Debug)]
pub struct LeaseConfig {
    /// Maximum interval from claim, start or heartbeat to the lease deadline.
    pub lease_ttl: Duration,
    /// Optional limit measured from claim, independent of lease renewal.
    pub max_attempt_duration: Option<Duration>,
    /// Total claims allowed per split, including its first attempt.
    pub max_attempts: u32,
    /// Require a plan whose operators allow speculative execution.
    /// The queue issues one active lease per split.
    pub speculation_allowed: bool,
}

impl Default for LeaseConfig {
    fn default() -> Self {
        Self {
            lease_ttl: Duration::from_secs(30),
            max_attempt_duration: None,
            max_attempts: 3,
            speculation_allowed: false,
        }
    }
}

impl LeaseConfig {
    /// Require positive durations and a positive attempt budget.
    ///
    /// A `max_attempt_duration` shorter than `lease_ttl` is valid. When set,
    /// the deadline at claim, start and heartbeat is
    /// `min(now.saturating_add(lease_ttl), claim_at.saturating_add(max_attempt_duration))`.
    pub fn validate(&self) -> Result<()> {
        if self.lease_ttl.is_zero() {
            return plan_err!("Lease TTL must be positive");
        }
        if self.max_attempts == 0 {
            return plan_err!("max_attempts must be positive");
        }
        if self.max_attempt_duration == Some(Duration::ZERO) {
            return plan_err!("max_attempt_duration must be positive when set");
        }
        Ok(())
    }

    /// Validate the configuration and check extension policies across the plan.
    /// Ambiguous policies are errors even when speculation is disabled.
    pub fn validate_for_plan(
        &self,
        plan: &dyn ExecutionPlan,
        registry: &DistExtensionRegistry,
    ) -> Result<()> {
        self.validate()?;
        let allows = registry.plan_allows_speculation(plan)?;
        if self.speculation_allowed && !allows {
            return plan_err!(
                "Speculation is not allowed for plans containing non-speculatable extension operators"
            );
        }
        Ok(())
    }
}

/// The scheduler-issued identity of one claim within a queue.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct LeaseToken {
    /// Split assigned to the worker.
    pub split_id: u64,
    /// Zero-based attempt number within the split.
    pub attempt_id: u32,
    /// Local identifier supplied by the claiming worker.
    pub worker_id: u64,
    /// Queue-wide sequence number of the claim.
    pub fencing_epoch: u64,
}

/// Lifecycle state of a split.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum SplitState {
    /// Available to claim.
    Queued,
    /// Claimed and awaiting start.
    Leased,
    /// Started and eligible to commit.
    Running,
    /// Lease elapsed. Recorded before requeueing or failing the split.
    Expired,
    /// Output accepted.
    Committed,
    /// No further attempts are allowed.
    Failed,
}

/// An audited state transition attributed to a specific claim.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct SplitTransition {
    /// State before the transition.
    pub from: SplitState,
    /// State after the transition.
    pub to: SplitState,
    /// Attempt responsible for the transition.
    pub token: LeaseToken,
    /// Time when the queue recorded the transition.
    pub at: MonotonicTime,
    /// Worker-provided reason for a reported failure.
    pub failure_reason: Option<String>,
}

/// Result of accepting an attempt's output.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum CommitOutcome {
    /// The queue published the output.
    Committed,
    /// The same claim already published identical output.
    AlreadyCommitted,
}

/// A rejected lease operation.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum LeaseError {
    /// The split is absent from the queue.
    UnknownSplit,
    /// The attempt or fencing epoch differs from the current claim.
    StaleAttempt,
    /// An active claim belongs to another worker.
    WrongWorker,
    /// The matching attempt's deadline elapsed.
    LeaseExpired,
    /// A split differs from its claim, or the output conflicts with the token or a prior commit.
    ConflictingCommit,
    /// The split has already reached a terminal state.
    AlreadyTerminal,
    /// The requested operation requires a different split state.
    InvalidState(SplitState),
    /// The queue cannot allocate a unique fencing epoch.
    FencingEpochExhausted,
    /// A previous panic poisoned the queue's state lock.
    Poisoned,
}

impl Display for LeaseError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let message = match self {
            Self::UnknownSplit => "Unknown split",
            Self::StaleAttempt => "Stale split attempt or fencing epoch",
            Self::WrongWorker => "Split lease belongs to another worker",
            Self::LeaseExpired => "Split lease expired",
            Self::ConflictingCommit => "Conflicting split or output",
            Self::AlreadyTerminal => "Split is already terminal",
            Self::InvalidState(state) => {
                return write!(f, "Invalid split state for lease operation: {state:?}");
            }
            Self::FencingEpochExhausted => "Split lease fencing epochs exhausted",
            Self::Poisoned => "Split queue mutex poisoned",
        };
        f.write_str(message)
    }
}

impl std::error::Error for LeaseError {}

impl From<LeaseError> for DataFusionError {
    fn from(error: LeaseError) -> Self {
        Self::External(Box::new(error))
    }
}

#[derive(Debug)]
struct ActiveLease {
    token: LeaseToken,
    lease_deadline: MonotonicTime,
    hard_deadline: Option<MonotonicTime>,
}

impl ActiveLease {
    fn renew(&mut self, now: MonotonicTime, ttl: Duration) {
        let deadline = now.saturating_add(ttl);
        self.lease_deadline = self
            .hard_deadline
            .map_or(deadline, |hard| deadline.min(hard));
    }
}

#[derive(Debug)]
struct Entry {
    split: Split,
    state: SplitState,
    next_attempt_id: u32,
    active_lease: Option<ActiveLease>,
    output: Option<Arc<SplitOutput>>,
    transitions: Vec<SplitTransition>,
}

impl Entry {
    fn transition(
        &mut self,
        to: SplitState,
        token: LeaseToken,
        at: MonotonicTime,
        failure_reason: Option<String>,
    ) {
        self.transitions.push(SplitTransition {
            from: self.state,
            to,
            token,
            at,
            failure_reason,
        });
        self.state = to;
    }

    fn expire(&mut self, now: MonotonicTime, max_attempts: u32) -> bool {
        let Some(lease) = &self.active_lease else {
            return false;
        };
        if now <= lease.lease_deadline {
            return false;
        }

        // Renewal caps lease_deadline at hard_deadline from the initial claim.
        let token = lease.token;
        self.transition(SplitState::Expired, token, now, None);
        self.finish_failure(token, now, true, max_attempts, None);
        true
    }

    fn finish_failure(
        &mut self,
        token: LeaseToken,
        now: MonotonicTime,
        retryable: bool,
        max_attempts: u32,
        reason: Option<String>,
    ) {
        self.active_lease = None;
        let next = if retryable && self.next_attempt_id < max_attempts {
            SplitState::Queued
        } else {
            SplitState::Failed
        };
        self.transition(next, token, now, reason);
    }

    fn validate_active(&self, token: &LeaseToken) -> Result<(), LeaseError> {
        let Some(last) = self.transitions.last() else {
            return Err(LeaseError::StaleAttempt);
        };
        let current = self
            .active_lease
            .as_ref()
            .map_or(&last.token, |lease| &lease.token);
        if current.attempt_id != token.attempt_id
            || current.fencing_epoch != token.fencing_epoch
        {
            return Err(LeaseError::StaleAttempt);
        }
        if current.worker_id != token.worker_id {
            return Err(LeaseError::WrongWorker);
        }
        if last.from == SplitState::Expired {
            return Err(LeaseError::LeaseExpired);
        }
        match self.state {
            SplitState::Leased | SplitState::Running => Ok(()),
            SplitState::Committed | SplitState::Failed => {
                Err(LeaseError::AlreadyTerminal)
            }
            SplitState::Expired => Err(LeaseError::LeaseExpired),
            SplitState::Queued => Err(LeaseError::StaleAttempt),
        }
    }
}

#[derive(Debug, Default)]
struct QueueState {
    pending: VecDeque<u64>,
    active: BTreeSet<u64>,
    entries: BTreeMap<u64, Entry>,
    fencing_epoch: u64,
}

impl QueueState {
    fn expire_stale(&mut self, now: MonotonicTime, max_attempts: u32) -> usize {
        let active_count = self.active.len();
        self.active.retain(|id| {
            let entry = self.entries.get_mut(id).unwrap();
            if !entry.expire(now, max_attempts) {
                return true;
            }
            if entry.state == SplitState::Queued {
                self.pending.push_back(*id);
            }
            false
        });
        active_count - self.active.len()
    }

    fn target(
        &mut self,
        token: &LeaseToken,
        now: MonotonicTime,
        max_attempts: u32,
    ) -> Result<&mut Entry, LeaseError> {
        let entry = self
            .entries
            .get_mut(&token.split_id)
            .ok_or(LeaseError::UnknownSplit)?;
        if entry.expire(now, max_attempts) {
            self.active.remove(&token.split_id);
            if entry.state == SplitState::Queued {
                self.pending.push_back(token.split_id);
            }
        }
        Ok(entry)
    }
}

/// A finite in-process queue with one active attempt per split.
///
/// Initial claims follow insertion order. Retries join the tail, with splits
/// recovered by the same sweep appended in split ID order. Deadlines are
/// inclusive. A lease expires when the clock moves past its deadline.
///
/// ~~~
/// use std::sync::Arc;
/// use datafusion_dist::{LeaseConfig, LeasedSplitQueue, ManualClock, Split};
///
/// let queue = LeasedSplitQueue::new(
///     [Split {
///         split_id: 0,
///         attempt_id: 0,
///         file_proto: vec![],
///         fingerprint: String::new(),
///     }],
///     LeaseConfig::default(),
///     Arc::new(ManualClock::default()),
/// )?;
/// let (split, token) = queue.claim(42)?.unwrap();
/// queue.start(&token)?;
/// assert_eq!(split.attempt_id, token.attempt_id);
/// # Ok::<(), datafusion::common::DataFusionError>(())
/// ~~~
#[derive(Debug)]
pub struct LeasedSplitQueue {
    config: LeaseConfig,
    clock: Arc<dyn MonotonicClock>,
    state: Mutex<QueueState>,
}

impl LeasedSplitQueue {
    /// Create a queue for a stage. The queue assigns attempt IDs starting at zero.
    ///
    /// Call [`LeaseConfig::validate_for_plan`] when configuring a physical plan.
    pub fn new(
        splits: impl IntoIterator<Item = Split>,
        config: LeaseConfig,
        clock: Arc<dyn MonotonicClock>,
    ) -> Result<Self> {
        config.validate()?;
        let mut state = QueueState::default();
        for split in splits {
            state.pending.push_back(split.split_id);
            let previous = state.entries.insert(
                split.split_id,
                Entry {
                    split,
                    state: SplitState::Queued,
                    next_attempt_id: 0,
                    active_lease: None,
                    output: None,
                    transitions: Vec::new(),
                },
            );
            if let Some(entry) = previous {
                return plan_err!("Duplicate split ID {}", entry.split.split_id);
            }
        }
        Ok(Self {
            config,
            clock,
            state: Mutex::new(state),
        })
    }

    fn lock(&self) -> Result<MutexGuard<'_, QueueState>, LeaseError> {
        self.state.lock().map_err(|_| LeaseError::Poisoned)
    }

    pub(crate) fn validate_plan(
        &self,
        plan: &dyn ExecutionPlan,
        registry: &DistExtensionRegistry,
    ) -> Result<()> {
        self.config.validate_for_plan(plan, registry)
    }

    pub(crate) fn verify_leased_split(
        &self,
        split: &Split,
        token: &LeaseToken,
    ) -> Result<(), LeaseError> {
        let mut state = self.lock()?;
        let entry = state.target(token, self.clock.now(), self.config.max_attempts)?;
        entry.validate_active(token)?;
        if split.fingerprint != entry.split.fingerprint
            || split.file_proto != entry.split.file_proto
        {
            return Err(LeaseError::ConflictingCommit);
        }
        Ok(())
    }

    /// Recover expired leases and claim the next available split for this worker.
    pub fn claim(
        &self,
        worker_id: u64,
    ) -> Result<Option<(Split, LeaseToken)>, LeaseError> {
        let mut state = self.lock()?;
        let now = self.clock.now();
        state.expire_stale(now, self.config.max_attempts);
        let Some(&split_id) = state.pending.front() else {
            return Ok(None);
        };
        let fencing_epoch = state
            .fencing_epoch
            .checked_add(1)
            .ok_or(LeaseError::FencingEpochExhausted)?;
        state.fencing_epoch = fencing_epoch;
        state.pending.pop_front();
        state.active.insert(split_id);
        let entry = state.entries.get_mut(&split_id).unwrap();
        let token = LeaseToken {
            split_id,
            attempt_id: entry.next_attempt_id,
            worker_id,
            fencing_epoch,
        };
        // Queued entries have issued fewer claims than max_attempts.
        entry.next_attempt_id += 1;
        let mut lease = ActiveLease {
            token,
            lease_deadline: now,
            hard_deadline: self
                .config
                .max_attempt_duration
                .map(|duration| now.saturating_add(duration)),
        };
        lease.renew(now, self.config.lease_ttl);
        entry.active_lease = Some(lease);
        entry.transition(SplitState::Leased, token, now, None);
        let mut split = entry.split.clone();
        split.attempt_id = token.attempt_id;
        Ok(Some((split, token)))
    }

    /// Start an unexpired claim and renew its lease.
    pub fn start(&self, token: &LeaseToken) -> Result<(), LeaseError> {
        self.renew(token, true)
    }

    /// Renew an unexpired claim, capped at its hard deadline.
    pub fn heartbeat(&self, token: &LeaseToken) -> Result<(), LeaseError> {
        self.renew(token, false)
    }

    fn renew(&self, token: &LeaseToken, start: bool) -> Result<(), LeaseError> {
        let mut state = self.lock()?;
        let now = self.clock.now();
        let entry = state.target(token, now, self.config.max_attempts)?;
        entry.validate_active(token)?;
        if start {
            if entry.state != SplitState::Leased {
                return Err(LeaseError::InvalidState(entry.state));
            }
            entry.transition(SplitState::Running, *token, now, None);
        }
        entry
            .active_lease
            .as_mut()
            .unwrap()
            .renew(now, self.config.lease_ttl);
        Ok(())
    }

    /// Recover all expired leases, returning the number of attempts expired.
    pub fn expire_stale(&self) -> Result<usize, LeaseError> {
        let mut state = self.lock()?;
        Ok(state.expire_stale(self.clock.now(), self.config.max_attempts))
    }

    /// Report failure of a valid claim, with optional retry subject to the attempt limit.
    pub fn report_failure(
        &self,
        token: &LeaseToken,
        retryable: bool,
        reason: impl Into<String>,
    ) -> Result<(), LeaseError> {
        let reason = reason.into();
        let mut state = self.lock()?;
        let now = self.clock.now();
        let entry = state.target(token, now, self.config.max_attempts)?;
        entry.validate_active(token)?;
        entry.finish_failure(
            *token,
            now,
            retryable,
            self.config.max_attempts,
            Some(reason),
        );
        if entry.state == SplitState::Queued {
            state.pending.push_back(token.split_id);
        }
        state.active.remove(&token.split_id);
        Ok(())
    }

    /// Publish a running claim's output exactly once.
    ///
    /// An identical replay using the original token succeeds even after the
    /// original deadline because the output was already accepted.
    pub fn commit(
        &self,
        token: &LeaseToken,
        output: SplitOutput,
    ) -> Result<CommitOutcome, LeaseError> {
        let mut state = self.lock()?;
        let now = self.clock.now();
        let entry = state.target(token, now, self.config.max_attempts)?;
        if let Some(committed) = &entry.output {
            if entry.transitions.last().map(|transition| &transition.token) != Some(token)
            {
                return Err(LeaseError::StaleAttempt);
            }
            return if committed.split_id == output.split_id
                && committed.attempt_id == output.attempt_id
                && committed.metrics == output.metrics
                && committed.batches == output.batches
            {
                Ok(CommitOutcome::AlreadyCommitted)
            } else {
                Err(LeaseError::ConflictingCommit)
            };
        }

        entry.validate_active(token)?;
        if entry.state != SplitState::Running {
            return Err(LeaseError::InvalidState(entry.state));
        }
        if (output.split_id, output.attempt_id) != (token.split_id, token.attempt_id) {
            return Err(LeaseError::ConflictingCommit);
        }
        entry.output = Some(Arc::new(output));
        entry.active_lease = None;
        entry.transition(SplitState::Committed, *token, now, None);
        state.active.remove(&token.split_id);
        Ok(CommitOutcome::Committed)
    }

    /// Read the recorded state, without advancing expiration processing.
    pub fn state(&self, split_id: u64) -> Result<SplitState, LeaseError> {
        let state = self.lock()?;
        Ok(state
            .entries
            .get(&split_id)
            .ok_or(LeaseError::UnknownSplit)?
            .state)
    }

    /// Return the recorded transitions in order, including expiration and recovery.
    pub fn transitions(&self, split_id: u64) -> Result<Vec<SplitTransition>, LeaseError> {
        let state = self.lock()?;
        Ok(state
            .entries
            .get(&split_id)
            .ok_or(LeaseError::UnknownSplit)?
            .transitions
            .clone())
    }

    /// Snapshot accepted outputs in split ID order, with one entry per committed split.
    pub fn committed_outputs(&self) -> Result<Vec<Arc<SplitOutput>>, LeaseError> {
        let state = self.lock()?;
        Ok(state
            .entries
            .values()
            .filter_map(|entry| entry.output.clone())
            .collect())
    }
}
