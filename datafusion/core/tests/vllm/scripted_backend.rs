// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: See the NOTICE file distributed with this work for additional information regarding copyright ownership
// SPDX-FileContributor: Licensed to the Apache Software Foundation (ASF) under one or more contributor license agreements

//! Step 2 can adapt its async backend to call [`ScriptedBackend::submit`] once
//! per eligible input row and [`Request::attempt`] on each transport attempt.
//! Withhold completions to measure backpressure, release them out of order to
//! check row alignment, drop the execution stream to check cancellation and
//! released slots, or reply with `RateLimited` to drive the retry policy.

use std::sync::{Arc, Mutex};
use std::time::Duration;

use futures::poll;
use tokio::sync::{mpsc, oneshot};

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub(super) struct Counts {
    pub(super) logical_rows_submitted: usize,
    pub(super) backend_attempts: usize,
    pub(super) in_flight: usize,
    pub(super) peak_in_flight: usize,
    pub(super) cancelled_attempts: usize,
}

#[derive(Debug, PartialEq, Eq)]
pub(super) enum Failure {
    RateLimited,
    ControllerDropped,
}

pub(super) type Reply = Result<String, Failure>;

#[derive(Clone)]
pub(super) struct ScriptedBackend {
    counts: Arc<Mutex<Counts>>,
    started: mpsc::UnboundedSender<PendingAttempt>,
}

pub(super) struct InferenceProbe {
    counts: Arc<Mutex<Counts>>,
    started: mpsc::UnboundedReceiver<PendingAttempt>,
}

pub(super) struct Request {
    backend: ScriptedBackend,
    id: i64,
    prompt: String,
}

pub(super) struct PendingAttempt {
    pub(super) id: i64,
    pub(super) prompt: String,
    pub(super) reply: oneshot::Sender<Reply>,
}

struct InFlight<'a> {
    counts: &'a Mutex<Counts>,
    completed: bool,
}

impl Drop for InFlight<'_> {
    fn drop(&mut self) {
        let mut counts = self.counts.lock().unwrap();
        counts.in_flight -= 1;
        counts.cancelled_attempts += usize::from(!self.completed);
    }
}

impl ScriptedBackend {
    pub(super) fn new() -> (Self, InferenceProbe) {
        let (sender, receiver) = mpsc::unbounded_channel();
        let counts = Arc::new(Mutex::new(Counts::default()));
        (
            Self {
                counts: Arc::clone(&counts),
                started: sender,
            },
            InferenceProbe {
                counts,
                started: receiver,
            },
        )
    }

    pub(super) fn submit(&self, id: i64, prompt: String) -> Request {
        self.counts.lock().unwrap().logical_rows_submitted += 1;
        Request {
            backend: self.clone(),
            id,
            prompt,
        }
    }
}

impl Request {
    /// The future registers an attempt on its first poll and holds an in-flight
    /// slot until the probe replies or the caller drops the future.
    pub(super) async fn attempt(&self) -> Reply {
        let counts = &self.backend.counts;
        {
            let mut counts = counts.lock().unwrap();
            counts.backend_attempts += 1;
            counts.in_flight += 1;
            counts.peak_in_flight = counts.peak_in_flight.max(counts.in_flight);
        }
        let mut in_flight = InFlight {
            counts,
            completed: false,
        };
        let (reply, wait) = oneshot::channel();
        let pending = PendingAttempt {
            id: self.id,
            prompt: self.prompt.clone(),
            reply,
        };
        let result = if self.backend.started.send(pending).is_ok() {
            wait.await.unwrap_or(Err(Failure::ControllerDropped))
        } else {
            Err(Failure::ControllerDropped)
        };
        in_flight.completed = true;
        result
    }
}

impl InferenceProbe {
    pub(super) fn counts(&self) -> Counts {
        *self.counts.lock().unwrap()
    }

    pub(super) async fn next_attempt(&mut self) -> PendingAttempt {
        tokio::time::timeout(Duration::from_secs(5), self.started.recv())
            .await
            .expect("Timed out waiting for an inference attempt")
            .expect("Scripted backend was dropped")
    }
}

#[tokio::test]
async fn scripted_backend_gates_and_completes_attempts_out_of_order() {
    let (backend, mut probe) = ScriptedBackend::new();
    let first = backend.submit(1, "first".into());
    let mut first_attempt = Box::pin(first.attempt());
    assert!(poll!(&mut first_attempt).is_pending());
    let first_gate = probe.next_attempt().await;
    assert_eq!((first_gate.id, first_gate.prompt.as_str()), (1, "first"));

    let second = backend.submit(2, "second".into());
    let mut second_attempt = Box::pin(second.attempt());
    assert!(poll!(&mut second_attempt).is_pending());
    let second_gate = probe.next_attempt().await;
    assert_eq!((second_gate.id, second_gate.prompt.as_str()), (2, "second"));
    assert_eq!(
        probe.counts(),
        Counts {
            logical_rows_submitted: 2,
            backend_attempts: 2,
            in_flight: 2,
            peak_in_flight: 2,
            cancelled_attempts: 0,
        }
    );

    second_gate.reply.send(Ok("second result".into())).unwrap();
    assert_eq!(second_attempt.await, Ok("second result".into()));
    assert!(poll!(&mut first_attempt).is_pending());
    assert_eq!(probe.counts().in_flight, 1);

    first_gate.reply.send(Ok("first result".into())).unwrap();
    assert_eq!(first_attempt.await, Ok("first result".into()));
    assert_eq!(probe.counts().in_flight, 0);
    assert_eq!(probe.counts().peak_in_flight, 2);
}

#[tokio::test]
async fn scripted_backend_records_cancellation_and_releases_the_slot() {
    let (backend, mut probe) = ScriptedBackend::new();
    let request = backend.submit(1, "pending".into());
    let mut attempt = Box::pin(request.attempt());
    assert!(poll!(&mut attempt).is_pending());
    let gate = probe.next_attempt().await;
    assert_eq!(probe.counts().in_flight, 1);

    drop(attempt);
    assert!(gate.reply.is_closed());
    assert_eq!(
        probe.counts(),
        Counts {
            logical_rows_submitted: 1,
            backend_attempts: 1,
            in_flight: 0,
            peak_in_flight: 1,
            cancelled_attempts: 1,
        }
    );
}

#[tokio::test]
async fn scripted_backend_counts_retries_as_attempts_of_one_submission() {
    let (backend, mut probe) = ScriptedBackend::new();
    let request = backend.submit(7, "retry".into());
    let mut first_attempt = Box::pin(request.attempt());
    assert!(poll!(&mut first_attempt).is_pending());
    let first = probe.next_attempt().await;
    first.reply.send(Err(Failure::RateLimited)).unwrap();
    assert_eq!(first_attempt.await, Err(Failure::RateLimited));
    assert_eq!(probe.counts().in_flight, 0);

    let mut retry_attempt = Box::pin(request.attempt());
    assert!(poll!(&mut retry_attempt).is_pending());
    let retry = probe.next_attempt().await;
    assert_eq!((retry.id, retry.prompt), (7, "retry".into()));
    retry.reply.send(Ok("retried".into())).unwrap();
    assert_eq!(retry_attempt.await, Ok("retried".into()));
    assert_eq!(
        probe.counts(),
        Counts {
            logical_rows_submitted: 1,
            backend_attempts: 2,
            in_flight: 0,
            peak_in_flight: 1,
            cancelled_attempts: 0,
        }
    );
}
