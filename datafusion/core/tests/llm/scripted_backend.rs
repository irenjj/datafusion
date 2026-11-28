// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: See the NOTICE file distributed with this work for additional information regarding copyright ownership
// SPDX-FileContributor: Licensed to the Apache Software Foundation (ASF) under one or more contributor license agreements

use std::sync::{Arc, Mutex};
use std::time::Duration;

use async_trait::async_trait;
use datafusion::llm::{
    CancellationToken, LlmBackend, LlmBackendError, LlmRequest, LlmResponse, LlmUsage,
};
use tokio::sync::{mpsc, oneshot};

#[derive(Debug, Default, Clone, Copy)]
pub(super) struct Counts {
    pub(super) backend_attempts: usize,
    pub(super) in_flight: usize,
    pub(super) peak_in_flight: usize,
    pub(super) cancelled_attempts: usize,
}

#[derive(Debug)]
pub(super) struct ScriptedBackend {
    counts: Arc<Mutex<Counts>>,
    started: mpsc::UnboundedSender<PendingAttempt>,
    max_batch_size: Option<usize>,
}

pub(super) struct InferenceProbe {
    counts: Arc<Mutex<Counts>>,
    started: mpsc::UnboundedReceiver<PendingAttempt>,
}

type Reply = Result<LlmResponse, LlmBackendError>;

#[derive(Debug)]
pub(super) struct PendingAttempt {
    pub(super) prompts: Vec<String>,
    pub(super) reply: oneshot::Sender<Reply>,
}

impl PendingAttempt {
    pub(super) fn succeed(self, prefix: &str, usage: Option<LlmUsage>) {
        let outputs = self
            .prompts
            .iter()
            .map(|prompt| Some(format!("{prefix}{prompt}")))
            .collect();
        self.reply.send(Ok(LlmResponse { outputs, usage })).unwrap();
    }
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
    pub(super) fn new() -> (Arc<Self>, InferenceProbe) {
        Self::with_max_batch_size(None)
    }

    pub(super) fn with_max_batch_size(
        max_batch_size: Option<usize>,
    ) -> (Arc<Self>, InferenceProbe) {
        let (sender, receiver) = mpsc::unbounded_channel();
        let counts = Arc::new(Mutex::new(Counts::default()));
        (
            Arc::new(Self {
                counts: Arc::clone(&counts),
                started: sender,
                max_batch_size,
            }),
            InferenceProbe {
                counts,
                started: receiver,
            },
        )
    }
}

#[async_trait]
impl LlmBackend for ScriptedBackend {
    fn max_batch_size(&self) -> Option<usize> {
        self.max_batch_size
    }

    async fn generate(&self, request: LlmRequest, cancel: CancellationToken) -> Reply {
        {
            let mut counts = self.counts.lock().unwrap();
            counts.backend_attempts += 1;
            counts.in_flight += 1;
            counts.peak_in_flight = counts.peak_in_flight.max(counts.in_flight);
        }
        let mut in_flight = InFlight {
            counts: &self.counts,
            completed: false,
        };
        let (reply, wait) = oneshot::channel();
        self.started
            .send(PendingAttempt {
                prompts: request.prompts,
                reply,
            })
            .unwrap();
        tokio::select! {
            biased;
            _ = cancel.cancelled() => Err(LlmBackendError::Cancelled),
            result = wait => {
                in_flight.completed = true;
                result.expect("The test dropped an inference gate without replying")
            }
        }
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

    pub(super) fn assert_no_new_attempt(&mut self) {
        assert!(matches!(
            self.started.try_recv(),
            Err(mpsc::error::TryRecvError::Empty)
        ));
    }
}
