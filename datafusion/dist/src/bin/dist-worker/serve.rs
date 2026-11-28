// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: See the NOTICE file distributed with this work for additional information regarding copyright ownership
// SPDX-FileContributor: Licensed to the Apache Software Foundation (ASF) under one or more contributor license agreements

use std::collections::BTreeSet;
use std::io::{self, BufRead};

use datafusion::common::{Result, exec_datafusion_err, plan_datafusion_err, plan_err};
use datafusion_dist::{
    LeaseConfig, LocalWorker, ParquetSplitSink, PreparedStage, ReplaySummary,
    StageManifest,
};
use serde::{Deserialize, Serialize};

use super::replay::{ReplayOptions, lease_config};
use super::{AttemptReceipt, StageJob, write_json};

#[derive(Deserialize)]
#[serde(tag = "command", rename_all = "snake_case", deny_unknown_fields)]
enum Request {
    Ping,
    LoadJob {
        job_id: String,
        job: Box<StageJob>,
        output_root: Option<String>,
    },
    RunSplit {
        job_id: String,
        split_id: u64,
        attempt_id: u32,
    },
    CommitSplit {
        job_id: String,
        receipt: Box<AttemptReceipt>,
    },
    FinalizeStage {
        job_id: String,
    },
    ReplayDlq {
        options: Box<ReplayOptions>,
        timeout_secs: Option<f64>,
        max_attempts: Option<u32>,
    },
    Shutdown,
}

#[derive(Serialize)]
#[serde(untagged)]
enum Reply<'a> {
    Success { ok: bool, result: &'a Response },
    Failure { ok: bool, error: Failure },
}

#[derive(Serialize)]
struct Failure {
    code: &'static str,
    message: String,
}

#[derive(Serialize)]
#[serde(untagged)]
enum Response {
    Pong { protocol: &'static str },
    Receipt(AttemptReceipt),
    Manifest(StageManifest),
    Replay(Box<ReplaySummary>),
    Ack,
}

struct LoadedJob {
    id: String,
    job: StageJob,
    _worker: LocalWorker,
    stage: PreparedStage,
    sink: ParquetSplitSink,
}

impl LoadedJob {
    fn prepare(id: String, job: StageJob, output_root: Option<&str>) -> Result<Self> {
        if id.trim().is_empty() {
            return plan_err!("job_id cannot be empty");
        }
        job.validate()?;
        let mut fingerprints = BTreeSet::new();
        for (index, split) in job.splits.iter().enumerate() {
            if split.split_id != index as u64 || !fingerprints.insert(&split.fingerprint)
            {
                return plan_err!("Invalid stage job split inventory");
            }
        }
        let mut pending_ids = BTreeSet::new();
        for pending in &job.pending_splits {
            if !pending_ids.insert(pending.split_id)
                || usize::try_from(pending.split_id)
                    .ok()
                    .and_then(|id| job.splits.get(id))
                    .is_none_or(|split| pending.attempt_id < split.attempt_id)
            {
                return plan_err!("Invalid pending split in stage job");
            }
        }
        let worker = job.worker()?;
        let stage = worker.prepare_stage(job.stage())?;
        let sink = job.sink_with_context(output_root, Some(worker.context()))?;
        Ok(Self {
            id,
            job,
            _worker: worker,
            stage,
            sink,
        })
    }
}

#[derive(Default)]
struct Server {
    loaded: Option<LoadedJob>,
}

impl Server {
    fn job(&self, id: &str) -> Result<&LoadedJob> {
        self.loaded
            .as_ref()
            .filter(|loaded| loaded.id == id)
            .ok_or_else(|| plan_datafusion_err!("No stage loaded for job_id"))
    }

    async fn handle(&mut self, request: Request) -> Result<Response> {
        match request {
            Request::Ping => Ok(Response::Pong {
                protocol: "df-worker-serve-v1",
            }),
            Request::LoadJob {
                job_id,
                job,
                output_root,
            } => {
                if self.loaded.is_some() {
                    return plan_err!("A stage job is already loaded");
                }
                self.loaded =
                    Some(LoadedJob::prepare(job_id, *job, output_root.as_deref())?);
                Ok(Response::Ack)
            }
            Request::RunSplit {
                job_id,
                split_id,
                attempt_id,
            } => {
                let loaded = self.job(&job_id)?;
                let split = loaded.job.split(split_id, attempt_id)?;
                let output = loaded.stage.execute_split(&split).await?;
                Ok(Response::Receipt(AttemptReceipt {
                    artifact: loaded.sink.write_attempt(&split, &output.batches).await?,
                    metrics: output.metrics,
                }))
            }
            Request::CommitSplit { job_id, receipt } => {
                let loaded = self.job(&job_id)?;
                let split = loaded
                    .job
                    .split(receipt.artifact.split_id, receipt.artifact.attempt_id)?;
                loaded
                    .sink
                    .commit_split_with_metrics(&split, receipt.artifact, receipt.metrics)
                    .await?;
                Ok(Response::Ack)
            }
            Request::FinalizeStage { job_id } => {
                let loaded = self.job(&job_id)?;
                let manifest = loaded.sink.finalize_stage(&loaded.job.splits).await?;
                Ok(Response::Manifest(manifest))
            }
            Request::ReplayDlq {
                options,
                timeout_secs,
                max_attempts,
            } => {
                let config = lease_config(
                    timeout_secs.unwrap_or(60.0),
                    max_attempts.unwrap_or_else(|| LeaseConfig::default().max_attempts),
                )?;
                Ok(Response::Replay(Box::new(options.run(config).await?)))
            }
            Request::Shutdown => Ok(Response::Ack),
        }
    }
}

pub(super) fn run() -> Result<()> {
    // The protocol thread waits on stdin between commands. Keep the Tokio reactor
    // and pooled connection tasks running independently of that blocking read.
    let runtime = tokio::runtime::Builder::new_multi_thread()
        .worker_threads(1)
        .enable_all()
        .build()?;
    let mut server = Server::default();
    let mut output = io::stdout().lock();
    // Read bytes so invalid UTF-8 is a request error and the next line remains usable.
    for line in io::stdin().lock().split(b'\n') {
        let (result, code, shutdown) = match serde_json::from_slice::<Request>(&line?) {
            Ok(request) => {
                let shutdown = matches!(request, Request::Shutdown);
                (
                    runtime.block_on(server.handle(request)),
                    "command_failed",
                    shutdown,
                )
            }
            Err(error) => (
                Err(exec_datafusion_err!("Invalid worker request: {error}")),
                "invalid_request",
                false,
            ),
        };
        let reply = match &result {
            Ok(result) => Reply::Success { ok: true, result },
            Err(error) => Reply::Failure {
                ok: false,
                error: Failure {
                    code,
                    message: error.to_string(),
                },
            },
        };
        write_json(&mut output, &reply)?;
        if shutdown {
            break;
        }
    }
    Ok(())
}
