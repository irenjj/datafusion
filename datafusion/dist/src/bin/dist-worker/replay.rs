// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: See the NOTICE file distributed with this work for additional information regarding copyright ownership
// SPDX-FileContributor: Licensed to the Apache Software Foundation (ASF) under one or more contributor license agreements

use std::collections::BTreeSet;
use std::num::NonZeroUsize;
use std::path::PathBuf;
use std::process::Command;
use std::time::Duration;

use datafusion::common::{Result, exec_err, plan_datafusion_err, plan_err};
use datafusion::prelude::{ParquetReadOptions, SessionContext};
use datafusion_dist::{DlqReplay, LeaseConfig, ReplaySummary, SinkOptions};
use serde::{Deserialize, Serialize};

use super::job::{Location, PlanningOptions, StageJob, table_binding};
use super::{Arguments, local};

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
pub(super) struct ReplayOptions {
    base_output_root: String,
    output_root: String,
    sql: String,
    row_key: String,
    table: Option<String>,
    stage_table: Option<String>,
    #[serde(default = "default_batch_size")]
    batch_size: NonZeroUsize,
}

struct Output {
    replay: DlqReplay,
    ctx: SessionContext,
    location: Location,
}

pub(super) struct PendingReplay {
    pub(super) job: StageJob,
    pub(super) attempt_id: u32,
    output: Output,
}

pub(super) enum Preparation {
    Complete(ReplaySummary),
    Pending(Box<PendingReplay>),
}

#[derive(Serialize)]
#[serde(untagged)]
pub(super) enum PlannedReplay {
    Complete { result: ReplaySummary },
    Pending { attempt_id: u32 },
}

fn default_batch_size() -> NonZeroUsize {
    NonZeroUsize::new(8192).unwrap()
}

impl ReplayOptions {
    pub(super) fn from_args(args: &mut Arguments) -> Result<Self> {
        Ok(Self {
            base_output_root: args.required("--base-output-root")?,
            output_root: args.required("--output-root")?,
            sql: args.required("--sql")?,
            row_key: args.required("--row-key")?,
            table: args.flag("--table")?,
            stage_table: args.flag("--stage-table")?,
            batch_size: args.number("--batch-size", Some(default_batch_size()))?,
        })
    }

    async fn open(&self) -> Result<Output> {
        if self.table.is_some() && self.stage_table.is_some() {
            return plan_err!("Replay accepts one source table");
        }
        let base = Location::parse(&self.base_output_root)?.resolve_local()?;
        let location = Location::parse(&self.output_root)?.resolve_local()?;
        if base.overlaps(&location)? {
            return plan_err!("Replay output and base directories must be disjoint");
        }
        let ctx = SessionContext::new();
        let mut stores = BTreeSet::new();
        base.register_input(&ctx, &mut stores)?;
        location.register_input(&ctx, &mut stores)?;
        let reader = base.open_stage(&ctx).await?;
        let identity = serde_json::to_vec(&(
            &self.sql,
            &self.table,
            &self.stage_table,
            &self.batch_size,
        ))
        .map_err(|error| plan_datafusion_err!("Cannot encode replay options: {error}"))?;
        let replay = DlqReplay::open(
            &ctx,
            reader,
            location.store_url()?,
            location.store_path()?,
            &self.row_key,
            &identity,
        )
        .await?;
        Ok(Output {
            replay,
            ctx,
            location,
        })
    }

    pub(super) async fn prepare(&self) -> Result<Preparation> {
        let output = self.open().await?;
        if let Some(result) = output.replay.completed(&output.ctx).await? {
            return Ok(Preparation::Complete(result));
        }
        let attempt_id = output.replay.reserve_attempt().await?;
        if !output.replay.needs_replay() {
            let result = output.replay.merge(&output.ctx, attempt_id, None).await?;
            return Ok(Preparation::Complete(result));
        }
        let (binding, stage_table) = match (&self.table, &self.stage_table) {
            (Some(binding), None) => (table_binding(binding, "--table")?, false),
            (None, Some(binding)) => (table_binding(binding, "--stage-table")?, true),
            _ => {
                return plan_err!(
                    "Replay with failed keys requires --table or --stage-table"
                );
            }
        };
        let (name, source_path) = binding;
        let source = Location::parse(source_path)?.resolve_local()?;
        if source.overlaps(&output.location)? {
            return plan_err!("Replay output and source directories must be disjoint");
        }
        let mut input_stores = BTreeSet::new();
        source.register_input(&output.ctx, &mut input_stores)?;
        let source = if stage_table {
            let stage = source.open_stage(&output.ctx).await?;
            output.ctx.read_table(stage.data_table())?
        } else {
            output
                .ctx
                .read_parquet(source.input_path()?, ParquetReadOptions::default())
                .await?
        };
        output
            .replay
            .stage_input(attempt_id, source, self.batch_size)
            .await?;
        let workspace = output.replay.workspace(attempt_id);
        let input = output.location.child_directory(&workspace.join("input"))?;
        let retry = output.retry_root(attempt_id)?;
        let job = StageJob::plan_with_stores(
            PlanningOptions {
                sql: self.sql.clone(),
                tables: vec![format!("{name}={}", input.input_path()?)],
                stage_tables: vec![],
                stage_dlq_tables: vec![],
                #[cfg(feature = "lake")]
                lance_tables: vec![],
                sink: SinkOptions {
                    row_key: Some(self.row_key.clone()),
                },
                output_root: Some(retry.input_path()?),
            },
            input_stores,
            Some(&input),
        )
        .await?;
        Ok(Preparation::Pending(Box::new(PendingReplay {
            job,
            attempt_id,
            output,
        })))
    }

    pub(super) async fn run(&self, config: LeaseConfig) -> Result<ReplaySummary> {
        config.validate()?;
        match self.prepare().await? {
            Preparation::Complete(result) => Ok(result),
            Preparation::Pending(pending) => {
                local::run_with_timeout(&pending.job, config).await?;
                pending.output.finalize(pending.attempt_id).await
            }
        }
    }

    pub(super) async fn finalize(&self, attempt_id: u32) -> Result<ReplaySummary> {
        self.open().await?.finalize(attempt_id).await
    }
}

impl Output {
    fn retry_root(&self, attempt_id: u32) -> Result<Location> {
        self.location
            .child_directory(&self.replay.workspace(attempt_id).join("retry"))
    }

    async fn finalize(&self, attempt_id: u32) -> Result<ReplaySummary> {
        if let Some(result) = self.replay.completed(&self.ctx).await? {
            return Ok(result);
        }
        let retry = self.retry_root(attempt_id)?.open_stage(&self.ctx).await?;
        self.replay.merge(&self.ctx, attempt_id, Some(retry)).await
    }
}

pub(super) fn lease_config(timeout_secs: f64, max_attempts: u32) -> Result<LeaseConfig> {
    let config = LeaseConfig {
        lease_ttl: Duration::try_from_secs_f64(timeout_secs)
            .map_err(|_| plan_datafusion_err!("Invalid replay timeout"))?,
        max_attempts,
        ..Default::default()
    };
    config.validate()?;
    Ok(config)
}

pub(super) async fn cli(mut args: Arguments) -> Result<()> {
    let replay = ReplayOptions::from_args(&mut args)?;
    let config = lease_config(
        args.number("--timeout-secs", Some(60.0))?,
        args.number("--max-attempts", Some(LeaseConfig::default().max_attempts))?,
    )?;
    let execution_mode = args.flag("--execution-mode")?;
    let address = args.flag("--ray-address")?;
    let pool = args.flag("--pool-size")?;
    let concurrency = args.flag("--concurrency")?;
    let driver = args.flag("--ray-driver")?;
    let worker = args.flag("--worker")?;
    let secrets = args.switch("--secrets-stdin")?;
    args.finish()?;
    let mode = execution_mode.as_deref().unwrap_or_else(|| {
        if address.is_some() || pool.is_some() {
            "actor"
        } else {
            "local"
        }
    });
    if !matches!(mode, "local" | "actor" | "task") {
        return plan_err!("--execution-mode must be local, actor or task");
    }
    if mode == "local" && (address.is_some() || pool.is_some() || concurrency.is_some()) {
        return plan_err!("Ray scheduling options require actor or task execution");
    }
    if mode == "local" && !secrets && worker.is_none() {
        return super::write_json(std::io::stdout().lock(), &replay.run(config).await?);
    }

    let executable = std::env::current_exe()?;
    let driver = driver
        .map(PathBuf::from)
        .or_else(|| std::env::var_os("DF_DIST_RAY_DRIVER").map(PathBuf::from))
        .or_else(|| {
            executable
                .with_file_name("ray_driver.py")
                .is_file()
                .then(|| executable.with_file_name("ray_driver.py"))
        })
        .unwrap_or_else(|| {
            PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("python/ray_driver.py")
        });
    if !driver.is_file() {
        return plan_err!("Cannot locate ray_driver.py, supply --ray-driver PATH");
    }
    let mut command = Command::new("python3");
    command.arg("-B").arg(driver).args([
        format!("--execution-mode={mode}"),
        format!("--base-output-root={}", replay.base_output_root),
        format!("--output-root={}", replay.output_root),
        format!("--sql={}", replay.sql),
        format!("--row-key={}", replay.row_key),
        format!("--batch-size={}", replay.batch_size),
        format!("--timeout={}", config.lease_ttl.as_secs_f64()),
        format!("--max-attempts={}", config.max_attempts),
    ]);
    for (flag, value) in [
        ("--table", replay.table),
        ("--stage-table", replay.stage_table),
        ("--ray-address", address),
        ("--pool-size", pool),
        ("--concurrency", concurrency),
    ] {
        if let Some(value) = value {
            command.arg(format!("{flag}={value}"));
        }
    }
    if let Some(worker) = worker {
        command.arg(format!("--worker={worker}"));
    } else {
        command.arg("--worker").arg(executable);
    }
    if secrets {
        command.arg("--secrets-stdin");
    }
    let status = command.status()?;
    if !status.success() {
        return exec_err!("Replay driver exited with {status}");
    }
    Ok(())
}
