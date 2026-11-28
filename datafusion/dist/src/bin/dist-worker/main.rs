// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: See the NOTICE file distributed with this work for additional information regarding copyright ownership
// SPDX-FileContributor: Licensed to the Apache Software Foundation (ASF) under one or more contributor license agreements

use std::collections::BTreeMap;
use std::fs::File;
use std::io::{self, BufReader, Read, Write};
use std::process::ExitCode;
use std::str::FromStr;
use std::sync::Arc;
use std::time::Duration;

use datafusion::arrow::json::writer::{LineDelimited, WriterBuilder};
use datafusion::common::{Result, exec_datafusion_err, plan_datafusion_err, plan_err};
use datafusion_dist::{
    AttemptArtifact, LeaseConfig, SinkOptions, SplitMetrics, SystemClock,
};
use futures::TryStreamExt;
use serde::{Deserialize, Serialize, de::DeserializeOwned};

mod job;
mod local;

use job::{PlanningOptions, StageJob};

const USAGE: &str = "\
dist-worker plan --sql SQL --out JOB [--table name=path] [--stage-table name=PATH] [--stage-dlq-table name=PATH] [--row-key COLUMN] [--output-root PATH]
dist-worker run-split --job JOB --split-id N --attempt-id N [--output-root PATH]
dist-worker commit-split --job JOB [--artifact FILE|-] [--output-root PATH]
dist-worker finalize-stage --job JOB [--output-root PATH]
dist-worker query-stage --output-root PATH --sql SQL
dist-worker run-local --sql SQL --table name=path --output-root PATH [--row-key COLUMN] [--timeout-secs N] [--max-attempts N]";

#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct AttemptReceipt {
    artifact: AttemptArtifact,
    metrics: SplitMetrics,
}

#[tokio::main(flavor = "current_thread")]
async fn main() -> ExitCode {
    let mut argv = std::env::args().skip(1);
    let command = argv.next();
    if matches!(command.as_deref(), None | Some("--help" | "-h")) {
        println!("{USAGE}");
        return ExitCode::SUCCESS;
    }
    match run(command.as_deref().unwrap(), Arguments::parse(argv)).await {
        Ok(()) => ExitCode::SUCCESS,
        Err(error) => {
            eprintln!("{error}");
            ExitCode::FAILURE
        }
    }
}

async fn run(command: &str, args: Result<Arguments>) -> Result<()> {
    let mut args = args?;
    match command {
        "plan" => {
            let file = args.required("--out")?;
            let options = args.planning()?;
            args.finish()?;
            let job = StageJob::plan(options).await?;
            write_json(File::create(file)?, &job)
        }
        "run-split" => {
            let path = args.required("--job")?;
            let root = args.flag("--output-root")?;
            let split_id = args.number("--split-id", None)?;
            let attempt_id = args.number("--attempt-id", None)?;
            args.finish()?;
            let job = read_job(&path)?;
            let split = job.split(split_id, attempt_id)?;
            let sink = job.sink(root.as_deref())?;
            let output = job.worker()?.execute_split(job.stage(), &split).await?;
            let receipt = AttemptReceipt {
                artifact: sink.write_attempt(&split, &output.batches).await?,
                metrics: output.metrics,
            };
            write_json(io::stdout().lock(), &receipt)
        }
        "commit-split" => {
            let path = args.required("--job")?;
            let root = args.flag("--output-root")?;
            let artifact = args.flag("--artifact")?;
            args.finish()?;
            let job = read_job(&path)?;
            let receipt: AttemptReceipt = match artifact.as_deref() {
                Some(file) if file != "-" => {
                    read_json(BufReader::new(File::open(file)?))?
                }
                _ => read_json(io::stdin().lock())?,
            };
            let split =
                job.split(receipt.artifact.split_id, receipt.artifact.attempt_id)?;
            job.sink(root.as_deref())?
                .commit_split_with_metrics(&split, receipt.artifact, receipt.metrics)
                .await
        }
        "finalize-stage" => {
            let path = args.required("--job")?;
            let root = args.flag("--output-root")?;
            args.finish()?;
            let job = read_job(&path)?;
            let manifest = job
                .sink(root.as_deref())?
                .finalize_stage(&job.splits)
                .await?;
            write_json(io::stdout().lock(), &manifest)
        }
        "query-stage" => {
            let root = args.required("--output-root")?;
            let sql = args.required("--sql")?;
            args.finish()?;
            let mut stream = job::query_stage(&root, &sql)
                .await?
                .execute_stream()
                .await?;
            let mut writer = WriterBuilder::new()
                .with_explicit_nulls(true)
                .build::<_, LineDelimited>(io::stdout().lock());
            while let Some(batch) = stream.try_next().await? {
                writer.write(&batch)?;
            }
            writer.finish()?;
            writer.into_inner().flush()?;
            Ok(())
        }
        "run-local" => {
            let options = args.planning()?;
            if options.output_root.is_none() {
                return plan_err!("--output-root is required");
            }
            let config = LeaseConfig {
                lease_ttl: Duration::from_secs(args.number("--timeout-secs", Some(30))?),
                max_attempts: args.number(
                    "--max-attempts",
                    Some(LeaseConfig::default().max_attempts),
                )?,
                ..Default::default()
            };
            args.finish()?;
            config.validate()?;
            let job = StageJob::plan(options).await?;
            let manifest = local::run(
                &job,
                config,
                Arc::new(SystemClock::default()),
                async |stage, queue, split, token| {
                    tokio::time::timeout(
                        config.lease_ttl,
                        stage.execute_lease(queue, split, token),
                    )
                    .await
                    .map_err(|_| {
                        exec_datafusion_err!(
                            "Split {} execution timed out",
                            split.split_id
                        )
                    })?
                },
            )
            .await?;
            write_json(io::stdout().lock(), &manifest)
        }
        _ => plan_err!("Unknown command '{command}'.\n{USAGE}"),
    }
}

fn read_job(path: &str) -> Result<StageJob> {
    let job: StageJob = read_json(BufReader::new(File::open(path)?))?;
    job.validate()?;
    Ok(job)
}

fn read_json<T: DeserializeOwned>(reader: impl Read) -> Result<T> {
    serde_json::from_reader(reader)
        .map_err(|error| exec_datafusion_err!("Invalid worker JSON: {error}"))
}

fn write_json(mut writer: impl Write, value: &impl Serialize) -> Result<()> {
    serde_json::to_writer(&mut writer, value)
        .map_err(|error| exec_datafusion_err!("Cannot write worker JSON: {error}"))?;
    writer.write_all(b"\n")?;
    writer.flush()?;
    Ok(())
}

#[derive(Default)]
struct Arguments(BTreeMap<String, Vec<String>>);

impl Arguments {
    fn parse(mut args: impl Iterator<Item = String>) -> Result<Self> {
        let mut parsed = Self::default();
        while let Some(key) = args.next() {
            if !key.starts_with("--") {
                return plan_err!("Expected a flag beginning with --");
            }
            let value = args
                .next()
                .ok_or_else(|| plan_datafusion_err!("{key} needs a value"))?;
            if value.starts_with("--") && key != "--sql" {
                return plan_err!("{key} needs a value");
            }
            let name = if key == "--df.split_row_key" {
                "--row-key".into()
            } else {
                key
            };
            parsed.0.entry(name).or_default().push(value);
        }
        Ok(parsed)
    }

    fn flag(&mut self, key: &str) -> Result<Option<String>> {
        match self.0.remove(key) {
            None => Ok(None),
            Some(mut values) if values.len() == 1 => Ok(values.pop()),
            Some(_) => plan_err!("{key} may be supplied only once"),
        }
    }

    fn required(&mut self, key: &str) -> Result<String> {
        self.flag(key)?
            .ok_or_else(|| plan_datafusion_err!("{key} is required"))
    }

    fn number<T: FromStr>(&mut self, key: &str, default: Option<T>) -> Result<T> {
        match self.flag(key)? {
            Some(value) => value
                .parse()
                .map_err(|_| plan_datafusion_err!("Invalid number for {key}")),
            None => default.ok_or_else(|| plan_datafusion_err!("{key} is required")),
        }
    }

    fn planning(&mut self) -> Result<PlanningOptions> {
        let sql = self.required("--sql")?;
        let tables = self.0.remove("--table").unwrap_or_default();
        let stage_tables = self.0.remove("--stage-table").unwrap_or_default();
        let stage_dlq_tables = self.0.remove("--stage-dlq-table").unwrap_or_default();
        if tables.is_empty() && stage_tables.is_empty() && stage_dlq_tables.is_empty() {
            return plan_err!(
                "--table is required (or use --stage-table or --stage-dlq-table)"
            );
        }
        Ok(PlanningOptions {
            sql,
            tables,
            stage_tables,
            stage_dlq_tables,
            sink: SinkOptions {
                row_key: self.flag("--row-key")?,
            },
            output_root: self.flag("--output-root")?,
        })
    }

    fn finish(self) -> Result<()> {
        if let Some(key) = self.0.keys().next() {
            return plan_err!("Unknown flag {key}");
        }
        Ok(())
    }
}
