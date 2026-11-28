// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: See the NOTICE file distributed with this work for additional information regarding copyright ownership
// SPDX-FileContributor: Licensed to the Apache Software Foundation (ASF) under one or more contributor license agreements

use std::fs::{self, File};
use std::io::Write;
use std::path::Path;
use std::process::{Command, Output, Stdio};
use std::sync::Arc;

use datafusion::arrow::array::Int64Array;
use datafusion::arrow::datatypes::{DataType, Field, Schema};
use datafusion::arrow::record_batch::RecordBatch;
use datafusion::parquet::arrow::ArrowWriter;
use datafusion::parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;
use serde_json::{Value, json};
use tempfile::tempdir;

#[cfg(feature = "ai")]
#[path = "worker/ai.rs"]
mod ai;

type Result<T> = std::result::Result<T, Box<dyn std::error::Error>>;

fn worker() -> Command {
    Command::new(env!("CARGO_BIN_EXE_dist-worker"))
}

fn successful(output: Output) -> Vec<u8> {
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    output.stdout
}

fn input_files(path: &Path) -> Result<()> {
    fs::create_dir(path)?;
    let schema = Arc::new(Schema::new(vec![Field::new("id", DataType::Int64, false)]));
    for id in [7, 11] {
        let batch = RecordBatch::try_new(
            Arc::clone(&schema),
            vec![Arc::new(Int64Array::from(vec![id]))],
        )?;
        let mut writer = ArrowWriter::try_new(
            File::create(path.join(format!("{id}.parquet")))?,
            Arc::clone(&schema),
            None,
        )?;
        writer.write(&batch)?;
        writer.close()?;
    }
    Ok(())
}

#[test]
fn run_local_and_resume_two_parquet_splits() -> Result<()> {
    let dir = tempdir()?;
    let input = dir.path().join("input");
    input_files(&input)?;
    let table = format!("t={}", input.display());
    let root = dir.path().join("out");
    let run = || -> Result<Value> {
        let output = worker()
            .args([
                "run-local",
                "--sql",
                "SELECT id + 1 AS id FROM t",
                "--table",
                &table,
            ])
            .arg("--output-root")
            .arg(&root)
            .output()?;
        Ok(serde_json::from_slice(&successful(output))?)
    };
    let manifest = run()?;
    assert_eq!(manifest["output_rows"], 2);
    assert_eq!(manifest["dlq_rows"], 0);
    let stage_dir = root
        .join("stages")
        .join(manifest["plan_fingerprint"].as_str().unwrap());
    let mut values = vec![];
    for id in 0..2 {
        let split = &manifest["splits"][id];
        assert_eq!(split["split_id"], id);
        assert_eq!(split["winning_attempt_id"], 0);
        assert_eq!(split["output_rows"], 1);
        assert!(split["dlq_rel_path"].is_null());
        let path = stage_dir.join(split["data_rel_path"].as_str().unwrap());
        for batch in
            ParquetRecordBatchReaderBuilder::try_new(File::open(path)?)?.build()?
        {
            let batch = batch?;
            let ids = batch
                .column(0)
                .as_any()
                .downcast_ref::<Int64Array>()
                .unwrap();
            values.extend(ids.values().iter().copied());
        }
    }
    values.sort_unstable();
    assert_eq!(values, [8, 12]);
    assert_eq!(run()?, manifest);
    Ok(())
}

#[test]
fn stateless_worker_receipts_and_partial_resume() -> Result<()> {
    let dir = tempdir()?;
    let input = dir.path().join("input");
    let root = dir.path().join("output");
    let job = dir.path().join("stage-job.json");
    let receipt_file = dir.path().join("receipt.json");
    input_files(&input)?;
    let table = format!("t={}", input.display());
    let plan = || -> Result<Value> {
        successful(
            worker()
                .args(["plan", "--sql", "SELECT id FROM t", "--table", &table])
                .arg("--out")
                .arg(&job)
                .arg("--output-root")
                .arg(&root)
                .output()?,
        );
        Ok(serde_json::from_slice(&fs::read(&job)?)?)
    };
    let planned = plan()?;
    assert_eq!(
        planned["pending_splits"],
        json!([{"split_id": 0, "attempt_id": 0}, {"split_id": 1, "attempt_id": 0}])
    );
    let stage = &planned["stage"]["plan_fingerprint"];

    let run_split = |id: &str, attempt: &str| -> Result<Vec<u8>> {
        Ok(successful(
            worker()
                .args(["run-split", "--split-id", id, "--attempt-id", attempt])
                .arg("--job")
                .arg(&job)
                .arg("--output-root")
                .arg(&root)
                .output()?,
        ))
    };
    let result = run_split("0", "2")?;
    let receipt: Value = serde_json::from_slice(&result)?;
    assert_eq!(receipt["metrics"]["output_rows"], 1);
    assert_eq!(receipt["artifact"]["attempt_id"], 2);
    assert!(receipt.get("batches").is_none());
    let artifact_path = receipt["artifact"]["data_rel_path"].as_str().unwrap();
    assert!(
        root.join("stages")
            .join(stage.as_str().unwrap())
            .join(artifact_path)
            .is_file()
    );
    fs::write(&receipt_file, result)?;
    let commit = || -> Result<()> {
        successful(
            worker()
                .arg("commit-split")
                .arg("--job")
                .arg(&job)
                .arg("--artifact")
                .arg(&receipt_file)
                .arg("--output-root")
                .arg(&root)
                .output()?,
        );
        Ok(())
    };
    commit()?;
    let mut replay = worker()
        .arg("commit-split")
        .arg("--job")
        .arg(&job)
        .stdin(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()?;
    replay
        .stdin
        .take()
        .unwrap()
        .write_all(&fs::read(&receipt_file)?)?;
    successful(replay.wait_with_output()?);
    run_split("1", "3")?;
    let resumed = plan()?;
    assert_eq!(resumed["stage"]["plan_fingerprint"], *stage);
    assert_eq!(
        resumed["pending_splits"],
        json!([{"split_id": 1, "attempt_id": 4}])
    );
    assert_eq!(resumed["splits"].as_array().unwrap().len(), 2);

    let unfinished = worker()
        .arg("finalize-stage")
        .arg("--job")
        .arg(&job)
        .output()?;
    assert!(!unfinished.status.success());
    assert!(String::from_utf8_lossy(&unfinished.stderr).contains("not committed"));
    fs::write(&receipt_file, run_split("1", "4")?)?;
    commit()?;
    let finalized = successful(
        worker()
            .arg("finalize-stage")
            .arg("--job")
            .arg(&job)
            .output()?,
    );
    let manifest: Value = serde_json::from_slice(&finalized)?;
    assert_eq!(manifest["output_rows"], 2);
    assert_eq!(manifest["splits"][0]["winning_attempt_id"], 2);
    assert_eq!(manifest["splits"][1]["winning_attempt_id"], 4);
    Ok(())
}

#[test]
fn planning_rejects_custom_bindings_and_embedded_credentials() -> Result<()> {
    let dir = tempdir()?;
    let job = dir.path().join("stage-job.json");
    for (path, expected) in [
        ("custom://bucket/part.parquet", "Unsupported object store"),
        (
            "s3://user:secret@bucket/part.parquet",
            "embedded credentials",
        ),
        (
            "s3://bucket/part.parquet?token=secret",
            "embedded credentials",
        ),
    ] {
        let output = worker()
            .args([
                "plan",
                "--sql",
                "SELECT id FROM t",
                "--table",
                &format!("t={path}"),
            ])
            .arg("--out")
            .arg(&job)
            .output()?;
        assert!(!output.status.success());
        let error = String::from_utf8_lossy(&output.stderr);
        assert!(error.contains(expected), "{error}");
        assert!(!error.contains("secret"), "{error}");
        assert!(!job.exists());
    }
    Ok(())
}

#[cfg(feature = "ai")]
#[test]
fn ai_cli_plans_and_runs_with_runtime_mock() -> Result<()> {
    let dir = tempdir()?;
    let input = dir.path().join("input");
    let output_root = dir.path().join("output");
    let job = dir.path().join("job.json");
    input_files(&input)?;
    let table = format!("t={}", input.display());
    successful(worker()
        .args(["plan", "--sql", r#"SELECT id, llm(CAST(id AS VARCHAR), 'test-model', '{}') AS answer FROM t"#, "--table", &table])
        .arg("--out").arg(&job)
        .arg("--output-root").arg(&output_root)
        .env("DF_DIST_LLM_BACKEND", "mock")
        .env("OPENAI_API_KEY", "runtime-only-fixture")
        .output()?);
    let job_bytes = fs::read(&job)?;
    let job_json: Value = serde_json::from_slice(&job_bytes)?;
    let plan_bytes: Vec<u8> =
        serde_json::from_value(job_json["stage"]["plan_bytes"].clone())?;
    for bytes in [&job_bytes, &plan_bytes] {
        assert!(
            !bytes
                .windows("runtime-only-fixture".len())
                .any(|window| window == b"runtime-only-fixture")
        );
    }
    let receipt = successful(
        worker()
            .args(["run-split", "--split-id", "0", "--attempt-id", "0"])
            .arg("--job")
            .arg(&job)
            .arg("--output-root")
            .arg(&output_root)
            .env("DF_DIST_LLM_BACKEND", "mock")
            .output()?,
    );
    assert_eq!(
        serde_json::from_slice::<Value>(&receipt)?["metrics"]["submitted_prompts"],
        1
    );
    Ok(())
}

#[cfg(not(feature = "s3"))]
#[test]
fn s3_inputs_and_outputs_require_s3_feature() -> Result<()> {
    let dir = tempdir()?;
    let job = dir.path().join("job.json");
    for (table, output_root) in [
        ("t=s3://bucket/part.parquet", None),
        ("t=unused.parquet", Some("s3://bucket/output")),
    ] {
        let mut command = worker();
        command
            .args(["plan", "--sql", "SELECT id FROM t", "--table", table])
            .arg("--out")
            .arg(&job);
        if let Some(root) = output_root {
            command.args(["--output-root", root]);
        }
        let output = command.output()?;
        assert!(!output.status.success());
        let error = String::from_utf8_lossy(&output.stderr);
        assert!(
            error.contains("S3 object store URLs require the s3 feature"),
            "{error}"
        );
        assert!(!job.exists());
    }
    Ok(())
}

#[cfg(feature = "ai")]
#[test]
fn openai_backend_selection_requires_ai_http_feature() -> Result<()> {
    let dir = tempdir()?;
    let input = dir.path().join("input");
    let job = dir.path().join("job.json");
    input_files(&input)?;
    let table = format!("t={}", input.display());
    let output = worker()
        .args([
            "plan", "--sql",
            r#"SELECT id, llm(CAST(id AS VARCHAR), 'test-model', '{"batch_size":1}') AS answer FROM t"#,
            "--table", &table,
        ])
        .arg("--out")
        .arg(&job)
        .env("DF_DIST_LLM_BACKEND", "openai")
        .env("OPENAI_BASE_URL", "http://127.0.0.1:1/v1")
        .env("OPENAI_API_KEY", "runtime-only-fixture")
        .env_remove("OPENAI_CA_CERT_FILE")
        .output()?;
    if cfg!(feature = "ai-http") {
        successful(output);
        let job_bytes = fs::read(&job)?;
        let job_json: Value = serde_json::from_slice(&job_bytes)?;
        let plan_bytes: Vec<u8> =
            serde_json::from_value(job_json["stage"]["plan_bytes"].clone())?;
        assert!(
            !plan_bytes
                .windows("runtime-only-fixture".len())
                .any(|w| w == b"runtime-only-fixture")
        );
    } else {
        assert!(!output.status.success());
        let error = String::from_utf8_lossy(&output.stderr);
        assert!(
            error.contains("DF_DIST_LLM_BACKEND=openai requires the ai-http feature"),
            "{error}"
        );
        assert!(!job.exists());
    }
    Ok(())
}
