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

#[cfg(feature = "lake")]
#[path = "worker/lake.rs"]
mod lake;

#[path = "worker/serve.rs"]
mod serve;

#[path = "worker/replay.rs"]
mod replay;

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

#[cfg(not(feature = "lake"))]
#[test]
fn lake_commands_require_the_lake_feature() -> Result<()> {
    for args in [
        vec![
            "plan",
            "--out",
            "unused",
            "--sql",
            "SELECT * FROM t",
            "--lance-table",
            "t=/unused",
        ],
        vec![
            "backfill-lance",
            "--dataset",
            "/unused",
            "--stage-output-root",
            "/unused",
        ],
    ] {
        let output = worker().args(args).output()?;
        assert!(!output.status.success());
        assert!(
            String::from_utf8_lossy(&output.stderr).contains("requires the lake feature"),
            "{output:?}"
        );
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

fn query_stage(root: &Path, sql: &str) -> Result<Vec<Value>> {
    let output = worker()
        .args(["query-stage", "--output-root"])
        .arg(root)
        .args(["--sql", sql])
        .output()?;
    Ok(std::str::from_utf8(&successful(output))?
        .lines()
        .map(serde_json::from_str)
        .collect::<std::result::Result<_, _>>()?)
}

#[test]
fn query_stage_and_plan_next_stage_from_manifest_winners() -> Result<()> {
    let dir = tempdir()?;
    let input = dir.path().join("input");
    input_files(&input)?;
    let root = dir.path().join("upstream");
    let output = worker()
        .args(["run-local", "--sql", "SELECT id FROM t", "--table"])
        .arg(format!("t={}", input.display()))
        .arg("--output-root")
        .arg(&root)
        .output()?;
    let manifest: Value = serde_json::from_slice(&successful(output))?;
    let stage_dir = root
        .join("stages")
        .join(manifest["plan_fingerprint"].as_str().unwrap());
    let first = stage_dir.join(manifest["splits"][0]["data_rel_path"].as_str().unwrap());
    let stale = stage_dir.join("attempts/split-0000000000000000-attempt-0001");
    fs::create_dir(&stale)?;
    fs::copy(&first, stale.join("data.parquet"))?;
    fs::copy(&first, stale.join("dlq.parquet"))?;

    assert_eq!(
        query_stage(&root, "SELECT COUNT(*) AS n FROM data")?,
        [json!({"n": 2})]
    );
    assert_eq!(
        query_stage(&stage_dir, "SELECT COUNT(*) AS n FROM dlq")?,
        [json!({"n": 0})]
    );
    let forbidden = dir.path().join("forbidden.parquet");
    let copy_attempt = worker()
        .args(["query-stage", "--output-root"])
        .arg(&root)
        .args([
            "--sql",
            &format!("COPY (SELECT id FROM data) TO '{}'", forbidden.display()),
        ])
        .output()?;
    assert!(!copy_attempt.status.success());
    assert!(copy_attempt.stdout.is_empty());
    assert!(
        String::from_utf8_lossy(&copy_attempt.stderr).contains("DML not supported: COPY")
    );
    assert!(!forbidden.exists());

    let job_path = dir.path().join("stage2-job.json");
    let second_root = dir.path().join("downstream");
    successful(
        worker()
            .args([
                "plan",
                "--sql",
                "SELECT id * 10 AS id FROM prior",
                "--stage-table",
            ])
            .arg(format!("prior={}", root.display()))
            .arg("--out")
            .arg(&job_path)
            .arg("--output-root")
            .arg(&second_root)
            .output()?,
    );
    let job: Value = serde_json::from_slice(&fs::read(&job_path)?)?;
    let pending = job["pending_splits"].as_array().unwrap();
    assert_eq!(pending.len(), 2);
    let receipt_path = dir.path().join("receipt.json");
    for split in pending {
        let result = worker()
            .args(["run-split", "--job"])
            .arg(&job_path)
            .args([
                "--split-id",
                &split["split_id"].to_string(),
                "--attempt-id",
                &split["attempt_id"].to_string(),
            ])
            .output()?;
        fs::write(&receipt_path, successful(result))?;
        successful(
            worker()
                .args(["commit-split", "--job"])
                .arg(&job_path)
                .arg("--artifact")
                .arg(&receipt_path)
                .output()?,
        );
    }
    successful(
        worker()
            .args(["finalize-stage", "--job"])
            .arg(&job_path)
            .output()?,
    );
    assert_eq!(
        query_stage(&second_root, "SELECT id FROM data ORDER BY id")?,
        [json!({"id": 70}), json!({"id": 110})]
    );

    let dlq_job = dir.path().join("dlq-job.json");
    successful(
        worker()
            .args([
                "plan",
                "--sql",
                "SELECT id FROM failures",
                "--stage-dlq-table",
            ])
            .arg(format!("failures={}", stage_dir.display()))
            .arg("--out")
            .arg(&dlq_job)
            .output()?,
    );
    let job: Value = serde_json::from_slice(&fs::read(dlq_job)?)?;
    assert_eq!(job["splits"], json!([]));

    #[cfg(unix)]
    {
        let outside = dir.path().join("outside.parquet");
        fs::rename(&first, &outside)?;
        std::os::unix::fs::symlink(&outside, &first)?;
        let output = worker()
            .args(["query-stage", "--output-root"])
            .arg(&root)
            .args(["--sql", "SELECT id FROM data"])
            .output()?;
        assert!(!output.status.success());
        assert!(output.stdout.is_empty());
        assert!(String::from_utf8_lossy(&output.stderr).contains("without symlinks"));
    }
    Ok(())
}

#[test]
fn stateless_worker_receipts_and_partial_resume() -> Result<()> {
    let counters = [
        ("input_tokens", 11, 23),
        ("output_tokens", 7, 13),
        ("llm_requests", 2, 5),
        ("llm_retries", 1, 3),
    ];
    let first_usage = json!({
        "qwen-openai": {"input_tokens": 11, "output_tokens": 7, "requests": 2}
    });
    let second_usage = json!({
        "qwen-openai": {"input_tokens": 5, "output_tokens": 3, "requests": 1},
        "gpt-gateway": {"input_tokens": 18, "output_tokens": 10, "requests": 4}
    });
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
    let mut receipt: Value = serde_json::from_slice(&result)?;
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
    for (name, first, _) in counters {
        assert_eq!(receipt["metrics"][name], 0);
        receipt["metrics"][name] = json!(first);
    }
    receipt["metrics"]["llm_usage_by_provider"] = first_usage.clone();
    fs::write(&receipt_file, serde_json::to_vec(&receipt)?)?;
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
    for (name, _, _) in counters {
        receipt["metrics"].as_object_mut().unwrap().remove(name);
    }
    receipt["metrics"]
        .as_object_mut()
        .unwrap()
        .remove("llm_usage_by_provider");
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
        .write_all(&serde_json::to_vec(&receipt)?)?;
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
    let mut receipt: Value = serde_json::from_slice(&run_split("1", "4")?)?;
    for (name, _, second) in counters {
        receipt["metrics"][name] = json!(second);
    }
    receipt["metrics"]["llm_usage_by_provider"] = second_usage.clone();
    fs::write(&receipt_file, serde_json::to_vec(&receipt)?)?;
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
    for (name, first, second) in counters {
        assert_eq!(manifest["splits"][0]["metrics"][name], first);
        assert_eq!(manifest["splits"][1]["metrics"][name], second);
        assert_eq!(manifest[name], first + second);
    }
    assert_eq!(
        manifest["splits"][0]["metrics"]["llm_usage_by_provider"],
        first_usage
    );
    assert_eq!(
        manifest["splits"][1]["metrics"]["llm_usage_by_provider"],
        second_usage
    );
    let usage = &manifest["llm_usage_by_provider"];
    assert_eq!(
        usage,
        &json!({
            "qwen-openai": {"input_tokens": 16, "output_tokens": 10, "requests": 3},
            "gpt-gateway": {"input_tokens": 18, "output_tokens": 10, "requests": 4}
        })
    );
    for (field, total) in [
        ("input_tokens", "input_tokens"),
        ("output_tokens", "output_tokens"),
        ("requests", "llm_requests"),
    ] {
        let sum: u64 = usage
            .as_object()
            .unwrap()
            .values()
            .map(|provider| provider[field].as_u64().unwrap())
            .sum();
        assert_eq!(manifest[total], sum);
    }
    let manifest_path = root
        .join("stages")
        .join(stage.as_str().unwrap())
        .join("STAGE_MANIFEST.json");
    assert_eq!(
        serde_json::from_slice::<Value>(&fs::read(manifest_path)?)?,
        manifest
    );
    assert_eq!(plan()?["pending_splits"], json!([]));
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
    let receipt: Value = serde_json::from_slice(&receipt)?;
    let metrics = &receipt["metrics"];
    assert_eq!(metrics["submitted_prompts"], 1);
    assert_eq!(metrics["llm_requests"], 1);
    assert_eq!(metrics["llm_retries"], 0);
    assert_eq!(metrics["input_tokens"], 0);
    assert_eq!(metrics["output_tokens"], 0);
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

#[cfg(not(feature = "ai-http"))]
#[test]
fn named_providers_require_ai_http() -> Result<()> {
    let dir = tempdir()?;
    let output = worker()
        .args(["plan", "--sql", "SELECT id FROM t", "--table", "t=unused.parquet", "--out"])
        .arg(dir.path().join("job.json"))
        .env("DF_DIST_LLM_PROVIDERS", r#"{"p":{"base_url":"https://provider.invalid/v1","api_key_env":"TEST_KEY"}}"#)
        .output()?;
    assert!(!output.status.success());
    assert!(
        String::from_utf8_lossy(&output.stderr)
            .contains("DF_DIST_LLM_PROVIDERS requires the ai-http feature")
    );
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
