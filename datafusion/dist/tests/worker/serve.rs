// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: See the NOTICE file distributed with this work for additional information regarding copyright ownership
// SPDX-FileContributor: Licensed to the Apache Software Foundation (ASF) under one or more contributor license agreements

use std::io::{BufRead, BufReader};
use std::process::{Child, ChildStdin, ChildStdout};

use super::*;

struct Daemon {
    child: Child,
    input: ChildStdin,
    output: BufReader<ChildStdout>,
}

impl Daemon {
    fn start() -> Result<Self> {
        let mut child = worker()
            .arg("serve")
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .spawn()?;
        Ok(Self {
            input: child.stdin.take().unwrap(),
            output: BufReader::new(child.stdout.take().unwrap()),
            child,
        })
    }

    fn send(&mut self, request: &Value) -> Result<Value> {
        self.send_bytes(&serde_json::to_vec(request)?)
    }

    fn send_bytes(&mut self, request: &[u8]) -> Result<Value> {
        self.input.write_all(request)?;
        self.input.write_all(b"\n")?;
        let mut line = String::new();
        assert_ne!(self.output.read_line(&mut line)?, 0);
        assert!(line.ends_with('\n'));
        Ok(serde_json::from_str(&line)?)
    }

    fn success(&mut self, request: &Value) -> Result<Value> {
        let reply = self.send(request)?;
        assert_eq!(reply["ok"], true, "{reply}");
        Ok(reply["result"].clone())
    }

    fn finish(self, close_input: bool) -> Result<()> {
        let Self {
            child,
            input,
            mut output,
        } = self;
        if close_input {
            drop(input);
            assert!(successful(child.wait_with_output()?).is_empty());
        } else {
            // Shutdown must exit even when the client keeps its write end open.
            assert!(successful(child.wait_with_output()?).is_empty());
            drop(input);
        }
        assert_eq!(output.read_line(&mut String::new())?, 0);
        Ok(())
    }
}

#[test]
fn serve_commits_two_splits_from_one_loaded_job() -> Result<()> {
    let dir = tempdir()?;
    let input = dir.path().join("input");
    let root = dir.path().join("output");
    let job_file = dir.path().join("job.json");
    input_files(&input)?;
    successful(
        worker()
            .args([
                "plan",
                "--sql",
                "SELECT id + 1 AS id FROM t",
                "--table",
                &format!("t={}", input.display()),
                "--out",
            ])
            .arg(&job_file)
            .arg("--output-root")
            .arg(&root)
            .output()?,
    );
    let job: Value = serde_json::from_slice(&fs::read(&job_file)?)?;
    fs::remove_file(&job_file)?;
    let ping = json!({"command":"ping"});
    let load = json!({"command":"load_job","job_id":"test","job":job});
    let run = |split_id, attempt_id| json!({"command":"run_split","job_id":"test","split_id":split_id,"attempt_id":attempt_id});
    let commit =
        |receipt| json!({"command":"commit_split","job_id":"test","receipt":receipt});
    let finalize = json!({"command":"finalize_stage","job_id":"test"});

    let mut runner = Daemon::start()?;
    let mut coordinator = Daemon::start()?;
    assert_eq!(runner.success(&ping)?["protocol"], "df-worker-serve-v1");
    assert_eq!(
        runner.send_bytes(b"\xff")?["error"]["code"],
        "invalid_request"
    );
    assert_eq!(
        runner.send(&json!({"command":"unknown"}))?["error"]["code"],
        "invalid_request"
    );
    assert_eq!(runner.send(&run(0, 0))?["ok"], false);
    let mut invalid = load.clone();
    invalid["job"]["schema"] = json!("invalid");
    assert_eq!(runner.send(&invalid)?["error"]["code"], "command_failed");
    invalid = load.clone();
    invalid["job"]["pending_splits"][0]["split_id"] = json!(42);
    assert_eq!(runner.send(&invalid)?["ok"], false);
    runner.success(&load)?;
    coordinator.success(&load)?;
    assert_eq!(runner.send(&load)?["ok"], false);
    let mut wrong_job = run(0, 0);
    wrong_job["job_id"] = json!("other");
    assert_eq!(runner.send(&wrong_job)?["ok"], false);
    assert_eq!(runner.send(&run(100, 0))?["ok"], false);
    let moved = dir.path().join("moved");
    fs::rename(&input, &moved)?;
    assert_eq!(runner.send(&run(0, 0))?["error"]["code"], "command_failed");
    fs::rename(&moved, &input)?;

    let first = runner.success(&run(0, 1))?;
    assert_eq!(coordinator.send(&finalize)?["ok"], false);
    let second = runner.success(&run(1, 0))?;
    assert_eq!(first["metrics"]["output_rows"], 1);
    assert_eq!(second["metrics"]["output_rows"], 1);
    coordinator.success(&commit(first))?;
    let later = runner.success(&run(0, 2))?;
    coordinator.success(&commit(later))?;
    coordinator.success(&commit(second))?;
    let manifest = coordinator.success(&finalize)?;
    assert_eq!(coordinator.success(&finalize)?, manifest);
    assert_eq!(manifest["output_rows"], 2);
    assert_eq!(manifest["splits"][0]["winning_attempt_id"], 1);
    assert_eq!(manifest["splits"][1]["winning_attempt_id"], 0);
    let stage_dir = root
        .join("stages")
        .join(manifest["plan_fingerprint"].as_str().unwrap());
    let published: Value =
        serde_json::from_slice(&fs::read(stage_dir.join("STAGE_MANIFEST.json"))?)?;
    assert_eq!(manifest, published);
    let mut ids = Vec::new();
    for split in manifest["splits"].as_array().unwrap() {
        let path = stage_dir.join(split["data_rel_path"].as_str().unwrap());
        for batch in
            ParquetRecordBatchReaderBuilder::try_new(File::open(path)?)?.build()?
        {
            let batch = batch?;
            let id = batch
                .column(0)
                .as_any()
                .downcast_ref::<Int64Array>()
                .unwrap();
            ids.extend(id.values().iter().copied());
        }
    }
    ids.sort_unstable();
    assert_eq!(ids, [8, 12]);
    runner.finish(true)?;
    coordinator.success(&json!({"command":"shutdown"}))?;
    coordinator.finish(false)
}
