// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: See the NOTICE file distributed with this work for additional information regarding copyright ownership
// SPDX-FileContributor: Licensed to the Apache Software Foundation (ASF) under one or more contributor license agreements

use super::*;

fn read_summary(output: Output) -> Result<Value> {
    Ok(serde_json::from_slice(&successful(output))?)
}

#[test]
fn replay_without_dlq_skips_source_and_sql() -> Result<()> {
    let dir = tempdir()?;
    let input = dir.path().join("input");
    input_files(&input)?;
    let base = dir.path().join("base");
    let out = dir.path().join("replay % data");
    let upstream = create_stage(&base, &input, "SELECT id FROM t")?;

    let run = || replay_command(&base, &out, "SELECT absent FROM missing").output();
    let summary = read_summary(run()?)?;
    assert_eq!(summary["prev_data_rows"], 2);
    assert_eq!(summary["new_data_rows"], 0);
    assert_eq!(summary["remaining_dlq_rows"], 0);
    assert_eq!(summary["llm_requests"], 0);
    assert_eq!(summary["llm_usage_by_provider"], json!({}));
    for (old, new) in upstream["splits"]
        .as_array()
        .unwrap()
        .iter()
        .zip(summary["splits"].as_array().unwrap())
    {
        assert_eq!(old["data_sha256"], new["data_sha256"]);
    }
    assert_eq!(summary, read_summary(run()?)?);
    fs::remove_dir_all(base)?;
    assert_eq!(
        query_stage(&out, "SELECT id FROM data ORDER BY id")?,
        [json!({"id": 7}), json!({"id": 11})]
    );
    assert!(query_stage(&out, "SELECT * FROM dlq")?.is_empty());
    Ok(())
}

fn create_stage(base: &Path, source: &Path, sql: &str) -> Result<Value> {
    read_summary(
        worker()
            .args(["run-local", "--sql", sql, "--table"])
            .arg(format!("t={}", source.display()))
            .args(["--row-key", "id", "--output-root"])
            .arg(base)
            .env("DF_DIST_LLM_BACKEND", "mock")
            .env_remove("DF_DIST_LLM_PROVIDERS")
            .env_remove("DF_DIST_LLM_PROVIDER")
            .output()?,
    )
}

fn replay_command(base: &Path, out: &Path, sql: &str) -> Command {
    let mut command = worker();
    command
        .args(["replay-dlq", "--base-output-root"])
        .arg(base)
        .arg("--output-root")
        .arg(out)
        .args(["--row-key", "id", "--sql", sql])
        .env("DF_DIST_LLM_BACKEND", "mock")
        .env_remove("DF_DIST_LLM_PROVIDERS")
        .env_remove("DF_DIST_LLM_PROVIDER");
    command
}

#[cfg(feature = "ai")]
mod ai {
    use datafusion::arrow::array::StringArray;
    use tempfile::TempDir;
    use url::Url;

    use super::*;

    const BASE_SQL: &str = r#"SELECT id,
        vlm(prompt, old_uri, 'base', '{"on_error":"envelope"}') AS label,
        vlm(prompt || '!', old_uri, 'base', '{"on_error":"envelope"}') AS detail
        FROM t"#;
    const RETRY_SQL: &str = r#"SELECT id,
        vlm(prompt, new_uri, 'retry', '{"on_error":"envelope"}') AS label,
        vlm(prompt || '!', new_uri, 'retry', '{"on_error":"envelope"}') AS detail
        FROM t"#;

    struct Fixture {
        _dir: TempDir,
        input: std::path::PathBuf,
        base: std::path::PathBuf,
        out: std::path::PathBuf,
        manifest: Value,
    }

    impl Fixture {
        fn new(ids: &[Option<i64>], base_sql: &str) -> Result<Self> {
            let dir = tempdir()?;
            let input = dir.path().join("input.parquet");
            let base = dir.path().join("base");
            let out = dir.path().join("out");
            let image = dir.path().join("image.ppm");
            fs::write(&image, b"P6\n1 1\n255\n\xff\0\0")?;
            let image = Url::from_file_path(image).unwrap();
            let missing = Url::from_file_path(dir.path().join("missing.ppm")).unwrap();
            let schema = Arc::new(Schema::new(vec![
                Field::new("id", DataType::Int64, true),
                Field::new("prompt", DataType::Utf8, false),
                Field::new("old_uri", DataType::Utf8, false),
                Field::new("new_uri", DataType::Utf8, false),
            ]));
            let uri = |success: i64| {
                Arc::new(StringArray::from_iter_values(ids.iter().map(|id| {
                    if *id == Some(success) {
                        image.as_str()
                    } else {
                        missing.as_str()
                    }
                })))
            };
            let batch = RecordBatch::try_new(
                Arc::clone(&schema),
                vec![
                    Arc::new(Int64Array::from(ids.to_vec())),
                    Arc::new(StringArray::from_iter_values(
                        ids.iter()
                            .enumerate()
                            .map(|(i, _)| format!("row {}", i + 1)),
                    )),
                    uri(1),
                    uri(2),
                ],
            )?;
            let mut writer = ArrowWriter::try_new(File::create(&input)?, schema, None)?;
            writer.write(&batch)?;
            writer.close()?;
            let manifest = create_stage(&base, &input, base_sql)?;
            Ok(Self {
                _dir: dir,
                input,
                base,
                out,
                manifest,
            })
        }

        fn run(&self, sql: &str) -> std::io::Result<Output> {
            replay_command(&self.base, &self.out, sql)
                .arg("--table")
                .arg(format!("t={}", self.input.display()))
                .args(["--batch-size", "1"])
                .output()
        }

        fn manifest_path(&self) -> std::path::PathBuf {
            self.base
                .join("stages")
                .join(self.manifest["plan_fingerprint"].as_str().unwrap())
                .join("STAGE_MANIFEST.json")
        }

        fn assert_rejected(&self, sql: &str, detail: &str) -> Result<()> {
            let result = self.run(sql)?;
            let stderr = String::from_utf8(result.stderr)?;
            assert!(!result.status.success(), "{stderr}");
            assert!(stderr.contains(detail), "{stderr}");
            assert!(!self.out.join("STAGE_MANIFEST.json").exists());
            Ok(())
        }
    }

    #[test]
    fn replay_deduplicates_cells_and_keeps_keys_that_still_fail() -> Result<()> {
        let fixture = Fixture::new(&[Some(1), Some(2), Some(3)], BASE_SQL)?;
        assert_eq!(fixture.manifest["output_rows"], 1);
        assert_eq!(fixture.manifest["dlq_rows"], 4);
        assert_eq!(fixture.manifest["llm_requests"], 2);
        let summary = read_summary(fixture.run(RETRY_SQL)?)?;
        assert_eq!(summary["prev_data_rows"], 1);
        assert_eq!(summary["new_data_rows"], 1);
        assert_eq!(summary["remaining_dlq_rows"], 1);
        assert_eq!(summary["dlq_rows"], 2);
        assert_eq!(summary["llm_requests"], 2);
        assert_eq!(summary["llm_usage_by_provider"]["default"]["requests"], 2);
        assert_eq!(
            query_stage(&fixture.out, "SELECT id, label FROM data ORDER BY id")?,
            [
                json!({"id": 1, "label": "mock[base]: row 1"}),
                json!({"id": 2, "label": "mock[retry]: row 2"})
            ],
        );
        assert_eq!(
            query_stage(&fixture.out, "SELECT DISTINCT row_key FROM dlq")?,
            [json!({"row_key": 3})],
        );
        Ok(())
    }

    #[test]
    fn replay_with_no_failures_reports_zero_mock_requests() -> Result<()> {
        let fixture = Fixture::new(&[Some(1)], BASE_SQL)?;
        assert_eq!(fixture.manifest["llm_requests"], 2);
        fs::remove_file(&fixture.input)?;
        let summary = read_summary(fixture.run(RETRY_SQL)?)?;
        assert_eq!(summary["llm_requests"], 0);
        assert_eq!(summary["llm_usage_by_provider"], json!({}));
        assert_eq!(
            query_stage(&fixture.out, "SELECT id FROM data")?,
            [json!({"id": 1})]
        );
        Ok(())
    }

    #[test]
    fn replay_rejects_duplicate_missing_or_schema_drift_in_output() -> Result<()> {
        let fixture = Fixture::new(&[Some(1), Some(2), Some(3)], BASE_SQL)?;
        // Both retry rows must produce id=2, even when one stays in the DLQ.
        let duplicate = RETRY_SQL.replace(
            "SELECT id,",
            "SELECT CASE WHEN id = 3 THEN 2 ELSE id END AS id,",
        );
        fixture.assert_rejected(&duplicate, "duplicate row key")?;
        fixture.assert_rejected(&format!("{RETRY_SQL} WHERE id = 2"), "conservation")?;
        let wrong_keys = RETRY_SQL.replace("SELECT id,", "SELECT id + 10 AS id,");
        fixture.assert_rejected(&wrong_keys, "conservation")?;
        let null_key =
            RETRY_SQL.replace("SELECT id,", "SELECT CAST(NULL AS BIGINT) AS id,");
        fixture.assert_rejected(&null_key, "non-null")?;
        let drift = RETRY_SQL.replace(" AS detail", " AS renamed");
        fixture.assert_rejected(&drift, "schema")?;
        Ok(())
    }

    #[test]
    fn replay_rejects_inconsistent_base_and_invalid_keys() -> Result<()> {
        let fixture = Fixture::new(&[Some(1), Some(2)], BASE_SQL)?;
        let mut manifest = fixture.manifest.clone();
        manifest["output_rows"] = 2.into();
        manifest["splits"][0]["output_rows"] = 2.into();
        fs::write(fixture.manifest_path(), serde_json::to_vec(&manifest)?)?;
        fixture.assert_rejected(RETRY_SQL, "row count")?;

        let null = Fixture::new(&[None], "SELECT id FROM t")?;
        null.assert_rejected(RETRY_SQL, "non-null")?;
        let empty = Fixture::new(&[Some(1)], "SELECT '' AS id FROM t")?;
        empty.assert_rejected(RETRY_SQL, "non-empty")?;
        let duplicate = Fixture::new(&[Some(1), Some(1)], "SELECT id FROM t")?;
        duplicate.assert_rejected(RETRY_SQL, "duplicate row key")?;
        let duplicate_failed = Fixture::new(&[Some(2), Some(2)], BASE_SQL)?;
        duplicate_failed.assert_rejected(RETRY_SQL, "duplicate")?;
        Ok(())
    }

    #[test]
    fn replay_requires_exact_source_coverage_before_planning_sql() -> Result<()> {
        let fixture = Fixture::new(&[Some(1), Some(2)], BASE_SQL)?;
        let batch =
            ParquetRecordBatchReaderBuilder::try_new(File::open(&fixture.input)?)?
                .build()?
                .next()
                .unwrap()?;
        let schema = batch.schema();
        let rewrite = |rows: &[RecordBatch]| -> Result<()> {
            let mut writer = ArrowWriter::try_new(
                File::create(&fixture.input)?,
                schema.clone(),
                None,
            )?;
            for row in rows {
                writer.write(row)?;
            }
            writer.close()?;
            Ok(())
        };
        rewrite(&[batch.slice(0, 1)])?;
        fixture.assert_rejected("invalid replay SQL", "missing failed row keys")?;
        rewrite(&[batch.clone(), batch.slice(1, 1)])?;
        fixture.assert_rejected("invalid replay SQL", "duplicate row key")?;
        Ok(())
    }

    #[test]
    fn replay_with_all_keys_still_failed_can_be_replayed_again() -> Result<()> {
        let fixture = Fixture::new(&[Some(1), Some(2)], BASE_SQL)?;
        let first = read_summary(fixture.run(BASE_SQL)?)?;
        assert_eq!(first["output_rows"], 1);
        assert_eq!(first["remaining_dlq_rows"], 1);
        assert_eq!(first["llm_requests"], 0);
        let second = fixture.out.with_file_name("second");
        let output = replay_command(&fixture.out, &second, RETRY_SQL)
            .arg("--table")
            .arg(format!("t={}", fixture.input.display()))
            .output()?;
        let summary = read_summary(output)?;
        assert_eq!(summary["output_rows"], 2);
        assert_eq!(summary["remaining_dlq_rows"], 0);
        assert_eq!(summary["llm_requests"], 2);
        assert!(query_stage(&second, "SELECT * FROM dlq")?.is_empty());
        Ok(())
    }

    #[test]
    fn concurrent_replays_return_one_complete_winner() -> Result<()> {
        let fixture = Fixture::new(&[Some(1), Some(2)], BASE_SQL)?;
        let gate = std::sync::Barrier::new(2);
        let run = || {
            gate.wait();
            fixture.run(RETRY_SQL)
        };
        let (left, right) = std::thread::scope(|scope| {
            let left = scope.spawn(run);
            let right = run();
            (left.join().unwrap(), right)
        });
        let result = read_summary(left?)?;
        assert_eq!(result, read_summary(right?)?);
        assert_eq!(result["output_rows"], 2);
        assert_eq!(result["remaining_dlq_rows"], 0);
        assert_eq!(result["llm_requests"], 2);
        assert_eq!(
            query_stage(&fixture.out, "SELECT id FROM data ORDER BY id")?,
            [json!({"id": 1}), json!({"id": 2})],
        );
        Ok(())
    }
}
