// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: See the NOTICE file distributed with this work for additional information regarding copyright ownership
// SPDX-FileContributor: Licensed to the Apache Software Foundation (ASF) under one or more contributor license agreements

use std::num::NonZeroUsize;
use std::{path::PathBuf, time::SystemTime};

use datafusion::arrow::array::{Int32Array, UInt64Array};
use datafusion::arrow::compute::{concat_batches, take_record_batch};
use datafusion::datasource::listing::PartitionedFile;
use datafusion::datasource::physical_plan::{
    FileGroup, FileScanConfig, FileScanConfigBuilder,
};
use datafusion::datasource::source::DataSourceExec;
use datafusion::prelude::SessionContext;
use datafusion_dist::{
    DistCodecs, LocalWorker, ParquetSplitSink, SinkOptions,
    bind_split_to_plan_with_codecs, split_plan_for_distribution_with_codecs,
};
use datafusion_lake::{InclusiveRowRange, LakeSplitExt, LakeSplitPayload};
use lance::{BlobArrayBuilder, BlobFieldOptions, blob_field_with_options};
use lance_table_runtime::{InsertBuilder, WriteMode, WriteParams};
use object_store::{local::LocalFileSystem, path::Path as StorePath};
use sha2::{Digest, Sha256};
use url::Url;

use super::*;

const IDS: [i32; 5] = [42, 7, 81, 15, 9];

async fn fixture(path: &Path, external: &Path) -> Result<()> {
    let mut blobs = BlobArrayBuilder::new(IDS.len());
    fs::write(external, b"external blob")?;
    blobs.push_uri(Url::from_file_path(external).unwrap().as_str())?;
    for id in IDS.into_iter().skip(1) {
        blobs.push_bytes(vec![u8::try_from(id)?; 32])?;
    }
    let field = blob_field_with_options(
        "payload",
        false,
        BlobFieldOptions::default()
            .with_inline_size_threshold(1)
            .with_dedicated_size_threshold(NonZeroUsize::new(8).unwrap()),
    );
    let batch = RecordBatch::try_new(
        Arc::new(Schema::new(vec![
            Field::new("id", DataType::Int32, false),
            field,
        ])),
        vec![Arc::new(Int32Array::from(IDS.to_vec())), blobs.finish()?],
    )?;
    InsertBuilder::new(path.to_str().unwrap())
        .with_params(&WriteParams {
            mode: WriteMode::Create,
            max_rows_per_file: 3,
            allow_external_blob_outside_bases: true,
            ..Default::default()
        })
        .execute(vec![batch])
        .await?;
    Ok(())
}

fn state(path: &Path) -> Result<(u64, SystemTime, [u8; 32])> {
    let metadata = fs::metadata(path)?;
    Ok((
        metadata.len(),
        metadata.modified()?,
        Sha256::digest(fs::read(path)?).into(),
    ))
}

fn list_files(dir: &Path, files: &mut Vec<PathBuf>) -> Result<()> {
    for entry in fs::read_dir(dir)? {
        let entry = entry?;
        if entry.file_type()?.is_dir() {
            list_files(&entry.path(), files)?;
        } else {
            files.push(entry.path());
        }
    }
    Ok(())
}

fn run_local(dataset: &Path, root: &Path, sql: &str, flags: &[&str]) -> Result<Value> {
    let output = worker()
        .args(["run-local", "--sql", sql, "--lance-table"])
        .arg(format!("source={}", dataset.display()))
        .args(flags)
        .arg("--output-root")
        .arg(root)
        .env("DF_DIST_LLM_BACKEND", "mock")
        .env_remove("DF_DIST_LLM_PROVIDER")
        .env_remove("DF_DIST_LLM_PROVIDERS")
        .output()?;
    Ok(serde_json::from_slice(&successful(output))?)
}

fn backfill(dataset: &Path, root: &Path) -> Result<Output> {
    Ok(worker()
        .args(["backfill-lance", "--dataset"])
        .arg(dataset)
        .arg("--stage-output-root")
        .arg(root)
        .output()?)
}

fn assert_failure(output: &Output, message: &str) {
    assert!(!output.status.success(), "{output:?}");
    assert!(
        String::from_utf8_lossy(&output.stderr).contains(message),
        "{output:?}"
    );
}

#[tokio::test]
async fn lake_worker_backfills_only_winners_without_rewriting_data_or_blobs() -> Result<()>
{
    let dir = tempdir()?;
    let dataset = dir.path().join("source.lance");
    let external = dir.path().join("external.blob");
    fixture(&dataset, &external).await?;
    let mut files = vec![];
    list_files(&dataset.join("data"), &mut files)?;
    assert!(
        files
            .iter()
            .any(|p| p.extension().is_some_and(|ext| ext == "blob"))
    );
    files.push(external.clone());
    let before = files.iter().map(|p| state(p)).collect::<Result<Vec<_>>>()?;

    let sql = "SELECT id * 7 AS score FROM source";
    let plan = |name: &str| -> Result<Value> {
        let path = dir.path().join(name);
        successful(
            worker()
                .args([
                    "plan",
                    "--sql",
                    "SELECT payload FROM source",
                    "--lance-table",
                ])
                .arg(format!("source={}", dataset.display()))
                .arg("--out")
                .arg(&path)
                .output()?,
        );
        Ok(serde_json::from_slice(&fs::read(path)?)?)
    };
    let first = plan("first.json")?;
    let second = plan("second.json")?;
    assert_eq!(first["stage"], second["stage"]);
    assert_eq!(first["splits"], second["splits"]);

    // Blob reference scans succeed even when the payload files are unavailable.
    let mut moved = Vec::new();
    for path in files
        .iter()
        .filter(|path| path.extension().is_some_and(|ext| ext == "blob"))
    {
        let backup = dir.path().join(format!("unavailable-{}", moved.len()));
        fs::rename(path, &backup)?;
        moved.push((path, backup));
    }
    let refs = dir.path().join("refs");
    let manifest = run_local(&dataset, &refs, "SELECT payload FROM source", &[])?;
    assert_eq!(manifest["output_rows"], IDS.len());
    assert_eq!(manifest["dlq_rows"], 0);
    assert_eq!(
        query_stage(&refs, "SELECT COUNT(*) AS n FROM dlq")?,
        [json!({"n": 0})]
    );
    let rows = query_stage(&refs, "SELECT payload FROM data")?;
    assert!(
        rows.iter().any(|row| row["payload"]["uri"]
            == Url::from_file_path(&external).unwrap().as_str())
    );
    for (path, backup) in moved {
        fs::rename(backup, path)?;
    }

    let output = dir.path().join("out");
    let older = dir.path().join("older");
    run_local(
        &dataset,
        &older,
        "SELECT id * 2 AS old_score FROM source",
        &[],
    )?;
    let manifest = run_local(&dataset, &output, sql, &[])?;
    assert_eq!(manifest["splits"].as_array().unwrap().len(), 2);
    let stage = output
        .join("stages")
        .join(manifest["plan_fingerprint"].as_str().unwrap());
    let stale = stage.join("attempts/split-0000000000000000-attempt-0001");
    fs::create_dir(&stale)?;
    fs::write(stale.join("data.parquet"), b"uncommitted bytes")?;
    let winner = stage.join(manifest["splits"][0]["data_rel_path"].as_str().unwrap());
    let original = fs::read(&winner)?;
    let mut corrupted = original.clone();
    corrupted[10] ^= 1;
    fs::write(&winner, corrupted)?;
    assert_failure(&backfill(&dataset, &output)?, "checksum");
    fs::write(winner, original)?;

    let response: Value =
        serde_json::from_slice(&successful(backfill(&dataset, &output)?))?;
    assert_eq!(response["dataset_version"], 2);
    for (path, before) in files.iter().zip(before) {
        assert_eq!(state(path)?, before, "{}", path.display());
    }
    let updated = run_local(
        &dataset,
        &dir.path().join("updated"),
        "SELECT id, score FROM source",
        &[],
    )?;
    assert_eq!(updated["output_rows"], IDS.len());
    let rows = query_stage(
        &dir.path().join("updated"),
        "SELECT id, score FROM data ORDER BY id",
    )?;
    let mut ids = IDS;
    ids.sort_unstable();
    assert_eq!(rows, ids.map(|id| json!({"id": id, "score": id * 7})));
    assert_failure(&backfill(&dataset, &older)?, "snapshot is stale");
    Ok(())
}

#[tokio::test]
async fn lake_replay_output_cannot_be_used_for_backfill() -> Result<()> {
    let dir = tempdir()?;
    let dataset = dir.path().join("source.lance");
    fixture(&dataset, &dir.path().join("external.blob")).await?;
    let base = dir.path().join("base");
    let replayed = dir.path().join("replayed");
    let sql = "SELECT id, id * 7 AS score FROM source";
    let manifest = run_local(&dataset, &base, sql, &["--row-key", "id"])?;
    let splits = manifest["splits"].as_array().unwrap();
    assert_eq!(splits.len(), 2);
    assert!(
        splits
            .iter()
            .all(|split| split.get("lake_source").is_some())
    );

    let output = worker()
        .args(["replay-dlq", "--base-output-root"])
        .arg(&base)
        .arg("--output-root")
        .arg(&replayed)
        .args(["--sql", sql, "--row-key", "id"])
        .output()?;
    let summary: Value = serde_json::from_slice(&successful(output))?;
    let splits = summary["splits"].as_array().unwrap();
    assert_eq!(splits.len(), 2);
    assert!(
        splits
            .iter()
            .all(|split| split.get("lake_source").is_none())
    );
    assert_eq!(summary["output_rows"], IDS.len());
    let select = "SELECT id, score FROM data ORDER BY id";
    assert_eq!(query_stage(&replayed, select)?, query_stage(&base, select)?);
    assert_failure(
        &backfill(&dataset, &replayed)?,
        "requires Lake split provenance",
    );
    Ok(())
}

#[tokio::test]
async fn lake_backfill_validates_coverage_and_keeps_only_new_columns() -> Result<()> {
    let dir = tempdir()?;
    let dataset = dir.path().join("source.lance");
    fixture(&dataset, &dir.path().join("external.blob")).await?;
    let other = dir.path().join("other.lance");
    fixture(&other, &dir.path().join("other.blob")).await?;
    let root = dir.path().join("out");
    run_local(
        &dataset,
        &root,
        "SELECT id, id * 3 AS score FROM source WHERE id > 30",
        &["--row-key", "id"],
    )?;
    assert_failure(&backfill(&dataset, &root)?, "physical rows");
    assert_failure(&backfill(&other, &root)?, "source dataset");

    let keyed = dir.path().join("keyed");
    let computed = dir.path().join("computed-key");
    run_local(
        &dataset,
        &computed,
        "SELECT -id AS row_key, id * 3 AS score FROM source",
        &["--row-key", "row_key"],
    )?;
    assert_failure(
        &backfill(&dataset, &computed)?,
        "unchanged source --row-key",
    );

    let ctx = SessionContext::new();
    ctx.register_table(
        "source",
        Arc::new(
            datafusion_lake::LanceReadOnlyTable::open(
                dataset.to_str().unwrap(),
                None,
                &Default::default(),
            )
            .await?,
        ),
    )?;
    let plan = ctx
        .sql("SELECT id AS row_key, id * 3 AS score FROM source")
        .await?
        .create_physical_plan()
        .await?;
    let codecs = DistCodecs::default().with_lake()?;
    let (stage, splits) = split_plan_for_distribution_with_codecs(plan, 0, &codecs)?;
    let sink = ParquetSplitSink::try_new(
        Arc::new(LocalFileSystem::new()),
        &StorePath::from_absolute_path(&keyed)?,
        &stage,
        &SinkOptions {
            row_key: Some("row_key".into()),
        },
    )?;
    let runner = LocalWorker::new().with_codecs(codecs);
    for split in &splits {
        let output = runner.execute_split(&stage, split).await?;
        let rows = concat_batches(&stage.output_schema, &output.batches)?;
        let order = UInt64Array::from_iter_values((0..rows.num_rows() as u64).rev());
        let rows = take_record_batch(&rows, &order)?;
        let attempt = sink.write_attempt(split, &[rows]).await?;
        sink.commit_split(split, attempt).await?;
    }
    sink.finalize_stage(&splits).await?;
    successful(backfill(&dataset, &keyed)?);
    let updated = lance::Dataset::open(dataset.to_str().unwrap()).await?;
    assert!(updated.schema().field("score").is_some());
    assert!(updated.schema().field("row_key").is_none());
    let verified = dir.path().join("verified");
    run_local(&dataset, &verified, "SELECT id, score FROM source", &[])?;
    assert_eq!(
        query_stage(
            &verified,
            "SELECT COUNT(*) AS n FROM data WHERE score <> 3 * id"
        )?,
        [json!({"n": 0})]
    );
    Ok(())
}

#[cfg(feature = "ai")]
#[tokio::test]
async fn lake_vision_resolves_blobs_and_rejects_backfill_with_dlq() -> Result<()> {
    let dir = tempdir()?;
    let dataset = dir.path().join("source.lance");
    let external = dir.path().join("external.blob");
    fixture(&dataset, &external).await?;
    let sql = r#"SELECT id, vlm(CAST(id AS VARCHAR), payload, 'vision',
                     '{"on_error":"envelope","max_media_bytes":64}') AS label FROM source"#;
    let broken = dir.path().join("broken");
    let payload = fs::read(&external)?;
    fs::remove_file(&external)?;
    let manifest = run_local(&dataset, &broken, sql, &["--row-key", "id"])?;
    assert_eq!(manifest["dlq_rows"], 1);
    assert_failure(&backfill(&dataset, &broken)?, "zero DLQ rows");

    fs::write(&external, payload)?;
    let success = dir.path().join("success");
    let manifest = run_local(&dataset, &success, sql, &["--row-key", "id"])?;
    assert_eq!(manifest["output_rows"], IDS.len());
    assert_eq!(manifest["dlq_rows"], 0);
    assert_eq!(manifest["llm_requests"], IDS.len());
    successful(backfill(&dataset, &success)?);
    let updated = dir.path().join("updated");
    run_local(&dataset, &updated, "SELECT id, label FROM source", &[])?;
    let mut ids = IDS;
    ids.sort_unstable();
    assert_eq!(
        query_stage(&updated, "SELECT id, label FROM data ORDER BY id")?,
        ids.map(|id| json!({"id": id, "label": format!("mock[vision]: {id}")}))
    );
    Ok(())
}

#[tokio::test]
async fn lake_row_ranges_replay_in_a_fresh_worker() -> Result<()> {
    let dir = tempdir()?;
    let dataset = dir.path().join("source.lance");
    fixture(&dataset, &dir.path().join("external.blob")).await?;
    let ctx = SessionContext::new();
    ctx.register_table(
        "source",
        Arc::new(
            datafusion_lake::LanceReadOnlyTable::open(
                dataset.to_str().unwrap(),
                None,
                &Default::default(),
            )
            .await?,
        ),
    )?;
    let original = ctx
        .sql("SELECT id FROM source")
        .await?
        .create_physical_plan()
        .await?;
    let scan = original
        .downcast_ref::<DataSourceExec>()
        .unwrap()
        .data_source()
        .downcast_ref::<FileScanConfig>()
        .unwrap();
    let mut split = scan.file_groups[0].files()[0]
        .extension::<LakeSplitExt>()
        .unwrap()
        .clone();
    let LakeSplitPayload::Lance { row_ranges, .. } = &mut split.payload else {
        unreachable!()
    };
    *row_ranges = Some(vec![InclusiveRowRange { from: 1, to: 1 }]);
    split.split_id = hex::encode(Sha256::digest(serde_json::to_vec(&split.payload)?));
    let file = PartitionedFile::new(split.split_id.clone(), 0).with_extension(split);
    let selected = DataSourceExec::from_data_source(
        FileScanConfigBuilder::from(scan.clone())
            .with_file_groups(vec![FileGroup::new(vec![file])])
            .build(),
    );
    let codecs = DistCodecs::default().with_lake()?;
    let (stage, splits) = split_plan_for_distribution_with_codecs(selected, 7, &codecs)?;
    assert_eq!(splits.len(), 1);
    assert!(
        bind_split_to_plan_with_codecs(&stage, &splits[0], &ctx, &DistCodecs::default())
            .is_err()
    );
    let worker = LocalWorker::new().with_codecs(codecs);
    let output = worker.execute_split(&stage, &splits[0]).await?;
    let batch = concat_batches(&stage.output_schema, &output.batches)?;
    assert_eq!(
        batch
            .column(0)
            .as_any()
            .downcast_ref::<Int32Array>()
            .unwrap()
            .values()
            .as_ref(),
        &[IDS[1]]
    );
    Ok(())
}
