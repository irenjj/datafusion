// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: See the NOTICE file distributed with this work for additional information regarding copyright ownership
// SPDX-FileContributor: Licensed to the Apache Software Foundation (ASF) under one or more contributor license agreements

use datafusion::arrow::array::{ArrayRef, Int64Array};
use datafusion::arrow::datatypes::{DataType, Field, Schema};
use datafusion::arrow::record_batch::RecordBatch;
use datafusion::assert_batches_eq;
use object_store::memory::InMemory;
use serde_json::{Value, json};

use crate::fingerprint::{plan_fingerprint, split_fingerprint};
use crate::{ParquetSplitSink, SinkOptions, Split, StagePlan};

use super::*;

struct Fixture {
    ctx: SessionContext,
    url: ObjectStoreUrl,
    store: Arc<dyn ObjectStore>,
    stage: StagePlan,
    sink: ParquetSplitSink,
    output_root: Path,
    stage_root: Path,
}

impl Fixture {
    fn new(schema: SchemaRef, options: &SinkOptions) -> Result<Self> {
        let ctx = SessionContext::new();
        let url = ObjectStoreUrl::parse("s3://stage-bucket")?;
        let store: Arc<dyn ObjectStore> = Arc::new(InMemory::new());
        ctx.register_object_store(url.as_ref(), Arc::clone(&store));
        let plan_bytes = vec![1, 2, 3];
        let stage = StagePlan {
            stage_id: 0,
            plan_fingerprint: plan_fingerprint(&plan_bytes, &schema)?,
            plan_bytes,
            output_schema: schema,
        };
        let output_root = Path::from("job/output");
        let stage_root = output_root
            .clone()
            .join("stages")
            .join(stage.plan_fingerprint.as_str());
        let sink =
            ParquetSplitSink::try_new(Arc::clone(&store), &output_root, &stage, options)?;
        Ok(Self {
            ctx,
            url,
            store,
            stage,
            sink,
            output_root,
            stage_root,
        })
    }

    fn split(&self, split_id: u64, attempt_id: u32) -> Split {
        let file_proto = split_id.to_le_bytes().to_vec();
        Split {
            split_id,
            attempt_id,
            fingerprint: split_fingerprint(&self.stage.plan_fingerprint, &file_proto),
            file_proto,
        }
    }

    async fn commit(&self, split: &Split, batch: RecordBatch) -> Result<()> {
        let receipt = self.sink.write_attempt(split, &[batch]).await?;
        self.sink.commit_split(split, receipt).await
    }

    async fn reader(&self) -> Result<StageOutputReader> {
        StageOutputReader::open(&self.ctx, self.url.clone(), &self.output_root).await
    }

    async fn put_manifest(&self, manifest: &Value) -> Result<()> {
        self.store
            .put(
                &self.stage_root.clone().join(MANIFEST_FILE),
                serde_json::to_vec(manifest).unwrap().into(),
            )
            .await?;
        Ok(())
    }

    async fn plain(values: Vec<i64>) -> Result<(Self, Value)> {
        let schema = Arc::new(Schema::new(vec![
            Field::new("id", DataType::Int64, false),
            Field::new("payload", DataType::Int64, false),
        ]));
        let payload: ArrayRef = Arc::new(Int64Array::from_iter_values(
            values.iter().map(|id| id * 10),
        ));
        let ids: ArrayRef = Arc::new(Int64Array::from(values));
        let fixture = Self::new(Arc::clone(&schema), &SinkOptions::default())?;
        let splits = [fixture.split(2, 1), fixture.split(5, 2)];
        for split in &splits {
            fixture
                .commit(
                    split,
                    RecordBatch::try_new(
                        Arc::clone(&schema),
                        vec![Arc::clone(&ids), Arc::clone(&payload)],
                    )?,
                )
                .await?;
        }
        let manifest =
            serde_json::to_value(fixture.sink.finalize_stage(&splits).await?).unwrap();
        Ok((fixture, manifest))
    }

    async fn register_tables(&self) -> Result<()> {
        let reader = self.reader().await?;
        self.ctx.register_table("data", reader.data_table())?;
        self.ctx.register_table("dlq", reader.dlq_table())?;
        Ok(())
    }
}

#[tokio::test]
async fn rejects_invalid_manifest_identity_inventory_and_paths() -> Result<()> {
    let (fixture, valid) = Fixture::plain(vec![1]).await?;
    for (pointer, value, expected) in [
        (
            "/schema",
            json!("df-stage-manifest-v2"),
            "Unsupported stage manifest schema",
        ),
        (
            "/plan_fingerprint",
            json!(""),
            "Invalid stage manifest plan fingerprint",
        ),
        (
            "/plan_fingerprint",
            json!("9".repeat(64)),
            "Stage directory does not match",
        ),
        (
            "/splits/0/plan_fingerprint",
            json!("9".repeat(64)),
            "Invalid manifest checkpoint identity",
        ),
        (
            "/splits/0/split_fingerprint",
            json!("../other"),
            "Invalid manifest checkpoint identity",
        ),
        (
            "/splits/1/split_fingerprint",
            valid["splits"][0]["split_fingerprint"].clone(),
            "Invalid manifest checkpoint identity",
        ),
        ("/splits/1/split_id", json!(2), "strictly increasing"),
        ("/splits/1/split_id", json!(1), "strictly increasing"),
        (
            "/splits/0/data_rel_path",
            json!("../data.parquet"),
            "Invalid checkpoint files",
        ),
        (
            "/splits/0/data_rel_path",
            json!("/attempts/split-0000000000000002-attempt-0001/data.parquet"),
            "Invalid checkpoint files",
        ),
        (
            "/splits/0/data_rel_path",
            json!("attempts/split-0000000000000002-attempt-0000/data.parquet"),
            "Invalid checkpoint files",
        ),
        ("/splits/0/dlq_rows", json!(1), "Invalid checkpoint files"),
        (
            "/splits/0/dlq_rel_path",
            json!("attempts/split-0000000000000002-attempt-0001/dlq.parquet"),
            "Invalid checkpoint files",
        ),
        ("/splits/0/data_bytes", json!(1), "data size mismatch"),
        ("/output_rows", json!(999), "row totals do not match"),
    ] {
        let mut invalid = valid.clone();
        *invalid.pointer_mut(pointer).unwrap() = value;
        fixture.put_manifest(&invalid).await?;
        let message = fixture.reader().await.unwrap_err().to_string();
        assert!(message.contains(expected), "{pointer}: {message}");
    }
    fixture.put_manifest(&valid).await?;
    fixture
        .store
        .delete(&stage_path(
            &fixture.stage_root,
            valid["splits"][0]["data_rel_path"].as_str().unwrap(),
        ))
        .await?;
    let missing = fixture.reader().await.unwrap_err().to_string();
    assert!(missing.contains("data.parquet"), "{missing}");
    assert!(missing.to_lowercase().contains("not found"), "{missing}");
    Ok(())
}

#[tokio::test]
async fn output_root_requires_a_unique_finalized_stage() -> Result<()> {
    let (fixture, valid) = Fixture::plain(vec![1]).await?;
    let other = fixture
        .output_root
        .clone()
        .join("stages")
        .join("9".repeat(64));
    fixture
        .store
        .put(&other.clone().join("unfinished"), vec![1].into())
        .await?;
    fixture.reader().await?;
    fixture
        .store
        .put(
            &other.join(MANIFEST_FILE),
            serde_json::to_vec(&valid).unwrap().into(),
        )
        .await?;
    let message = fixture.reader().await.unwrap_err().to_string();
    assert!(message.contains("Multiple finalized stages"), "{message}");
    StageOutputReader::open(&fixture.ctx, fixture.url.clone(), &fixture.stage_root)
        .await?;
    Ok(())
}

#[tokio::test]
async fn empty_data_and_dlq_tables_keep_the_data_footer_schema() -> Result<()> {
    let (fixture, manifest) = Fixture::plain(vec![]).await?;
    assert_eq!(manifest["output_rows"], 0);
    assert_eq!(manifest["dlq_rows"], 0);
    fixture.register_tables().await?;
    for table in ["data", "dlq"] {
        let projection = fixture.ctx.sql(&format!("SELECT id FROM {table}")).await?;
        assert_eq!(*projection.schema().field(0).data_type(), DataType::Int64);
        assert_eq!(projection.count().await?, 0);
    }
    Ok(())
}

#[tokio::test]
async fn multi_split_scan_projects_limits_and_keeps_residual_filters() -> Result<()> {
    let (fixture, _) = Fixture::plain(vec![1, 2]).await?;
    let reader = fixture.reader().await?;
    let table = reader.data_table();
    let state = fixture.ctx.state();
    let plan = table.scan(&state, Some(&[0]), &[], Some(3)).await?;
    assert_eq!(plan.schema().fields().len(), 1);
    assert_eq!(plan.schema().field(0).name(), "id");
    let rows = datafusion::physical_plan::collect(plan, fixture.ctx.task_ctx()).await?;
    assert_eq!(rows.iter().map(RecordBatch::num_rows).sum::<usize>(), 3);
    assert!(
        table
            .scan(&state, None, &[datafusion::logical_expr::lit(true)], None)
            .await
            .is_err()
    );
    fixture.ctx.register_table("data", table)?;
    let matches = fixture
        .ctx
        .sql("SELECT id FROM data WHERE id = 2 LIMIT 1")
        .await?
        .collect()
        .await?;
    assert_batches_eq!(
        ["+----+", "| id |", "+----+", "| 2  |", "+----+",],
        &matches
    );
    Ok(())
}

#[cfg(feature = "ai")]
#[tokio::test]
async fn s3_store_reads_only_winning_data_and_dlq() -> Result<()> {
    use datafusion::arrow::array::{BooleanArray, StringArray, StructArray};
    use datafusion_ai::AiErrorType;

    let envelope = AiErrorType::new(DataType::Utf8).field("result", true)?;
    let DataType::Struct(fields) = envelope.data_type() else {
        unreachable!()
    };
    let fields = fields.clone();
    let schema = Arc::new(Schema::new(vec![
        Field::new("id", DataType::Int64, false),
        envelope,
    ]));
    let fixture = Fixture::new(
        Arc::clone(&schema),
        &SinkOptions {
            row_key: Some("id".into()),
        },
    )?;
    let batch = |first| {
        RecordBatch::try_new(
            Arc::clone(&schema),
            vec![
                Arc::new(Int64Array::from(vec![first, first + 1])),
                Arc::new(StructArray::new(
                    fields.clone(),
                    vec![
                        Arc::new(StringArray::from(vec![Some("ok"), None])),
                        Arc::new(StringArray::from(vec![None, Some("TIMEOUT")])),
                        Arc::new(StringArray::from(vec![None, Some("provider failure")])),
                        Arc::new(BooleanArray::from(vec![None, Some(true)])),
                    ],
                    None,
                )),
            ],
        )
    };
    let winners = [fixture.split(0, 1), fixture.split(1, 3)];
    fixture
        .sink
        .write_attempt(&fixture.split(0, 0), &[batch(99)?])
        .await?;
    fixture.commit(&winners[0], batch(10)?).await?;
    fixture.commit(&winners[1], batch(20)?).await?;
    let manifest = fixture.sink.finalize_stage(&winners).await?;
    fixture
        .sink
        .write_attempt(&fixture.split(1, 4), &[batch(199)?])
        .await?;
    fixture.register_tables().await?;
    let data = fixture
        .ctx
        .sql("SELECT id FROM data ORDER BY id")
        .await?
        .collect()
        .await?;
    assert_batches_eq!(
        ["+----+", "| id |", "+----+", "| 10 |", "| 20 |", "+----+",],
        &data
    );
    let dlq = fixture
        .ctx
        .sql("SELECT row_key, error_code FROM dlq ORDER BY row_key")
        .await?
        .collect()
        .await?;
    assert_batches_eq!(
        [
            "+---------+------------+",
            "| row_key | error_code |",
            "+---------+------------+",
            "| 11      | TIMEOUT    |",
            "| 21      | TIMEOUT    |",
            "+---------+------------+",
        ],
        &dlq
    );
    let path = manifest.splits[0].files.dlq_rel_path.as_deref().unwrap();
    fixture
        .store
        .delete(&stage_path(&fixture.stage_root, path))
        .await?;
    let missing = fixture.reader().await.unwrap_err().to_string();
    assert!(missing.contains("dlq.parquet"), "{missing}");
    assert!(missing.to_lowercase().contains("not found"), "{missing}");
    Ok(())
}
