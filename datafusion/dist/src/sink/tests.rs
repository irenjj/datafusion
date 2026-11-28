// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: See the NOTICE file distributed with this work for additional information regarding copyright ownership
// SPDX-FileContributor: Licensed to the Apache Software Foundation (ASF) under one or more contributor license agreements

use datafusion::arrow::array::Int64Array;
use datafusion::arrow::datatypes::{DataType, Field, Schema};
use datafusion::object_store::local::LocalFileSystem;
use datafusion::parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;
use tempfile::{TempDir, tempdir};

use super::*;

struct Fixture {
    _dir: TempDir,
    store: Arc<dyn ObjectStore>,
    sink: ParquetSplitSink,
    stage: StagePlan,
}

impl Fixture {
    fn new(schema: SchemaRef, options: &SinkOptions) -> Result<Self> {
        let dir = tempdir()?;
        let store: Arc<dyn ObjectStore> =
            Arc::new(LocalFileSystem::new_with_prefix(dir.path())?);
        let plan_bytes = vec![1, 2, 3];
        let stage = StagePlan {
            stage_id: 0,
            plan_fingerprint: plan_fingerprint(&plan_bytes, &schema)?,
            plan_bytes,
            output_schema: schema,
        };
        let sink = ParquetSplitSink::try_new(
            Arc::clone(&store),
            &Path::from("output"),
            &stage,
            options,
        )?;
        Ok(Self {
            _dir: dir,
            store,
            sink,
            stage,
        })
    }

    fn plain() -> Result<Self> {
        Self::new(
            Arc::new(Schema::new(vec![Field::new("id", DataType::Int64, false)])),
            &SinkOptions::default(),
        )
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

    async fn write(&self, split: &Split, values: Vec<i64>) -> Result<AttemptArtifact> {
        self.sink
            .write_attempt(
                split,
                &[RecordBatch::try_new(
                    Arc::clone(&self.stage.output_schema),
                    vec![Arc::new(Int64Array::from(values))],
                )?],
            )
            .await
    }

    async fn batches(&self, relative_path: &str) -> Result<Vec<RecordBatch>> {
        let bytes = self
            .store
            .get(&self.object(relative_path))
            .await?
            .bytes()
            .await?;
        let reader = ParquetRecordBatchReaderBuilder::try_new(bytes)?.build()?;
        Ok(reader.collect::<std::result::Result<_, _>>()?)
    }

    fn object(&self, relative_path: &str) -> Path {
        self.sink.stage_path(relative_path)
    }
}

#[tokio::test]
async fn competing_attempts_only_publish_one_winner() -> Result<()> {
    let fixture = Fixture::plain()?;
    let attempts = [
        fixture.split(7, 0),
        fixture.split(7, 1),
        fixture.split(7, 2),
    ];
    let race = async |split: &Split| -> Result<()> {
        let receipt = fixture
            .write(split, vec![i64::from(split.attempt_id)])
            .await?;
        fixture.sink.commit_split(split, receipt).await
    };
    let (a, b, c) =
        tokio::join!(race(&attempts[0]), race(&attempts[1]), race(&attempts[2]));
    a?;
    b?;
    c?;
    let marker = fixture.sink.read_checkpoint(&attempts[0]).await?.unwrap();
    let marker_path = fixture.sink.checkpoint_path(&attempts[0]);
    let marker_bytes = fixture.store.get(&marker_path).await?.bytes().await?;
    let winner_path = fixture.object(&marker.files.data_rel_path);
    let before = fixture.store.get(&winner_path).await?.bytes().await?;
    for split in &attempts {
        let batches = fixture
            .batches(&attempt_path(
                split.split_id,
                split.attempt_id,
                "data.parquet",
            ))
            .await?;
        let ids = batches[0]
            .column(0)
            .as_any()
            .downcast_ref::<Int64Array>()
            .unwrap();
        assert_eq!(ids.values().as_ref(), &[i64::from(split.attempt_id)]);
    }
    let late = fixture.split(7, 9);
    let late_receipt = fixture.write(&late, vec![999]).await?;
    fixture.sink.commit_split(&late, late_receipt).await?;
    assert_eq!(
        fixture.store.get(&winner_path).await?.bytes().await?,
        before
    );
    assert_eq!(
        fixture.sink.read_checkpoint(&attempts[0]).await?.unwrap(),
        marker
    );
    assert_eq!(
        fixture.store.get(&marker_path).await?.bytes().await?,
        marker_bytes
    );
    Ok(())
}

#[tokio::test]
async fn replay_is_idempotent_and_tampered_marker_is_rejected() -> Result<()> {
    let fixture = Fixture::plain()?;
    let first = fixture.split(1, 0);
    let replay = fixture.split(1, 4);
    let receipt = fixture.write(&first, vec![1, 2]).await?;
    fixture.sink.commit_split(&first, receipt.clone()).await?;
    assert_eq!(fixture.write(&first, vec![1, 2]).await?, receipt);
    assert!(fixture.write(&first, vec![1, 3]).await.is_err());
    fixture.sink.commit_split(&first, receipt).await?;

    let replay_receipt = fixture.write(&replay, vec![1, 2]).await?;
    fixture
        .sink
        .commit_split(&replay, replay_receipt.clone())
        .await?;
    let path = fixture.sink.checkpoint_path(&first);
    let marker = fixture.store.get(&path).await?.bytes().await?;
    for field in ["data_sha256", "split_fingerprint"] {
        let mut changed: serde_json::Value =
            serde_json::from_slice(&marker).map_err(json_error)?;
        changed[field] = "f".repeat(64).into();
        fixture
            .store
            .put(&path, json_bytes(&changed)?.into())
            .await?;
        assert!(
            fixture
                .sink
                .commit_split(&replay, replay_receipt.clone())
                .await
                .is_err()
        );
    }
    assert!(fixture.sink.resume_filter(&[first]).await.is_err());
    Ok(())
}

#[tokio::test]
async fn missing_or_altered_winner_fails_closed() -> Result<()> {
    let fixture = Fixture::plain()?;
    let first = fixture.split(1, 0);
    let replay = fixture.split(1, 1);
    let receipt = fixture.write(&first, vec![1]).await?;
    let data = fixture.object(&receipt.files.data_rel_path);
    fixture.sink.commit_split(&first, receipt.clone()).await?;
    let retry = fixture.write(&replay, vec![2]).await?;

    let mut changed = fixture.store.get(&data).await?.bytes().await?.to_vec();
    changed[4] ^= 1;
    fixture.store.put(&data, changed.into()).await?;
    assert!(
        fixture
            .sink
            .commit_split(&replay, retry.clone())
            .await
            .is_err()
    );

    fixture.store.delete(&data).await?;
    assert!(
        fixture
            .sink
            .resume_filter(std::slice::from_ref(&first))
            .await
            .is_err()
    );
    assert!(fixture.sink.finalize_stage(&[first]).await.is_err());
    assert!(fixture.sink.commit_split(&replay, retry).await.is_err());
    Ok(())
}

#[tokio::test]
async fn resume_skips_committed_and_finalizes_in_split_order() -> Result<()> {
    let fixture = Fixture::plain()?;
    let earlier = fixture.split(2, 0);
    let later = fixture.split(8, 5);
    let splits = [later.clone(), earlier.clone()];

    let receipt = fixture.write(&later, vec![7, 8, 9]).await?;
    assert!(receipt.files.dlq_rel_path.is_none());
    assert!(
        fixture
            .store
            .head(&fixture.object(&attempt_path(
                later.split_id,
                later.attempt_id,
                "dlq.parquet"
            )))
            .await
            .is_err()
    );
    fixture.sink.commit_split(&later, receipt).await?;
    assert_eq!(fixture.sink.resume_filter(&splits).await?[0].split_id, 2);
    assert!(fixture.sink.finalize_stage(&splits).await.is_err());

    let receipt = fixture.write(&earlier, vec![4]).await?;
    fixture.sink.commit_split(&earlier, receipt).await?;
    assert!(fixture.sink.resume_filter(&splits).await?.is_empty());
    let manifest = fixture.sink.finalize_stage(&splits).await?;
    assert_eq!(manifest.output_rows, 4);
    assert_eq!(manifest.dlq_rows, 0);
    assert_eq!(
        manifest
            .splits
            .iter()
            .map(|split| split.split_id)
            .collect::<Vec<_>>(),
        [2, 8]
    );
    let persisted = fixture
        .store
        .get(&fixture.object("STAGE_MANIFEST.json"))
        .await?
        .bytes()
        .await?;
    assert_eq!(
        serde_json::from_slice::<StageManifest>(&persisted).map_err(json_error)?,
        manifest
    );
    assert_eq!(fixture.sink.finalize_stage(&splits).await?, manifest);
    assert!(fixture.sink.finalize_stage(&[later]).await.is_err());
    Ok(())
}

#[cfg(feature = "ai")]
mod ai {
    use std::fmt;

    use datafusion::arrow::array::{
        Array, ArrayRef, BooleanArray, FixedSizeListArray, Float32Array, StringArray,
        StructArray,
    };
    use datafusion::common::tree_node::TreeNodeRecursion;
    use datafusion::datasource::memory::MemorySourceConfig;
    use datafusion::execution::TaskContext;
    use datafusion::physical_expr::PhysicalExpr;
    use datafusion::physical_plan::{
        DisplayAs, DisplayFormatType, PlanProperties, SendableRecordBatchStream, collect,
    };
    use datafusion::prelude::SessionContext;
    use datafusion_ai::{AiErrorType, CheckpointOutputColumn, CheckpointableTarget};

    use super::*;

    #[derive(Debug)]
    struct EnvelopeExec {
        memory: Arc<dyn ExecutionPlan>,
        declares_valid_target: bool,
    }

    impl DisplayAs for EnvelopeExec {
        fn fmt_as(
            &self,
            _: DisplayFormatType,
            f: &mut fmt::Formatter<'_>,
        ) -> fmt::Result {
            write!(f, "EnvelopeExec")
        }
    }

    impl ExecutionPlan for EnvelopeExec {
        fn name(&self) -> &'static str {
            "EnvelopeExec"
        }
        fn properties(&self) -> &Arc<PlanProperties> {
            self.memory.properties()
        }
        fn children(&self) -> Vec<&Arc<dyn ExecutionPlan>> {
            vec![]
        }

        fn apply_expressions(
            &self,
            _: &mut dyn FnMut(&Arc<dyn PhysicalExpr>) -> Result<TreeNodeRecursion>,
        ) -> Result<TreeNodeRecursion> {
            Ok(TreeNodeRecursion::Continue)
        }

        fn with_new_children(
            self: Arc<Self>,
            children: Vec<Arc<dyn ExecutionPlan>>,
        ) -> Result<Arc<dyn ExecutionPlan>> {
            assert!(children.is_empty());
            Ok(self)
        }

        fn execute(
            &self,
            partition: usize,
            ctx: Arc<TaskContext>,
        ) -> Result<SendableRecordBatchStream> {
            self.memory.execute(partition, ctx)
        }
    }

    impl CheckpointableTarget for EnvelopeExec {
        fn passthrough_column_count(&self) -> usize {
            1
        }
        fn ai_output_columns(&self) -> Vec<CheckpointOutputColumn> {
            ["z_embedding", "a_label"]
                .into_iter()
                .enumerate()
                .map(|(index, name)| CheckpointOutputColumn {
                    index: index + 1,
                    name: if self.declares_valid_target {
                        name
                    } else {
                        "incorrect"
                    }
                    .into(),
                    is_error_envelope: true,
                })
                .collect()
        }
        fn preserves_input_order(&self) -> bool {
            true
        }
    }

    fn example_batch(keys: Vec<Option<i64>>, failures: bool) -> Result<RecordBatch> {
        let item = Arc::new(Field::new("item", DataType::Float32, true));
        let vector_values = Arc::new(FixedSizeListArray::new(
            item,
            2,
            Arc::new(Float32Array::from(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0])),
            None,
        ));
        let label_values = Arc::new(StringArray::from(vec!["ok", "quota", "reject"]));
        let embedding = AiErrorType::new(vector_values.data_type().clone())
            .field("z_embedding", true)?;
        let label = AiErrorType::new(DataType::Utf8).field("a_label", true)?;
        let envelope =
            |field: &Field, values: ArrayRef, errors: Vec<Option<&str>>, retryable| {
                let messages = errors
                    .iter()
                    .map(|code| code.map(|_| "provider failure"))
                    .collect::<Vec<_>>();
                let retries = errors
                    .iter()
                    .map(|code| code.map(|_| retryable))
                    .collect::<Vec<_>>();
                let DataType::Struct(fields) = field.data_type() else {
                    unreachable!()
                };
                Arc::new(StructArray::new(
                    fields.clone(),
                    vec![
                        values,
                        Arc::new(StringArray::from(errors)),
                        Arc::new(StringArray::from(messages)),
                        Arc::new(BooleanArray::from(retries)),
                    ],
                    None,
                ))
            };
        let vector = envelope(
            &embedding,
            vector_values,
            vec![None, None, failures.then_some("TIMEOUT")],
            true,
        );
        let label_array = envelope(
            &label,
            label_values,
            vec![
                None,
                failures.then_some("RATE_LIMITED"),
                failures.then_some("PERMANENT_HTTP"),
            ],
            false,
        );
        Ok(RecordBatch::try_new(
            Arc::new(Schema::new(vec![
                Field::new("id", DataType::Int64, true),
                embedding,
                label,
            ])),
            vec![Arc::new(Int64Array::from(keys)), vector, label_array],
        )?)
    }

    #[tokio::test]
    async fn routes_mixed_and_double_failures_with_typed_values_and_sorted_dlq()
    -> Result<()> {
        let batch = example_batch(vec![Some(99), Some(10), Some(2)], true)?;
        let plan = Arc::new(EnvelopeExec {
            memory: MemorySourceConfig::try_new_exec(
                &[vec![batch.clone()]],
                batch.schema(),
                None,
            )?,
            declares_valid_target: true,
        });
        let options = SinkOptions {
            row_key: Some("id".into()),
        };
        let fixture = Fixture::new(batch.schema(), &options)?;
        fixture.sink.validate_target(plan.as_ref())?;
        let bad = EnvelopeExec {
            memory: Arc::clone(&plan.memory),
            declares_valid_target: false,
        };
        assert!(fixture.sink.validate_target(&bad).is_err());

        let split = fixture.split(10, 0);
        let rows = collect(plan, SessionContext::new().task_ctx()).await?;
        let artifact = fixture.sink.write_attempt(&split, &rows).await?;
        assert_eq!(artifact.files.output_rows, 1);
        assert_eq!(artifact.files.dlq_rows, 3);

        let data = fixture.batches(&artifact.files.data_rel_path).await?;
        assert_eq!(data[0].num_rows(), 1);
        let vector = data[0]
            .column(1)
            .as_any()
            .downcast_ref::<FixedSizeListArray>()
            .unwrap();
        assert_eq!(vector.value_length(), 2);
        let value = vector.value(0);
        assert_eq!(
            value
                .as_any()
                .downcast_ref::<Float32Array>()
                .unwrap()
                .values()
                .as_ref(),
            &[1.0, 2.0]
        );
        assert_eq!(data[0].schema().field(1).extension_type_name(), None);
        let labels = data[0]
            .column(2)
            .as_any()
            .downcast_ref::<StringArray>()
            .unwrap();
        assert_eq!(labels.value(0), "ok");

        let dlq = fixture
            .batches(artifact.files.dlq_rel_path.as_deref().unwrap())
            .await?;
        let batch = &dlq[0];
        let keys = batch
            .column(0)
            .as_any()
            .downcast_ref::<Int64Array>()
            .unwrap();
        let text = |column| {
            batch
                .column(column)
                .as_any()
                .downcast_ref::<StringArray>()
                .unwrap()
                .iter()
                .flatten()
                .collect::<Vec<_>>()
        };
        let retries = batch
            .column(4)
            .as_any()
            .downcast_ref::<BooleanArray>()
            .unwrap();
        assert_eq!(keys.values().as_ref(), &[2, 2, 10]);
        assert_eq!(text(1), ["a_label", "z_embedding", "a_label"]);
        assert_eq!(text(2), ["PERMANENT_HTTP", "TIMEOUT", "RATE_LIMITED"]);
        assert_eq!(text(3), ["provider failure"; 3]);
        assert_eq!(
            retries.iter().collect::<Vec<_>>(),
            [Some(false), Some(true), Some(false)]
        );

        let second_split = fixture.split(3, 1);
        let second = fixture.sink.write_attempt(&second_split, &rows).await?;
        fixture.sink.commit_split(&split, artifact).await?;
        fixture.sink.commit_split(&second_split, second).await?;
        let manifest = fixture.sink.finalize_stage(&[split, second_split]).await?;
        assert_eq!(manifest.output_rows, 2);
        assert_eq!(manifest.dlq_rows, 6);
        assert_eq!(
            manifest.dlq_by_code,
            BTreeMap::from([
                ("PERMANENT_HTTP".to_owned(), 2),
                ("RATE_LIMITED".to_owned(), 2),
                ("TIMEOUT".to_owned(), 2),
            ])
        );
        Ok(())
    }

    #[tokio::test]
    async fn envelopes_require_non_null_keys_and_omit_empty_dlq() -> Result<()> {
        let batch = example_batch(vec![Some(99), Some(10), Some(2)], false)?;
        for row_key in [None, Some("missing".into())] {
            assert!(matches!(
                Fixture::new(batch.schema(), &SinkOptions { row_key }),
                Err(DataFusionError::Plan(_))
            ));
        }
        let options = SinkOptions {
            row_key: Some("id".into()),
        };
        let fixture = Fixture::new(batch.schema(), &options)?;
        let split = fixture.split(0, 0);
        let null_key = example_batch(vec![None, Some(10), Some(2)], true)?;
        assert!(matches!(
            fixture.sink.write_attempt(&split, &[null_key]).await,
            Err(DataFusionError::Plan(_))
        ));

        let artifact = fixture.sink.write_attempt(&split, &[batch]).await?;
        assert_eq!(artifact.files.output_rows, 3);
        assert!(artifact.files.dlq_rel_path.is_none());
        assert!(
            fixture
                .store
                .head(&fixture.object(&attempt_path(
                    split.split_id,
                    split.attempt_id,
                    "dlq.parquet"
                )))
                .await
                .is_err()
        );
        fixture.sink.commit_split(&split, artifact).await?;
        Ok(())
    }
}
