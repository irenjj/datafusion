// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: See the NOTICE file distributed with this work for additional information regarding copyright ownership
// SPDX-FileContributor: Licensed to the Apache Software Foundation (ASF) under one or more contributor license agreements

use std::sync::Arc;
use std::time::Duration;

use datafusion::arrow::{
    array::Int64Array,
    datatypes::{DataType, Field, Schema},
    record_batch::RecordBatch,
};
use datafusion::common::tree_node::TreeNodeRecursion;
use datafusion::common::{DataFusionError, Result, ScalarValue, Statistics, plan_err};
use datafusion::datasource::{
    listing::PartitionedFile,
    physical_plan::{
        FileGroup, FileOpenFuture, FileOpener, FileScanConfig, FileScanConfigBuilder,
        FileSource,
    },
    source::DataSourceExec,
    table_schema::TableSchema,
};
use datafusion::execution::{TaskContext, object_store::ObjectStoreUrl};
use datafusion::object_store::ObjectStore;
use datafusion::physical_expr::{Partitioning, PhysicalExpr};
use datafusion::physical_plan::{
    ExecutionPlan, collect, metrics::ExecutionPlanMetricsSet, projection::ProjectionExec,
};
use datafusion::prelude::SessionContext;
use datafusion_dist::{
    CommitOutcome, DistCodecs, ExtensionOperatorPolicy, LeaseConfig, LeaseError,
    LeasedSplitQueue, ManualClock, NamedPhysicalExtensionCodec, bind_split_to_template,
    decode_stage_template, decode_stage_template_with_codecs,
    split_plan_for_distribution, split_plan_for_distribution_with_codecs,
};
use datafusion_proto::physical_plan::{
    PhysicalExtensionCodec, PhysicalProtoConverterExtension,
};
use datafusion_proto::protobuf::{
    PhysicalPlanNode, physical_plan_node::PhysicalPlanType,
};
use futures::{StreamExt, TryStreamExt, stream};
use prost::Message;

use super::contract::{LegacySplitEnvelope, plan_hash, set_file_bytes};
use super::harness::{assert_plan_error, column_expr, rows};

#[derive(Clone)]
struct MockLeafSource {
    schema: TableSchema,
    rows_per_file: u32,
    batch_size: usize,
    metrics: ExecutionPlanMetricsSet,
}

impl FileSource for MockLeafSource {
    fn table_schema(&self) -> &TableSchema {
        &self.schema
    }

    fn create_file_opener(
        &self,
        _object_store: Arc<dyn ObjectStore>,
        _config: &FileScanConfig,
        _partition: usize,
    ) -> Result<Arc<dyn FileOpener>> {
        Ok(Arc::new(self.clone()))
    }

    fn with_batch_size(&self, batch_size: usize) -> Arc<dyn FileSource> {
        Arc::new(Self {
            batch_size,
            ..self.clone()
        })
    }

    fn supports_repartitioning(&self) -> bool {
        false
    }

    fn file_type(&self) -> &str {
        "mock_leaf"
    }

    fn metrics(&self) -> &ExecutionPlanMetricsSet {
        &self.metrics
    }

    fn apply_expressions(
        &self,
        _f: &mut dyn FnMut(&Arc<dyn PhysicalExpr>) -> Result<TreeNodeRecursion>,
    ) -> Result<TreeNodeRecursion> {
        Ok(TreeNodeRecursion::Continue)
    }
}

impl FileOpener for MockLeafSource {
    fn open(&self, file: PartitionedFile) -> Result<FileOpenFuture> {
        let first = file
            .object_meta
            .location
            .filename()
            .unwrap()
            .parse::<i64>()
            .unwrap();
        let ids = (first..first + i64::from(self.rows_per_file)).collect::<Vec<_>>();
        let batches = ids
            .chunks(self.batch_size)
            .map(|chunk| {
                RecordBatch::try_new(
                    Arc::clone(self.schema.file_schema()),
                    vec![Arc::new(Int64Array::from_iter_values(
                        chunk.iter().copied(),
                    ))],
                )
                .map_err(DataFusionError::from)
            })
            .collect::<Vec<_>>();
        Ok(Box::pin(async move { Ok(stream::iter(batches).boxed()) }))
    }
}

fn leaf_file(first: i64) -> PartitionedFile {
    PartitionedFile::new(first.to_string(), 100)
}

fn leaf_plan(
    rows_per_file: u32,
    files: Vec<PartitionedFile>,
    limit: Option<usize>,
) -> Arc<dyn ExecutionPlan> {
    let schema = Arc::new(Schema::new(vec![Field::new("id", DataType::Int64, false)]));
    let source = Arc::new(MockLeafSource {
        schema: TableSchema::from(&schema),
        rows_per_file,
        batch_size: 2,
        metrics: ExecutionPlanMetricsSet::new(),
    });
    let groups = files
        .into_iter()
        .map(|file| FileGroup::new(vec![file]))
        .collect();
    DataSourceExec::from_data_source(
        FileScanConfigBuilder::new(ObjectStoreUrl::local_filesystem(), source)
            .with_file_groups(groups)
            .with_limit(limit)
            .build(),
    )
}

#[derive(Debug, Default)]
enum Fault {
    #[default]
    None,
    DecodeFiles,
    DecodeLimit,
    BoundSchema,
    BoundPartitions,
}

#[derive(Debug)]
struct MockLeafCodec {
    name: &'static str,
    fault: Fault,
}

impl Default for MockLeafCodec {
    fn default() -> Self {
        Self {
            name: "test.mock_leaf",
            fault: Fault::None,
        }
    }
}

impl NamedPhysicalExtensionCodec for MockLeafCodec {
    fn extension_name(&self) -> &str {
        self.name
    }

    fn operator_policy(
        &self,
        node: &dyn ExecutionPlan,
    ) -> Option<ExtensionOperatorPolicy> {
        node.downcast_ref::<DataSourceExec>()?
            .downcast_to_file_source::<MockLeafSource>()?;
        Some(ExtensionOperatorPolicy {
            map_only: true,
            speculation_allowed: true,
        })
    }

    fn try_encode(
        &self,
        node: &Arc<dyn ExecutionPlan>,
        _converter: &dyn PhysicalProtoConverterExtension,
        _root_codec: &dyn PhysicalExtensionCodec,
    ) -> Result<Option<(u32, Vec<u8>)>> {
        Ok(node
            .downcast_ref::<DataSourceExec>()
            .and_then(DataSourceExec::downcast_to_file_source::<MockLeafSource>)
            .map(|(_, source)| (1, source.rows_per_file.to_le_bytes().to_vec())))
    }

    fn try_decode(
        &self,
        version: u32,
        payload: &[u8],
        inputs: &[Arc<dyn ExecutionPlan>],
        _ctx: &TaskContext,
        _converter: &dyn PhysicalProtoConverterExtension,
        _root_codec: &dyn PhysicalExtensionCodec,
    ) -> Result<Arc<dyn ExecutionPlan>> {
        let Ok(encoded_rows) = payload.try_into() else {
            return plan_err!("Invalid mock leaf row count");
        };
        if version != 1 || !inputs.is_empty() {
            return plan_err!("Invalid mock leaf version or inputs");
        }
        let files = if matches!(self.fault, Fault::DecodeFiles) {
            vec![leaf_file(10)]
        } else {
            vec![]
        };
        let limit = matches!(self.fault, Fault::DecodeLimit).then_some(1);
        Ok(leaf_plan(u32::from_le_bytes(encoded_rows), files, limit))
    }

    fn bind_split(
        &self,
        node: &Arc<dyn ExecutionPlan>,
        file: PartitionedFile,
    ) -> Result<Option<Arc<dyn ExecutionPlan>>> {
        let Some((scan, _)) = node
            .downcast_ref::<DataSourceExec>()
            .and_then(DataSourceExec::downcast_to_file_source::<MockLeafSource>)
        else {
            return Ok(None);
        };
        assert!(scan.file_groups.is_empty());
        assert!(scan.limit.is_none());
        assert!(file.range.is_none());
        let mut groups = vec![FileGroup::new(vec![file])];
        if matches!(self.fault, Fault::BoundPartitions) {
            groups.push(FileGroup::new(vec![leaf_file(99)]));
        }
        let bound: Arc<dyn ExecutionPlan> = DataSourceExec::from_data_source(
            FileScanConfigBuilder::from(scan.clone())
                .with_file_groups(groups)
                .build(),
        );
        if matches!(self.fault, Fault::BoundSchema) {
            let id = column_expr("id", bound.as_ref())?;
            return Ok(Some(Arc::new(ProjectionExec::try_new(
                [(id, "changed".into())],
                bound,
            )?)));
        }
        Ok(Some(bound))
    }
}

fn leaf_codecs(fault: Fault) -> Result<DistCodecs> {
    let mut codecs = DistCodecs::default();
    codecs.physical.register(Arc::new(MockLeafCodec {
        fault,
        ..Default::default()
    }))?;
    Ok(codecs)
}

#[tokio::test]
async fn zero_input_leaf_round_trips_and_executes_in_a_new_session() -> Result<()> {
    let native = leaf_plan(3, vec![leaf_file(10), leaf_file(20)], None);
    let coordinator = SessionContext::new();
    let baseline = collect(Arc::clone(&native), coordinator.task_ctx()).await?;
    assert_eq!(baseline.iter().map(RecordBatch::num_rows).sum::<usize>(), 6);

    let codecs = leaf_codecs(Fault::None)?;
    let (stage, splits) = split_plan_for_distribution_with_codecs(native, 1, &codecs)?;
    assert_eq!(splits.len(), 2);
    let proto = PhysicalPlanNode::decode(stage.plan_bytes.as_slice()).unwrap();
    let Some(PhysicalPlanType::Extension(extension)) = proto.physical_plan_type else {
        panic!("expected an extension leaf");
    };
    assert!(extension.inputs.is_empty());
    assert!(extension.plan_name.is_none());

    let worker = SessionContext::new();
    let prepared = decode_stage_template_with_codecs(&stage, &worker, &codecs)?;
    let mut batches = Vec::new();
    for split in splits {
        let bound = bind_split_to_template(&prepared, &split)?;
        assert_eq!(bound.schema(), stage.output_schema);
        batches.extend(
            bound
                .execute(0, worker.task_ctx())?
                .try_collect::<Vec<_>>()
                .await?,
        );
    }
    assert_eq!(rows(&batches)?, rows(&baseline)?);
    Ok(())
}

#[test]
fn leaf_requires_one_split_binder() -> Result<()> {
    let native = leaf_plan(3, vec![leaf_file(10)], None);
    assert_plan_error(
        &split_plan_for_distribution(Arc::clone(&native), 1).unwrap_err(),
        "Map-only distribution does not support mock_leaf FileSource",
    );
    let mut codecs = leaf_codecs(Fault::None)?;
    codecs.physical.register(Arc::new(MockLeafCodec {
        name: "test.other_leaf",
        ..Default::default()
    }))?;
    assert_plan_error(
        &split_plan_for_distribution_with_codecs(native, 1, &codecs).unwrap_err(),
        "Ambiguous physical extension split binding for DataSourceExec",
    );
    Ok(())
}

#[test]
fn leaf_wire_rejects_unregistered_codec_and_multiple_inputs() -> Result<()> {
    let codecs = leaf_codecs(Fault::None)?;
    let (mut stage, _) = split_plan_for_distribution_with_codecs(
        leaf_plan(3, vec![leaf_file(10)], None),
        1,
        &codecs,
    )?;
    let worker = SessionContext::new();
    assert_plan_error(
        &decode_stage_template(&stage, &worker).unwrap_err(),
        "Unregistered physical extension",
    );
    let mut proto = PhysicalPlanNode::decode(stage.plan_bytes.as_slice()).unwrap();
    let Some(PhysicalPlanType::Extension(extension)) = &mut proto.physical_plan_type
    else {
        panic!("expected an extension leaf");
    };
    extension.inputs = vec![PhysicalPlanNode::default(); 2];
    stage.plan_bytes = proto.encode_to_vec();
    stage.plan_fingerprint = plan_hash(&stage)?;
    assert_plan_error(
        &decode_stage_template_with_codecs(&stage, &worker, &codecs).unwrap_err(),
        "extensions require exactly one input for unary operators or zero inputs for scan leaves",
    );
    Ok(())
}

#[test]
fn leaf_template_requires_empty_file_groups_and_no_limit() -> Result<()> {
    let codecs = leaf_codecs(Fault::None)?;
    let (stage, _) = split_plan_for_distribution_with_codecs(
        leaf_plan(3, vec![leaf_file(10)], None),
        1,
        &codecs,
    )?;
    let worker = SessionContext::new();
    for (fault, expected) in [
        (
            Fault::DecodeFiles,
            "must have empty file groups for mock_leaf FileSource",
        ),
        (Fault::DecodeLimit, "does not support FileScanConfig.limit"),
    ] {
        assert_plan_error(
            &decode_stage_template_with_codecs(&stage, &worker, &leaf_codecs(fault)?)
                .unwrap_err(),
            expected,
        );
    }
    assert_plan_error(
        &split_plan_for_distribution_with_codecs(
            leaf_plan(3, vec![leaf_file(10)], Some(1)),
            1,
            &codecs,
        )
        .unwrap_err(),
        "does not support FileScanConfig.limit",
    );
    Ok(())
}

#[test]
fn leaf_rejects_ordered_and_explicitly_partitioned_scans() -> Result<()> {
    let codecs = leaf_codecs(Fault::None)?;
    let plan = leaf_plan(3, vec![leaf_file(10)], None);
    let (scan, _) = plan
        .downcast_ref::<DataSourceExec>()
        .unwrap()
        .downcast_to_file_source::<MockLeafSource>()
        .unwrap();
    for builder in [
        FileScanConfigBuilder::from(scan.clone()).with_preserve_order(true),
        FileScanConfigBuilder::from(scan.clone())
            .with_output_partitioning(Some(Partitioning::UnknownPartitioning(1))),
    ] {
        let plan = DataSourceExec::from_data_source(builder.build());
        assert_plan_error(
            &split_plan_for_distribution_with_codecs(plan, 1, &codecs).unwrap_err(),
            "Map-only distribution does not support ordered, explicitly partitioned, or custom-adapter mock_leaf FileSource scans",
        );
    }
    Ok(())
}

#[test]
fn leaf_binding_checks_split_metadata_and_range_support() -> Result<()> {
    let codecs = leaf_codecs(Fault::None)?;
    let file = leaf_file(10);
    let (stage, splits) = split_plan_for_distribution_with_codecs(
        leaf_plan(3, vec![file.clone()], None),
        1,
        &codecs,
    )?;
    let prepared =
        decode_stage_template_with_codecs(&stage, &SessionContext::new(), &codecs)?;
    let split = &splits[0];
    assert!(bind_split_to_template(&prepared, split).is_ok());
    let mut tampered = split.clone();
    tampered.file_proto.push(0);
    assert_plan_error(
        &bind_split_to_template(&prepared, &tampered).unwrap_err(),
        "fingerprint does not match",
    );

    for (file, expected) in [
        (
            file.clone()
                .with_partition_values(vec![ScalarValue::Int64(Some(5))]),
            "partition values do not match",
        ),
        (
            file.clone()
                .with_statistics(Arc::new(Statistics::new_unknown(&Schema::empty()))),
            "statistics do not match",
        ),
        (
            file.clone().with_range(0, 101),
            "byte range is outside the file or empty",
        ),
        (
            file.clone().with_range(0, 10),
            "byte range is not supported by mock_leaf FileSource",
        ),
    ] {
        let mut malformed = split.clone();
        let wire = LegacySplitEnvelope {
            file: Some((&file).try_into()?),
            e_tag: None,
        }
        .encode_to_vec();
        set_file_bytes(&stage, &mut malformed, wire);
        assert_plan_error(
            &bind_split_to_template(&prepared, &malformed).unwrap_err(),
            expected,
        );
    }
    assert_plan_error(
        &split_plan_for_distribution_with_codecs(
            leaf_plan(3, vec![file.with_range(0, 10)], None),
            1,
            &codecs,
        )
        .unwrap_err(),
        "byte range is not supported by mock_leaf FileSource",
    );
    Ok(())
}

#[test]
fn bound_leaf_preserves_schema_and_single_partition() -> Result<()> {
    let (stage, splits) = split_plan_for_distribution_with_codecs(
        leaf_plan(3, vec![leaf_file(10)], None),
        1,
        &leaf_codecs(Fault::None)?,
    )?;
    let worker = SessionContext::new();
    for (fault, expected) in [
        (Fault::BoundSchema, "changed its output schema"),
        (Fault::BoundPartitions, "must have one partition"),
    ] {
        let prepared =
            decode_stage_template_with_codecs(&stage, &worker, &leaf_codecs(fault)?)?;
        assert_plan_error(
            &bind_split_to_template(&prepared, &splits[0]).unwrap_err(),
            expected,
        );
    }
    Ok(())
}

#[tokio::test]
async fn leased_leaf_rejects_stale_execution_and_commit() -> Result<()> {
    let codecs = leaf_codecs(Fault::None)?;
    let (stage, splits) = split_plan_for_distribution_with_codecs(
        leaf_plan(3, vec![leaf_file(10)], None),
        1,
        &codecs,
    )?;
    let prepared =
        decode_stage_template_with_codecs(&stage, &SessionContext::new(), &codecs)?;
    let config = LeaseConfig {
        speculation_allowed: true,
        ..Default::default()
    };
    let clock = Arc::new(ManualClock::default());
    let queue = LeasedSplitQueue::new(splits, config, clock.clone())?;
    let (first, first_token) = queue.claim(0)?.unwrap();
    let first_output = prepared.execute_lease(&queue, &first, &first_token).await?;
    clock.advance(config.lease_ttl + Duration::from_nanos(1));

    let (retry, retry_token) = queue.claim(1)?.unwrap();
    assert_eq!((retry.split_id, retry.attempt_id), (first.split_id, 1));
    let retry_output = prepared.execute_lease(&queue, &retry, &retry_token).await?;
    assert_eq!(first_output.batches, retry_output.batches);
    assert_eq!(retry_output.metrics.output_rows, 3);
    assert_eq!(
        queue.commit(&retry_token, retry_output)?,
        CommitOutcome::Committed
    );

    let error = prepared
        .execute_lease(&queue, &first, &first_token)
        .await
        .unwrap_err();
    let DataFusionError::External(error) = error else {
        panic!("expected lease error: {error}");
    };
    assert_eq!(
        error.downcast_ref::<LeaseError>(),
        Some(&LeaseError::StaleAttempt)
    );
    assert_eq!(
        queue.commit(&first_token, first_output),
        Err(LeaseError::StaleAttempt)
    );
    assert!(queue.claim(2)?.is_none());
    Ok(())
}
