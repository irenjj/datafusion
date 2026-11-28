// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: See the NOTICE file distributed with this work for additional information regarding copyright ownership
// SPDX-FileContributor: Licensed to the Apache Software Foundation (ASF) under one or more contributor license agreements

#![cfg(feature = "ai")]

use std::sync::Arc;
use std::time::Duration;

use datafusion::arrow::datatypes::{DataType, Field, Schema};
use datafusion::common::{DFSchema, Result};
use datafusion::physical_expr::expressions::Column;
use datafusion::physical_plan::{
    ExecutionPlan, collect, empty::EmptyExec, filter::FilterExec,
};
use datafusion::prelude::{SessionConfig, SessionContext, col, lit};
use datafusion_ai::llm::{LlmExec, LlmProviderRegistry, MockLlmBackend};
use datafusion_dist::{
    CommitOutcome, DistCodecs, DistExtensionRegistry, DistPhysicalCodec, LeaseConfig,
    LeasedSplitQueue, LlmExecCodec, ManualClock, NamedPhysicalExtensionCodec,
    bind_split_to_template, decode_stage_template_with_codecs,
    split_plan_for_distribution_with_codecs,
};
use datafusion_proto::physical_plan::DefaultPhysicalProtoConverter;
use datafusion_proto::protobuf::PhysicalExprNode;
use prost::Message;
use tempfile::tempdir;

use super::gated_queue::take_outputs;
use super::harness::{
    assert_matches_single_node, assert_plan_error, metric, project, register_parquet,
    rows, scan_plan, write_parquet,
};

fn registry() -> Result<LlmProviderRegistry> {
    let registry = LlmProviderRegistry::default();
    registry.register("offline", Arc::new(MockLlmBackend))?;
    Ok(registry)
}

fn session(registry: &LlmProviderRegistry) -> SessionContext {
    SessionContext::new_with_config(
        SessionConfig::new()
            .with_target_partitions(1)
            .with_extension(Arc::new(registry.clone())),
    )
}

#[tokio::test]
async fn llm_splits_match_local_results_and_isolate_retry_metrics() -> Result<()> {
    let dir = tempdir()?;
    write_parquet(&dir.path().join("a.parquet"), &[(0, 0), (2, 20)])?;
    write_parquet(&dir.path().join("b.parquet"), &[(4, 30), (6, 40), (8, 50)])?;
    let coordinator_registry = registry()?;
    let coordinator = session(&coordinator_registry);
    register_parquet(&coordinator, dir.path(), false).await?;
    let scan = scan_plan(&coordinator).await?;
    let projected = project(scan, ["id", "prompt_col", "filter_only_col"])?;
    let predicate = coordinator.create_physical_expr(
        col("filter_only_col").gt(lit(10_i64)),
        &DFSchema::try_from(projected.schema())?,
    )?;
    let input = Arc::new(FilterExec::try_new(predicate, projected)?);
    let prompt = coordinator
        .create_physical_expr(col("prompt_col"), &DFSchema::try_from(input.schema())?)?;
    let llm = Arc::new(LlmExec::try_new_with_registry(
        input,
        prompt,
        "test-model".into(),
        r#"{ "provider": "offline", "extra_body": {"z": 1, "a": 2}, "batch_size": 2 }"#,
        "answer".into(),
        &coordinator_registry,
    )?);
    let plan: Arc<dyn ExecutionPlan> = llm.clone();
    let baseline = collect(Arc::clone(&plan), coordinator.task_ctx()).await?;
    assert_eq!(metric(llm.as_ref(), "submitted_prompts"), 4);

    let mut physical = DistExtensionRegistry::default();
    physical.register(Arc::new(LlmExecCodec::new()))?;
    let codecs = DistCodecs::default().with_physical_codecs(physical);
    let config = LeaseConfig::default();
    config.validate_for_plan(plan.as_ref(), &codecs.physical)?;
    let speculative = LeaseConfig {
        speculation_allowed: true,
        ..config
    };
    assert_plan_error(
        &speculative
            .validate_for_plan(plan.as_ref(), &codecs.physical)
            .unwrap_err(),
        "Speculation is not allowed",
    );
    let (stage, splits) = split_plan_for_distribution_with_codecs(plan, 2, &codecs)?;
    assert_eq!(splits.len(), 2);
    assert_eq!(
        stage.output_schema.field(3),
        &Field::new("answer", DataType::Utf8, true)
    );
    assert_plan_error(
        &decode_stage_template_with_codecs(&stage, &SessionContext::new(), &codecs)
            .unwrap_err(),
        "requires a LlmProviderRegistry extension",
    );

    let worker = session(&registry()?);
    assert_ne!(worker.session_id(), coordinator.session_id());
    let prepared = decode_stage_template_with_codecs(&stage, &worker, &codecs)?;
    let bound = bind_split_to_template(&prepared, &splits[0])?;
    let decoded = bound.downcast_ref::<LlmExec>().unwrap();
    assert_eq!(decoded.model(), llm.model());
    assert_eq!(decoded.output_col(), llm.output_col());
    assert_eq!(decoded.options_json(), llm.options_json());
    assert_eq!(
        decoded.options_json(),
        r#"{"batch_size":2,"extra_body":{"a":2,"z":1},"provider":"offline"}"#
    );
    assert_eq!(
        decoded.prompt().downcast_ref::<Column>(),
        llm.prompt().downcast_ref::<Column>()
    );
    assert_eq!(metric(decoded, "submitted_prompts"), 0);

    let clock = Arc::new(ManualClock::default());
    let queue = LeasedSplitQueue::new(splits, config, clock.clone())?;
    let (first_split, first_token) = queue.claim(0)?.unwrap();
    let first = prepared
        .execute_lease(&queue, &first_split, &first_token)
        .await?;
    clock.advance(config.lease_ttl + Duration::from_nanos(1));
    while let Some((split, token)) = queue.claim(1)? {
        let output = prepared.execute_lease(&queue, &split, &token).await?;
        if split.split_id == first_split.split_id {
            assert_eq!(output.attempt_id, 1);
            assert_eq!(
                output.metrics.submitted_prompts,
                first.metrics.submitted_prompts
            );
            assert_eq!(rows(&output.batches)?, rows(&first.batches)?);
        } else {
            assert_eq!(output.attempt_id, 0);
        }
        assert_eq!(
            output.metrics.output_rows,
            2 * output.metrics.submitted_prompts
        );
        assert_eq!(queue.commit(&token, output)?, CommitOutcome::Committed);
    }
    let outputs = take_outputs(queue)?;
    let mut prompt_counts: Vec<_> = outputs
        .iter()
        .map(|output| output.metrics.submitted_prompts)
        .collect();
    prompt_counts.sort_unstable();
    assert_eq!(prompt_counts, [1, 3]);
    assert_eq!(metric(decoded, "submitted_prompts"), 0);
    assert_matches_single_node(&stage, llm.schema().as_ref(), &baseline, &outputs)
}

fn llm_test_plan(registry: &LlmProviderRegistry, options_json: &str) -> Result<LlmExec> {
    let input: Arc<dyn ExecutionPlan> = Arc::new(EmptyExec::new(Arc::new(Schema::new(
        vec![Field::new("prompt", DataType::Utf8, false)],
    ))));
    LlmExec::try_new_with_registry(
        input,
        Arc::new(Column::new("prompt", 0)),
        "test-model".into(),
        options_json,
        "answer".into(),
        registry,
    )
}

#[test]
fn llm_codec_requires_one_input_and_its_supported_version() -> Result<()> {
    let registry = registry()?;
    let ctx = session(&registry);
    let llm = llm_test_plan(&registry, "{}")?;
    let input = Arc::clone(llm.input());
    let plan: Arc<dyn ExecutionPlan> = Arc::new(llm);
    let codec = LlmExecCodec::new();
    let converter = DefaultPhysicalProtoConverter {};
    let root = DistPhysicalCodec::default();
    assert!(codec.operator_policy(input.as_ref()).is_none());
    assert!(codec.try_encode(&input, &converter, &root)?.is_none());
    let (version, payload) = codec.try_encode(&plan, &converter, &root)?.unwrap();
    assert_eq!(codec.extension_name(), "df.llm.exec");
    assert_eq!(version, 1);
    let task = ctx.task_ctx();
    for inputs in [vec![], vec![Arc::clone(&input), Arc::clone(&input)]] {
        assert_plan_error(
            &codec
                .try_decode(version, &payload, &inputs, &task, &converter, &root)
                .unwrap_err(),
            "LlmExec requires exactly one input",
        );
    }
    assert_plan_error(
        &codec
            .try_decode(2, &payload, &[input], &task, &converter, &root)
            .unwrap_err(),
        "Unsupported LlmExec schema version 2",
    );
    Ok(())
}

#[derive(Message)]
struct WirePayload {
    #[prost(message, optional, tag = "1")]
    prompt: Option<PhysicalExprNode>,
    #[prost(string, tag = "2")]
    model: String,
    #[prost(string, tag = "3")]
    options_json: String,
    #[prost(string, tag = "4")]
    output_col: String,
}

#[test]
fn llm_codec_rejects_noncanonical_options_json() -> Result<()> {
    let registry = registry()?;
    let ctx = session(&registry);
    let canonical = r#"{"batch_size":2,"provider":"offline"}"#;
    let llm = llm_test_plan(&registry, canonical)?;
    let inputs = [Arc::clone(llm.input())];
    let plan: Arc<dyn ExecutionPlan> = Arc::new(llm);
    let codec = LlmExecCodec::new();
    let converter = DefaultPhysicalProtoConverter {};
    let root = DistPhysicalCodec::default();
    let (version, payload) = codec.try_encode(&plan, &converter, &root)?.unwrap();
    let mut wire = WirePayload::decode(payload.as_slice()).unwrap();
    assert_eq!(wire.options_json, canonical);
    wire.options_json = r#"{"provider":"offline","batch_size":2}"#.into();
    assert_plan_error(
        &codec
            .try_decode(
                version,
                &wire.encode_to_vec(),
                &inputs,
                &ctx.task_ctx(),
                &converter,
                &root,
            )
            .unwrap_err(),
        "requires canonical options JSON",
    );
    Ok(())
}
