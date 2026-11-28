// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: See the NOTICE file distributed with this work for additional information regarding copyright ownership
// SPDX-FileContributor: Licensed to the Apache Software Foundation (ASF) under one or more contributor license agreements

use std::sync::Arc;

use datafusion::common::{Result, plan_datafusion_err, plan_err};
use datafusion::execution::TaskContext;
use datafusion::physical_plan::ExecutionPlan;
use datafusion_ai::llm::{LlmExec, MediaPlan, TaskKind};
use datafusion_proto::physical_plan::{
    PhysicalExtensionCodec, PhysicalPlanDecodeContext, PhysicalProtoConverterExtension,
    from_proto::parse_physical_expr_with_converter,
    to_proto::serialize_physical_expr_with_converter,
};
use datafusion_proto::protobuf::PhysicalExprNode;
use prost::Message;

use crate::{ExtensionOperatorPolicy, NamedPhysicalExtensionCodec};

const VERSION: u32 = 2;

/// Versioned transport for [`LlmExec`]. The worker's [`TaskContext`] must
/// provide an [`LlmProviderRegistry`](datafusion_ai::llm::LlmProviderRegistry).
/// Vision plans also require a [`MediaResolver`](datafusion_ai::MediaResolver)
/// registered in the worker's session config.
/// Plans using this codec disallow speculation.
///
/// ```
/// use std::sync::Arc;
/// use datafusion_dist::{DistExtensionRegistry, LlmExecCodec};
///
/// let mut registry = DistExtensionRegistry::default();
/// registry.register(Arc::new(LlmExecCodec::new()))?;
/// # Ok::<(), datafusion::common::DataFusionError>(())
/// ```
#[derive(Debug, Default)]
pub struct LlmExecCodec;

impl LlmExecCodec {
    /// Create a codec for [`LlmExec`].
    pub fn new() -> Self {
        Self
    }
}

#[derive(Message)]
struct LlmExecPayload {
    #[prost(message, optional, tag = "1")]
    prompt: Option<PhysicalExprNode>,
    #[prost(string, tag = "2")]
    model: String,
    #[prost(string, tag = "3")]
    options_json: String,
    #[prost(string, tag = "4")]
    output_col: String,
    #[prost(uint32, tag = "5")]
    kind: u32,
    #[prost(message, optional, tag = "6")]
    media: Option<PhysicalExprNode>,
}

impl NamedPhysicalExtensionCodec for LlmExecCodec {
    fn extension_name(&self) -> &str {
        "df.llm.exec"
    }

    fn operator_policy(
        &self,
        node: &dyn ExecutionPlan,
    ) -> Option<ExtensionOperatorPolicy> {
        node.is::<LlmExec>().then_some(ExtensionOperatorPolicy {
            map_only: true,
            speculation_allowed: false,
        })
    }

    fn try_encode(
        &self,
        node: &Arc<dyn ExecutionPlan>,
        converter: &dyn PhysicalProtoConverterExtension,
        root_codec: &dyn PhysicalExtensionCodec,
    ) -> Result<Option<(u32, Vec<u8>)>> {
        let Some(node) = node.downcast_ref::<LlmExec>() else {
            return Ok(None);
        };
        let prompt =
            serialize_physical_expr_with_converter(node.prompt(), root_codec, converter)?;
        let kind = match node.task_kind() {
            TaskKind::Chat => 0,
            TaskKind::Embed => 1,
            TaskKind::Vision => 2,
        };
        let media = node
            .media()
            .map(|plan| {
                serialize_physical_expr_with_converter(plan.expr(), root_codec, converter)
            })
            .transpose()?;
        let payload = LlmExecPayload {
            prompt: Some(prompt),
            model: node.model().to_owned(),
            options_json: node.options_json().to_owned(),
            output_col: node.output_col().to_owned(),
            kind,
            media,
        };
        Ok(Some((VERSION, payload.encode_to_vec())))
    }

    fn try_decode(
        &self,
        version: u32,
        payload: &[u8],
        inputs: &[Arc<dyn ExecutionPlan>],
        ctx: &TaskContext,
        converter: &dyn PhysicalProtoConverterExtension,
        root_codec: &dyn PhysicalExtensionCodec,
    ) -> Result<Arc<dyn ExecutionPlan>> {
        if !matches!(version, 1 | VERSION) {
            return plan_err!("Unsupported LlmExec schema version {version}");
        }
        let [input] = inputs else {
            return plan_err!("LlmExec requires exactly one input");
        };
        let payload = LlmExecPayload::decode(payload)
            .map_err(|error| plan_datafusion_err!("Invalid LlmExec protobuf: {error}"))?;
        if version == 1 && (payload.kind != 0 || payload.media.is_some()) {
            return plan_err!(
                "LlmExec schema version 1 only supports Chat without media"
            );
        }
        let prompt = payload.prompt.ok_or_else(|| {
            plan_datafusion_err!("LlmExec payload is missing its prompt")
        })?;
        let schema = input.schema();
        let decode_ctx = PhysicalPlanDecodeContext::new(ctx, root_codec);
        let prompt =
            parse_physical_expr_with_converter(&prompt, &schema, &decode_ctx, converter)?;
        let llm = if version == 1 {
            LlmExec::try_new_from_task_context(
                Arc::clone(input),
                prompt,
                payload.model,
                &payload.options_json,
                payload.output_col,
                ctx,
            )?
        } else {
            let kind = match payload.kind {
                0 => TaskKind::Chat,
                1 => TaskKind::Embed,
                2 => TaskKind::Vision,
                kind => return plan_err!("Unsupported LlmExec task kind {kind}"),
            };
            let media = match (kind, payload.media) {
                (TaskKind::Vision, Some(media)) => {
                    Some(MediaPlan::new(parse_physical_expr_with_converter(
                        &media,
                        &schema,
                        &decode_ctx,
                        converter,
                    )?))
                }
                (TaskKind::Vision, None) => {
                    return plan_err!("Vision requires a media expression");
                }
                (_, Some(_)) => return plan_err!("Only vision inference accepts media"),
                (_, None) => None,
            };
            LlmExec::try_new_task_from_task_context(
                Arc::clone(input),
                prompt,
                media,
                kind,
                payload.model,
                &payload.options_json,
                payload.output_col,
                ctx,
            )?
        };
        if llm.options_json() != payload.options_json {
            return plan_err!("LlmExec payload requires canonical options JSON");
        }
        Ok(Arc::new(llm))
    }
}
