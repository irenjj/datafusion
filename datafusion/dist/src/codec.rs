// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: See the NOTICE file distributed with this work for additional information regarding copyright ownership
// SPDX-FileContributor: Licensed to the Apache Software Foundation (ASF) under one or more contributor license agreements

use std::sync::Arc;

use datafusion::common::{Result, plan_datafusion_err, plan_err};
use datafusion::execution::TaskContext;
use datafusion::llm::{LlmExec, canonical_options};
use datafusion::physical_plan::ExecutionPlan;
use datafusion_proto::physical_plan::{
    PhysicalExtensionCodec, PhysicalPlanDecodeContext, PhysicalProtoConverterExtension,
};
use datafusion_proto::protobuf;
use prost::Message;

/// Codec for the LlmExec extension. Built-in expressions use DataFusion's codec.
#[derive(Debug, Default)]
pub struct LlmPhysicalCodec;

#[derive(Clone, PartialEq, Message)]
struct LlmExecNode {
    #[prost(message, optional, tag = "1")]
    prompt: Option<protobuf::PhysicalExprNode>,
    #[prost(string, tag = "2")]
    model: String,
    #[prost(string, tag = "3")]
    options_json: String,
    #[prost(string, tag = "4")]
    output_col: String,
}

impl PhysicalExtensionCodec for LlmPhysicalCodec {
    fn try_decode(
        &self,
        buf: &[u8],
        inputs: &[Arc<dyn ExecutionPlan>],
        ctx: &TaskContext,
        converter: &dyn PhysicalProtoConverterExtension,
    ) -> Result<Arc<dyn ExecutionPlan>> {
        let [input] = inputs else {
            return plan_err!("LlmExec requires exactly one input");
        };
        let node = LlmExecNode::decode(buf)
            .map_err(|error| plan_datafusion_err!("Invalid LlmExec protobuf: {error}"))?;
        if node.model.trim().is_empty() || node.output_col.trim().is_empty() {
            return plan_err!("LlmExec requires a model and output column");
        }
        let options_json = canonical_options(&node.options_json)?;
        let prompt = node.prompt.as_ref().ok_or_else(|| {
            plan_datafusion_err!("LlmExec protobuf is missing its prompt")
        })?;
        let prompt = converter.proto_to_physical_expr(
            prompt,
            &input.schema(),
            &PhysicalPlanDecodeContext::new(ctx, self),
        )?;
        Ok(Arc::new(LlmExec::new(
            Arc::clone(input),
            prompt,
            node.model,
            options_json,
            node.output_col,
        )))
    }

    fn try_encode(
        &self,
        node: Arc<dyn ExecutionPlan>,
        buf: &mut Vec<u8>,
        converter: &dyn PhysicalProtoConverterExtension,
    ) -> Result<()> {
        let Some(exec) = node.downcast_ref::<LlmExec>() else {
            return plan_err!("Unsupported physical extension {}", node.name());
        };
        if exec.model().trim().is_empty() || exec.output_col().trim().is_empty() {
            return plan_err!("LlmExec requires a model and output column");
        }
        let node = LlmExecNode {
            prompt: Some(converter.physical_expr_to_proto(exec.prompt(), self)?),
            model: exec.model().to_owned(),
            options_json: canonical_options(exec.options_json())?,
            output_col: exec.output_col().to_owned(),
        };
        node.encode(buf).map_err(|error| {
            plan_datafusion_err!("Cannot encode LlmExec protobuf: {error}")
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use datafusion::arrow::datatypes::Schema;
    use datafusion::common::ScalarValue;
    use datafusion::physical_expr::expressions::Literal;
    use datafusion::physical_plan::empty::EmptyExec;
    use datafusion::prelude::SessionContext;
    use datafusion_proto::physical_plan::DefaultPhysicalProtoConverter;

    #[test]
    fn validates_extension_configuration_on_both_sides_of_the_wire() -> Result<()> {
        let codec = LlmPhysicalCodec;
        let converter = DefaultPhysicalProtoConverter {};
        let task_ctx = SessionContext::new().task_ctx();
        let input: Arc<dyn ExecutionPlan> =
            Arc::new(EmptyExec::new(Arc::new(Schema::empty())));
        let prompt: Arc<dyn datafusion::physical_expr::PhysicalExpr> =
            Arc::new(Literal::new(ScalarValue::from("prompt")));
        let options = r#"{ "z": 1, "a": 2 }"#;
        let exec = |model: &str, options: &str, output: &str| {
            Arc::new(LlmExec::new(
                Arc::clone(&input),
                Arc::clone(&prompt),
                model.into(),
                options.into(),
                output.into(),
            ))
        };
        let mut bytes = Vec::new();
        codec.try_encode(exec("m", options, "resp"), &mut bytes, &converter)?;
        let mut node = LlmExecNode::decode(bytes.as_slice()).unwrap();
        assert_eq!(node.options_json, r#"{"a":2,"z":1}"#);
        node.options_json = options.into();
        let decoded = codec.try_decode(
            &node.encode_to_vec(),
            std::slice::from_ref(&input),
            &task_ctx,
            &converter,
        )?;
        assert_eq!(
            decoded.downcast_ref::<LlmExec>().unwrap().options_json(),
            r#"{"a":2,"z":1}"#
        );
        for (model, options, output, message) in [
            (" ", "{}", "resp", "requires a model and output column"),
            ("m", "{}", "", "requires a model and output column"),
            (
                "m",
                r#"{"a":{"token":"test"}}"#,
                "resp",
                "sensitive credential key",
            ),
        ] {
            let error = codec
                .try_encode(exec(model, options, output), &mut Vec::new(), &converter)
                .unwrap_err();
            assert!(error.to_string().contains(message), "{error}");
            node.model = model.into();
            node.options_json = options.into();
            node.output_col = output.into();
            let error = codec
                .try_decode(
                    &node.encode_to_vec(),
                    std::slice::from_ref(&input),
                    &task_ctx,
                    &converter,
                )
                .unwrap_err();
            assert!(error.to_string().contains(message), "{error}");
        }
        Ok(())
    }
}
