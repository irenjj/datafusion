// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: See the NOTICE file distributed with this work for additional information regarding copyright ownership
// SPDX-FileContributor: Licensed to the Apache Software Foundation (ASF) under one or more contributor license agreements

use std::sync::Arc;

use async_trait::async_trait;
use datafusion::arrow::array::{Array, AsArray};
use datafusion::arrow::datatypes::{DataType, Field, Int64Type};
use datafusion::common::Result;
use datafusion::llm::{
    CancellationToken, LlmBackend, LlmBackendError, LlmRequest, LlmResponse,
};
use datafusion::prelude::{SessionConfig, SessionContext, col, lit};
use datafusion_ai::AiErrorType;

#[derive(Debug)]
struct AssemblyBackend;

#[async_trait]
impl LlmBackend for AssemblyBackend {
    async fn generate(
        &self,
        request: LlmRequest,
        _cancel: CancellationToken,
    ) -> std::result::Result<LlmResponse, LlmBackendError> {
        let outputs = request
            .prompts
            .iter()
            .map(|prompt| {
                let text = match prompt.as_str() {
                    "valid" => "```json\n{\"count\":7,\"accepted\":true}\n```",
                    "wrong type" => r#"{"count":"7","accepted":true}"#,
                    prompt => panic!("unexpected prompt {prompt}"),
                };
                Some(text.to_owned())
            })
            .collect();
        Ok(LlmResponse {
            outputs,
            usage: None,
        })
    }
}

#[tokio::test]
async fn dataframe_and_sql_preserve_typed_ai_envelopes() -> Result<()> {
    let ctx =
        SessionContext::new_with_config(SessionConfig::new().with_target_partitions(1));
    ctx.state()
        .llm_provider_registry()
        .register("assembly", Arc::new(AssemblyBackend))?;
    let prompts = ctx
        .sql("SELECT * FROM (VALUES (2, 'wrong type'), (1, 'valid'), (3, 'filtered')) AS t(id, prompt)")
        .await?
        .filter(col("id").lt(lit(3_i64)))?;
    ctx.register_table("prompts", prompts.into_view())?;
    let inference = ctx
        .sql(
            r#"SELECT id, llm(prompt, 'test-model', '{
                "provider":"assembly", "schema_mode":"prompt_only", "on_error":"envelope",
                "response_schema":{
                    "type":"object", "properties":{
                        "count":{"type":"integer"}, "accepted":{"type":"boolean"}
                    }, "required":["count", "accepted"]
                }
            }') AS answer FROM prompts"#,
        )
        .await?
        .sort(vec![col("id").sort(true, false)])?;
    let value_type = DataType::Struct(
        vec![
            Field::new("count", DataType::Int64, false),
            Field::new("accepted", DataType::Boolean, false),
        ]
        .into(),
    );
    let expected = AiErrorType::new(value_type).field("answer", false)?;
    assert_eq!(inference.schema().field(1).as_ref(), &expected);

    let batches = inference.collect().await?;
    assert_eq!(batches.len(), 1);
    let batch = &batches[0];
    assert_eq!(batch.schema().field(1), &expected);
    assert_eq!(
        batch.column(0).as_primitive::<Int64Type>().values(),
        &[1, 2]
    );
    let answer = batch.column(1).as_struct();
    let values = answer.column(0).as_struct();
    assert!(values.is_valid(0) && values.is_null(1));
    assert_eq!(values.column(0).as_primitive::<Int64Type>().value(0), 7);
    assert!(values.column(1).as_boolean().value(0));
    assert_eq!(
        answer
            .column(1)
            .as_string::<i32>()
            .iter()
            .collect::<Vec<_>>(),
        [None, Some("SCHEMA_MISMATCH")]
    );
    let messages = answer.column(2).as_string::<i32>();
    assert!(messages.is_null(0));
    assert!(messages.value(1).contains("64-bit integer"));
    assert_eq!(
        answer.column(3).as_boolean().iter().collect::<Vec<_>>(),
        [None, Some(false)]
    );
    Ok(())
}
