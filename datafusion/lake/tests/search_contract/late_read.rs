// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: See the NOTICE file distributed with this work for additional information regarding copyright ownership
// SPDX-FileContributor: Licensed to the Apache Software Foundation (ASF) under one or more contributor license agreements

use std::sync::Arc;

use arrow::array::{Float32Array, Int64Array, RecordBatch};
use arrow::datatypes::{DataType, Field, Schema};
use datafusion_execution::TaskContext;
use datafusion_physical_plan::{collect, test::TestMemoryExec};
use paimon::spec::ROW_ID_FIELD_NAME;

use super::{FusedRow, FusedScoreExec};

#[tokio::test]
async fn null_paimon_row_id_errors_without_panicking() {
    let input = RecordBatch::try_from_iter([(
        ROW_ID_FIELD_NAME,
        Arc::new(Int64Array::from(vec![None])) as _,
    )])
    .unwrap();
    let plan = TestMemoryExec::try_new_exec(&[vec![input.clone()]], input.schema(), None)
        .unwrap();
    let plan = FusedScoreExec::try_new(
        plan,
        Arc::new(Schema::empty()),
        vec![],
        ROW_ID_FIELD_NAME,
        vec![FusedRow {
            row_key: 0,
            row_addr: None,
            score: 1.0 / 61.0,
        }],
    )
    .unwrap();
    let result = collect(Arc::new(plan), Arc::new(TaskContext::default())).await;
    assert!(
        result
            .unwrap_err()
            .to_string()
            .contains("Null fused row key")
    );
}

#[tokio::test]
async fn late_read_discards_superset_rows_and_reports_missing_keys() {
    let input = RecordBatch::try_from_iter([
        ("id", Arc::new(Int64Array::from(vec![10, 30, 20])) as _),
        (
            ROW_ID_FIELD_NAME,
            Arc::new(Int64Array::from(vec![1, 3, 2])) as _,
        ),
    ])
    .unwrap();
    let schema = Arc::new(Schema::new(vec![
        input.schema().field(0).clone(),
        Field::new("_score", DataType::Float32, false),
    ]));
    let input = TestMemoryExec::try_new_exec(
        &[vec![input]],
        Arc::new(Schema::new(vec![
            schema.field(0).clone(),
            Field::new(ROW_ID_FIELD_NAME, DataType::Int64, false),
        ])),
        None,
    )
    .unwrap();

    let run = |keys: [u64; 2]| {
        let fused = keys
            .into_iter()
            .enumerate()
            .map(|(index, row_key)| FusedRow {
                row_key,
                row_addr: None,
                score: 1.0 / (index + 1) as f64,
            })
            .collect();
        let plan = FusedScoreExec::try_new(
            input.clone(),
            schema.clone(),
            vec![Some(0), None],
            ROW_ID_FIELD_NAME,
            fused,
        )
        .unwrap();
        collect(Arc::new(plan), Arc::new(TaskContext::default()))
    };
    let batches = run([2, 1]).await.unwrap();
    assert_eq!(
        batches[0]["id"]
            .as_any()
            .downcast_ref::<Int64Array>()
            .unwrap()
            .values(),
        &[20, 10]
    );
    assert_eq!(
        batches[0]["_score"]
            .as_any()
            .downcast_ref::<Float32Array>()
            .unwrap()
            .values(),
        &[1.0, 0.5]
    );
    let error = run([2, 4]).await.unwrap_err();
    assert!(error.to_string().contains("Missing fused row key 4"));
}
