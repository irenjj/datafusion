// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: See the NOTICE file distributed with this work for additional information regarding copyright ownership
// SPDX-FileContributor: Licensed to the Apache Software Foundation (ASF) under one or more contributor license agreements

use std::sync::Arc;

use arrow::array::{Int32Array, RecordBatch};
use arrow::datatypes::{DataType, Field, Schema};
use datafusion_lake::backfill::{
    BackfillPlan, BackfillReceipt, BackfillTarget, BackfillWorkUnit, finalize_backfill,
    prepare_backfill, write_backfill_unit,
};
use futures::stream;
use lance_file::version::LanceFileVersion;
use lance_io::object_store::ObjectStoreParams;
use lance_table_runtime::write::{InsertBuilder, WriteParams};
use serde_json::{Value, json};
use sha2::{Digest, Sha256};

#[tokio::test]
async fn backfill_wire_round_trips_and_rejects_invalid_messages() {
    let dir = tempfile::tempdir().unwrap();
    let uri = dir.path().to_str().unwrap();
    InsertBuilder::new(uri)
        .with_params(&WriteParams::with_storage_version(LanceFileVersion::V2_2))
        .execute(vec![column("id")])
        .await
        .unwrap();
    let runtime = ObjectStoreParams::default();
    let new_columns = column("value");
    let (plan, units) =
        prepare_backfill(BackfillTarget::Lance(uri), &new_columns.schema(), &runtime)
            .await
            .unwrap();
    let plan_bytes = plan.to_json_bytes().unwrap();
    let transported = BackfillPlan::from_json_bytes(&plan_bytes).unwrap();
    assert_eq!(plan_bytes, transported.to_json_bytes().unwrap());
    let unit_bytes = units[0].to_json_bytes(&plan).unwrap();
    let unit = BackfillWorkUnit::from_json_bytes(&unit_bytes, &transported).unwrap();
    assert_eq!(unit, units[0]);
    let receipt = write_backfill_unit(
        &transported,
        &unit,
        "worker-1",
        stream::iter([Ok(new_columns)]),
        &runtime,
    )
    .await
    .unwrap();
    let receipt_bytes = receipt.to_json_bytes(&transported).unwrap();
    let receipt = BackfillReceipt::from_json_bytes(&receipt_bytes, &plan).unwrap();
    assert_eq!(receipt_bytes, receipt.to_json_bytes(&plan).unwrap());

    let bad_plan = |edit: fn(&mut Value), expected: &str| {
        let err = BackfillPlan::from_json_bytes(&change(&plan_bytes, edit)).unwrap_err();
        assert!(err.to_string().contains(expected), "{err}");
    };
    bad_plan(|wire| wire["magic"] = json!("df.lake.split"), "magic");
    bad_plan(|wire| wire["version"] = json!(2), "envelope version");
    bad_plan(
        |wire| wire["payload"]["plan_fingerprint"] = json!("0".repeat(64)),
        "fingerprint",
    );
    bad_plan(
        |wire| {
            let payload = &mut wire["payload"]["payload"];
            payload["version"] = json!(1);
            payload.as_object_mut().unwrap().remove("fragments");
        },
        "Unsupported backfill plan version 1",
    );
    bad_plan(
        |wire| {
            wire["payload"]["payload"]["row_count"] = json!(3);
            refresh_fingerprint(wire);
        },
        "row count does not match",
    );
    bad_plan(
        |wire| {
            let payload = &mut wire["payload"]["payload"];
            let rows = u64::from(u32::MAX) + 1;
            payload["fragments"][0]["unit"]["row_count"] = json!(rows);
            payload["row_count"] = json!(rows);
            refresh_fingerprint(wire);
        },
        "exceeds u32 rows",
    );
    bad_plan(
        |wire| {
            let layout = &mut wire["payload"]["payload"]["fragments"][0];
            layout["deletion_file_rev"] = json!(1);
            layout["deleted_ranges"] = json!([{ "start": 1, "end": 3 }]);
            refresh_fingerprint(wire);
        },
        "Invalid deletion ranges",
    );
    let oversized = vec![b' '; 16 * 1024 * 1024 + 1];
    let err = BackfillPlan::from_json_bytes(&oversized).unwrap_err();
    assert!(err.to_string().contains("byte limit"), "{err}");
    assert!(BackfillPlan::from_json_bytes(&plan_bytes[..plan_bytes.len() - 1]).is_err());
    assert!(BackfillPlan::from_json_bytes(&serde_json::to_vec(&plan).unwrap()).is_err());

    let mismatch = change(&unit_bytes, |wire| {
        wire["payload"]["work_unit"]["row_count"] = json!(1);
    });
    assert!(BackfillWorkUnit::from_json_bytes(&mismatch, &plan).is_err());
    let mismatch = change(&unit_bytes, |wire| {
        wire["payload"]["work_unit"]["fragment_id"] = json!(u64::from(u32::MAX) + 1);
    });
    assert!(BackfillWorkUnit::from_json_bytes(&mismatch, &plan).is_err());
    let unknown_version = change(&unit_bytes, |wire| wire["version"] = json!(2));
    assert!(BackfillWorkUnit::from_json_bytes(&unknown_version, &plan).is_err());
    assert!(BackfillWorkUnit::from_json_bytes(&receipt_bytes, &plan).is_err());

    let other_plan = prepare_backfill(
        BackfillTarget::Lance(uri),
        &column("other").schema(),
        &runtime,
    )
    .await
    .unwrap()
    .0;
    assert!(BackfillWorkUnit::from_json_bytes(&unit_bytes, &other_plan).is_err());
    assert!(BackfillReceipt::from_json_bytes(&receipt_bytes, &other_plan).is_err());
    let bad_receipt = |edit: fn(&mut Value), expected: &str| {
        let err = BackfillReceipt::from_json_bytes(&change(&receipt_bytes, edit), &plan)
            .unwrap_err();
        assert!(err.to_string().contains(expected), "{err}");
    };
    bad_receipt(|wire| wire["version"] = json!(2), "envelope version");
    bad_receipt(
        |wire| wire["payload"]["plan_fingerprint"] = json!("0".repeat(64)),
        "fingerprint",
    );
    bad_receipt(
        |wire| wire["payload"]["row_count"] = json!(1),
        "does not match its work unit",
    );
    bad_receipt(
        |wire| {
            wire["payload"]["row_count"] = json!(1);
            wire["payload"]["work_unit"]["row_count"] = json!(1);
        },
        "does not match the pinned fragment",
    );
    bad_receipt(
        |wire| wire["payload"]["attempt_id"] = json!("different-attempt"),
        "Invalid backfill data file",
    );
    bad_receipt(
        |wire| wire["payload"]["data_file"]["column_indices"] = json!([100]),
        "Invalid backfill data file",
    );
    assert_eq!(
        finalize_backfill(&plan, &[receipt], &runtime)
            .await
            .unwrap(),
        plan.dataset_version() + 1,
    );
}

fn column(name: &str) -> RecordBatch {
    RecordBatch::try_new(
        Arc::new(Schema::new(vec![Field::new(name, DataType::Int32, true)])),
        vec![Arc::new(Int32Array::from(vec![1, 2]))],
    )
    .unwrap()
}

fn change(bytes: &[u8], edit: fn(&mut Value)) -> Vec<u8> {
    let mut wire = serde_json::from_slice(bytes).unwrap();
    edit(&mut wire);
    serde_json::to_vec(&wire).unwrap()
}

fn refresh_fingerprint(wire: &mut Value) {
    let payload = &mut wire["payload"]["payload"];
    payload.sort_all_objects();
    let fingerprint = Sha256::digest(serde_json::to_vec(payload).unwrap());
    wire["payload"]["plan_fingerprint"] = json!(hex::encode(fingerprint));
}
