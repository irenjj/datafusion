// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: See the NOTICE file distributed with this work for additional information regarding copyright ownership
// SPDX-FileContributor: Licensed to the Apache Software Foundation (ASF) under one or more contributor license agreements

use datafusion_common::DataFusionError;
use datafusion_datasource::FileExtensions;
use datafusion_lake::{
    InclusiveRowRange, LAKE_SPLIT_EXT_NAME, LAKE_SPLIT_EXT_VERSION, LakeSplitExt,
    LakeSplitPayload,
};
use serde_json::json;

#[test]
fn lance_split_codec_round_trip_preserves_unrelated_extensions() {
    let mut original = FileExtensions::default();
    assert!(
        LakeSplitExt::try_encode_from_extensions(&original)
            .unwrap()
            .is_none()
    );
    let split = LakeSplitExt {
        version: 1,
        split_id: "fragment-7".into(),
        payload: LakeSplitPayload::Lance {
            dataset_uri: "file:///test.lance".into(),
            dataset_version: 5,
            fragment_id: 7,
            row_ranges: Some(vec![
                InclusiveRowRange { from: 0, to: 2 },
                InclusiveRowRange { from: 5, to: 9 },
            ]),
            deletion_file_rev: Some(3),
        },
    };
    original.insert(split.clone());
    let (version, bytes) = LakeSplitExt::try_encode_from_extensions(&original)
        .unwrap()
        .unwrap();
    assert_eq!(LAKE_SPLIT_EXT_NAME, "df.lake.split");
    assert_eq!(version, LAKE_SPLIT_EXT_VERSION);
    let mut decoded = FileExtensions::default();
    decoded.insert(42u64);
    LakeSplitExt::try_decode_into_extensions(version, &bytes, &mut decoded).unwrap();
    assert_eq!(decoded.get::<LakeSplitExt>(), Some(&split));
    assert_eq!(decoded.get::<u64>(), Some(&42));
    assert_eq!(
        LakeSplitExt::try_encode_from_extensions(&decoded).unwrap(),
        Some((version, bytes))
    );
}

#[test]
fn split_codec_rejects_versions_invalid_ranges_and_corrupt_payloads() {
    let valid = json!({
        "version": 1,
        "split_id": "id",
        "payload": {
            "engine": "lance",
            "dataset_uri": "file:///test.lance",
            "dataset_version": 2,
            "fragment_id": 0,
            "row_ranges": [{"from": 0, "to": 1}],
            "deletion_file_rev": 1
        }
    });
    let mut ext = FileExtensions::default();
    for (version, payload) in [
        (2, serde_json::to_vec(&valid).unwrap()),
        (1, b"{broken".to_vec()),
    ] {
        assert!(matches!(
            LakeSplitExt::try_decode_into_extensions(version, &payload, &mut ext),
            Err(DataFusionError::Plan(_))
        ));
    }
    for (path, value) in [
        ("/version", json!(2)),
        ("/split_id", json!(" \t")),
        ("/payload/dataset_uri", json!("")),
        ("/payload/dataset_version", json!(0)),
        ("/payload/row_ranges", json!([])),
        ("/payload/row_ranges/0/from", json!(2)),
    ] {
        let mut invalid = valid.clone();
        *invalid.pointer_mut(path).unwrap() = value;
        let bytes = serde_json::to_vec(&invalid).unwrap();
        assert!(matches!(
            LakeSplitExt::try_decode_into_extensions(1, &bytes, &mut ext),
            Err(DataFusionError::Plan(_))
        ));
    }
    let paimon = json!({
        "version": 1, "split_id": "id",
        "payload": {
            "engine": "paimon", "table_uri": "memory://fixture/table",
            "snapshot_id": 1, "bucket": 0, "partition_key_b64": "AAAAAA==",
            "raw_convertible": true, "row_ranges": null, "data_split_json": "[]"
        }
    });
    assert!(matches!(
        LakeSplitExt::try_decode_into_extensions(
            1,
            &serde_json::to_vec(&paimon).unwrap(),
            &mut ext
        ),
        Err(DataFusionError::Plan(_))
    ));
    assert!(ext.get::<LakeSplitExt>().is_none());
}
