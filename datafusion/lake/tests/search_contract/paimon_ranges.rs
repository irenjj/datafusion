// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: See the NOTICE file distributed with this work for additional information regarding copyright ownership
// SPDX-FileContributor: Licensed to the Apache Software Foundation (ASF) under one or more contributor license agreements

use datafusion_common::DataFusionError;
use paimon::table::GlobalIndexResult;
use paimon::vector_search::ScoredRowIds;

use super::{ranked_rows, row_ranges};

#[test]
fn rejects_row_keys_above_i64_max_without_panicking() {
    let result = std::panic::catch_unwind(|| row_ranges([i64::MAX as u64 + 1]));
    let error = result
        .expect("range construction must not panic")
        .unwrap_err();
    assert!(error.to_string().contains("exceeds i64::MAX"));
}

#[test]
fn validates_best_first_scores() {
    for (scores, expected) in [
        ([f32::NAN, 1.0], "non-finite scores"),
        ([f32::INFINITY, 1.0], "non-finite scores"),
        ([1.0, f32::NEG_INFINITY], "non-finite scores"),
        ([1.0, 2.0], "scores out of best-first order"),
    ] {
        let result = GlobalIndexResult::from_scored_row_ids(ScoredRowIds::new(
            vec![7, 3],
            scores.to_vec(),
        ));
        let error = ranked_rows(&result).unwrap_err();
        assert!(
            matches!(error, DataFusionError::Execution(ref message) if message.contains(expected)),
            "{error}"
        );
    }

    let result = GlobalIndexResult::from_scored_row_ids(ScoredRowIds::new(
        vec![7, 3, 4],
        vec![2.0, 2.0, 1.0],
    ));
    let row_keys: Vec<_> = ranked_rows(&result)
        .unwrap()
        .into_iter()
        .map(|row| row.row_key)
        .collect();
    assert_eq!(row_keys, [7, 3, 4]);
}
