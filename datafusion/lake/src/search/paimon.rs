// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: See the NOTICE file distributed with this work for additional information regarding copyright ownership
// SPDX-FileContributor: Licensed to the Apache Software Foundation (ASF) under one or more contributor license agreements

use std::collections::HashMap;
use std::sync::Arc;

use arrow::datatypes::SchemaRef;
use datafusion_common::{Result, exec_datafusion_err, exec_err, plan_err};
use datafusion_physical_plan::ExecutionPlan;
use paimon::spec::{Predicate, PredicateBuilder};
use paimon::table::GlobalIndexResult;
use paimon::{RowRange, Table};

use crate::PaimonReadOnlyTable;
use crate::filter;
use crate::paimon_table::paimon_scan_error;

use super::{FusedRow, RankedRow, SearchRequest, TextQuery};

const TABLE_REQUIREMENT: &str = "Paimon vector_search and hybrid_search require a data-evolution table with row tracking and global indexes. Primary-key search is unsupported";

pub(super) fn validate_table(table: &PaimonReadOnlyTable) -> Result<()> {
    let schema = table.table().schema();
    let options = schema.core_options();
    if !options.data_evolution_enabled()
        || !options.global_index_enabled()
        || !options.row_tracking_enabled()
        || !schema.primary_keys().is_empty()
    {
        return plan_err!("{TABLE_REQUIREMENT}");
    }
    Ok(())
}

pub(super) fn predicate(
    table: &PaimonReadOnlyTable,
    request: &SearchRequest,
) -> Result<Option<Predicate>> {
    let builder = PredicateBuilder::new(table.table().schema().fields());
    request
        .filter
        .as_ref()
        .map(|expr| filter::paimon(expr, &builder, false))
        .transpose()
}

pub(super) async fn search(
    table: &PaimonReadOnlyTable,
    request: &SearchRequest,
    filter: Option<&Predicate>,
) -> Result<(Vec<RankedRow>, Vec<RankedRow>)> {
    let table = table.table();
    // Scalar FAST mode can discard matches when a predicate has no scalar
    // index. Refine against the pinned data while keeping the retrieval indexes.
    let refined = filter.map(|_| {
        table.copy_with_options(HashMap::from([
            ("scalar-index.search-mode".into(), "full".into()),
            ("global-index.filter.refine-from-data".into(), "true".into()),
        ]))
    });
    let table = refined.as_ref().unwrap_or(table);
    let mut builder = table.new_vector_search_builder();
    builder
        .with_vector_column(&request.vector_column)
        .with_query_vector(request.vector.clone())
        .with_limit(request.k);
    if let Some(filter) = filter {
        builder.with_filter(filter.clone());
    }
    let vector = async {
        ranked_rows(
            &Box::pin(builder.execute())
                .await
                .map_err(paimon_scan_error)?,
        )
    };
    if let Some(text) = &request.text {
        futures::try_join!(vector, text_search(table, request.k, text, filter))
    } else {
        Ok((vector.await?, Vec::new()))
    }
}

async fn text_search(
    table: &Table,
    k: usize,
    text: &TextQuery,
    filter: Option<&Predicate>,
) -> Result<Vec<RankedRow>> {
    let mut builder = table.new_full_text_search_builder();
    builder
        .with_text_column(&text.column)
        .with_query_text(&text.query)
        .with_limit(k);
    if let Some(filter) = filter {
        builder.with_filter(filter.clone());
    }
    ranked_rows(
        &Box::pin(builder.execute())
            .await
            .map_err(paimon_scan_error)?,
    )
}

fn ranked_rows(result: &GlobalIndexResult) -> Result<Vec<RankedRow>> {
    let hits = result
        .row_ids()
        .map_err(|err| exec_datafusion_err!("{TABLE_REQUIREMENT}: {err}"))?;
    if hits.row_ids.len() != hits.scores.len() {
        return exec_err!("Paimon search returned misaligned row IDs and scores");
    }
    if hits.scores.iter().any(|score| !score.is_finite()) {
        return exec_err!("Paimon search returned non-finite scores");
    }
    if hits.scores.windows(2).any(|pair| pair[0] < pair[1]) {
        return exec_err!("Paimon search returned scores out of best-first order");
    }
    Ok(hits
        .row_ids
        .iter()
        .map(|&row_key| RankedRow {
            row_key,
            row_addr: None,
        })
        .collect())
}

fn row_ranges(keys: impl IntoIterator<Item = u64>) -> Result<Vec<RowRange>> {
    let mut keys = keys
        .into_iter()
        .map(|key| {
            i64::try_from(key)
                .map_err(|_| exec_datafusion_err!("Paimon row ID {key} exceeds i64::MAX"))
        })
        .collect::<Result<Vec<_>>>()?;
    keys.sort_unstable();
    keys.dedup();
    let mut ranges = Vec::new();
    for group in keys.chunk_by(|left, right| left.checked_add(1) == Some(*right)) {
        let from = group[0];
        let to = group[group.len() - 1];
        if from > to {
            return exec_err!("Invalid Paimon late read row range {from}..={to}");
        }
        ranges.push(RowRange::new(from, to));
    }
    Ok(ranges)
}

pub(super) async fn scan_with_row_ranges(
    table: &PaimonReadOnlyTable,
    schema: SchemaRef,
    rows: &[FusedRow],
) -> Result<Arc<dyn ExecutionPlan>> {
    let ranges = row_ranges(rows.iter().map(|row| row.row_key))?;
    table.scan_row_ranges(schema, ranges).await
}

#[cfg(test)]
#[path = "../../tests/search_contract/paimon_ranges.rs"]
mod tests;
