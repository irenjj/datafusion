// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: See the NOTICE file distributed with this work for additional information regarding copyright ownership
// SPDX-FileContributor: Licensed to the Apache Software Foundation (ASF) under one or more contributor license agreements

use std::collections::BTreeMap;
use std::sync::Arc;

use arrow::array::{Array, Float32Array, RecordBatch, UInt64Array};
use arrow::datatypes::SchemaRef;
use datafusion_common::{Result, exec_datafusion_err, exec_err};
use datafusion_physical_plan::ExecutionPlan;
use futures::TryStreamExt;
use lance::dataset::scanner::Scanner;
use lance_core::datatypes::BlobHandling;
use lance_core::utils::address::RowAddress;
use lance_core::{ROW_ADDR, ROW_ID};
use lance_index::scalar::FullTextSearchQuery;

use crate::lance_table::lance_scan_error;
use crate::{InclusiveRowRange, LanceReadOnlyTable};

use super::{FusedRow, RankedRow, SearchRequest, TextQuery};

const DISTANCE: &str = "_distance";

pub(super) async fn search(
    table: &LanceReadOnlyTable,
    request: &SearchRequest,
) -> Result<(Vec<RankedRow>, Vec<RankedRow>)> {
    let query = Float32Array::from(request.vector.clone());
    let mut vector = narrow_scan(table, request)?;
    vector
        .nearest(&request.vector_column, &query, request.k)
        .map_err(lance_scan_error)?;
    table
        .select_vector_index(&request.vector_column, &mut vector)
        .await?;
    let vector = ranked_rows(vector, DISTANCE, request.k);
    if let Some(text) = &request.text {
        futures::try_join!(vector, text_search(table, request, text))
    } else {
        Ok((vector.await?, Vec::new()))
    }
}

async fn text_search(
    table: &LanceReadOnlyTable,
    request: &SearchRequest,
    text: &TextQuery,
) -> Result<Vec<RankedRow>> {
    let mut scanner = narrow_scan(table, request)?;
    let query = FullTextSearchQuery::new(text.query.clone())
        .with_column(text.column.clone())
        .map_err(lance_scan_error)?
        .limit(Some(request.k as i64));
    scanner.full_text_search(query).map_err(lance_scan_error)?;
    ranked_rows(scanner, super::SCORE, request.k).await
}

fn narrow_scan(table: &LanceReadOnlyTable, request: &SearchRequest) -> Result<Scanner> {
    let mut scanner = table.dataset().scan();
    scanner
        .empty_project()
        .map_err(lance_scan_error)?
        .blob_handling(BlobHandling::BlobsDescriptions)
        .with_row_id()
        .with_row_address()
        .prefilter(true);
    if let Some(filter) = &request.filter {
        scanner.filter_expr(filter.clone());
    }
    Ok(scanner)
}

fn column<'a, T: Array + 'static>(batch: &'a RecordBatch, name: &str) -> Result<&'a T> {
    batch
        .column_by_name(name)
        .and_then(|column| column.as_any().downcast_ref::<T>())
        .ok_or_else(|| {
            exec_datafusion_err!("Lance search omitted or mistyped column {name:?}")
        })
}

async fn ranked_rows(scanner: Scanner, score: &str, k: usize) -> Result<Vec<RankedRow>> {
    let mut stream = scanner.try_into_stream().await.map_err(lance_scan_error)?;
    let mut ranked = Vec::with_capacity(k);
    while let Some(batch) = stream.try_next().await.map_err(lance_scan_error)? {
        let ids = column::<UInt64Array>(&batch, ROW_ID)?;
        let addresses = column::<UInt64Array>(&batch, ROW_ADDR)?;
        let scores = column::<Float32Array>(&batch, score)?;
        for ((id, addr), raw) in ids.iter().zip(addresses.iter()).zip(scores.iter()) {
            let (Some(row_key), Some(row_addr), Some(raw)) = (id, addr, raw) else {
                return exec_err!("Lance search returned a null score or row identity");
            };
            if !raw.is_finite() {
                return exec_err!("Lance search returned a non-finite score");
            }
            let relevance = if score == DISTANCE { -raw } else { raw };
            ranked.push((
                RankedRow {
                    row_key,
                    row_addr: Some(row_addr),
                },
                relevance,
            ));
        }
    }
    ranked.sort_unstable_by(|(left, a), (right, b)| {
        b.total_cmp(a).then(left.row_key.cmp(&right.row_key))
    });
    Ok(ranked.into_iter().take(k).map(|(hit, _)| hit).collect())
}

pub(super) fn scan_with_row_addrs(
    table: &LanceReadOnlyTable,
    schema: SchemaRef,
    rows: &[FusedRow],
) -> Result<Arc<dyn ExecutionPlan>> {
    let mut fragments = BTreeMap::<_, Vec<_>>::new();
    for row in rows {
        let addr = row.row_addr.ok_or_else(|| {
            exec_datafusion_err!(
                "Lance search row {} has no physical address",
                row.row_key
            )
        })?;
        let addr = RowAddress::from(addr);
        let offset = u64::from(addr.row_offset());
        fragments
            .entry(addr.fragment_id())
            .or_default()
            .push(InclusiveRowRange {
                from: offset,
                to: offset,
            });
    }
    for ranges in fragments.values_mut() {
        ranges.sort_unstable_by_key(|range| range.from);
    }
    table.scan_row_ranges(schema, fragments)
}
