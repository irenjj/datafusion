// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: See the NOTICE file distributed with this work for additional information regarding copyright ownership
// SPDX-FileContributor: Licensed to the Apache Software Foundation (ASF) under one or more contributor license agreements

//! Snapshot-pinned, attempt-scoped column backfills.
//!
//! # Public Only for Internal Use:
//! This is not a public API and is for internal use only; see [API policy] for details.
//! [API policy]: https://datafusion.apache.org/contributor-guide/api-health.html#internal-public-apis

mod lance;
mod paimon;
mod plan;
mod stage;
mod wire;

use arrow::array::RecordBatch;
use arrow::datatypes::Schema;
use datafusion_common::Result;
use futures::Stream;
use lance_io::object_store::ObjectStoreParams;

pub use plan::{BackfillPlan, BackfillReceipt, BackfillWorkUnit};
pub use stage::bridge_backfill_stage;

/// The existing table that the coordinator will backfill.
///
/// # Public Only for Internal Use:
/// This is not a public API and is for internal use only; see [API policy] for details.
/// [API policy]: https://datafusion.apache.org/contributor-guide/api-health.html#internal-public-apis
#[doc(hidden)]
#[derive(Debug, Clone, Copy)]
pub enum BackfillTarget<'a> {
    /// Lance dataset at an absolute filesystem path or credential-free URL.
    Lance(&'a str),
    /// Catalog-resolved Paimon table.
    Paimon(&'a ::paimon::Table),
}

/// Pin the current snapshot and enumerate physical work units without writing to the table.
/// Supply store settings locally at each stage.
///
/// # Errors
/// Returns an error for an invalid target, a conflicting column, or an unsupported backfill.
///
/// # Public Only for Internal Use:
/// This is not a public API and is for internal use only; see [API policy] for details.
/// [API policy]: https://datafusion.apache.org/contributor-guide/api-health.html#internal-public-apis
#[doc(hidden)]
pub async fn prepare_backfill(
    target: BackfillTarget<'_>,
    new_columns: &Schema,
    runtime: &ObjectStoreParams,
) -> Result<(BackfillPlan, Vec<BackfillWorkUnit>)> {
    match target {
        BackfillTarget::Lance(uri) => lance::prepare(uri, new_columns, runtime).await,
        BackfillTarget::Paimon(table) => paimon::prepare(table, new_columns).await,
    }
}

/// Write only the new columns for one attempt. The stream must supply every row
/// in physical offset order, including deleted offsets, with the planned Arrow
/// schema on every batch. Use [`bridge_backfill_stage`] to adapt a logical stream
/// of visible rows from a distributed stage. Wrap a single batch in [futures::stream::iter].
///
/// # Errors
/// Returns an error for invalid inputs, storage failures, or a previously published
/// file for this work unit and attempt ID. Use a new attempt ID when retrying.
///
/// # Public Only for Internal Use:
/// This is not a public API and is for internal use only; see [API policy] for details.
/// [API policy]: https://datafusion.apache.org/contributor-guide/api-health.html#internal-public-apis
#[doc(hidden)]
pub async fn write_backfill_unit(
    plan: &BackfillPlan,
    work_unit: &BackfillWorkUnit,
    attempt_id: &str,
    batches: impl Stream<Item = Result<RecordBatch>> + Send,
    runtime: &ObjectStoreParams,
) -> Result<BackfillReceipt> {
    plan.validate()?;
    lance::write_unit(plan, work_unit, attempt_id, batches, runtime).await
}

/// Attach exactly one selected attempt per work unit and publish the schema and
/// column files in a single coordinator commit. Returns the new dataset version.
///
/// # Errors
/// Returns an error for incomplete or inconsistent receipts, missing artifacts,
/// or a concurrent change to the pinned dataset.
///
/// # Public Only for Internal Use:
/// This is not a public API and is for internal use only; see [API policy] for details.
/// [API policy]: https://datafusion.apache.org/contributor-guide/api-health.html#internal-public-apis
#[doc(hidden)]
pub async fn finalize_backfill(
    plan: &BackfillPlan,
    receipts: &[BackfillReceipt],
    runtime: &ObjectStoreParams,
) -> Result<u64> {
    let winners = plan.validate_receipts(receipts)?;
    lance::finalize(plan, &winners, runtime).await
}
