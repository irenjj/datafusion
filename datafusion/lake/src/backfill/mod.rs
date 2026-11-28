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

use arrow::array::RecordBatch;
use arrow::datatypes::Schema;
use datafusion_common::Result;
use futures::Stream;
use lance_io::object_store::ObjectStoreParams;
use lance_table_runtime::TableBuilder;
use url::Url;

use crate::LakeObjectStores;

pub use plan::{BackfillPlan, BackfillReceipt, BackfillWorkUnit};

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

/// Process-local storage binding for a backfill stage. Pass `&LakeObjectStores`
/// or the existing `&ObjectStoreParams` to each of the backfill operations.
///
/// # Public Only for Internal Use:
/// This is not a public API and is for internal use only; see [API policy] for details.
/// [API policy]: https://datafusion.apache.org/contributor-guide/api-health.html#internal-public-apis
#[doc(hidden)]
#[derive(Clone, Copy)]
pub struct BackfillRuntime<'a>(StoreBinding<'a>);

#[derive(Clone, Copy)]
enum StoreBinding<'a> {
    Stores(&'a LakeObjectStores),
    LanceParams(&'a ObjectStoreParams),
}

impl<'a> From<&'a LakeObjectStores> for BackfillRuntime<'a> {
    fn from(stores: &'a LakeObjectStores) -> Self {
        Self(StoreBinding::Stores(stores))
    }
}

impl<'a> From<&'a ObjectStoreParams> for BackfillRuntime<'a> {
    fn from(params: &'a ObjectStoreParams) -> Self {
        Self(StoreBinding::LanceParams(params))
    }
}

impl BackfillRuntime<'_> {
    fn lance_builder(self, uri: &Url) -> TableBuilder {
        match self.0 {
            StoreBinding::LanceParams(params) => {
                TableBuilder::from_uri(uri.as_str()).with_store_params(params.clone())
            }
            StoreBinding::Stores(stores) => stores.lance_table_builder(uri),
        }
    }
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
pub async fn prepare_backfill<'a>(
    target: BackfillTarget<'_>,
    new_columns: &Schema,
    runtime: impl Into<BackfillRuntime<'a>>,
) -> Result<(BackfillPlan, Vec<BackfillWorkUnit>)> {
    match target {
        BackfillTarget::Lance(uri) => {
            lance::prepare(uri, new_columns, runtime.into()).await
        }
        BackfillTarget::Paimon(table) => paimon::prepare(table, new_columns).await,
    }
}

/// Write only the new columns for one attempt. The stream must supply every row
/// in physical offset order, including deleted offsets, with the planned Arrow
/// schema on every batch. Wrap a single batch in [futures::stream::iter].
/// The store must support [`object_store::CopyMode::Create`] to publish attempt
/// files. Enable `S3CopyIfNotExists::Multipart` when building an Amazon S3 store.
///
/// # Errors
/// Returns an error for invalid inputs, storage failures, or a previously published
/// file for this work unit and attempt ID. Use a new attempt ID when retrying.
///
/// # Public Only for Internal Use:
/// This is not a public API and is for internal use only; see [API policy] for details.
/// [API policy]: https://datafusion.apache.org/contributor-guide/api-health.html#internal-public-apis
#[doc(hidden)]
pub async fn write_backfill_unit<'a>(
    plan: &BackfillPlan,
    work_unit: &BackfillWorkUnit,
    attempt_id: &str,
    batches: impl Stream<Item = Result<RecordBatch>> + Send,
    runtime: impl Into<BackfillRuntime<'a>>,
) -> Result<BackfillReceipt> {
    plan.validate()?;
    lance::write_unit(plan, work_unit, attempt_id, batches, runtime.into()).await
}

/// Attach exactly one selected attempt per work unit and publish the schema and
/// column files in a single coordinator commit. Returns the new dataset version.
/// For S3 the store must support atomic [`object_store::PutMode::Create`].
///
/// # Errors
/// Returns an error for incomplete or inconsistent receipts, missing artifacts,
/// or a concurrent change to the pinned dataset.
///
/// # Public Only for Internal Use:
/// This is not a public API and is for internal use only; see [API policy] for details.
/// [API policy]: https://datafusion.apache.org/contributor-guide/api-health.html#internal-public-apis
#[doc(hidden)]
pub async fn finalize_backfill<'a>(
    plan: &BackfillPlan,
    receipts: &[BackfillReceipt],
    runtime: impl Into<BackfillRuntime<'a>>,
) -> Result<u64> {
    let winners = plan.validate_receipts(receipts)?;
    lance::finalize(plan, &winners, runtime.into()).await
}
