// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: See the NOTICE file distributed with this work for additional information regarding copyright ownership
// SPDX-FileContributor: Licensed to the Apache Software Foundation (ASF) under one or more contributor license agreements

//! Indexed retrieval on snapshot-pinned lake tables.
//!
//! # Public Only for Internal Use:
//!
//! This is not a public API and is for internal use only; see [API policy] for details.
//! [API policy]: https://datafusion.apache.org/contributor-guide/api-health.html#internal-public-apis

mod exec;
mod lance;
mod paimon;
mod udtf;
mod validate;

use std::collections::{HashMap, HashSet};
use std::sync::{Arc, RwLock};

use arrow::datatypes::SchemaRef;
use datafusion_catalog::TableProvider;
use datafusion_common::{
    Result, exec_err, internal_datafusion_err, plan_datafusion_err, plan_err,
};
use datafusion_expr::Expr;

use crate::{LanceReadOnlyTable, PaimonReadOnlyTable, RequiredIndexSpec};

pub use udtf::{hybrid_search_function, vector_search_function};

const SCORE: &str = "_score";
const DEFAULT_RRF_K: u64 = 60;

#[derive(Debug)]
struct TextQuery {
    column: String,
    query: String,
    weight: f64,
}

/// A validated search over one [`PinnedLakeTable`].
///
/// # Public Only for Internal Use:
///
/// This is not a public API and is for internal use only; see [API policy] for details.
/// [API policy]: https://datafusion.apache.org/contributor-guide/api-health.html#internal-public-apis
#[doc(hidden)]
#[derive(Debug)]
pub struct SearchRequest {
    vector_column: String,
    vector: Vec<f32>,
    text: Option<TextQuery>,
    k: usize,
    rrf_k: u64,
    vector_weight: f64,
    filter: Option<Expr>,
}

/// A hit in a best-first list supplied to [`rrf_fuse`].
///
/// # Public Only for Internal Use:
///
/// This is not a public API and is for internal use only; see [API policy] for details.
/// [API policy]: https://datafusion.apache.org/contributor-guide/api-health.html#internal-public-apis
#[doc(hidden)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RankedRow {
    /// Stable row key in the pinned snapshot.
    pub row_key: u64,
    /// Physical address for Lance fragment reads.
    pub row_addr: Option<u64>,
}

/// A hit in the fused result of [`rrf_fuse`].
///
/// # Public Only for Internal Use:
///
/// This is not a public API and is for internal use only; see [API policy] for details.
/// [API policy]: https://datafusion.apache.org/contributor-guide/api-health.html#internal-public-apis
#[doc(hidden)]
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct FusedRow {
    /// Stable row key in the pinned snapshot.
    pub row_key: u64,
    /// Physical address for Lance fragment reads.
    pub row_addr: Option<u64>,
    /// Reciprocal rank score before conversion to the result column's type.
    pub score: f64,
}

/// Fuse best-first hit lists using one-based weighted reciprocal ranks.
///
/// # Errors
///
/// Returns a planning error for invalid parameters or an execution error for
/// inconsistent row identities or duplicate hits within a route.
///
/// # Public Only for Internal Use:
///
/// This is not a public API and is for internal use only; see [API policy] for details.
/// [API policy]: https://datafusion.apache.org/contributor-guide/api-health.html#internal-public-apis
#[doc(hidden)]
pub fn rrf_fuse(
    rrf_k: u64,
    k: usize,
    routes: &[(f64, &[RankedRow])],
) -> Result<Vec<FusedRow>> {
    validate::limits(k, rrf_k)?;
    let mut fused = HashMap::new();
    for &(weight, hits) in routes {
        validate::weight(weight, "route weight")?;
        let mut seen = HashSet::with_capacity(hits.len());
        for (index, hit) in hits.iter().enumerate() {
            let entry = fused.entry(hit.row_key).or_insert(FusedRow {
                row_key: hit.row_key,
                row_addr: hit.row_addr,
                score: 0.0,
            });
            if !seen.insert(hit.row_key) || entry.row_addr != hit.row_addr {
                return exec_err!("Inconsistent ranked hit for lake row {}", hit.row_key);
            }
            entry.score += weight / (rrf_k as f64 + (index + 1) as f64);
        }
    }
    let mut rows: Vec<_> = fused.into_values().collect();
    rows.sort_unstable_by(|left, right| {
        right
            .score
            .total_cmp(&left.score)
            .then(left.row_key.cmp(&right.row_key))
    });
    rows.truncate(k);
    Ok(rows)
}

#[derive(Debug)]
enum Engine {
    Lance(LanceReadOnlyTable),
    Paimon(PaimonReadOnlyTable),
}

/// An open snapshot handle for registration in a [`LakeSearchCatalog`].
///
/// # Public Only for Internal Use:
///
/// This is not a public API and is for internal use only; see [API policy] for details.
/// [API policy]: https://datafusion.apache.org/contributor-guide/api-health.html#internal-public-apis
#[doc(hidden)]
#[derive(Debug)]
pub struct PinnedLakeTable(Engine);

impl PinnedLakeTable {
    /// Return the lake table's schema, including descriptor types for media columns.
    ///
    /// # Public Only for Internal Use:
    ///
    /// This is not a public API and is for internal use only; see [API policy] for details.
    /// [API policy]: https://datafusion.apache.org/contributor-guide/api-health.html#internal-public-apis
    #[doc(hidden)]
    #[must_use]
    pub fn schema(&self) -> SchemaRef {
        match &self.0 {
            Engine::Lance(table) => table.schema(),
            Engine::Paimon(table) => table.schema(),
        }
    }

    fn validate_index(&self, required: &RequiredIndexSpec) -> Result<()> {
        let verified = match &self.0 {
            Engine::Lance(table) => table.has_required_index(required),
            Engine::Paimon(table) => table.has_required_index(required),
        };
        if !verified {
            return plan_err!(
                "Lake search requires index {required:?} to be validated in LakeTableOptions::required_indices at open"
            );
        }
        Ok(())
    }
}

impl From<LanceReadOnlyTable> for PinnedLakeTable {
    fn from(table: LanceReadOnlyTable) -> Self {
        Self(Engine::Lance(table))
    }
}

impl From<PaimonReadOnlyTable> for PinnedLakeTable {
    fn from(table: PaimonReadOnlyTable) -> Self {
        Self(Engine::Paimon(table))
    }
}

/// Resolves names to open [`PinnedLakeTable`] handles for synchronous UDTF planning.
///
/// # Public Only for Internal Use:
///
/// This is not a public API and is for internal use only; see [API policy] for details.
/// [API policy]: https://datafusion.apache.org/contributor-guide/api-health.html#internal-public-apis
#[doc(hidden)]
#[derive(Debug, Default)]
pub struct LakeSearchCatalog {
    tables: RwLock<HashMap<String, Arc<PinnedLakeTable>>>,
}

impl LakeSearchCatalog {
    /// Register an open table. Re-registration replaces the name for future calls.
    ///
    /// # Errors
    ///
    /// Returns an internal error if the catalog lock was poisoned.
    ///
    /// # Public Only for Internal Use:
    ///
    /// This is not a public API and is for internal use only; see [API policy] for details.
    /// [API policy]: https://datafusion.apache.org/contributor-guide/api-health.html#internal-public-apis
    #[doc(hidden)]
    pub fn register(
        &self,
        name: impl Into<String>,
        table: PinnedLakeTable,
    ) -> Result<()> {
        let name = name.into();
        self.tables
            .write()
            .map_err(|_| internal_datafusion_err!("Lake search catalog lock poisoned"))?
            .insert(name, Arc::new(table));
        Ok(())
    }

    fn resolve(&self, name: &str) -> Result<Arc<PinnedLakeTable>> {
        self.tables
            .read()
            .map_err(|_| internal_datafusion_err!("Lake search catalog lock poisoned"))?
            .get(name)
            .cloned()
            .ok_or_else(|| plan_datafusion_err!("Unknown lake search table {name:?}"))
    }
}
