// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: See the NOTICE file distributed with this work for additional information regarding copyright ownership
// SPDX-FileContributor: Licensed to the Apache Software Foundation (ASF) under one or more contributor license agreements

use arrow::datatypes::DataType;
use datafusion_common::{Result, plan_err};

/// Read constraints checked when opening a lake table.
///
/// # Public Only for Internal Use:
///
/// This is not a public API and is for internal use only; see [API policy] for details.
/// [API policy]: https://datafusion.apache.org/contributor-guide/api-health.html#internal-public-apis
#[doc(hidden)]
#[derive(Debug, Clone, Default)]
pub struct LakeTableOptions {
    /// Every listed index must be usable at the pinned snapshot.
    pub required_indices: Vec<RequiredIndexSpec>,
}

/// One column and kind of index required at table open.
///
/// # Public Only for Internal Use:
///
/// This is not a public API and is for internal use only; see [API policy] for details.
/// [API policy]: https://datafusion.apache.org/contributor-guide/api-health.html#internal-public-apis
#[doc(hidden)]
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum RequiredIndexSpec {
    /// A vector index at the expected fixed width.
    Vector {
        /// Indexed column.
        column: String,
        /// Number of values per vector.
        dimension: u32,
    },
    /// A full-text index.
    FullText {
        /// Indexed column.
        column: String,
    },
    /// An index that supports scalar predicates.
    Scalar {
        /// Indexed column.
        column: String,
    },
}

impl RequiredIndexSpec {
    pub(crate) fn column(&self) -> &str {
        match self {
            Self::Vector { column, .. }
            | Self::FullText { column }
            | Self::Scalar { column } => column,
        }
    }

    pub(crate) fn validate_dimension(&self, data_type: &DataType) -> Result<()> {
        if let Self::Vector { column, dimension } = self {
            let actual = match data_type {
                DataType::FixedSizeList(_, size) => u32::try_from(*size).ok(),
                _ => None,
            };
            if *dimension == 0 || actual != Some(*dimension) {
                return plan_err!(
                    "Lake vector index on {column:?} requires dimension {dimension}, column type is {data_type}"
                );
            }
        }
        Ok(())
    }
}

const MIME_KEYS: [&str; 3] = ["df.media.mime", "mime", "content-type"];

pub(crate) fn blob_mime<'a>(
    column: &str,
    column_metadata: impl Fn(&str) -> Option<&'a String>,
    table_metadata: impl Fn(&str) -> Option<&'a String>,
) -> Result<Option<String>> {
    let mut selected = None;
    for value in MIME_KEYS
        .iter()
        .filter_map(|key| column_metadata(key))
        .chain(
            MIME_KEYS
                .iter()
                .filter_map(|key| table_metadata(&format!("fields.{column}.{key}"))),
        )
        .chain(MIME_KEYS.iter().filter_map(|key| table_metadata(key)))
    {
        if value.trim().is_empty() {
            return plan_err!("Lake blob column {column:?} has blank MIME metadata");
        }
        selected.get_or_insert_with(|| value.clone());
    }
    Ok(selected)
}
