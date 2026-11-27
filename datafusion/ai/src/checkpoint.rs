// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: See the NOTICE file distributed with this work for additional information regarding copyright ownership
// SPDX-FileContributor: Licensed to the Apache Software Foundation (ASF) under one or more contributor license agreements

use std::fmt::Debug;

use arrow::datatypes::Schema;
use arrow_schema::extension::ExtensionType;
use datafusion_common::{Result, plan_err};

use crate::AiErrorType;

/// An AI-produced column declared by a [`CheckpointableTarget`].
///
/// # Public Only for Internal Use:
///
/// This is not a public API and is for internal use only; see [API policy] for details.
/// [API policy]: https://datafusion.apache.org/contributor-guide/api-health.html#internal-public-apis
#[doc(hidden)]
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CheckpointOutputColumn {
    /// Zero-based column index in the operator's output [`Schema`].
    pub index: usize,
    /// Column name in the operator's output [`Schema`].
    pub name: String,
    /// Whether this column uses the [`AiErrorType`] (`df.ai_error`) extension.
    pub is_error_envelope: bool,
}

/// Output columns and row alignment guarantees for checkpointing an AI operator.
///
/// # Public Only for Internal Use:
///
/// This is not a public API and is for internal use only; see [API policy] for details.
/// [API policy]: https://datafusion.apache.org/contributor-guide/api-health.html#internal-public-apis
#[doc(hidden)]
pub trait CheckpointableTarget: Debug + Send + Sync {
    /// Number of leading input columns passed through verbatim at indices
    /// `0..passthrough_column_count()`.
    fn passthrough_column_count(&self) -> usize;

    /// AI-produced output columns, sorted in strictly increasing index order.
    /// Every index must be at least [`Self::passthrough_column_count`].
    fn ai_output_columns(&self) -> Vec<CheckpointOutputColumn>;

    /// Whether output rows follow the exact input row order within each partition.
    fn preserves_input_order(&self) -> bool;

    /// Whether the operator emits exactly one output row per input row.
    fn preserves_row_cardinality(&self) -> bool {
        true
    }
}

/// Validate the column declarations of a [`CheckpointableTarget`] against its
/// output [`Schema`].
///
/// # Errors
/// Returns a planning error for invalid column counts, indices, names, or
/// error envelope extension types.
///
/// # Public Only for Internal Use:
///
/// This is not a public API and is for internal use only; see [API policy] for details.
/// [API policy]: https://datafusion.apache.org/contributor-guide/api-health.html#internal-public-apis
#[doc(hidden)]
pub fn validate_checkpoint_target(
    target: &dyn CheckpointableTarget,
    output_schema: &Schema,
) -> Result<()> {
    let fields = output_schema.fields();
    let field_count = fields.len();
    let passthrough_count = target.passthrough_column_count();
    if passthrough_count > field_count {
        return plan_err!(
            "Checkpoint passthrough column count {passthrough_count} exceeds output schema field count {field_count}"
        );
    }

    let columns = target.ai_output_columns();
    if columns.is_empty() {
        return plan_err!("Checkpoint target must declare at least one AI output column");
    }

    let mut previous_index = None;
    for column in columns {
        let index = column.index;
        let Some(field) = fields.get(index) else {
            return plan_err!(
                "Checkpoint AI output column index {index} is out of bounds for {field_count} fields"
            );
        };
        if index < passthrough_count {
            return plan_err!(
                "Checkpoint AI output column index {index} overlaps the first {passthrough_count} passthrough columns"
            );
        }
        if let Some(previous) = previous_index
            && index <= previous
        {
            return plan_err!(
                "Checkpoint AI output column indices must be strictly increasing, got {index} after {previous}"
            );
        }
        if &column.name != field.name() {
            return plan_err!(
                "Checkpoint AI output column {index} declares name {:?}, but the output schema has {:?}",
                column.name,
                field.name()
            );
        }
        if column.is_error_envelope
            && let Err(error) = field.try_extension_type::<AiErrorType>()
        {
            return plan_err!(
                "Checkpoint AI output column {index} ({:?}) must use the {} extension: {error}",
                column.name,
                AiErrorType::NAME
            );
        }
        previous_index = Some(index);
    }
    Ok(())
}
