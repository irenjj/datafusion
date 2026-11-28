// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: See the NOTICE file distributed with this work for additional information regarding copyright ownership
// SPDX-FileContributor: Licensed to the Apache Software Foundation (ASF) under one or more contributor license agreements

use std::collections::{HashMap, HashSet};
use std::path::Path;

use arrow::datatypes::{DataType, Fields, Schema};
use datafusion_common::{Result, not_impl_err, plan_datafusion_err, plan_err};
use lance::table::format::DataFile;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use url::Url;

/// Snapshot and Arrow column schema shared by the coordinator and workers.
///
/// # Public Only for Internal Use:
/// This is not a public API and is for internal use only; see [API policy] for details.
/// [API policy]: https://datafusion.apache.org/contributor-guide/api-health.html#internal-public-apis
#[doc(hidden)]
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct BackfillPlan {
    /// SHA-256 of the versioned payload with JSON object keys sorted recursively.
    pub plan_fingerprint: String,
    pub(super) payload: PlanPayload,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub(super) struct PlanPayload {
    pub(super) version: u8,
    pub(super) dataset_uri: String,
    pub(super) dataset_version: u64,
    pub(super) manifest_identity: String,
    #[serde(with = "schema_wire")]
    pub(super) new_columns: Schema,
    pub(super) unit_count: usize,
    pub(super) row_count: u64,
}

/// One physical Lance fragment to backfill.
///
/// # Public Only for Internal Use:
/// This is not a public API and is for internal use only; see [API policy] for details.
/// [API policy]: https://datafusion.apache.org/contributor-guide/api-health.html#internal-public-apis
#[doc(hidden)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct BackfillWorkUnit {
    /// Physical fragment ID in the pinned manifest.
    pub fragment_id: u32,
    /// Physical row count, including any deleted offsets.
    pub row_count: u64,
}

/// An immutable worker artifact eligible for selection by the coordinator.
///
/// # Public Only for Internal Use:
/// This is not a public API and is for internal use only; see [API policy] for details.
/// [API policy]: https://datafusion.apache.org/contributor-guide/api-health.html#internal-public-apis
#[doc(hidden)]
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct BackfillReceipt {
    /// Fingerprint of the plan that produced the artifact.
    pub plan_fingerprint: String,
    /// Physical fragment covered by the artifact.
    pub work_unit: BackfillWorkUnit,
    /// Attempt identity assigned by the caller.
    pub attempt_id: String,
    /// Location of the staged Lance column file.
    pub artifact_uri: String,
    /// Number of rows written by the worker.
    pub row_count: u64,
    pub(super) data_file: DataFile,
}

impl BackfillPlan {
    pub(super) fn new(payload: PlanPayload) -> Result<Self> {
        let plan_fingerprint = fingerprint(&payload)?;
        Ok(Self {
            plan_fingerprint,
            payload,
        })
    }

    /// Credential-free URI of the planned dataset.
    ///
    /// # Public Only for Internal Use:
    /// This is not a public API and is for internal use only; see [API policy] for details.
    /// [API policy]: https://datafusion.apache.org/contributor-guide/api-health.html#internal-public-apis
    #[must_use]
    #[doc(hidden)]
    pub fn dataset_uri(&self) -> &str {
        &self.payload.dataset_uri
    }

    /// The dataset version whose physical offsets the worker must use.
    ///
    /// # Public Only for Internal Use:
    /// This is not a public API and is for internal use only; see [API policy] for details.
    /// [API policy]: https://datafusion.apache.org/contributor-guide/api-health.html#internal-public-apis
    #[must_use]
    #[doc(hidden)]
    pub fn dataset_version(&self) -> u64 {
        self.payload.dataset_version
    }

    /// The exact Arrow schema expected in the worker's batches.
    ///
    /// # Public Only for Internal Use:
    /// This is not a public API and is for internal use only; see [API policy] for details.
    /// [API policy]: https://datafusion.apache.org/contributor-guide/api-health.html#internal-public-apis
    #[must_use]
    #[doc(hidden)]
    pub fn new_columns(&self) -> &Schema {
        &self.payload.new_columns
    }

    pub(super) fn validate(&self) -> Result<()> {
        if self.payload.version != 1 {
            return plan_err!(
                "Unsupported backfill plan version {}",
                self.payload.version
            );
        }
        if self.plan_fingerprint != fingerprint(&self.payload)? {
            return plan_err!("Backfill plan fingerprint does not match its payload");
        }
        Ok(())
    }

    pub(super) fn validate_receipts<'a>(
        &self,
        receipts: &'a [BackfillReceipt],
    ) -> Result<HashMap<u32, &'a BackfillReceipt>> {
        self.validate()?;
        let mut winners = HashMap::with_capacity(receipts.len());
        let mut row_count = 0;
        for receipt in receipts {
            if receipt.plan_fingerprint != self.plan_fingerprint {
                return plan_err!("Backfill receipt fingerprint does not match the plan");
            }
            if receipt.row_count != receipt.work_unit.row_count {
                return plan_err!(
                    "Backfill receipt row count does not match its work unit"
                );
            }
            let fragment = receipt.work_unit.fragment_id;
            if winners.insert(fragment, receipt).is_some() {
                return plan_err!(
                    "Duplicate backfill receipt for Lance fragment {fragment}"
                );
            }
            row_count = add_rows(row_count, receipt.row_count)?;
        }
        if winners.len() != self.payload.unit_count {
            return plan_err!(
                "Expected {} backfill receipts, received {}",
                self.payload.unit_count,
                winners.len()
            );
        }
        if row_count != self.payload.row_count {
            return plan_err!(
                "Expected {} backfill rows, received {row_count}",
                self.payload.row_count
            );
        }
        Ok(winners)
    }
}

fn fingerprint(payload: &PlanPayload) -> Result<String> {
    let mut canonical = serde_json::to_value(payload)
        .map_err(|e| plan_datafusion_err!("Cannot encode backfill plan: {e}"))?;
    canonical.sort_all_objects();
    let bytes = serde_json::to_vec(&canonical)
        .map_err(|e| plan_datafusion_err!("Cannot encode backfill plan: {e}"))?;
    Ok(hex::encode(Sha256::digest(bytes)))
}

mod schema_wire {
    use arrow::datatypes::Schema;
    use arrow::ipc::{
        convert::{IpcSchemaEncoder, try_fb_to_schema},
        root_as_schema,
    };
    use base64::{Engine, engine::general_purpose::STANDARD};
    use serde::{Deserialize, Deserializer, Serializer, de::Error};

    pub(super) fn serialize<S: Serializer>(
        schema: &Schema,
        serializer: S,
    ) -> Result<S::Ok, S::Error> {
        let ipc = IpcSchemaEncoder::new().schema_to_fb(schema);
        serializer.serialize_str(&STANDARD.encode(ipc.finished_data()))
    }

    pub(super) fn deserialize<'de, D: Deserializer<'de>>(
        deserializer: D,
    ) -> Result<Schema, D::Error> {
        let encoded = String::deserialize(deserializer)?;
        let bytes = STANDARD.decode(encoded).map_err(D::Error::custom)?;
        let schema = root_as_schema(&bytes).map_err(D::Error::custom)?;
        try_fb_to_schema(schema).map_err(D::Error::custom)
    }
}

pub(super) fn add_rows(total: u64, rows: u64) -> Result<u64> {
    total
        .checked_add(rows)
        .ok_or_else(|| plan_datafusion_err!("Backfill row count overflows u64"))
}

pub(super) fn dataset_url(uri: &str) -> Result<Url> {
    let url = if Path::new(uri).is_absolute() {
        Url::from_file_path(uri).ok()
    } else {
        Url::parse(uri).ok()
    }
    .ok_or_else(|| plan_datafusion_err!("Backfill requires an absolute dataset URI"))?;
    if url.cannot_be_a_base()
        || !url.username().is_empty()
        || url.password().is_some()
        || url.query().is_some()
        || url.fragment().is_some()
    {
        return plan_err!("Backfill requires a credential-free dataset URI");
    }
    Ok(url)
}

pub(super) fn validate_columns<'a>(
    existing: impl IntoIterator<Item = &'a str>,
    columns: &Schema,
) -> Result<()> {
    let existing: HashSet<_> = existing.into_iter().collect();
    if let Some(field) = columns
        .fields()
        .iter()
        .find(|field| existing.contains(field.name().as_str()))
    {
        return plan_err!("Backfill column {:?} already exists", field.name());
    }
    validate_fields(columns.fields())
}

fn validate_fields(fields: &Fields) -> Result<()> {
    if fields.is_empty() {
        return plan_err!(
            "Backfill requires a nonempty set of new columns and struct fields"
        );
    }
    let mut names = HashSet::with_capacity(fields.len());
    for field in fields {
        if field.name().trim().is_empty() || !names.insert(field.name()) {
            return plan_err!("Invalid or duplicate backfill field {:?}", field.name());
        }
        if field.extension_type_name() == Some("df.ai_error")
            || matches!(
                field.data_type(),
                DataType::FixedSizeList(item, _) if item.extension_type_name() == Some("df.ai_error")
            )
        {
            return plan_err!(
                "Unpack df.ai_error before backfilling field {:?}",
                field.name()
            );
        }
        match field.data_type() {
            DataType::Struct(fields) => validate_fields(fields)?,
            DataType::FixedSizeList(item, dim)
                if *dim > 0
                    && item.data_type() == &DataType::Float32
                    && item.extension_type_name().is_none()
                    && item.extension_type_metadata().is_none() => {}
            DataType::Boolean
            | DataType::Int8
            | DataType::Int16
            | DataType::Int32
            | DataType::Int64
            | DataType::UInt8
            | DataType::UInt16
            | DataType::UInt32
            | DataType::UInt64
            | DataType::Float32
            | DataType::Float64
            | DataType::Utf8
            | DataType::LargeUtf8
            | DataType::Binary
            | DataType::LargeBinary => {}
            data_type => {
                return not_impl_err!("Unsupported backfill Arrow type {data_type}");
            }
        }
    }
    Ok(())
}
