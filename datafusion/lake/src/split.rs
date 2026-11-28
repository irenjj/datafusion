// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: See the NOTICE file distributed with this work for additional information regarding copyright ownership
// SPDX-FileContributor: Licensed to the Apache Software Foundation (ASF) under one or more contributor license agreements

use base64::{Engine, engine::general_purpose::STANDARD};
use datafusion_common::{Result, plan_datafusion_err, plan_err};
use datafusion_datasource::{FileExtensions, PartitionedFile};
use paimon::{DataSplit, spec::BinaryRow};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

/// Stable name for the file extension understood by distributed scan codecs.
///
/// # Public Only for Internal Use:
///
/// This is not a public API and is for internal use only; see [API policy] for details.
/// [API policy]: https://datafusion.apache.org/contributor-guide/api-health.html#internal-public-apis
#[doc(hidden)]
pub const LAKE_SPLIT_EXT_NAME: &str = "df.lake.split";

/// The supported version of [`LAKE_SPLIT_EXT_NAME`].
///
/// # Public Only for Internal Use:
///
/// This is not a public API and is for internal use only; see [API policy] for details.
/// [API policy]: https://datafusion.apache.org/contributor-guide/api-health.html#internal-public-apis
#[doc(hidden)]
pub const LAKE_SPLIT_EXT_VERSION: u32 = 1;

/// An inclusive physical Lance row offset or Paimon row ID range.
///
/// # Public Only for Internal Use:
///
/// This is not a public API and is for internal use only; see [API policy] for details.
/// [API policy]: https://datafusion.apache.org/contributor-guide/api-health.html#internal-public-apis
#[doc(hidden)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct InclusiveRowRange {
    /// First selected row.
    pub from: u64,
    /// Last selected row.
    pub to: u64,
}

/// The snapshot-bound task consumed by the lake file opener.
///
/// # Public Only for Internal Use:
///
/// This is not a public API and is for internal use only; see [API policy] for details.
/// [API policy]: https://datafusion.apache.org/contributor-guide/api-health.html#internal-public-apis
#[doc(hidden)]
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "engine", rename_all = "snake_case", deny_unknown_fields)]
pub enum LakeSplitPayload {
    /// One fragment in a Lance manifest.
    Lance {
        /// Dataset location understood by Lance.
        dataset_uri: String,
        /// Manifest version selected when the table was opened.
        dataset_version: u64,
        /// Fragment ID in the pinned manifest.
        fragment_id: u32,
        /// Optional physical row offsets within the fragment.
        row_ranges: Option<Vec<InclusiveRowRange>>,
        /// Read version of the fragment's deletion file, when present.
        deletion_file_rev: Option<u64>,
    },
    /// One Paimon data split that may contain a group of merge-related files.
    Paimon {
        /// Table location understood by Paimon.
        table_uri: String,
        /// Snapshot ID selected when the table was opened.
        snapshot_id: i64,
        /// Bucket within the partition.
        bucket: i32,
        /// Standard base64 encoding of the Paimon binary partition row.
        partition_key_b64: String,
        /// Whether the engine can read this split without a merge.
        raw_convertible: bool,
        /// Selection in the serialized split's row coordinate system.
        row_ranges: Option<Vec<InclusiveRowRange>>,
        /// Serialized Paimon [`DataSplit`].
        data_split_json: String,
    },
}

/// Versioned file extension carried by a range-free [`PartitionedFile`].
///
/// # Public Only for Internal Use:
///
/// This is not a public API and is for internal use only; see [API policy] for details.
/// [API policy]: https://datafusion.apache.org/contributor-guide/api-health.html#internal-public-apis
#[doc(hidden)]
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct LakeSplitExt {
    /// Payload version, currently 1.
    pub version: u8,
    /// Deterministic identity assigned by planning.
    pub split_id: String,
    /// Engine-specific task.
    pub payload: LakeSplitPayload,
}

impl LakeSplitExt {
    pub(crate) fn into_file(
        payload: LakeSplitPayload,
        size: u64,
    ) -> Result<PartitionedFile> {
        let split_id = payload_id(&payload)?;
        let split = Self {
            version: 1,
            split_id: split_id.clone(),
            payload,
        };
        Ok(PartitionedFile::new(split_id, size).with_extension(split))
    }

    /// Encode the lake split when present in this map.
    ///
    /// # Errors
    /// Returns a plan error for an invalid extension.
    #[doc(hidden)]
    pub fn try_encode_from_extensions(
        extensions: &FileExtensions,
    ) -> Result<Option<(u32, Vec<u8>)>> {
        let Some(split) = extensions.get::<Self>() else {
            return Ok(None);
        };
        split.validate()?;
        Ok(Some((LAKE_SPLIT_EXT_VERSION, encode(split)?)))
    }

    /// Decode versioned bytes into the lake extension slot.
    ///
    /// # Errors
    /// Returns a plan error for unsupported versions or invalid payloads.
    #[doc(hidden)]
    pub fn try_decode_into_extensions(
        version: u32,
        payload: &[u8],
        extensions: &mut FileExtensions,
    ) -> Result<()> {
        if version != LAKE_SPLIT_EXT_VERSION {
            return plan_err!("Unsupported lake split wire version {version}");
        }
        let split: Self = serde_json::from_slice(payload)
            .map_err(|e| plan_datafusion_err!("Invalid lake split JSON: {e}"))?;
        split.validate()?;
        extensions.insert(split);
        Ok(())
    }

    pub(crate) fn for_read(file: &PartitionedFile) -> Result<&Self> {
        if file.range.is_some() {
            return plan_err!("Lake splits cannot have a byte range");
        }
        let split = file.extension::<Self>().ok_or_else(|| {
            plan_datafusion_err!("Missing {LAKE_SPLIT_EXT_NAME} extension")
        })?;
        split.validate_header()?;
        Ok(split)
    }

    pub(crate) fn validate_identity(&self) -> Result<()> {
        if self.split_id != payload_id(&self.payload)? {
            return plan_err!("Lake split_id does not match the SHA-256 of its payload");
        }
        Ok(())
    }

    fn validate_header(&self) -> Result<()> {
        if u32::from(self.version) != LAKE_SPLIT_EXT_VERSION {
            return plan_err!("Unsupported lake split payload version {}", self.version);
        }
        if self.split_id.trim().is_empty() {
            return plan_err!("Lake split_id must not be blank");
        }
        let (uri, ranges) = match &self.payload {
            LakeSplitPayload::Lance {
                dataset_uri,
                dataset_version,
                row_ranges,
                deletion_file_rev,
                ..
            } => {
                if *dataset_version == 0
                    || deletion_file_rev.is_some_and(|rev| rev > *dataset_version)
                {
                    return plan_err!("Invalid Lance split version or deletion revision");
                }
                (dataset_uri, row_ranges)
            }
            LakeSplitPayload::Paimon {
                table_uri,
                snapshot_id,
                row_ranges,
                ..
            } => {
                if *snapshot_id <= 0 {
                    return plan_err!("Invalid Paimon split snapshot ID {snapshot_id}");
                }
                (table_uri, row_ranges)
            }
        };
        if uri.trim().is_empty() {
            return plan_err!("Lake split table URI must not be blank");
        }
        if let Some(ranges) = ranges {
            if ranges.is_empty() {
                return plan_err!("Lake split row_ranges must be nonempty when present");
            }
            if ranges.iter().any(|r| r.from > r.to) {
                return plan_err!("Lake row ranges must satisfy from <= to");
            }
        }
        Ok(())
    }

    fn validate(&self) -> Result<()> {
        self.validate_header()?;
        if matches!(self.payload, LakeSplitPayload::Paimon { .. }) {
            self.payload.paimon_data_split()?;
        }
        Ok(())
    }
}

impl LakeSplitPayload {
    pub(crate) fn paimon_data_split(&self) -> Result<DataSplit> {
        let Self::Paimon {
            snapshot_id,
            bucket,
            partition_key_b64,
            raw_convertible,
            row_ranges,
            data_split_json,
            ..
        } = self
        else {
            return plan_err!("Expected a Paimon split");
        };
        let split: DataSplit = serde_json::from_str(data_split_json)
            .map_err(|e| plan_datafusion_err!("Invalid Paimon DataSplit JSON: {e}"))?;
        let partition = split.partition().to_serialized_bytes();
        let decoded_partition =
            BinaryRow::from_serialized_bytes(&partition).map_err(|e| {
                plan_datafusion_err!("Invalid Paimon split partition row: {e}")
            })?;
        if (decoded_partition.arity() != 0 && &decoded_partition != split.partition())
            || split.snapshot_id() != *snapshot_id
            || split.bucket() != *bucket
            || split.raw_convertible() != *raw_convertible
            || STANDARD.encode(partition) != *partition_key_b64
            || paimon_ranges(&split)? != *row_ranges
            || split.data_files().is_empty()
            || split.is_streaming()
            || split.bucket_path().trim().is_empty()
            || split
                .data_deletion_files()
                .is_some_and(|files| files.len() != split.data_files().len())
        {
            return plan_err!(
                "Paimon DataSplit metadata does not match its lake payload"
            );
        }
        Ok(split)
    }
}

pub(crate) fn partition_key(split: &DataSplit) -> String {
    STANDARD.encode(split.partition().to_serialized_bytes())
}

pub(crate) fn paimon_ranges(split: &DataSplit) -> Result<Option<Vec<InclusiveRowRange>>> {
    split
        .row_ranges()
        .map(|ranges| {
            ranges
                .iter()
                .map(|range| {
                    let from = u64::try_from(range.from()).map_err(|_| {
                        plan_datafusion_err!("Paimon row range has a negative start")
                    })?;
                    let to = u64::try_from(range.to()).map_err(|_| {
                        plan_datafusion_err!("Paimon row range has a negative end")
                    })?;
                    if from > to || to - from == i64::MAX.unsigned_abs() {
                        return plan_err!(
                            "Invalid inclusive Paimon row range {from}..={to}"
                        );
                    }
                    Ok(InclusiveRowRange { from, to })
                })
                .collect()
        })
        .transpose()
}

fn encode(value: &impl Serialize) -> Result<Vec<u8>> {
    serde_json::to_vec(value)
        .map_err(|e| plan_datafusion_err!("Cannot encode lake split: {e}"))
}

fn payload_id(payload: &LakeSplitPayload) -> Result<String> {
    Ok(hex::encode(Sha256::digest(encode(payload)?)))
}
