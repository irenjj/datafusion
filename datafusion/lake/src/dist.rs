// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: See the NOTICE file distributed with this work for additional information regarding copyright ownership
// SPDX-FileContributor: Licensed to the Apache Software Foundation (ASF) under one or more contributor license agreements

//! Snapshot-bound scan templates and split binding for distributed execution.
//!
//! # Public Only for Internal Use:
//!
//! This is not a public API and is for internal use only; see [API policy] for details.
//! [API policy]: https://datafusion.apache.org/contributor-guide/api-health.html#internal-public-apis

mod filter;

use std::path::Path;
use std::sync::Arc;

use arrow::datatypes::SchemaRef;
use datafusion_common::{Result, plan_datafusion_err, plan_err};
use datafusion_datasource::file_groups::FileGroup;
use datafusion_datasource::file_scan_config::{FileScanConfig, FileScanConfigBuilder};
use datafusion_datasource::source::DataSourceExec;
use datafusion_datasource::{FileExtensions, PartitionedFile};
use datafusion_execution::{TaskContext, config::SessionConfig};
use datafusion_physical_plan::ExecutionPlan;
use lance::dataset::builder::DatasetBuilder;
use paimon::io::{FileIO, FileIOBuilder};
use serde::{Deserialize, Serialize};
use url::Url;

use self::filter::ScanFilter;
use crate::lance_table::{LanceScan, LanceScanSpec};
use crate::paimon_table::{PaimonScan, PaimonScanSpec, paimon_scan_error};
use crate::scan::{LakeScan, LakeSource};
use crate::{LAKE_SPLIT_EXT_VERSION, LakeSplitExt, LakeSplitPayload, RequiredIndexSpec};

/// Stable wire name for a split-free lake scan template.
///
/// # Public Only for Internal Use:
///
/// This is not a public API and is for internal use only; see [API policy] for details.
/// [API policy]: https://datafusion.apache.org/contributor-guide/api-health.html#internal-public-apis
#[doc(hidden)]
pub const LAKE_SCAN_PROTOCOL_NAME: &str = "df.lake.scan";

/// Wire version of [`LAKE_SCAN_PROTOCOL_NAME`].
///
/// # Public Only for Internal Use:
///
/// This is not a public API and is for internal use only; see [API policy] for details.
/// [API policy]: https://datafusion.apache.org/contributor-guide/api-health.html#internal-public-apis
#[doc(hidden)]
pub const LAKE_SCAN_PROTOCOL_VERSION: u32 = 1;

const SCAN_MAGIC: &[u8] = b"DFLKS1\0";
const MAX_SCAN_BYTES: usize = 1024 * 1024;
const MAX_SPLIT_BYTES: usize = 16 * 1024 * 1024;

type StorageProvider<T> = Arc<dyn Fn(&str) -> Result<T> + Send + Sync>;

/// Worker-local providers for resolving storage when a split is opened.
///
/// # Public Only for Internal Use:
///
/// This is not a public API and is for internal use only; see [API policy] for details.
/// [API policy]: https://datafusion.apache.org/contributor-guide/api-health.html#internal-public-apis
#[doc(hidden)]
#[derive(Default)]
pub struct LakeRuntimeConfig {
    lance: Option<StorageProvider<DatasetBuilder>>,
    paimon: Option<StorageProvider<FileIO>>,
}

impl LakeRuntimeConfig {
    /// Supply a Lance builder for the dataset URI. The scan sets the pinned version.
    ///
    /// # Public Only for Internal Use:
    ///
    /// This is not a public API and is for internal use only; see [API policy] for details.
    /// [API policy]: https://datafusion.apache.org/contributor-guide/api-health.html#internal-public-apis
    #[doc(hidden)]
    #[must_use]
    pub fn with_lance_builder(
        mut self,
        provider: impl Fn(&str) -> Result<DatasetBuilder> + Send + Sync + 'static,
    ) -> Self {
        self.lance = Some(Arc::new(provider));
        self
    }

    /// Supply Paimon FileIO for the table URI.
    ///
    /// # Public Only for Internal Use:
    ///
    /// This is not a public API and is for internal use only; see [API policy] for details.
    /// [API policy]: https://datafusion.apache.org/contributor-guide/api-health.html#internal-public-apis
    #[doc(hidden)]
    #[must_use]
    pub fn with_paimon_file_io(
        mut self,
        provider: impl Fn(&str) -> Result<FileIO> + Send + Sync + 'static,
    ) -> Self {
        self.paimon = Some(Arc::new(provider));
        self
    }

    /// Register the providers as opaque session state.
    ///
    /// # Public Only for Internal Use:
    ///
    /// This is not a public API and is for internal use only; see [API policy] for details.
    /// [API policy]: https://datafusion.apache.org/contributor-guide/api-health.html#internal-public-apis
    #[doc(hidden)]
    pub fn register(self, session: &mut SessionConfig) {
        session.set_extension(Arc::new(self));
    }

    /// Get the providers registered on this session, if present.
    ///
    /// # Public Only for Internal Use:
    ///
    /// This is not a public API and is for internal use only; see [API policy] for details.
    /// [API policy]: https://datafusion.apache.org/contributor-guide/api-health.html#internal-public-apis
    #[doc(hidden)]
    pub fn get(session: &SessionConfig) -> Option<Arc<Self>> {
        session.get_extension()
    }

    pub(crate) fn lance_builder(&self, uri: &str) -> Result<DatasetBuilder> {
        match &self.lance {
            Some(provider) => provider(uri),
            None => Ok(DatasetBuilder::from_uri(uri)),
        }
    }

    pub(crate) fn paimon_file_io(&self, uri: &str) -> Result<FileIO> {
        if let Some(provider) = &self.paimon {
            return provider(uri);
        }
        let url = Url::parse(uri).ok();
        let scheme = if Path::new(uri).is_absolute() {
            "file"
        } else {
            url.as_ref().map_or("file", Url::scheme)
        };
        FileIOBuilder::new(scheme)
            .build()
            .map_err(paimon_scan_error)
    }
}

#[derive(Serialize, Deserialize)]
#[serde(tag = "engine", rename_all = "snake_case", deny_unknown_fields)]
enum ScanTemplate {
    Lance {
        dataset_uri: String,
        dataset_version: u64,
        filter: Option<ScanFilter>,
        required_indices: Vec<RequiredIndexSpec>,
    },
    Paimon {
        table_uri: String,
        schema_id: i64,
        snapshot_id: i64,
        filter_expr: Option<ScanFilter>,
        required_indices: Vec<RequiredIndexSpec>,
    },
}

/// Encode a lake scan template with no split assignments or storage providers.
/// Returns `None` for other file sources. Carry the plan's output schema
/// alongside the payload for [`rebuild_scan`].
///
/// # Errors
///
/// Returns a plan error for a pushed fetch, ordering, custom file scan state,
/// or unsupported snapshot or filter. The encoded template is limited to 1 MiB.
///
/// # Public Only for Internal Use:
///
/// This is not a public API and is for internal use only; see [API policy] for details.
/// [API policy]: https://datafusion.apache.org/contributor-guide/api-health.html#internal-public-apis
#[doc(hidden)]
pub fn encode_scan(config: &FileScanConfig) -> Result<Option<Vec<u8>>> {
    let Some(source) = config.file_source().downcast_ref::<LakeSource>() else {
        return Ok(None);
    };
    if config.limit.is_some()
        || config.preserve_order
        || !config.output_ordering.is_empty()
        || config.output_partitioning.is_some()
        || config.expr_adapter_factory.is_some()
    {
        return plan_err!(
            "Lake scan transport requires independent, unordered splits without a pushed fetch or custom expression adapter"
        );
    }
    let template = match &source.engine {
        LakeScan::Lance(scan) => {
            let spec = &scan.spec;
            validate_uri(&spec.dataset_uri)?;
            ScanTemplate::Lance {
                dataset_uri: spec.dataset_uri.clone(),
                dataset_version: spec.dataset_version,
                filter: spec
                    .filter
                    .as_ref()
                    .map(ScanFilter::from_expr)
                    .transpose()?,
                required_indices: spec.required_indices.clone(),
            }
        }
        LakeScan::Paimon(scan) => {
            scan.check_transportable()?;
            let spec = &scan.spec;
            validate_uri(&spec.table_uri)?;
            ScanTemplate::Paimon {
                table_uri: spec.table_uri.clone(),
                schema_id: spec.schema_id,
                snapshot_id: spec.snapshot_id,
                filter_expr: spec
                    .filter_expr
                    .as_ref()
                    .map(ScanFilter::from_expr)
                    .transpose()?,
                required_indices: spec.required_indices.clone(),
            }
        }
    };
    let mut bytes = SCAN_MAGIC.to_vec();
    bytes.extend(LAKE_SCAN_PROTOCOL_VERSION.to_le_bytes());
    serde_json::to_writer(&mut bytes, &template).map_err(|error| {
        plan_datafusion_err!("Cannot encode lake scan template: {error}")
    })?;
    check_size(&bytes, MAX_SCAN_BYTES, "scan")?;
    Ok(Some(bytes))
}

/// Rebuild a zero-file lake scan. The first bound split opens the snapshot;
/// later splits bound to the same template share the opened handle.
///
/// # Errors
///
/// Returns a plan error for malformed templates, unsupported versions or invalid pins.
///
/// # Public Only for Internal Use:
///
/// This is not a public API and is for internal use only; see [API policy] for details.
/// [API policy]: https://datafusion.apache.org/contributor-guide/api-health.html#internal-public-apis
#[doc(hidden)]
pub fn rebuild_scan(
    payload: &[u8],
    output_schema: SchemaRef,
    ctx: &TaskContext,
) -> Result<Arc<dyn ExecutionPlan>> {
    check_size(payload, MAX_SCAN_BYTES, "scan")?;
    let Some(header) = payload.strip_prefix(SCAN_MAGIC) else {
        return plan_err!("Invalid lake scan magic");
    };
    let Some((version, json)) = header.split_first_chunk::<4>() else {
        return plan_err!("Missing lake scan version");
    };
    let version = u32::from_le_bytes(*version);
    if version != LAKE_SCAN_PROTOCOL_VERSION {
        return plan_err!("Unsupported lake scan version {version}");
    }
    let template: ScanTemplate = serde_json::from_slice(json)
        .map_err(|error| plan_datafusion_err!("Invalid lake scan template: {error}"))?;
    let runtime = LakeRuntimeConfig::get(ctx.session_config()).unwrap_or_default();
    let engine = match template {
        ScanTemplate::Lance {
            dataset_uri,
            dataset_version,
            filter,
            required_indices,
        } => {
            validate_uri(&dataset_uri)?;
            if dataset_version == 0 {
                return plan_err!("Invalid Lance dataset version {dataset_version}");
            }
            LakeScan::Lance(LanceScan::worker(
                LanceScanSpec {
                    dataset_uri,
                    dataset_version,
                    filter: filter.map(ScanFilter::into_expr),
                    required_indices,
                },
                runtime,
            ))
        }
        ScanTemplate::Paimon {
            table_uri,
            schema_id,
            snapshot_id,
            filter_expr,
            required_indices,
        } => {
            validate_uri(&table_uri)?;
            if schema_id < 0 || snapshot_id <= 0 {
                return plan_err!(
                    "Invalid Paimon schema ID {schema_id} or snapshot ID {snapshot_id}"
                );
            }
            LakeScan::Paimon(PaimonScan::worker(
                PaimonScanSpec {
                    table_uri,
                    schema_id,
                    snapshot_id,
                    filter_expr: filter_expr.map(ScanFilter::into_expr),
                    required_indices,
                },
                runtime,
            ))
        }
    };
    LakeSource::plan(engine, output_schema, None, vec![], None)
}

/// Encode the planned splits in file-group order using the `df.lake.split` v1 codec.
///
/// # Errors
///
/// Returns a plan error for invalid split identities, incompatible file sources
/// or a split exceeding 16 MiB.
///
/// # Public Only for Internal Use:
///
/// This is not a public API and is for internal use only; see [API policy] for details.
/// [API policy]: https://datafusion.apache.org/contributor-guide/api-health.html#internal-public-apis
#[doc(hidden)]
pub fn encode_splits(config: &FileScanConfig) -> Result<Vec<Vec<u8>>> {
    let source = lake_source(config)?;
    config
        .file_groups
        .iter()
        .flat_map(FileGroup::files)
        .map(|file| {
            let split = LakeSplitExt::for_read(file)?;
            validate_split_identity(split)?;
            source.engine.validate_split(&split.payload)?;
            let (_, bytes) = LakeSplitExt::try_encode_from_extensions(&file.extensions)?
                .ok_or_else(|| plan_datafusion_err!("Missing lake split extension"))?;
            check_size(&bytes, MAX_SPLIT_BYTES, "split")?;
            Ok(bytes)
        })
        .collect()
}

/// Decode a `df.lake.split` v1 payload into a range-free file with unknown size.
///
/// # Errors
///
/// Returns a plan error for malformed splits or invalid versions or identities.
///
/// # Public Only for Internal Use:
///
/// This is not a public API and is for internal use only; see [API policy] for details.
/// [API policy]: https://datafusion.apache.org/contributor-guide/api-health.html#internal-public-apis
#[doc(hidden)]
pub fn decode_split(bytes: &[u8]) -> Result<PartitionedFile> {
    check_size(bytes, MAX_SPLIT_BYTES, "split")?;
    let mut extensions = FileExtensions::new();
    LakeSplitExt::try_decode_into_extensions(
        LAKE_SPLIT_EXT_VERSION,
        bytes,
        &mut extensions,
    )?;
    let split = extensions
        .get::<LakeSplitExt>()
        .ok_or_else(|| plan_datafusion_err!("Missing lake split extension"))?;
    validate_split_identity(split)?;
    let mut file = PartitionedFile::new(split.split_id.clone(), 0);
    file.extensions = extensions;
    Ok(file)
}

/// Bind one split to a lake source, preserving its output schema.
///
/// # Errors
///
/// Returns a plan error if the split is invalid or does not match the scan's snapshot.
///
/// # Public Only for Internal Use:
///
/// This is not a public API and is for internal use only; see [API policy] for details.
/// [API policy]: https://datafusion.apache.org/contributor-guide/api-health.html#internal-public-apis
#[doc(hidden)]
pub fn bind_split(
    config: &FileScanConfig,
    file: PartitionedFile,
) -> Result<Arc<dyn ExecutionPlan>> {
    let source = lake_source(config)?;
    let split = LakeSplitExt::for_read(&file)?;
    validate_split_identity(split)?;
    source.engine.validate_split(&split.payload)?;
    if matches!(split.payload, LakeSplitPayload::Paimon { .. }) {
        split.payload.paimon_data_split()?;
    }
    let config = FileScanConfigBuilder::from(config.clone())
        .with_source(Arc::new(source.with_fresh_metrics()))
        .with_file_groups(vec![FileGroup::new(vec![file])])
        .build();
    Ok(DataSourceExec::from_data_source(config))
}

fn lake_source(config: &FileScanConfig) -> Result<&LakeSource> {
    config
        .file_source()
        .downcast_ref()
        .ok_or_else(|| plan_datafusion_err!("Expected a lake FileScanConfig"))
}

fn validate_split_identity(split: &LakeSplitExt) -> Result<()> {
    split.validate_identity()?;
    let uri = match &split.payload {
        LakeSplitPayload::Lance { dataset_uri, .. } => dataset_uri,
        LakeSplitPayload::Paimon { table_uri, .. } => table_uri,
    };
    validate_uri(uri)
}

fn validate_uri(uri: &str) -> Result<()> {
    if Path::new(uri).is_absolute() {
        return Ok(());
    }
    let url = Url::parse(uri).map_err(|_| {
        plan_datafusion_err!("Lake transport requires an absolute table URI or path")
    })?;
    if url.cannot_be_a_base()
        || !url.username().is_empty()
        || url.password().is_some()
        || url.query().is_some()
        || url.fragment().is_some()
    {
        return plan_err!(
            "Lake transport requires a table URI without user info, query, or fragment"
        );
    }
    Ok(())
}

fn check_size(bytes: &[u8], max: usize, kind: &str) -> Result<()> {
    if bytes.len() > max {
        return plan_err!("Lake {kind} payload exceeds the {max}-byte limit");
    }
    Ok(())
}

#[cfg(test)]
#[path = "../tests/dist/row_ranges.rs"]
mod row_range_tests;
