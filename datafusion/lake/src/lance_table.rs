// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: See the NOTICE file distributed with this work for additional information regarding copyright ownership
// SPDX-FileContributor: Licensed to the Apache Software Foundation (ASF) under one or more contributor license agreements

use std::collections::{BTreeMap, BTreeSet, HashMap};
use std::sync::Arc;

use arrow::array::{Array, RecordBatch, StructArray, UInt64Array};
use arrow::datatypes::{Field, Schema, SchemaRef};
use async_trait::async_trait;
use datafusion_ai::{MediaRef, MediaRefType};
use datafusion_catalog::{Session, TableProvider};
use datafusion_common::{
    DataFusionError, Result, exec_err, not_impl_err, plan_datafusion_err, plan_err,
};
use datafusion_datasource::PartitionedFile;
use datafusion_expr::utils::conjunction;
use datafusion_expr::{Expr, TableProviderFilterPushDown, TableType};
use datafusion_physical_plan::ExecutionPlan;
use futures::{TryStreamExt, stream::BoxStream};
use lance::Dataset;
use lance::dataset::builder::DatasetBuilder;
use lance::dataset::fragment::FileFragment;
use lance::dataset::scanner::{FragmentSlice, Scanner};
use lance::index::{DatasetIndexExt, DatasetIndexInternalExt};
use lance::session::UnregisteredExternalBlobPolicy;
use lance_core::ROW_ID;
use lance_core::datatypes::{BlobHandling, BlobVersion};
use lance_core::utils::blob::blob_path;
use lance_index::metrics::NoOpMetricsCollector;
use lance_table_runtime::blob::locate::{
    StoredBlob, blob_version_from_descriptions, data_file_key_from_path,
};
use object_store::path::Path;
use url::Url;

use crate::dist::LakeRuntimeConfig;
use crate::filter;
use crate::media::{external, media_array, media_url};
use crate::scan::{LakeScan, ScanHandle, make_batch, scan_plan};
use crate::split::{InclusiveRowRange, LakeSplitExt, LakeSplitPayload};
use crate::validation::{LakeTableOptions, RequiredIndexSpec, blob_mime};

#[derive(Debug)]
struct BlobColumn {
    field_id: u32,
    mime: Option<String>,
}

type BlobColumns = Arc<HashMap<String, BlobColumn>>;

struct BlobLocation<'a> {
    base_id: Option<u32>,
    data_dir: Url,
    file_path: &'a str,
}

/// Read-only [`TableProvider`] for a pinned Lance dataset.
///
/// # Public Only for Internal Use:
///
/// This is not a public API and is for internal use only; see [API policy] for details.
/// [API policy]: https://datafusion.apache.org/contributor-guide/api-health.html#internal-public-apis
#[doc(hidden)]
#[derive(Debug)]
pub struct LanceReadOnlyTable {
    dataset: Arc<Dataset>,
    schema: SchemaRef,
    blobs: BlobColumns,
    required_indices: Vec<RequiredIndexSpec>,
    vector_indices: HashMap<String, String>,
}

impl LanceReadOnlyTable {
    /// Open `version` or pin the latest version at `uri`.
    ///
    /// # Errors
    /// Returns a planning error for an unreadable dataset or failed table validation.
    #[doc(hidden)]
    pub async fn open(
        uri: &str,
        version: Option<u64>,
        options: &LakeTableOptions,
    ) -> Result<Self> {
        let builder = DatasetBuilder::from_uri(uri);
        let builder = match version {
            Some(version) => builder.with_version(version),
            None => builder,
        };
        Self::open_with_builder(builder, options).await
    }

    /// Open with a caller-configured Lance builder, including any store bindings.
    /// The loaded manifest supplies the pinned version.
    ///
    /// # Errors
    /// Returns a planning error for an unreadable dataset or failed table validation.
    #[doc(hidden)]
    pub async fn open_with_builder(
        builder: DatasetBuilder,
        options: &LakeTableOptions,
    ) -> Result<Self> {
        let dataset = builder
            .load()
            .await
            .map_err(|e| plan_datafusion_err!("Cannot open Lance snapshot: {e}"))?;
        let raw = Schema::from(dataset.schema());
        let (schema, blobs) = lake_schema(&dataset, &raw)?;
        let vector_indices = validate_indices(&dataset, &raw, options).await?;

        Ok(Self {
            dataset: Arc::new(dataset),
            schema,
            blobs,
            required_indices: options.required_indices.clone(),
            vector_indices,
        })
    }

    /// The immutable manifest version used by subsequent scans.
    #[must_use]
    #[doc(hidden)]
    pub fn dataset_version(&self) -> u64 {
        self.dataset.version_id()
    }

    /// Validate the caller's entire scalar prefilter before constructing a scan or retrieval plan.
    ///
    /// # Errors
    /// Returns [`DataFusionError::NotImplemented`] for unsupported predicates.
    #[doc(hidden)]
    pub fn validate_prefilter(&self, filters: &[Expr]) -> Result<()> {
        for expr in filters {
            filter::normalize(expr, &self.schema)?;
        }
        Ok(())
    }

    pub(crate) fn dataset(&self) -> &Dataset {
        &self.dataset
    }

    pub(crate) fn has_required_index(&self, index: &RequiredIndexSpec) -> bool {
        self.required_indices.contains(index)
    }

    pub(crate) async fn select_vector_index(
        &self,
        column: &str,
        scan: &mut Scanner,
    ) -> Result<()> {
        let name = self.vector_indices.get(column).ok_or_else(|| {
            plan_datafusion_err!("Lance vector index for {column:?} was not validated")
        })?;
        let indices = self
            .dataset
            .load_indices()
            .await
            .map_err(lance_scan_error)?;
        let segments = indices
            .iter()
            .filter(|index| &index.name == name)
            .map(|index| index.uuid)
            .collect();
        scan.with_index_segments(segments)
            .map_err(lance_scan_error)?;
        Ok(())
    }

    pub(crate) fn scan_row_ranges(
        &self,
        schema: SchemaRef,
        fragments: BTreeMap<u32, Vec<InclusiveRowRange>>,
    ) -> Result<Arc<dyn ExecutionPlan>> {
        let files = fragments
            .into_iter()
            .map(|(fragment_id, ranges)| {
                let fragment = self
                    .dataset
                    .get_fragment(fragment_id as usize)
                    .ok_or_else(|| {
                        plan_datafusion_err!(
                            "Missing Lance fragment {fragment_id} at the pinned snapshot"
                        )
                    })?;
                let metadata = fragment.metadata();
                let size = metadata
                    .files
                    .iter()
                    .filter_map(|file| file.file_size_bytes.get())
                    .map(std::num::NonZeroU64::get)
                    .fold(0, u64::saturating_add);
                LakeSplitExt::into_file(
                    LakeSplitPayload::Lance {
                        dataset_uri: self.dataset.uri().into(),
                        dataset_version: self.dataset_version(),
                        fragment_id,
                        row_ranges: Some(ranges),
                        deletion_file_rev: metadata
                            .deletion_file
                            .as_ref()
                            .map(|file| file.read_version),
                    },
                    size,
                )
            })
            .collect::<Result<Vec<_>>>()?;
        let scan = self.local_scan(None);
        scan_plan(LakeScan::Lance(scan), schema, None, files, None)
    }

    fn local_scan(&self, filter: Option<Expr>) -> LanceScan {
        LanceScan {
            spec: Arc::new(LanceScanSpec {
                dataset_uri: self.dataset.uri().into(),
                dataset_version: self.dataset_version(),
                required_indices: self.required_indices.clone(),
                filter,
            }),
            handle: Arc::new(ScanHandle::Local(Arc::new(OpenedLanceScan {
                dataset: Arc::clone(&self.dataset),
                blobs: Arc::clone(&self.blobs),
            }))),
        }
    }

    fn files(&self) -> Result<Vec<PartitionedFile>> {
        let mut fragments: Vec<_> = self.dataset.iter_fragments().collect();
        fragments.sort_unstable_by_key(|fragment| fragment.id);
        fragments
            .into_iter()
            .map(|fragment| {
                let fragment_id = u32::try_from(fragment.id).map_err(|_| {
                    plan_datafusion_err!("Invalid Lance fragment ID {}", fragment.id)
                })?;
                let size = fragment
                    .files
                    .iter()
                    .filter_map(|file| file.file_size_bytes.get())
                    .map(std::num::NonZeroU64::get)
                    .fold(0, u64::saturating_add);
                LakeSplitExt::into_file(
                    LakeSplitPayload::Lance {
                        dataset_uri: self.dataset.uri().into(),
                        dataset_version: self.dataset_version(),
                        fragment_id,
                        row_ranges: None,
                        deletion_file_rev: fragment
                            .deletion_file
                            .as_ref()
                            .map(|file| file.read_version),
                    },
                    size,
                )
            })
            .collect()
    }
}

#[async_trait]
impl TableProvider for LanceReadOnlyTable {
    fn schema(&self) -> SchemaRef {
        Arc::clone(&self.schema)
    }

    fn table_type(&self) -> TableType {
        TableType::Base
    }

    fn supports_filters_pushdown(
        &self,
        filters: &[&Expr],
    ) -> Result<Vec<TableProviderFilterPushDown>> {
        filters
            .iter()
            .map(|expr| {
                filter::normalize(expr, &self.schema)?;
                Ok(TableProviderFilterPushDown::Exact)
            })
            .collect()
    }

    async fn scan(
        &self,
        _state: &dyn Session,
        projection: Option<&[usize]>,
        filters: &[Expr],
        limit: Option<usize>,
    ) -> Result<Arc<dyn ExecutionPlan>> {
        let filters = filters
            .iter()
            .map(|expr| filter::normalize(expr, &self.schema))
            .collect::<Result<Vec<_>>>()?;
        let scan = self.local_scan(conjunction(filters));
        let files = if limit == Some(0) {
            vec![]
        } else {
            self.files()?
        };
        scan_plan(
            LakeScan::Lance(scan),
            self.schema(),
            projection,
            files,
            limit,
        )
    }
}

pub(crate) struct LanceScanSpec {
    pub(crate) dataset_uri: String,
    pub(crate) dataset_version: u64,
    pub(crate) filter: Option<Expr>,
    pub(crate) required_indices: Vec<RequiredIndexSpec>,
}

#[derive(Clone)]
pub(crate) struct LanceScan {
    pub(crate) spec: Arc<LanceScanSpec>,
    handle: Arc<ScanHandle<OpenedLanceScan>>,
}

struct OpenedLanceScan {
    dataset: Arc<Dataset>,
    blobs: BlobColumns,
}

impl LanceScan {
    pub(crate) fn worker(spec: LanceScanSpec, runtime: Arc<LakeRuntimeConfig>) -> Self {
        Self {
            spec: Arc::new(spec),
            handle: Arc::new(ScanHandle::worker(runtime)),
        }
    }

    pub(crate) fn is_worker(&self) -> bool {
        self.handle.local().is_none()
    }

    pub(crate) fn validate_split(&self, payload: &LakeSplitPayload) -> Result<()> {
        let LakeSplitPayload::Lance {
            dataset_uri,
            dataset_version,
            ..
        } = payload
        else {
            return plan_err!("Expected a Lance split");
        };
        if *dataset_version != self.spec.dataset_version
            || *dataset_uri != self.spec.dataset_uri
        {
            return plan_err!("Lance split does not match the pinned dataset snapshot");
        }
        Ok(())
    }

    async fn open_snapshot(
        &self,
        runtime: Arc<LakeRuntimeConfig>,
    ) -> Result<OpenedLanceScan> {
        let dataset = runtime
            .lance_builder(&self.spec.dataset_uri)?
            .with_version(self.spec.dataset_version)
            .load()
            .await
            .map_err(lance_scan_error)?;
        if dataset.version_id() != self.spec.dataset_version
            || dataset.uri() != self.spec.dataset_uri
        {
            return plan_err!("Opened Lance dataset does not match the pinned snapshot");
        }
        let raw = Schema::from(dataset.schema());
        let (schema, blobs) = lake_schema(&dataset, &raw)?;
        if let Some(filter) = &self.spec.filter {
            filter::normalize(filter, &schema)?;
        }
        for required in &self.spec.required_indices {
            required.validate_column(&raw)?;
        }
        Ok(OpenedLanceScan {
            dataset: Arc::new(dataset),
            blobs,
        })
    }

    pub(crate) async fn open(
        self,
        payload: LakeSplitPayload,
        schema: SchemaRef,
        batch_size: usize,
        limit: Option<usize>,
    ) -> Result<BoxStream<'static, Result<RecordBatch>>> {
        let LakeSplitPayload::Lance {
            fragment_id,
            row_ranges,
            deletion_file_rev,
            ..
        } = payload
        else {
            return plan_err!("Expected a Lance split");
        };
        let opened = self
            .handle
            .get_or_open(|runtime| self.open_snapshot(runtime))
            .await?;
        let fragment = opened
            .dataset
            .get_fragment(fragment_id as usize)
            .ok_or_else(|| {
                plan_datafusion_err!("Missing Lance fragment {fragment_id}")
            })?;
        if fragment
            .metadata()
            .deletion_file
            .as_ref()
            .map(|file| file.read_version)
            != deletion_file_rev
        {
            return plan_err!(
                "Lance split deletion revision does not match the pinned fragment"
            );
        }

        let mut scanner = Scanner::from_fragment(
            Arc::clone(&opened.dataset),
            fragment.metadata().clone(),
        );
        let mut columns: BTreeSet<_> =
            schema.fields().iter().map(|f| f.name().as_str()).collect();
        if row_ranges.is_some() && columns.remove(ROW_ID) {
            scanner.with_row_id();
        }
        scanner
            .blob_handling(BlobHandling::BlobsDescriptions)
            .batch_size(batch_size)
            .project(&columns.into_iter().collect::<Vec<_>>())
            .map_err(lance_scan_error)?;
        if let Some(filter) = self.spec.filter.clone() {
            scanner.filter_expr(filter);
        }
        if let Some(ranges) = row_ranges {
            let slices = ranges
                .iter()
                .map(|range| {
                    let row_count = range
                        .to
                        .checked_sub(range.from)
                        .and_then(|diff| diff.checked_add(1))
                        .ok_or_else(|| {
                            plan_datafusion_err!("Lance split row count overflows u64")
                        })?;
                    Ok(FragmentSlice {
                        fragment_id,
                        row_offset: range.from,
                        row_count,
                    })
                })
                .collect::<Result<Vec<_>>>()?;
            scanner
                .with_fragment_slices(&slices)
                .await
                .map_err(lance_scan_error)?;
        }
        if let Some(limit) = limit {
            scanner
                .limit(Some(i64::try_from(limit).unwrap_or(i64::MAX)), None)
                .map_err(lance_scan_error)?;
        }
        let stream = scanner.try_into_stream().await.map_err(lance_scan_error)?;
        Ok(Box::pin(stream.map_err(lance_scan_error).and_then(
            move |batch| {
                futures::future::ready(opened.project(&batch, &schema, &fragment))
            },
        )))
    }
}

impl OpenedLanceScan {
    fn project(
        &self,
        batch: &RecordBatch,
        schema: &SchemaRef,
        fragment: &FileFragment,
    ) -> Result<RecordBatch> {
        let columns = schema
            .fields()
            .iter()
            .map(|field| {
                let input = batch.column_by_name(field.name()).ok_or_else(|| {
                    plan_datafusion_err!("Lance scan omitted column {:?}", field.name())
                })?;
                if let Some(blob) = self.blobs.get(field.name()) {
                    self.blob_array(input.as_ref(), field, blob, fragment)
                } else {
                    Ok(Arc::clone(input))
                }
            })
            .collect::<Result<Vec<_>>>()?;
        make_batch(Arc::clone(schema), columns, batch.num_rows())
    }

    fn blob_array(
        &self,
        input: &dyn Array,
        field: &Field,
        blob: &BlobColumn,
        fragment: &FileFragment,
    ) -> Result<arrow::array::ArrayRef> {
        let descriptions =
            input
                .as_any()
                .downcast_ref::<StructArray>()
                .ok_or_else(|| {
                    plan_datafusion_err!(
                        "Lance scan did not return blob descriptors for {:?}",
                        field.name()
                    )
                })?;
        let loc = fragment
            .data_file_for_field(blob.field_id)
            .map(|file| {
                let base = self.base_uri(file.base_id)?;
                let dir = if file.base_id.is_none_or(|id| {
                    self.dataset.manifest().base_paths[&id].is_dataset_root
                }) {
                    join_uri(base, "data")?
                } else {
                    base
                };
                Ok::<_, DataFusionError>(BlobLocation {
                    base_id: file.base_id,
                    data_dir: dir,
                    file_path: &file.path,
                })
            })
            .transpose()?;
        let version =
            blob_version_from_descriptions(descriptions).map_err(lance_scan_error)?;
        let values = (0..descriptions.len()).map(|row| {
            let stored = match version {
                BlobVersion::V1 => v1_descriptor(descriptions, row),
                BlobVersion::V2 => {
                    StoredBlob::parse(descriptions, row).map_err(lance_scan_error)?
                }
            };
            stored
                .map(|stored| self.media_ref(stored, loc.as_ref(), blob.mime.as_deref()))
                .transpose()
        });
        media_array(field.data_type(), values)
    }

    fn base_uri(&self, base_id: Option<u32>) -> Result<Url> {
        let uri = match base_id {
            Some(id) => self
                .dataset
                .manifest()
                .base_paths
                .get(&id)
                .ok_or_else(|| plan_datafusion_err!("Unknown Lance storage base {id}"))?
                .path
                .as_str(),
            None => self.dataset.uri(),
        };
        let uri = lance_io::object_store::uri_to_url(uri).map_err(lance_scan_error)?;
        media_url(uri.as_str())
    }

    fn media_ref(
        &self,
        blob: StoredBlob<'_>,
        loc: Option<&BlobLocation<'_>>,
        mime: Option<&str>,
    ) -> Result<MediaRef> {
        let location = || {
            loc.ok_or_else(|| {
                plan_datafusion_err!("Lance blob descriptor has no data file")
            })
        };
        let (uri, position, len) = match blob {
            StoredBlob::Empty | StoredBlob::Inline { size: 0, .. } => {
                (self.base_uri(None)?, 0, Some(0))
            }
            StoredBlob::Inline { position, size } => {
                let file = location()?;
                (
                    join_uri(file.data_dir.clone(), file.file_path)?,
                    position,
                    Some(size),
                )
            }
            StoredBlob::Packed {
                blob_id,
                position,
                size,
            } => (sidecar_uri(location()?, blob_id)?, position, Some(size)),
            StoredBlob::Dedicated { blob_id, size } => {
                (sidecar_uri(location()?, blob_id)?, 0, Some(size))
            }
            StoredBlob::External {
                base_id,
                uri,
                position,
                size,
            } => {
                let uri = if base_id == 0 {
                    if self.dataset.session().unregistered_external_blob_policy()
                        == UnregisteredExternalBlobPolicy::Deny
                    {
                        return exec_err!(
                            "Lance session denies unregistered external blob URIs"
                        );
                    }
                    media_url(uri)?
                } else {
                    join_uri(self.base_uri(Some(base_id))?, uri)?
                };
                (uri, position, (size != 0).then_some(size))
            }
            StoredBlob::Managed {
                base_id,
                uri,
                position,
                size,
            } => {
                let base_id = match base_id {
                    Some(id) => Some(id),
                    None => location()?.base_id,
                };
                (
                    join_uri(self.base_uri(base_id)?, uri)?,
                    position,
                    Some(size),
                )
            }
        };
        Ok(MediaRef {
            uri: uri.into(),
            byte_offset: Some(position),
            byte_len: len,
            etag: None,
            mime: mime.map(str::to_owned),
        })
    }
}

fn lake_schema(dataset: &Dataset, raw: &Schema) -> Result<(SchemaRef, BlobColumns)> {
    let mut blobs = HashMap::new();
    let fields = raw
        .fields()
        .iter()
        .zip(&dataset.schema().fields)
        .map(|(arrow, lance)| {
            if !lance.is_blob() {
                if lance.children.iter().any(contains_blob) {
                    return not_impl_err!("Lance nested blob column {:?}", arrow.name());
                }
                return Ok(Arc::clone(arrow));
            }
            let field_id = u32::try_from(lance.id).map_err(|_| {
                plan_datafusion_err!("Invalid Lance field ID {}", lance.id)
            })?;
            blobs.insert(
                arrow.name().clone(),
                BlobColumn {
                    field_id,
                    mime: blob_mime(
                        arrow.name(),
                        |key| arrow.metadata().get(key),
                        |key| raw.metadata().get(key),
                    )?,
                },
            );
            Ok(Arc::new(MediaRefType::field(
                arrow.name(),
                arrow.is_nullable(),
            )?))
        })
        .collect::<Result<Vec<_>>>()?;
    if dataset.iter_fragments().any(|fragment| {
        fragment.overlays.iter().any(|overlay| {
            blobs.values().any(|blob| {
                i32::try_from(blob.field_id)
                    .is_ok_and(|id| overlay.data_file.fields.contains(&id))
            })
        })
    }) {
        return not_impl_err!("Lance blob descriptors from overlay files");
    }
    let schema = Schema::new_with_metadata(fields, raw.metadata().clone());
    Ok((Arc::new(schema), Arc::new(blobs)))
}

fn contains_blob(field: &lance_core::datatypes::Field) -> bool {
    field.is_blob() || field.children.iter().any(contains_blob)
}

fn v1_descriptor(descriptions: &StructArray, row: usize) -> Option<StoredBlob<'_>> {
    let positions = descriptions
        .column(0)
        .as_any()
        .downcast_ref::<UInt64Array>()?;
    let sizes = descriptions
        .column(1)
        .as_any()
        .downcast_ref::<UInt64Array>()?;
    if descriptions.is_null(row) || positions.is_null(row) || sizes.is_null(row) {
        return None;
    }
    let (position, size) = (positions.value(row), sizes.value(row));
    // V1 uses a nonzero position with zero size to represent null definition levels.
    match (position, size) {
        (0, 0) => Some(StoredBlob::Empty),
        (_, 0) => None,
        (position, size) => Some(StoredBlob::Inline { position, size }),
    }
}

fn sidecar_uri(file: &BlobLocation<'_>, id: u32) -> Result<Url> {
    let path = blob_path(&Path::ROOT, data_file_key_from_path(file.file_path), id);
    join_uri(file.data_dir.clone(), path.as_ref())
}

fn join_uri(mut root: Url, relative: &str) -> Result<Url> {
    let path = Path::parse(relative).map_err(|e| {
        plan_datafusion_err!("Invalid Lance blob object key {relative:?}: {e}")
    })?;
    if path.is_root() {
        return plan_err!("Lance blob object key must not be empty");
    }
    root.path_segments_mut()
        .map_err(|()| plan_datafusion_err!("Invalid Lance blob storage base"))?
        .pop_if_empty()
        .extend(path.as_ref().split('/'));
    Ok(root)
}

pub(crate) fn lance_scan_error(error: lance::Error) -> DataFusionError {
    if matches!(error, lance::Error::NotSupported { .. }) {
        DataFusionError::NotImplemented(error.to_string())
    } else {
        external(error)
    }
}

async fn validate_indices(
    dataset: &Dataset,
    schema: &Schema,
    options: &LakeTableOptions,
) -> Result<HashMap<String, String>> {
    let mut vector_indices = HashMap::new();
    if options.required_indices.is_empty() {
        return Ok(vector_indices);
    }
    let indices = dataset
        .load_indices()
        .await
        .map_err(|err| plan_datafusion_err!("Cannot load Lance index metadata: {err}"))?;
    for required in &options.required_indices {
        let column = required.column();
        let field = schema.field_with_name(column).map_err(|_| {
            plan_datafusion_err!("Lance index column {column:?} does not exist")
        })?;
        required.validate_dimension(field.data_type())?;
        let id = dataset
            .schema()
            .field_id(column)
            .map_err(lance_scan_error)?;
        let candidates: BTreeSet<_> = indices
            .iter()
            .filter(|index| index.fields.as_slice() == [id])
            .map(|index| index.name.as_str())
            .collect();
        let mut ready = false;
        for name in candidates {
            let stats = dataset.index_statistics(name).await.map_err(|err| {
                plan_datafusion_err!("Lance index {name:?} is unreadable: {err}")
            })?;
            let stats: serde_json::Value =
                serde_json::from_str(&stats).map_err(|err| {
                    plan_datafusion_err!(
                        "Lance index {name:?} has invalid statistics: {err}"
                    )
                })?;
            let kind = stats["index_type"]
                .as_str()
                .unwrap_or_default()
                .to_ascii_lowercase();
            let matching = match required {
                RequiredIndexSpec::Vector { .. } => {
                    kind == "vector" || kind.starts_with("ivf_")
                }
                RequiredIndexSpec::FullText { .. } => kind == "inverted",
                RequiredIndexSpec::Scalar { .. } => matches!(
                    kind.as_str(),
                    "btree"
                        | "bitmap"
                        | "labellist"
                        | "zonemap"
                        | "bloomfilter"
                        | "ngram"
                        | "fm"
                ),
            };
            if matching
                && stats["num_unindexed_fragments"].as_u64() == Some(0)
                && stats["num_unindexed_rows"].as_u64() == Some(0)
            {
                if let RequiredIndexSpec::Vector { dimension, .. } = required {
                    for index in indices.iter().filter(|index| index.name == name) {
                        let opened = dataset
                            .open_vector_index_from_metadata(
                                column,
                                index,
                                &NoOpMetricsCollector,
                            )
                            .await
                            .map_err(|err| {
                                plan_datafusion_err!(
                                    "Cannot open Lance vector index {name:?}: {err}"
                                )
                            })?;
                        let actual = opened.ivf_model().dimension();
                        if u32::try_from(actual).ok() != Some(*dimension) {
                            return plan_err!(
                                "Lance vector index {name:?} has dimension {actual}, expected {dimension}"
                            );
                        }
                    }
                    vector_indices.insert(column.to_owned(), name.to_owned());
                }
                ready = true;
                break;
            }
        }
        if !ready {
            return plan_err!(
                "Required Lance index {required:?} is missing or does not cover snapshot {}",
                dataset.version_id()
            );
        }
    }
    Ok(vector_indices)
}
