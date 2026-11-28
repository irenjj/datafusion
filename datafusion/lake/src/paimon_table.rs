// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: See the NOTICE file distributed with this work for additional information regarding copyright ownership
// SPDX-FileContributor: Licensed to the Apache Software Foundation (ASF) under one or more contributor license agreements

use std::collections::{BTreeSet, HashMap, HashSet};
use std::sync::Arc;

use arrow::array::{Array, BinaryArray, BooleanArray, LargeBinaryArray, RecordBatch};
use arrow::compute::filter_record_batch;
use arrow::datatypes::{Field, Schema, SchemaRef};
use async_trait::async_trait;
use bytes::Buf;
use datafusion_ai::{MediaRef, MediaRefType};
use datafusion_catalog::{Session, TableProvider};
use datafusion_common::{
    DFSchema, DataFusionError, Result, exec_err, not_impl_err, plan_datafusion_err,
    plan_err,
};
use datafusion_datasource::PartitionedFile;
use datafusion_expr::execution_props::ExecutionProps;
use datafusion_expr::physical_planning_context::PhysicalPlanningContext;
use datafusion_expr::utils::conjunction;
use datafusion_expr::{Expr, TableProviderFilterPushDown, TableType};
use datafusion_physical_expr::{PhysicalExpr, create_physical_expr};
use datafusion_physical_plan::ExecutionPlan;
use futures::{TryStreamExt, stream::BoxStream};
use paimon::arrow::build_read_arrow_schema;
use paimon::catalog::Identifier;
use paimon::io::{FileRead, InputFile};
use paimon::spec::{
    BlobDescriptor, DataEvolutionIndexSourceMeta, DataType, FileKind, IndexManifest,
    Predicate, PredicateBuilder, Snapshot,
};
use paimon::table::{SchemaManager, SnapshotManager};
use paimon::{DataSplit, ReadBuilder, RowRange, Table};
use paimon_vindex_core::{diskann_io, io as ivfpq_io, ivfflat_io, ivfrq_io, ivfsq_io};
use sha2::{Digest, Sha256};

use crate::dist::LakeRuntimeConfig;
use crate::filter;
use crate::media::{external, media_array};
use crate::scan::{LakeScan, ScanHandle, make_batch, projected_schema, scan_plan};
use crate::split::{LakeSplitExt, LakeSplitPayload, paimon_ranges, partition_key};
use crate::validation::{LakeTableOptions, RequiredIndexSpec, blob_mime};

type BlobColumns = Arc<HashMap<String, Option<String>>>;

/// Read-only [`TableProvider`] for a pinned Paimon snapshot.
///
/// # Public Only for Internal Use:
///
/// This is not a public API and is for internal use only; see [API policy] for details.
/// [API policy]: https://datafusion.apache.org/contributor-guide/api-health.html#internal-public-apis
#[doc(hidden)]
#[derive(Debug)]
pub struct PaimonReadOnlyTable {
    table: Arc<Table>,
    snapshot_id: i64,
    schema: SchemaRef,
    blobs: BlobColumns,
    required_indices: Vec<RequiredIndexSpec>,
}

impl PaimonReadOnlyTable {
    /// Open a catalog-resolved Paimon table and pin `snapshot_id` or its latest snapshot.
    ///
    /// # Errors
    /// Returns a planning error for an unreadable snapshot or failed table validation.
    #[doc(hidden)]
    pub async fn open(
        table: Table,
        snapshot_id: Option<i64>,
        options: &LakeTableOptions,
    ) -> Result<Self> {
        if table.schema().core_options().is_format_table() {
            return plan_err!("Paimon format tables have no snapshot to pin");
        }
        let manager = table.snapshot_manager();
        let snapshot = match snapshot_id {
            Some(id) => manager.get_snapshot(id).await.map(Some),
            None => manager.get_latest_snapshot().await,
        }
        .map_err(|e| plan_datafusion_err!("Cannot open Paimon snapshot: {e}"))?
        .ok_or_else(|| plan_datafusion_err!("Paimon table has no readable snapshot"))?;
        let table = pin_table(&table, &snapshot).await?;
        if snapshot.id() <= 0 || snapshot_id.is_some_and(|id| id != snapshot.id()) {
            return plan_err!("Invalid Paimon snapshot identity {}", snapshot.id());
        }
        let raw = raw_schema(&table)?;
        let (schema, blobs) = lake_schema(&table, &raw)?;
        validate_indices(&table, &snapshot, &raw, options).await?;
        Ok(Self {
            table: Arc::new(table),
            snapshot_id: snapshot.id(),
            schema,
            blobs,
            required_indices: options.required_indices.clone(),
        })
    }

    /// The immutable Paimon snapshot ID used by subsequent scans.
    #[must_use]
    #[doc(hidden)]
    pub fn snapshot_id(&self) -> i64 {
        self.snapshot_id
    }

    /// Validate the caller's entire scalar prefilter before constructing a scan or retrieval plan.
    ///
    /// # Errors
    /// Returns [`DataFusionError::NotImplemented`] for unsupported predicates.
    #[doc(hidden)]
    pub fn validate_prefilter(&self, filters: &[Expr]) -> Result<()> {
        self.translate(filters.iter()).map(|_| ())
    }

    pub(crate) fn table(&self) -> &Table {
        &self.table
    }

    pub(crate) fn has_required_index(&self, index: &RequiredIndexSpec) -> bool {
        self.required_indices.contains(index)
    }

    pub(crate) async fn scan_row_ranges(
        &self,
        schema: SchemaRef,
        ranges: Vec<RowRange>,
    ) -> Result<Arc<dyn ExecutionPlan>> {
        let columns: Vec<_> = schema
            .fields()
            .iter()
            .map(|field| field.name().clone())
            .collect();
        let mut read = self.table.new_read_builder();
        configure_read(&mut read, &columns, None)?;
        let plan = read
            .new_scan()
            .with_row_ranges(ranges)
            .plan()
            .await
            .map_err(paimon_scan_error)?;
        let (files, planned) =
            split_files(self.table.location(), self.snapshot_id, plan.splits())?;
        let scan = self.local_scan(None, columns, None, None, planned);
        scan_plan(LakeScan::Paimon(scan), schema, None, files, None)
    }

    fn local_scan(
        &self,
        filter_expr: Option<Expr>,
        columns: Vec<String>,
        predicate: Option<Predicate>,
        filter: Option<Arc<dyn PhysicalExpr>>,
        planned: HashSet<SplitFingerprint>,
    ) -> PaimonScan {
        PaimonScan {
            spec: Arc::new(PaimonScanSpec {
                table_uri: self.table.location().into(),
                schema_id: self.table.schema().id(),
                snapshot_id: self.snapshot_id,
                filter_expr,
                required_indices: self.required_indices.clone(),
            }),
            handle: Arc::new(ScanHandle::Local(Arc::new(OpenedPaimonScan {
                table: Arc::clone(&self.table),
                blobs: Arc::clone(&self.blobs),
                columns,
                predicate,
                filter,
            }))),
            planned: Some(Arc::new(planned)),
        }
    }

    fn translate<'a>(
        &self,
        filters: impl IntoIterator<Item = &'a Expr>,
    ) -> Result<Vec<(Expr, Predicate)>> {
        let builder = PredicateBuilder::new(self.table.schema().fields());
        filters
            .into_iter()
            .map(|expr| {
                let expr = filter::normalize(expr, &self.schema)?;
                let predicate = filter::paimon(&expr, &builder, false)?;
                Ok((expr, predicate))
            })
            .collect()
    }
}

#[async_trait]
impl TableProvider for PaimonReadOnlyTable {
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
        // Invalid predicates must reach the caller as errors. Returning
        // Unsupported would move the filter above an otherwise unfiltered scan.
        self.translate(filters.iter().copied())
            .map(|filters| vec![TableProviderFilterPushDown::Exact; filters.len()])
    }

    async fn scan(
        &self,
        state: &dyn Session,
        projection: Option<&[usize]>,
        filters: &[Expr],
        limit: Option<usize>,
    ) -> Result<Arc<dyn ExecutionPlan>> {
        let (logical, predicates): (Vec<_>, Vec<_>) =
            self.translate(filters)?.into_iter().unzip();
        let predicate = (!predicates.is_empty()).then(|| Predicate::and(predicates));
        let filter_expr = conjunction(logical);
        let output = projected_schema(&self.schema, projection)?;
        let mut read = self.table.new_read_builder();
        let residual = push_filter(&mut read, predicate.as_ref(), filter_expr.as_ref());
        let columns = read_columns(&output, residual);
        let filter = residual
            .map(|expr| {
                let input = filter_schema(&self.schema, &columns)?;
                state.create_physical_expr(expr.clone(), &input)
            })
            .transpose()?;
        let native_limit = filter.is_none().then_some(limit).flatten();
        configure_read(&mut read, &columns, native_limit)?;

        let (files, planned) = if limit == Some(0) {
            (vec![], HashSet::new())
        } else {
            let plan = read.new_scan().plan().await.map_err(paimon_scan_error)?;
            split_files(self.table.location(), self.snapshot_id, plan.splits())?
        };
        let scan = self.local_scan(filter_expr, columns, predicate, filter, planned);
        scan_plan(
            LakeScan::Paimon(scan),
            self.schema(),
            projection,
            files,
            limit,
        )
    }
}

type SplitFingerprint = [u8; 32];

fn split_files(
    table_uri: &str,
    snapshot_id: i64,
    splits: &[DataSplit],
) -> Result<(Vec<PartitionedFile>, HashSet<SplitFingerprint>)> {
    let mut ordered = Vec::with_capacity(splits.len());
    let mut planned = HashSet::with_capacity(splits.len());
    for split in splits {
        let row_ranges = paimon_ranges(split)?;
        let partition_key_b64 = partition_key(split);
        let first = split.data_files().first().ok_or_else(|| {
            plan_datafusion_err!("Paimon planned a split without data files")
        })?;
        if split.snapshot_id() != snapshot_id || split.is_streaming() {
            return plan_err!(
                "Paimon planned a split outside pinned snapshot {snapshot_id}"
            );
        }
        let mut json = serde_json::to_value(split)
            .map_err(|e| plan_datafusion_err!("Cannot serialize Paimon split: {e}"))?;
        json.sort_all_objects();
        let data_split_json = json.to_string();
        let fingerprint: SplitFingerprint = Sha256::digest(&data_split_json).into();
        if !planned.insert(fingerprint) {
            return plan_err!("Paimon planned a duplicate split");
        }
        let key = (
            partition_key_b64.clone(),
            split.bucket(),
            first.file_name.as_str(),
            row_ranges
                .as_ref()
                .and_then(|ranges| ranges.first())
                .map(|range| range.from),
            fingerprint,
        );
        let size = split
            .data_files()
            .iter()
            .filter_map(|file| u64::try_from(file.file_size).ok())
            .fold(0, u64::saturating_add);
        let file = LakeSplitExt::into_file(
            LakeSplitPayload::Paimon {
                table_uri: table_uri.into(),
                snapshot_id,
                bucket: split.bucket(),
                partition_key_b64,
                raw_convertible: split.raw_convertible(),
                row_ranges,
                data_split_json,
            },
            size,
        )?;
        ordered.push((key, file));
    }
    ordered.sort_unstable_by(|(left, _), (right, _)| left.cmp(right));
    Ok((ordered.into_iter().map(|(_, file)| file).collect(), planned))
}

pub(crate) struct PaimonScanSpec {
    pub(crate) table_uri: String,
    pub(crate) schema_id: i64,
    pub(crate) snapshot_id: i64,
    pub(crate) filter_expr: Option<Expr>,
    pub(crate) required_indices: Vec<RequiredIndexSpec>,
}

#[derive(Clone)]
pub(crate) struct PaimonScan {
    pub(crate) spec: Arc<PaimonScanSpec>,
    handle: Arc<ScanHandle<OpenedPaimonScan>>,
    planned: Option<Arc<HashSet<SplitFingerprint>>>,
}

struct OpenedPaimonScan {
    table: Arc<Table>,
    blobs: BlobColumns,
    columns: Vec<String>,
    predicate: Option<Predicate>,
    filter: Option<Arc<dyn PhysicalExpr>>,
}

impl PaimonScan {
    pub(crate) fn worker(spec: PaimonScanSpec, runtime: Arc<LakeRuntimeConfig>) -> Self {
        Self {
            spec: Arc::new(spec),
            handle: Arc::new(ScanHandle::worker(runtime)),
            planned: None,
        }
    }

    pub(crate) fn is_worker(&self) -> bool {
        self.handle.local().is_none()
    }

    pub(crate) fn check_transportable(&self) -> Result<()> {
        if let Some(opened) = self.handle.local() {
            if !opened.table.is_main_branch() {
                return plan_err!(
                    "The lake scan protocol requires the Paimon main branch"
                );
            }
            check_direct_read(&opened.table)?;
        }
        Ok(())
    }

    pub(crate) fn validate_split(&self, payload: &LakeSplitPayload) -> Result<()> {
        let LakeSplitPayload::Paimon {
            table_uri,
            snapshot_id,
            data_split_json,
            ..
        } = payload
        else {
            return plan_err!("Expected a Paimon split");
        };
        if *snapshot_id != self.spec.snapshot_id
            || *table_uri != self.spec.table_uri
            || self.planned.as_ref().is_some_and(|planned| {
                let fingerprint: SplitFingerprint =
                    Sha256::digest(data_split_json).into();
                !planned.contains(&fingerprint)
            })
        {
            return plan_err!("Paimon split does not match the pinned table scan");
        }
        Ok(())
    }

    async fn open_snapshot(
        &self,
        runtime: Arc<LakeRuntimeConfig>,
        output: &Schema,
    ) -> Result<OpenedPaimonScan> {
        let spec = &self.spec;
        let file_io = runtime.paimon_file_io(&spec.table_uri)?;
        let snapshot = SnapshotManager::new(file_io.clone(), spec.table_uri.clone())
            .get_snapshot(spec.snapshot_id)
            .await
            .map_err(paimon_scan_error)?;
        if snapshot.id() != spec.snapshot_id || snapshot.schema_id() != spec.schema_id {
            return plan_err!(
                "Opened Paimon snapshot does not match the pinned snapshot and schema"
            );
        }
        let schema = SchemaManager::new(file_io.clone(), spec.table_uri.clone())
            .schema(spec.schema_id)
            .await
            .map_err(paimon_scan_error)?;
        if schema.id() != spec.schema_id {
            return plan_err!("Opened Paimon schema does not match the pinned schema");
        }
        let table = Table::from_resolved_schema(
            file_io,
            Identifier::new("lake", "scan"),
            spec.table_uri.clone(),
            schema.as_ref().clone(),
            "main",
        )
        .map_err(paimon_scan_error)?;
        check_direct_read(&table)?;
        let table = pin_table(&table, &snapshot).await?;
        let raw = raw_schema(&table)?;
        let (schema, blobs) = lake_schema(&table, &raw)?;
        for required in &spec.required_indices {
            required.validate_column(&raw)?;
        }

        let filter_expr = spec
            .filter_expr
            .as_ref()
            .map(|expr| filter::normalize(expr, &schema))
            .transpose()?;
        let predicate = filter_expr
            .as_ref()
            .map(|expr| {
                filter::paimon(
                    expr,
                    &PredicateBuilder::new(table.schema().fields()),
                    false,
                )
            })
            .transpose()?;
        let residual = push_filter(
            &mut table.new_read_builder(),
            predicate.as_ref(),
            filter_expr.as_ref(),
        );
        let columns = read_columns(output, residual);
        let filter = residual
            .map(|expr| {
                create_physical_expr(
                    expr,
                    &filter_schema(&schema, &columns)?,
                    &ExecutionProps::new(),
                    &PhysicalPlanningContext::default(),
                )
            })
            .transpose()?;
        Ok(OpenedPaimonScan {
            table: Arc::new(table),
            blobs,
            columns,
            predicate,
            filter,
        })
    }

    pub(crate) async fn open(
        self,
        payload: LakeSplitPayload,
        schema: SchemaRef,
        limit: Option<usize>,
    ) -> Result<BoxStream<'static, Result<RecordBatch>>> {
        let split = payload.paimon_data_split()?;
        let opened = self
            .handle
            .get_or_open(|runtime| self.open_snapshot(runtime, &schema))
            .await?;
        let mut builder = opened.table.new_read_builder();
        if let Some(predicate) = &opened.predicate {
            builder.with_filter(predicate.clone());
        }
        // Paimon may use a data predicate only for pruning. Let FileStream
        // enforce fetch after exact row filtering in that case.
        let native_limit = opened.filter.is_none().then_some(limit).flatten();
        configure_read(&mut builder, &opened.columns, native_limit)?;
        let read = builder.new_read().map_err(paimon_scan_error)?;
        let splits = Box::pin(read.authorize_splits(&[split]))
            .await
            .map_err(paimon_scan_error)?;
        let stream = read.to_arrow(&splits).map_err(paimon_scan_error)?;
        Ok(Box::pin(stream.map_err(paimon_scan_error).and_then(
            move |batch| futures::future::ready(opened.project(batch, &schema)),
        )))
    }

    pub(crate) fn row_filter(&self) -> Option<&Arc<dyn PhysicalExpr>> {
        self.handle
            .local()
            .and_then(|opened| opened.filter.as_ref())
    }
}

impl OpenedPaimonScan {
    fn project(&self, batch: RecordBatch, schema: &SchemaRef) -> Result<RecordBatch> {
        let batch = if let Some(filter) = &self.filter {
            let array = filter.evaluate(&batch)?.into_array(batch.num_rows())?;
            let mask =
                array
                    .as_any()
                    .downcast_ref::<BooleanArray>()
                    .ok_or_else(|| {
                        plan_datafusion_err!("Paimon row filter must return Boolean")
                    })?;
            filter_record_batch(&batch, mask)?
        } else {
            batch
        };
        let columns = schema
            .fields()
            .iter()
            .map(|field| {
                let input = batch.column_by_name(field.name()).ok_or_else(|| {
                    plan_datafusion_err!("Paimon scan omitted column {:?}", field.name())
                })?;
                if let Some(mime) = self.blobs.get(field.name()) {
                    blob_array(input.as_ref(), field, mime.as_deref())
                } else {
                    Ok(Arc::clone(input))
                }
            })
            .collect::<Result<Vec<_>>>()?;
        make_batch(Arc::clone(schema), columns, batch.num_rows())
    }
}

fn check_direct_read(table: &Table) -> Result<()> {
    let options = table.schema().core_options();
    if options.is_format_table() || options.query_auth_enabled() {
        return plan_err!(
            "The lake scan protocol requires a Paimon snapshot readable through FileIO"
        );
    }
    Ok(())
}

async fn pin_table(table: &Table, snapshot: &Snapshot) -> Result<Table> {
    Ok(table
        .copy_with_resolved_snapshot(snapshot)
        .await
        .map_err(|e| plan_datafusion_err!("Cannot load Paimon snapshot schema: {e}"))?
        .copy_with_options(HashMap::from([
            ("blob-as-descriptor".into(), "true".into()),
            ("blob-view.resolve.enabled".into(), "false".into()),
        ])))
}

fn raw_schema(table: &Table) -> Result<SchemaRef> {
    build_read_arrow_schema(table.schema().fields())
        .map_err(|e| plan_datafusion_err!("Cannot convert Paimon snapshot schema: {e}"))
}

fn lake_schema(table: &Table, raw: &Schema) -> Result<(SchemaRef, BlobColumns)> {
    let mut blobs = HashMap::new();
    let fields = raw
        .fields()
        .iter()
        .zip(table.schema().fields())
        .map(|(arrow, paimon)| {
            if matches!(paimon.data_type(), DataType::Blob(_)) {
                blobs.insert(
                    arrow.name().clone(),
                    blob_mime(
                        arrow.name(),
                        |key| arrow.metadata().get(key),
                        |key| table.schema().options().get(key),
                    )?,
                );
                Ok(Arc::new(MediaRefType::field(
                    arrow.name(),
                    arrow.is_nullable(),
                )?))
            } else if contains_blob(paimon.data_type()) {
                not_impl_err!("Paimon nested blob column {:?}", arrow.name())
            } else {
                Ok(Arc::clone(arrow))
            }
        })
        .collect::<Result<Vec<_>>>()?;
    Ok((Arc::new(Schema::new(fields)), Arc::new(blobs)))
}

fn push_filter<'a>(
    read: &mut ReadBuilder<'_>,
    predicate: Option<&Predicate>,
    logical: Option<&'a Expr>,
) -> Option<&'a Expr> {
    let predicate = predicate?;
    read.with_filter(predicate.clone());
    logical.filter(|_| !read.is_exact_filter_pushdown(predicate))
}

fn read_columns(output: &Schema, residual: Option<&Expr>) -> Vec<String> {
    let mut columns: BTreeSet<_> = output
        .fields()
        .iter()
        .map(|field| field.name().as_str())
        .collect();
    if let Some(residual) = residual {
        columns.extend(
            residual
                .column_refs()
                .into_iter()
                .map(|col| col.name.as_str()),
        );
    }
    columns.into_iter().map(str::to_owned).collect()
}

fn filter_schema(schema: &Schema, columns: &[String]) -> Result<DFSchema> {
    let fields = columns
        .iter()
        .map(|name| schema.field_with_name(name).cloned())
        .collect::<std::result::Result<Vec<_>, _>>()?;
    DFSchema::try_from(Schema::new(fields))
}

fn configure_read(
    read: &mut ReadBuilder<'_>,
    columns: &[String],
    limit: Option<usize>,
) -> Result<()> {
    read.with_projection(&columns.iter().map(String::as_str).collect::<Vec<_>>())
        .map_err(paimon_scan_error)?;
    if let Some(limit) = limit {
        read.with_limit(limit);
    }
    Ok(())
}

fn blob_array(
    input: &dyn Array,
    field: &Field,
    mime: Option<&str>,
) -> Result<arrow::array::ArrayRef> {
    fn media(bytes: &[u8], mime: Option<&str>) -> Result<MediaRef> {
        let descriptor = BlobDescriptor::deserialize(bytes).map_err(paimon_scan_error)?;
        descriptor.validate().map_err(paimon_scan_error)?;
        Ok(MediaRef {
            uri: descriptor.uri().into(),
            byte_offset: u64::try_from(descriptor.offset()).ok(),
            byte_len: u64::try_from(descriptor.length()).ok(),
            etag: None,
            mime: mime.map(str::to_owned),
        })
    }
    if let Some(array) = input.as_any().downcast_ref::<LargeBinaryArray>() {
        media_array(
            field.data_type(),
            array
                .iter()
                .map(|bytes| bytes.map(|bytes| media(bytes, mime)).transpose()),
        )
    } else if let Some(array) = input.as_any().downcast_ref::<BinaryArray>() {
        media_array(
            field.data_type(),
            array
                .iter()
                .map(|bytes| bytes.map(|bytes| media(bytes, mime)).transpose()),
        )
    } else {
        exec_err!(
            "Paimon scan did not return binary blob descriptors for {:?}",
            field.name()
        )
    }
}

fn contains_blob(dt: &DataType) -> bool {
    match dt {
        DataType::Blob(_) => true,
        DataType::Row(row) => row
            .fields()
            .iter()
            .any(|field| contains_blob(field.data_type())),
        DataType::Array(array) => contains_blob(array.element_type()),
        DataType::Map(map) => {
            contains_blob(map.key_type()) || contains_blob(map.value_type())
        }
        DataType::Multiset(set) => contains_blob(set.element_type()),
        _ => false,
    }
}

pub(crate) fn paimon_scan_error(error: paimon::Error) -> DataFusionError {
    if matches!(error, paimon::Error::Unsupported { .. }) {
        DataFusionError::NotImplemented(error.to_string())
    } else {
        external(error)
    }
}

async fn validate_indices(
    table: &Table,
    snapshot: &Snapshot,
    schema: &Schema,
    options: &LakeTableOptions,
) -> Result<()> {
    if options.required_indices.is_empty() {
        return Ok(());
    }
    for required in &options.required_indices {
        let column = required.column();
        let field = schema.field_with_name(column).map_err(|_| {
            plan_datafusion_err!("Paimon index column {column:?} does not exist")
        })?;
        required.validate_dimension(field.data_type())?;
    }
    let table_uri = table.location().trim_end_matches('/');
    let Some(manifest) = snapshot.index_manifest() else {
        return plan_err!("Paimon snapshot {} has no index manifest", snapshot.id());
    };
    let entries =
        IndexManifest::read(table.file_io(), &format!("{table_uri}/manifest/{manifest}"))
            .await
            .map_err(|e| {
                plan_datafusion_err!("Cannot read Paimon index manifest: {e}")
            })?;
    let plan = table
        .new_read_builder()
        .new_scan()
        .plan()
        .await
        .map_err(|e| {
            plan_datafusion_err!("Cannot plan Paimon snapshot {}: {e}", snapshot.id())
        })?;
    for required in &options.required_indices {
        let field = table
            .schema()
            .fields()
            .iter()
            .find(|field| field.name() == required.column())
            .ok_or_else(|| {
                plan_datafusion_err!("Missing Paimon index column {}", required.column())
            })?;
        let mut coverage = HashMap::<String, PartitionCoverage<'_>>::new();
        for entry in &entries {
            let file = &entry.index_file;
            let kind = file.index_type.to_ascii_lowercase();
            let matching = match required {
                RequiredIndexSpec::Vector { .. } => {
                    paimon::vindex::is_vindex_index_type(&kind)
                }
                RequiredIndexSpec::FullText { .. } => kind == "full-text",
                RequiredIndexSpec::Scalar { .. } => matches!(
                    kind.as_str(),
                    "btree" | "bitmap" | "multivalue" | "fm" | "rtree"
                ),
            };
            let Some(meta) = &file.global_index_meta else {
                continue;
            };
            if !matching
                || entry.kind != FileKind::Add
                || meta.index_field_id != field.id()
            {
                continue;
            }
            // Only table-global ranges can establish coverage in the stable row-ID
            // space. Primary-key indexes have file-local positions in this field.
            if let Some(source) = &meta.source_meta {
                let Ok(source) = DataEvolutionIndexSourceMeta::deserialize(source) else {
                    continue;
                };
                if source.scan_snapshot_id() > snapshot.id()
                    || source.source_schema_id() != snapshot.schema_id()
                {
                    continue;
                }
            }
            if meta.row_range_start < 0
                || meta.row_range_end < meta.row_range_start
                || file.file_size <= 0
            {
                continue;
            }
            let local = format!("{table_uri}/index/{}", file.file_name);
            let path = file.external_path.as_deref().unwrap_or(&local);
            let input = table
                .file_io()
                .new_input(path)
                .map_err(|e| plan_datafusion_err!("Invalid Paimon index path: {e}"))?;
            let status = input.metadata().await.map_err(|e| {
                plan_datafusion_err!("Paimon index {path:?} is unreadable: {e}")
            })?;
            if status.is_dir || i64::try_from(status.size).ok() != Some(file.file_size) {
                return plan_err!("Paimon index {path:?} is incomplete");
            }
            if let RequiredIndexSpec::Vector { dimension, .. } = required {
                validate_vector_header(&input, &kind, status.size, *dimension).await?;
            }
            coverage
                .entry(kind)
                .or_default()
                .entry((entry.partition.as_slice(), entry.bucket))
                .or_default()
                .push((meta.row_range_start, meta.row_range_end));
        }
        let ready = coverage.values_mut().any(|partitions| {
            for ranges in partitions.values_mut() {
                ranges.sort_unstable();
            }
            covers_splits(partitions, plan.splits())
        });
        if !ready {
            return plan_err!(
                "Required Paimon index {required:?} is missing or does not cover snapshot {}",
                snapshot.id()
            );
        }
    }
    Ok(())
}

type PartitionCoverage<'a> = HashMap<(&'a [u8], i32), Vec<(i64, i64)>>;

fn covers_splits(coverage: &PartitionCoverage<'_>, splits: &[DataSplit]) -> bool {
    splits.iter().all(|split| {
        let partition = split.partition().to_serialized_bytes();
        let ranges = coverage
            .get(&(partition.as_slice(), split.bucket()))
            .map_or(&[][..], Vec::as_slice);
        split.data_files().iter().all(|file| {
            if file.row_count == 0 {
                return true;
            }
            let Some(start) = file.first_row_id else {
                return false;
            };
            let Some(end) = start
                .checked_add(file.row_count)
                .and_then(|end| end.checked_sub(1))
            else {
                return false;
            };
            covers(ranges, start, end)
        })
    })
}

async fn validate_vector_header(
    input: &InputFile,
    kind: &str,
    size: u64,
    expected: u32,
) -> Result<()> {
    use paimon::vindex::{
        DISKANN_IDENTIFIER, IVF_FLAT_IDENTIFIER, IVF_PQ_IDENTIFIER, IVF_RQ_IDENTIFIER,
        IVF_SQ_IDENTIFIER,
    };
    let (magic, version, header_size) = match kind {
        IVF_FLAT_IDENTIFIER => (
            ivfflat_io::IVFFLAT_MAGIC,
            ivfflat_io::IVFFLAT_VERSION,
            ivfflat_io::IVFFLAT_HEADER_SIZE,
        ),
        IVF_PQ_IDENTIFIER => (ivfpq_io::MAGIC, ivfpq_io::VERSION, ivfpq_io::HEADER_SIZE),
        IVF_RQ_IDENTIFIER => (
            ivfrq_io::IVF_RQ_MAGIC,
            ivfrq_io::IVF_RQ_VERSION,
            ivfrq_io::IVF_RQ_HEADER_SIZE,
        ),
        IVF_SQ_IDENTIFIER => (
            ivfsq_io::IVF_SQ_MAGIC,
            ivfsq_io::IVF_SQ_VERSION,
            ivfsq_io::IVF_SQ_HEADER_SIZE,
        ),
        DISKANN_IDENTIFIER => (
            diskann_io::DISKANN_MAGIC,
            diskann_io::DISKANN_VERSION,
            diskann_io::DISKANN_HEADER_SIZE,
        ),
        _ => return plan_err!("Unknown Paimon vector index type {kind:?}"),
    };
    let path = input.location();
    if size < header_size as u64 {
        return plan_err!("Truncated Paimon vector index header at {path:?}");
    }
    let reader = input.reader().await.map_err(|e| {
        plan_datafusion_err!("Cannot open Paimon vector index {path:?}: {e}")
    })?;
    let mut header = reader.read(0..header_size as u64).await.map_err(|e| {
        plan_datafusion_err!("Cannot read Paimon vector index header at {path:?}: {e}")
    })?;
    if header.len() != header_size
        || header.get_u32_le() != magic
        || header.get_u32_le() != version
    {
        return plan_err!("Invalid Paimon vector index header at {path:?}");
    }
    // DiskANN places the header length and flags before the dimension.
    if kind == DISKANN_IDENTIFIER {
        header.advance(8);
    }
    let actual = header.get_u32_le();
    if actual != expected {
        return plan_err!(
            "Paimon vector index {path:?} has dimension {actual}, expected {expected}"
        );
    }
    Ok(())
}

fn covers(ranges: &[(i64, i64)], start: i64, end: i64) -> bool {
    if start < 0 || end < start {
        return false;
    }
    let mut next = start;
    for &(from, to) in ranges {
        if from > next {
            break;
        }
        if to >= end {
            return true;
        }
        next = next.max(to + 1);
    }
    false
}
