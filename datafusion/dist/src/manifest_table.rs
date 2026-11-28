// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: See the NOTICE file distributed with this work for additional information regarding copyright ownership
// SPDX-FileContributor: Licensed to the Apache Software Foundation (ASF) under one or more contributor license agreements

use std::collections::BTreeSet;
use std::path::{Path as FsPath, PathBuf};
use std::sync::Arc;

use datafusion::arrow::datatypes::SchemaRef;
use datafusion::catalog::{Session, TableProvider};
use datafusion::common::{Result, exec_datafusion_err, exec_err, not_impl_err};
use datafusion::datasource::file_format::{FileFormat, parquet::ParquetFormat};
use datafusion::datasource::listing::PartitionedFile;
use datafusion::datasource::physical_plan::{
    FileGroup, FileScanConfigBuilder, ParquetSource,
};
use datafusion::execution::object_store::ObjectStoreUrl;
use datafusion::logical_expr::{Expr, TableType};
use datafusion::physical_plan::ExecutionPlan;
use datafusion::prelude::SessionContext;
use futures::{StreamExt, TryStreamExt, future::BoxFuture, stream};
use object_store::path::Path;
use object_store::{
    Error as StoreError, GetResult, GetResultPayload, ObjectMeta, ObjectStore,
    ObjectStoreExt,
};

use crate::StageManifest;
use crate::sink::{
    CHECKPOINT_SCHEMA, MANIFEST_FILE, MANIFEST_SCHEMA, is_sha256, stage_path,
};

#[cfg(test)]
mod tests;

/// Read the data and dead-letter tables of a finalized stage.
///
/// The tables use the winning Parquet paths recorded in `STAGE_MANIFEST.json`.
/// Register them in a session with the same object store binding used at open.
/// Projection and limit can be pushed into the scan. DataFusion evaluates SQL
/// filters above the scan.
///
/// ```no_run
/// # async fn example() -> datafusion::common::Result<()> {
/// use datafusion::execution::object_store::ObjectStoreUrl;
/// use datafusion::object_store::path::Path;
/// use datafusion::prelude::SessionContext;
/// use datafusion_dist::StageOutputReader;
///
/// let ctx = SessionContext::new();
/// let stage = StageOutputReader::open(
///     &ctx,
///     ObjectStoreUrl::local_filesystem(),
///     &Path::from("tmp/job-output"),
/// ).await?;
/// ctx.register_table("upstream", stage.data_table())?;
/// let batches = ctx.sql("SELECT COUNT(*) FROM upstream").await?.collect().await?;
/// # Ok(())
/// # }
/// ```
#[derive(Debug)]
pub struct StageOutputReader {
    data: Arc<dyn TableProvider>,
    dlq: Arc<dyn TableProvider>,
    pub(crate) store: Arc<dyn ObjectStore>,
    pub(crate) store_url: ObjectStoreUrl,
    pub(crate) stage_root: Path,
    pub(crate) manifest: StageManifest,
}

impl StageOutputReader {
    /// Open a stage directory or a job output root with one finalized stage.
    ///
    /// `output_root` is relative to the registered object store. Under a job
    /// root, the reader discovers manifests in `stages/<plan_fingerprint>/`.
    /// Ambiguous job roots and manifests with no splits return an error.
    /// The reader checks each winner's presence and the data file byte count.
    pub async fn open(
        ctx: &SessionContext,
        store_url: ObjectStoreUrl,
        output_root: &Path,
    ) -> Result<Self> {
        let state = ctx.state();
        let store = state.runtime_env().object_store(&store_url)?;
        let (manifest_path, object) = find_manifest(store.as_ref(), output_root).await?;
        let stage_root = manifest_path.parent().expect("manifest has a filename");
        let local_root = match &object.payload {
            GetResultPayload::File(_, path) => {
                Some(local_stage_root(path, &manifest_path, output_root).await?)
            }
            GetResultPayload::Stream(_) => None,
        };
        let manifest: StageManifest = serde_json::from_slice(&object.bytes().await?)
            .map_err(|error| {
                exec_datafusion_err!("Invalid stage manifest at {manifest_path}: {error}")
            })?;
        validate_manifest(&manifest, &stage_root)?;

        let (data, dlq): (Vec<_>, Vec<_>) = stream::iter(&manifest.splits)
            .map(|split| async {
                let files = &split.files;
                let data = head_winner(
                    store.as_ref(),
                    &stage_root,
                    local_root.as_deref(),
                    &files.data_rel_path,
                )
                .await?;
                if data.size != files.data_bytes {
                    return exec_err!(
                        "Manifest data size mismatch at {}: expected {}, got {}",
                        data.location,
                        files.data_bytes,
                        data.size
                    );
                }
                let dlq = match &files.dlq_rel_path {
                    Some(path) => Some(
                        head_winner(
                            store.as_ref(),
                            &stage_root,
                            local_root.as_deref(),
                            path,
                        )
                        .await?,
                    ),
                    None => None,
                };
                Ok((data, dlq))
            })
            .buffered(
                state
                    .config_options()
                    .execution
                    .meta_fetch_concurrency
                    .get(),
            )
            .try_collect::<Vec<_>>()
            .await?
            .into_iter()
            .unzip();
        let dlq: Vec<_> = dlq.into_iter().flatten().collect();
        let format = ParquetFormat::new();
        let data_schema = format.infer_schema(&state, &store, &data).await?;
        let dlq_schema = if dlq.is_empty() {
            Arc::clone(&data_schema)
        } else {
            format.infer_schema(&state, &store, &dlq).await?
        };
        Ok(Self {
            data: Arc::new(WinnerTable::new(store_url.clone(), data_schema, data)),
            dlq: Arc::new(WinnerTable::new(store_url.clone(), dlq_schema, dlq)),
            store,
            store_url,
            stage_root,
            manifest,
        })
    }

    /// Return a provider for the winning data files, including zero-row outputs.
    pub fn data_table(&self) -> Arc<dyn TableProvider> {
        Arc::clone(&self.data)
    }

    /// Return a provider for the winning dead-letter files.
    ///
    /// When the stage produced no dead-letter files, this is a zero-row table
    /// with the data table's schema. Use `COUNT(*)` with either schema.
    pub fn dlq_table(&self) -> Arc<dyn TableProvider> {
        Arc::clone(&self.dlq)
    }
}

async fn find_manifest(
    store: &dyn ObjectStore,
    output_root: &Path,
) -> Result<(Path, GetResult)> {
    let direct = output_root.clone().join(MANIFEST_FILE);
    match store.get(&direct).await {
        Ok(object) => return Ok((direct, object)),
        Err(StoreError::NotFound { .. }) => (),
        Err(error) => return Err(error.into()),
    }

    let stages = output_root.clone().join("stages");
    let mut selected = None;
    for stage in store
        .list_with_delimiter(Some(&stages))
        .await?
        .common_prefixes
    {
        if stage.parent().as_ref() != Some(&stages)
            || !stage.filename().is_some_and(is_sha256)
        {
            continue;
        }
        let path = stage.join(MANIFEST_FILE);
        match store.head(&path).await {
            Ok(_) if selected.is_some() => {
                return exec_err!(
                    "Multiple finalized stages under {output_root}; use a stage directory"
                );
            }
            Ok(_) => selected = Some(path),
            Err(StoreError::NotFound { .. }) => (),
            Err(error) => return Err(error.into()),
        }
    }
    let path = selected
        .ok_or_else(|| exec_datafusion_err!("No {MANIFEST_FILE} under {output_root}"))?;
    let object = store.get(&path).await?;
    Ok((path, object))
}

async fn local_stage_root(
    path: &FsPath,
    manifest_path: &Path,
    output_root: &Path,
) -> Result<PathBuf> {
    // LocalFileSystem follows symlinks. Resolve the caller's root first, then
    // require the manifest and its winners to resolve to their exact paths.
    let relative: PathBuf = manifest_path
        .prefix_match(output_root)
        .ok_or_else(|| exec_datafusion_err!("Stage manifest is outside its output root"))?
        .map(|part| part.as_ref().to_owned())
        .collect();
    let base = path
        .ancestors()
        .nth(relative.components().count())
        .ok_or_else(|| {
            exec_datafusion_err!("Cannot resolve the local stage output root")
        })?;
    let expected = tokio::fs::canonicalize(base).await?.join(relative);
    check_local_file(&expected).await?;
    Ok(expected
        .parent()
        .expect("manifest has a filename")
        .to_path_buf())
}

async fn check_local_file(path: &FsPath) -> Result<()> {
    let resolved = tokio::fs::canonicalize(path).await.map_err(|error| {
        exec_datafusion_err!("Cannot resolve stage file {}: {error}", path.display())
    })?;
    if resolved != path || !tokio::fs::metadata(&resolved).await?.is_file() {
        return exec_err!(
            "Stage file must be a regular file without symlinks under its output root: {}",
            path.display()
        );
    }
    Ok(())
}

pub(crate) fn validate_manifest(
    manifest: &StageManifest,
    stage_root: &Path,
) -> Result<()> {
    if manifest.schema != MANIFEST_SCHEMA {
        return exec_err!("Unsupported stage manifest schema");
    }
    if !is_sha256(&manifest.plan_fingerprint) {
        return exec_err!("Invalid stage manifest plan fingerprint");
    }
    if stage_root
        .parent()
        .is_some_and(|parent| parent.filename() == Some("stages"))
        && stage_root.filename() != Some(manifest.plan_fingerprint.as_str())
    {
        return exec_err!("Stage directory does not match the manifest plan fingerprint");
    }
    if manifest.splits.is_empty() {
        return exec_err!("Stage manifest has no winning splits to infer a data schema");
    }
    let mut previous_id = None;
    let mut fingerprints = BTreeSet::new();
    let mut data_rows = 0_u64;
    let mut dlq_rows = 0_u64;
    for split in &manifest.splits {
        if previous_id.is_some_and(|id| id >= split.split_id) {
            return exec_err!("Manifest split IDs must be strictly increasing");
        }
        previous_id = Some(split.split_id);
        if split.schema != CHECKPOINT_SCHEMA
            || split.plan_fingerprint != manifest.plan_fingerprint
            || !is_sha256(&split.split_fingerprint)
            || !fingerprints.insert(&split.split_fingerprint)
        {
            return exec_err!(
                "Invalid manifest checkpoint identity or fingerprint for split {}",
                split.split_id
            );
        }
        split
            .files
            .validate(split.split_id, split.winning_attempt_id)?;
        data_rows = checked_add_rows(data_rows, split.files.output_rows)?;
        dlq_rows = checked_add_rows(dlq_rows, split.files.dlq_rows)?;
    }
    if (data_rows, dlq_rows) != (manifest.output_rows, manifest.dlq_rows) {
        return exec_err!("Manifest row totals do not match the winning splits");
    }
    Ok(())
}

fn checked_add_rows(total: u64, rows: u64) -> Result<u64> {
    total
        .checked_add(rows)
        .ok_or_else(|| exec_datafusion_err!("Manifest row count overflow"))
}

async fn head_winner(
    store: &dyn ObjectStore,
    stage_root: &Path,
    local_root: Option<&FsPath>,
    rel_path: &str,
) -> Result<ObjectMeta> {
    if let Some(root) = local_root {
        check_local_file(&root.join(rel_path)).await?;
    }
    let path = stage_path(stage_root, rel_path);
    let meta = store.head(&path).await?;
    if meta.size == 0 || meta.location != path {
        return exec_err!("Invalid metadata for winning Parquet file {path}");
    }
    Ok(meta)
}

#[derive(Debug)]
struct WinnerTable {
    store_url: ObjectStoreUrl,
    schema: SchemaRef,
    files: Vec<PartitionedFile>,
}

impl WinnerTable {
    fn new(store_url: ObjectStoreUrl, schema: SchemaRef, files: Vec<ObjectMeta>) -> Self {
        Self {
            store_url,
            schema,
            files: files
                .into_iter()
                .map(PartitionedFile::new_from_meta)
                .collect(),
        }
    }
}

impl TableProvider for WinnerTable {
    fn schema(&self) -> SchemaRef {
        Arc::clone(&self.schema)
    }

    fn table_type(&self) -> TableType {
        TableType::Base
    }

    fn scan<'a, 'b, 'c, 'd, 'f>(
        &'a self,
        state: &'b dyn Session,
        projection: Option<&'c [usize]>,
        filters: &'d [Expr],
        limit: Option<usize>,
    ) -> BoxFuture<'f, Result<Arc<dyn ExecutionPlan>>>
    where
        'a: 'f,
        'b: 'f,
        'c: 'f,
        'd: 'f,
        Self: 'f,
    {
        Box::pin(async move {
            if !filters.is_empty() {
                return not_impl_err!(
                    "Stage manifest tables do not support filter pushdown"
                );
            }
            let files = FileGroup::new(self.files.clone());
            let groups = if limit.is_some() || files.is_empty() {
                vec![files]
            } else {
                files.split_files(state.config_options().execution.target_partitions)
            };
            let scan = FileScanConfigBuilder::new(
                self.store_url.clone(),
                Arc::new(ParquetSource::new(Arc::clone(&self.schema))),
            )
            .with_file_groups(groups)
            .with_projection_indices(projection.map(<[usize]>::to_vec))?
            .with_limit(limit)
            .build();
            ParquetFormat::new().create_physical_plan(state, scan).await
        })
    }
}
