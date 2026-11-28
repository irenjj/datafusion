// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: See the NOTICE file distributed with this work for additional information regarding copyright ownership
// SPDX-FileContributor: Licensed to the Apache Software Foundation (ASF) under one or more contributor license agreements

use std::collections::{BTreeMap, HashSet};
use std::num::NonZeroUsize;
use std::sync::Arc;

use datafusion::arrow::array::{Array, ArrayRef, AsArray, BooleanArray};
use datafusion::arrow::compute::{cast, filter_record_batch};
use datafusion::arrow::datatypes::{DataType, Schema, SchemaRef};
use datafusion::arrow::record_batch::RecordBatch;
use datafusion::arrow::row::{RowConverter, Rows, SortField};
use datafusion::common::{
    Result, exec_datafusion_err, exec_err, plan_datafusion_err, plan_err,
};
use datafusion::dataframe::DataFrame;
use datafusion::execution::object_store::ObjectStoreUrl;
use datafusion::parquet::arrow::{
    ProjectionMask, arrow_reader::ParquetRecordBatchReaderBuilder,
};
use datafusion::parquet::file::reader::ChunkReader;
use datafusion::prelude::SessionContext;
use futures::TryStreamExt;
use object_store::path::Path;
use object_store::{ObjectStore, ObjectStoreExt, PutPayload};
use serde::Serialize;

use crate::fingerprint::split_fingerprint;
use crate::manifest_table::validate_manifest;
use crate::sink::{
    DATA_FILE, DLQ_FILE, DLQ_HASH_METADATA, MANIFEST_FILE, SplitCheckpoint, attempt_path,
    checksum, encode_parquet, json_bytes, put_create, read_json, stage_path,
    write_immutable,
};
use crate::{SplitMetrics, StageManifest, StageOutputReader};

type Keys = HashSet<Box<[u8]>>;

/// A dead-letter replay rooted at a validated upstream stage and a new output directory.
///
/// Materialize the failed source rows with [`Self::stage_input`], run the replay
/// SQL as an ordinary distributed stage over that snapshot, then call [`Self::merge`].
/// The final manifest is published atomically after validating the winning rows.
///
/// ```no_run
/// # async fn example() -> datafusion::common::Result<()> {
/// use datafusion::execution::object_store::ObjectStoreUrl;
/// use datafusion::object_store::path::Path;
/// use datafusion::prelude::SessionContext;
/// use datafusion_dist::{DlqReplay, StageOutputReader};
///
/// let ctx = SessionContext::new();
/// let store = ObjectStoreUrl::local_filesystem();
/// let base = StageOutputReader::open(&ctx, store.clone(), &Path::from("tmp/base")).await?;
/// let replay = DlqReplay::open(&ctx, base, store, Path::from("tmp/clean"), "id", b"copy").await?;
/// if !replay.needs_replay() && replay.completed(&ctx).await?.is_none() {
///     replay.merge(&ctx, replay.reserve_attempt().await?, None).await?;
/// }
/// # Ok(())
/// # }
/// ```
pub struct DlqReplay {
    base: StageOutputReader,
    schema: SchemaRef,
    failed: Keys,
    row_key: String,
    output_url: ObjectStoreUrl,
    output_root: Path,
    store: Arc<dyn ObjectStore>,
    fingerprint: String,
}

/// Row counts and incremental inference usage for the winning replay.
///
/// The two data counts and `remaining_dlq_rows` count unique row keys.
/// `manifest.dlq_rows` counts failed cells, which can exceed the failed key count.
#[derive(Debug, Serialize)]
pub struct ReplaySummary {
    /// Successful upstream rows reused byte for byte.
    pub prev_data_rows: u64,
    /// Failed upstream rows that succeeded on replay.
    pub new_data_rows: u64,
    /// Upstream keys that remain in the dead-letter queue.
    pub remaining_dlq_rows: u64,
    /// Published manifest with inference counters for this replay's winning splits.
    #[serde(flatten)]
    pub manifest: StageManifest,
}

impl DlqReplay {
    /// Verify the upstream manifest, its winning file contents, and its row keys.
    ///
    /// The output root must be dedicated to this replay. Use `identity` to bind
    /// it to the exact SQL and source configuration.
    pub async fn open(
        ctx: &SessionContext,
        base: StageOutputReader,
        output_url: ObjectStoreUrl,
        output_root: Path,
        row_key: &str,
        identity: &[u8],
    ) -> Result<Self> {
        let store = ctx.runtime_env().object_store(&output_url)?;
        if (base.store_url == output_url || Arc::ptr_eq(&base.store, &store))
            && (base.stage_root.prefix_matches(&output_root)
                || output_root.prefix_matches(&base.stage_root))
        {
            return plan_err!(
                "Replay output and base stage directories must be disjoint"
            );
        }
        let fingerprint = checksum(&json_bytes(&(
            "df.dist.replay.v1",
            &base.manifest,
            row_key,
            identity,
        ))?);
        let inventory = inspect_stage(&base, row_key, None).await?;
        let replay = Self {
            base,
            schema: inventory.schema,
            failed: inventory.failed,
            row_key: row_key.into(),
            output_url,
            output_root,
            store,
            fingerprint,
        };
        let staged = replay
            .store
            .list_with_delimiter(Some(&replay.output_root.clone().join("stages")))
            .await?;
        if !staged.common_prefixes.is_empty() || !staged.objects.is_empty() {
            return plan_err!("Replay output root already contains a standard stage");
        }
        Ok(replay)
    }

    /// Whether the base contains failed keys requiring inference.
    pub fn needs_replay(&self) -> bool {
        !self.failed.is_empty()
    }

    /// Load the winner if this replay already published its manifest.
    pub async fn completed(&self, ctx: &SessionContext) -> Result<Option<ReplaySummary>> {
        let path = self.output_root.clone().join(MANIFEST_FILE);
        let Some(manifest) =
            read_json::<StageManifest>(self.store.as_ref(), &path).await?
        else {
            return Ok(None);
        };
        if manifest.plan_fingerprint != self.fingerprint {
            return exec_err!(
                "Replay output root already has a different published stage"
            );
        }
        let winner =
            StageOutputReader::open(ctx, self.output_url.clone(), &self.output_root)
                .await?;
        self.summarize(winner.manifest).map(Some)
    }

    /// Reserve an isolated workspace and output attempt ID using atomic create.
    pub async fn reserve_attempt(&self) -> Result<u32> {
        for id in 0..=u32::MAX {
            let reservation = self.workspace(id).join("RESERVED");
            if put_create(self.store.as_ref(), &reservation, self.fingerprint.clone())
                .await?
            {
                return Ok(id);
            }
        }
        exec_err!("Replay output has exhausted attempt IDs")
    }

    /// Store-relative directory for this attempt's snapshot and intermediate stage.
    pub fn workspace(&self, attempt_id: u32) -> Path {
        self.output_root
            .clone()
            .join("_replay")
            .join(format!("{attempt_id:08x}"))
    }

    /// Materialize only the failed keys in Parquet under `workspace/input/`.
    ///
    /// The caller supplies a scan of the source table. Each failed key must
    /// appear exactly once in the source. `batch_size` bounds rows per file,
    /// which become independent splits for the standard distributed scheduler.
    pub async fn stage_input(
        &self,
        attempt_id: u32,
        source: DataFrame,
        batch_size: NonZeroUsize,
    ) -> Result<()> {
        self.verify_reservation(attempt_id).await?;
        let index = key_index(source.schema().as_arrow(), &self.row_key)?;
        let codec = KeyCodec::new(
            self.schema
                .field(key_index(&self.schema, &self.row_key)?)
                .data_type(),
        )?;
        let mut stream = source.execute_stream().await?;
        let schema = stream.schema();
        let folder = self.workspace(attempt_id).join("input");
        let mut pending = Vec::new();
        let mut selected = Keys::new();
        let mut rows = 0;
        let mut part = 0_u64;
        while let Some(batch) = stream.try_next().await? {
            let keys = codec.encode(batch.column(index))?;
            let mut mask = Vec::with_capacity(batch.num_rows());
            for key in &keys {
                let keep = self.failed.contains(key.as_ref());
                if keep && !selected.insert(key.as_ref().into()) {
                    return exec_err!(
                        "Replay source has a duplicate row key for a failed row"
                    );
                }
                mask.push(keep);
            }
            let filtered = filter_record_batch(&batch, &BooleanArray::from(mask))?;
            let mut offset = 0;
            while offset < filtered.num_rows() {
                let count = (batch_size.get() - rows).min(filtered.num_rows() - offset);
                pending.push(filtered.slice(offset, count));
                rows += count;
                offset += count;
                if rows == batch_size.get() {
                    self.write_input_part(&folder, part, &schema, &pending)
                        .await?;
                    part += 1;
                    pending.clear();
                    rows = 0;
                }
            }
        }
        if selected.len() != self.failed.len() {
            return exec_err!("Replay source is missing failed row keys");
        }
        if !pending.is_empty() {
            self.write_input_part(&folder, part, &schema, &pending)
                .await?;
        }
        Ok(())
    }

    /// Publish unchanged base data plus this attempt's winning retry data and DLQ.
    ///
    /// Supply `None` when [`Self::needs_replay`] is false. If two valid attempts
    /// compete, the first manifest published wins and both callers read its result.
    pub async fn merge(
        &self,
        ctx: &SessionContext,
        attempt_id: u32,
        retried: Option<StageOutputReader>,
    ) -> Result<ReplaySummary> {
        self.verify_reservation(attempt_id).await?;
        if retried.is_some() != self.needs_replay() {
            return exec_err!(
                "Replay requires a retry stage exactly when failed keys exist"
            );
        }
        if let Some(stage) = &retried {
            let retry = inspect_stage(stage, &self.row_key, Some(&self.schema)).await?;
            if retry.data.len().checked_add(retry.failed.len()) != Some(self.failed.len())
                || !retry.data.is_subset(&self.failed)
                || !retry.failed.is_subset(&self.failed)
            {
                return exec_err!(
                    "Replay row conservation failed: output must cover exactly the upstream failed keys"
                );
            }
        }
        let mut checkpoints = Vec::new();
        for (stage, reuse) in std::iter::once((&self.base, true))
            .chain(retried.iter().map(|stage| (stage, false)))
        {
            for checkpoint in &stage.manifest.splits {
                let copied = self
                    .copy_split(
                        stage,
                        checkpoint,
                        checkpoints.len() as u64,
                        attempt_id,
                        reuse,
                    )
                    .await?;
                checkpoints.push(copied);
            }
        }
        let manifest = StageManifest::from_checkpoints(&self.fingerprint, checkpoints)?;
        validate_manifest(&manifest, &self.output_root)?;
        let summary = self.summarize(manifest)?;
        let path = self.output_root.clone().join(MANIFEST_FILE);
        if put_create(self.store.as_ref(), &path, json_bytes(&summary.manifest)?).await? {
            Ok(summary)
        } else {
            self.completed(ctx).await?.ok_or_else(|| {
                exec_datafusion_err!("Published replay manifest disappeared")
            })
        }
    }

    async fn verify_reservation(&self, attempt_id: u32) -> Result<()> {
        let path = self.workspace(attempt_id).join("RESERVED");
        if self.store.get(&path).await?.bytes().await?.as_ref()
            != self.fingerprint.as_bytes()
        {
            return exec_err!("Replay attempt belongs to a different replay request");
        }
        Ok(())
    }

    async fn write_input_part(
        &self,
        folder: &Path,
        part: u64,
        schema: &SchemaRef,
        batches: &[RecordBatch],
    ) -> Result<()> {
        let path = folder.clone().join(format!("part-{part:016x}.parquet"));
        let data = encode_parquet(schema, batches, None)?;
        let sha256 = checksum(&data);
        write_immutable(self.store.as_ref(), &path, data, &sha256).await
    }

    async fn copy_split(
        &self,
        stage: &StageOutputReader,
        original: &SplitCheckpoint,
        split_id: u64,
        attempt_id: u32,
        reuse: bool,
    ) -> Result<SplitCheckpoint> {
        let mut split = original.clone();
        split.plan_fingerprint.clone_from(&self.fingerprint);
        split.split_fingerprint = split_fingerprint(
            &self.fingerprint,
            &json_bytes(&(attempt_id, split_id, &original.split_fingerprint))?,
        );
        split.winning_attempt_id = attempt_id;
        split.split_id = split_id;
        // Replay assigns new fingerprints that no longer identify the original Lake splits.
        split.lake_source = None;
        let files = &mut split.files;
        let data =
            verified_payload(stage, &files.data_rel_path, &files.data_sha256).await?;
        files.data_rel_path = attempt_path(split_id, attempt_id, DATA_FILE);
        write_immutable(
            self.store.as_ref(),
            &stage_path(&self.output_root, &files.data_rel_path),
            data,
            &files.data_sha256,
        )
        .await?;
        if reuse {
            files.dlq_rows = 0;
            files.dlq_by_code.clear();
            files.dlq_rel_path = None;
            files.dlq_sha256 = None;
            split.metrics = SplitMetrics {
                output_rows: files.output_rows as usize,
                ..Default::default()
            };
        } else if let (Some(path), Some(sha256)) =
            (&files.dlq_rel_path, &files.dlq_sha256)
        {
            let dlq = verified_payload(stage, path, sha256).await?;
            let path = attempt_path(split_id, attempt_id, DLQ_FILE);
            write_immutable(
                self.store.as_ref(),
                &stage_path(&self.output_root, &path),
                dlq,
                sha256,
            )
            .await?;
            files.dlq_rel_path = Some(path);
        }
        let marker = self
            .output_root
            .clone()
            .join("splits")
            .join(split.split_fingerprint.as_str())
            .join("COMMITTED.json");
        let checkpoint = json_bytes(&split)?;
        let digest = checksum(&checkpoint);
        write_immutable(self.store.as_ref(), &marker, checkpoint, &digest).await?;
        Ok(split)
    }

    fn summarize(&self, manifest: StageManifest) -> Result<ReplaySummary> {
        let prev_data_rows = self.base.manifest.output_rows;
        let new_data_rows = manifest
            .output_rows
            .checked_sub(prev_data_rows)
            .ok_or_else(|| exec_datafusion_err!("Replay lost upstream data rows"))?;
        let remaining_dlq_rows = (self.failed.len() as u64)
            .checked_sub(new_data_rows)
            .ok_or_else(|| {
                exec_datafusion_err!("Replay output exceeds upstream row count")
            })?;
        if (manifest.dlq_rows == 0) != (remaining_dlq_rows == 0)
            || manifest.dlq_rows < remaining_dlq_rows
        {
            return exec_err!("Replay row conservation failed: invalid DLQ totals");
        }
        Ok(ReplaySummary {
            prev_data_rows,
            new_data_rows,
            remaining_dlq_rows,
            manifest,
        })
    }
}

struct Inventory {
    schema: SchemaRef,
    data: Keys,
    failed: Keys,
}

async fn inspect_stage(
    stage: &StageOutputReader,
    row_key: &str,
    expected_schema: Option<&SchemaRef>,
) -> Result<Inventory> {
    let manifest = &stage.manifest;
    let totals = StageManifest::from_checkpoints(
        &manifest.plan_fingerprint,
        manifest.splits.clone(),
    )?;
    if totals != *manifest {
        return exec_err!("Replay manifest totals do not match its checkpoints");
    }
    let mut schema = expected_schema.cloned();
    let mut data_keys = Keys::new();
    let mut failed_keys = Keys::new();
    for split in &manifest.splits {
        let files = &split.files;
        let bytes =
            verified_payload(stage, &files.data_rel_path, &files.data_sha256).await?;
        let builder = ParquetRecordBatchReaderBuilder::try_new(bytes)?;
        // The sink stores the DLQ checksum as a per-file Parquet footer entry.
        let mut file_schema = builder.schema().as_ref().clone();
        file_schema.metadata.remove(DLQ_HASH_METADATA);
        let inventory_schema =
            schema.get_or_insert_with(|| Arc::new(file_schema.clone()));
        if inventory_schema.as_ref() != &file_schema {
            return exec_err!(
                "Replay data schema differs from the upstream winner schema"
            );
        }
        check_footer_rows(&builder, files.output_rows)?;
        let index = key_index(inventory_schema, row_key)?;
        let codec = KeyCodec::new(inventory_schema.field(index).data_type())?;
        let mask = ProjectionMask::roots(builder.parquet_schema(), [index]);
        let mut data_rows = 0_u64;
        for batch in builder.with_projection(mask).build()? {
            let batch = batch?;
            for key in &codec.encode(batch.column(0))? {
                if !data_keys.insert(key.as_ref().into()) {
                    return exec_err!("Replay data contains a duplicate row key");
                }
            }
            data_rows += batch.num_rows() as u64;
        }
        let mut failed_in_split = 0;
        if let (Some(path), Some(sha256)) = (&files.dlq_rel_path, &files.dlq_sha256) {
            failed_in_split = inspect_dlq(
                stage,
                split,
                path,
                sha256,
                &codec,
                inventory_schema,
                &mut failed_keys,
            )
            .await?;
        }
        let expected_rows = data_rows
            .checked_add(failed_in_split)
            .ok_or_else(|| exec_datafusion_err!("Replay split row count overflow"))?;
        // Without a root row count, multiple cells could represent one row or
        // duplicate keys on different rows that failed different columns.
        if split.metrics.output_rows == 0 && files.dlq_rows > failed_in_split {
            return exec_err!(
                "Replay requires a split row count to verify multiple DLQ cells for one row key"
            );
        }
        if data_rows != files.output_rows
            || ((expected_schema.is_some() || split.metrics.output_rows > 0)
                && expected_rows != split.metrics.output_rows as u64)
        {
            return exec_err!(
                "Replay split row count disagrees with its checkpoint (possible duplicate row key)"
            );
        }
    }
    if !data_keys.is_disjoint(&failed_keys) {
        return exec_err!("Replay data and DLQ contain a duplicate row key");
    }
    Ok(Inventory {
        schema: schema
            .ok_or_else(|| exec_datafusion_err!("Replay stage has no data schema"))?,
        data: data_keys,
        failed: failed_keys,
    })
}

async fn inspect_dlq(
    stage: &StageOutputReader,
    split: &SplitCheckpoint,
    path: &str,
    sha256: &str,
    codec: &KeyCodec,
    data_schema: &Schema,
    failed: &mut Keys,
) -> Result<u64> {
    let bytes = verified_payload(stage, path, sha256).await?;
    let builder = ParquetRecordBatchReaderBuilder::try_new(bytes)?;
    let [key, column, code, message, retryable] = builder.schema().fields().as_ref()
    else {
        return exec_err!("Invalid replay DLQ schema");
    };
    if key.name() != "row_key"
        || [
            (column, "failed_column"),
            (code, "error_code"),
            (message, "error_message"),
        ]
        .iter()
        .any(|(field, name)| {
            field.name() != *name || field.data_type() != &DataType::Utf8
        })
        || retryable.name() != "retryable"
        || retryable.data_type() != &DataType::Boolean
        || builder
            .schema()
            .fields()
            .iter()
            .any(|field| field.is_nullable())
    {
        return exec_err!("Invalid replay DLQ schema");
    }
    check_footer_rows(&builder, split.files.dlq_rows)?;
    let mut cells: BTreeMap<Box<[u8]>, HashSet<String>> = BTreeMap::new();
    let mut by_code = BTreeMap::new();
    for batch in builder.build()? {
        let batch = batch?;
        let keys = codec.encode(batch.column(0))?;
        let columns = batch.column(1).as_string::<i32>();
        let codes = batch.column(2).as_string::<i32>();
        for (i, key) in keys.iter().enumerate() {
            let name = columns.value(i);
            if data_schema.index_of(name).is_err() {
                return exec_err!("Replay DLQ names an unknown failed column");
            }
            if !cells
                .entry(key.as_ref().into())
                .or_default()
                .insert(name.to_owned())
            {
                return exec_err!("Replay DLQ has a duplicate row key and failed column");
            }
            *by_code.entry(codes.value(i).to_owned()).or_insert(0_u64) += 1;
        }
    }
    if by_code != split.files.dlq_by_code {
        return exec_err!("Replay DLQ row counts disagree with its checkpoint");
    }
    let count = cells.len() as u64;
    for key in cells.into_keys() {
        if !failed.insert(key) {
            return exec_err!("Replay DLQ has a duplicate row key across splits");
        }
    }
    Ok(count)
}

async fn verified_payload(
    stage: &StageOutputReader,
    relative: &str,
    sha256: &str,
) -> Result<impl ChunkReader + Into<PutPayload> + use<>> {
    let path = stage_path(&stage.stage_root, relative);
    let bytes = stage.store.get(&path).await?.bytes().await?;
    if checksum(&bytes) != sha256 {
        return exec_err!("Replay winner checksum does not match at {path}");
    }
    Ok(bytes)
}

fn check_footer_rows<T: ChunkReader + 'static>(
    builder: &ParquetRecordBatchReaderBuilder<T>,
    expected: u64,
) -> Result<()> {
    if u64::try_from(builder.metadata().file_metadata().num_rows()).ok() != Some(expected)
    {
        return exec_err!(
            "Replay winner Parquet row count disagrees with its checkpoint"
        );
    }
    Ok(())
}

fn key_index(schema: &Schema, name: &str) -> Result<usize> {
    let mut matches = schema
        .fields()
        .iter()
        .enumerate()
        .filter(|(_, field)| field.name() == name);
    match (matches.next(), matches.next()) {
        (Some((index, _)), None) if !name.trim().is_empty() => Ok(index),
        _ => plan_err!("Replay row key must name exactly one column: '{name}'"),
    }
}

fn key_type(data_type: &DataType) -> DataType {
    match data_type {
        DataType::Utf8 | DataType::LargeUtf8 | DataType::Utf8View => DataType::Utf8View,
        DataType::Binary | DataType::LargeBinary | DataType::BinaryView => {
            DataType::BinaryView
        }
        DataType::Dictionary(_, value) => key_type(value),
        other => other.clone(),
    }
}

struct KeyCodec {
    data_type: DataType,
    converter: RowConverter,
}

impl KeyCodec {
    fn new(data_type: &DataType) -> Result<Self> {
        let data_type = key_type(data_type);
        let converter = RowConverter::new(vec![SortField::new(data_type.clone())])
            .map_err(|error| {
                plan_datafusion_err!("Unsupported replay row key type: {error}")
            })?;
        Ok(Self {
            data_type,
            converter,
        })
    }

    fn encode(&self, array: &ArrayRef) -> Result<Rows> {
        let target = &self.data_type;
        if &key_type(array.data_type()) != target {
            return plan_err!(
                "Replay row key type is inconsistent with the upstream data"
            );
        }
        let array = if array.data_type() == target {
            Arc::clone(array)
        } else {
            cast(array, target)?
        };
        if array.logical_null_count() != 0 {
            return exec_err!("Replay row key must be non-null");
        }
        let empty = match target {
            DataType::Utf8View => {
                array.as_string_view().iter().flatten().any(str::is_empty)
            }
            DataType::BinaryView => array
                .as_binary_view()
                .iter()
                .flatten()
                .any(<[u8]>::is_empty),
            _ => false,
        };
        if empty {
            return exec_err!("Replay row key must be non-empty");
        }
        Ok(self.converter.convert_columns(&[array])?)
    }
}
