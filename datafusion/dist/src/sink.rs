// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: See the NOTICE file distributed with this work for additional information regarding copyright ownership
// SPDX-FileContributor: Licensed to the Apache Software Foundation (ASF) under one or more contributor license agreements

use std::collections::{BTreeMap, BTreeSet};
use std::sync::Arc;

use datafusion::arrow::datatypes::SchemaRef;
use datafusion::arrow::record_batch::RecordBatch;
use datafusion::common::{
    DataFusionError, Result, exec_datafusion_err, exec_err, plan_err,
};
use datafusion::object_store::path::Path;
use datafusion::object_store::{
    Error as StoreError, ObjectStore, ObjectStoreExt, PutMode, PutPayload,
};
use datafusion::parquet::arrow::ArrowWriter;
use datafusion::parquet::file::metadata::KeyValue;
use datafusion::physical_plan::ExecutionPlan;
#[cfg(feature = "ai")]
use datafusion_ai::{CheckpointableTarget, llm::LlmExec, validate_checkpoint_target};
use futures::TryStreamExt;
use serde::{Deserialize, Serialize, de::DeserializeOwned};
use sha2::{Digest, Sha256};

use crate::fingerprint::{plan_fingerprint, split_fingerprint};
use crate::{ProviderUsage, Split, SplitMetrics, StagePlan};

mod routing;
#[cfg(test)]
mod tests;

use routing::Routing;

pub(crate) const CHECKPOINT_SCHEMA: &str = "df-checkpoint-v1";
pub(crate) const MANIFEST_SCHEMA: &str = "df-stage-manifest-v1";
pub(crate) const MANIFEST_FILE: &str = "STAGE_MANIFEST.json";
const DATA_FILE: &str = "data.parquet";
const DLQ_FILE: &str = "dlq.parquet";

/// Options for routing a stage's output to Parquet.
#[derive(Clone, Debug, Default, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct SinkOptions {
    /// Output column used as the non-null row key when routing AI envelopes.
    pub row_key: Option<String>,
}

impl SinkOptions {
    /// Validate the root's output schema and the declarations of a built-in checkpoint target.
    pub fn validate_plan(&self, plan: &dyn ExecutionPlan) -> Result<()> {
        let schema = plan.schema();
        #[cfg(feature = "ai")]
        if let Some(target) = plan.downcast_ref::<LlmExec>() {
            validate_checkpoint_target(target, &schema)?;
        }
        Routing::try_new(schema, self).map(|_| ())
    }
}

/// JSON receipt for the immutable Parquet files written by one split attempt.
///
/// Pass the receipt to [`ParquetSplitSink::commit_split`] to publish it.
#[derive(Clone, Debug, Deserialize, PartialEq, Eq, Serialize)]
pub struct AttemptArtifact {
    /// Split that produced the files.
    pub split_id: u64,
    /// Attempt that produced the files.
    pub attempt_id: u32,
    plan_fingerprint: String,
    split_fingerprint: String,
    #[serde(flatten)]
    files: ArtifactFiles,
}

#[derive(Clone, Debug, Deserialize, PartialEq, Eq, Serialize)]
pub(crate) struct ArtifactFiles {
    pub(crate) data_rel_path: String,
    data_sha256: String,
    pub(crate) data_bytes: u64,
    pub(crate) output_rows: u64,
    pub(crate) dlq_rel_path: Option<String>,
    dlq_sha256: Option<String>,
    pub(crate) dlq_rows: u64,
    dlq_by_code: BTreeMap<String, u64>,
}

#[derive(Clone, Debug, Deserialize, PartialEq, Eq, Serialize)]
pub(crate) struct SplitCheckpoint {
    pub(crate) schema: String,
    pub(crate) plan_fingerprint: String,
    pub(crate) split_fingerprint: String,
    pub(crate) split_id: u64,
    pub(crate) winning_attempt_id: u32,
    #[serde(flatten)]
    pub(crate) files: ArtifactFiles,
    #[serde(default)]
    metrics: SplitMetrics,
}

/// JSON inventory of winning Parquet files and totals for a completed stage.
///
/// The serialized split entries appear in ascending split ID order. Their paths
/// are relative to the stage directory.
#[derive(Clone, Debug, Deserialize, PartialEq, Eq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct StageManifest {
    pub(crate) schema: String,
    pub(crate) plan_fingerprint: String,
    pub(crate) splits: Vec<SplitCheckpoint>,
    /// Total successful rows across the winning attempts.
    pub output_rows: u64,
    /// Total failed cells written to the dead-letter queue (one row per failure).
    pub dlq_rows: u64,
    /// Dead-letter queue row counts grouped by error code.
    pub dlq_by_code: BTreeMap<String, u64>,
    /// Input tokens reported by providers across the winning attempts.
    #[serde(default)]
    pub input_tokens: usize,
    /// Output tokens reported by providers across the winning attempts.
    #[serde(default)]
    pub output_tokens: usize,
    /// Inference request attempts across the winning attempts, including retries.
    #[serde(default)]
    pub llm_requests: usize,
    /// Inference retries across the winning attempts.
    #[serde(default)]
    pub llm_retries: usize,
    /// Inference counters by provider across the winning attempts.
    #[serde(default)]
    pub llm_usage_by_provider: BTreeMap<String, ProviderUsage>,
}

/// Writes immutable Parquet attempt files and publishes one checkpoint per split.
///
/// Use an [`ObjectStore`] that supports atomic create. Readers use the paths in
/// the checkpoint or stage manifest to locate the winning files.
///
/// ```no_run
/// # async fn example(stage: &datafusion_dist::StagePlan, splits: &[datafusion_dist::Split])
/// # -> datafusion::common::Result<()> {
/// use std::sync::Arc;
/// use datafusion::object_store::{local::LocalFileSystem, path::Path};
/// use datafusion_dist::{ParquetSplitSink, SinkOptions};
///
/// let sink = ParquetSplitSink::try_new(
///     Arc::new(LocalFileSystem::new()),
///     &Path::from("tmp/stage-output"),
///     stage,
///     &SinkOptions::default(),
/// )?;
/// let pending = sink.resume_filter(splits).await?;
/// # Ok(())
/// # }
/// ```
#[derive(Debug)]
pub struct ParquetSplitSink {
    store: Arc<dyn ObjectStore>,
    stage_root: Path,
    plan_fingerprint: String,
    input_schema: SchemaRef,
    routing: Routing,
}

impl ParquetSplitSink {
    /// Bind a stage and its routing options to a store-relative output root.
    pub fn try_new(
        store: Arc<dyn ObjectStore>,
        output_root: &Path,
        stage: &StagePlan,
        options: &SinkOptions,
    ) -> Result<Self> {
        if plan_fingerprint(&stage.plan_bytes, &stage.output_schema)?
            != stage.plan_fingerprint
        {
            return plan_err!("Plan fingerprint does not match stage {}", stage.stage_id);
        }
        Ok(Self {
            store,
            stage_root: output_root
                .clone()
                .join("stages")
                .join(stage.plan_fingerprint.as_str()),
            plan_fingerprint: stage.plan_fingerprint.clone(),
            input_schema: Arc::clone(&stage.output_schema),
            routing: Routing::try_new(Arc::clone(&stage.output_schema), options)?,
        })
    }

    /// Validate a caller-provided [`CheckpointableTarget`] for the stage's root.
    #[cfg(feature = "ai")]
    pub fn validate_target(&self, target: &dyn CheckpointableTarget) -> Result<()> {
        validate_checkpoint_target(target, &self.input_schema)
    }

    /// Write attempt-scoped Parquet files and return a receipt for publication.
    ///
    /// Rewriting an attempt is allowed only when its existing bytes match.
    pub async fn write_attempt(
        &self,
        split: &Split,
        batches: &[RecordBatch],
    ) -> Result<AttemptArtifact> {
        self.validate_split(split)?;
        if batches
            .iter()
            .any(|batch| batch.schema_ref() != &self.input_schema)
        {
            return plan_err!("Attempt output schema differs from its stage");
        }
        let routed = self.routing.route(batches)?;
        let dlq = routed
            .dlq
            .as_ref()
            .map(|batch| {
                encode_parquet(&batch.schema(), std::slice::from_ref(batch), None)
            })
            .transpose()?;
        let dlq_sha256 = dlq.as_ref().map(|bytes| checksum(bytes));
        // Bind the two files before a writer can claim this attempt directory.
        let data = encode_parquet(
            self.routing.data_schema(),
            &routed.data,
            dlq_sha256.as_deref(),
        )?;
        let data_rel_path = attempt_path(split.split_id, split.attempt_id, DATA_FILE);
        let data_sha256 = checksum(&data);
        let data_bytes = data.len() as u64;
        self.write_immutable(&self.stage_path(&data_rel_path), data, &data_sha256)
            .await?;
        let dlq_rel_path = if let (Some(bytes), Some(sha256)) = (dlq, &dlq_sha256) {
            let path = attempt_path(split.split_id, split.attempt_id, DLQ_FILE);
            self.write_immutable(&self.stage_path(&path), bytes, sha256)
                .await?;
            Some(path)
        } else {
            None
        };
        Ok(AttemptArtifact {
            split_id: split.split_id,
            attempt_id: split.attempt_id,
            plan_fingerprint: self.plan_fingerprint.clone(),
            split_fingerprint: split.fingerprint.clone(),
            files: ArtifactFiles {
                data_rel_path,
                data_sha256,
                data_bytes,
                output_rows: count_rows(&routed.data),
                dlq_rel_path,
                dlq_sha256,
                dlq_rows: routed
                    .dlq
                    .as_ref()
                    .map_or(0, |batch| batch.num_rows() as u64),
                dlq_by_code: routed.dlq_by_code,
            },
        })
    }

    /// Publish a verified attempt. The first durable writer wins.
    ///
    /// If another attempt has already published, verify its checkpoint and
    /// files and accept it as the winner, even when its output differs.
    ///
    /// A caller using leases must check the attempt's eligibility before
    /// publication and commit the queue entry only after this method succeeds.
    pub async fn commit_split(
        &self,
        split: &Split,
        attempt: AttemptArtifact,
    ) -> Result<()> {
        self.commit_split_with_metrics(split, attempt, SplitMetrics::default())
            .await
    }

    /// Publish a verified attempt and its execution counters in one checkpoint.
    ///
    /// The same first-writer and lease rules as [`Self::commit_split`] apply.
    pub async fn commit_split_with_metrics(
        &self,
        split: &Split,
        attempt: AttemptArtifact,
        metrics: SplitMetrics,
    ) -> Result<()> {
        self.validate_split(split)?;
        if attempt.attempt_id != split.attempt_id {
            return exec_err!("Receipt attempt does not match split {}", split.split_id);
        }
        let candidate = SplitCheckpoint {
            schema: CHECKPOINT_SCHEMA.into(),
            plan_fingerprint: attempt.plan_fingerprint,
            split_fingerprint: attempt.split_fingerprint,
            split_id: attempt.split_id,
            winning_attempt_id: attempt.attempt_id,
            files: attempt.files,
            metrics,
        };
        self.validate_checkpoint(split, &candidate)?;
        self.verify_contents(&candidate.files).await?;
        let path = self.checkpoint_path(split);
        if self.create(&path, json_bytes(&candidate)?).await? {
            return Ok(());
        }

        let winner = self.read_checkpoint(split).await?.ok_or_else(|| {
            exec_datafusion_err!(
                "Split {} checkpoint disappeared after publication",
                split.split_id
            )
        })?;
        self.verify_contents(&winner.files).await
    }

    /// Return uncommitted splits after validating existing markers and file metadata.
    ///
    /// Pending splits start after all previously staged attempt directories.
    pub async fn resume_filter(&self, splits: &[Split]) -> Result<Vec<Split>> {
        validate_inventory(splits)?;
        if let Some(manifest) = self
            .read_json::<StageManifest>(&self.stage_root.clone().join(MANIFEST_FILE))
            .await?
        {
            if manifest != self.collect_manifest(splits).await? {
                return exec_err!(
                    "Conflicting manifest for stage {}",
                    self.plan_fingerprint
                );
            }
            return Ok(vec![]);
        }

        let mut pending = Vec::new();
        for split in splits {
            if self.read_checkpoint(split).await?.is_none() {
                pending.push(split.clone());
            }
        }
        if !pending.is_empty() {
            self.advance_attempt_ids(&mut pending).await?;
        }
        Ok(pending)
    }

    /// Atomically publish the manifest once every supplied split is committed.
    pub async fn finalize_stage(&self, splits: &[Split]) -> Result<StageManifest> {
        validate_inventory(splits)?;
        let manifest = self.collect_manifest(splits).await?;
        let path = self.stage_root.clone().join(MANIFEST_FILE);
        if !self.create(&path, json_bytes(&manifest)?).await?
            && self.read_json::<StageManifest>(&path).await?.as_ref() != Some(&manifest)
        {
            return exec_err!("Conflicting manifest for stage {}", self.plan_fingerprint);
        }
        Ok(manifest)
    }

    async fn collect_manifest(&self, splits: &[Split]) -> Result<StageManifest> {
        let mut manifest = StageManifest {
            schema: MANIFEST_SCHEMA.into(),
            plan_fingerprint: self.plan_fingerprint.clone(),
            splits: Vec::with_capacity(splits.len()),
            output_rows: 0,
            dlq_rows: 0,
            dlq_by_code: BTreeMap::new(),
            input_tokens: 0,
            output_tokens: 0,
            llm_requests: 0,
            llm_retries: 0,
            llm_usage_by_provider: BTreeMap::new(),
        };
        let mut sorted: Vec<_> = splits.iter().collect();
        sorted.sort_unstable_by_key(|split| split.split_id);
        for split in sorted {
            let checkpoint = self.read_checkpoint(split).await?.ok_or_else(|| {
                exec_datafusion_err!("Split {} is not committed", split.split_id)
            })?;
            add_rows(&mut manifest.output_rows, checkpoint.files.output_rows)?;
            add_rows(&mut manifest.dlq_rows, checkpoint.files.dlq_rows)?;
            for (code, rows) in &checkpoint.files.dlq_by_code {
                add_rows(manifest.dlq_by_code.entry(code.clone()).or_default(), *rows)?;
            }
            for (name, total, value) in [
                (
                    "input_tokens",
                    &mut manifest.input_tokens,
                    checkpoint.metrics.input_tokens,
                ),
                (
                    "output_tokens",
                    &mut manifest.output_tokens,
                    checkpoint.metrics.output_tokens,
                ),
                (
                    "llm_requests",
                    &mut manifest.llm_requests,
                    checkpoint.metrics.llm_requests,
                ),
                (
                    "llm_retries",
                    &mut manifest.llm_retries,
                    checkpoint.metrics.llm_retries,
                ),
            ] {
                *total = total
                    .checked_add(value)
                    .ok_or_else(|| exec_datafusion_err!("Stage {name} count overflow"))?;
            }
            for (provider, usage) in &checkpoint.metrics.llm_usage_by_provider {
                let total = manifest
                    .llm_usage_by_provider
                    .entry(provider.clone())
                    .or_default();
                for (name, total, value) in [
                    ("input_tokens", &mut total.input_tokens, usage.input_tokens),
                    (
                        "output_tokens",
                        &mut total.output_tokens,
                        usage.output_tokens,
                    ),
                    ("requests", &mut total.requests, usage.requests),
                ] {
                    *total = total.checked_add(value).ok_or_else(|| {
                        exec_datafusion_err!(
                            "Stage provider '{provider}' {name} count overflow"
                        )
                    })?;
                }
            }
            manifest.splits.push(checkpoint);
        }
        Ok(manifest)
    }

    async fn advance_attempt_ids(&self, pending: &mut [Split]) -> Result<()> {
        let prefix = self.stage_root.clone().join("attempts");
        let attempts = self.store.list_with_delimiter(Some(&prefix)).await?;
        let mut by_id: BTreeMap<_, _> = pending
            .iter_mut()
            .map(|split| (split.split_id, split))
            .collect();
        for path in attempts.common_prefixes {
            if let Some((split_id, attempt_id)) =
                path.filename().and_then(parse_attempt_dir)
                && let Some(split) = by_id.get_mut(&split_id)
            {
                let next_id = attempt_id.checked_add(1).ok_or_else(|| {
                    exec_datafusion_err!("Attempt IDs exhausted for split {split_id}")
                })?;
                split.attempt_id = split.attempt_id.max(next_id);
            }
        }
        Ok(())
    }

    fn validate_split(&self, split: &Split) -> Result<()> {
        if split.fingerprint
            != split_fingerprint(&self.plan_fingerprint, &split.file_proto)
        {
            return plan_err!(
                "Split {} fingerprint does not match its stage",
                split.split_id
            );
        }
        Ok(())
    }

    fn validate_checkpoint(
        &self,
        split: &Split,
        checkpoint: &SplitCheckpoint,
    ) -> Result<()> {
        if checkpoint.schema != CHECKPOINT_SCHEMA
            || checkpoint.plan_fingerprint != self.plan_fingerprint
            || checkpoint.split_fingerprint != split.fingerprint
            || checkpoint.split_id != split.split_id
        {
            return exec_err!(
                "Checkpoint identity does not match split {}",
                split.split_id
            );
        }
        checkpoint
            .files
            .validate(split.split_id, checkpoint.winning_attempt_id)
    }

    async fn read_checkpoint(&self, split: &Split) -> Result<Option<SplitCheckpoint>> {
        self.validate_split(split)?;
        let Some(checkpoint) = self
            .read_json::<SplitCheckpoint>(&self.checkpoint_path(split))
            .await?
        else {
            return Ok(None);
        };
        self.validate_checkpoint(split, &checkpoint)?;
        let files = &checkpoint.files;
        let data = self
            .store
            .head(&self.stage_path(&files.data_rel_path))
            .await?;
        if data.size != files.data_bytes {
            return exec_err!(
                "Checkpoint data size mismatch for split {}",
                split.split_id
            );
        }
        if let Some(path) = &files.dlq_rel_path {
            self.store.head(&self.stage_path(path)).await?;
        }
        Ok(Some(checkpoint))
    }

    async fn verify_contents(&self, files: &ArtifactFiles) -> Result<()> {
        let path = self.stage_path(&files.data_rel_path);
        if self.store.head(&path).await?.size != files.data_bytes {
            return exec_err!("Checkpoint data size mismatch at {path}");
        }
        self.verify_sha256(&path, &files.data_sha256).await?;
        if let (Some(path), Some(sha256)) = (&files.dlq_rel_path, &files.dlq_sha256) {
            self.verify_sha256(&self.stage_path(path), sha256).await?;
        }
        Ok(())
    }

    async fn verify_sha256(&self, path: &Path, expected: &str) -> Result<()> {
        let mut content = self.store.get(path).await?.into_stream();
        let mut sha256 = Sha256::new();
        while let Some(bytes) = content.try_next().await? {
            sha256.update(bytes);
        }
        if hex::encode(sha256.finalize()) != expected {
            return exec_err!("Artifact checksum does not match at {path}");
        }
        Ok(())
    }

    async fn write_immutable(
        &self,
        path: &Path,
        bytes: Vec<u8>,
        sha256: &str,
    ) -> Result<()> {
        if !self.create(path, bytes).await? {
            self.verify_sha256(path, sha256).await?;
        }
        Ok(())
    }

    async fn create(&self, path: &Path, bytes: impl Into<PutPayload>) -> Result<bool> {
        match self
            .store
            .put_opts(path, bytes.into(), PutMode::Create.into())
            .await
        {
            Ok(_) => Ok(true),
            Err(StoreError::AlreadyExists { .. }) => Ok(false),
            Err(error) => Err(error.into()),
        }
    }

    async fn read_json<T: DeserializeOwned>(&self, path: &Path) -> Result<Option<T>> {
        match self.store.get(path).await {
            Ok(result) => serde_json::from_slice(&result.bytes().await?)
                .map(Some)
                .map_err(|error| exec_datafusion_err!("Invalid JSON at {path}: {error}")),
            Err(StoreError::NotFound { .. }) => Ok(None),
            Err(error) => Err(error.into()),
        }
    }

    fn checkpoint_path(&self, split: &Split) -> Path {
        self.stage_root
            .clone()
            .join("splits")
            .join(split.fingerprint.as_str())
            .join("COMMITTED.json")
    }

    fn stage_path(&self, relative_path: &str) -> Path {
        stage_path(&self.stage_root, relative_path)
    }
}

impl ArtifactFiles {
    pub(crate) fn validate(&self, split_id: u64, attempt_id: u32) -> Result<()> {
        let expected_data = attempt_path(split_id, attempt_id, DATA_FILE);
        let expected_dlq = attempt_path(split_id, attempt_id, DLQ_FILE);
        let valid_dlq = match (&self.dlq_rel_path, &self.dlq_sha256) {
            (None, None) => self.dlq_rows == 0,
            (Some(path), Some(sha256)) => {
                self.dlq_rows > 0 && *path == expected_dlq && is_sha256(sha256)
            }
            _ => false,
        };
        let dlq_total = self.dlq_by_code.values().try_fold(0_u64, |total, rows| {
            if *rows == 0 {
                None
            } else {
                total.checked_add(*rows)
            }
        });
        if self.data_rel_path != expected_data
            || !is_sha256(&self.data_sha256)
            || self.data_bytes == 0
            || !valid_dlq
            || dlq_total != Some(self.dlq_rows)
        {
            return exec_err!(
                "Invalid checkpoint files or statistics for split {split_id}"
            );
        }
        Ok(())
    }
}

pub(crate) fn stage_path(stage_root: &Path, relative_path: &str) -> Path {
    relative_path
        .split('/')
        .fold(stage_root.clone(), |path, part| path.join(part))
}

fn validate_inventory(splits: &[Split]) -> Result<()> {
    let mut ids = BTreeSet::new();
    let mut fingerprints = BTreeSet::new();
    for split in splits {
        if !ids.insert(split.split_id) || !fingerprints.insert(&split.fingerprint) {
            return plan_err!(
                "Duplicate split ID or fingerprint in the checkpoint inventory"
            );
        }
    }
    Ok(())
}

fn attempt_dir(split_id: u64, attempt_id: u32) -> String {
    format!("split-{split_id:016x}-attempt-{attempt_id:04x}")
}

fn attempt_path(split_id: u64, attempt_id: u32, filename: &str) -> String {
    format!("attempts/{}/{filename}", attempt_dir(split_id, attempt_id))
}

fn parse_attempt_dir(name: &str) -> Option<(u64, u32)> {
    let (split, attempt) = name.strip_prefix("split-")?.split_once("-attempt-")?;
    let split_id = u64::from_str_radix(split, 16).ok()?;
    let attempt_id = u32::from_str_radix(attempt, 16).ok()?;
    (name == attempt_dir(split_id, attempt_id)).then_some((split_id, attempt_id))
}

fn add_rows(total: &mut u64, rows: u64) -> Result<()> {
    *total = total
        .checked_add(rows)
        .ok_or_else(|| exec_datafusion_err!("Checkpoint row count overflow"))?;
    Ok(())
}

fn count_rows(batches: &[RecordBatch]) -> u64 {
    batches.iter().map(|batch| batch.num_rows() as u64).sum()
}

fn checksum(bytes: &[u8]) -> String {
    hex::encode(Sha256::digest(bytes))
}

pub(crate) fn is_sha256(text: &str) -> bool {
    text.len() == 64
        && text
            .bytes()
            .all(|byte| byte.is_ascii_digit() || matches!(byte, b'a'..=b'f'))
}

fn json_bytes(value: &impl Serialize) -> Result<Vec<u8>> {
    serde_json::to_vec(value).map_err(json_error)
}

fn json_error(error: serde_json::Error) -> DataFusionError {
    DataFusionError::External(Box::new(error))
}

fn encode_parquet(
    schema: &SchemaRef,
    batches: &[RecordBatch],
    dlq_sha256: Option<&str>,
) -> Result<Vec<u8>> {
    let mut writer = ArrowWriter::try_new(Vec::new(), Arc::clone(schema), None)?;
    if let Some(sha256) = dlq_sha256 {
        writer.append_key_value_metadata(KeyValue::new(
            "df.dist.dlq_sha256".into(),
            sha256.to_owned(),
        ));
    }
    for batch in batches {
        writer.write(batch)?;
    }
    Ok(writer.into_inner()?)
}
