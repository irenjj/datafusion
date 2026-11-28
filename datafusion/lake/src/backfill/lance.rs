// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: See the NOTICE file distributed with this work for additional information regarding copyright ownership
// SPDX-FileContributor: Licensed to the Apache Software Foundation (ASF) under one or more contributor license agreements

use std::collections::{HashMap, HashSet};
use std::sync::Arc;

use arrow::array::RecordBatch;
use arrow::datatypes::Schema;
use datafusion_common::{Result, exec_err, not_impl_err, plan_datafusion_err, plan_err};
use futures::{Stream, StreamExt, TryStreamExt, pin_mut, stream};
use lance::dataset::transaction::{Operation, Transaction};
use lance::table::format::{Fragment, Manifest, ManifestBuildConfig};
use lance_core::datatypes::Schema as LanceSchema;
use lance_io::object_store::{ObjectStore, ObjectStoreParams};
use lance_table_runtime::commit::{
    CommitHooks, CommitTransactionOptions, commit_transaction,
};
use lance_table_runtime::write::open_update_writer;
use lance_table_runtime::{FileFragment, Table, TableBuilder};
use object_store::ObjectStoreExt;
use sha2::{Digest, Sha256};
use url::Url;

use crate::lance_table::lance_scan_error;
use crate::media::external;

use super::plan::{
    BackfillPlan, BackfillReceipt, BackfillWorkUnit, PlanPayload, add_rows, dataset_url,
    validate_columns,
};

const METADATA_CONCURRENCY: usize = 16;

pub(super) async fn prepare(
    uri: &str,
    columns: &Schema,
    runtime: &ObjectStoreParams,
) -> Result<(BackfillPlan, Vec<BackfillWorkUnit>)> {
    let uri = dataset_url(uri)?;
    let table = open(&uri, None, runtime).await?;
    column_schema(&table, columns)?;
    let units = work_units(&table).await?;
    let row_count = units
        .iter()
        .try_fold(0, |total, unit| add_rows(total, unit.row_count))?;
    let plan = BackfillPlan::new(PlanPayload {
        version: 1,
        dataset_uri: uri.into(),
        dataset_version: table.version_id(),
        manifest_identity: manifest_identity(table.manifest()),
        new_columns: columns.clone(),
        unit_count: units.len(),
        row_count,
    })?;
    Ok((plan, units))
}

pub(super) async fn write_unit(
    plan: &BackfillPlan,
    unit: &BackfillWorkUnit,
    attempt_id: &str,
    batches: impl Stream<Item = Result<RecordBatch>> + Send,
    runtime: &ObjectStoreParams,
) -> Result<BackfillReceipt> {
    let file_name = artifact_name(plan, unit, attempt_id)?;
    let uri = dataset_url(plan.dataset_uri())?;
    let artifact_uri = artifact_url(uri.clone(), &file_name)?;
    let table = open_baseline(plan, &uri, runtime).await?;
    let fragment = table
        .get_fragment(unit.fragment_id as usize)
        .ok_or_else(|| {
            plan_datafusion_err!("Missing Lance backfill fragment {}", unit.fragment_id)
        })?;
    if work_unit(fragment).await? != *unit {
        return plan_err!("Lance backfill work unit does not match the pinned fragment");
    }
    let schema = column_schema(&table, plan.new_columns())?;
    let store = table.object_store(None).await.map_err(lance_scan_error)?;
    let data_dir = table.data_dir();
    let version = table.manifest().data_storage_format.lance_file_format();
    let mut writer = open_update_writer(version, &table, &schema, false)
        .await
        .map_err(lance_scan_error)?;
    let staged = data_dir.clone().join(writer.data_file_path().0);
    let result = async {
        pin_mut!(batches);
        let mut rows = 0;
        while let Some(batch) = batches.try_next().await? {
            if batch.schema_ref().as_ref() != plan.new_columns() {
                return exec_err!(
                    "Backfill batch schema differs from the planned new columns"
                );
            }
            rows = add_rows(rows, batch.num_rows() as u64)?;
            if rows > unit.row_count {
                return exec_err!(
                    "Backfill row count exceeds {} physical rows",
                    unit.row_count
                );
            }
            writer.write(&[batch]).await.map_err(lance_scan_error)?;
        }
        if rows != unit.row_count {
            return exec_err!(
                "Expected {} backfill rows, received {rows}",
                unit.row_count
            );
        }
        let (written, mut file) = writer.finish().await.map_err(lance_scan_error)?;
        if u64::from(written) != rows {
            return exec_err!("Lance wrote {written} backfill rows, expected {rows}");
        }
        // Use create-only publication even when the same attempt is submitted twice.
        store
            .copy_if_not_exists(&staged, &data_dir.join(file_name.as_str()))
            .await
            .map_err(external)?;
        file.path = file_name;
        Ok(BackfillReceipt {
            plan_fingerprint: plan.plan_fingerprint.clone(),
            work_unit: *unit,
            attempt_id: attempt_id.to_owned(),
            artifact_uri: artifact_uri.into(),
            row_count: rows,
            data_file: file,
        })
    }
    .await;
    drop(writer);
    let _ = store.delete(&staged).await;
    result
}

pub(super) async fn finalize(
    plan: &BackfillPlan,
    winners: &HashMap<u32, &BackfillReceipt>,
    runtime: &ObjectStoreParams,
) -> Result<u64> {
    let uri = dataset_url(plan.dataset_uri())?;
    let table = open_baseline(plan, &uri, runtime).await?;
    let latest = table.latest_version_id().await.map_err(lance_scan_error)?;
    if latest != plan.dataset_version() {
        return plan_err!(
            "Lance backfill baseline {} is stale. Latest version is {latest}",
            plan.dataset_version()
        );
    }
    let columns = column_schema(&table, plan.new_columns())?;
    let mut manifest = table.manifest().clone();
    manifest
        .schema
        .fields
        .extend(columns.fields.iter().cloned());
    let preview = Arc::new(table.with_manifest_view(Arc::new(manifest)));
    let store = table.object_store(None).await.map_err(lance_scan_error)?;
    let units = work_units(&table).await?;
    if units.len() != winners.len() {
        return plan_err!("Backfill receipts do not cover the pinned Lance fragments");
    }
    let assignments = units
        .iter()
        .map(|unit| {
            let receipt = winners.get(&unit.fragment_id).ok_or_else(|| {
                plan_datafusion_err!(
                    "Missing backfill receipt for Lance fragment {}",
                    unit.fragment_id
                )
            })?;
            if receipt.work_unit != *unit {
                return plan_err!(
                    "Backfill receipt row count does not match the pinned fragment"
                );
            }
            Ok(*receipt)
        })
        .collect::<Result<Vec<_>>>()?;
    let field_ids: HashSet<_> = columns.field_ids().into_iter().collect();
    stream::iter(assignments.iter().copied())
        .map(|receipt| {
            verify_artifact(&uri, &preview, &store, plan, &field_ids, &columns, receipt)
        })
        .buffer_unordered(METADATA_CONCURRENCY)
        .try_for_each(|()| async { Ok(()) })
        .await?;

    let mut fragments = table.manifest().fragments.as_ref().clone();
    for (fragment, receipt) in fragments.iter_mut().zip(assignments) {
        fragment.files.push(receipt.data_file.clone());
    }
    let transaction = Transaction::new(
        plan.dataset_version(),
        Operation::Merge {
            fragments,
            schema: preview.schema().clone(),
            preserves_nullability: columns.fields.iter().all(|field| field.nullable),
        },
        None,
    );
    let build_config = ManifestBuildConfig::default();
    let commit_config = Default::default();
    let options = CommitTransactionOptions::new(
        &build_config,
        &commit_config,
        table.manifest_location().naming_scheme,
    )
    .with_expected_latest_version(plan.dataset_version());
    let (manifest, _) = Box::pin(commit_transaction(
        &table,
        &store,
        table.commit_handler().as_ref(),
        &transaction,
        options,
        CommitHooks::default(),
    ))
    .await
    .map_err(lance_scan_error)?;
    Ok(manifest.version)
}

async fn verify_artifact(
    uri: &Url,
    preview: &Arc<Table>,
    store: &ObjectStore,
    plan: &BackfillPlan,
    field_ids: &HashSet<i32>,
    columns: &LanceSchema,
    receipt: &BackfillReceipt,
) -> Result<()> {
    let unit = receipt.work_unit;
    let file_name = artifact_name(plan, &unit, &receipt.attempt_id)?;
    let file = &receipt.data_file;
    let data_dir = preview.data_dir();
    if file.path != file_name
        || artifact_url(uri.clone(), &file_name)?.as_str() != receipt.artifact_uri
        || file.base_id.is_some()
        || file.file_version().map_err(lance_scan_error)?
            != preview.manifest().data_storage_format.lance_file_format()
        || file.fields.iter().any(|id| !field_ids.contains(id))
        || file.schema(columns) != *columns
    {
        return plan_err!(
            "Invalid backfill data file for Lance fragment {}",
            unit.fragment_id
        );
    }
    file.validate(&data_dir).map_err(lance_scan_error)?;
    let meta = store
        .inner
        .head(&data_dir.join(file_name.as_str()))
        .await
        .map_err(external)?;
    if file.file_size_bytes.get().map(u64::from) != Some(meta.size) {
        return plan_err!("Backfill data file size differs from the worker receipt");
    }
    let mut metadata = Fragment::new(u64::from(unit.fragment_id));
    metadata.files.push(file.clone());
    let fragment = FileFragment::new(Arc::clone(preview), metadata);
    let rows = fragment.physical_rows().await.map_err(lance_scan_error)? as u64;
    if rows != unit.row_count {
        return plan_err!(
            "Backfill data file has {rows} rows, expected {}",
            unit.row_count
        );
    }
    Ok(())
}

async fn open(
    uri: &Url,
    version: Option<u64>,
    runtime: &ObjectStoreParams,
) -> Result<Table> {
    let builder = TableBuilder::from_uri(uri.as_str()).with_store_params(runtime.clone());
    let builder = match version {
        Some(version) => builder.with_version(version),
        None => builder,
    };
    let table = builder.load().await.map_err(lance_scan_error)?;
    if table
        .manifest()
        .data_storage_format
        .lance_file_format()
        .to_manifest_string()
        == "0.1"
    {
        return not_impl_err!(concat!(
            "Lance backfill requires V2 storage. Legacy V1 pairs files by batch ",
            "boundary and cannot preserve all Arrow nulls in a new column file."
        ));
    }
    Ok(table)
}

async fn open_baseline(
    plan: &BackfillPlan,
    uri: &Url,
    runtime: &ObjectStoreParams,
) -> Result<Table> {
    let table = open(uri, Some(plan.dataset_version()), runtime).await?;
    if table.version_id() != plan.dataset_version()
        || manifest_identity(table.manifest()) != plan.payload.manifest_identity
    {
        return plan_err!("Lance manifest identity differs from the backfill baseline");
    }
    Ok(table)
}

fn manifest_identity(manifest: &Manifest) -> String {
    let hash = Sha256::new()
        .chain_update(manifest.timestamp_nanos.to_le_bytes())
        .chain_update(manifest.transaction_file.as_deref().unwrap_or_default());
    hex::encode(hash.finalize())
}

fn column_schema(table: &Table, columns: &Schema) -> Result<LanceSchema> {
    validate_columns(
        table.schema().fields.iter().map(|f| f.name.as_str()),
        columns,
    )?;
    let mut schema = LanceSchema::try_from(columns).map_err(lance_scan_error)?;
    if schema.fields_pre_order().any(|field| {
        field.is_blob()
            || field.metadata.keys().any(|key| {
                key.starts_with("lance-schema:") || key.starts_with("lance-encoding:")
            })
    }) {
        return plan_err!(
            "Backfill columns cannot change Lance encoding or key configuration"
        );
    }
    schema
        .try_reassign_field_ids(Some(table.manifest().max_field_id()))
        .map_err(lance_scan_error)?;
    schema.validate_writable().map_err(lance_scan_error)?;
    Ok(schema)
}

async fn work_units(table: &Table) -> Result<Vec<BackfillWorkUnit>> {
    let units: Vec<_> = stream::iter(table.get_fragments())
        .map(work_unit)
        .buffered(METADATA_CONCURRENCY)
        .try_collect()
        .await?;
    if !units
        .windows(2)
        .all(|pair| pair[0].fragment_id < pair[1].fragment_id)
    {
        return plan_err!("Lance backfill requires unique and ordered fragment IDs");
    }
    Ok(units)
}

async fn work_unit(fragment: FileFragment) -> Result<BackfillWorkUnit> {
    let fragment_id = u32::try_from(fragment.metadata().id).map_err(|_| {
        plan_datafusion_err!("Invalid Lance fragment ID {}", fragment.metadata().id)
    })?;
    let rows = fragment.physical_rows().await.map_err(lance_scan_error)?;
    let row_count = u32::try_from(rows).map_err(|_| {
        plan_datafusion_err!("Lance fragment {fragment_id} exceeds u32 rows")
    })?;
    Ok(BackfillWorkUnit {
        fragment_id,
        row_count: u64::from(row_count),
    })
}

fn artifact_name(
    plan: &BackfillPlan,
    unit: &BackfillWorkUnit,
    attempt_id: &str,
) -> Result<String> {
    if attempt_id.trim().is_empty() {
        return plan_err!("Backfill attempt ID must be nonempty");
    }
    let attempt_hash = hex::encode(Sha256::digest(attempt_id));
    Ok(format!(
        "backfill-{}-{}-{attempt_hash}.lance",
        plan.plan_fingerprint, unit.fragment_id
    ))
}

fn artifact_url(mut uri: Url, file_name: &str) -> Result<Url> {
    uri.path_segments_mut()
        .map_err(|()| plan_datafusion_err!("Invalid Lance backfill artifact base"))?
        .pop_if_empty()
        .extend(["data", file_name]);
    Ok(uri)
}
