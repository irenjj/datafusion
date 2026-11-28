// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: See the NOTICE file distributed with this work for additional information regarding copyright ownership
// SPDX-FileContributor: Licensed to the Apache Software Foundation (ASF) under one or more contributor license agreements

use std::collections::{BTreeMap, HashMap};
use std::path::Path as FsPath;
use std::sync::Arc;

use datafusion::arrow::array::{AsArray, RecordBatchReader, UInt64Array};
use datafusion::arrow::compute::{concat_batches, take_record_batch};
use datafusion::arrow::datatypes::UInt64Type;
use datafusion::arrow::record_batch::RecordBatch;
use datafusion::arrow::row::{RowConverter, SortField};
use datafusion::common::{
    DataFusionError, Result, exec_err, not_impl_err, plan_datafusion_err, plan_err,
};
use datafusion::execution::object_store::ObjectStoreUrl;
use datafusion::parquet::arrow::arrow_reader::{
    ParquetRecordBatchReader, ParquetRecordBatchReaderBuilder,
};
use datafusion::prelude::SessionContext;
use datafusion_lake::LakeSplitPayload;
use datafusion_lake::backfill::{
    BackfillTarget, BackfillWorkUnit, finalize_backfill, prepare_backfill,
    write_backfill_unit,
};
use datafusion_lake::dist::{LAKE_SCAN_PROTOCOL_NAME, LAKE_SCAN_PROTOCOL_VERSION};
use datafusion_proto::protobuf::{
    PhysicalPlanNode, Schema as WireSchema, physical_expr_node::ExprType,
    physical_plan_node::PhysicalPlanType,
};
use futures::{TryStreamExt, stream};
use lance::Dataset;
use lance::dataset::{builder::DatasetBuilder, scanner::Scanner};
use lance_core::{ROW_ADDR, datatypes::BlobHandling, utils::address::RowAddress};
use lance_io::object_store::ObjectStoreParams;
use object_store::{ObjectStore, ObjectStoreExt, path::Path};
use prost::Message;
use sha2::{Digest, Sha256};
use url::Url;
use uuid::Uuid;

use super::LakeScanPayload;
use crate::StagePlan;
use crate::codec::VersionedExtensionNode;
use crate::fingerprint::split_fingerprint;
use crate::manifest_table::{PublishedStage, check_local_file, load_manifest};
use crate::plan::decode_lake_split;
use crate::sink::{BackfillOrder, SplitCheckpoint, stage_path};

pub(crate) fn backfill_order(
    stage: &StagePlan,
    row_key: Option<&str>,
) -> Option<BackfillOrder> {
    let plan = PhysicalPlanNode::decode(stage.plan_bytes.as_slice()).ok()?;
    let mut key = row_key.and_then(|name| {
        let mut fields = stage
            .output_schema
            .fields()
            .iter()
            .enumerate()
            .filter(|(_, field)| field.name() == name);
        let (index, _) = fields.next()?;
        fields.next().is_none().then_some(index)
    });
    let mut preserves_order = true;
    let mut node = &plan;
    loop {
        node = match node.physical_plan_type.as_ref()? {
            PhysicalPlanType::Projection(projection) => {
                key = key.and_then(|index| {
                    match projection.expr.get(index)?.expr_type.as_ref()? {
                        ExprType::Column(column) => Some(column.index as usize),
                        _ => None,
                    }
                });
                projection.input.as_deref()?
            }
            PhysicalPlanType::Filter(filter) => {
                if !filter.projection.is_empty() {
                    key = key.and_then(|index| {
                        filter.projection.get(index).map(|&i| i as usize)
                    });
                }
                filter.input.as_deref()?
            }
            PhysicalPlanType::CoalesceBatches(coalesce) => coalesce.input.as_deref()?,
            PhysicalPlanType::Extension(extension) if extension.plan_name.is_none() => {
                let named =
                    VersionedExtensionNode::decode(extension.node.as_slice()).ok()?;
                match (named.extension_name.as_str(), extension.inputs.as_slice()) {
                    (LAKE_SCAN_PROTOCOL_NAME, [])
                        if named.schema_version == LAKE_SCAN_PROTOCOL_VERSION =>
                    {
                        let payload =
                            LakeScanPayload::decode(named.payload.as_slice()).ok()?;
                        if let Some(output) = row_key {
                            let schema =
                                WireSchema::decode(payload.output_schema.as_deref()?)
                                    .ok()?;
                            return Some(
                                match key.and_then(|index| schema.columns.get(index)) {
                                    Some(field) => BackfillOrder::Key {
                                        source: field.name.clone(),
                                        output: output.into(),
                                    },
                                    None => BackfillOrder::RequiresKey,
                                },
                            );
                        }
                        return Some(
                            if preserves_order && unfiltered_lance(&payload.scan) {
                                BackfillOrder::Position
                            } else {
                                BackfillOrder::RequiresKey
                            },
                        );
                    }
                    ("df.llm.exec", [child]) if matches!(named.schema_version, 1 | 2) => {
                        preserves_order = false;
                        child
                    }
                    _ => return None,
                }
            }
            _ => return None,
        };
    }
}

fn unfiltered_lance(scan: &[u8]) -> bool {
    // The Lake v1 route exposes a scan payload, but no accessor for its pushed filter.
    let header = b"DFLKS1\0\x01\0\0\0";
    scan.strip_prefix(header)
        .and_then(|json| serde_json::from_slice::<serde_json::Value>(json).ok())
        .is_some_and(|scan| {
            scan["engine"] == "lance" && scan.get("filter").is_some_and(|f| f.is_null())
        })
}

/// Add the winning result columns of a complete Lance stage to its source dataset.
///
/// The source snapshot must still be current. Each fragment must supply every
/// physical row and the published stage must contain no DLQ rows. For plans
/// that may reorder rows, pass an unchanged, non-null unique source column
/// through the stage as its row key. The row key is consumed for alignment.
/// Returns the published Lance dataset version.
///
/// # Public Only for Internal Use:
///
/// This is not a public API and is for internal use only; see [API policy] for details.
/// [API policy]: https://datafusion.apache.org/contributor-guide/api-health.html#internal-public-apis
#[doc(hidden)]
pub async fn backfill_lance(
    ctx: &SessionContext,
    stage_store: ObjectStoreUrl,
    output_root: &Path,
    dataset: &str,
    runtime: &ObjectStoreParams,
) -> Result<u64> {
    let store = ctx.runtime_env().object_store(&stage_store)?;
    let stage = load_manifest(store.as_ref(), output_root).await?;
    let (version, order, winners) = lance_winners(&stage, dataset)?;
    let first = *winners
        .values()
        .next()
        .expect("validated nonempty manifest");
    let first_reader = winner_reader(store.as_ref(), &stage, first).await?;
    let input_schema = first_reader.schema();
    let mut first_reader = Some(first_reader);
    let key_index = match &order {
        BackfillOrder::Key { output, .. } => Some(input_schema.index_of(output)?),
        _ => None,
    };
    let columns: Vec<_> = (0..input_schema.fields().len())
        .filter(|index| Some(*index) != key_index)
        .collect();
    let new_schema = input_schema.project(&columns)?;
    let (plan, units) =
        prepare_backfill(BackfillTarget::Lance(dataset), &new_schema, runtime).await?;
    if plan.dataset_version() != version {
        return plan_err!("Lance backfill source snapshot is stale");
    }
    if units.len() != winners.len()
        || units.iter().any(|unit| {
            winners
                .get(&unit.fragment_id)
                .is_none_or(|split| split.files.output_rows != unit.row_count)
        })
    {
        return plan_err!("Lance backfill requires all physical rows of every fragment");
    }
    let keyed_dataset = if key_index.is_some() {
        Some(Arc::new(
            DatasetBuilder::from_uri(plan.dataset_uri())
                .with_version(version)
                .with_store_params(runtime.clone())
                .load()
                .await
                .map_err(lance_error)?,
        ))
    } else {
        None
    };
    let attempt = Uuid::new_v4().to_string();
    let mut receipts = Vec::with_capacity(units.len());
    for unit in units {
        let split = winners[&unit.fragment_id];
        let reader = if split.split_id == first.split_id {
            first_reader.take().unwrap()
        } else {
            winner_reader(store.as_ref(), &stage, split).await?
        };
        if reader.schema() != input_schema {
            return exec_err!("Lance backfill winner schemas differ across splits");
        }
        let batches = match (&order, keyed_dataset.as_ref(), key_index) {
            (BackfillOrder::Key { source, .. }, Some(dataset), Some(index)) => {
                let rows = reader.collect::<std::result::Result<Vec<_>, _>>()?;
                let rows = concat_batches(&input_schema, &rows)?;
                vec![align_by_key(dataset, unit, source, index, rows).await?]
            }
            _ => reader.collect::<std::result::Result<Vec<_>, _>>()?,
        };
        let batches = stream::iter(
            batches
                .into_iter()
                .map(|batch| Ok(batch.project(&columns)?)),
        );
        receipts
            .push(write_backfill_unit(&plan, &unit, &attempt, batches, runtime).await?);
    }
    finalize_backfill(&plan, &receipts, runtime).await
}

fn lance_winners<'a>(
    stage: &'a PublishedStage,
    target: &str,
) -> Result<(u64, BackfillOrder, BTreeMap<u32, &'a SplitCheckpoint>)> {
    let target = dataset_url(target)?;
    let mut baseline = None;
    let mut winners = BTreeMap::new();
    for split in &stage.manifest.splits {
        let source = split.lake_source.as_ref().ok_or_else(|| {
            plan_datafusion_err!(
                "Lance backfill requires Lake split provenance in the stage manifest"
            )
        })?;
        if split_fingerprint(&stage.manifest.plan_fingerprint, &source.file_proto)
            != split.split_fingerprint
        {
            return plan_err!(
                "Lake split provenance fingerprint differs from the winning checkpoint"
            );
        }
        let LakeSplitPayload::Lance {
            dataset_uri,
            dataset_version,
            fragment_id,
            row_ranges,
            deletion_file_rev,
        } = decode_lake_split(&source.file_proto)?.payload
        else {
            return not_impl_err!(
                "Paimon backfill requires atomic schema-and-snapshot publication"
            );
        };
        if dataset_url(&dataset_uri)? != target {
            return plan_err!(
                "Lance backfill target differs from the stage source dataset"
            );
        }
        if row_ranges.is_some() || deletion_file_rev.is_some() {
            return plan_err!(
                "Lance backfill requires complete fragments without row ranges or deletion vectors"
            );
        }
        let identity = (dataset_version, &source.order);
        if *baseline.get_or_insert(identity) != identity {
            return plan_err!(
                "Lance backfill winners disagree about snapshot or row order"
            );
        }
        if winners.insert(fragment_id, split).is_some() {
            return plan_err!("Duplicate winning split for Lance fragment {fragment_id}");
        }
    }
    if stage.manifest.dlq_rows != 0 {
        return plan_err!("Lance backfill requires a stage with zero DLQ rows");
    }
    let (version, order) = baseline.expect("validated nonempty manifest");
    if matches!(order, BackfillOrder::RequiresKey) {
        return plan_err!(
            "Lance backfill requires all physical rows and an unchanged source --row-key for this stage"
        );
    }
    Ok((version, order.clone(), winners))
}

fn dataset_url(uri: &str) -> Result<Url> {
    let parsed = if FsPath::new(uri).is_absolute() {
        Url::from_file_path(uri).ok()
    } else {
        Url::parse(uri).ok()
    };
    let mut url = parsed
        .ok_or_else(|| plan_datafusion_err!("Lance backfill requires a dataset URI"))?;
    if !url.username().is_empty()
        || url.password().is_some()
        || url.query().is_some()
        || url.fragment().is_some()
    {
        return plan_err!("Lance backfill requires a credential-free dataset URI");
    }
    url.path_segments_mut()
        .map_err(|()| plan_datafusion_err!("Invalid Lance dataset URI"))?
        .pop_if_empty();
    Ok(url)
}

async fn winner_reader(
    store: &dyn ObjectStore,
    stage: &PublishedStage,
    split: &SplitCheckpoint,
) -> Result<ParquetRecordBatchReader> {
    let relative = &split.files.data_rel_path;
    if let Some(root) = &stage.local_root {
        check_local_file(&root.join(relative)).await?;
    }
    let path = stage_path(&stage.stage_root, relative);
    let bytes = store.get(&path).await?.bytes().await?;
    if bytes.len() as u64 != split.files.data_bytes
        || hex::encode(Sha256::digest(&bytes)) != split.files.data_sha256
    {
        return exec_err!("Lance backfill winner size or checksum differs at {path}");
    }
    let reader = ParquetRecordBatchReaderBuilder::try_new(bytes)?;
    if u64::try_from(reader.metadata().file_metadata().num_rows()).ok()
        != Some(split.files.output_rows)
    {
        return exec_err!("Lance backfill winner row count differs at {path}");
    }
    Ok(reader.build()?)
}

async fn align_by_key(
    dataset: &Arc<Dataset>,
    unit: BackfillWorkUnit,
    source: &str,
    key_index: usize,
    rows: RecordBatch,
) -> Result<RecordBatch> {
    let fragment = dataset
        .get_fragment(unit.fragment_id as usize)
        .ok_or_else(|| {
            plan_datafusion_err!("Missing Lance backfill fragment {}", unit.fragment_id)
        })?;
    let mut scanner =
        Scanner::from_fragment(Arc::clone(dataset), fragment.metadata().clone());
    scanner
        .blob_handling(BlobHandling::BlobsDescriptions)
        .project(&[source])
        .map_err(lance_error)?
        .with_row_address();
    let mut keys = HashMap::new();
    let mut offsets = vec![None; rows.num_rows()];
    let winner_keys = rows.column(key_index);
    let encode_keys =
        RowConverter::new(vec![SortField::new(winner_keys.data_type().clone())])?;
    let mut source_batches = scanner.try_into_stream().await.map_err(lance_error)?;
    while let Some(batch) = source_batches.try_next().await.map_err(lance_error)? {
        let addresses = batch
            .column_by_name(ROW_ADDR)
            .and_then(|column| column.as_primitive_opt::<UInt64Type>())
            .ok_or_else(|| {
                plan_datafusion_err!("Missing Lance physical row addresses")
            })?;
        let source_keys = batch.column_by_name(source).ok_or_else(|| {
            plan_datafusion_err!("Missing Lance backfill source row key")
        })?;
        if source_keys.logical_null_count() != 0 {
            return plan_err!("Lance backfill source row keys must be non-null");
        }
        let encoded = encode_keys.convert_columns(std::slice::from_ref(source_keys))?;
        for (key, address) in encoded.iter().zip(addresses.iter()) {
            let address = address.map(RowAddress::new_from_u64).ok_or_else(|| {
                plan_datafusion_err!("Missing physical address for Lance backfill row")
            })?;
            let offset = address.row_offset() as usize;
            if address.fragment_id() != unit.fragment_id
                || offset >= offsets.len()
                || keys.insert(key.data().to_vec(), offset).is_some()
            {
                return plan_err!(
                    "Lance backfill source row keys must be unique and non-null per fragment"
                );
            }
        }
    }
    let encoded = encode_keys.convert_columns(std::slice::from_ref(winner_keys))?;
    for (row, key) in encoded.iter().enumerate() {
        let offset = keys.remove(key.data()).ok_or_else(|| {
            plan_datafusion_err!("Lance backfill winner row key is missing or repeated")
        })?;
        if offsets[offset].replace(row as u64).is_some() {
            return plan_err!("Duplicate Lance backfill physical row offset");
        }
    }
    let indices: Option<Vec<_>> = offsets.into_iter().collect();
    let indices = indices.ok_or_else(|| {
        plan_datafusion_err!("Lance backfill did not cover every physical row offset")
    })?;
    Ok(take_record_batch(&rows, &UInt64Array::from(indices))?)
}

fn lance_error(error: lance::Error) -> DataFusionError {
    DataFusionError::External(Box::new(error))
}
