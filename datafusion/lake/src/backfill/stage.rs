// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: See the NOTICE file distributed with this work for additional information regarding copyright ownership
// SPDX-FileContributor: Licensed to the Apache Software Foundation (ASF) under one or more contributor license agreements

use std::collections::{BTreeMap, HashMap, HashSet};
use std::future::Future;
use std::iter::Peekable;
use std::ops::Range;
use std::sync::Arc;

use arrow::array::{
    ArrayRef, FixedSizeListArray, RecordBatch, StructArray, new_null_array,
};
use arrow::datatypes::{DataType, Field, SchemaRef};
use datafusion_common::{Result, ScalarValue, exec_err, plan_datafusion_err, plan_err};
use datafusion_execution::SendableRecordBatchStream;
use datafusion_physical_plan::stream::RecordBatchStreamAdapter;
use futures::{TryStreamExt, stream};

use crate::dist::decode_split;
use crate::split::{LakeSplitExt, LakeSplitPayload};

use super::plan::{BackfillPlan, BackfillWorkUnit, FragmentLayout, dataset_url};

/// Bind the winning Parquet files of a distributed stage to backfill work units.
/// `splits` pairs each stage split ID with its `df.lake.split` v1 bytes. Supply
/// one winner per split and an opener that streams its exact Arrow schema and
/// rows in Parquet file order, including schema for a zero-row file. The stage
/// must preserve every visible row in fragment offset order.
///
/// Each returned stream can be passed to [`super::write_backfill_unit`]. Streams
/// share live Arrow buffers and insert NULLs at deleted physical offsets. For
/// non-nullable fields they insert type defaults, which the deletion vector
/// keeps invisible. Files are read as their streams are polled.
///
/// # Errors
/// Returns an error for incomplete or inconsistent split and winner sets. Each
/// stream reports opener failures and schema or row-count mismatches.
///
/// # Public Only for Internal Use:
/// This is not a public API and is for internal use only; see [API policy] for details.
/// [API policy]: https://datafusion.apache.org/contributor-guide/api-health.html#internal-public-apis
#[doc(hidden)]
pub fn bridge_backfill_stage<'a, File, Read>(
    plan: &BackfillPlan,
    splits: impl IntoIterator<Item = (u64, &'a [u8])>,
    winners: impl IntoIterator<Item = (u64, File)>,
    mut open_parquet: impl FnMut(File) -> Read,
) -> Result<Vec<(BackfillWorkUnit, SendableRecordBatchStream)>>
where
    Read: Future<Output = Result<SendableRecordBatchStream>> + Send + 'static,
{
    plan.validate()?;
    let uri = dataset_url(plan.dataset_uri())?;
    let mut bindings = HashMap::new();
    let mut fragments = HashSet::new();
    for (split_id, bytes) in splits {
        let file = decode_split(bytes)?;
        let LakeSplitPayload::Lance {
            dataset_uri,
            dataset_version,
            fragment_id,
            row_ranges: None,
            deletion_file_rev,
        } = &LakeSplitExt::for_read(&file)?.payload
        else {
            return plan_err!(
                "Backfill split {split_id} must cover one whole Lance fragment"
            );
        };
        let layout = plan.fragment(*fragment_id)?;
        if dataset_url(dataset_uri)? != uri
            || *dataset_version != plan.dataset_version()
            || *deletion_file_rev != layout.deletion_file_rev
        {
            return plan_err!(
                "Backfill split {split_id} does not match the pinned Lance fragment"
            );
        }
        if bindings.insert(split_id, layout).is_some() || !fragments.insert(*fragment_id)
        {
            return plan_err!(
                "Duplicate backfill split {split_id} or Lance fragment {fragment_id}"
            );
        }
    }
    if bindings.len() != plan.payload.fragments.len() {
        return plan_err!("Missing stage splits for planned Lance fragments");
    }

    let mut files = BTreeMap::new();
    for (split_id, file) in winners {
        let layout = bindings.get(&split_id).ok_or_else(|| {
            plan_datafusion_err!("Unknown backfill stage split {split_id}")
        })?;
        if files
            .insert(layout.unit.fragment_id, (*layout, file))
            .is_some()
        {
            return plan_err!(
                "Duplicate winning Parquet file for stage split {split_id}"
            );
        }
    }
    if files.len() != bindings.len() {
        return plan_err!("Missing winning Parquet files for backfill stage splits");
    }
    let schema = Arc::new(plan.new_columns().clone());
    Ok(files
        .into_values()
        .map(|(layout, file)| {
            let input = open_parquet(file);
            (
                layout.unit,
                physical_batches(layout.clone(), Arc::clone(&schema), input),
            )
        })
        .collect())
}

fn physical_batches(
    layout: FragmentLayout,
    schema: SchemaRef,
    open: impl Future<Output = Result<SendableRecordBatchStream>> + Send + 'static,
) -> SendableRecordBatchStream {
    let open_stream = {
        let schema = Arc::clone(&schema);
        async move {
            let input = open.await?;
            if input.schema() != schema {
                return exec_err!(
                    "Backfill Parquet schema differs from the planned new columns"
                );
            }
            let state = PhysicalBatches {
                input,
                schema,
                unit: layout.unit,
                offset: 0,
                deleted: layout.deleted_ranges.into_iter().peekable(),
                pending: None,
            };
            Ok(stream::try_unfold(state, |mut state| async move {
                Ok(state.next_batch().await?.map(|batch| (batch, state)))
            }))
        }
    };
    let stream = stream::once(open_stream).try_flatten();
    Box::pin(RecordBatchStreamAdapter::new(schema, stream))
}

struct PhysicalBatches {
    input: SendableRecordBatchStream,
    schema: SchemaRef,
    unit: BackfillWorkUnit,
    offset: u32,
    deleted: Peekable<std::vec::IntoIter<Range<u32>>>,
    pending: Option<RecordBatch>,
}

impl PhysicalBatches {
    async fn next_batch(&mut self) -> Result<Option<RecordBatch>> {
        const MAX_PADDING_ROWS: u32 = 1024;
        let physical_end = self.unit.row_count as u32;
        let fragment = self.unit.fragment_id;

        let mut live_end = physical_end;
        if let Some(range) = self.deleted.peek() {
            if self.offset >= range.start {
                let count = (range.end - self.offset).min(MAX_PADDING_ROWS);
                self.offset += count;
                if self.offset == range.end {
                    self.deleted.next();
                }
                let columns = self
                    .schema
                    .fields()
                    .iter()
                    .map(|field| deleted_values(field, count as usize))
                    .collect::<Result<Vec<_>>>()?;
                return Ok(Some(RecordBatch::try_new(
                    Arc::clone(&self.schema),
                    columns,
                )?));
            }
            live_end = range.start;
        }

        loop {
            if let Some(batch) = self.pending.take() {
                let count = batch.num_rows().min((live_end - self.offset) as usize);
                if count == 0 {
                    return exec_err!(
                        "Too many logical rows in Parquet for Lance fragment {fragment}"
                    );
                }
                self.offset += count as u32;
                if count == batch.num_rows() {
                    return Ok(Some(batch));
                }
                self.pending = Some(batch.slice(count, batch.num_rows() - count));
                return Ok(Some(batch.slice(0, count)));
            }

            let Some(batch) = self.input.try_next().await? else {
                if self.offset != physical_end {
                    return exec_err!(
                        "Too few logical rows in Parquet for Lance fragment {fragment}"
                    );
                }
                return Ok(None);
            };
            if batch.schema_ref() != &self.schema {
                return exec_err!(
                    "Backfill Parquet batch schema differs from the planned new columns"
                );
            }
            if batch.num_rows() > 0 {
                self.pending = Some(batch);
            }
        }
    }
}

fn deleted_values(field: &Field, count: usize) -> Result<ArrayRef> {
    let data_type = field.data_type();
    if field.is_nullable() {
        return Ok(new_null_array(data_type, count));
    }
    match data_type {
        DataType::Struct(fields) => {
            let children = fields
                .iter()
                .map(|field| deleted_values(field, count))
                .collect::<Result<Vec<_>>>()?;
            Ok(Arc::new(StructArray::try_new(
                fields.clone(),
                children,
                None,
            )?))
        }
        DataType::FixedSizeList(item, size) => {
            let children = count.checked_mul(*size as usize).ok_or_else(|| {
                plan_datafusion_err!(
                    "Backfill vector padding exceeds Arrow array capacity"
                )
            })?;
            Ok(Arc::new(FixedSizeListArray::try_new(
                Arc::clone(item),
                *size,
                deleted_values(item, children)?,
                None,
            )?))
        }
        _ => ScalarValue::new_default(data_type)?.to_array_of_size(count),
    }
}
