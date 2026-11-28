// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: See the NOTICE file distributed with this work for additional information regarding copyright ownership
// SPDX-FileContributor: Licensed to the Apache Software Foundation (ASF) under one or more contributor license agreements

use std::collections::BTreeMap;

use datafusion::arrow::datatypes::SchemaRef;
use datafusion::arrow::record_batch::RecordBatch;
use datafusion::common::{Result, plan_err};

use super::SinkOptions;

#[derive(Debug)]
pub(super) enum Routing {
    Plain(SchemaRef),
    #[cfg(feature = "ai")]
    Envelopes(EnvelopeRouting),
}

pub(super) struct Routed {
    pub(super) data: Vec<RecordBatch>,
    pub(super) dlq: Option<RecordBatch>,
    pub(super) dlq_by_code: BTreeMap<String, u64>,
}

impl Routing {
    pub(super) fn try_new(schema: SchemaRef, options: &SinkOptions) -> Result<Self> {
        #[cfg(feature = "ai")]
        if let Some(routing) = EnvelopeRouting::try_new(&schema, options)? {
            return Ok(Self::Envelopes(routing));
        }
        #[cfg(not(feature = "ai"))]
        {
            let _ = options;
            if schema
                .fields()
                .iter()
                .any(|field| field.extension_type_name() == Some("df.ai_error"))
            {
                return plan_err!("Routing df.ai_error requires the ai feature");
            }
        }
        Ok(Self::Plain(schema))
    }

    pub(super) fn data_schema(&self) -> &SchemaRef {
        match self {
            Self::Plain(schema) => schema,
            #[cfg(feature = "ai")]
            Self::Envelopes(routing) => &routing.data_schema,
        }
    }

    pub(super) fn route(&self, batches: &[RecordBatch]) -> Result<Routed> {
        match self {
            Self::Plain(_) => Ok(Routed {
                data: batches.to_vec(),
                dlq: None,
                dlq_by_code: BTreeMap::new(),
            }),
            #[cfg(feature = "ai")]
            Self::Envelopes(routing) => routing.route(batches),
        }
    }
}

#[cfg(feature = "ai")]
use envelopes::EnvelopeRouting;

#[cfg(feature = "ai")]
mod envelopes {
    use std::sync::Arc;

    use arrow_schema::extension::ExtensionType;
    use datafusion::arrow::array::{
        Array, ArrayRef, AsArray, BooleanArray, BooleanBuilder, StringBuilder,
        UInt64Array, make_comparator,
    };
    use datafusion::arrow::compute::{
        concat_batches, filter_record_batch, take, take_record_batch,
    };
    use datafusion::arrow::datatypes::{DataType, Field, Schema};
    use datafusion::common::plan_datafusion_err;
    use datafusion_ai::AiErrorType;

    use super::*;

    #[derive(Debug)]
    pub(in crate::sink) struct EnvelopeRouting {
        pub(super) data_schema: SchemaRef,
        dlq_schema: SchemaRef,
        key_index: usize,
        columns: Vec<usize>,
    }

    impl EnvelopeRouting {
        pub(super) fn try_new(
            schema: &SchemaRef,
            options: &SinkOptions,
        ) -> Result<Option<Self>> {
            let mut columns = Vec::new();
            let mut fields = schema.fields().to_vec();
            for (index, field) in schema.fields().iter().enumerate() {
                if field.extension_type_name() != Some(AiErrorType::NAME) {
                    continue;
                }
                field.try_extension_type::<AiErrorType>().map_err(|error| {
                    plan_datafusion_err!(
                        "Invalid df.ai_error column '{}': {error}",
                        field.name()
                    )
                })?;
                let DataType::Struct(storage) = field.data_type() else {
                    unreachable!()
                };
                fields[index] =
                    Arc::new(storage[0].as_ref().clone().with_name(field.name()));
                columns.push(index);
            }
            if columns.is_empty() {
                return Ok(None);
            }
            let key_name = options.row_key.as_deref().ok_or_else(|| {
                plan_datafusion_err!(
                    "df.split_row_key is required for df.ai_error output"
                )
            })?;
            let matches = schema
                .fields()
                .iter()
                .enumerate()
                .filter(|(_, field)| field.name() == key_name)
                .collect::<Vec<_>>();
            let [(key_index, key)] = matches.as_slice() else {
                return plan_err!(
                    "df.split_row_key '{key_name}' must name exactly one output column"
                );
            };
            let key = key
                .as_ref()
                .clone()
                .with_name("row_key")
                .with_nullable(false);
            let dlq_schema = Schema::new(vec![
                key,
                Field::new("failed_column", DataType::Utf8, false),
                Field::new("error_code", DataType::Utf8, false),
                Field::new("error_message", DataType::Utf8, false),
                Field::new("retryable", DataType::Boolean, false),
            ]);
            Ok(Some(Self {
                data_schema: Arc::new(Schema::new_with_metadata(
                    fields,
                    schema.metadata().clone(),
                )),
                dlq_schema: Arc::new(dlq_schema),
                key_index: *key_index,
                columns,
            }))
        }

        pub(super) fn route(&self, batches: &[RecordBatch]) -> Result<Routed> {
            let mut data = Vec::with_capacity(batches.len());
            let mut dlq_batches = Vec::new();
            let mut dlq_by_code = BTreeMap::new();
            for batch in batches {
                let key = batch.column(self.key_index);
                if key.logical_null_count() != 0 {
                    return plan_err!(
                        "df.split_row_key must be non-null for every output row"
                    );
                }
                let mut success = vec![true; batch.num_rows()];
                let mut columns = batch.columns().to_vec();
                let mut dlq = DlqBuilder::default();
                for &index in &self.columns {
                    let envelope = batch.column(index).as_struct();
                    let (_, children) = envelope.flatten();
                    let [value, code, message, retryable] = children.as_slice() else {
                        unreachable!()
                    };
                    let code = code.as_string::<i32>();
                    let message = message.as_string::<i32>();
                    let retryable = retryable.as_boolean();
                    let name = self.data_schema.field(index).name();
                    for (row, keep) in success.iter_mut().enumerate() {
                        if code.is_null(row) {
                            continue;
                        }
                        if message.is_null(row) || retryable.is_null(row) {
                            return plan_err!(
                                "Invalid error fields in df.ai_error column '{name}' at row {row}"
                            );
                        }
                        *keep = false;
                        let error_code = code.value(row);
                        dlq.push(
                            row,
                            name,
                            error_code,
                            message.value(row),
                            retryable.value(row),
                        );
                        *dlq_by_code.entry(error_code.to_owned()).or_default() += 1;
                    }
                    columns[index] = Arc::clone(value);
                }
                let batch = RecordBatch::try_new(Arc::clone(&self.data_schema), columns)?;
                data.push(filter_record_batch(&batch, &BooleanArray::from(success))?);
                if !dlq.rows.is_empty() {
                    dlq_batches.push(dlq.finish(key, Arc::clone(&self.dlq_schema))?);
                }
            }
            let dlq = if dlq_batches.is_empty() {
                None
            } else {
                let batch = concat_batches(&self.dlq_schema, &dlq_batches)?;
                let keys = batch.column(0);
                let names = batch.column(1).as_string::<i32>();
                let compare =
                    make_comparator(keys.as_ref(), keys.as_ref(), Default::default())?;
                let mut order: Vec<_> = (0..batch.num_rows()).collect();
                order.sort_by(|&a, &b| {
                    compare(a, b).then_with(|| names.value(a).cmp(names.value(b)))
                });
                let indices = UInt64Array::from_iter_values(
                    order.into_iter().map(|row| row as u64),
                );
                Some(take_record_batch(&batch, &indices)?)
            };
            Ok(Routed {
                data,
                dlq,
                dlq_by_code,
            })
        }
    }

    #[derive(Default)]
    struct DlqBuilder {
        rows: Vec<u64>,
        columns: StringBuilder,
        codes: StringBuilder,
        messages: StringBuilder,
        retryable: BooleanBuilder,
    }

    impl DlqBuilder {
        fn push(
            &mut self,
            row: usize,
            column: &str,
            code: &str,
            message: &str,
            retryable: bool,
        ) {
            self.rows.push(row as u64);
            self.columns.append_value(column);
            self.codes.append_value(code);
            self.messages.append_value(message);
            self.retryable.append_value(retryable);
        }

        fn finish(mut self, key: &ArrayRef, schema: SchemaRef) -> Result<RecordBatch> {
            Ok(RecordBatch::try_new(
                schema,
                vec![
                    take(key.as_ref(), &UInt64Array::from(self.rows), None)?,
                    Arc::new(self.columns.finish()),
                    Arc::new(self.codes.finish()),
                    Arc::new(self.messages.finish()),
                    Arc::new(self.retryable.finish()),
                ],
            )?)
        }
    }
}
