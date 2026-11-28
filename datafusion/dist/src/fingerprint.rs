// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: See the NOTICE file distributed with this work for additional information regarding copyright ownership
// SPDX-FileContributor: Licensed to the Apache Software Foundation (ASF) under one or more contributor license agreements

use std::collections::BTreeMap;

use datafusion::arrow::datatypes::Schema;
use datafusion::common::Result;
use datafusion_proto::generated::datafusion_common::arrow_type::ArrowTypeEnum;
use datafusion_proto::protobuf;
use prost::{Message, encoding::bytes};
use protobuf::physical_plan_node::PhysicalPlanType;
use sha2::{Digest, Sha256};

pub(crate) fn plan_fingerprint(plan: &[u8], schema: &Schema) -> Result<String> {
    let wire_schema = encode_schema(protobuf::Schema::try_from(schema)?);
    let mut hash = Sha256::new();
    hash.update(b"df.dist.plan.v1\0");
    hash.update((plan.len() as u64).to_le_bytes());
    hash.update(plan);
    hash.update((wire_schema.len() as u64).to_le_bytes());
    hash.update(wire_schema);
    Ok(hex::encode(hash.finalize()))
}

// Plan fingerprints include the wire bytes. Preserve protobuf field tags while
// ordering the schema metadata maps in Parquet scans and projections.
// Named extensions must canonicalize their own payloads before reaching here.
pub(crate) fn encode_plan(plan: protobuf::PhysicalPlanNode) -> Vec<u8> {
    let mut node = Vec::new();
    let tag = match plan.physical_plan_type {
        Some(PhysicalPlanType::ParquetScan(mut scan)) => {
            if let Some(mut conf) = scan.base_conf.take() {
                let mut base = Vec::new();
                if let Some(schema) = conf.schema.take() {
                    bytes::encode(2, &encode_schema(schema), &mut base);
                }
                conf.encode_raw(&mut base);
                bytes::encode(1, &base, &mut node);
            }
            scan.encode_raw(&mut node);
            1
        }
        Some(PhysicalPlanType::Projection(mut projection)) => {
            encode_input(projection.input.take(), &mut node);
            let schema = projection.schema.take();
            projection.encode_raw(&mut node);
            if let Some(schema) = schema {
                bytes::encode(4, &encode_schema(schema), &mut node);
            }
            4
        }
        Some(PhysicalPlanType::CoalesceBatches(mut coalesce)) => {
            encode_input(coalesce.input.take(), &mut node);
            coalesce.encode_raw(&mut node);
            11
        }
        Some(PhysicalPlanType::Filter(mut filter)) => {
            encode_input(filter.input.take(), &mut node);
            filter.encode_raw(&mut node);
            12
        }
        Some(PhysicalPlanType::Extension(mut extension)) => {
            let inputs = std::mem::take(&mut extension.inputs);
            extension.encode_raw(&mut node);
            for input in inputs {
                bytes::encode(2, &encode_plan(input), &mut node);
            }
            18
        }
        _ => return plan.encode_to_vec(),
    };
    let mut buf = Vec::new();
    bytes::encode(tag, &node, &mut buf);
    buf
}

fn encode_input(input: Option<Box<protobuf::PhysicalPlanNode>>, buf: &mut Vec<u8>) {
    if let Some(input) = input {
        bytes::encode(1, &encode_plan(*input), buf);
    }
}

pub(crate) fn split_fingerprint(plan_fingerprint: &str, file: &[u8]) -> String {
    let mut hash = Sha256::new();
    hash.update(b"df.dist.split.v1\0");
    hash.update(plan_fingerprint.as_bytes());
    hash.update((file.len() as u64).to_le_bytes());
    hash.update(file);
    hex::encode(hash.finalize())
}

// Wire-compatible with protobuf::Schema and protobuf::Field. Sort the maps so
// independently converted Arrow schemas hash identically across processes.
#[derive(Message)]
struct OrderedSchema {
    #[prost(bytes, repeated, tag = "1")]
    columns: Vec<Vec<u8>>,
    #[prost(btree_map = "string, string", tag = "2")]
    metadata: BTreeMap<String, String>,
}

#[derive(Message)]
struct OrderedField {
    #[prost(string, tag = "1")]
    name: String,
    #[prost(bytes, optional, tag = "2")]
    arrow_type: Option<Vec<u8>>,
    #[prost(bool, tag = "3")]
    nullable: bool,
    #[prost(bytes, repeated, tag = "4")]
    children: Vec<Vec<u8>>,
    #[prost(btree_map = "string, string", tag = "5")]
    metadata: BTreeMap<String, String>,
}

pub(crate) fn encode_schema(schema: protobuf::Schema) -> Vec<u8> {
    OrderedSchema {
        columns: schema.columns.into_iter().map(encode_field).collect(),
        metadata: schema.metadata.into_iter().collect(),
    }
    .encode_to_vec()
}

fn encode_field(field: protobuf::Field) -> Vec<u8> {
    OrderedField {
        name: field.name,
        arrow_type: field.arrow_type.map(|ty| encode_arrow_type(*ty)),
        nullable: field.nullable,
        children: field.children.into_iter().map(encode_field).collect(),
        metadata: field.metadata.into_iter().collect(),
    }
    .encode_to_vec()
}

fn encode_fields<const N: usize>(fields: [Option<Box<protobuf::Field>>; N]) -> Vec<u8> {
    let mut buf = Vec::new();
    for (index, field) in (1_u32..).zip(fields) {
        if let Some(field) = field {
            bytes::encode(index, &encode_field(*field), &mut buf);
        }
    }
    buf
}

// Re-encode types that can carry nested Field metadata, retaining the
// ArrowType oneof tags and the same field encodings as the upstream protobuf.
fn encode_arrow_type(ty: protobuf::ArrowType) -> Vec<u8> {
    let (tag, value) = match ty.arrow_type_enum {
        Some(ArrowTypeEnum::List(item)) => (25, encode_fields([item.field_type])),
        Some(ArrowTypeEnum::LargeList(item)) => (26, encode_fields([item.field_type])),
        Some(ArrowTypeEnum::FixedSizeList(mut list)) => {
            let mut value = encode_fields([list.field_type.take()]);
            list.encode_raw(&mut value);
            (27, value)
        }
        Some(ArrowTypeEnum::Struct(fields)) => {
            let mut value = Vec::new();
            for field in fields.sub_field_types {
                bytes::encode(1, &encode_field(field), &mut value);
            }
            (28, value)
        }
        Some(ArrowTypeEnum::Union(mut union)) => {
            let mut value = Vec::new();
            for field in std::mem::take(&mut union.union_types) {
                bytes::encode(1, &encode_field(field), &mut value);
            }
            union.encode_raw(&mut value);
            (29, value)
        }
        Some(ArrowTypeEnum::Dictionary(dict)) => {
            let mut value = Vec::new();
            if let Some(key) = dict.key {
                bytes::encode(1, &encode_arrow_type(*key), &mut value);
            }
            if let Some(item) = dict.value {
                bytes::encode(2, &encode_arrow_type(*item), &mut value);
            }
            (30, value)
        }
        Some(ArrowTypeEnum::Map(mut map)) => {
            let mut value = encode_fields([map.field_type.take()]);
            map.encode_raw(&mut value);
            (33, value)
        }
        Some(ArrowTypeEnum::RunEndEncoded(ree)) => {
            (42, encode_fields([ree.run_ends_field, ree.values_field]))
        }
        Some(ArrowTypeEnum::ListView(item)) => (43, encode_fields([item.field_type])),
        Some(ArrowTypeEnum::LargeListView(item)) => {
            (44, encode_fields([item.field_type]))
        }
        _ => return ty.encode_to_vec(),
    };
    let mut buf = Vec::new();
    bytes::encode(tag, &value, &mut buf);
    buf
}
