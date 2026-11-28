// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: See the NOTICE file distributed with this work for additional information regarding copyright ownership
// SPDX-FileContributor: Licensed to the Apache Software Foundation (ASF) under one or more contributor license agreements

use std::sync::Arc;

use datafusion::arrow::datatypes::Schema;
use datafusion::common::extensions::Extensions;
use datafusion::common::{Result, plan_datafusion_err, plan_err};
use datafusion::datasource::listing::PartitionedFile;
use datafusion::datasource::physical_plan::FileScanConfig;
use datafusion::datasource::source::DataSourceExec;
use datafusion::execution::TaskContext;
use datafusion::physical_plan::ExecutionPlan;
use datafusion_lake::dist::{
    LAKE_SCAN_PROTOCOL_NAME, LAKE_SCAN_PROTOCOL_VERSION, bind_split, decode_split,
    encode_scan, rebuild_scan,
};
use datafusion_lake::{LAKE_SPLIT_EXT_NAME, LAKE_SPLIT_EXT_VERSION, LakeSplitExt};
use datafusion_proto::physical_plan::{
    PhysicalExtensionCodec, PhysicalProtoConverterExtension,
};
use datafusion_proto::protobuf;
use prost::Message;

use crate::fingerprint::encode_schema;
use crate::{
    DistCodecs, ExtensionOperatorPolicy, FileExtensionCodec, NamedPhysicalExtensionCodec,
};

mod backfill;

pub use backfill::backfill_lance;
pub(crate) use backfill::backfill_order;

impl DistCodecs {
    /// Register the Lake scan and split codecs for both planning and execution.
    ///
    /// Worker sessions can register a
    /// [`LakeRuntimeConfig`](datafusion_lake::dist::LakeRuntimeConfig) to supply
    /// process-local storage providers when opening a pinned snapshot.
    ///
    /// ```
    /// use datafusion_dist::DistCodecs;
    /// let codecs = DistCodecs::default().with_lake()?;
    /// # Ok::<(), datafusion::common::DataFusionError>(())
    /// ```
    pub fn with_lake(mut self) -> Result<Self> {
        self.physical.register(Arc::new(LakeScanCodec))?;
        self.file_extensions.register(Arc::new(LakeSplitCodec))?;
        Ok(self)
    }
}

#[derive(Debug)]
struct LakeScanCodec;

#[derive(Message)]
struct LakeScanPayload {
    #[prost(bytes, tag = "1")]
    scan: Vec<u8>,
    #[prost(bytes, optional, tag = "2")]
    output_schema: Option<Vec<u8>>,
}

fn lake_config(node: &dyn ExecutionPlan) -> Option<&FileScanConfig> {
    if node.downcast_delegate().is_some() {
        return None;
    }
    let config = node
        .downcast_ref::<DataSourceExec>()?
        .data_source()
        .downcast_ref::<FileScanConfig>()?;
    matches!(config.file_source().file_type(), "lance" | "paimon").then_some(config)
}

impl NamedPhysicalExtensionCodec for LakeScanCodec {
    fn extension_name(&self) -> &str {
        LAKE_SCAN_PROTOCOL_NAME
    }

    fn operator_policy(
        &self,
        node: &dyn ExecutionPlan,
    ) -> Option<ExtensionOperatorPolicy> {
        lake_config(node).map(|_| ExtensionOperatorPolicy {
            map_only: true,
            speculation_allowed: true,
        })
    }

    fn try_encode(
        &self,
        node: &Arc<dyn ExecutionPlan>,
        _: &dyn PhysicalProtoConverterExtension,
        _: &dyn PhysicalExtensionCodec,
    ) -> Result<Option<(u32, Vec<u8>)>> {
        let Some(config) = lake_config(node.as_ref()) else {
            return Ok(None);
        };
        let Some(scan) = encode_scan(config)? else {
            return Ok(None);
        };
        let output_schema =
            encode_schema(protobuf::Schema::try_from(node.schema().as_ref())?);
        Ok(Some((
            LAKE_SCAN_PROTOCOL_VERSION,
            LakeScanPayload {
                scan,
                output_schema: Some(output_schema),
            }
            .encode_to_vec(),
        )))
    }

    fn try_decode(
        &self,
        version: u32,
        payload: &[u8],
        inputs: &[Arc<dyn ExecutionPlan>],
        ctx: &TaskContext,
        _: &dyn PhysicalProtoConverterExtension,
        _: &dyn PhysicalExtensionCodec,
    ) -> Result<Arc<dyn ExecutionPlan>> {
        if version != LAKE_SCAN_PROTOCOL_VERSION || !inputs.is_empty() {
            return plan_err!(
                "Lake scan requires version {LAKE_SCAN_PROTOCOL_VERSION} and zero inputs"
            );
        }
        let payload = LakeScanPayload::decode(payload).map_err(|error| {
            plan_datafusion_err!("Invalid Lake scan protobuf: {error}")
        })?;
        let schema = payload.output_schema.ok_or_else(|| {
            plan_datafusion_err!("Lake scan payload is missing its output schema")
        })?;
        let schema = protobuf::Schema::decode(schema.as_slice()).map_err(|error| {
            plan_datafusion_err!("Invalid Lake scan output schema: {error}")
        })?;
        rebuild_scan(&payload.scan, Arc::new(Schema::try_from(&schema)?), ctx)
    }

    fn bind_split(
        &self,
        node: &Arc<dyn ExecutionPlan>,
        file: PartitionedFile,
    ) -> Result<Option<Arc<dyn ExecutionPlan>>> {
        lake_config(node.as_ref())
            .map(|config| bind_split(config, file))
            .transpose()
    }
}

#[derive(Debug)]
struct LakeSplitCodec;

impl FileExtensionCodec for LakeSplitCodec {
    fn extension_name(&self) -> &str {
        LAKE_SPLIT_EXT_NAME
    }

    fn try_encode(&self, extensions: &Extensions) -> Result<Option<(u32, Vec<u8>)>> {
        let encoded = LakeSplitExt::try_encode_from_extensions(extensions)?;
        if let Some((_, bytes)) = &encoded
            && bytes.len() > 16 * 1024 * 1024
        {
            return plan_err!("Lake split payload exceeds the 16 MiB limit");
        }
        Ok(encoded)
    }

    fn try_decode(
        &self,
        version: u32,
        payload: &[u8],
        extensions: &mut Extensions,
    ) -> Result<()> {
        if version != LAKE_SPLIT_EXT_VERSION {
            return plan_err!("Unsupported lake split wire version {version}");
        }
        extensions.merge(&decode_split(payload)?.extensions);
        Ok(())
    }
}
