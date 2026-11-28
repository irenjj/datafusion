// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: See the NOTICE file distributed with this work for additional information regarding copyright ownership
// SPDX-FileContributor: Licensed to the Apache Software Foundation (ASF) under one or more contributor license agreements

use std::collections::BTreeMap;
use std::fmt::Debug;
use std::sync::Arc;

use datafusion::arrow::datatypes::Schema;
use datafusion::common::{Result, plan_datafusion_err, plan_err};
use datafusion::datasource::listing::PartitionedFile;
use datafusion::datasource::physical_plan::ParquetSource;
use datafusion::datasource::source::DataSourceExec;
use datafusion::execution::TaskContext;
use datafusion::physical_expr::PhysicalExpr;
#[expect(deprecated)]
use datafusion::physical_plan::coalesce_batches::CoalesceBatchesExec;
use datafusion::physical_plan::{
    ExecutionPlan, filter::FilterExec, projection::ProjectionExec,
};
use datafusion_proto::physical_plan::{
    PhysicalExtensionCodec, PhysicalPlanDecodeContext, PhysicalPlanNodeExt,
    PhysicalProtoConverterExtension, to_proto::serialize_physical_expr_with_converter,
};
use datafusion_proto::protobuf::{
    PhysicalExprNode, PhysicalExtensionNode, PhysicalPlanNode,
    physical_plan_node::PhysicalPlanType,
};
use prost::Message;

/// Execution constraints for an operator claimed by a [`NamedPhysicalExtensionCodec`].
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct ExtensionOperatorPolicy {
    /// Whether the operator can run independently on each split.
    pub map_only: bool,
    /// Whether multiple attempts can safely execute the same split concurrently.
    pub speculation_allowed: bool,
}

/// A versioned codec for a custom [`ExecutionPlan`] operator.
///
/// Use a stable wire name. A codec should claim the same operators in
/// [`Self::operator_policy`] and [`Self::try_encode`].
pub trait NamedPhysicalExtensionCodec: Debug + Send + Sync {
    /// Return the stable, nonblank wire name of this codec.
    fn extension_name(&self) -> &str;

    /// Return the policy for a supported operator, or `None` for other operators.
    fn operator_policy(
        &self,
        node: &dyn ExecutionPlan,
    ) -> Option<ExtensionOperatorPolicy>;

    /// Encode a supported operator as a version and payload, or return `Ok(None)`.
    ///
    /// Use the supplied root codec and converter for nested protobuf values.
    fn try_encode(
        &self,
        node: &Arc<dyn ExecutionPlan>,
        converter: &dyn PhysicalProtoConverterExtension,
        root_codec: &dyn PhysicalExtensionCodec,
    ) -> Result<Option<(u32, Vec<u8>)>>;

    /// Decode an operator and return an error for unsupported versions or payloads.
    ///
    /// `inputs` are the decoded children of the node. Each invocation must
    /// construct an operator with fresh execution metrics.
    fn try_decode(
        &self,
        version: u32,
        payload: &[u8],
        inputs: &[Arc<dyn ExecutionPlan>],
        ctx: &TaskContext,
        converter: &dyn PhysicalProtoConverterExtension,
        root_codec: &dyn PhysicalExtensionCodec,
    ) -> Result<Arc<dyn ExecutionPlan>>;

    /// Bind one split to an empty non-Parquet [`DataSourceExec`] template.
    ///
    /// Return `Some` only when this codec claims `node`. The returned
    /// [`ExecutionPlan`] must retain the template's output schema and have
    /// exactly one output partition. The coordinator also calls this method
    /// during admission, so binding must only construct a plan.
    /// Multiple codecs claiming the same binding cause a planning error.
    fn bind_split(
        &self,
        _node: &Arc<dyn ExecutionPlan>,
        _file: PartitionedFile,
    ) -> Result<Option<Arc<dyn ExecutionPlan>>> {
        Ok(None)
    }
}

/// Registered [`NamedPhysicalExtensionCodec`]s, ordered by wire name.
#[derive(Clone, Debug, Default)]
pub struct DistExtensionRegistry {
    codecs: Arc<BTreeMap<String, Arc<dyn NamedPhysicalExtensionCodec>>>,
}

impl DistExtensionRegistry {
    /// Register a codec. Blank names and duplicate names are plan errors.
    pub fn register(
        &mut self,
        codec: Arc<dyn NamedPhysicalExtensionCodec>,
    ) -> Result<()> {
        let name = codec.extension_name();
        if name.trim().is_empty() {
            return plan_err!("Physical extension codec name must not be empty");
        }
        if self.codecs.contains_key(name) {
            return plan_err!("Physical extension codec '{name}' is already registered");
        }
        Arc::make_mut(&mut self.codecs).insert(name.to_owned(), codec);
        Ok(())
    }

    /// Look up the policy for an operator. Multiple claims are a plan error.
    pub fn policy_for(
        &self,
        node: &dyn ExecutionPlan,
    ) -> Result<Option<ExtensionOperatorPolicy>> {
        let mut matching = self
            .codecs
            .values()
            .filter_map(|codec| codec.operator_policy(node));
        let policy = matching.next();
        if matching.next().is_some() {
            return plan_err!("Ambiguous physical extension policy for {}", node.name());
        }
        Ok(policy)
    }

    pub(crate) fn bind_split(
        &self,
        node: &Arc<dyn ExecutionPlan>,
        file: PartitionedFile,
    ) -> Result<Option<Arc<dyn ExecutionPlan>>> {
        let mut bound = None;
        let files = std::iter::repeat_n(file, self.codecs.len());
        for ((name, codec), file) in self.codecs.iter().zip(files) {
            let candidate = codec.bind_split(node, file).map_err(|error| {
                plan_datafusion_err!("Cannot bind physical extension '{name}': {error}")
            })?;
            if let Some(candidate) = candidate {
                if bound.is_some() {
                    return plan_err!(
                        "Ambiguous physical extension split binding for {}",
                        node.name()
                    );
                }
                bound = Some(candidate);
            }
        }
        Ok(bound)
    }

    /// Check speculation policy at every node, propagating ambiguous claims.
    ///
    /// Split admission is checked separately by
    /// [`crate::split_plan_for_distribution_with_codecs`].
    pub fn plan_allows_speculation(&self, plan: &dyn ExecutionPlan) -> Result<bool> {
        let mut pending = vec![plan];
        let mut allowed = true;
        while let Some(node) = pending.pop() {
            allowed &= self
                .policy_for(node)?
                .is_none_or(|policy| policy.speculation_allowed);
            pending.extend(node.children().into_iter().map(Arc::as_ref));
        }
        Ok(allowed)
    }
}

/// A [`PhysicalExtensionCodec`] that routes versioned operators by wire name.
#[derive(Debug, Default)]
pub struct DistPhysicalCodec {
    registry: DistExtensionRegistry,
}

impl DistPhysicalCodec {
    /// Create a codec using the supplied [`DistExtensionRegistry`].
    pub fn new(registry: DistExtensionRegistry) -> Self {
        Self { registry }
    }

    fn encode_extension(
        &self,
        node: &Arc<dyn ExecutionPlan>,
        converter: &dyn PhysicalProtoConverterExtension,
    ) -> Result<Option<VersionedExtensionNode>> {
        let mut encoded = None;
        for (name, codec) in self.registry.codecs.iter() {
            let candidate = codec.try_encode(node, converter, self).map_err(|error| {
                plan_datafusion_err!("Cannot encode physical extension '{name}': {error}")
            })?;
            if let Some((schema_version, payload)) = candidate {
                if encoded.is_some() {
                    return plan_err!(
                        "Ambiguous physical extension encoding for {}",
                        node.name()
                    );
                }
                encoded = Some(VersionedExtensionNode {
                    extension_name: name.clone(),
                    schema_version,
                    payload,
                });
            }
        }
        Ok(encoded)
    }
}

#[derive(Clone, PartialEq, Message)]
pub(crate) struct VersionedExtensionNode {
    #[prost(string, tag = "1")]
    pub(crate) extension_name: String,
    #[prost(uint32, tag = "2")]
    pub(crate) schema_version: u32,
    #[prost(bytes = "vec", tag = "3")]
    pub(crate) payload: Vec<u8>,
}

impl PhysicalExtensionCodec for DistPhysicalCodec {
    fn try_decode(
        &self,
        buf: &[u8],
        inputs: &[Arc<dyn ExecutionPlan>],
        ctx: &TaskContext,
        converter: &dyn PhysicalProtoConverterExtension,
    ) -> Result<Arc<dyn ExecutionPlan>> {
        let node = VersionedExtensionNode::decode(buf).map_err(|error| {
            plan_datafusion_err!("Invalid physical extension protobuf: {error}")
        })?;
        let name = &node.extension_name;
        if name.trim().is_empty() {
            return plan_err!("Physical extension name must not be empty");
        }
        let codec = self.registry.codecs.get(name).ok_or_else(|| {
            plan_datafusion_err!("Unregistered physical extension '{name}'")
        })?;
        codec
            .try_decode(
                node.schema_version,
                &node.payload,
                inputs,
                ctx,
                converter,
                self,
            )
            .map_err(|error| {
                plan_datafusion_err!(
                    "Cannot decode physical extension '{name}' version {}: {error}",
                    node.schema_version
                )
            })
    }

    fn try_encode(
        &self,
        node: Arc<dyn ExecutionPlan>,
        buf: &mut Vec<u8>,
        converter: &dyn PhysicalProtoConverterExtension,
    ) -> Result<()> {
        let encoded = self.encode_extension(&node, converter)?.ok_or_else(|| {
            plan_datafusion_err!("Unsupported physical extension {}", node.name())
        })?;
        encoded.encode_raw(buf);
        Ok(())
    }
}

#[expect(deprecated)]
pub(crate) fn is_builtin(plan: &dyn ExecutionPlan) -> bool {
    plan.downcast_delegate().is_none()
        && (plan.is::<DataSourceExec>()
            || plan.is::<ProjectionExec>()
            || plan.is::<FilterExec>()
            || plan.is::<CoalesceBatchesExec>())
}

// DataFusion's session registry can bypass the extension codec for named nodes.
pub(crate) struct MapPlanProtoConverter<'a>(pub(crate) &'a DistPhysicalCodec);

impl PhysicalProtoConverterExtension for MapPlanProtoConverter<'_> {
    fn proto_to_execution_plan(
        &self,
        proto: &PhysicalPlanNode,
        ctx: &PhysicalPlanDecodeContext<'_>,
    ) -> Result<Arc<dyn ExecutionPlan>> {
        match &proto.physical_plan_type {
            Some(
                PhysicalPlanType::ParquetScan(_)
                | PhysicalPlanType::Projection(_)
                | PhysicalPlanType::Filter(_)
                | PhysicalPlanType::CoalesceBatches(_),
            ) => self.default_proto_to_execution_plan(proto, ctx),
            Some(PhysicalPlanType::Extension(extension)) => {
                if extension.plan_name.is_some() {
                    return plan_err!(
                        "Map-only stage extensions must use the versioned codec"
                    );
                }
                if extension.inputs.len() > 1 {
                    return plan_err!(
                        "Map-only stage extensions require exactly one input for unary operators or zero inputs for scan leaves"
                    );
                }
                self.default_proto_to_execution_plan(proto, ctx)
            }
            _ => plan_err!("Unsupported physical plan node in map-only stage"),
        }
    }

    fn execution_plan_to_proto(
        &self,
        plan: &Arc<dyn ExecutionPlan>,
        codec: &dyn PhysicalExtensionCodec,
    ) -> Result<PhysicalPlanNode> {
        if plan.downcast_delegate().is_none()
            && let Some(source) = plan.downcast_ref::<DataSourceExec>()
            && source.downcast_to_file_source::<ParquetSource>().is_none()
            && let Some(node) = self.0.encode_extension(plan, self)?
        {
            return Ok(extension_proto(node.encode_to_vec(), vec![]));
        }
        if is_builtin(plan.as_ref()) {
            let proto = PhysicalPlanNode::try_from_physical_plan_with_converter(
                Arc::clone(plan),
                codec,
                self,
            )?;
            if matches!(
                proto.physical_plan_type,
                Some(PhysicalPlanType::Extension(_))
            ) {
                return plan_err!("Cannot serialize map-only operator {}", plan.name());
            }
            return Ok(proto);
        }
        let [child] = plan.children()[..] else {
            return plan_err!(
                "Map-only operator {} requires exactly one input",
                plan.name()
            );
        };
        let mut node = Vec::new();
        codec.try_encode(Arc::clone(plan), &mut node, self)?;
        Ok(extension_proto(
            node,
            vec![self.execution_plan_to_proto(child, codec)?],
        ))
    }

    fn proto_to_physical_expr(
        &self,
        proto: &PhysicalExprNode,
        schema: &Schema,
        ctx: &PhysicalPlanDecodeContext<'_>,
    ) -> Result<Arc<dyn PhysicalExpr>> {
        self.default_proto_to_physical_expr(proto, schema, ctx)
    }

    fn physical_expr_to_proto(
        &self,
        expr: &Arc<dyn PhysicalExpr>,
        codec: &dyn PhysicalExtensionCodec,
    ) -> Result<PhysicalExprNode> {
        serialize_physical_expr_with_converter(expr, codec, self)
    }
}

fn extension_proto(node: Vec<u8>, inputs: Vec<PhysicalPlanNode>) -> PhysicalPlanNode {
    PhysicalPlanNode {
        physical_plan_type: Some(PhysicalPlanType::Extension(PhysicalExtensionNode {
            node,
            inputs,
            plan_name: None,
        })),
    }
}
