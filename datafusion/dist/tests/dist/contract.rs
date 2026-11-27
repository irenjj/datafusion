// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: See the NOTICE file distributed with this work for additional information regarding copyright ownership
// SPDX-FileContributor: Licensed to the Apache Software Foundation (ASF) under one or more contributor license agreements

use std::collections::HashMap;
use std::fs;
use std::sync::Arc;

use datafusion::arrow::{
    datatypes::{DataType, Field, Schema},
    record_batch::RecordBatch,
};
use datafusion::common::extensions::Extensions as FileExtensions;
use datafusion::common::stats::Precision;
use datafusion::common::{Result, ScalarValue, Statistics, TableReference, exec_err};
use datafusion::datasource::{
    listing::PartitionedFile,
    physical_plan::{
        FileGroup, FileScanConfigBuilder, FileSource, ParquetFileReaderFactory,
        ParquetSource,
    },
    source::DataSourceExec,
};
use datafusion::execution::object_store::ObjectStoreUrl;
use datafusion::functions::core::expr_fn::file_row_index;
use datafusion::functions_aggregate::expr_fn::count;
use datafusion::logical_expr::{JoinType, Operator};
use datafusion::parquet::arrow::async_reader::AsyncFileReader;
use datafusion::physical_expr::{
    LexOrdering, Partitioning, PhysicalExpr, PhysicalSortExpr,
    expressions::{BinaryExpr, CastExpr, Column, Literal, in_list},
};
use datafusion::physical_expr_adapter::DefaultPhysicalExprAdapterFactory;
#[expect(deprecated)]
use datafusion::physical_plan::coalesce_batches::CoalesceBatchesExec;
use datafusion::physical_plan::{
    ExecutionPlan, coalesce_partitions::CoalescePartitionsExec, collect,
    empty::EmptyExec, filter::FilterExec, limit::LocalLimitExec,
    metrics::ExecutionPlanMetricsSet, repartition::RepartitionExec,
    sorts::sort_preserving_merge::SortPreservingMergeExec,
};
use datafusion::prelude::{ParquetReadOptions, SessionConfig, SessionContext, col, lit};
use datafusion_dist::{
    DistCodecs, DistPhysicalCodec, ExtensionOperatorPolicy, FileExtensionCodec,
    FileExtensionCodecRegistry, LocalWorker, Split, StagePlan, bind_split_to_plan,
    bind_split_to_plan_with_codecs, bind_split_to_plan_with_extensions,
    bind_split_to_template, bind_split_to_template_with_extensions,
    decode_stage_template, decode_stage_template_with_codecs,
    decode_stage_template_with_extensions, split_plan_for_distribution,
    split_plan_for_distribution_with_codecs, split_plan_for_distribution_with_extensions,
};
use datafusion_proto::bytes::{
    physical_plan_from_bytes_with_extension_codec,
    physical_plan_to_bytes_with_extension_codec,
};
use datafusion_proto::physical_plan::{
    DefaultPhysicalProtoConverter, PhysicalExtensionCodec,
};
use datafusion_proto::protobuf::{
    self, PhysicalPlanNode, physical_plan_node::PhysicalPlanType,
};
use prost::Message;
use sha2::{Digest, Sha256};
use tempfile::tempdir;

use super::harness::{
    MockAiCodec, MockAiExec, MockAiNode, SpeculatableCodec, SpeculatableExec, ai_codecs,
    assert_attempts, assert_independent, assert_matches_single_node, assert_plan_error,
    column_expr, context, coordinator, filtered_plan, map_input, nodes, project,
    register_parquet, rejected, response_plan, rewrite_files, rows, scan_node, scan_plan,
    submitted, write_parquet,
};

#[tokio::test]
async fn file_ranges_preserve_pruning_and_empty_split_output() -> Result<()> {
    let codecs = ai_codecs()?;
    let dir = tempdir()?;
    let path = dir.path().join("wide.parquet");
    let metadata = write_parquet(&path, &[(0, -20), (2, 0), (4, 20), (6, 40)])?;
    assert_eq!(metadata.num_row_groups(), 4);
    let column = metadata.row_group(2).column(0);
    let boundary = column
        .dictionary_page_offset()
        .unwrap_or_else(|| column.data_page_offset());
    let size = fs::metadata(&path)?.len() as i64;
    assert!(boundary > 0 && boundary < size);

    let ctx = coordinator();
    ctx.register_parquet("t", path.to_str().unwrap(), ParquetReadOptions::default())
        .await?;
    let native = filtered_plan(&ctx, None).await?;
    let baseline_schema = native.schema();
    let baseline = collect(Arc::clone(&native), ctx.task_ctx()).await?;
    assert_eq!(
        rows(&baseline)?,
        HashMap::from([
            (
                vec![
                    ScalarValue::Int64(Some(4)),
                    ScalarValue::Utf8(Some("mock[mock-m]: a prompt".into()))
                ],
                1
            ),
            (
                vec![ScalarValue::Int64(Some(5)), ScalarValue::Utf8(None)],
                1
            ),
            (
                vec![
                    ScalarValue::Int64(Some(6)),
                    ScalarValue::Utf8(Some("mock[mock-m]: a prompt".into()))
                ],
                1
            ),
            (
                vec![ScalarValue::Int64(Some(7)), ScalarValue::Utf8(None)],
                1
            ),
        ])
    );
    let (scan, _) = scan_node(native.as_ref());
    let mut statistics = Statistics::new_unknown(scan.file_schema());
    statistics.num_rows = Precision::Exact(8);
    let statistics = Arc::new(statistics);
    let ranged = rewrite_files(native, &|file| {
        assert!(file.range.is_none());
        let file = file.clone().with_statistics(Arc::clone(&statistics));
        vec![
            file.clone().with_range(0, boundary),
            file.clone().with_range(boundary, size),
        ]
    })?;
    let ranged = Arc::new(RepartitionExec::try_new(
        ranged,
        Partitioning::RoundRobinBatch(3),
    )?);
    let (stage, mut splits) =
        split_plan_for_distribution_with_codecs(ranged, 7, &codecs)?;
    assert_eq!(stage.stage_id, 7);
    assert_eq!(splits.len(), 2);
    assert!(splits.iter().all(|split| split.attempt_id == 0));
    splits[0].attempt_id = 3;
    splits[1].attempt_id = 9;

    let workers = [
        LocalWorker::new().with_codecs(codecs.clone()),
        LocalWorker::new().with_codecs(codecs.clone()),
    ];
    assert_independent(&ctx, &workers);
    let template = physical_plan_from_bytes_with_extension_codec(
        &stage.plan_bytes,
        &workers[0].context().task_ctx(),
        &DistPhysicalCodec::new(codecs.physical.clone()),
    )?;
    assert!(scan_node(template.as_ref()).0.file_groups.is_empty());
    let mut outputs = Vec::new();
    for ((worker, split), (start, end)) in workers
        .iter()
        .zip(&splits)
        .zip([(0, boundary), (boundary, size)])
    {
        let bound =
            bind_split_to_plan_with_codecs(&stage, split, worker.context(), &codecs)?;
        assert_eq!(
            bound.properties().output_partitioning().partition_count(),
            1
        );
        assert!(!bound.is::<RepartitionExec>());
        let (scan, parquet) = scan_node(bound.as_ref());
        assert_eq!(scan.file_groups.len(), 1);
        let [file] = scan.file_groups[0].files() else {
            panic!("expected one file in the split");
        };
        let range = file.range.as_ref().unwrap();
        assert_eq!((range.start, range.end), (start, end));
        assert!(file.statistics.is_none());
        assert_eq!(file.object_meta.size as i64, size);
        assert_eq!(
            scan.object_store_url,
            scan_node(template.as_ref()).0.object_store_url
        );
        assert!(parquet.table_parquet_options().global.pushdown_filters);
        assert!(
            parquet
                .table_parquet_options()
                .global
                .schema_force_view_types
        );
        let projected = scan.projected_schema()?;
        assert!(projected.field_with_name("unused_wide_col").is_err());
        assert!(
            parquet
                .filter()
                .unwrap()
                .to_string()
                .contains("filter_only_col")
        );
        assert!(scan.statistics().num_rows.get_value().is_none());

        let output = worker.execute_split(&stage, split).await?;
        assert_eq!(
            (output.split_id, output.attempt_id),
            (split.split_id, split.attempt_id)
        );
        outputs.push(output);
    }

    assert_ne!(outputs[0].split_id, outputs[1].split_id);
    assert!(outputs[0].batches.is_empty());
    assert_eq!(outputs[0].metrics.output_rows, 0);
    assert_eq!(outputs[0].metrics.submitted_prompts, 0);
    assert_eq!(outputs[0].metrics.row_groups_pruned_statistics, 2);
    assert_eq!(outputs[1].metrics.output_rows, 4);
    assert_eq!(outputs[1].metrics.submitted_prompts, 2);
    assert_eq!(outputs[1].metrics.row_groups_matched_statistics, 2);
    assert_matches_single_node(&stage, &baseline_schema, &baseline, &outputs)?;
    eprintln!(
        "Range metrics: {:?}",
        outputs.iter().map(|out| &out.metrics).collect::<Vec<_>>()
    );
    Ok(())
}

#[tokio::test]
async fn hive_values_file_metadata_and_nested_prompt_round_trip() -> Result<()> {
    let codecs = ai_codecs()?;
    let dir = tempdir()?;
    for (id, region) in [(10, "east"), (20, "west")] {
        let partition = dir.path().join(format!("part_col={region}"));
        fs::create_dir(&partition)?;
        write_parquet(&partition.join("data.parquet"), &[(id, 20)])?;
    }
    let ctx = coordinator();
    ctx.register_parquet(
        "t",
        dir.path().to_str().unwrap(),
        ParquetReadOptions::default()
            .table_partition_cols(vec![("part_col".into(), DataType::Utf8)]),
    )
    .await?;
    let native = map_input(&ctx, &["id", "part_col", "prompt_col"], None).await?;
    let join = |left: Arc<dyn PhysicalExpr>, right: Arc<dyn PhysicalExpr>| {
        Arc::new(BinaryExpr::new(left, Operator::StringConcat, right))
    };
    let part_label = join(
        column_expr("part_col", native.as_ref())?,
        Arc::new(Literal::new(ScalarValue::from(":"))),
    );
    let prompt = join(
        part_label,
        Arc::new(CastExpr::new(
            column_expr("prompt_col", native.as_ref())?,
            DataType::Utf8,
            None,
        )),
    );
    let native = project(
        Arc::new(MockAiExec::new(
            native,
            prompt,
            "hive-m".into(),
            r#"{"temperature":0}"#.into(),
            "resp".into(),
        )),
        ["id", "part_col", "resp"],
    )?;
    let baseline_schema = native.schema();
    let baseline = collect(Arc::clone(&native), ctx.task_ctx()).await?;
    let (native_scan, _) = scan_node(native.as_ref());
    let file_schema = native_scan.file_schema().clone();
    let table_schema = native_scan.file_source().table_schema().table_schema();
    let part_col = Arc::new(Column::new("part_col", table_schema.index_of("part_col")?));
    // IN consults the input schema during decoding, so a file-only schema fails here.
    let part_is_east = in_list(
        part_col.clone(),
        vec![Arc::new(Literal::new(ScalarValue::from("east")))],
        &false,
        table_schema,
    )?;
    let ordering = LexOrdering::new([
        PhysicalSortExpr::new_default(part_col),
        PhysicalSortExpr::new_default(part_is_east),
    ]);
    let with_metadata = rewrite_files(native, &|file| {
        let mut file = file
            .clone()
            .with_statistics(Arc::new(Statistics::new_unknown(&file_schema)))
            .with_table_reference(Some(TableReference::bare("t")));
        file.object_meta.e_tag = Some("tag".into());
        file.object_meta.version = Some("version".into());
        file.metadata_size_hint = Some(2048);
        file.ordering = ordering.clone();
        vec![file]
    })?;
    let (stage, splits) =
        split_plan_for_distribution_with_codecs(Arc::clone(&with_metadata), 8, &codecs)?;
    let (source, _) = scan_node(with_metadata.as_ref());
    assert_eq!(splits.len(), 2);
    let workers = [
        LocalWorker::with_context(SessionContext::new())
            .with_physical_codecs(codecs.physical.clone()),
        LocalWorker::new().with_codecs(codecs.clone()),
    ];
    assert_independent(&ctx, &workers);
    let mut outputs = Vec::new();
    for ((split, worker), expected_file) in splits
        .iter()
        .zip(&workers)
        .zip(source.file_groups.iter().flat_map(FileGroup::files))
    {
        let bound =
            bind_split_to_plan_with_codecs(&stage, split, worker.context(), &codecs)?;
        let node = bound.as_ref();
        let ai = nodes(node)
            .find_map(|op| op.downcast_ref::<MockAiExec>())
            .expect("expected a mock AI operator");
        assert_eq!(ai.model, "hive-m");
        assert_eq!(ai.options_json, r#"{"temperature":0}"#);
        assert_eq!(ai.output_col, "resp");
        let (scan, _) = scan_node(bound.as_ref());
        assert_eq!(scan.table_partition_cols(), source.table_partition_cols());
        let file = &scan.file_groups[0].files()[0];
        assert_eq!(file.partition_values, expected_file.partition_values);
        assert_eq!(file.object_meta, expected_file.object_meta);
        assert_eq!(file.statistics, expected_file.statistics);
        assert_eq!(file.metadata_size_hint, expected_file.metadata_size_hint);
        assert_eq!(file.table_reference, expected_file.table_reference);
        assert_eq!(file.ordering, expected_file.ordering);
        assert_eq!(file.ordering, ordering);
        outputs.push(worker.execute_split(&stage, split).await?);
    }
    assert_matches_single_node(&stage, &baseline_schema, &baseline, &outputs)?;
    Ok(())
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct LanceFragment(u64);

#[derive(Clone, Debug, PartialEq, Eq)]
struct PaimonSplit(String);

#[derive(Debug)]
struct LanceCodec(&'static str);

impl FileExtensionCodec for LanceCodec {
    fn extension_name(&self) -> &str {
        self.0
    }

    fn try_encode(&self, extensions: &FileExtensions) -> Result<Option<(u32, Vec<u8>)>> {
        Ok(extensions
            .get::<LanceFragment>()
            .map(|fragment| (1, fragment.0.to_le_bytes().to_vec())))
    }

    fn try_decode(
        &self,
        version: u32,
        payload: &[u8],
        extensions: &mut FileExtensions,
    ) -> Result<()> {
        if version != 1 {
            return exec_err!("unsupported Lance version {version}");
        }
        let Ok(fragment) = payload.try_into() else {
            return exec_err!("invalid Lance fragment ID");
        };
        extensions.insert(LanceFragment(u64::from_le_bytes(fragment)));
        Ok(())
    }
}

#[derive(Debug)]
struct PaimonCodec;

impl FileExtensionCodec for PaimonCodec {
    fn extension_name(&self) -> &str {
        "paimon.split"
    }

    fn try_encode(&self, extensions: &FileExtensions) -> Result<Option<(u32, Vec<u8>)>> {
        Ok(extensions
            .get::<PaimonSplit>()
            .map(|split| (2, split.0.as_bytes().to_vec())))
    }

    fn try_decode(
        &self,
        version: u32,
        payload: &[u8],
        extensions: &mut FileExtensions,
    ) -> Result<()> {
        if version != 2 {
            return exec_err!("unsupported Paimon version {version}");
        }
        let Ok(split) = str::from_utf8(payload) else {
            return exec_err!("invalid Paimon split");
        };
        extensions.insert(PaimonSplit(split.to_owned()));
        Ok(())
    }
}

#[derive(Debug)]
struct NoOpCodec;

impl FileExtensionCodec for NoOpCodec {
    fn extension_name(&self) -> &str {
        "noop"
    }

    fn try_encode(&self, _: &FileExtensions) -> Result<Option<(u32, Vec<u8>)>> {
        Ok(None)
    }

    fn try_decode(&self, _: u32, _: &[u8], _: &mut FileExtensions) -> Result<()> {
        Ok(())
    }
}

#[derive(Debug)]
struct MultipleFileTypesCodec;

impl FileExtensionCodec for MultipleFileTypesCodec {
    fn extension_name(&self) -> &str {
        "multiple"
    }

    fn try_encode(&self, _extensions: &FileExtensions) -> Result<Option<(u32, Vec<u8>)>> {
        Ok(None)
    }

    fn try_decode(
        &self,
        _version: u32,
        _payload: &[u8],
        extensions: &mut FileExtensions,
    ) -> Result<()> {
        extensions.insert(LanceFragment(1));
        extensions.insert(PaimonSplit("another type".into()));
        Ok(())
    }
}

#[derive(Clone, PartialEq, Message)]
struct WireExtension {
    #[prost(string, tag = "1")]
    name: String,
    #[prost(uint32, tag = "2")]
    version: u32,
    #[prost(bytes = "vec", tag = "3")]
    payload: Vec<u8>,
}

#[derive(Clone, PartialEq, Message)]
struct WireExtensions {
    #[prost(message, repeated, tag = "7")]
    entries: Vec<WireExtension>,
}

#[derive(Clone, PartialEq, Message)]
struct LegacySplitEnvelope {
    #[prost(message, optional, tag = "1")]
    file: Option<protobuf::PartitionedFile>,
    #[prost(string, optional, tag = "2")]
    e_tag: Option<String>,
}

#[tokio::test]
async fn file_extensions_round_trip_and_encode_deterministically() -> Result<()> {
    let dir = tempdir()?;
    let path = dir.path().join("data.parquet");
    write_parquet(&path, &[(1, 20), (2, 40)])?;
    let ctx = coordinator();
    register_parquet(&ctx, &path, false).await?;
    let native = map_input(&ctx, &["id"], Some(10)).await?;

    let fragment = LanceFragment(0x0123_4567_89ab_cdef);
    let paimon = PaimonSplit("split-🍃".into());
    let make_plan = |include_paimon, reverse_inserts| {
        rewrite_files(Arc::clone(&native), &|file| {
            let mut file = file.clone();
            if reverse_inserts && include_paimon {
                file.extensions.insert(paimon.clone());
            }
            file.extensions.insert(fragment);
            if !reverse_inserts && include_paimon {
                file.extensions.insert(paimon.clone());
            }
            vec![file]
        })
    };

    let mut lance_only = FileExtensionCodecRegistry::default();
    lance_only.register(Arc::new(LanceCodec("lance.fragment")))?;
    let mut codecs = lance_only.clone();
    codecs.register(Arc::new(PaimonCodec))?;
    let mut reverse_codecs = FileExtensionCodecRegistry::default();
    reverse_codecs.register(Arc::new(PaimonCodec))?;
    reverse_codecs.register(Arc::new(LanceCodec("lance.fragment")))?;
    let worker = LocalWorker::new().with_extension_codecs(codecs.clone());

    for include_paimon in [false, true] {
        let plan = make_plan(include_paimon, false)?;
        let baseline_schema = plan.schema();
        let baseline = collect(Arc::clone(&plan), ctx.task_ctx()).await?;
        let (stage, splits) =
            split_plan_for_distribution_with_extensions(plan, 30, &codecs)?;
        assert_eq!(splits.len(), 1);
        let split = &splits[0];
        let prepared =
            decode_stage_template_with_extensions(&stage, worker.context(), &codecs)?;
        let worker_prepared = worker.prepare_stage(&stage)?;
        let legacy_prepared = decode_stage_template(&stage, worker.context())?;
        for bound in [
            bind_split_to_plan_with_extensions(&stage, split, worker.context(), &codecs)?,
            bind_split_to_template(&prepared, split)?,
            bind_split_to_template(&worker_prepared, split)?,
            bind_split_to_template_with_extensions(&legacy_prepared, split, &codecs)?,
        ] {
            let (scan, _) = scan_node(bound.as_ref());
            let extensions = &scan.file_groups[0].files()[0].extensions;
            assert_eq!(extensions.len(), 1 + usize::from(include_paimon));
            assert_eq!(extensions.get::<LanceFragment>(), Some(&fragment));
            assert_eq!(
                extensions.get::<PaimonSplit>(),
                include_paimon.then_some(&paimon)
            );
        }
        assert_plan_error(
            &bind_split_to_template(&legacy_prepared, split).unwrap_err(),
            "Unregistered split file extension 'lance.fragment'",
        );
        let output = worker_prepared.execute_split(split).await?;
        assert_matches_single_node(&stage, &baseline_schema, &baseline, &[output])?;

        let entries = WireExtensions::decode(split.file_proto.as_slice())
            .unwrap()
            .entries;
        let names: Vec<_> = entries.iter().map(|entry| entry.name.as_str()).collect();
        if include_paimon {
            assert_eq!(names, ["lance.fragment", "paimon.split"]);
            for (reverse_inserts, registry) in [
                (false, &reverse_codecs),
                (true, &codecs),
                (true, &reverse_codecs),
            ] {
                let (_, alternate) = split_plan_for_distribution_with_extensions(
                    make_plan(true, reverse_inserts)?,
                    30,
                    registry,
                )?;
                assert_eq!(alternate[0].file_proto, split.file_proto);
            }
        } else {
            assert_eq!(names, ["lance.fragment"]);
        }
    }
    assert_plan_error(
        &split_plan_for_distribution_with_extensions(
            make_plan(true, false)?,
            30,
            &lance_only,
        )
        .unwrap_err(),
        "PartitionedFile.extensions: encoded 1 of 2",
    );
    Ok(())
}

#[tokio::test]
async fn legacy_split_and_invalid_extension_wire_entries() -> Result<()> {
    let dir = tempdir()?;
    let path = dir.path().join("data.parquet");
    write_parquet(&path, &[(1, 20)])?;
    let ctx = coordinator();
    register_parquet(&ctx, &path, false).await?;
    let native = map_input(&ctx, &["id"], Some(10)).await?;
    let (scan, _) = scan_node(native.as_ref());
    let file = &scan.file_groups[0].files()[0];
    let legacy_bytes = LegacySplitEnvelope {
        file: Some(file.try_into()?),
        e_tag: Some("legacy-tag".into()),
    }
    .encode_to_vec();
    let (stage, mut splits) = split_plan_for_distribution(native, 31)?;
    assert_eq!(splits.len(), 1);
    let split = &mut splits[0];
    set_file_bytes(&stage, split, legacy_bytes.clone());

    let mut codecs = FileExtensionCodecRegistry::default();
    codecs.register(Arc::new(LanceCodec("lance.fragment")))?;
    codecs.register(Arc::new(LanceCodec("lance.alias")))?;
    codecs.register(Arc::new(NoOpCodec))?;
    codecs.register(Arc::new(MultipleFileTypesCodec))?;
    let prepared = decode_stage_template_with_extensions(&stage, &ctx, &codecs)?;
    for bound in [
        bind_split_to_plan(&stage, split, &ctx)?,
        bind_split_to_template(&prepared, split)?,
    ] {
        let (scan, _) = scan_node(bound.as_ref());
        let file = &scan.file_groups[0].files()[0];
        assert_eq!(file.object_meta.e_tag.as_deref(), Some("legacy-tag"));
        assert!(file.extensions.is_empty());
    }

    let lance = WireExtension {
        name: "lance.fragment".into(),
        version: 1,
        payload: 42_u64.to_le_bytes().to_vec(),
    };
    for (entries, expected) in [
        (
            vec![WireExtension {
                name: "unknown".into(),
                ..lance.clone()
            }],
            "Unregistered split file extension 'unknown'",
        ),
        (
            vec![lance.clone(), lance.clone()],
            "Duplicate split file extension 'lance.fragment'",
        ),
        (
            vec![WireExtension {
                name: String::new(),
                ..lance.clone()
            }],
            "extension name must not be empty",
        ),
        (
            vec![WireExtension {
                name: " \t".into(),
                ..lance.clone()
            }],
            "extension name must not be empty",
        ),
        (
            vec![WireExtension {
                version: 9,
                ..lance.clone()
            }],
            "unsupported Lance version 9",
        ),
        (
            vec![WireExtension {
                payload: vec![0],
                ..lance.clone()
            }],
            "invalid Lance fragment ID",
        ),
        (
            vec![WireExtension {
                name: "noop".into(),
                ..lance.clone()
            }],
            "must decode exactly one value",
        ),
        (
            vec![WireExtension {
                name: "multiple".into(),
                ..lance.clone()
            }],
            "must decode exactly one value",
        ),
        (
            vec![
                lance.clone(),
                WireExtension {
                    name: "lance.alias".into(),
                    ..lance.clone()
                },
            ],
            "repeats a decoded Rust type",
        ),
    ] {
        set_file_bytes(&stage, split, legacy_bytes.clone());
        WireExtensions { entries }
            .encode(&mut split.file_proto)
            .unwrap();
        split.fingerprint = split_hash(&stage, &split.file_proto);
        assert_plan_error(
            &bind_split_to_template(&prepared, split).unwrap_err(),
            expected,
        );
    }
    Ok(())
}

#[test]
fn rejects_blank_and_duplicate_file_extension_codec_names() -> Result<()> {
    let mut codecs = FileExtensionCodecRegistry::default();
    for name in ["", " \t"] {
        assert_plan_error(
            &codecs.register(Arc::new(LanceCodec(name))).unwrap_err(),
            "name must not be empty",
        );
    }
    codecs.register(Arc::new(LanceCodec("lance.fragment")))?;
    assert_plan_error(
        &codecs
            .register(Arc::new(LanceCodec("lance.fragment")))
            .unwrap_err(),
        "already registered",
    );
    Ok(())
}

#[derive(Debug)]
struct CustomParquetReader;

impl ParquetFileReaderFactory for CustomParquetReader {
    fn create_reader(
        &self,
        _partition_index: usize,
        _file: PartitionedFile,
        _metadata_size_hint: Option<usize>,
        _metrics: &ExecutionPlanMetricsSet,
    ) -> Result<Box<dyn AsyncFileReader + Send>> {
        exec_err!("the reader must be rejected before execution")
    }
}

#[tokio::test]
async fn rejects_non_map_only_operators_and_file_extensions() -> Result<()> {
    let dir = tempdir()?;
    let ctx = coordinator();
    for (table, id) in [("t", 0), ("s", 2)] {
        let path = dir.path().join(format!("{table}.parquet"));
        write_parquet(&path, &[(id, 20)])?;
        ctx.register_parquet(
            table,
            path.to_str().unwrap(),
            ParquetReadOptions::default(),
        )
        .await?;
    }
    let t = ctx.table("t").await?;
    let other = ctx.table("s").await?;
    for (df, operator) in [
        (
            t.clone().sort(vec![col("id").sort(true, false)])?,
            "SortExec",
        ),
        (
            t.clone().aggregate(vec![col("id")], vec![count(lit(1))])?,
            "AggregateExec",
        ),
        (t.clone().limit(0, Some(1))?, "limit"),
        (
            t.clone()
                .join(other.clone(), JoinType::Inner, &["id"], &["id"], None)?,
            "HashJoinExec",
        ),
        (t.clone().union(other)?, "UnionExec"),
    ] {
        rejected(df.create_physical_plan().await?, operator);
    }
    let virtual_scan = t
        .select(vec![file_row_index(), col("id")])?
        .create_physical_plan()
        .await?;
    let (_, parquet) = scan_node(virtual_scan.as_ref());
    assert!(!parquet.table_schema().virtual_columns().is_empty());
    rejected(virtual_scan, "virtual columns");

    let scan = scan_plan(&ctx).await?;
    let (config, parquet) = scan_node(scan.as_ref());
    let limited_scan = DataSourceExec::from_data_source(
        FileScanConfigBuilder::from(config.clone())
            .with_limit(Some(1))
            .build(),
    );
    rejected(limited_scan, "FileScanConfig.limit");
    let mut options = parquet.table_parquet_options().clone();
    options.crypto.factory_id = Some("encrypted-parquet".into());
    let encrypted_scan = DataSourceExec::from_data_source(
        FileScanConfigBuilder::from(config.clone())
            .with_source(Arc::new(
                parquet.clone().with_table_parquet_options(options),
            ))
            .build(),
    );
    rejected(encrypted_scan, "Parquet encryption");
    rejected(
        DataSourceExec::from_data_source(
            FileScanConfigBuilder::from(config.clone())
                .with_source(Arc::new(
                    parquet
                        .clone()
                        .with_parquet_file_reader_factory(Arc::new(CustomParquetReader)),
                ))
                .build(),
        ),
        "custom Parquet reader factories",
    );
    let ordering = LexOrdering::new([PhysicalSortExpr::new_default(Arc::new(
        Column::new("id", 0),
    ))])
    .unwrap();
    for builder in [
        FileScanConfigBuilder::from(config.clone()).with_preserve_order(true),
        FileScanConfigBuilder::from(config.clone())
            .with_output_ordering(vec![ordering.clone()]),
        FileScanConfigBuilder::from(config.clone())
            .with_output_partitioning(Some(Partitioning::UnknownPartitioning(1))),
        FileScanConfigBuilder::from(config.clone())
            .with_expr_adapter(Some(Arc::new(DefaultPhysicalExprAdapterFactory))),
    ] {
        rejected(
            DataSourceExec::from_data_source(builder.build()),
            "ordered, explicitly partitioned, or custom-adapter",
        );
    }
    let ordered = DataSourceExec::from_data_source(
        FileScanConfigBuilder::from(config.clone())
            .with_file_groups(vec![config.file_groups[0].clone(); 2])
            .with_output_ordering(vec![ordering.clone()])
            .build(),
    );
    let preserving = RepartitionExec::try_new(ordered, Partitioning::RoundRobinBatch(2))?
        .with_preserve_order();
    assert!(preserving.preserve_order());
    rejected(Arc::new(preserving), "RepartitionExec partitioning");
    rejected(
        Arc::new(SortPreservingMergeExec::new(ordering, Arc::clone(&scan))),
        "input ordering or distribution",
    );
    let filter = Arc::new(FilterExec::try_new(
        Arc::new(Literal::new(ScalarValue::Boolean(Some(true)))),
        Arc::clone(&scan),
    )?);
    rejected(
        filter.with_fetch(Some(1)).unwrap(),
        "FilterExec with a fetch limit",
    );

    rejected(
        Arc::new(LocalLimitExec::new(Arc::clone(&scan), 1)),
        "LocalLimitExec",
    );
    rejected(
        Arc::new(CoalescePartitionsExec::new(Arc::clone(&scan))),
        "CoalescePartitionsExec",
    );
    rejected(
        Arc::new(RepartitionExec::try_new(
            Arc::clone(&scan),
            Partitioning::Hash(vec![Arc::new(Column::new("id", 0))], 2),
        )?),
        "RepartitionExec",
    );
    rejected(
        rewrite_files(scan, &|file| {
            vec![file.clone().with_extension("external metadata")]
        })?,
        "PartitionedFile.extensions",
    );
    Ok(())
}

#[cfg(feature = "parquet_encryption")]
#[test]
fn rejects_direct_parquet_encryption_factory() {
    use datafusion::arrow::datatypes::SchemaRef;
    use datafusion::common::config::EncryptionFactoryOptions;
    use datafusion::execution::parquet_encryption::EncryptionFactory;
    use datafusion::object_store::path::Path;
    use datafusion::parquet::encryption::{
        decrypt::FileDecryptionProperties, encrypt::FileEncryptionProperties,
    };
    use futures::future::BoxFuture;

    #[derive(Debug)]
    struct TestEncryptionFactory;

    impl EncryptionFactory for TestEncryptionFactory {
        fn get_file_encryption_properties<'factory, 'config, 'schema, 'path, 'future>(
            &'factory self,
            _config: &'config EncryptionFactoryOptions,
            _schema: &'schema SchemaRef,
            _path: &'path Path,
        ) -> BoxFuture<'future, Result<Option<Arc<FileEncryptionProperties>>>>
        where
            'factory: 'future,
            'config: 'future,
            'schema: 'future,
            'path: 'future,
            Self: 'future,
        {
            unreachable!("plan admission must not call the encryption factory")
        }

        fn get_file_decryption_properties<'factory, 'config, 'path, 'future>(
            &'factory self,
            _config: &'config EncryptionFactoryOptions,
            _path: &'path Path,
        ) -> BoxFuture<'future, Result<Option<Arc<FileDecryptionProperties>>>>
        where
            'factory: 'future,
            'config: 'future,
            'path: 'future,
            Self: 'future,
        {
            unreachable!("plan admission must not call the encryption factory")
        }
    }

    let source = ParquetSource::new(Arc::new(Schema::empty()))
        .with_encryption_factory(Arc::new(TestEncryptionFactory));
    assert_eq!(source.table_parquet_options().crypto, Default::default());
    let plan = DataSourceExec::from_data_source(
        FileScanConfigBuilder::new(ObjectStoreUrl::local_filesystem(), Arc::new(source))
            .with_file_groups(vec![FileGroup::new(vec![PartitionedFile::new(
                "encrypted.parquet",
                1024,
            )])])
            .build(),
    );
    rejected(plan, "Parquet encryption");
}

#[tokio::test]
async fn rejects_invalid_split_descriptors_and_nonempty_templates() -> Result<()> {
    let dir = tempdir()?;
    let partition = dir.path().join("part_col=east");
    fs::create_dir(&partition)?;
    write_parquet(&partition.join("data.parquet"), &[(0, 20)])?;
    let ctx = coordinator();
    ctx.register_parquet(
        "t",
        dir.path().to_str().unwrap(),
        ParquetReadOptions::default()
            .table_partition_cols(vec![("part_col".into(), DataType::Utf8)]),
    )
    .await?;
    let native = scan_plan(&ctx).await?;
    let (scan, _) = scan_node(native.as_ref());
    let file = &scan.file_groups[0].files()[0];
    let (stage, splits) = split_plan_for_distribution(Arc::clone(&native), 11)?;
    let worker = LocalWorker::new();
    let prepared = worker.prepare_stage(&stage)?;

    for (file_proto, expected) in [
        (vec![0x0a, 0x02, 0x01], "Invalid split protobuf"),
        (vec![], "missing its file"),
    ] {
        let mut split = splits[0].clone();
        set_file_bytes(&stage, &mut split, file_proto);
        assert_plan_error(
            &bind_split_to_template(&prepared, &split).unwrap_err(),
            expected,
        );
    }
    let mut invalid_stats = file.clone();
    invalid_stats.statistics = Some(Arc::new(Statistics::new_unknown(&Schema::empty())));
    for (file, expected) in [
        (
            file.clone().with_range(0, file.object_meta.size as i64 + 1),
            "byte range",
        ),
        (file.clone().with_range(10, 10), "byte range"),
        (file.clone().with_range(-1, 10), "byte range"),
        (
            file.clone().with_partition_values(vec![]),
            "Hive partition values",
        ),
        (
            file.clone()
                .with_partition_values(vec![ScalarValue::Int64(Some(1))]),
            "Hive partition values",
        ),
        (invalid_stats, "statistics do not match"),
    ] {
        let malformed = rewrite_files(Arc::clone(&native), &|_| vec![file.clone()])?;
        let (_, malformed_splits) =
            split_plan_for_distribution(malformed, stage.stage_id)?;
        assert_plan_error(
            &bind_split_to_template(&prepared, &malformed_splits[0]).unwrap_err(),
            expected,
        );
    }

    let mut nonempty = StagePlan {
        plan_bytes: physical_plan_to_bytes_with_extension_codec(
            native,
            &DistPhysicalCodec::default(),
        )?
        .to_vec(),
        ..stage
    };
    nonempty.plan_fingerprint = plan_hash(&nonempty)?;
    assert_plan_error(
        &worker.prepare_stage(&nonempty).unwrap_err(),
        "must have an empty Parquet scan",
    );
    Ok(())
}

#[tokio::test]
#[expect(deprecated)]
async fn empty_file_groups_round_trip_with_no_splits() -> Result<()> {
    let codecs = ai_codecs()?;
    let dir = tempdir()?;
    let path = dir.path().join("data.parquet");
    write_parquet(&path, &[(0, 20)])?;
    let ctx = coordinator();
    ctx.register_parquet("t", path.to_str().unwrap(), ParquetReadOptions::default())
        .await?;
    let native = Arc::new(CoalesceBatchesExec::new(
        filtered_plan(&ctx, None).await?,
        32,
    ));
    let output_schema = native.schema();
    let empty = rewrite_files(native, &|_| vec![])?;
    let (stage, splits) = split_plan_for_distribution_with_codecs(empty, 12, &codecs)?;
    assert!(splits.is_empty());
    assert!(!stage.plan_bytes.is_empty());
    assert_eq!(stage.output_schema, output_schema);
    let worker = LocalWorker::new().with_codecs(codecs.clone());
    worker.prepare_stage(&stage)?;
    let template = physical_plan_from_bytes_with_extension_codec(
        &stage.plan_bytes,
        &worker.context().task_ctx(),
        &DistPhysicalCodec::new(codecs.physical.clone()),
    )?;
    assert!(template.is::<CoalesceBatchesExec>());
    assert!(scan_node(template.as_ref()).0.file_groups.is_empty());
    Ok(())
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn naturally_planned_parallel_parquet_ranges_match_single_node() -> Result<()> {
    let codecs = ai_codecs()?;
    let dir = tempdir()?;
    let path = dir.path().join("data.parquet");
    let groups: Vec<_> = (0..16)
        .map(|i| (2 * i, if i % 2 == 0 { -20 + i } else { 20 + i }))
        .collect();
    write_parquet(&path, &groups)?;
    let size = fs::metadata(&path)?.len() as i64;

    let ctx = context(8192, 4, true, true);
    ctx.register_parquet("t", path.to_str().unwrap(), ParquetReadOptions::default())
        .await?;
    let native = filtered_plan(&ctx, None).await?;
    let (scan, _) = scan_node(native.as_ref());
    assert_eq!(scan.file_groups.len(), 4);
    let mut ranges: Vec<_> = scan
        .file_groups
        .iter()
        .flat_map(FileGroup::files)
        .map(|file| {
            let range = file
                .range
                .as_ref()
                .expect("DataFusion should plan a byte range");
            (range.start, range.end)
        })
        .collect();
    assert_eq!(ranges.len(), 4);
    ranges.sort_unstable();
    assert_eq!(ranges[0].0, 0);
    assert_eq!(ranges.last().unwrap().1, size);
    assert!(ranges.windows(2).all(|pair| pair[0].1 == pair[1].0));

    let baseline_schema = native.schema();
    let baseline = collect(Arc::clone(&native), ctx.task_ctx()).await?;
    assert_eq!(
        baseline.iter().map(RecordBatch::num_rows).sum::<usize>(),
        16
    );
    let (stage, splits) = split_plan_for_distribution_with_codecs(native, 13, &codecs)?;
    assert_eq!(splits.len(), ranges.len());
    let workers = [
        LocalWorker::new().with_codecs(codecs.clone()),
        LocalWorker::new().with_codecs(codecs.clone()),
    ];
    assert_independent(&ctx, &workers);
    let first = workers[0].prepare_stage(&stage)?;
    let second = workers[1].prepare_stage(&stage)?;
    let mut outputs = Vec::new();
    let (pairs, remainder) = splits.as_chunks::<2>();
    assert!(remainder.is_empty());
    for pair in pairs {
        let (a, b) = tokio::try_join!(
            first.execute_split(&pair[0]),
            second.execute_split(&pair[1])
        )?;
        assert_eq!(a.split_id, pair[0].split_id);
        assert_eq!(b.split_id, pair[1].split_id);
        outputs.extend([a, b]);
    }
    assert_matches_single_node(&stage, &baseline_schema, &baseline, &outputs)?;
    assert_eq!(
        outputs
            .iter()
            .map(|o| o.metrics.submitted_prompts)
            .sum::<usize>(),
        8
    );
    eprintln!(
        "Native DataFusion plan: {} contiguous byte ranges over {size} bytes",
        ranges.len()
    );
    Ok(())
}

#[tokio::test]
async fn default_parquet_pushdown_keeps_filters_on_both_sides_of_mock_ai() -> Result<()> {
    let codecs = ai_codecs()?;
    let dir = tempdir()?;
    for id in [0, 4] {
        write_parquet(
            &dir.path().join(format!("{id}.parquet")),
            &[(id, 0), (id + 2, 20)],
        )?;
    }
    let ctx =
        SessionContext::new_with_config(SessionConfig::new().with_target_partitions(1));
    register_parquet(&ctx, dir.path(), false).await?;
    let native = filtered_plan(&ctx, Some("mock%")).await?;
    let schema = native.schema();
    let baseline = collect(Arc::clone(&native), ctx.task_ctx()).await?;
    assert_eq!(submitted(native.as_ref()), 2);
    let (stage, splits) = split_plan_for_distribution_with_codecs(native, 14, &codecs)?;
    assert_eq!(splits.len(), 2);
    let worker = LocalWorker::new().with_codecs(codecs.clone());
    let prepared = worker.prepare_stage(&stage)?;
    let mut outputs = Vec::new();
    for split in &splits {
        let bound = bind_split_to_template(&prepared, split)?;
        let operators = nodes(bound.as_ref())
            .filter(|node| node.is::<MockAiExec>() || node.is::<FilterExec>())
            .map(ExecutionPlan::name)
            .collect::<Vec<_>>();
        assert_eq!(operators, ["FilterExec", "MockAiExec", "FilterExec"]);
        let (_, parquet) = scan_node(bound.as_ref());
        assert!(!parquet.table_parquet_options().global.pushdown_filters);
        let output = prepared.execute_split(split).await?;
        assert_eq!(output.metrics.output_rows, 1);
        assert_eq!(output.metrics.submitted_prompts, 1);
        outputs.push(output);
    }
    assert_attempts(&splits, &outputs);
    assert_matches_single_node(&stage, &schema, &baseline, &outputs)
}

#[tokio::test]
async fn nested_mock_ai_counts_submissions_from_both_layers() -> Result<()> {
    let codecs = ai_codecs()?;
    let dir = tempdir()?;
    for id in [0, 4] {
        write_parquet(
            &dir.path().join(format!("{id}.parquet")),
            &[(id, 20), (id + 2, 40)],
        )?;
    }
    let ctx = coordinator();
    register_parquet(&ctx, dir.path(), false).await?;
    let input = map_input(&ctx, &["id", "prompt_col"], None).await?;
    let native = response_plan(input, &["id"], true, None)?;
    let schema = native.schema();
    let baseline = collect(Arc::clone(&native), ctx.task_ctx()).await?;
    assert_eq!(submitted(native.as_ref()), 8);
    let (stage, splits) = split_plan_for_distribution_with_codecs(native, 15, &codecs)?;
    let workers = [
        LocalWorker::new().with_codecs(codecs.clone()),
        LocalWorker::new().with_codecs(codecs.clone()),
    ];
    assert_independent(&ctx, &workers);
    assert_eq!(splits.len(), workers.len());
    let mut outputs = Vec::new();
    for (split, worker) in splits.iter().zip(&workers) {
        let prepared = worker.prepare_stage(&stage)?;
        let bound = bind_split_to_template(&prepared, split)?;
        assert_eq!(
            nodes(bound.as_ref())
                .filter(|node| node.is::<MockAiExec>())
                .count(),
            2
        );
        let output = prepared.execute_split(split).await?;
        assert_eq!(output.metrics.submitted_prompts, 4);
        outputs.push(output);
    }
    assert_attempts(&splits, &outputs);
    assert_matches_single_node(&stage, &schema, &baseline, &outputs)
}

#[tokio::test]
async fn reused_prepared_stage_isolates_scan_filter_and_mock_ai_metrics() -> Result<()> {
    let codecs = ai_codecs()?;
    let dir = tempdir()?;
    write_parquet(&dir.path().join("a.parquet"), &[(0, 0), (2, 20)])?;
    write_parquet(&dir.path().join("b.parquet"), &[(4, 0), (6, 20), (8, 40)])?;
    let ctx = context(2, 1, false, false);
    register_parquet(&ctx, dir.path(), false).await?;
    let native = filtered_plan(&ctx, Some("mock%")).await?;
    let schema = native.schema();
    let baseline = collect(Arc::clone(&native), ctx.task_ctx()).await?;
    let (stage, splits) = split_plan_for_distribution_with_codecs(native, 16, &codecs)?;
    let worker = LocalWorker::new().with_codecs(codecs.clone());
    let prepared = worker.prepare_stage(&stage)?;
    let filter_rows = |filter: &dyn ExecutionPlan| {
        filter.metrics().and_then(|metrics| metrics.output_rows())
    };
    let mut outputs = Vec::new();
    for split in &splits {
        let bound = bind_split_to_template(&prepared, split)?;
        let (scan, _) = scan_node(bound.as_ref());
        let location = &scan.file_groups[0].files()[0].object_meta.location;
        let matched: usize = if location.as_ref().ends_with("a.parquet") {
            1
        } else {
            2
        };
        let filters = nodes(bound.as_ref())
            .filter(|node| node.is::<FilterExec>())
            .collect::<Vec<_>>();
        assert_eq!(filters.len(), 2);
        assert!(filters.iter().all(|filter| filter_rows(*filter).is_none()));
        assert_eq!(submitted(bound.as_ref()), 0);
        collect(Arc::clone(&bound), worker.context().task_ctx()).await?;
        assert_eq!(filter_rows(filters[0]), Some(matched));
        assert_eq!(filter_rows(filters[1]), Some(2 * matched));
        assert_eq!(submitted(bound.as_ref()), matched);
        let output = prepared.execute_split(split).await?;
        assert_eq!(output.metrics.output_rows, matched);
        assert_eq!(output.metrics.submitted_prompts, matched);
        assert_eq!(output.metrics.row_groups_pruned_statistics, 1);
        assert_eq!(output.metrics.row_groups_matched_statistics, matched);
        outputs.push(output);
    }
    assert_eq!(outputs.len(), 2);
    assert_attempts(&splits, &outputs);
    assert_matches_single_node(&stage, &schema, &baseline, &outputs)
}

fn plan_hash(stage: &StagePlan) -> Result<String> {
    let schema =
        protobuf::Schema::try_from(stage.output_schema.as_ref())?.encode_to_vec();
    let mut hash = Sha256::new();
    hash.update(b"df.dist.plan.v1\0");
    hash.update((stage.plan_bytes.len() as u64).to_le_bytes());
    hash.update(&stage.plan_bytes);
    hash.update((schema.len() as u64).to_le_bytes());
    hash.update(schema);
    Ok(hex::encode(hash.finalize()))
}

fn split_hash(stage: &StagePlan, file: &[u8]) -> String {
    let mut hash = Sha256::new();
    hash.update(b"df.dist.split.v1\0");
    hash.update(stage.plan_fingerprint.as_bytes());
    hash.update((file.len() as u64).to_le_bytes());
    hash.update(file);
    hex::encode(hash.finalize())
}

fn set_file_bytes(stage: &StagePlan, split: &mut Split, file: Vec<u8>) {
    split.fingerprint = split_hash(stage, &file);
    split.file_proto = file;
}

#[derive(Clone, PartialEq, Message)]
struct WireOperator {
    #[prost(string, tag = "1")]
    name: String,
    #[prost(uint32, tag = "2")]
    version: u32,
    #[prost(bytes, tag = "3")]
    payload: Vec<u8>,
}

#[test]
fn physical_codec_validates_operator_envelope_and_config_on_both_sides() -> Result<()> {
    let ctx = SessionContext::new().task_ctx();
    let converter = DefaultPhysicalProtoConverter {};
    let codec = DistPhysicalCodec::new(ai_codecs()?.physical);
    let input: Arc<dyn ExecutionPlan> =
        Arc::new(EmptyExec::new(Arc::new(Schema::empty())));
    let children = std::slice::from_ref(&input);
    let make_exec =
        |model: &str, options: &str, output: &str| -> Arc<dyn ExecutionPlan> {
            Arc::new(MockAiExec::new(
                Arc::clone(&input),
                Arc::new(Literal::new(ScalarValue::from("prompt"))),
                model.into(),
                options.into(),
                output.into(),
            ))
        };
    let options = "{\n  \"z\": {\"b\": 1, \"a\": 2}, \"a\": 3\n}";
    let canonical = r#"{"a":3,"z":{"a":2,"b":1}}"#;
    let mut buf = Vec::new();
    let plan = make_exec("m", options, "response");
    codec.try_encode(Arc::clone(&plan), &mut buf, &converter)?;
    let wire = WireOperator::decode(buf.as_slice()).unwrap();
    assert_eq!(wire.name, "test.mock_ai");
    assert_eq!(wire.version, 1);
    let payload = MockAiNode::decode(wire.payload.as_slice()).unwrap();
    assert_eq!(payload.options_json, canonical);

    let mut noncanonical = payload.clone();
    noncanonical.options_json = options.into();
    let decode = |wire: &WireOperator| {
        codec.try_decode(&wire.encode_to_vec(), children, &ctx, &converter)
    };
    let decoded = decode(&WireOperator {
        payload: noncanonical.encode_to_vec(),
        ..wire.clone()
    })?;
    assert_eq!(
        decoded.downcast_ref::<MockAiExec>().unwrap().options_json,
        canonical
    );
    let mut roundtrip = Vec::new();
    codec.try_encode(decoded, &mut roundtrip, &converter)?;
    assert_eq!(roundtrip, buf);

    for (model, options, output, expected) in [
        (" ", "{}", "response", "requires a model and output column"),
        ("m", "{}", "", "requires a model and output column"),
        ("m", "{", "response", "Invalid mock AI options_json"),
        (
            "m",
            r#"{"a":"first
second"}"#,
            "response",
            "Invalid mock AI options_json",
        ),
        ("m", "[]", "response", "options_json must be an object"),
        (
            "m",
            r#"{"a":[{"To\u006ben":"test"}]}"#,
            "response",
            "sensitive credential key",
        ),
        (
            "m",
            r#"{"a":{"password":"test"},"a":{}}"#,
            "response",
            "sensitive credential key",
        ),
    ] {
        let mut untouched = vec![0xaa];
        assert_plan_error(
            &codec
                .try_encode(
                    make_exec(model, options, output),
                    &mut untouched,
                    &converter,
                )
                .unwrap_err(),
            expected,
        );
        assert_eq!(untouched, [0xaa]);
        let invalid = MockAiNode {
            model: model.into(),
            options_json: options.into(),
            output_col: output.into(),
            ..payload.clone()
        };
        assert_plan_error(
            &decode(&WireOperator {
                payload: invalid.encode_to_vec(),
                ..wire.clone()
            })
            .unwrap_err(),
            expected,
        );
    }
    for (invalid, expected) in [
        (
            WireOperator {
                version: 9,
                ..wire.clone()
            },
            "schema version 9",
        ),
        (
            WireOperator {
                name: "other.codec".into(),
                ..wire.clone()
            },
            "Unregistered physical extension",
        ),
        (
            WireOperator {
                name: " \t".into(),
                ..wire.clone()
            },
            "name must not be empty",
        ),
        (
            WireOperator {
                payload: vec![0xff],
                ..wire.clone()
            },
            "Invalid mock AI protobuf",
        ),
        (
            WireOperator {
                payload: MockAiNode {
                    prompt: None,
                    ..payload
                }
                .encode_to_vec(),
                ..wire.clone()
            },
            "missing its prompt",
        ),
    ] {
        assert_plan_error(&decode(&invalid).unwrap_err(), expected);
    }
    for (bytes, inputs, expected) in [
        (buf.as_slice(), &[][..], "requires exactly one input"),
        (&[0xff][..], children, "Invalid physical extension protobuf"),
        (&[][..], children, "name must not be empty"),
    ] {
        assert_plan_error(
            &codec
                .try_decode(bytes, inputs, &ctx, &converter)
                .unwrap_err(),
            expected,
        );
    }
    let empty = DistPhysicalCodec::default();
    assert_plan_error(
        &empty
            .try_encode(plan, &mut Vec::new(), &converter)
            .unwrap_err(),
        "Unsupported physical extension",
    );
    assert_plan_error(
        &empty
            .try_decode(&buf, children, &ctx, &converter)
            .unwrap_err(),
        "Unregistered physical extension",
    );
    Ok(())
}

#[tokio::test]
async fn extension_policy_enforces_admission_and_checks_the_whole_tree() -> Result<()> {
    let dir = tempdir()?;
    let path = dir.path().join("data.parquet");
    write_parquet(&path, &[(1, 20)])?;
    let ctx = coordinator();
    register_parquet(&ctx, &path, false).await?;
    let scan = map_input(&ctx, &["id", "prompt_col"], None).await?;
    let speculatable: Arc<dyn ExecutionPlan> = Arc::new(SpeculatableExec::new(
        Arc::clone(&scan),
        column_expr("prompt_col", scan.as_ref())?,
        "s".into(),
        "{}".into(),
        "safe_resp".into(),
    ));
    let nested: Arc<dyn ExecutionPlan> = Arc::new(MockAiExec::new(
        Arc::clone(&speculatable),
        column_expr("safe_resp", speculatable.as_ref())?,
        "m".into(),
        "{}".into(),
        "resp".into(),
    ));
    let mut registry = ai_codecs()?.physical;
    for name in ["", " \t", "test.mock_ai"] {
        let error = registry
            .register(Arc::new(MockAiCodec::new(name, true)))
            .unwrap_err();
        let expected = if name.trim().is_empty() {
            "name must not be empty"
        } else {
            "already registered"
        };
        assert_plan_error(&error, expected);
    }
    let mut nonspeculative = registry.clone();
    registry.register(Arc::new(SpeculatableCodec::new("test.speculatable", true)))?;
    assert_eq!(registry.policy_for(scan.as_ref())?, None);
    assert!(registry.plan_allows_speculation(scan.as_ref())?);
    assert_eq!(
        registry.policy_for(nested.as_ref())?,
        Some(ExtensionOperatorPolicy {
            map_only: true,
            speculation_allowed: false,
        })
    );
    assert!(!ExtensionOperatorPolicy::default().speculation_allowed);
    assert!(registry.plan_allows_speculation(speculatable.as_ref())?);
    assert!(!registry.plan_allows_speculation(nested.as_ref())?);
    let safe_parent = SpeculatableExec::new(
        Arc::clone(&nested),
        column_expr("resp", nested.as_ref())?,
        "s".into(),
        "{}".into(),
        "outer_resp".into(),
    );
    assert!(!registry.plan_allows_speculation(&safe_parent)?);

    rejected(
        Arc::clone(&nested),
        "Unsupported map-only operator MockAiExec",
    );
    let codecs = DistCodecs::default().with_physical_codecs(registry.clone());
    let (stage, splits) =
        split_plan_for_distribution_with_codecs(Arc::clone(&nested), 21, &codecs)?;
    assert_plan_error(
        &decode_stage_template(&stage, &ctx).unwrap_err(),
        "Unregistered physical extension",
    );
    let bound = bind_split_to_plan_with_codecs(&stage, &splits[0], &ctx, &codecs)?;
    assert!(!registry.plan_allows_speculation(bound.as_ref())?);
    assert_eq!(
        collect(bound, ctx.task_ctx())
            .await?
            .iter()
            .map(RecordBatch::num_rows)
            .sum::<usize>(),
        2
    );

    registry.register(Arc::new(SpeculatableCodec::new(
        "test.speculatable_alias",
        true,
    )))?;
    assert_plan_error(
        &registry
            .plan_allows_speculation(nested.as_ref())
            .unwrap_err(),
        "Ambiguous physical extension policy",
    );
    let mut ambiguous = codecs.clone();
    ambiguous.physical = registry;
    assert_plan_error(
        &split_plan_for_distribution_with_codecs(Arc::clone(&nested), 21, &ambiguous)
            .unwrap_err(),
        "Ambiguous physical extension policy",
    );
    let codec = DistPhysicalCodec::new(ambiguous.physical);
    let mut buf = vec![0xaa];
    assert_plan_error(
        &codec
            .try_encode(speculatable, &mut buf, &DefaultPhysicalProtoConverter {})
            .unwrap_err(),
        "Ambiguous physical extension encoding",
    );
    assert_eq!(buf, [0xaa]);

    nonspeculative
        .register(Arc::new(SpeculatableCodec::new("test.speculatable", false)))?;
    let rejected_codecs = DistCodecs::default().with_physical_codecs(nonspeculative);
    assert_plan_error(
        &split_plan_for_distribution_with_codecs(nested, 21, &rejected_codecs)
            .unwrap_err(),
        "Unsupported map-only operator SpeculatableExec",
    );
    assert_plan_error(
        &decode_stage_template_with_codecs(&stage, &ctx, &rejected_codecs).unwrap_err(),
        "Unsupported map-only operator SpeculatableExec",
    );
    Ok(())
}

#[tokio::test]
async fn stage_wire_extensions_use_the_versioned_codec() -> Result<()> {
    let dir = tempdir()?;
    let path = dir.path().join("data.parquet");
    write_parquet(&path, &[(1, 20)])?;
    let ctx = coordinator();
    register_parquet(&ctx, &path, false).await?;
    let input = map_input(&ctx, &["prompt_col"], None).await?;
    let prompt = column_expr("prompt_col", input.as_ref())?;
    let plan = Arc::new(MockAiExec::new(
        input,
        prompt,
        "m".into(),
        "{}".into(),
        "resp".into(),
    ));
    let codecs = ai_codecs()?;
    let (stage, _) = split_plan_for_distribution_with_codecs(plan, 1, &codecs)?;

    let proto = PhysicalPlanNode::decode(stage.plan_bytes.as_slice()).unwrap();
    let Some(PhysicalPlanType::Extension(extension)) = &proto.physical_plan_type else {
        panic!("expected the distributed extension envelope");
    };
    assert!(extension.plan_name.is_none());
    let wire = WireOperator::decode(extension.node.as_slice()).unwrap();
    let mut unsupported_version = extension.clone();
    unsupported_version.node = WireOperator { version: 9, ..wire }.encode_to_vec();
    let mut named = extension.clone();
    named.plan_name = Some("test.mock_ai".into());
    let mut multiple_inputs = extension.clone();
    multiple_inputs.inputs.push(extension.inputs[0].clone());
    for (extension, expected) in [
        (unsupported_version, "schema version 9"),
        (named, "extensions must use the versioned codec"),
        (multiple_inputs, "extensions require exactly one input"),
    ] {
        let mut altered = stage.clone();
        altered.plan_bytes = PhysicalPlanNode {
            physical_plan_type: Some(PhysicalPlanType::Extension(extension)),
        }
        .encode_to_vec();
        altered.plan_fingerprint = plan_hash(&altered)?;
        assert_plan_error(
            &decode_stage_template_with_codecs(&altered, &ctx, &codecs).unwrap_err(),
            expected,
        );
    }
    Ok(())
}

#[tokio::test]
async fn physical_and_file_codecs_can_be_used_together() -> Result<()> {
    let dir = tempdir()?;
    let path = dir.path().join("data.parquet");
    write_parquet(&path, &[(1, 20)])?;
    let ctx = coordinator();
    register_parquet(&ctx, &path, false).await?;
    let native = filtered_plan(&ctx, None).await?;
    let plan = rewrite_files(Arc::clone(&native), &|file| {
        vec![file.clone().with_extension(LanceFragment(7))]
    })?;
    let mut codecs = ai_codecs()?;
    codecs
        .file_extensions
        .register(Arc::new(LanceCodec("lance.fragment")))?;
    let (stage, splits) = split_plan_for_distribution_with_codecs(plan, 1, &codecs)?;
    let [split] = splits.as_slice() else {
        panic!("expected one split");
    };
    let worker = LocalWorker::new()
        .with_physical_codecs(codecs.physical)
        .with_extension_codecs(codecs.file_extensions);
    let prepared = worker.prepare_stage(&stage)?;
    let bound = bind_split_to_template(&prepared, split)?;
    assert_eq!(
        scan_node(bound.as_ref()).0.file_groups[0].files()[0]
            .extensions
            .get(),
        Some(&LanceFragment(7)),
    );
    let output = prepared.execute_split(split).await?;
    let baseline = collect(Arc::clone(&native), ctx.task_ctx()).await?;
    assert_matches_single_node(&stage, native.schema().as_ref(), &baseline, &[output])
}

#[tokio::test]
async fn fingerprints_bind_plan_schema_and_file_bytes() -> Result<()> {
    let dir = tempdir()?;
    let path = dir.path().join("data.parquet");
    write_parquet(&path, &[(1, 20)])?;
    let ctx = coordinator();
    register_parquet(&ctx, &path, false).await?;
    let plan = map_input(&ctx, &["id"], Some(10)).await?;
    let (stage, splits) = split_plan_for_distribution(Arc::clone(&plan), 1)?;
    let split = &splits[0];
    assert_eq!(stage.plan_fingerprint, plan_hash(&stage)?);
    assert_eq!(split.fingerprint, split_hash(&stage, &split.file_proto));

    let (repeat, repeat_splits) = split_plan_for_distribution(Arc::clone(&plan), 2)?;
    assert_eq!(stage.plan_bytes, repeat.plan_bytes);
    assert_eq!(stage.plan_fingerprint, repeat.plan_fingerprint);
    assert_eq!(split.fingerprint, repeat_splits[0].fingerprint);

    let (other, _) =
        split_plan_for_distribution(map_input(&ctx, &["id"], Some(11)).await?, 3)?;
    assert_ne!(stage.plan_fingerprint, other.plan_fingerprint);
    assert_plan_error(
        &bind_split_to_plan(&other, split, &ctx).unwrap_err(),
        "fingerprint does not match stage",
    );

    let mut altered_plan = stage.clone();
    altered_plan.plan_bytes = other.plan_bytes;
    assert_plan_error(
        &decode_stage_template(&altered_plan, &ctx).unwrap_err(),
        "Plan fingerprint does not match",
    );

    let mut altered_schema = stage.clone();
    altered_schema.output_schema = Arc::new(
        stage
            .output_schema
            .as_ref()
            .clone()
            .with_metadata(HashMap::from([("version".into(), "new".into())])),
    );
    assert_ne!(plan_hash(&altered_schema)?, stage.plan_fingerprint);
    assert_plan_error(
        &decode_stage_template(&altered_schema, &ctx).unwrap_err(),
        "Plan fingerprint does not match",
    );
    altered_schema.plan_fingerprint = plan_hash(&altered_schema)?;
    assert_plan_error(
        &decode_stage_template(&altered_schema, &ctx).unwrap_err(),
        "Template output schema differs",
    );

    let retagged = rewrite_files(plan, &|file| {
        let mut file = file.clone();
        file.object_meta.e_tag = Some("new-tag".into());
        vec![file]
    })?;
    let (retagged_stage, retagged_splits) = split_plan_for_distribution(retagged, 1)?;
    assert_eq!(retagged_stage.plan_fingerprint, stage.plan_fingerprint);
    assert_ne!(retagged_splits[0].fingerprint, split.fingerprint);
    let mut altered_file = split.clone();
    altered_file
        .file_proto
        .clone_from(&retagged_splits[0].file_proto);
    assert_plan_error(
        &bind_split_to_plan(&stage, &altered_file, &ctx).unwrap_err(),
        "fingerprint does not match stage",
    );
    Ok(())
}

#[test]
fn plan_fingerprint_accepts_independently_converted_schemas_with_nested_metadata()
-> Result<()> {
    let metadata = HashMap::from([
        ("source".into(), "fixture".into()),
        ("revision".into(), "9".into()),
        ("kind".into(), "nested".into()),
    ]);
    let field = Field::new("item", DataType::Int64, true).with_metadata(metadata.clone());
    let schema = Arc::new(Schema::new_with_metadata(
        vec![Field::new_list("items", field, true).with_metadata(metadata.clone())],
        metadata,
    ));
    let plan = DataSourceExec::from_data_source(
        FileScanConfigBuilder::new(
            ObjectStoreUrl::local_filesystem(),
            Arc::new(ParquetSource::new(schema)),
        )
        .build(),
    );
    let (stage, _) = split_plan_for_distribution(plan, 1)?;
    let ctx = coordinator();
    for _ in 0..10 {
        let mut independently_decoded = stage.clone();
        let proto = protobuf::Schema::try_from(stage.output_schema.as_ref())?;
        independently_decoded.output_schema = Arc::new(Schema::try_from(&proto)?);
        decode_stage_template(&independently_decoded, &ctx)?;
    }
    Ok(())
}
