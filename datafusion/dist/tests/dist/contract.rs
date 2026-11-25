// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: See the NOTICE file distributed with this work for additional information regarding copyright ownership
// SPDX-FileContributor: Licensed to the Apache Software Foundation (ASF) under one or more contributor license agreements

use std::collections::HashMap;
use std::fs;
use std::sync::Arc;

use datafusion::arrow::{
    array::Array,
    datatypes::{DataType, Schema},
    record_batch::RecordBatch,
};
use datafusion::common::stats::Precision;
use datafusion::common::{Result, ScalarValue, Statistics, TableReference};
use datafusion::datasource::{
    physical_plan::{FileGroup, FileScanConfigBuilder, FileSource},
    source::DataSourceExec,
};
use datafusion::llm::LlmExec;
use datafusion::physical_expr::{
    LexOrdering, Partitioning, PhysicalSortExpr,
    expressions::{Column, Literal, in_list},
};
use datafusion::physical_plan::{
    ExecutionPlan, coalesce_partitions::CoalescePartitionsExec, collect,
    filter::FilterExec, limit::LocalLimitExec, repartition::RepartitionExec,
};
use datafusion::prelude::{ParquetReadOptions, SessionConfig, SessionContext};
use datafusion_dist::{
    LlmPhysicalCodec, LocalWorker, StagePlan, bind_split_to_plan, bind_split_to_template,
    split_plan_for_distribution,
};
use datafusion_proto::bytes::{
    physical_plan_from_bytes_with_extension_codec,
    physical_plan_to_bytes_with_extension_codec,
};
use tempfile::tempdir;

use super::harness::{
    FILTERED, assert_attempts, assert_independent, assert_matches_single_node,
    assert_plan_error, context, coordinator, nodes, optimized, register_parquet,
    rejected, rewrite_files, rows, scan_node, submitted, write_parquet,
};

#[tokio::test]
async fn file_ranges_preserve_pruning_and_empty_split_output() -> Result<()> {
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
    let native = optimized(&ctx, FILTERED).await?;
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
    let (stage, mut splits) = split_plan_for_distribution(ranged, 7)?;
    assert_eq!(stage.stage_id, 7);
    assert_eq!(splits.len(), 2);
    assert!(splits.iter().all(|split| split.attempt_id == 0));
    splits[0].attempt_id = 3;
    splits[1].attempt_id = 9;

    let workers = [LocalWorker::new(), LocalWorker::new()];
    assert_independent(&ctx, &workers);
    let template = physical_plan_from_bytes_with_extension_codec(
        &stage.plan_bytes,
        &workers[0].context().task_ctx(),
        &LlmPhysicalCodec,
    )?;
    assert!(scan_node(template.as_ref()).0.file_groups.is_empty());
    let mut outputs = Vec::new();
    for ((worker, split), (start, end)) in workers
        .iter()
        .zip(&splits)
        .zip([(0, boundary), (boundary, size)])
    {
        let bound = bind_split_to_plan(&stage, split, worker.context())?;
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
    let native = optimized(&ctx, r#"SELECT id, part_col,
        llm(concat(part_col, ':', CAST(id AS VARCHAR), ':', prompt_col), 'hive-m', '{"temperature":0}') AS resp
        FROM t"#).await?;
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
    let (stage, splits) = split_plan_for_distribution(Arc::clone(&with_metadata), 8)?;
    let (source, _) = scan_node(with_metadata.as_ref());
    assert_eq!(splits.len(), 2);
    let workers = [
        LocalWorker::with_context(SessionContext::new()),
        LocalWorker::new(),
    ];
    assert_independent(&ctx, &workers);
    let mut outputs = Vec::new();
    for ((split, worker), expected_file) in splits
        .iter()
        .zip(&workers)
        .zip(source.file_groups.iter().flat_map(FileGroup::files))
    {
        let bound = bind_split_to_plan(&stage, split, worker.context())?;
        let mut node = bound.as_ref();
        let llm = loop {
            if let Some(exec) = node.downcast_ref::<LlmExec>() {
                break exec;
            }
            node = node.children()[0].as_ref();
        };
        assert_eq!(llm.model(), "hive-m");
        assert_eq!(llm.options_json(), r#"{"temperature":0}"#);
        assert_eq!(llm.output_col(), "__df_llm_0");
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
    for (sql, operator) in [
        ("SELECT id FROM t ORDER BY id", "SortExec"),
        ("SELECT id, count(*) FROM t GROUP BY id", "AggregateExec"),
        ("SELECT id FROM t LIMIT 1", "limit"),
        ("SELECT t.id FROM t JOIN s ON t.id = s.id", "HashJoinExec"),
        ("SELECT id FROM t UNION ALL SELECT id FROM s", "UnionExec"),
    ] {
        rejected(optimized(&ctx, sql).await?, operator);
    }
    let virtual_scan = optimized(&ctx, "SELECT file_row_index(), id FROM t").await?;
    let (_, parquet) = scan_node(virtual_scan.as_ref());
    assert!(!parquet.table_schema().virtual_columns().is_empty());
    rejected(virtual_scan, "virtual columns");

    let scan = optimized(&ctx, "SELECT * FROM t").await?;
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
    let native = optimized(&ctx, "SELECT * FROM t").await?;
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
        split.file_proto = file_proto;
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

    let nonempty = StagePlan {
        plan_bytes: physical_plan_to_bytes_with_extension_codec(
            native,
            &LlmPhysicalCodec,
        )?
        .to_vec(),
        ..stage
    };
    assert_plan_error(
        &worker.prepare_stage(&nonempty).unwrap_err(),
        "must have an empty Parquet scan",
    );
    Ok(())
}

#[tokio::test]
async fn empty_file_groups_round_trip_with_no_splits() -> Result<()> {
    let dir = tempdir()?;
    let path = dir.path().join("data.parquet");
    write_parquet(&path, &[(0, 20)])?;
    let ctx = coordinator();
    ctx.register_parquet("t", path.to_str().unwrap(), ParquetReadOptions::default())
        .await?;
    let native = optimized(&ctx, FILTERED).await?;
    let output_schema = native.schema();
    let empty = rewrite_files(native, &|_| vec![])?;
    let (stage, splits) = split_plan_for_distribution(empty, 12)?;
    assert!(splits.is_empty());
    assert!(!stage.plan_bytes.is_empty());
    assert_eq!(stage.output_schema, output_schema);
    let worker = LocalWorker::new();
    worker.prepare_stage(&stage)?;
    let template = physical_plan_from_bytes_with_extension_codec(
        &stage.plan_bytes,
        &worker.context().task_ctx(),
        &LlmPhysicalCodec,
    )?;
    assert!(scan_node(template.as_ref()).0.file_groups.is_empty());
    Ok(())
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn naturally_planned_parallel_parquet_ranges_match_single_node() -> Result<()> {
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
    let native = optimized(&ctx, FILTERED).await?;
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
    let (stage, splits) = split_plan_for_distribution(native, 13)?;
    assert_eq!(splits.len(), ranges.len());
    let workers = [LocalWorker::new(), LocalWorker::new()];
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

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn faster_worker_claims_more_splits_without_losing_or_repeating_rows() -> Result<()>
{
    let dir = tempdir()?;
    const FILES: usize = 16;
    for id in 0..FILES {
        // Give identical IDs and values to pairs of files to verify row multiplicity.
        let first_id = (id / 2 * 10) as i64;
        write_parquet(
            &dir.path().join(format!("{id}.parquet")),
            &[(first_id, 0), (first_id + 2, 20)],
        )?;
    }
    let ctx = coordinator();
    ctx.register_parquet(
        "t",
        dir.path().to_str().unwrap(),
        ParquetReadOptions::default(),
    )
    .await?;
    let native = optimized(&ctx, FILTERED).await?;
    let baseline_schema = native.schema();
    let baseline = collect(Arc::clone(&native), ctx.task_ctx()).await?;
    let (stage, splits) = split_plan_for_distribution(native, 10)?;
    assert_eq!(splits.len(), FILES);

    let (slow, fast) = super::gated_queue::gated_drain(&ctx, &stage, &splits).await?;
    assert_eq!(slow.len(), 1);
    assert_eq!(slow[0].split_id, splits[0].split_id);
    assert_eq!(fast.len(), FILES - 1);
    let outputs: Vec<_> = slow.into_iter().chain(fast).collect();
    assert_attempts(&splits, &outputs);
    assert_matches_single_node(&stage, &baseline_schema, &baseline, &outputs)?;
    for output in &outputs {
        assert_eq!(output.metrics.row_groups_pruned_statistics, 1);
        assert_eq!(output.metrics.row_groups_matched_statistics, 1);
    }
    let non_null_prompts: usize = baseline
        .iter()
        .map(|b| b.num_rows() - b.column(1).null_count())
        .sum();
    assert_eq!(non_null_prompts, FILES);
    assert_eq!(
        outputs
            .iter()
            .map(|out| out.metrics.submitted_prompts)
            .sum::<usize>(),
        non_null_prompts
    );
    Ok(())
}

const OUTPUT_FILTERED: &str = "SELECT id, resp FROM (
    SELECT id, llm(prompt_col, 'mock-m', '{}') AS resp FROM t WHERE filter_only_col > 10
) q WHERE resp LIKE 'mock%'";

#[tokio::test]
async fn default_parquet_pushdown_keeps_filters_on_both_sides_of_llm() -> Result<()> {
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
    let native = optimized(&ctx, OUTPUT_FILTERED).await?;
    let schema = native.schema();
    let baseline = collect(Arc::clone(&native), ctx.task_ctx()).await?;
    assert_eq!(submitted(native.as_ref()), 2);
    let (stage, splits) = split_plan_for_distribution(native, 14)?;
    assert_eq!(splits.len(), 2);
    let worker = LocalWorker::new();
    let prepared = worker.prepare_stage(&stage)?;
    let mut outputs = Vec::new();
    for split in &splits {
        let bound = bind_split_to_template(&prepared, split)?;
        let operators = nodes(bound.as_ref())
            .filter(|node| node.is::<LlmExec>() || node.is::<FilterExec>())
            .map(ExecutionPlan::name)
            .collect::<Vec<_>>();
        assert_eq!(operators, ["FilterExec", "LlmExec", "FilterExec"]);
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
async fn nested_llm_counts_submissions_from_both_layers() -> Result<()> {
    let dir = tempdir()?;
    for id in [0, 4] {
        write_parquet(
            &dir.path().join(format!("{id}.parquet")),
            &[(id, 20), (id + 2, 40)],
        )?;
    }
    let ctx = coordinator();
    register_parquet(&ctx, dir.path(), false).await?;
    let native = optimized(
        &ctx,
        "SELECT id,
        llm(concat('outer:', llm(prompt_col, 'inner-m', '{}')), 'outer-m', '{}') AS resp
        FROM t",
    )
    .await?;
    let schema = native.schema();
    let baseline = collect(Arc::clone(&native), ctx.task_ctx()).await?;
    // concat ignores NULL inputs: four prompts reach the inner operator and
    // all eight rows reach the outer operator.
    assert_eq!(submitted(native.as_ref()), 12);
    let (stage, splits) = split_plan_for_distribution(native, 15)?;
    let workers = [LocalWorker::new(), LocalWorker::new()];
    assert_independent(&ctx, &workers);
    assert_eq!(splits.len(), workers.len());
    let mut outputs = Vec::new();
    for (split, worker) in splits.iter().zip(&workers) {
        let prepared = worker.prepare_stage(&stage)?;
        let bound = bind_split_to_template(&prepared, split)?;
        assert_eq!(
            nodes(bound.as_ref())
                .filter(|node| node.is::<LlmExec>())
                .count(),
            2
        );
        let output = prepared.execute_split(split).await?;
        assert_eq!(output.metrics.submitted_prompts, 6);
        outputs.push(output);
    }
    assert_attempts(&splits, &outputs);
    assert_matches_single_node(&stage, &schema, &baseline, &outputs)
}

#[tokio::test]
async fn reused_prepared_stage_isolates_scan_filter_and_llm_metrics() -> Result<()> {
    let dir = tempdir()?;
    write_parquet(&dir.path().join("a.parquet"), &[(0, 0), (2, 20)])?;
    write_parquet(&dir.path().join("b.parquet"), &[(4, 0), (6, 20), (8, 40)])?;
    let ctx = context(2, 1, false, false);
    register_parquet(&ctx, dir.path(), false).await?;
    let native = optimized(&ctx, OUTPUT_FILTERED).await?;
    let schema = native.schema();
    let baseline = collect(Arc::clone(&native), ctx.task_ctx()).await?;
    let (stage, splits) = split_plan_for_distribution(native, 16)?;
    let worker = LocalWorker::new();
    let prepared = worker.prepare_stage(&stage)?;
    let filter_rows = |filter: &dyn ExecutionPlan| {
        filter.metrics().and_then(|metrics| metrics.output_rows())
    };
    let mut previous_plans: Vec<Arc<dyn ExecutionPlan>> = Vec::new();
    let mut outputs = Vec::new();
    for split in &splits {
        let bound = bind_split_to_template(&prepared, split)?;
        let (scan, source) = scan_node(bound.as_ref());
        let location = &scan.file_groups[0].files()[0].object_meta.location;
        let matched: usize = if location.as_ref().ends_with("a.parquet") {
            1
        } else {
            2
        };
        for previous in &previous_plans {
            let (_, old_source) = scan_node(previous.as_ref());
            assert!(Arc::ptr_eq(
                source.parquet_file_reader_factory().unwrap(),
                old_source.parquet_file_reader_factory().unwrap(),
            ));
        }
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
        previous_plans.push(bound);
        outputs.push(output);
    }
    assert_eq!(outputs.len(), 2);
    assert_attempts(&splits, &outputs);
    assert_matches_single_node(&stage, &schema, &baseline, &outputs)
}
