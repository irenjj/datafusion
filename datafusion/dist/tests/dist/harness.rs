// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: See the NOTICE file distributed with this work for additional information regarding copyright ownership
// SPDX-FileContributor: Licensed to the Apache Software Foundation (ASF) under one or more contributor license agreements

use std::collections::{HashMap, HashSet};
use std::fs::File;
use std::path::Path;
use std::sync::Arc;

use datafusion::arrow::{
    array::{ArrayRef, Int64Array, LargeStringArray, StringArray, StringViewArray},
    datatypes::{DataType, Field, Schema},
    record_batch::RecordBatch,
};
use datafusion::common::{DataFusionError, Result, ScalarValue};
use datafusion::datasource::{
    listing::PartitionedFile,
    physical_plan::{
        FileGroup, FileScanConfig, FileScanConfigBuilder, FileSource, ParquetSource,
    },
    source::DataSourceExec,
};
use datafusion::llm::LlmExec;
use datafusion::parquet::{arrow::ArrowWriter, file::metadata::ParquetMetaData};
use datafusion::physical_plan::{
    ChildrenPropertiesMode, ExecutionPlan, ReplaceChildrenOptions,
};
use datafusion::prelude::{ParquetReadOptions, SessionConfig, SessionContext};
use datafusion_dist::{
    LocalWorker, Split, SplitOutput, StagePlan, split_plan_for_distribution,
};

pub(super) const FILTERED: &str = r#"SELECT id, llm(prompt_col, 'mock-m', '{"temperature":0}') AS resp
    FROM t WHERE filter_only_col > 10"#;

pub(super) fn coordinator() -> SessionContext {
    context(8192, 1, true, false)
}

pub(super) fn context(
    batch_size: usize,
    target_partitions: usize,
    pushdown_filters: bool,
    natural_ranges: bool,
) -> SessionContext {
    let mut config = SessionConfig::new()
        .with_batch_size(batch_size)
        .with_target_partitions(target_partitions);
    config.options_mut().execution.parquet.pushdown_filters = pushdown_filters;
    config.options_mut().optimizer.repartition_file_min_size = 1;
    config.options_mut().optimizer.repartition_file_scans = natural_ranges;
    SessionContext::new_with_config(config)
}

pub(super) fn write_parquet(
    path: &Path,
    groups: &[(i64, i64)],
) -> Result<ParquetMetaData> {
    let groups = groups.iter().map(|&(id, filter)| {
        vec![
            Row {
                id,
                filter,
                prompt: Some("a prompt"),
            },
            Row {
                id: id + 1,
                filter: filter + 1,
                prompt: None,
            },
        ]
    });
    write_row_groups(path, &DataType::Utf8, groups)
}

#[derive(Clone, Copy)]
pub(super) struct Row {
    pub(super) id: i64,
    pub(super) filter: i64,
    pub(super) prompt: Option<&'static str>,
}

pub(super) fn write_row_groups(
    path: &Path,
    string_type: &DataType,
    groups: impl IntoIterator<Item = Vec<Row>>,
) -> Result<ParquetMetaData> {
    let schema = Arc::new(Schema::new(vec![
        Field::new("id", DataType::Int64, false),
        Field::new("filter_only_col", DataType::Int64, false),
        Field::new("unused_wide_col", DataType::Utf8, false),
        Field::new("prompt_col", string_type.clone(), true),
    ]));
    let mut writer =
        ArrowWriter::try_new(File::create(path)?, Arc::clone(&schema), None)?;
    let unused = "unused data".repeat(128);
    for group in groups {
        let prompts = group.iter().map(|row| row.prompt).collect::<Vec<_>>();
        let prompts: ArrayRef = match string_type {
            DataType::Utf8 => Arc::new(StringArray::from(prompts)),
            DataType::LargeUtf8 => Arc::new(LargeStringArray::from(prompts)),
            DataType::Utf8View => Arc::new(StringViewArray::from(prompts)),
            _ => unreachable!("unsupported test prompt type {string_type}"),
        };
        let batch = RecordBatch::try_new(
            Arc::clone(&schema),
            vec![
                Arc::new(Int64Array::from_iter_values(group.iter().map(|row| row.id))),
                Arc::new(Int64Array::from_iter_values(
                    group.iter().map(|row| row.filter),
                )),
                Arc::new(StringArray::from(vec![unused.as_str(); group.len()])),
                prompts,
            ],
        )?;
        writer.write(&batch)?;
        writer.flush()?;
    }
    Ok(writer.close()?)
}

pub(super) fn hive_options() -> ParquetReadOptions<'static> {
    ParquetReadOptions::default()
        .table_partition_cols(vec![("part_col".into(), DataType::Utf8)])
}

pub(super) async fn register_parquet(
    ctx: &SessionContext,
    path: &Path,
    hive: bool,
) -> Result<()> {
    let options = if hive {
        hive_options()
    } else {
        ParquetReadOptions::default()
    };
    ctx.register_parquet("t", path.to_str().unwrap(), options)
        .await
}

pub(super) async fn optimized(
    ctx: &SessionContext,
    sql: &str,
) -> Result<Arc<dyn ExecutionPlan>> {
    ctx.sql(sql).await?.create_physical_plan().await
}

pub(super) fn scan_node(plan: &dyn ExecutionPlan) -> (&FileScanConfig, &ParquetSource) {
    if let Some(source) = plan.downcast_ref::<DataSourceExec>() {
        return source.downcast_to_file_source::<ParquetSource>().unwrap();
    }
    let [child] = plan.children()[..] else {
        panic!("expected unary map-only plan at {}", plan.name());
    };
    scan_node(child.as_ref())
}

pub(super) fn rewrite_files(
    plan: Arc<dyn ExecutionPlan>,
    update: &impl Fn(&PartitionedFile) -> Vec<PartitionedFile>,
) -> Result<Arc<dyn ExecutionPlan>> {
    if let Some(exec) = plan.downcast_ref::<DataSourceExec>() {
        let (scan, _) = exec.downcast_to_file_source::<ParquetSource>().unwrap();
        let groups = scan
            .file_groups
            .iter()
            .flat_map(FileGroup::files)
            .flat_map(update)
            .map(|file| FileGroup::new(vec![file]))
            .collect();
        return Ok(DataSourceExec::from_data_source(
            FileScanConfigBuilder::from(scan.clone())
                .with_file_groups(groups)
                .build(),
        ));
    }
    let [child] = plan.children()[..] else {
        panic!("expected unary map-only plan at {}", plan.name());
    };
    let child = rewrite_files(Arc::clone(child), update)?;
    plan.replace_children(
        vec![child],
        ReplaceChildrenOptions::new(ChildrenPropertiesMode::Recompute),
    )
}

pub(super) fn assert_independent(coordinator: &SessionContext, workers: &[LocalWorker]) {
    let contexts = std::iter::once(coordinator)
        .chain(workers.iter().map(LocalWorker::context))
        .collect::<Vec<_>>();
    for (index, a) in contexts.iter().enumerate() {
        for b in &contexts[index + 1..] {
            assert_ne!(a.session_id(), b.session_id());
            assert!(!Arc::ptr_eq(&a.runtime_env(), &b.runtime_env()));
        }
    }
    for worker in workers {
        assert!(!worker.context().table_exist("t").unwrap());
    }
}

pub(super) fn nodes(
    plan: &dyn ExecutionPlan,
) -> impl Iterator<Item = &dyn ExecutionPlan> {
    std::iter::successors(Some(plan), |node| {
        let mut children = node.children().into_iter();
        let child = children.next()?;
        assert!(
            children.next().is_none(),
            "expected unary map-only plan at {}",
            node.name()
        );
        Some(child.as_ref())
    })
}

pub(super) fn submitted(plan: &dyn ExecutionPlan) -> usize {
    nodes(plan)
        .filter(|node| node.is::<LlmExec>())
        .map(|node| metric(node, "submitted_prompts"))
        .sum()
}

pub(super) fn metric(plan: &dyn ExecutionPlan, name: &str) -> usize {
    plan.metrics()
        .and_then(|metrics| metrics.sum_by_name(name))
        .map_or(0, |value| value.as_usize())
}

pub(super) fn assert_pruned_projection(plan: &dyn ExecutionPlan) {
    let (scan, parquet) = scan_node(plan);
    let unused = scan.file_schema().index_of("unused_wide_col").unwrap();
    let indices = parquet.projection().unwrap().column_indices();
    assert!(
        !indices.contains(&unused),
        "scan read projection: {indices:?}"
    );
    if let Some(predicate) = parquet.filter() {
        assert!(!predicate.to_string().contains("unused_wide_col"));
    }
    assert_eq!(plan.properties().output_partitioning().partition_count(), 1);
}

pub(super) fn assert_attempts(expected: &[Split], outputs: &[SplitOutput]) {
    let expected_ids: HashSet<_> = expected
        .iter()
        .map(|s| (s.split_id, s.attempt_id))
        .collect();
    let actual_ids: HashSet<_> =
        outputs.iter().map(|s| (s.split_id, s.attempt_id)).collect();
    assert_eq!(
        expected_ids.len(),
        expected.len(),
        "duplicate split IDs in fixture"
    );
    assert_eq!(
        actual_ids.len(),
        outputs.len(),
        "duplicate split execution outputs"
    );
    assert_eq!(actual_ids, expected_ids);
}

pub(super) fn rows(batches: &[RecordBatch]) -> Result<HashMap<Vec<ScalarValue>, usize>> {
    let mut counts = HashMap::new();
    for batch in batches {
        for row in 0..batch.num_rows() {
            let values = batch
                .columns()
                .iter()
                .map(|array| ScalarValue::try_from_array(array, row))
                .collect::<Result<Vec<_>>>()?;
            *counts.entry(values).or_default() += 1;
        }
    }
    Ok(counts)
}

pub(super) fn assert_matches_single_node(
    stage: &StagePlan,
    baseline_schema: &Schema,
    baseline: &[RecordBatch],
    outputs: &[SplitOutput],
) -> Result<()> {
    let batches: Vec<_> = outputs
        .iter()
        .flat_map(|output| output.batches.iter().cloned())
        .collect();
    assert_eq!(stage.output_schema.as_ref(), baseline_schema);
    assert!(
        baseline
            .iter()
            .all(|b| b.schema().as_ref() == baseline_schema)
    );
    for batch in &batches {
        assert_eq!(batch.schema(), stage.output_schema);
    }
    assert_eq!(rows(&batches)?, rows(baseline)?);
    for output in outputs {
        assert_eq!(
            output.metrics.output_rows,
            output
                .batches
                .iter()
                .map(RecordBatch::num_rows)
                .sum::<usize>()
        );
    }
    Ok(())
}

pub(super) fn rejected(plan: Arc<dyn ExecutionPlan>, operator: &str) {
    let error = split_plan_for_distribution(plan, 9).unwrap_err();
    assert_plan_error(&error, operator);
}

pub(super) fn assert_plan_error(error: &DataFusionError, expected: &str) {
    assert!(
        matches!(error.find_root(), DataFusionError::Plan(_)),
        "{error}"
    );
    assert!(error.to_string().contains(expected), "{error}");
}
