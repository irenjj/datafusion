// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: See the NOTICE file distributed with this work for additional information regarding copyright ownership
// SPDX-FileContributor: Licensed to the Apache Software Foundation (ASF) under one or more contributor license agreements

use std::env;
use std::fs;
use std::future::{Future, ready};
use std::sync::Arc;

use datafusion::arrow::datatypes::DataType;
use datafusion::common::Result;
use datafusion::datasource::physical_plan::FileGroup;
use datafusion::physical_expr::Partitioning;
use datafusion::physical_plan::{ExecutionPlan, collect, repartition::RepartitionExec};
use datafusion::prelude::{SessionConfig, SessionContext};
use datafusion_dist::{
    LocalWorker, bind_split_to_template, split_plan_for_distribution_with_codecs,
};
use futures::future::try_join_all;
use rand::prelude::*;
use rand::seq::index::sample;
use tempfile::{TempDir, tempdir};
use test_utils::fuzz::{self, FuzzTest, RunConfig};

use super::harness::{
    Row, ai_codecs, assert_attempts, assert_independent, assert_matches_single_node,
    assert_pruned_projection, context, map_input, nodes, register_parquet, response_plan,
    rewrite_files, scan_node, submitted, write_row_groups,
};

struct Dataset {
    dir: TempDir,
    files: usize,
    hive: bool,
    string_type: DataType,
}

impl Dataset {
    fn generate(seed: u64) -> Result<Self> {
        const PROMPTS: [&str; 6] = [
            "",
            "a-1",
            "a-2",
            "longer than twelve bytes",
            "βeta-1",
            "βeta-2",
        ];
        let mut rng = SmallRng::seed_from_u64(seed);
        let dir = tempdir()?;
        let files = rng.random_range(1..=4);
        let hive = rng.random_bool(0.5);
        let string_type = [DataType::Utf8, DataType::LargeUtf8, DataType::Utf8View]
            [rng.random_range(0..3)]
        .clone();
        let null_rate = rng.random_range(0.0..=1.0);

        for file in 0..files {
            let parent = if hive {
                let partition = dir.path().join(format!("part_col={}", file % 2));
                fs::create_dir_all(&partition)?;
                partition
            } else {
                dir.path().to_owned()
            };
            let groups = (0..rng.random_range(1..=4)).map(|index| {
                let filter = rng.random_range(-8..16);
                let mut rows = (0..rng.random_range(1..=12))
                    .map(|_| Row {
                        id: rng.random_range(0..10),
                        filter: filter + rng.random_range(0..6),
                        prompt: (!rng.random_bool(null_rate))
                            .then(|| PROMPTS[rng.random_range(0..PROMPTS.len())]),
                    })
                    .collect::<Vec<_>>();
                // Keep a scan in every optimized plan even when statistics can
                // eliminate all the randomly generated rows for this predicate.
                if index == 0 {
                    rows[0].filter = 20;
                }
                rows.push(rows[0]);
                rows
            });
            write_row_groups(
                &parent.join(format!("{file}.parquet")),
                &string_type,
                groups,
            )?;
        }
        eprintln!("Dataset: files={files}, hive={hive}, prompt_type={string_type}");
        Ok(Self {
            dir,
            files,
            hive,
            string_type,
        })
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum SplitMode {
    WholeFile,
    NativeRanges,
    ByteRanges,
}

#[derive(Debug)]
struct Case {
    mode: SplitMode,
    pushdown_filters: bool,
    min_filter: Option<i64>,
    output_filter: Option<&'static str>,
    nested: bool,
    keep_prompt_col: bool,
    round_robin: bool,
    workers: usize,
    batch_size: usize,
}

impl Case {
    fn generate(seed: u64) -> Self {
        let mut rng = SmallRng::seed_from_u64(seed);
        Self {
            mode: [
                SplitMode::WholeFile,
                SplitMode::NativeRanges,
                SplitMode::ByteRanges,
            ][rng.random_range(0..3)],
            pushdown_filters: rng.random_bool(0.5),
            min_filter: rng.random_bool(0.75).then(|| rng.random_range(-8..18)),
            output_filter: [None, Some("mock%"), Some("%-1")][rng.random_range(0..3)],
            nested: rng.random_bool(0.5),
            keep_prompt_col: rng.random_bool(0.5),
            round_robin: rng.random_bool(0.5),
            workers: rng.random_range(1..=3),
            batch_size: rng.random_range(1..=16),
        }
    }

    async fn plan(
        &self,
        ctx: &SessionContext,
        hive: bool,
    ) -> Result<Arc<dyn ExecutionPlan>> {
        let mut output = vec!["id"];
        if hive {
            output.push("part_col");
        }
        let mut input = output.clone();
        input.push("prompt_col");
        if self.keep_prompt_col {
            output.push("prompt_col");
        }
        let scan = map_input(ctx, &input, self.min_filter).await?;
        response_plan(scan, &output, self.nested, self.output_filter)
    }
}

struct DistributionFuzz;

impl FuzzTest for DistributionFuzz {
    type Dataset = Dataset;

    fn name(&self) -> &str {
        "dist_parquet_map_only"
    }

    fn prepare_dataset(&self, seed: u64) -> impl Future<Output = Result<Dataset>> {
        ready(Dataset::generate(seed))
    }

    async fn run_case(&self, dataset: &Dataset, seed: u64) -> Result<()> {
        let case = Case::generate(seed);
        let codecs = ai_codecs()?;
        eprintln!("Case: {case:?}");

        let ctx = context(case.batch_size, 1, case.pushdown_filters, false);
        ctx.state_ref()
            .write()
            .config_mut()
            .options_mut()
            .execution
            .parquet
            .schema_force_view_types = dataset.string_type == DataType::Utf8View;
        register_parquet(&ctx, dataset.dir.path(), dataset.hive).await?;
        let native = case.plan(&ctx, dataset.hive).await?;
        let schema = native.schema();
        let (scan, _) = scan_node(native.as_ref());
        assert_eq!(
            scan.file_schema()
                .field_with_name("prompt_col")?
                .data_type(),
            &dataset.string_type
        );
        let baseline = collect(Arc::clone(&native), ctx.task_ctx()).await?;
        let expected_submissions = submitted(native.as_ref());

        let mut distributed = match case.mode {
            SplitMode::WholeFile => native,
            SplitMode::NativeRanges => {
                {
                    let state = ctx.state_ref();
                    let mut state = state.write();
                    let options = state.config_mut().options_mut();
                    options.execution.target_partitions = dataset.files * 2 + 2;
                    options.optimizer.repartition_file_scans = true;
                }
                let plan = case.plan(&ctx, dataset.hive).await?;
                let (scan, _) = scan_node(plan.as_ref());
                assert!(
                    scan.file_groups
                        .iter()
                        .flat_map(FileGroup::files)
                        .any(|file| file.range.is_some())
                );
                plan
            }
            SplitMode::ByteRanges => rewrite_files(native, &|file| {
                let filename = file.object_meta.location.filename().unwrap();
                let file_index: u64 =
                    filename.strip_suffix(".parquet").unwrap().parse().unwrap();
                let mut rng = SmallRng::seed_from_u64(seed.wrapping_add(file_index));
                let size = file.object_meta.size as usize;
                // [0, 1) is inside the header, so even a fully matching file
                // has a zero-row split. Other cuts are arbitrary byte offsets.
                let mut cuts = vec![0, 1, size];
                let extra_cuts = rng.random_range(0..=2);
                cuts.extend(
                    sample(&mut rng, size - 2, extra_cuts)
                        .iter()
                        .map(|offset| offset + 2),
                );
                cuts.sort_unstable();
                cuts.windows(2)
                    .map(|bounds| {
                        file.clone().with_range(bounds[0] as i64, bounds[1] as i64)
                    })
                    .collect()
            })?,
        };
        if case.round_robin {
            distributed = Arc::new(RepartitionExec::try_new(
                distributed,
                Partitioning::RoundRobinBatch(3),
            )?);
        }
        let (stage, splits) =
            split_plan_for_distribution_with_codecs(distributed, seed, &codecs)?;
        assert!(!splits.is_empty());
        let workers = (0..case.workers)
            .map(|_| {
                LocalWorker::with_context(SessionContext::new_with_config(
                    SessionConfig::new().with_batch_size(case.batch_size),
                ))
                .with_codecs(codecs.clone())
            })
            .collect::<Vec<_>>();
        assert_independent(&ctx, &workers);

        let results = try_join_all(workers.iter().enumerate().map(|(index, worker)| {
            let stage = &stage;
            let assigned = splits.iter().skip(index).step_by(workers.len());
            async move {
                let prepared = worker.prepare_stage(stage)?;
                let mut outputs = Vec::new();
                for split in assigned {
                    let bound = bind_split_to_template(&prepared, split)?;
                    assert!(
                        nodes(bound.as_ref()).all(|node| !node.is::<RepartitionExec>())
                    );
                    assert_pruned_projection(bound.as_ref());
                    outputs.push(prepared.execute_split(split).await?);
                }
                Ok::<_, datafusion::common::DataFusionError>(outputs)
            }
        }))
        .await?;
        let outputs = results.into_iter().flatten().collect::<Vec<_>>();
        assert_matches_single_node(&stage, &schema, &baseline, &outputs)?;
        assert_attempts(&splits, &outputs);
        assert_eq!(
            outputs
                .iter()
                .map(|output| output.metrics.submitted_prompts)
                .sum::<usize>(),
            expected_submissions,
        );
        if case.mode == SplitMode::ByteRanges {
            assert!(
                outputs
                    .iter()
                    .filter(|output| output.metrics.output_rows == 0)
                    .count()
                    >= dataset.files
            );
        }
        Ok(())
    }
}

#[tokio::test]
async fn differential_fuzz() -> Result<()> {
    if let Ok(replay) = env::var("DIST_FUZZ_REPLAY") {
        let (dataset_seed, case_seed) =
            replay.split_once(',').expect("Use dataset_seed,case_seed");
        return DistributionFuzz
            .reproduce(dataset_seed.parse().unwrap(), case_seed.parse().unwrap())
            .await;
    }

    let cases = env::var("DIST_FUZZ_CASES").ok().map_or(36, |value| {
        value
            .parse()
            .expect("DIST_FUZZ_CASES must be a positive integer")
    });
    let seed = env::var("DIST_FUZZ_SEED").ok().map_or(2026, |value| {
        value.parse().expect("DIST_FUZZ_SEED must be a u64")
    });
    let stats = fuzz::run(
        &DistributionFuzz,
        RunConfig {
            run_seed: Some(seed),
            max_cases: Some(cases),
            time_limit: None,
            cases_per_dataset: 4,
        },
    )
    .await?;
    println!("Distribution differential fuzz: {stats:?}");
    Ok(())
}
