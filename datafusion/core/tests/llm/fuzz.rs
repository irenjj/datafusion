// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: See the NOTICE file distributed with this work for additional information regarding copyright ownership
// SPDX-FileContributor: Licensed to the Apache Software Foundation (ASF) under one or more contributor license agreements

use std::collections::HashMap;
use std::env;
use std::fmt::Write;
use std::future::{Future, ready};
use std::hash::{DefaultHasher, Hash, Hasher};
use std::iter::once;
use std::sync::Arc;
use std::time::Duration;

use async_trait::async_trait;

use datafusion::arrow::array::Int64Array;
use datafusion::arrow::datatypes::{DataType, Field, Schema};
use datafusion::arrow::record_batch::RecordBatch;
use datafusion::common::Result;
use datafusion::datasource::MemTable;
use datafusion::llm::{
    CancellationToken, LlmBackend, LlmBackendError, LlmRequest, LlmResponse,
    MockLlmBackend,
};
use rand::prelude::*;
use test_utils::fuzz::{self, FuzzTest, RunConfig};

use super::harness::{
    AnswerRow, answer_rows, assert_rows, run, string_array, submitted, test_context,
};

const PROMPT_COLUMNS: [(&str, DataType); 3] = [
    ("regular", DataType::Utf8),
    ("large", DataType::LargeUtf8),
    ("view_text", DataType::Utf8View),
];
const PROMPTS: [&str; 6] = [
    "",
    "shared-prefix/alpha-1",
    "shared-prefix/alpha-2",
    "shared-prefix/βeta-1",
    "shared-prefix/βeta-2",
    "other prompt",
];

struct Row {
    id: i64,
    prompts: [Option<&'static str>; PROMPT_COLUMNS.len()],
}

struct Dataset {
    rows: Vec<Row>,
    table: Arc<MemTable>,
    partitions: usize,
}

#[derive(Debug)]
struct Case {
    column: usize,
    min_id: Option<i64>,
    input_limit: Option<usize>,
    nested: bool,
    filter_response: bool,
    order_by_response: bool,
    output_limit: Option<usize>,
}

impl Case {
    fn generate(rng: &mut SmallRng, nrows: usize) -> Self {
        let order_by_response = rng.random_bool(0.5);
        let mut case = Self {
            column: rng.random_range(0..PROMPT_COLUMNS.len()),
            min_id: rng
                .random_bool(0.6)
                .then(|| rng.random_range(1..=nrows as i64)),
            input_limit: rng.random_bool(0.4).then(|| rng.random_range(1..nrows)),
            nested: rng.random_bool(0.5),
            filter_response: rng.random_bool(0.5),
            order_by_response,
            output_limit: None,
        };
        // An unordered limit on a response filter may have to infer additional
        // rows before it finds enough matches.
        case.output_limit = ((order_by_response || !case.filter_response)
            && rng.random_bool(0.6))
        .then(|| rng.random_range(1..nrows));
        case
    }

    fn sql(&self, options: &str) -> String {
        let column = PROMPT_COLUMNS[self.column].0;
        let mut prompt = format!("llm({column}, 'inner', '{options}')");
        if self.nested {
            prompt = format!("llm({prompt}, 'outer', '{options}')");
        }

        let id_filter = self.min_id.map(|min_id| format!("id >= {min_id}"));
        let mut predicates = Vec::new();
        let source = if let Some(limit) = self.input_limit {
            let filter = id_filter
                .map(|filter| format!(" WHERE {filter}"))
                .unwrap_or_default();
            format!(
                "(SELECT id, {column} FROM t{filter} ORDER BY id LIMIT {limit}) input"
            )
        } else {
            // With no input limit, the passthrough predicate starts above inference.
            predicates.extend(id_filter);
            "t".to_owned()
        };
        if self.filter_response {
            predicates.push("r LIKE '%-1'".into());
        }

        let mut sql =
            format!("SELECT id, r FROM (SELECT id, {prompt} AS r FROM {source}) q");
        if !predicates.is_empty() {
            write!(sql, " WHERE {}", predicates.join(" AND ")).unwrap();
        }
        if self.order_by_response {
            sql.push_str(" ORDER BY r ASC NULLS LAST, id DESC");
        }
        if let Some(limit) = self.output_limit {
            write!(sql, " LIMIT {limit}").unwrap();
        }
        sql
    }

    /// Evaluate the query stages over the source rows without SQL or physical plan code.
    fn reference(&self, rows: &[Row]) -> (Vec<AnswerRow>, usize) {
        let mut input = rows
            .iter()
            .filter(|row| self.min_id.is_none_or(|min_id| row.id >= min_id))
            .collect::<Vec<_>>();
        if let Some(limit) = self.input_limit {
            input.sort_unstable_by_key(|row| row.id);
            input.truncate(limit);
        }

        let mut submitted = 0;
        let mut result = Vec::new();
        for row in input {
            let answer = row.prompts[self.column].map(|prompt| {
                submitted += 1;
                let answer = format!("mock[inner]: {prompt}");
                if self.nested {
                    submitted += 1;
                    format!("mock[outer]: {answer}")
                } else {
                    answer
                }
            });
            if !self.filter_response || answer.as_ref().is_some_and(|r| r.ends_with("-1"))
            {
                result.push((row.id, answer));
            }
        }

        if self.order_by_response {
            result.sort_unstable_by(|(id_a, a), (id_b, b)| {
                a.is_none()
                    .cmp(&b.is_none())
                    .then_with(|| a.cmp(b))
                    .then_with(|| id_b.cmp(id_a))
            });
            if let Some(limit) = self.output_limit {
                result.truncate(limit);
            }
        }
        (result, submitted)
    }
}

impl Dataset {
    fn generate(dataset_seed: u64) -> Result<Self> {
        let mut rng = SmallRng::seed_from_u64(dataset_seed);
        let nrows = rng.random_range(16..=64);
        let null_rates: [f64; PROMPT_COLUMNS.len()] =
            std::array::from_fn(|_| rng.random_range(0.0..=1.0));
        let mut rows = (1..=nrows)
            .map(|id| Row {
                id,
                prompts: std::array::from_fn(|column| {
                    (!rng.random_bool(null_rates[column]))
                        .then(|| PROMPTS[rng.random_range(0..PROMPTS.len())])
                }),
            })
            .collect::<Vec<_>>();
        rows.shuffle(&mut rng);

        let schema = Arc::new(Schema::new(
            once(Field::new("id", DataType::Int64, false))
                .chain(
                    PROMPT_COLUMNS
                        .iter()
                        .map(|(name, kind)| Field::new(*name, kind.clone(), true)),
                )
                .collect::<Vec<_>>(),
        ));
        let partitions = rng.random_range(2..=4);
        let mut batches = vec![Vec::new(); partitions];
        let mut remaining = rows.as_slice();
        for partition in (0..partitions).cycle() {
            if remaining.is_empty() {
                break;
            }
            let count = rng.random_range(1..=8).min(remaining.len());
            let (chunk, rest) = remaining.split_at(count);
            let mut columns = vec![Arc::new(Int64Array::from_iter_values(
                chunk.iter().map(|row| row.id),
            )) as _];
            columns.extend(PROMPT_COLUMNS.iter().enumerate().map(
                |(index, (_, kind))| {
                    string_array(
                        kind,
                        chunk.iter().map(|row| row.prompts[index]).collect(),
                    )
                },
            ));
            batches[partition].push(RecordBatch::try_new(Arc::clone(&schema), columns)?);
            remaining = rest;
        }
        Ok(Dataset {
            rows,
            table: Arc::new(MemTable::try_new(schema, batches)?),
            partitions,
        })
    }
}

#[derive(Debug)]
struct DelayedMock(u64);

#[async_trait]
impl LlmBackend for DelayedMock {
    async fn generate(
        &self,
        request: LlmRequest,
        cancel: CancellationToken,
    ) -> std::result::Result<LlmResponse, LlmBackendError> {
        let mut hash = DefaultHasher::new();
        (self.0, &request.model, &request.prompts).hash(&mut hash);
        let delay = SmallRng::seed_from_u64(hash.finish()).random_range(0..=8_000);
        tokio::time::sleep(Duration::from_micros(delay)).await;
        MockLlmBackend.generate(request, cancel).await
    }
}

struct LlmFuzz;

impl FuzzTest for LlmFuzz {
    type Dataset = Dataset;

    fn name(&self) -> &str {
        "llm_sql_oracle"
    }

    fn prepare_dataset(
        &self,
        dataset_seed: u64,
    ) -> impl Future<Output = Result<Dataset>> {
        ready(Dataset::generate(dataset_seed))
    }

    async fn run_case(&self, dataset: &Dataset, case_seed: u64) -> Result<()> {
        let mut rng = SmallRng::seed_from_u64(case_seed);
        let case = Case::generate(&mut rng, dataset.rows.len());
        let batch_size = rng.random_range(1..=16);
        let request_batch = rng.random_range(1..=5);
        let inflight = rng.random_range(1..=6);
        let delayed_options = format!(
            r#"{{"provider":"delayed","batch_size":{request_batch},"inflight_limit":{inflight}}}"#
        );
        let ctx = test_context(batch_size, dataset.partitions);
        ctx.state()
            .llm_provider_registry()
            .register("delayed", Arc::new(DelayedMock(case_seed)))?;
        ctx.register_table("t", dataset.table.clone())?;
        for options in ["{}", &delayed_options] {
            let sql = case.sql(options);
            eprintln!("SQL: {sql}\nBatch size: {batch_size}");
            let (plan, batches) = run(&ctx, &sql).await?;
            check_rows(&case, dataset, &batches, submitted(plan.as_ref()));
        }
        Ok(())
    }
}

fn check_rows(case: &Case, dataset: &Dataset, batches: &[RecordBatch], observed: usize) {
    let (expected, count) = case.reference(&dataset.rows);
    if let Some(limit) = case.output_limit.filter(|_| !case.order_by_response) {
        // The selected rows can vary across partitions. Validate the whole
        // eligible set and charge only the non-NULL rows actually returned.
        let actual = answer_rows(batches);
        let mut eligible = expected.into_iter().collect::<HashMap<_, _>>();
        assert_eq!(actual.len(), eligible.len().min(limit));
        for (id, answer) in &actual {
            assert_eq!(
                eligible.remove(id).as_ref(),
                Some(answer),
                "ineligible row, duplicate id, or incorrect answer for id {id}"
            );
        }
        let calls_per_prompt = if case.nested { 2 } else { 1 };
        assert!(
            observed <= limit * calls_per_prompt,
            "{observed} submissions for {case:?}"
        );
        let non_null = actual.iter().filter(|(_, answer)| answer.is_some()).count();
        assert_eq!(
            observed,
            non_null * calls_per_prompt,
            "prompt submissions for {case:?}"
        );
    } else {
        assert_rows(batches, expected, case.order_by_response);
        assert_eq!(observed, count, "prompt submissions for {case:?}");
    }
}

#[tokio::test(start_paused = true)]
async fn differential_fuzz() -> Result<()> {
    if let Ok(replay) = env::var("LLM_FUZZ_REPLAY") {
        let (dataset_seed, case_seed) =
            replay.split_once(',').expect("Use dataset_seed,case_seed");
        return LlmFuzz
            .reproduce(dataset_seed.parse().unwrap(), case_seed.parse().unwrap())
            .await;
    }

    let cases = env::var("LLM_FUZZ_CASES").ok().map_or(36, |value| {
        value
            .parse()
            .expect("LLM_FUZZ_CASES must be a positive integer")
    });
    let seed = env::var("LLM_FUZZ_SEED").ok().map_or(2026, |value| {
        value.parse().expect("LLM_FUZZ_SEED must be a u64")
    });
    let stats = fuzz::run(
        &LlmFuzz,
        RunConfig {
            run_seed: Some(seed),
            max_cases: Some(cases),
            time_limit: None,
            cases_per_dataset: 4,
        },
    )
    .await?;
    println!("LLM differential fuzz: {stats:?}");
    Ok(())
}
