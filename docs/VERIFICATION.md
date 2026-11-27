<!--
SPDX-License-Identifier: Apache-2.0
SPDX-FileCopyrightText: See the NOTICE file distributed with this work for additional information regarding copyright ownership
SPDX-FileContributor: Licensed to the Apache Software Foundation (ASF) under one or more contributor license agreements
-->

# AI engine verification

Select every surface touched by the change and use [the test index](TEST_INDEX.md) to identify the tests and fixtures. Run `cargo fmt --all -- --check` for every PR in this workflow. The table uses these shell functions for the crate checks.

```bash
lint() { cargo clippy --locked -p "$@" --all-targets -- -D warnings; }
tests() { cargo nextest run --locked --cargo-profile ci -p "$@"; }
```

| Change surface  | PR gate after fmt                                                                                                                                                                                                            | Nightly or targeted replay                                                                                                                                                                                              |
| --------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| ai-contract     | `lint datafusion-ai --no-default-features` and `tests datafusion-ai --no-default-features` (includes `--test contract`)                                                                                                      | `PROPTEST_CASES=4096 PROPTEST_RNG_SEED=2026 cargo test --locked -p datafusion-ai --no-default-features --test contract`. Mutate envelope or schema validation when it changes.                                          |
| ai-runtime      | `lint datafusion-ai` and `tests datafusion-ai` (default `runtime`). TODO add scripted retry, cancellation and row-alignment coverage to this crate as operators move.                                                        | Reuse `LLM_FUZZ_CASES=1000 LLM_FUZZ_SEED=2026 cargo test --locked -p datafusion --features llm --test llm differential_fuzz` until the runtime harness moves. Mutate retry and cancellation decisions when they change. |
| lake-storage    | `lint datafusion-lake` and `tests datafusion-lake`. Covers pinned Lance and Paimon scans, split replay, scalar pushdown, descriptor-only media projection and bounded resolution.                                            | Replay `cargo nextest run --locked --cargo-profile ci -p datafusion-lake --test readonly_scan --test split_contract` using generated local fixtures.                                                                    |
| dist-checkpoint | TODO `lint datafusion-dist` and `tests datafusion-dist`. The lake tables are read-only. Checkpoint and DLQ routing belong to `datafusion-dist`.                                                                              | TODO restart, duplicate replay and interruption tests at the checkpoint commit boundary.                                                                                                                                |
| lake-retrieval  | TODO `lint datafusion-lake` and `tests datafusion-lake`. Add exact-oracle, ties and empty-input cases.                                                                                                                       | TODO differential retrieval fuzz with fixed seeds and stored failing cases.                                                                                                                                             |
| dist-plan       | `lint datafusion-dist` and `tests datafusion-dist` (target `dist`). Covers map-only admission, versioned operator and file codecs, split fingerprints and metric isolation.                                                  | `DIST_FUZZ_CASES=36 DIST_FUZZ_SEED=2026 cargo nextest run --locked --cargo-profile ci -p datafusion-dist --test dist fuzz`. Use `DIST_FUZZ_REPLAY=dataset_seed,case_seed` for one failing case.                         |
| dist-runtime    | TODO `lint datafusion-dist` and `tests datafusion-dist`. Add worker loss, late completion and retry fencing cases.                                                                                                           | TODO replay worker-loss and retry fixtures on Ray.                                                                                                                                                                      |
| assembly        | Once core exposes `ai`, `lint datafusion --features ai` and `tests datafusion --features ai --lib --test core_integration`. TODO add feature wiring and end-to-end cases in core. Existing Rust CI covers the default build. | TODO AI-enabled SQL fixtures in `sqllogictests` and end-to-end composed execution.                                                                                                                                      |
| repo-infra      | `scripts/test-evidence.sh` and the workflow fmt gate. Run `./ci/scripts/doc_prettier_check.sh` for Markdown changes.                                                                                                         | Replay the three PR jobs with the toolchain or runner image being updated.                                                                                                                                              |

The PR workflow applies the crate commands to directories changed relative to the merge commit's first parent. Its evidence records the tested merge commit. AI changes run both feature modes. Lake, dist and assembly start their crate checks once their package or feature exists. Existing DataFusion workflows provide the upstream SQL, extended and legacy LLM suites. Schedule the larger fuzz and fault-injection suites in the nightly lane. Run focused mutations manually when a listed decision changes.

For focused mutation testing, run `cargo mutants -p datafusion-ai --no-default-features --file 'datafusion/ai/src/**/*.rs'` from the root. Substitute the affected package, feature flags and changed source paths for another surface. Retain the mutation outcomes and record the replay command and fixture seed.

## Lake scan contract

The lake providers pin the dataset version or table snapshot at open. Paimon reads the data manifest lists at scan planning or when index requirements need coverage validation at open. Each logical split carries `df.lake.split` v1 in `PartitionedFile.extensions` and leaves `PartitionedFile.range` unset. A split either omits row ranges for full coverage or carries a nonempty list. Replaying a file group consumes that extension at execution. Replay uses the pinned table and scan configuration.

The providers accept comparisons between a top-level scalar column and a typed literal, `IN`, `IS NULL`, `IS NOT NULL`, `AND`, `OR` and `NOT`. Both engines support comparisons on booleans, signed integers, strings and Arrow `Date32`. Lance also accepts unsigned integers and floats. Unsupported expressions presented to either `supports_filters_pushdown` or `scan` return `NotImplemented`. For strict admission before planning, callers pass the complete predicate list to `validate_prefilter`. DataFusion's optimizer handles volatile and subquery predicates above provider scans and can be configured to skip failed optimization rules.

Lance applies predicates in the fragment scanner. Paimon pushes predicates into split planning and reading, and applies a row filter before returning a batch if the native reader only guarantees pruning. Limits cross splits through a single file group when supplied to `scan`. A Paimon native limit is applied when it can run after exact filtering. Projection reads output columns and any scalar columns needed for filtering.

Top-level blobs project as `df.media_ref` v1. MIME takes the first declared value from field metadata, `fields.<column>.<key>`, or table metadata, with key order `df.media.mime`, `mime`, `content-type`. Any applicable blank declaration fails at open. Nested blob projections and Lance blob overlays are rejected at open. Lance preserves the storage block size configured by its builder. Its metadata tail prefetch can overlap inline payload bytes.

The media resolver uses a caller's object store registry, with optional scoped Paimon FileIO bindings. It validates the URI for every reference. An explicit zero-length range without an ETag returns empty bytes without storage IO. With an ETag, the resolver checks object metadata without reading payload bytes. Paimon FileIO bindings expose object length and reject a supplied ETag because their API does not expose ETags.

## Evidence

Start from a clean checkout with Bash 5, Git, GNU coreutils and `jq` on PATH. Use a separate lane name per command and a fresh output directory for a rerun.

```bash
scripts/evidence.sh --name ai-contract --output-dir evidence/ -- \
  cargo nextest run --locked --cargo-profile ci -p datafusion-ai --no-default-features --test contract
scripts/verify-changed-crates.sh "$(git merge-base origin/main HEAD)" evidence/
```

In the PR, record each command, exit code and evidence manifest or artifact path. Record the reason for every applicable check that did not run, including TODO targets, unavailable fixtures and nightly scope. For randomized runs, include the seed, case count and replay command.

`df-evidence-v1` captures the tested Git revision, dirty status, exact argument vector, Cargo feature flags and build profile, UTC start time, duration, tool versions, host and notes. `nextest_profile` records the nextest runner profile separately. Cargo settings describe direct Cargo invocations. Use `--notes` for fixture or environment details outside the argument vector.

Check the log against its SHA-256 from the manifest directory.

```bash
cd evidence
jq -r '.artifacts[] | "\(.sha256)  \(.path)"' "ai-contract-$(git rev-parse HEAD).json" | sha256sum --check
```

A failed command leaves its log and returns its exit code. A moved revision or an unexpected dirty tree prevents publication of a manifest. `--allow-dirty` marks local diagnostic evidence `source_dirty: true` if edits are present.

Local baseline `6a3f4adcfa7118f2615caae36ab6fbedc272ca3e` (`origin/main`) passed workspace fmt, strict core Clippy with `llm` and strict all-feature workspace Clippy. The new workflow uses blocking crate-scoped checks to keep the PR build bounded.
