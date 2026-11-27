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
| lake-storage    | TODO `lint datafusion-lake` and `tests datafusion-lake` when the crate lands. Add storage and range cases. Replay `cargo test --locked -p datafusion --test parquet_integration` when changing shared Parquet behavior.      | TODO fault-injection fixtures for missing objects, schema drift and corrupt data.                                                                                                                                       |
| lake-checkpoint | TODO `lint datafusion-lake` and `tests datafusion-lake`. Add restart, duplicate replay and DLQ routing cases against `df.ai_error`.                                                                                          | TODO exhaust interruption points in the checkpoint and DLQ fixtures. Mutate checkpoint and routing decisions when they change.                                                                                          |
| lake-retrieval  | TODO `lint datafusion-lake` and `tests datafusion-lake`. Add exact-oracle, ties and empty-input cases.                                                                                                                       | TODO differential retrieval fuzz with fixed seeds and stored failing cases.                                                                                                                                             |
| dist-plan       | `lint datafusion-dist` and `tests datafusion-dist` (target `dist`). Covers map-only admission, versioned operator and file codecs, split fingerprints and metric isolation.                                                  | `DIST_FUZZ_CASES=36 DIST_FUZZ_SEED=2026 cargo nextest run --locked --cargo-profile ci -p datafusion-dist --test dist fuzz`. Use `DIST_FUZZ_REPLAY=dataset_seed,case_seed` for one failing case.                         |
| dist-runtime    | TODO `lint datafusion-dist` and `tests datafusion-dist`. Add worker loss, late completion and retry fencing cases.                                                                                                           | TODO replay worker-loss and retry fixtures on Ray.                                                                                                                                                                      |
| assembly        | Once core exposes `ai`, `lint datafusion --features ai` and `tests datafusion --features ai --lib --test core_integration`. TODO add feature wiring and end-to-end cases in core. Existing Rust CI covers the default build. | TODO AI-enabled SQL fixtures in `sqllogictests` and end-to-end composed execution.                                                                                                                                      |
| repo-infra      | `scripts/test-evidence.sh` and the workflow fmt gate. Run `./ci/scripts/doc_prettier_check.sh` for Markdown changes.                                                                                                         | Replay the three PR jobs with the toolchain or runner image being updated.                                                                                                                                              |

The PR workflow applies the crate commands to directories changed relative to the merge commit's first parent. Its evidence records the tested merge commit. AI changes run both feature modes. Lake, dist and assembly start their crate checks once their package or feature exists. Existing DataFusion workflows provide the upstream SQL, extended and legacy LLM suites. Schedule the larger fuzz and fault-injection suites in the nightly lane. Run focused mutations manually when a listed decision changes.

For focused mutation testing, run `cargo mutants -p datafusion-ai --no-default-features --file 'datafusion/ai/src/**/*.rs'` from the root. Substitute the affected package, feature flags and changed source paths for another surface. Retain the mutation outcomes and record the replay command and fixture seed.

## Evidence

Start from a clean checkout with Bash 5, Git, GNU coreutils and `jq` on PATH. Use the local two-node `df-ci` pool for routine verification and store evidence outside the checkout. Use a separate lane name per command and a fresh output directory for a rerun.

```bash
DF_CI=1 scripts/verify-changed-crates.sh "$(git merge-base origin/main HEAD)" ~/df-ci-data/evidence/my-task/
```

This command checks for oversized Git blobs before selecting crate gates. `DF_CI_BIN` overrides the default `/home/renjj/repo/datafusion-ci/bin/df-ci` path. To capture a single pool job, use its full log as the evidence output.

```bash
scripts/evidence.sh --name ai-contract --output-dir ~/df-ci-data/evidence/my-task/ -- \
  df-ci run --repo "$PWD" --ensure-submodules --cat-logs --command \
  'cargo nextest run --locked --cargo-profile ci -p datafusion-ai --no-default-features --test contract'
```

GitHub Actions verification runs on demand via `workflow_dispatch`. The `base_ref` input accepts a branch or commit and defaults to `main`.

```bash
gh workflow run ci.yml -R irenjj/datafusion --ref my-branch
```

In the PR, record each command, exit code and evidence manifest or artifact path. Record the reason for every applicable check that did not run, including TODO targets, unavailable fixtures and nightly scope. For randomized runs, include the seed, case count and replay command.

`df-evidence-v1` captures the tested Git revision, dirty status, exact argument vector, Cargo feature flags and build profile, UTC start time, duration, host tool versions and notes. `nextest_profile` records the nextest runner profile separately. Cargo settings describe direct invocations, invocations prefixed with `env`, and a single literal Cargo command passed to `df-ci run --command`. Use `--notes` for fixture, image or environment details outside the argument vector.

Check the log against its SHA-256 from the manifest directory.

```bash
revision=$(git rev-parse HEAD)
cd ~/df-ci-data/evidence/my-task/
jq -r '.artifacts[] | "\(.sha256)  \(.path)"' "ai-contract-$revision.json" | sha256sum --check
```

A failed command leaves its log and returns its exit code. A moved revision or an unexpected dirty tree prevents publication of a manifest. `--allow-dirty` marks local diagnostic evidence `source_dirty: true` if edits are present.
