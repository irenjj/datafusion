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

| Change surface  | PR gate after fmt                                                                                                                                                                                                                                          | Nightly or targeted replay                                                                                                                                                                      |
| --------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| ai-contract     | `lint datafusion-ai --no-default-features` and `tests datafusion-ai --no-default-features` (includes `--test contract`)                                                                                                                                    | `PROPTEST_CASES=4096 PROPTEST_RNG_SEED=2026 cargo test --locked -p datafusion-ai --no-default-features --test contract`. Mutate envelope or schema validation when it changes.                  |
| ai-runtime      | `lint datafusion-ai` and `tests datafusion-ai` with default `runtime`. Nextest runs the LLM unit tests and the `contract` and `runtime` targets.                                                                                                           | `cargo test --locked --profile ci -p datafusion-ai --test runtime` replays worker decode, checkpoint, concurrency, retry and cancellation cases.                                                |
| ai-http         | `lint datafusion-ai --features http` and `tests datafusion-ai --features http`. Loopback tests cover transport and registry execution.                                                                                                                     | `cargo nextest run --locked --cargo-profile ci -p datafusion-ai --features http --test http`. Check core wiring with `lint datafusion --features ai-http`.                                      |
| lake-storage    | `lint datafusion-lake` and `tests datafusion-lake`. Covers pinned Lance and Paimon scans, split replay, scalar pushdown, descriptor-only media projection and bounded resolution.                                                                          | Replay `cargo nextest run --locked --cargo-profile ci -p datafusion-lake --test readonly_scan --test split_contract` using generated local fixtures.                                            |
| dist-checkpoint | TODO `lint datafusion-dist` and `tests datafusion-dist`. The lake tables are read-only. Checkpoint and DLQ routing belong to `datafusion-dist`.                                                                                                            | TODO restart, duplicate replay and interruption tests at the checkpoint commit boundary.                                                                                                        |
| lake-retrieval  | TODO `lint datafusion-lake` and `tests datafusion-lake`. Add exact-oracle, ties and empty-input cases.                                                                                                                                                     | TODO differential retrieval fuzz with fixed seeds and stored failing cases.                                                                                                                     |
| dist-plan       | `lint datafusion-dist` and `tests datafusion-dist` (target `dist`). Covers map-only admission, zero-input file scan extensions, versioned operator and file codecs, split fingerprints and metric isolation.                                               | `DIST_FUZZ_CASES=36 DIST_FUZZ_SEED=2026 cargo nextest run --locked --cargo-profile ci -p datafusion-dist --test dist fuzz`. Use `DIST_FUZZ_REPLAY=dataset_seed,case_seed` for one failing case. |
| dist-ai         | `lint datafusion-dist --features ai` and `tests datafusion-dist --features ai`. Covers `LlmExec` transport, worker provider registration, speculation policy and metrics across attempts.                                                                  | `cargo nextest run --locked --cargo-profile ci -p datafusion-dist --features ai --test dist ai_codec`.                                                                                          |
| dist-runtime    | `lint datafusion-dist` and `tests datafusion-dist`. Covers deterministic worker recovery, attempt fencing and 256 seeded lease sequences.                                                                                                                  | [Replay high-sample proptest and both TLC configurations](#distributed-scheduler-replay).                                                                                                       |
| assembly        | `lint datafusion --features ai` and `tests datafusion --features ai --lib --test core_integration`. The manual LLM workflow will run `--test llm` with default features plus `llm` and with `--no-default-features --features sql,string_expressions,llm`. | `LLM_FUZZ_CASES=1000 LLM_FUZZ_SEED=2026 cargo test --locked -p datafusion --features llm --test llm differential_fuzz` replays SQL planning and execution against the row oracle.               |
| repo-infra      | `scripts/test-evidence.sh` and the workflow fmt gate. Run `./ci/scripts/doc_prettier_check.sh` for Markdown changes.                                                                                                                                       | Replay the three PR jobs with the toolchain or runner image being updated.                                                                                                                      |

The crate verifier selects gates for directories changed from the supplied base. AI changes run the contract, runtime and opt-in HTTP gates. Lake and dist changes run their crate gates. Core changes run the assembly lane with `ai`. Existing DataFusion workflows provide the SQL, extended and LLM suites. Schedule the larger fuzz and fault-injection suites in the nightly lane. Run focused mutations manually when a listed decision changes.

For focused mutation testing, run `cargo mutants -p datafusion-ai --no-default-features --file 'datafusion/ai/src/**/*.rs'` from the root. Substitute the affected package, feature flags and changed source paths for another surface. Retain the mutation outcomes and record the replay command and fixture seed.

## Lake scan contract

The lake providers pin the dataset version or table snapshot at open. Paimon reads the data manifest lists at scan planning or when index requirements need coverage validation at open. Each logical split carries `df.lake.split` v1 in `PartitionedFile.extensions` and leaves `PartitionedFile.range` unset. A split either omits row ranges for full coverage or carries a nonempty list. Replaying a file group consumes that extension at execution. Replay uses the pinned table and scan configuration.

The providers accept comparisons between a top-level scalar column and a typed literal, `IN`, `IS NULL`, `IS NOT NULL`, `AND`, `OR` and `NOT`. Both engines support comparisons on booleans, signed integers, strings and Arrow `Date32`. Lance also accepts unsigned integers and floats. Unsupported expressions presented to either `supports_filters_pushdown` or `scan` return `NotImplemented`. For strict admission before planning, callers pass the complete predicate list to `validate_prefilter`. DataFusion's optimizer handles volatile and subquery predicates above provider scans and can be configured to skip failed optimization rules.

Lance applies predicates in the fragment scanner. Paimon pushes predicates into split planning and reading, and applies a row filter before returning a batch if the native reader only guarantees pruning. Limits cross splits through a single file group when supplied to `scan`. A Paimon native limit is applied when it can run after exact filtering. Projection reads output columns and any scalar columns needed for filtering.

Top-level blobs project as `df.media_ref` v1. MIME takes the first declared value from field metadata, `fields.<column>.<key>`, or table metadata, with key order `df.media.mime`, `mime`, `content-type`. Any applicable blank declaration fails at open. Nested blob projections and Lance blob overlays are rejected at open. Lance preserves the storage block size configured by its builder. Its metadata tail prefetch can overlap inline payload bytes.

The media resolver uses a caller's object store registry, with optional scoped Paimon FileIO bindings. It validates the URI for every reference. An explicit zero-length range without an ETag returns empty bytes without storage IO. With an ETag, the resolver checks object metadata without reading payload bytes. Paimon FileIO bindings expose object length and reject a supplied ETag because their API does not expose ETags.

## Distributed scheduler replay

```bash
PROPTEST_CASES=256 DIST_LEASE_PROPTEST_SEED=3099 \
  cargo nextest run --locked --cargo-profile ci -p datafusion-dist --test dist scheduler_props
PROPTEST_CASES=4096 DIST_LEASE_PROPTEST_SEED=3099 \
  cargo nextest run --locked --cargo-profile ci -p datafusion-dist --test dist scheduler_props
```

The default seed is `0xc1b` (3099). If both seed variables are set, proptest's
`PROPTEST_RNG_SEED` overrides `DIST_LEASE_PROPTEST_SEED`.

With the [official TLC JAR in `/home/renjj/df-ci-data/tools`](../datafusion/dist/specs/README.md#replay-with-tlc),
run both exhaustive model configurations with deadlock checking enabled.

```bash
for config in LeasedSplitQueue LeasedSplitQueueHardCap; do
  java -XX:+UseParallelGC -cp /home/renjj/df-ci-data/tools/tla2tools.jar tlc2.TLC \
    -workers auto -noGenerateSpecTE -metadir "/tmp/tlc-states-$config" \
    -config "datafusion/dist/specs/$config.cfg" datafusion/dist/specs/LeasedSplitQueue.tla
done
```

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
