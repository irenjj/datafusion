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
| ai-http         | `lint datafusion-ai --features http` and `tests datafusion-ai --features http`. Loopback tests cover chat, embedding batches and vision data URIs.                                                                                                         | `cargo nextest run --locked --cargo-profile ci -p datafusion-ai --features http --test http`. Check core wiring with `lint datafusion --features ai-http`.                                      |
| core-ai-http    | `cargo nextest run --locked --cargo-profile ci -p datafusion --features ai-http --test llm` covers embedding and vision through SQL and generated Parquet.                                                                                                 | [Replay loopback TLS with a custom CA](#ai-custom-ca-loopback).                                                                                                                                 |
| lake-storage    | `lint datafusion-lake` and `tests datafusion-lake`. Covers pinned Lance and Paimon scans, split replay, scalar pushdown, descriptor-only media projection and bounded resolution.                                                                          | Replay `cargo nextest run --locked --cargo-profile ci -p datafusion-lake --test readonly_scan --test split_contract` using generated local fixtures.                                            |
| dist-checkpoint | `lint datafusion-dist --features ai` and `tests datafusion-dist --features ai`. The first durable writer wins. Covers immutable artifacts, `df.ai_error` routing and restart from partial checkpoints.                                                     | Run `tests datafusion-dist --all-features` for the combined feature build.                                                                                                                      |
| dist-worker     | `lint datafusion-dist`, `tests datafusion-dist` and `python3 -m unittest discover -s datafusion/dist/python -v`. Covers stateless execution, manifest reads, stage chaining, Ray scheduling and the offline SDK.                                           | [Run the Parquet worker protocol](#distributed-parquet-worker).                                                                                                                                 |
| lake-retrieval  | `lint datafusion-lake`, `tests datafusion-lake` and `bash datafusion/lake/tests/check-dependencies.sh`. Covers indexed retrieval, weighted fusion, late reads and dependency isolation.                                                                    | `cargo nextest run --locked --cargo-profile ci -p datafusion-lake -E 'test(search::)'` replays the retrieval contract with local fixtures.                                                      |
| dist-plan       | `lint datafusion-dist` and `tests datafusion-dist` (target `dist`). Covers map-only admission, zero-input file scan extensions, versioned operator and file codecs, split fingerprints and metric isolation.                                               | `DIST_FUZZ_CASES=36 DIST_FUZZ_SEED=2026 cargo nextest run --locked --cargo-profile ci -p datafusion-dist --test dist fuzz`. Use `DIST_FUZZ_REPLAY=dataset_seed,case_seed` for one failing case. |
| dist-ai         | `lint datafusion-dist --features ai` and `tests datafusion-dist --features ai`. Covers `LlmExec` transport, worker provider registration, speculation policy and metrics across attempts.                                                                  | `cargo nextest run --locked --cargo-profile ci -p datafusion-dist --features ai --test dist ai_codec`.                                                                                          |
| dist-ai-http    | `lint datafusion-dist --features ai-http` and `tests datafusion-dist --features ai-http`. Covers worker HTTP backend selection and keeps reqwest out of the `ai` dependency graph.                                                                         | [Compare feature dependencies](#distributed-parquet-worker).                                                                                                                                    |
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

## Lake retrieval contract

Open a snapshot with `LakeTableOptions::required_indices` listing the vector index and, for hybrid retrieval, the full-text index. Register the open handle in `LakeSearchCatalog` and register the two table functions with the host. Paimon retrieval requires Data Evolution, row tracking and global indexes. Primary-key search is unsupported.

```sql
SELECT id, _score FROM vector_search('items', 'embedding', '[1,2,3]', 10);

SELECT id, _score FROM hybrid_search(
  'items', 'embedding', '[1,2,3]', 'description', 'quartz',
  10, 60, 1.0, 2.0, 'active = true'
);
```

The UDTFs require literal `Expr` arguments at `TableFunctionImpl::call_with_args`. The host SQL planner can fold an argument before that call. The optional hybrid arguments are `rrf_k`, vector weight, text weight and `filter_sql`, in that order. The default `rrf_k` is 60 and each default weight is 1.0. The prefilter uses the scalar predicate contract above. An outer `WHERE` applies to the fused result after search.

Each route retrieves up to `k` indexed candidates. Fusion uses one-based ranks and `Float64` arithmetic, orders equal fused scores by ascending row key and returns `_score` as non-null `Float32`. The read plan fetches projected columns for the fused row ranges, restores fusion order and errors if a fused row key is missing. The exact vector oracles use single-partition IVF-flat indexes. Recall on approximate indexes depends on the engine and index configuration.

## Distributed Parquet worker

Use an output root dedicated to a job. On restart reuse its SQL, input inventory and `--row-key` (`--df.split_row_key`). The worker supports filesystem paths by default. Enable `--features s3` for `s3://bucket/key` URLs. End an S3 input prefix with `/` to read multiple files. Paths must be accessible to every worker, for example on a shared filesystem. Use the runtime `AWS_*` environment for S3. Enable `--features ai` for the mock backend. Enable `--features ai-http` to select `DF_DIST_LLM_BACKEND=openai`. `DF_DIST_LLM_PROVIDER` selects the provider name, and `OPENAI_BASE_URL`, `OPENAI_API_KEY` and optional `OPENAI_CA_CERT_FILE` configure the HTTP backend at runtime. Set `OPENAI_CA_CERT_FILE` to a PEM file on the planner and workers when the endpoint uses a custom CA.

With `--features ai-http`, set `DF_DIST_LLM_PROVIDERS` to a JSON object mapping provider names to HTTP settings. This selects the named configuration ahead of the single-provider environment settings.

```bash
export DF_DIST_LLM_PROVIDERS='{
  "qwen-openai": {
    "base_url": "https://chat.example.com/v1",
    "api_key_env": "DF_LLM_KEY_QWEN_OPENAI"
  },
  "gpt-gateway": {
    "base_url": "https://embed.example.com/v1",
    "api_key_env": "DF_LLM_KEY_GPT_GATEWAY"
  }
}'
export DF_DIST_LLM_PROVIDER=qwen-openai
```

Supply each API key through its named environment variable. The JSON contains variable names and endpoint configuration. An optional `ca_cert_env` names a variable containing PEM text or the path to a PEM file on the coordinator and workers. For the single-provider setup, `OPENAI_CA_CERT_PEM` supplies PEM text directly and takes precedence over `OPENAI_CA_CERT_FILE`. The backend validates the key at startup and reads it for each request.

In one SQL statement, use `"provider":"qwen-openai"` in the options for `llm()` or `vlm()`, and `"provider":"gpt-gateway"` in the options for `embed()`. Calls that omit `provider` use the name in `DF_DIST_LLM_PROVIDER` or the provider named `default` if that variable is unset. When the map has no `default` entry and no default name is configured, supply the provider name in each call.

The `df.llm.exec` v2 codec carries the Chat, Embed or Vision task kind and the Vision media expression. Workers resolve media URIs and `df.media_ref` ranges through their session's object store registry. v2 workers can decode v1 Chat jobs. Deploy v2 workers before submitting v2 plans.

For a custom HTTP S3 endpoint, set `AWS_ENDPOINT`, `AWS_ALLOW_HTTP=true`, `AWS_VIRTUAL_HOSTED_STYLE_REQUEST=false` and `AWS_REGION`. Supply credentials through the AWS runtime environment. Input and output URLs each select their own bucket.

```bash
dist-worker run-local --sql 'SELECT id FROM input' --table input=/shared/input --output-root /shared/out

dist-worker plan --sql 'SELECT id FROM input' --table input=/shared/input --out stage-job.json --output-root /shared/out
dist-worker run-split --job stage-job.json --split-id 0 --attempt-id 0 > attempt.json
dist-worker commit-split --job stage-job.json --artifact attempt.json
# Run and commit each remaining pending split before finalizing.
dist-worker finalize-stage --job stage-job.json

python3 datafusion/dist/python/ray_driver.py --worker dist-worker --ray-address auto \\
  --sql 'SELECT id FROM input' --table input=/shared/input --job stage-job.json --output-root /shared/out
```

Run the driver on the Ray head to use the cluster's Python and Ray versions. On the `ray-vlm` head the driver runs with Python 3.10.18 and Ray 2.47.1. Copy only the driver script to the head and stream the worker environment from the caller's stdin.

```bash
kubectl -n ray-vlm cp datafusion/dist/python/ray_driver.py vlm-head-tfczb:/tmp/ray_driver.py -c ray-head
kubectl -n ray-vlm exec -i vlm-head-tfczb -c ray-head -- \\
  python3 -B /tmp/ray_driver.py --ray-address auto --secrets-stdin \\
  --worker s3://artifacts/dist-worker --sql 'SELECT id FROM input' \\
  --table input=s3://datasets/input/ --job /tmp/stage-job.json \\
  --output-root s3://results/df-dist-run < /tmp/secrets.json
```

With `--ray-address auto` or no address, the driver calls `ray.init()`. Ray Client addresses such as `ray://head:10001` require matching Ray versions and matching Python major and minor versions. For example, Python 3.12 cannot connect as a Ray Client to Python 3.10. HTTP dashboard addresses belong to `ray job submit --address http://head:8265`, whose entrypoint can run the driver with `--ray-address auto`.

The input is one JSON object mapping environment names to strings. Include the worker's S3 settings, provider configuration, credential values, and any CA variables. Use `--secrets-file PATH` with a regular `0600` file as an alternative to `--secrets-stdin`. In Ray Jobs mode the file must be available to the coordinator outside the submitted working directory. With neither option, the driver can collect recognized worker settings from its existing environment.

The driver puts the environment object into Ray's object store before planning. The coordinator and tasks use it in their child process environments. The task decorator omits `runtime_env` because Ray's agent can log it at INFO level. Keep credentials out of the job submission runtime environment as well. Stage jobs carry plans and store locations. Data in the Ray object store is subject to the cluster's access and object spilling configuration.

For a shared worker binary, pass its local path or command name. With `--worker s3://bucket/key`, both the coordinator and each task download the executable into a temporary directory using boto3 first and the AWS CLI as a fallback. The binary must run on both the coordinator and worker nodes.

`plan` writes the complete inventory and a `pending_splits` list of split and first-attempt IDs. Use the `pending_splits` list when resuming. The driver checks attempt IDs before publication. Each accepted receipt names immutable files and carries byte counts, row counts and hashes. The sink streams the full candidate files to verify their hashes during commit and also reads an existing winner before accepting it. It validates the marker identity and file presence during resume and finalization. `COMMITTED.json` selects the first durable winner per split. Later attempts accept that winner after verification, even if their output differs. `STAGE_MANIFEST.json` lists those winners in split ID order. Read the Parquet files named by the manifest.

Use `--stage-table name=PATH` or `--stage-dlq-table name=PATH` in `plan` to bind an upstream stage. The job keeps the existing map-only plan requirements. `query-stage` registers the two relations as `data` and `dlq` and streams one JSON object per row to stdout. Use a job output root with one finalized stage or its explicit `stages/<plan_fingerprint>/` directory. A job root with multiple finalized stages requires an explicit stage directory.

```bash
dist-worker query-stage --output-root /shared/out --sql 'SELECT count(*) AS rows FROM data'
dist-worker query-stage --output-root /shared/out --sql 'SELECT count(*) AS rows FROM dlq'
dist-worker plan --sql 'SELECT id + 1 AS id FROM upstream' \
  --stage-table upstream=/shared/out --out downstream.json --output-root /shared/downstream
```

Rust callers can register the providers returned by `StageOutputReader::data_table()` and `StageOutputReader::dlq_table()` in the session used to open the reader. The manifest paths must match the winning attempt names and the readers check every winning file's presence and every data file's byte count. The Parquet footers provide the schema, including for zero-row data files. When the manifest names zero DLQ files, `dlq` uses the data schema and returns zero rows. `COUNT(*)` works for either DLQ schema. A manifest with zero splits has no data footer and the Rust reader returns a schema inference error. SQL filters run above the provider scan. The scan supports projection and limit. Keep the published files immutable for the duration of the query.

Add `datafusion/dist/python` to `PYTHONPATH` to use the SDK. `run_stage()` launches the Ray driver with the current Python interpreter and worker environment. It returns the metrics and winner paths from finalization. `StageResult.load()` accepts the same root as `query-stage`, and `StageResult.from_manifest()` accepts a specific manifest file or S3 object. Install `boto3` to load an S3 manifest. The result exposes only the selected paths, with absolute local paths or S3 URIs in split ID order. Storage engines open those paths directly.

```python
from stage_sdk import StageResult, run_stage

result = run_stage(
    sql="SELECT id FROM input",
    tables={"input": "/shared/input"},
    output_root="/shared/out",
    worker="dist-worker",
    ray_address="auto",
)
assert StageResult.load("/shared/out").data_uris() == result.data_uris()
print(result.output_rows, result.dlq_uris())
```

Worker receipts carry `input_tokens`, `output_tokens`, `llm_requests` and `llm_retries` under `metrics`. Token counts use the usage reported by the provider for successful responses. A response without usage contributes zero tokens. Requests count backend attempts, including retries. The manifest keeps the counters under each winning split's `metrics` and sums them at the top level. Older receipts and checkpoints default missing usage counters to zero.

`llm_usage_by_provider` records `input_tokens`, `output_tokens` and `requests` by resolved provider name in split metrics and in the stage manifest. The stage map merges winning splits. Older checkpoints without the map still contribute their aggregate counters. In a stage resumed from those checkpoints, provider totals cover the splits that recorded provider attribution. Provider names use ASCII letters, digits, underscores, periods or hyphens.

`df.ai_error` columns require an explicit non-null output row key. A row with any failed envelope is routed to the DLQ, with one entry per failed column and stable ordering by row key and column name. A successful row retains the Arrow value type. The Python suite uses an injected executor and requires neither Ray nor a network connection.

```bash
cargo nextest run --locked --cargo-profile ci -p datafusion-dist --bin dist-worker
cargo nextest run --locked --cargo-profile ci -p datafusion-dist --features ai -E 'test(manifest_table::) + test(query_stage_and_plan_next_stage)'
cargo nextest run --locked --cargo-profile ci -p datafusion-dist --features ai --test worker multimodal_parquet_outputs
cargo nextest run --locked --cargo-profile ci -p datafusion-dist --features ai-http --test worker openai_ca_cert
cargo tree --locked -p datafusion-dist --features ai
cargo tree --locked -p datafusion-dist --features ai-http
python3 -m unittest discover -s datafusion/dist/python -v
```

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

## AI custom CA loopback

Python 3 and the OpenSSL CLI generate an ephemeral CA and server certificate. The test checks that the default trust store rejects the server and that `ca_cert_pem` enables a successful request.

```bash
cargo test --locked --profile ci -p datafusion-ai --features http --test http -- --ignored custom_ca_controls_trust
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
