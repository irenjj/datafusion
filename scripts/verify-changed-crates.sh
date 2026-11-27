#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: See the NOTICE file distributed with this work for additional information regarding copyright ownership
# SPDX-FileContributor: Licensed to the Apache Software Foundation (ASF) under one or more contributor license agreements

set -euo pipefail

if (( $# != 2 )); then
    printf 'Usage: scripts/verify-changed-crates.sh BASE_REVISION OUTPUT_DIR\n' >&2
    exit 2
fi

base=$(git rev-parse --verify --end-of-options "$1^{commit}")
root_dir=$(git rev-parse --show-toplevel)
output_dir=$2
df_ci_bin=${DF_CI_BIN:-/home/renjj/repo/datafusion-ci/bin/df-ci}
failed=0
selected=0

run_gate() {
    local name=$1
    shift
    scripts/evidence.sh --name "$name" --output-dir "$output_dir" -- "$@" || failed=1
}

run_cargo_gate() {
    local name=$1 command
    shift
    if [[ ${DF_CI:-0} == 1 ]]; then
        printf -v command '%q ' cargo "$@"
        run_gate "$name" "$df_ci_bin" run --repo "$root_dir" --ensure-submodules --cat-logs --command "$command"
    else
        run_gate "$name" cargo "$@"
    fi
}

verify_crate() {
    local lane=$1 package=$2
    shift 2
    local -a targets=()
    if [[ $package == datafusion ]]; then
        targets=(--lib --test core_integration)
    fi
    run_cargo_gate "$lane-clippy" clippy --locked -p "$package" "$@" --all-targets -- -D warnings
    run_cargo_gate "$lane-nextest" nextest run --locked --cargo-profile ci -p "$package" "$@" "${targets[@]}"
}

run_gate "large-files" ./ci/scripts/check_large_files.sh --base "$base" --head HEAD

for module in ai lake dist core; do
    directory="datafusion/$module"
    changed=$(git diff --name-only --no-renames "$base" HEAD -- "$directory/")
    if [[ -z $changed || ! -f $directory/Cargo.toml ]]; then
        continue
    fi
    case "$module" in
        ai)
            verify_crate ai-contract datafusion-ai --no-default-features
            verify_crate ai-runtime datafusion-ai
            ;;
        core)
            has_ai=$(cargo metadata --locked --no-deps --format-version 1 | jq -r '
                any(.packages[]; .name == "datafusion" and (.features | has("ai")))
            ')
            if [[ $has_ai != true ]]; then
                printf 'Assembly lane will start when datafusion enables the ai feature.\n'
                continue
            fi
            verify_crate assembly datafusion --features ai
            ;;
        *) verify_crate "$module" "datafusion-$module" ;;
    esac
    selected=1
done
if (( selected == 0 )); then
    printf 'No verification crate checks selected for %s..HEAD.\n' "$base"
fi
exit "$failed"
