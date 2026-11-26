#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: See the NOTICE file distributed with this work for additional information regarding copyright ownership
# SPDX-FileContributor: Licensed to the Apache Software Foundation (ASF) under one or more contributor license agreements

set -euo pipefail

if (( $# != 2 )); then
    printf 'Usage: scripts/verify-changed-crates.sh BASE_REVISION OUTPUT_DIR\n' >&2
    exit 2
fi

base=$(git rev-parse --verify "$1^{commit}")
output_dir=$2
failed=0
selected=0

run_gate() {
    local name=$1
    shift
    scripts/evidence.sh --name "$name" --output-dir "$output_dir" -- "$@" || failed=1
}

verify_crate() {
    local lane=$1 package=$2
    shift 2
    local -a targets=()
    if [[ $package == datafusion ]]; then
        targets=(--lib --test core_integration)
    fi
    run_gate "$lane-clippy" cargo clippy --locked -p "$package" "$@" --all-targets -- -D warnings
    run_gate "$lane-nextest" cargo nextest run --locked --cargo-profile ci -p "$package" "$@" "${targets[@]}"
}

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
