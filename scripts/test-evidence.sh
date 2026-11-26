#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: See the NOTICE file distributed with this work for additional information regarding copyright ownership
# SPDX-FileContributor: Licensed to the Apache Software Foundation (ASF) under one or more contributor license agreements

set -euo pipefail

evidence="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/evidence.sh"
scratch=$(mktemp -d)
trap 'rm -rf -- "$scratch"' EXIT
git init --quiet "$scratch/source"
cd -- "$scratch/source"
git -c user.name='Evidence self-test' -c user.email='evidence@example.com' \
    -c commit.gpgsign=false -c core.hooksPath=/dev/null \
    commit --quiet --allow-empty -m 'initial source'

output="$scratch/output directory"
revision=$(git rev-parse HEAD)
message="literal \$HOME, spaces, and \"quotes\""
command=(printf '%s\n' "$message")
"$evidence" --name success --output-dir "$output" --notes 'self-test' -- "${command[@]}"
manifest="$output/success-$revision.json"
log="$output/success-$revision.log"
hash=$(sha256sum < "$log")
argv=$(jq -nc --args '$ARGS.positional' -- "${command[@]}")
jq -e --arg revision "$revision" --arg log "${log##*/}" \
    --arg hash "${hash%% *}" --argjson argv "$argv" '
    .schema == "df-evidence-v1"
    and .source_revision == $revision and .source_dirty == false
    and .command == $argv and .exit_code == 0
    and .duration_ms >= 0 and .notes == "self-test"
    and .artifacts == [{path: $log, sha256: $hash}]
' "$manifest"
[[ $(cat "$log") == "$message" ]]
printf 'PASS success (exit 0, argv, source SHA and log checksum)\n'

expect_exit() {
    local expected=$1 actual=0
    shift
    "$@" || actual=$?
    if (( actual != expected )); then
        printf 'expected exit %s, got %s\n' "$expected" "$actual" >&2
        exit 1
    fi
}

printf 'untracked edit\n' > dirty
expect_exit 1 "$evidence" --name dirty --output-dir "$scratch/refused" -- true
[[ ! -e "$scratch/refused" ]]
printf 'PASS dirty worktree refused (exit 1)\n'
"$evidence" --allow-dirty --name dirty --output-dir "$output" -- true
jq -e '.source_dirty == true' "$output/dirty-$revision.json"
rm -- dirty

expect_exit 42 "$evidence" --name failure --output-dir "$output" -- bash -c \
    'printf "failure output\n" >&2; exit 42'
[[ ! -e "$output/failure-$revision.json" ]]
[[ $(cat "$output/failure-$revision.log") == 'failure output' ]]
printf 'PASS failed command (exit 42, log only)\n'

expect_exit 1 "$evidence" --name success --output-dir "$output" -- touch "$scratch/would-run"
[[ ! -e "$scratch/would-run" && $(sha256sum < "$log") == "$hash" ]]
printf 'PASS collision refused (exit 1, existing log preserved)\n'

expect_exit 1 "$evidence" --name changed --output-dir "$output" -- touch changed
[[ -f "$output/changed-$revision.log" && ! -e "$output/changed-$revision.json" ]]
printf 'PASS source changed by command (exit 1, log only)\n'
