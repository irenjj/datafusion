#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: See the NOTICE file distributed with this work for additional information regarding copyright ownership
# SPDX-FileContributor: Licensed to the Apache Software Foundation (ASF) under one or more contributor license agreements

set -euo pipefail

usage() {
    cat <<'EOF'
Usage: scripts/evidence.sh --name LANE --output-dir DIR [--allow-dirty] [--notes TEXT] -- COMMAND [ARG...]

Run COMMAND from the current directory inside a Git checkout. Save a log
and, on success, a df-evidence-v1 JSON manifest. Existing files are preserved.
EOF
}

die() {
    printf 'evidence: %s\n' "$*" >&2
    exit 1
}

name=''
output_dir=''
notes=''
allow_dirty=false
while (( $# )); do
    case "$1" in
        --name|--output-dir|--notes)
            (( $# >= 2 )) || die "missing argument for $1"
            case "$1" in
                --name) name=$2 ;;
                --output-dir) output_dir=$2 ;;
                --notes) notes=$2 ;;
            esac
            shift 2
            ;;
        --allow-dirty) allow_dirty=true; shift ;;
        -h|--help) usage; exit 0 ;;
        --) shift; break ;;
        *) die "unknown option $1 (see --help)" ;;
    esac
done
[[ $name =~ ^[a-zA-Z0-9][a-zA-Z0-9._-]*$ ]] || die "provide a valid --name (see --help)"
[[ -n $output_dir && $# -gt 0 ]] || die "provide --output-dir and a command (see --help)"
command -v jq >/dev/null || die "jq is required; install it and retry"

git_status() {
    git status --porcelain --untracked-files=all --ignore-submodules=none
}

revision=$(git rev-parse --verify HEAD) || die "run from a Git worktree with a commit"
initial_status=$(git_status) || die "could not read the worktree status"
if [[ -n $initial_status && $allow_dirty == false ]]; then
    die "worktree is dirty; commit changes or explicitly pass --allow-dirty"
fi

mkdir -p -- "$output_dir"
output_dir=$(cd -- "$output_dir" && pwd -P)
stem="$name-$revision"
log_path="$output_dir/$stem.log"
json_path="$output_dir/$stem.json"
for path in "$log_path" "$json_path"; do
    if [[ -e $path || -L $path ]]; then
        die "refusing to overwrite $path"
    fi
done

cargo_features=null
cargo_profile=''
nextest_profile=''
toolchain=()
capture_cargo_settings() {
    local executable=$1 mode arg value
    shift
    [[ ${executable##*/} == cargo ]] || return 0
    if [[ ${1:-} == +* ]]; then
        toolchain=("$1")
        shift
    fi
    mode=${1:-}
    case "$mode" in
        test|nextest|mutants) cargo_profile='test' ;;
        bench) cargo_profile=bench ;;
        build|check|clippy|doc|run) cargo_profile=dev ;;
        *) return 0 ;;
    esac
    shift
    if [[ $mode == nextest ]]; then
        nextest_profile=default
    fi
    local defaults=true all=false
    local -a requested=()
    while (( $# )); do
        arg=$1
        shift
        case "$arg" in
            --) break ;;
            --no-default-features) defaults=false ;;
            --all-features) all=true ;;
            --release) cargo_profile=release ;;
            --features=*) requested+=("${arg#*=}") ;;
            -F?*) requested+=("${arg#-F}") ;;
            --features|-F|--profile|--cargo-profile)
                (( $# )) || break
                value=$1
                shift
                case "$arg" in
                    --features|-F) requested+=("$value") ;;
                    --profile)
                        if [[ $mode == nextest ]]; then
                            nextest_profile=$value
                        else
                            cargo_profile=$value
                        fi
                        ;;
                    --cargo-profile) cargo_profile=$value ;;
                esac
                ;;
            --profile=*)
                if [[ $mode == nextest ]]; then
                    nextest_profile=${arg#*=}
                else
                    cargo_profile=${arg#*=}
                fi
                ;;
            --cargo-profile=*) cargo_profile=${arg#*=} ;;
        esac
    done
    cargo_features=$(jq -nc --argjson defaults "$defaults" --argjson all "$all" \
        --args '{defaults: $defaults, all: $all, requested: $ARGS.positional}' -- "${requested[@]}")
}

capture_cargo_settings "$@"
argv=$(jq -nc --args '$ARGS.positional' -- "$@")

version_line() {
    local version
    if version=$("$@" --version 2>/dev/null); then
        printf '%s' "${version%%$'\n'*}"
    fi
}

tool_versions=$(jq -nc \
    --arg bash "$BASH_VERSION" --arg git "$(version_line git)" --arg jq "$(version_line jq)" \
    --arg rustc "$(version_line rustc "${toolchain[@]}")" \
    --arg cargo "$(version_line cargo "${toolchain[@]}")" \
    --arg nextest "$(version_line cargo "${toolchain[@]}" nextest)" \
    --arg clippy "$(version_line cargo "${toolchain[@]}" clippy)" \
    '$ARGS.named | with_entries(select(.value != ""))')

now_ms() {
    local stamp=$EPOCHREALTIME micros
    micros=${stamp#*.}
    printf '%s\n' "$(( ${stamp%.*} * 1000 + 10#${micros:0:3} ))"
}

scratch=$(mktemp -d)
publish_dir=''
cleanup() {
    rm -rf -- "$scratch"
    if [[ -n $publish_dir ]]; then
        rm -rf -- "$publish_dir"
    fi
}
trap cleanup EXIT

started_at=$(date -u +%Y-%m-%dT%H:%M:%SZ)
start_ms=$(now_ms)
set +e
"$@" 2>&1 | tee -- "$scratch/command.log"
pipeline_status=("${PIPESTATUS[@]}")
set -e
duration_ms=$(( $(now_ms) - start_ms ))
if (( duration_ms < 0 )); then
    duration_ms=0
fi

source_issue=''
source_dirty=false
if (( pipeline_status[0] == 0 && pipeline_status[1] == 0 )); then
    if ! final_revision=$(git rev-parse --verify HEAD) || ! final_status=$(git_status); then
        source_issue='could not recheck the source after the command'
    elif [[ $revision != "$final_revision" ]]; then
        source_issue='HEAD changed during the command'
    elif [[ -n $final_status && $allow_dirty == false ]]; then
        source_issue='command left a dirty worktree'
    elif [[ -n $initial_status || -n $final_status ]]; then
        source_dirty=true
    fi
fi

publish_dir=$(mktemp -d "$output_dir/.evidence.XXXXXX")
mv -- "$scratch/command.log" "$publish_dir/command.log"
ln -T -- "$publish_dir/command.log" "$log_path" || die "could not publish $log_path"
if (( pipeline_status[0] != 0 )); then
    printf 'evidence: command exited %d after %d ms (log %s)\n' \
        "${pipeline_status[0]}" "$duration_ms" "$log_path" >&2
    exit "${pipeline_status[0]}"
fi
(( pipeline_status[1] == 0 )) || die "could not capture the full command log"
[[ -z $source_issue ]] || die "$source_issue (log $log_path)"

sha256=$(sha256sum < "$log_path")
jq -n --arg lane "$name" --arg revision "$revision" --argjson dirty "$source_dirty" \
    --argjson argv "$argv" --argjson features "$cargo_features" \
    --arg profile "$cargo_profile" --arg nextest_profile "$nextest_profile" \
    --arg started_at "$started_at" --argjson duration_ms "$duration_ms" \
    --argjson tool_versions "$tool_versions" --arg log "$stem.log" --arg sha256 "${sha256%% *}" \
    --arg notes "$notes" --arg host "$(uname -srm)" '
    {
        schema: "df-evidence-v1", lane: $lane, source_revision: $revision,
        source_dirty: $dirty, command: $argv, features: $features,
        profile: ($profile | if . == "" then null else . end),
        nextest_profile: ($nextest_profile | if . == "" then null else . end),
        exit_code: 0, started_at: $started_at, duration_ms: $duration_ms,
        tool_versions: $tool_versions, artifacts: [{path: $log, sha256: $sha256}],
        notes: $notes, host: $host
    }' > "$publish_dir/manifest.json"
ln -T -- "$publish_dir/manifest.json" "$json_path" || die "could not publish $json_path"
printf '%s\n' "$json_path"
