#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: See the NOTICE file distributed with this work for additional information regarding copyright ownership
# SPDX-FileContributor: Licensed to the Apache Software Foundation (ASF) under one or more contributor license agreements

set -euo pipefail
tree=$(cargo tree --locked -p datafusion-lake --prefix none -e normal,build,dev)
forbidden=$(awk '$1 == "datafusion" || $1 == "datafusion-dist" || $1 ~ /^aws-sdk-/ || ($1 ~ /^(arrow|parquet)(-|$)/ && $2 ~ /^v59\./)' <<< "$tree")
if [[ -n $forbidden ]]; then
    printf 'Forbidden datafusion-lake dependencies:\n%s\n' "$forbidden" >&2
    exit 1
fi
printf 'datafusion-lake dependency boundaries passed.\n'
