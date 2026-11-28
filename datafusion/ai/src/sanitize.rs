// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: See the NOTICE file distributed with this work for additional information regarding copyright ownership
// SPDX-FileContributor: Licensed to the Apache Software Foundation (ASF) under one or more contributor license agreements

pub(super) fn single_line(value: &str) -> String {
    value
        .split(|ch: char| {
            ch.is_control() || ch.is_whitespace()
                || matches!(ch, '\u{061c}' | '\u{200b}'..='\u{200f}' | '\u{202a}'..='\u{202e}' | '\u{2060}'..='\u{206f}' | '\u{feff}')
        })
        .filter(|part| !part.is_empty())
        .collect::<Vec<_>>()
        .join(" ")
}

pub(super) fn sanitized_detail(value: &str) -> Option<String> {
    let detail = single_line(value);
    (!detail.is_empty() && !detail.contains("://")).then_some(detail)
}
