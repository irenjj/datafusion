// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: See the NOTICE file distributed with this work for additional information regarding copyright ownership
// SPDX-FileContributor: Licensed to the Apache Software Foundation (ASF) under one or more contributor license agreements

use arrow::datatypes::Schema;
use datafusion_common::{Result, not_impl_err, plan_datafusion_err, plan_err};
use paimon::Table;
use paimon::arrow::arrow_field_to_paimon_type;
use paimon::spec::TableSchema;

use crate::media::external;

use super::plan::{BackfillPlan, BackfillWorkUnit, validate_columns};

pub(super) async fn prepare(
    table: &Table,
    columns: &Schema,
) -> Result<(BackfillPlan, Vec<BackfillWorkUnit>)> {
    validate_table(table.schema())?;
    let snapshot = table
        .snapshot_manager()
        .get_latest_snapshot()
        .await
        .map_err(external)?
        .ok_or_else(|| {
            plan_datafusion_err!("Paimon backfill requires an existing snapshot")
        })?;
    let schema = table
        .schema_manager()
        .schema(snapshot.schema_id())
        .await
        .map_err(external)?;
    validate_table(&schema)?;
    validate_columns(schema.fields().iter().map(|field| field.name()), columns)?;
    for column in columns.fields() {
        arrow_field_to_paimon_type(column).map_err(external)?;
    }

    not_impl_err!(
        concat!(
            "Paimon backfill at snapshot {} requires atomic schema-and-snapshot publication. ",
            "Publishing the schema file through FileIO would expose it to the catalog before ",
            "the snapshot commit. DataEvolutionPartialWriter and ",
            "TableCommit::commit_if_latest_snapshot are crate-private."
        ),
        snapshot.id()
    )
}

fn validate_table(schema: &TableSchema) -> Result<()> {
    let options = schema.core_options();
    if options.is_format_table() || !schema.primary_keys().is_empty() {
        return plan_err!("Paimon backfill requires a native table without primary keys");
    }
    if !options.data_evolution_enabled() || !options.row_tracking_enabled() {
        return plan_err!(
            "Paimon backfill requires data-evolution.enabled=true and row-tracking.enabled=true"
        );
    }
    Ok(())
}
