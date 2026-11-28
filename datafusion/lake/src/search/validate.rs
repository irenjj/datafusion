// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: See the NOTICE file distributed with this work for additional information regarding copyright ownership
// SPDX-FileContributor: Licensed to the Apache Software Foundation (ASF) under one or more contributor license agreements

use std::sync::Arc;

use arrow::datatypes::{DataType, SchemaRef};
use datafusion_catalog::Session;
use datafusion_common::{
    DFSchema, DataFusionError, Result, ScalarValue, TableReference,
    config::ConfigOptions,
    not_impl_err, plan_datafusion_err, plan_err,
    tree_node::{Transformed, TreeNode},
};
use datafusion_expr::planner::ContextProvider;
use datafusion_expr::{
    AggregateUDF, Expr, HigherOrderUDF, Operator, ScalarUDF, TableSource, WindowUDF,
};
use datafusion_sql::parser::DFParser;
use datafusion_sql::planner::{PlannerContext, SqlToRel};

use crate::{RequiredIndexSpec, filter};

use super::{PinnedLakeTable, SearchRequest};

const MAX_K: usize = 10_000;

pub(super) fn limits(k: usize, rrf_k: u64) -> Result<()> {
    if !(1..=MAX_K).contains(&k) {
        return plan_err!("Lake search k must be in 1..={MAX_K}");
    }
    if rrf_k == 0 {
        return plan_err!("Lake search rrf_k must be a positive integer");
    }
    Ok(())
}

pub(super) fn weight(value: f64, name: &str) -> Result<f64> {
    if !value.is_finite() || value <= 0.0 {
        return plan_err!("Lake search {name} must be finite and positive");
    }
    Ok(value)
}

pub(super) fn vector(json: &str) -> Result<Vec<f32>> {
    let values: Vec<f32> = serde_json::from_str(json).map_err(|err| {
        plan_datafusion_err!(
            "Lake search query vector must be a JSON array of f32: {err}"
        )
    })?;
    if values.iter().any(|value| !value.is_finite()) {
        return plan_err!("Lake search query vector values must be finite");
    }
    Ok(values)
}

pub(super) fn indices(table: &PinnedLakeTable, request: &SearchRequest) -> Result<()> {
    let schema = table.schema();
    let vector = schema
        .field_with_name(&request.vector_column)
        .map_err(|_| {
            plan_datafusion_err!(
                "Unknown lake search vector column {:?}",
                request.vector_column
            )
        })?;
    let DataType::FixedSizeList(item, dimension) = vector.data_type() else {
        return plan_err!("Lake search vector column must be FixedSizeList(Float32, n)");
    };
    if item.data_type() != &DataType::Float32 || *dimension <= 0 {
        return plan_err!("Lake search vector column must be FixedSizeList(Float32, n)");
    }
    if request.vector.len() != *dimension as usize {
        return plan_err!(
            "Lake search query vector dimension {} does not match index dimension {dimension}",
            request.vector.len()
        );
    }
    table.validate_index(&RequiredIndexSpec::Vector {
        column: request.vector_column.clone(),
        dimension: *dimension as u32,
    })?;
    if let Some(text) = &request.text {
        schema.field_with_name(&text.column).map_err(|_| {
            plan_datafusion_err!("Unknown lake search text column {:?}", text.column)
        })?;
        table.validate_index(&RequiredIndexSpec::FullText {
            column: text.column.clone(),
        })?;
    }
    Ok(())
}

pub(super) fn filter_sql(
    sql: &str,
    schema: &SchemaRef,
    session: &dyn Session,
) -> Result<Expr> {
    let ast = DFParser::parse_sql_into_expr(sql)
        .map_err(|err| plan_datafusion_err!("Invalid lake search filter_sql: {err}"))?;
    if ast.alias.is_some() {
        return not_impl_err!("Aliases in lake search filter_sql");
    }
    let df_schema = DFSchema::try_from(Arc::clone(schema))?;
    let context = FilterContext(session.config_options());
    let expr = SqlToRel::new(&context)
        .sql_to_expr(ast.expr, &df_schema, &mut PlannerContext::new())
        .map_err(|err| {
            DataFusionError::NotImplemented(format!("Lake search filter_sql: {err}"))
        })?;
    // SQL parses ordinary numeric literals as Int64 or Float64, regardless of
    // the comparison's field type. Preserve the value when typing the literal.
    let expr = expr
        .transform_up(|expr| {
            let changed = match expr {
                Expr::Cast(ref cast) if cast.field.data_type() == &DataType::Date32 => {
                    if let Expr::Literal(ScalarValue::Utf8(Some(value)), _) =
                        cast.expr.as_ref()
                    {
                        let date = ScalarValue::Utf8(Some(value.clone()))
                            .cast_to(&DataType::Date32)
                            .map_err(|err| {
                                plan_datafusion_err!(
                                    "Invalid lake search filter date: {err}"
                                )
                            })?;
                        Expr::Literal(date, None)
                    } else {
                        return Ok(Transformed::no(expr));
                    }
                }
                Expr::BinaryExpr(mut binary)
                    if matches!(
                        binary.op,
                        Operator::Eq
                            | Operator::NotEq
                            | Operator::Lt
                            | Operator::LtEq
                            | Operator::Gt
                            | Operator::GtEq
                    ) =>
                {
                    if !type_sql_literal(&binary.left, &mut binary.right, schema) {
                        type_sql_literal(&binary.right, &mut binary.left, schema);
                    }
                    Expr::BinaryExpr(binary)
                }
                Expr::InList(mut list) => {
                    for argument in &mut list.list {
                        type_sql_literal(&list.expr, argument, schema);
                    }
                    Expr::InList(list)
                }
                other => return Ok(Transformed::no(other)),
            };
            Ok(Transformed::yes(changed))
        })?
        .data;
    filter::normalize(&expr, schema)
}

fn type_sql_literal(column: &Expr, literal: &mut Expr, schema: &SchemaRef) -> bool {
    let Expr::Column(column) = column else {
        return false;
    };
    let Ok(field) = schema.field_with_name(&column.name) else {
        return false;
    };
    let Expr::Literal(value, _) = literal else {
        return false;
    };
    let source = value.data_type();
    let target = field.data_type();
    if &source != target
        && source.is_numeric()
        && target.is_numeric()
        && let Ok(converted) = value.cast_to(target)
        && converted
            .cast_to(&source)
            .as_ref()
            .is_ok_and(|restored| restored == value)
    {
        *literal = Expr::Literal(converted, None);
    }
    true
}

struct FilterContext<'a>(&'a ConfigOptions);

impl ContextProvider for FilterContext<'_> {
    fn get_table_source(&self, name: TableReference) -> Result<Arc<dyn TableSource>> {
        not_impl_err!("Table references in lake search filter_sql: {name}")
    }

    fn get_function_meta(&self, _: &str) -> Option<Arc<ScalarUDF>> {
        None
    }

    fn get_higher_order_meta(&self, _: &str) -> Option<Arc<HigherOrderUDF>> {
        None
    }

    fn get_aggregate_meta(&self, _: &str) -> Option<Arc<AggregateUDF>> {
        None
    }

    fn get_window_meta(&self, _: &str) -> Option<Arc<WindowUDF>> {
        None
    }

    fn get_variable_type(&self, _: &[String]) -> Option<DataType> {
        None
    }

    fn options(&self) -> &ConfigOptions {
        self.0
    }

    fn udf_names(&self) -> Vec<String> {
        Vec::new()
    }

    fn higher_order_function_names(&self) -> Vec<String> {
        Vec::new()
    }

    fn udaf_names(&self) -> Vec<String> {
        Vec::new()
    }

    fn udwf_names(&self) -> Vec<String> {
        Vec::new()
    }
}
