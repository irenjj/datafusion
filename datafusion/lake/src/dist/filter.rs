// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: See the NOTICE file distributed with this work for additional information regarding copyright ownership
// SPDX-FileContributor: Licensed to the Apache Software Foundation (ASF) under one or more contributor license agreements

use datafusion_common::{Column, Result, ScalarValue, plan_err};
use datafusion_expr::expr::BinaryExpr;
use datafusion_expr::{Expr, Operator};
use serde::{Deserialize, Serialize};

#[derive(Serialize, Deserialize)]
#[serde(rename_all = "snake_case", deny_unknown_fields)]
pub(super) enum ScanFilter {
    And(Box<Self>, Box<Self>),
    Or(Box<Self>, Box<Self>),
    Not(Box<Self>),
    Compare {
        column: String,
        op: Comparison,
        value: Literal,
    },
    InList {
        column: String,
        values: Vec<Literal>,
        negated: bool,
    },
    IsNull(String),
    IsNotNull(String),
    Boolean(Option<bool>),
}

impl ScanFilter {
    pub(super) fn from_expr(expr: &Expr) -> Result<Self> {
        Self::from_expr_at_depth(expr, 0)
    }

    fn from_expr_at_depth(expr: &Expr, depth: usize) -> Result<Self> {
        // Keep encoded predicates within the JSON decoder's recursion limit.
        if depth >= 48 {
            return plan_err!("Lake scan filter exceeds maximum nesting");
        }
        let child = |expr| Self::from_expr_at_depth(expr, depth + 1).map(Box::new);
        Ok(match expr {
            Expr::BinaryExpr(expr) => match expr.op {
                Operator::And => Self::And(child(&expr.left)?, child(&expr.right)?),
                Operator::Or => Self::Or(child(&expr.left)?, child(&expr.right)?),
                op => Self::Compare {
                    column: column_name(&expr.left)?,
                    op: Comparison::try_from(op)?,
                    value: Literal::from_expr(&expr.right)?,
                },
            },
            Expr::InList(expr) => Self::InList {
                column: column_name(&expr.expr)?,
                values: expr
                    .list
                    .iter()
                    .map(Literal::from_expr)
                    .collect::<Result<_>>()?,
                negated: expr.negated,
            },
            Expr::IsNull(expr) => Self::IsNull(column_name(expr)?),
            Expr::IsNotNull(expr) => Self::IsNotNull(column_name(expr)?),
            Expr::Not(expr) => Self::Not(child(expr)?),
            Expr::Literal(ScalarValue::Boolean(value), _) => Self::Boolean(*value),
            _ => return plan_err!("Unsupported lake scan filter"),
        })
    }

    pub(super) fn into_expr(self) -> Expr {
        let column = |name| Expr::Column(Column::from_name(name));
        match self {
            Self::And(left, right) => left.into_expr().and(right.into_expr()),
            Self::Or(left, right) => left.into_expr().or(right.into_expr()),
            Self::Not(expr) => Expr::Not(Box::new(expr.into_expr())),
            Self::Compare {
                column: name,
                op,
                value,
            } => Expr::BinaryExpr(BinaryExpr::new(
                Box::new(column(name)),
                op.into(),
                Box::new(value.into_expr()),
            )),
            Self::InList {
                column: name,
                values,
                negated,
            } => column(name).in_list(
                values.into_iter().map(Literal::into_expr).collect(),
                negated,
            ),
            Self::IsNull(name) => column(name).is_null(),
            Self::IsNotNull(name) => column(name).is_not_null(),
            Self::Boolean(value) => Expr::Literal(ScalarValue::Boolean(value), None),
        }
    }
}

fn column_name(expr: &Expr) -> Result<String> {
    let Expr::Column(column) = expr else {
        return plan_err!("Expected a column in the lake scan filter");
    };
    Ok(column.name.clone())
}

#[derive(Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub(super) enum Comparison {
    Eq,
    NotEq,
    Lt,
    LtEq,
    Gt,
    GtEq,
}

impl TryFrom<Operator> for Comparison {
    type Error = datafusion_common::DataFusionError;

    fn try_from(op: Operator) -> Result<Self> {
        Ok(match op {
            Operator::Eq => Self::Eq,
            Operator::NotEq => Self::NotEq,
            Operator::Lt => Self::Lt,
            Operator::LtEq => Self::LtEq,
            Operator::Gt => Self::Gt,
            Operator::GtEq => Self::GtEq,
            _ => return plan_err!("Unsupported lake scan comparison operator {op}"),
        })
    }
}

impl From<Comparison> for Operator {
    fn from(op: Comparison) -> Self {
        match op {
            Comparison::Eq => Self::Eq,
            Comparison::NotEq => Self::NotEq,
            Comparison::Lt => Self::Lt,
            Comparison::LtEq => Self::LtEq,
            Comparison::Gt => Self::Gt,
            Comparison::GtEq => Self::GtEq,
        }
    }
}

#[derive(Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub(super) enum Literal {
    Boolean(Option<bool>),
    Int8(Option<i8>),
    Int16(Option<i16>),
    Int32(Option<i32>),
    Int64(Option<i64>),
    UInt8(Option<u8>),
    UInt16(Option<u16>),
    UInt32(Option<u32>),
    UInt64(Option<u64>),
    Float32(Option<u32>),
    Float64(Option<u64>),
    Utf8(Option<String>),
    LargeUtf8(Option<String>),
    Utf8View(Option<String>),
    Date32(Option<i32>),
}

impl Literal {
    fn from_expr(expr: &Expr) -> Result<Self> {
        let Expr::Literal(value, _) = expr else {
            return plan_err!("Expected a literal in the lake scan filter");
        };
        Ok(match value {
            ScalarValue::Boolean(value) => Self::Boolean(*value),
            ScalarValue::Int8(value) => Self::Int8(*value),
            ScalarValue::Int16(value) => Self::Int16(*value),
            ScalarValue::Int32(value) => Self::Int32(*value),
            ScalarValue::Int64(value) => Self::Int64(*value),
            ScalarValue::UInt8(value) => Self::UInt8(*value),
            ScalarValue::UInt16(value) => Self::UInt16(*value),
            ScalarValue::UInt32(value) => Self::UInt32(*value),
            ScalarValue::UInt64(value) => Self::UInt64(*value),
            // JSON numbers cannot distinguish NaN and infinity from a null literal.
            ScalarValue::Float32(value) => Self::Float32(value.map(f32::to_bits)),
            ScalarValue::Float64(value) => Self::Float64(value.map(f64::to_bits)),
            ScalarValue::Utf8(value) => Self::Utf8(value.clone()),
            ScalarValue::LargeUtf8(value) => Self::LargeUtf8(value.clone()),
            ScalarValue::Utf8View(value) => Self::Utf8View(value.clone()),
            ScalarValue::Date32(value) => Self::Date32(*value),
            _ => return plan_err!("Unsupported scalar in the lake scan filter"),
        })
    }

    fn into_expr(self) -> Expr {
        let value = match self {
            Self::Boolean(value) => ScalarValue::Boolean(value),
            Self::Int8(value) => ScalarValue::Int8(value),
            Self::Int16(value) => ScalarValue::Int16(value),
            Self::Int32(value) => ScalarValue::Int32(value),
            Self::Int64(value) => ScalarValue::Int64(value),
            Self::UInt8(value) => ScalarValue::UInt8(value),
            Self::UInt16(value) => ScalarValue::UInt16(value),
            Self::UInt32(value) => ScalarValue::UInt32(value),
            Self::UInt64(value) => ScalarValue::UInt64(value),
            Self::Float32(value) => ScalarValue::Float32(value.map(f32::from_bits)),
            Self::Float64(value) => ScalarValue::Float64(value.map(f64::from_bits)),
            Self::Utf8(value) => ScalarValue::Utf8(value),
            Self::LargeUtf8(value) => ScalarValue::LargeUtf8(value),
            Self::Utf8View(value) => ScalarValue::Utf8View(value),
            Self::Date32(value) => ScalarValue::Date32(value),
        };
        Expr::Literal(value, None)
    }
}
