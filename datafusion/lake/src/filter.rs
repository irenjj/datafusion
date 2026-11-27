// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: See the NOTICE file distributed with this work for additional information regarding copyright ownership
// SPDX-FileContributor: Licensed to the Apache Software Foundation (ASF) under one or more contributor license agreements

use arrow::datatypes::{DataType, Schema};
use datafusion_common::{Column, DataFusionError, Result, ScalarValue, not_impl_err};
use datafusion_expr::expr::{BinaryExpr, InList};
use datafusion_expr::{Expr, Operator};
use paimon::spec::{Datum, Predicate, PredicateBuilder, PredicateOperator};

pub(crate) fn normalize(expr: &Expr, schema: &Schema) -> Result<Expr> {
    let normalized = match expr {
        Expr::BinaryExpr(binary) if matches!(binary.op, Operator::And | Operator::Or) => {
            Expr::BinaryExpr(BinaryExpr::new(
                Box::new(normalize(&binary.left, schema)?),
                binary.op,
                Box::new(normalize(&binary.right, schema)?),
            ))
        }
        Expr::BinaryExpr(binary) => {
            let (column, value, op) = if matches!(binary.left.as_ref(), Expr::Column(_)) {
                (binary.left.as_ref(), binary.right.as_ref(), binary.op)
            } else {
                (
                    binary.right.as_ref(),
                    binary.left.as_ref(),
                    binary.op.swap().ok_or_else(|| unsupported(expr))?,
                )
            };
            comparison_op(op).ok_or_else(|| unsupported(expr))?;
            let (column, data_type) = scalar_column(column, schema)?;
            Expr::BinaryExpr(BinaryExpr::new(
                Box::new(column),
                op,
                Box::new(literal(value, data_type)?),
            ))
        }
        Expr::InList(in_list) if !in_list.list.is_empty() => {
            let (column, data_type) = scalar_column(&in_list.expr, schema)?;
            let list = in_list
                .list
                .iter()
                .map(|arg| literal(arg, data_type))
                .collect::<Result<Vec<_>>>()?;
            Expr::InList(InList::new(Box::new(column), list, in_list.negated))
        }
        Expr::IsNull(inner) => Expr::IsNull(Box::new(scalar_column(inner, schema)?.0)),
        Expr::IsNotNull(inner) => {
            Expr::IsNotNull(Box::new(scalar_column(inner, schema)?.0))
        }
        Expr::Not(inner) => Expr::Not(Box::new(normalize(inner, schema)?)),
        Expr::Literal(ScalarValue::Boolean(_), _) => expr.clone(),
        _ => return Err(unsupported(expr)),
    };
    Ok(normalized)
}

fn scalar_column<'a>(expr: &Expr, schema: &'a Schema) -> Result<(Expr, &'a DataType)> {
    let Expr::Column(column) = expr else {
        return Err(unsupported(expr));
    };
    let field = schema
        .field_with_name(&column.name)
        .map_err(|_| unsupported(expr))?;
    let data_type = field.data_type();
    if !matches!(
        data_type,
        DataType::Boolean
            | DataType::Int8
            | DataType::Int16
            | DataType::Int32
            | DataType::Int64
            | DataType::UInt8
            | DataType::UInt16
            | DataType::UInt32
            | DataType::UInt64
            | DataType::Float32
            | DataType::Float64
            | DataType::Utf8
            | DataType::LargeUtf8
            | DataType::Utf8View
            | DataType::Date32
    ) {
        return Err(unsupported(expr));
    }
    Ok((Expr::Column(Column::from_name(&column.name)), data_type))
}

fn literal(expr: &Expr, data_type: &DataType) -> Result<Expr> {
    let Expr::Literal(value, _) = expr else {
        return Err(unsupported(expr));
    };
    let value = if &value.data_type() == data_type {
        value.clone()
    } else if value.is_null()
        || (matches!(
            data_type,
            DataType::Utf8 | DataType::LargeUtf8 | DataType::Utf8View
        ) && matches!(
            value,
            ScalarValue::Utf8(_) | ScalarValue::LargeUtf8(_) | ScalarValue::Utf8View(_)
        ))
    {
        value.cast_to(data_type).map_err(|_| unsupported(expr))?
    } else {
        return Err(unsupported(expr));
    };
    Ok(Expr::Literal(value, None))
}

fn unsupported(expr: &Expr) -> DataFusionError {
    DataFusionError::NotImplemented(format!("Lake cannot push down scalar filter {expr}"))
}

fn comparison_op(op: Operator) -> Option<PredicateOperator> {
    Some(match op {
        Operator::Eq => PredicateOperator::Eq,
        Operator::NotEq => PredicateOperator::NotEq,
        Operator::Lt => PredicateOperator::Lt,
        Operator::LtEq => PredicateOperator::LtEq,
        Operator::Gt => PredicateOperator::Gt,
        Operator::GtEq => PredicateOperator::GtEq,
        _ => return None,
    })
}

// A negative subtree selects rows on which its operand is SQL FALSE. Paimon
// predicates are two-valued, so negating the set of TRUE rows would admit NULL.
pub(crate) fn paimon(
    expr: &Expr,
    builder: &PredicateBuilder,
    negated: bool,
) -> Result<Predicate> {
    let (column, op, values) = match expr {
        Expr::BinaryExpr(binary) if matches!(binary.op, Operator::And | Operator::Or) => {
            let children = vec![
                paimon(&binary.left, builder, negated)?,
                paimon(&binary.right, builder, negated)?,
            ];
            return Ok(if (binary.op == Operator::And) ^ negated {
                Predicate::and(children)
            } else {
                Predicate::or(children)
            });
        }
        Expr::Not(inner) => return paimon(inner, builder, !negated),
        Expr::Literal(ScalarValue::Boolean(value), _) => {
            return Ok(if value.is_some_and(|value| value ^ negated) {
                Predicate::AlwaysTrue
            } else {
                Predicate::AlwaysFalse
            });
        }
        Expr::BinaryExpr(binary) => {
            let values = paimon_literal(&binary.right)?;
            let Some(value) = values else {
                return Ok(Predicate::AlwaysFalse);
            };
            let op = comparison_op(binary.op).ok_or_else(|| unsupported(expr))?;
            (binary.left.as_ref(), op, vec![value])
        }
        Expr::InList(in_list) => {
            let values = in_list
                .list
                .iter()
                .map(paimon_literal)
                .collect::<Result<Vec<_>>>()?;
            if (in_list.negated ^ negated) && values.iter().any(Option::is_none) {
                return Ok(Predicate::AlwaysFalse);
            }
            let values: Vec<_> = values.into_iter().flatten().collect();
            if values.is_empty() {
                return Ok(Predicate::AlwaysFalse);
            }
            let op = if in_list.negated {
                PredicateOperator::NotIn
            } else {
                PredicateOperator::In
            };
            (in_list.expr.as_ref(), op, values)
        }
        Expr::IsNull(column) => (column.as_ref(), PredicateOperator::IsNull, vec![]),
        Expr::IsNotNull(column) => {
            (column.as_ref(), PredicateOperator::IsNotNull, vec![])
        }
        _ => return Err(unsupported(expr)),
    };
    let Expr::Column(column) = column else {
        return Err(unsupported(expr));
    };

    use PredicateOperator as Op;
    let name = column.name.as_str();
    let predicate = match (op, values.as_slice()) {
        (Op::Eq, [value]) => builder.equal(name, value.clone()),
        (Op::NotEq, [value]) => builder.not_equal(name, value.clone()),
        (Op::Lt, [value]) => builder.less_than(name, value.clone()),
        (Op::LtEq, [value]) => builder.less_or_equal(name, value.clone()),
        (Op::Gt, [value]) => builder.greater_than(name, value.clone()),
        (Op::GtEq, [value]) => builder.greater_or_equal(name, value.clone()),
        (Op::IsNull, []) => builder.is_null(name),
        (Op::IsNotNull, []) => builder.is_not_null(name),
        (Op::In, _) => builder.is_in(name, values),
        (Op::NotIn, _) => builder.is_not_in(name, values),
        _ => return Err(unsupported(expr)),
    }
    .map_err(|err| {
        DataFusionError::NotImplemented(format!("Paimon cannot push down {expr}: {err}"))
    })?;
    Ok(if negated {
        Predicate::negate(predicate)
    } else {
        predicate
    })
}

fn paimon_literal(expr: &Expr) -> Result<Option<Datum>> {
    let Expr::Literal(value, _) = expr else {
        return Err(unsupported(expr));
    };
    Ok(Some(match value {
        ScalarValue::Boolean(Some(value)) => Datum::Bool(*value),
        ScalarValue::Int8(Some(value)) => Datum::TinyInt(*value),
        ScalarValue::Int16(Some(value)) => Datum::SmallInt(*value),
        ScalarValue::Int32(Some(value)) => Datum::Int(*value),
        ScalarValue::Int64(Some(value)) => Datum::Long(*value),
        ScalarValue::Utf8(Some(value))
        | ScalarValue::LargeUtf8(Some(value))
        | ScalarValue::Utf8View(Some(value)) => Datum::String(value.clone()),
        ScalarValue::Date32(Some(value)) => Datum::Date(*value),
        value if value.is_null() => return Ok(None),
        _ => {
            return not_impl_err!(
                "Paimon cannot conservatively push down literal {expr}"
            );
        }
    }))
}
