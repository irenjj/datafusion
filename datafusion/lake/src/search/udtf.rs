// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: See the NOTICE file distributed with this work for additional information regarding copyright ownership
// SPDX-FileContributor: Licensed to the Apache Software Foundation (ASF) under one or more contributor license agreements

use std::sync::Arc;

use arrow::datatypes::{DataType, Field, Schema, SchemaRef};
use async_trait::async_trait;
use datafusion_catalog::{Session, TableFunctionArgs, TableFunctionImpl, TableProvider};
use datafusion_common::{
    Result, ScalarValue, internal_err, plan_datafusion_err, plan_err,
};
use datafusion_expr::{Expr, TableProviderFilterPushDown, TableType};
use datafusion_physical_plan::{ExecutionPlan, empty::EmptyExec};
use lance_core::{ROW_ADDR, ROW_ID};
use paimon::spec::{Predicate, ROW_ID_FIELD_NAME};

use crate::scan::projected_schema;

use super::exec::FusedScoreExec;
use super::{
    DEFAULT_RRF_K, Engine, LakeSearchCatalog, PinnedLakeTable, SCORE, SearchRequest,
    TextQuery, lance, paimon as paimon_search, rrf_fuse, validate,
};

/// Create the vector retrieval table function for a [`LakeSearchCatalog`].
///
/// Calling the function returns planning errors for invalid arguments or unverified
/// indexes. Arguments passed to [`TableFunctionImpl::call_with_args`] must be literals.
///
/// # Public Only for Internal Use:
///
/// This is not a public API and is for internal use only; see [API policy] for details.
/// [API policy]: https://datafusion.apache.org/contributor-guide/api-health.html#internal-public-apis
#[doc(hidden)]
#[must_use]
pub fn vector_search_function(
    catalog: Arc<LakeSearchCatalog>,
) -> Arc<dyn TableFunctionImpl> {
    Arc::new(VectorSearch(catalog))
}

/// Create the weighted vector and text retrieval table function for a [`LakeSearchCatalog`].
///
/// Calling the function returns planning errors for invalid arguments or unverified
/// indexes and rejects unsupported predicates in its optional SQL prefilter.
///
/// # Public Only for Internal Use:
///
/// This is not a public API and is for internal use only; see [API policy] for details.
/// [API policy]: https://datafusion.apache.org/contributor-guide/api-health.html#internal-public-apis
#[doc(hidden)]
#[must_use]
pub fn hybrid_search_function(
    catalog: Arc<LakeSearchCatalog>,
) -> Arc<dyn TableFunctionImpl> {
    Arc::new(HybridSearch(catalog))
}

#[derive(Debug)]
struct VectorSearch(Arc<LakeSearchCatalog>);

impl TableFunctionImpl for VectorSearch {
    fn call_with_args(&self, args: TableFunctionArgs) -> Result<Arc<dyn TableProvider>> {
        let [table, column, vector, k] = args.exprs() else {
            return plan_err!("vector_search expects 4 arguments");
        };
        let table = self.0.resolve(string(table, "table name")?)?;
        let request = SearchRequest {
            vector_column: string(column, "vector column")?.into(),
            vector: validate::vector(string(vector, "query vector")?)?,
            text: None,
            k: top_k(k)?,
            rrf_k: DEFAULT_RRF_K,
            vector_weight: 1.0,
            filter: None,
        };
        SearchProvider::try_new(table, request)
    }
}

#[derive(Debug)]
struct HybridSearch(Arc<LakeSearchCatalog>);

impl TableFunctionImpl for HybridSearch {
    fn call_with_args(&self, args: TableFunctionArgs) -> Result<Arc<dyn TableProvider>> {
        let [
            table,
            vector_column,
            vector,
            text_column,
            text,
            k,
            optional @ ..,
        ] = args.exprs()
        else {
            return plan_err!("hybrid_search expects 6 to 10 arguments");
        };
        if optional.len() > 4 {
            return plan_err!("hybrid_search expects 6 to 10 arguments");
        }
        let table = self.0.resolve(string(table, "table name")?)?;
        let rrf_k = optional
            .first()
            .map(|expr| integer(expr, "rrf_k"))
            .transpose()?
            .unwrap_or(DEFAULT_RRF_K);
        let vector_weight = optional
            .get(1)
            .map(|expr| weight(expr, "vector_weight"))
            .transpose()?
            .unwrap_or(1.0);
        let text_weight = optional
            .get(2)
            .map(|expr| weight(expr, "text_weight"))
            .transpose()?
            .unwrap_or(1.0);
        let filter = optional
            .get(3)
            .map(|expr| {
                validate::filter_sql(
                    string(expr, "filter_sql")?,
                    &table.schema(),
                    args.session(),
                )
            })
            .transpose()?;
        let request = SearchRequest {
            vector_column: string(vector_column, "vector column")?.into(),
            vector: validate::vector(string(vector, "query vector")?)?,
            text: Some(TextQuery {
                column: string(text_column, "text column")?.into(),
                query: string(text, "query text")?.into(),
                weight: text_weight,
            }),
            k: top_k(k)?,
            rrf_k,
            vector_weight,
            filter,
        };
        SearchProvider::try_new(table, request)
    }
}

fn literal<'a>(expr: &'a Expr, name: &str) -> Result<&'a ScalarValue> {
    match expr {
        Expr::Literal(value, _) => Ok(value),
        _ => plan_err!("Lake search {name} must be a literal"),
    }
}

fn string<'a>(expr: &'a Expr, name: &str) -> Result<&'a str> {
    match literal(expr, name)? {
        ScalarValue::Utf8(Some(value))
        | ScalarValue::LargeUtf8(Some(value))
        | ScalarValue::Utf8View(Some(value)) => Ok(value),
        _ => plan_err!("Lake search {name} must be a non-null string literal"),
    }
}

fn integer(expr: &Expr, name: &str) -> Result<u64> {
    let value = match literal(expr, name)? {
        ScalarValue::Int8(Some(value)) => u64::try_from(*value).ok(),
        ScalarValue::Int16(Some(value)) => u64::try_from(*value).ok(),
        ScalarValue::Int32(Some(value)) => u64::try_from(*value).ok(),
        ScalarValue::Int64(Some(value)) => u64::try_from(*value).ok(),
        ScalarValue::UInt8(Some(value)) => Some(u64::from(*value)),
        ScalarValue::UInt16(Some(value)) => Some(u64::from(*value)),
        ScalarValue::UInt32(Some(value)) => Some(u64::from(*value)),
        ScalarValue::UInt64(Some(value)) => Some(*value),
        _ => None,
    };
    value.ok_or_else(|| {
        plan_datafusion_err!("Lake search {name} must be a nonnegative integer literal")
    })
}

fn top_k(expr: &Expr) -> Result<usize> {
    usize::try_from(integer(expr, "k")?)
        .map_err(|_| plan_datafusion_err!("Lake search k is too large"))
}

fn weight(expr: &Expr, name: &str) -> Result<f64> {
    let value = literal(expr, name)?;
    if value.data_type().is_numeric()
        && let Ok(ScalarValue::Float64(Some(weight))) = value.cast_to(&DataType::Float64)
    {
        return validate::weight(weight, name);
    }
    plan_err!("Lake search {name} must be a non-null numeric literal")
}

#[derive(Debug)]
struct SearchProvider {
    table: Arc<PinnedLakeTable>,
    schema: SchemaRef,
    request: SearchRequest,
    paimon_filter: Option<Predicate>,
}

impl SearchProvider {
    fn try_new(
        table: Arc<PinnedLakeTable>,
        request: SearchRequest,
    ) -> Result<Arc<dyn TableProvider>> {
        validate::limits(request.k, request.rrf_k)?;
        let paimon_filter = match &table.0 {
            Engine::Lance(_) => None,
            Engine::Paimon(table) => {
                paimon_search::validate_table(table)?;
                paimon_search::predicate(table, &request)?
            }
        };
        validate::indices(&table, &request)?;
        let source = table.schema();
        let reserved: &[&str] = match &table.0 {
            Engine::Lance(_) => &[SCORE, ROW_ID, ROW_ADDR],
            Engine::Paimon(_) => &[SCORE, ROW_ID_FIELD_NAME],
        };
        for &name in reserved {
            if source.field_with_name(name).is_ok() {
                return plan_err!("Lake search table uses reserved column name {name:?}");
            }
        }
        let mut fields = source.fields().to_vec();
        fields.push(Arc::new(Field::new(SCORE, DataType::Float32, false)));
        let schema =
            Arc::new(Schema::new_with_metadata(fields, source.metadata().clone()));
        Ok(Arc::new(Self {
            table,
            schema,
            request,
            paimon_filter,
        }))
    }
}

#[async_trait]
impl TableProvider for SearchProvider {
    fn schema(&self) -> SchemaRef {
        Arc::clone(&self.schema)
    }

    fn table_type(&self) -> TableType {
        TableType::Base
    }

    fn supports_filters_pushdown(
        &self,
        filters: &[&Expr],
    ) -> Result<Vec<TableProviderFilterPushDown>> {
        Ok(vec![
            TableProviderFilterPushDown::Unsupported;
            filters.len()
        ])
    }

    async fn scan(
        &self,
        _state: &dyn Session,
        projection: Option<&[usize]>,
        filters: &[Expr],
        limit: Option<usize>,
    ) -> Result<Arc<dyn ExecutionPlan>> {
        if !filters.is_empty() {
            return internal_err!(
                "Lake search does not accept outer filters in its scan"
            );
        }
        let schema = projected_schema(&self.schema, projection)?;
        if limit == Some(0) {
            return Ok(Arc::new(EmptyExec::new(schema)));
        }
        let (vector, text) = match &self.table.0 {
            Engine::Lance(table) => lance::search(table, &self.request).await?,
            Engine::Paimon(table) => {
                paimon_search::search(table, &self.request, self.paimon_filter.as_ref())
                    .await?
            }
        };
        let text_weight = self.request.text.as_ref().map_or(1.0, |query| query.weight);
        let rows = rrf_fuse(
            self.request.rrf_k,
            self.request.k,
            &[(self.request.vector_weight, &vector), (text_weight, &text)],
        )?;
        if rows.is_empty() {
            return Ok(Arc::new(EmptyExec::new(schema)));
        }
        let fields = schema.fields().iter().filter(|field| field.name() != SCORE);
        let mut input_fields = Vec::new();
        for field in fields {
            if !input_fields.contains(field) {
                input_fields.push(Arc::clone(field));
            }
        }
        let projection = schema
            .fields()
            .iter()
            .map(|field| input_fields.iter().position(|input| input == field))
            .collect();
        let (key_name, key_type, nullable) = match &self.table.0 {
            Engine::Lance(_) => (ROW_ID, DataType::UInt64, false),
            Engine::Paimon(_) => (ROW_ID_FIELD_NAME, DataType::Int64, true),
        };
        input_fields.push(Arc::new(Field::new(key_name, key_type, nullable)));
        let input_schema = Arc::new(Schema::new_with_metadata(
            input_fields,
            schema.metadata().clone(),
        ));
        let input = match &self.table.0 {
            Engine::Lance(table) => {
                lance::scan_with_row_addrs(table, input_schema, &rows)?
            }
            Engine::Paimon(table) => {
                paimon_search::scan_with_row_ranges(table, input_schema, &rows).await?
            }
        };
        Ok(Arc::new(FusedScoreExec::try_new(
            input, schema, projection, key_name, rows,
        )?))
    }
}
