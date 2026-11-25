// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: See the NOTICE file distributed with this work for additional information regarding copyright ownership
// SPDX-FileContributor: Licensed to the Apache Software Foundation (ASF) under one or more contributor license agreements

use std::cmp::Ordering;
use std::collections::{BTreeSet, HashMap, HashSet};
use std::fmt;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering as AtomicOrdering};

use super::{LlmOptions, LlmProviderRegistry};
use crate::arrow::datatypes::{DataType, Field, FieldRef};
use crate::common::config::ConfigOptions;
use crate::common::tree_node::{Transformed, TreeNode, TreeNodeRecursion};
use crate::common::{
    Column, DFSchema, DFSchemaRef, Result, ScalarValue, internal_err, plan_err,
};
use crate::logical_expr::expr::ScalarFunction;
use crate::logical_expr::expr_rewriter::NamePreserver;
use crate::logical_expr::{
    ColumnarValue, Expr, Extension, LogicalPlan, Projection, ReturnFieldArgs,
    ScalarFunctionArgs, ScalarUDFImpl, Signature, TypeSignature,
    UserDefinedLogicalNodeCore, Volatility,
};
use crate::optimizer::{AnalyzerRule, ApplyOrder, OptimizerConfig, OptimizerRule};

#[derive(Debug, PartialEq, Eq, Hash)]
pub(super) struct LlmMarkerUdf {
    signature: Signature,
}

impl Default for LlmMarkerUdf {
    fn default() -> Self {
        Self {
            signature: Signature::one_of(
                [DataType::Utf8, DataType::LargeUtf8, DataType::Utf8View]
                    .into_iter()
                    .map(|prompt| {
                        TypeSignature::Exact(vec![prompt, DataType::Utf8, DataType::Utf8])
                    })
                    .collect(),
                Volatility::Volatile,
            ),
        }
    }
}

impl ScalarUDFImpl for LlmMarkerUdf {
    fn name(&self) -> &str {
        "llm"
    }

    fn signature(&self) -> &Signature {
        &self.signature
    }

    fn return_type(&self, _arg_types: &[DataType]) -> Result<DataType> {
        Ok(DataType::Utf8)
    }

    fn return_field_from_args(&self, args: ReturnFieldArgs<'_>) -> Result<FieldRef> {
        for (index, name) in CONFIG_ARGS {
            if let Some(Some(value)) = args.scalar_arguments.get(index) {
                required_string(value, name)?;
            }
            // None means any non-literal, including concat of string literals. The analyzer
            // detects column dependencies and the extraction rule checks the folded result.
        }
        Ok(llm_field(self.name()))
    }

    fn invoke_with_args(&self, _args: ScalarFunctionArgs) -> Result<ColumnarValue> {
        internal_err!(
            "llm() can only be used in a projection and must be planned as a LlmNode operator"
        )
    }
}

const CONFIG_ARGS: [(usize, &str); 2] = [(1, "model"), (2, "options_json")];

fn required_string<'a>(value: &'a ScalarValue, name: &str) -> Result<&'a str> {
    match value {
        ScalarValue::Utf8(Some(value)) if !value.trim().is_empty() => Ok(value),
        _ => {
            plan_err!("llm() {name} must be a non-empty, non-NULL constant Utf8 string")
        }
    }
}

pub(super) fn llm_field(name: &str) -> FieldRef {
    Arc::new(Field::new(name, DataType::Utf8, true))
}

#[derive(Debug, Eq, PartialEq, Hash)]
pub(super) struct LlmNode {
    pub(super) input: LogicalPlan,
    pub(super) prompt_expr: Expr,
    pub(super) model: String,
    pub(super) options_json: String,
    pub(super) output_col: String,
    schema: DFSchemaRef,
}

impl PartialOrd for LlmNode {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        (
            &self.input,
            &self.prompt_expr,
            &self.model,
            &self.options_json,
            &self.output_col,
        )
            .partial_cmp(&(
                &other.input,
                &other.prompt_expr,
                &other.model,
                &other.options_json,
                &other.output_col,
            ))
            .filter(|ordering| *ordering != Ordering::Equal || self == other)
    }
}

impl LlmNode {
    fn try_new(
        input: LogicalPlan,
        prompt_expr: Expr,
        model: String,
        options_json: String,
        output_col: String,
    ) -> Result<Self> {
        let input_schema = input.schema();
        let output_schema = DFSchema::from_unqualified_fields(
            vec![llm_field(&output_col)].into(),
            HashMap::new(),
        )?;
        let schema = input_schema
            .join(&output_schema)?
            .with_functional_dependencies(
                input_schema.functional_dependencies().clone(),
            )?;
        Ok(Self {
            input,
            prompt_expr,
            model,
            options_json,
            output_col,
            schema: Arc::new(schema),
        })
    }
}

impl UserDefinedLogicalNodeCore for LlmNode {
    fn name(&self) -> &str {
        "LlmProject"
    }

    fn inputs(&self) -> Vec<&LogicalPlan> {
        vec![&self.input]
    }

    fn schema(&self) -> &DFSchemaRef {
        &self.schema
    }

    fn expressions(&self) -> Vec<Expr> {
        vec![self.prompt_expr.clone()]
    }

    fn fmt_for_explain(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "LlmProject: model={}, output={}",
            self.model, self.output_col
        )
    }

    fn with_exprs_and_inputs(
        &self,
        exprs: Vec<Expr>,
        inputs: Vec<LogicalPlan>,
    ) -> Result<Self> {
        let (Ok([prompt_expr]), Ok([input])): (Result<[_; 1], _>, Result<[_; 1], _>) =
            (exprs.try_into(), inputs.try_into())
        else {
            return internal_err!(
                "LlmProject requires one prompt expression and one input"
            );
        };
        Self::try_new(
            input,
            prompt_expr,
            self.model.clone(),
            self.options_json.clone(),
            self.output_col.clone(),
        )
    }

    fn prevent_predicate_push_down_columns(&self) -> HashSet<String> {
        HashSet::from([self.output_col.clone()])
    }

    fn necessary_children_exprs(
        &self,
        output_columns: &[usize],
    ) -> Option<Vec<Vec<usize>>> {
        let schema = self.input.schema();
        let mut required = output_columns
            .iter()
            .copied()
            .filter(|index| *index < schema.fields().len())
            .collect::<BTreeSet<_>>();
        for column in self.prompt_expr.column_refs() {
            required.insert(schema.index_of_column(column).ok()?);
        }
        Some(vec![required.into_iter().collect()])
    }

    fn bypass_if_unused(&self, required_columns: &[usize]) -> Option<&LogicalPlan> {
        // An unused output skips inference and evaluation of its prompt,
        // including any volatile subexpressions.
        let output_index = self.input.schema().fields().len();
        (!required_columns.contains(&output_index)).then_some(&self.input)
    }

    fn supports_limit_pushdown(&self) -> bool {
        true
    }
}

fn marker(expr: &Expr) -> Option<&ScalarFunction> {
    match expr {
        Expr::ScalarFunction(call) if call.func.inner().is::<LlmMarkerUdf>() => {
            Some(call)
        }
        _ => None,
    }
}

fn validate_usage(plan: &LogicalPlan) -> Result<bool> {
    let projection = matches!(plan, LogicalPlan::Projection(_));
    let mut found = false;
    plan.apply_expressions(|root| {
        root.apply(|expr| {
            if expr.short_circuits() {
                if expr.exists(|child| Ok(marker(child).is_some()))? {
                    return plan_err!("llm() is not supported inside CASE, AND/OR, or COALESCE short-circuit expressions");
                }
                return Ok(TreeNodeRecursion::Jump);
            }
            if let Some(call) = marker(expr) {
                if !projection {
                    return plan_err!("llm() can only be used in a projection");
                }
                for (index, name) in CONFIG_ARGS {
                    if call.args.get(index).is_some_and(|arg| {
                        arg.any_column_refs() || arg.contains_outer() || arg.is_volatile()
                    }) {
                        return plan_err!("llm() {name} must be a non-empty, non-NULL constant Utf8 string");
                    }
                }
                found = true;
            }
            Ok(TreeNodeRecursion::Continue)
        })
    })?;
    Ok(found)
}

// Validate before simplification can erase a dead CASE/AND/OR branch or fold away
// other evidence of an unsupported context. Extraction still runs after folding.
#[derive(Debug)]
pub(super) struct ValidateLlm;

impl AnalyzerRule for ValidateLlm {
    fn name(&self) -> &str {
        "validate_llm_usage"
    }

    fn analyze(&self, plan: LogicalPlan, _config: &ConfigOptions) -> Result<LogicalPlan> {
        plan.apply_with_subqueries(|plan| {
            validate_usage(plan)?;
            Ok(TreeNodeRecursion::Continue)
        })?;
        Ok(plan)
    }
}

#[derive(Debug)]
pub(super) struct LlmProjectRule {
    next_column: AtomicU64,
    registry: LlmProviderRegistry,
}

impl LlmProjectRule {
    pub(super) fn new(registry: LlmProviderRegistry) -> Self {
        Self {
            next_column: AtomicU64::new(0),
            registry,
        }
    }
}

impl OptimizerRule for LlmProjectRule {
    fn name(&self) -> &str {
        "extract_llm_project"
    }

    fn apply_order(&self) -> Option<ApplyOrder> {
        Some(ApplyOrder::BottomUp)
    }

    fn rewrite(
        &self,
        plan: LogicalPlan,
        _config: &dyn OptimizerConfig,
    ) -> Result<Transformed<LogicalPlan>> {
        if !validate_usage(&plan)? {
            return Ok(Transformed::no(plan));
        }
        let names = NamePreserver::new_for_projection();
        let LogicalPlan::Projection(projection) = plan else {
            return internal_err!("Validated llm() call outside a projection");
        };

        let mut input = projection.input;
        // Match by expression equality without storing Expr as a map key. The
        // cache is scoped to the inference calls in this projection.
        let mut extracted = Vec::new();
        let mut exprs = Vec::with_capacity(projection.expr.len());
        for expr in projection.expr {
            let name = names.save(&expr);
            let rewritten = expr.transform_up(|expr| {
                if marker(&expr).is_none() {
                    return Ok(Transformed::no(expr));
                }
                let Expr::ScalarFunction(ScalarFunction { args, .. }) = expr else {
                    unreachable!("marker is a scalar function")
                };
                let Ok([prompt, model, options]): Result<[_; 3], _> = args.try_into()
                else {
                    return plan_err!("llm() requires prompt, model, and options_json");
                };
                let model = literal_string(&model, "model")?.to_owned();
                let (typed_options, options) =
                    LlmOptions::parse(literal_string(&options, "options_json")?)?;
                self.registry.resolve(typed_options.provider.as_deref())?;
                let reusable =
                    typed_options.deduplicate.unwrap_or(true) && !prompt.is_volatile();
                if reusable
                    && let Some((_, _, _, output)) = extracted.iter().find(
                        |(prior_prompt, prior_model, prior_options, _)| {
                            prior_prompt == &prompt
                                && prior_model == &model
                                && prior_options == &options
                        },
                    )
                {
                    return Ok(Transformed::yes(Expr::Column(Column::from_name(output))));
                }

                let output = loop {
                    let id = self.next_column.fetch_add(1, AtomicOrdering::Relaxed);
                    let candidate = format!("__df_llm_{id}");
                    if input
                        .schema()
                        .fields()
                        .iter()
                        .chain(projection.schema.fields())
                        .all(|field| field.name() != &candidate)
                    {
                        break candidate;
                    }
                };
                if reusable {
                    extracted.push((
                        prompt.clone(),
                        model.clone(),
                        options.clone(),
                        output.clone(),
                    ));
                }
                input = Arc::new(LogicalPlan::Extension(Extension {
                    node: Arc::new(LlmNode::try_new(
                        input.as_ref().clone(),
                        prompt,
                        model,
                        options,
                        output.clone(),
                    )?),
                }));
                Ok(Transformed::yes(Expr::Column(Column::from_name(output))))
            })?;
            exprs.push(name.restore(rewritten.data));
        }

        Ok(Transformed::yes(LogicalPlan::Projection(
            Projection::try_new_with_schema(exprs, input, projection.schema)?,
        )))
    }
}

fn literal_string<'a>(expr: &'a Expr, name: &str) -> Result<&'a str> {
    match expr {
        Expr::Literal(value, _) => required_string(value, name),
        _ => {
            plan_err!("llm() {name} must be a non-empty, non-NULL constant Utf8 string")
        }
    }
}
