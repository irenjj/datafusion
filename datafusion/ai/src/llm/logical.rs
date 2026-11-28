// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: See the NOTICE file distributed with this work for additional information regarding copyright ownership
// SPDX-FileContributor: Licensed to the Apache Software Foundation (ASF) under one or more contributor license agreements

use std::cmp::Ordering;
use std::collections::{BTreeSet, HashMap, HashSet};
use std::fmt;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering as AtomicOrdering};

use super::{LlmOptions, LlmProviderRegistry, OnError, TaskKind};
use crate::MediaRefType;
use arrow::datatypes::{DataType, Field, FieldRef};
use datafusion_common::config::ConfigOptions;
use datafusion_common::tree_node::{Transformed, TreeNode, TreeNodeRecursion};
use datafusion_common::{
    Column, DFSchema, DFSchemaRef, Result, ScalarValue, internal_err, plan_err,
};
use datafusion_expr::expr::ScalarFunction;
use datafusion_expr::expr_rewriter::NamePreserver;
use datafusion_expr::planner::{ExprPlanner, PlannerResult, RawScalarExpr};
use datafusion_expr::simplify::SimplifyContext;
use datafusion_expr::{
    ColumnarValue, Expr, Extension, LogicalPlan, Projection, ReturnFieldArgs,
    ScalarFunctionArgs, ScalarUDFImpl, Signature, TypeSignature,
    UserDefinedLogicalNodeCore, Volatility,
};
use datafusion_optimizer::simplify_expressions::ExprSimplifier;
use datafusion_optimizer::{AnalyzerRule, ApplyOrder, OptimizerConfig, OptimizerRule};

#[derive(Debug, PartialEq, Eq, Hash)]
pub(super) struct LlmMarkerUdf {
    kind: TaskKind,
    signature: Signature,
}

impl LlmMarkerUdf {
    pub(super) fn new(kind: TaskKind) -> Self {
        let signature = if kind == TaskKind::Vision {
            Signature::user_defined(Volatility::Volatile)
        } else {
            let signatures = [DataType::Utf8, DataType::LargeUtf8, DataType::Utf8View]
                .into_iter()
                .map(|prompt| {
                    TypeSignature::Exact(vec![prompt, DataType::Utf8, DataType::Utf8])
                })
                .collect();
            Signature::one_of(signatures, Volatility::Volatile)
        };
        Self { kind, signature }
    }
}

impl ScalarUDFImpl for LlmMarkerUdf {
    fn name(&self) -> &str {
        self.kind.function()
    }

    fn signature(&self) -> &Signature {
        &self.signature
    }

    fn coerce_types(&self, arg_types: &[DataType]) -> Result<Vec<DataType>> {
        let [prompt, media, _, _] = arg_types else {
            return plan_err!("vlm() requires prompt, media, model, and options_json");
        };
        let prompt = if prompt.is_string() {
            prompt.clone()
        } else {
            DataType::Utf8
        };
        // Preserve the actual type and child field metadata for media validation.
        let media = if media.is_null() {
            DataType::Utf8
        } else {
            media.clone()
        };
        Ok(vec![prompt, media, DataType::Utf8, DataType::Utf8])
    }

    fn return_type(&self, _arg_types: &[DataType]) -> Result<DataType> {
        if self.kind == TaskKind::Embed {
            return plan_err!("embed() requires a dimension in constant options_json");
        }
        Ok(DataType::Utf8)
    }

    fn return_field_from_args(&self, args: ReturnFieldArgs<'_>) -> Result<FieldRef> {
        if self.kind == TaskKind::Vision
            && let Some(media) = args.arg_fields.get(1)
        {
            validate_media_field(media)?;
        }
        let [(model_index, model_name), (options_index, options_name)] =
            config_args(self.kind);
        if let Some(Some(value)) = args.scalar_arguments.get(model_index) {
            required_string(value, self.kind, model_name)?;
        }
        let (options, schema) = match args.scalar_arguments.get(options_index) {
            Some(Some(value)) => {
                let (options, _, schema) =
                    LlmOptions::parse(required_string(value, self.kind, options_name)?)?;
                (options, schema)
            }
            _ => (LlmOptions::default(), None),
        };
        options.output_field(self.kind, self.name(), schema.as_ref())
    }

    fn invoke_with_args(&self, _args: ScalarFunctionArgs) -> Result<ColumnarValue> {
        internal_err!(
            "{}() can only be used in a projection and must be planned as a LlmNode operator",
            self.name()
        )
    }
}

fn config_args(kind: TaskKind) -> [(usize, &'static str); 2] {
    let model = if kind == TaskKind::Vision { 2 } else { 1 };
    [(model, "model"), (model + 1, "options_json")]
}

fn required_string<'a>(
    value: &'a ScalarValue,
    kind: TaskKind,
    name: &str,
) -> Result<&'a str> {
    match value {
        ScalarValue::Utf8(Some(value)) if !value.trim().is_empty() => Ok(value),
        _ => constant_string_error(kind, name),
    }
}

fn constant_string_error<T>(kind: TaskKind, name: &str) -> Result<T> {
    plan_err!(
        "{}() {name} must be a non-empty, non-NULL constant Utf8 string",
        kind.function()
    )
}

pub(super) fn validate_media_field(field: &Field) -> Result<()> {
    let item = match field.data_type() {
        DataType::List(item)
        | DataType::LargeList(item)
        | DataType::FixedSizeList(item, _) => item,
        _ => field,
    };
    if item.data_type().is_string()
        || (matches!(item.data_type(), DataType::Struct(_))
            && item.try_extension_type::<MediaRefType>().is_ok())
    {
        return Ok(());
    }
    plan_err!(
        "vlm() media must be a string URI or a df.media_ref v1 field, or a List, LargeList, or FixedSizeList of either, got {}",
        field.data_type()
    )
}

#[derive(Debug)]
pub(super) struct MarkerExprPlanner;

impl ExprPlanner for MarkerExprPlanner {
    fn plan_scalar(
        &self,
        mut expr: RawScalarExpr,
        schema: &DFSchema,
    ) -> Result<PlannerResult<RawScalarExpr>> {
        if let Some(marker) = expr.func.inner().downcast_ref::<LlmMarkerUdf>()
            && let Some(options) = expr.args.get_mut(config_args(marker.kind)[1].0)
            && !matches!(options, Expr::Literal(..))
            && !options.any_column_refs()
            && !options.contains_outer()
            && !options.exists(|expr| {
                Ok(matches!(expr, Expr::ScalarFunction(function)
                if function.func.signature().volatility != Volatility::Immutable))
            })?
        {
            // The SQL planner needs the output type before the optimizer folds arguments.
            let simplifier = ExprSimplifier::new(
                SimplifyContext::builder()
                    .with_schema(Arc::new(schema.clone()))
                    .build(),
            );
            if let Ok(folded @ Expr::Literal(..)) = simplifier
                .coerce(options.clone(), schema)
                .and_then(|expr| simplifier.simplify(expr))
                && (marker.kind != TaskKind::Chat
                    || literal_string(&folded, marker.kind, "options_json")
                        .and_then(LlmOptions::parse)
                        .is_ok_and(|(options, _, schema)| {
                            options.on_error == OnError::Envelope || schema.is_some()
                        }))
            {
                *options = folded;
            }
        }
        Ok(PlannerResult::Original(expr))
    }
}

/// Logical extension that appends an LLM inference result to each input row.
#[derive(Debug, Eq, PartialEq, Hash)]
pub struct LlmNode {
    input: LogicalPlan,
    prompt_expr: Expr,
    media_expr: Option<Expr>,
    kind: TaskKind,
    model: String,
    options_json: String,
    output_col: String,
    schema: DFSchemaRef,
}

impl PartialOrd for LlmNode {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        (
            &self.input,
            &self.prompt_expr,
            &self.media_expr,
            self.kind,
            &self.model,
            &self.options_json,
            &self.output_col,
        )
            .partial_cmp(&(
                &other.input,
                &other.prompt_expr,
                &other.media_expr,
                other.kind,
                &other.model,
                &other.options_json,
                &other.output_col,
            ))
            .filter(|ordering| *ordering != Ordering::Equal || self == other)
    }
}

impl LlmNode {
    /// Returns the input plan.
    pub fn input(&self) -> &LogicalPlan {
        &self.input
    }

    /// Returns the expression that produces the prompt for each row.
    pub fn prompt_expr(&self) -> &Expr {
        &self.prompt_expr
    }

    /// Returns the model identifier.
    pub fn model(&self) -> &str {
        &self.model
    }

    /// Returns the validated, canonical JSON options.
    pub fn options_json(&self) -> &str {
        &self.options_json
    }

    /// Returns the name of the appended result column.
    pub fn output_col(&self) -> &str {
        &self.output_col
    }

    pub(super) fn task_kind(&self) -> TaskKind {
        self.kind
    }

    pub(super) fn media_expr(&self) -> Option<&Expr> {
        self.media_expr.as_ref()
    }

    fn try_new(
        input: LogicalPlan,
        prompt_expr: Expr,
        media_expr: Option<Expr>,
        kind: TaskKind,
        model: String,
        options_json: String,
        output_field: FieldRef,
    ) -> Result<Self> {
        let input_schema = input.schema();
        let output_col = output_field.name().clone();
        let output_schema =
            DFSchema::from_unqualified_fields(vec![output_field].into(), HashMap::new())?;
        let schema = input_schema
            .join(&output_schema)?
            .with_functional_dependencies(
                input_schema.functional_dependencies().clone(),
            )?;
        Ok(Self {
            input,
            prompt_expr,
            media_expr,
            kind,
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
        std::iter::once(self.prompt_expr.clone())
            .chain(self.media_expr.clone())
            .collect()
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
        let mut exprs = exprs.into_iter();
        let inputs: Result<[_; 1], _> = inputs.try_into();
        let (Some(prompt_expr), Ok([input])) = (exprs.next(), inputs) else {
            return internal_err!(
                "LlmProject requires one prompt expression and one input"
            );
        };
        let media_expr = exprs.next();
        if media_expr.is_some() != self.media_expr.is_some() || exprs.next().is_some() {
            return internal_err!(
                "LlmProject received an unexpected number of expressions"
            );
        }
        Self::try_new(
            input,
            prompt_expr,
            media_expr,
            self.kind,
            self.model.clone(),
            self.options_json.clone(),
            Arc::clone(self.schema.fields().last().unwrap()),
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
        for expr in std::iter::once(&self.prompt_expr).chain(self.media_expr.as_ref()) {
            for column in expr.column_refs() {
                required.insert(schema.index_of_column(column).ok()?);
            }
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

fn marker(expr: &Expr) -> Option<(&ScalarFunction, TaskKind)> {
    match expr {
        Expr::ScalarFunction(call) => call
            .func
            .inner()
            .downcast_ref::<LlmMarkerUdf>()
            .map(|marker| (call, marker.kind)),
        _ => None,
    }
}

fn validate_usage(plan: &LogicalPlan) -> Result<bool> {
    let projection = matches!(plan, LogicalPlan::Projection(_));
    let mut found = false;
    plan.apply_expressions(|root| {
        root.apply(|expr| {
            if expr.short_circuits() {
                let mut kind = None;
                if expr.exists(|child| {
                    kind = marker(child).map(|(_, kind)| kind);
                    Ok(kind.is_some())
                })? {
                    return plan_err!("{}() is not supported inside CASE, AND/OR, or COALESCE short-circuit expressions", kind.unwrap().function());
                }
                return Ok(TreeNodeRecursion::Jump);
            }
            if let Some((call, kind)) = marker(expr) {
                if !projection {
                    return plan_err!("{}() can only be used in a projection", kind.function());
                }
                for (index, name) in config_args(kind) {
                    if call.args.get(index).is_some_and(|arg| {
                        arg.any_column_refs() || arg.contains_outer() || arg.is_volatile()
                    }) {
                        return constant_string_error(kind, name);
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
                let Some((_, kind)) = marker(&expr) else {
                    return Ok(Transformed::no(expr));
                };
                let Expr::ScalarFunction(ScalarFunction { args, .. }) = expr else {
                    unreachable!("marker is a scalar function")
                };
                let (prompt, media, model, options) = match kind {
                    TaskKind::Vision => {
                        let Ok([prompt, media, model, options]): Result<[_; 4], _> =
                            args.try_into()
                        else {
                            return plan_err!(
                                "vlm() requires prompt, media, model, and options_json"
                            );
                        };
                        (prompt, Some(media), model, options)
                    }
                    TaskKind::Chat | TaskKind::Embed => {
                        let Ok([prompt, model, options]): Result<[_; 3], _> =
                            args.try_into()
                        else {
                            return plan_err!(
                                "{}() requires prompt, model, and options_json",
                                kind.function()
                            );
                        };
                        (prompt, None, model, options)
                    }
                };
                let model = literal_string(&model, kind, "model")?.to_owned();
                let (typed_options, options, schema) =
                    LlmOptions::parse(literal_string(&options, kind, "options_json")?)?;
                self.registry.resolve(typed_options.provider.as_deref())?;
                let reusable = typed_options.deduplicate.unwrap_or(true)
                    && !prompt.is_volatile()
                    && !media.as_ref().is_some_and(Expr::is_volatile);
                if reusable
                    && let Some((_, _, _, _, _, output)) = extracted.iter().find(
                        |(
                            prior_kind,
                            prior_prompt,
                            prior_media,
                            prior_model,
                            prior_options,
                            _,
                        )| {
                            prior_kind == &kind
                                && prior_prompt == &prompt
                                && prior_media == &media
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
                        kind,
                        prompt.clone(),
                        media.clone(),
                        model.clone(),
                        options.clone(),
                        output.clone(),
                    ));
                }
                input = Arc::new(LogicalPlan::Extension(Extension {
                    node: Arc::new(LlmNode::try_new(
                        input.as_ref().clone(),
                        prompt,
                        media,
                        kind,
                        model,
                        options,
                        typed_options.output_field(kind, &output, schema.as_ref())?,
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

fn literal_string<'a>(expr: &'a Expr, kind: TaskKind, name: &str) -> Result<&'a str> {
    match expr {
        Expr::Literal(value, _) => required_string(value, kind, name),
        _ => constant_string_error(kind, name),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use datafusion_common::DataFusionError;

    fn result_field(kind: TaskKind, json: &str) -> Result<FieldRef> {
        let udf = LlmMarkerUdf::new(kind);
        let model = ScalarValue::from("model");
        let options = ScalarValue::from(json);
        let mut names = vec!["text", "model", "options"];
        let mut scalars = vec![None, Some(&model), Some(&options)];
        if kind == TaskKind::Vision {
            names.insert(1, "media");
            scalars.insert(1, None);
        }
        let arg_fields: Vec<_> = names
            .into_iter()
            .map(|name| Arc::new(Field::new(name, DataType::Utf8, true)))
            .collect();
        udf.return_field_from_args(ReturnFieldArgs {
            arg_fields: &arg_fields,
            scalar_arguments: &scalars,
        })
    }

    #[test]
    fn response_schema_derives_nested_arrow_types_in_declaration_order() {
        let schema = r#"{"response_schema":{
            "type":"object",
            "properties":{
                "total":{"type":"integer"},
                "state":{"enum":["on","off"],"type":"string"},
                "flag":{"type":["null","boolean"]},
                "events":{"type":"array","items":{
                    "type":"object",
                    "properties":{"when":{"type":"number"},"label":{"type":"string"}},
                    "required":["label"]
                }}
            },
            "required":["events","flag","total"],
            "additionalProperties":false
        }}"#;
        let events = DataType::Struct(
            vec![
                Field::new("when", DataType::Float64, true),
                Field::new("label", DataType::Utf8, false),
            ]
            .into(),
        );
        let expected = DataType::Struct(
            vec![
                Field::new("total", DataType::Int64, false),
                Field::new("state", DataType::Utf8, true),
                Field::new("flag", DataType::Boolean, true),
                Field::new(
                    "events",
                    DataType::List(Arc::new(Field::new_list_field(events, false))),
                    false,
                ),
            ]
            .into(),
        );
        for kind in [TaskKind::Chat, TaskKind::Vision] {
            let field = result_field(kind, schema).unwrap();
            assert_eq!(field.data_type(), &expected);
            assert!(field.is_nullable());
        }
    }

    #[test]
    fn response_schema_rejects_unsupported_or_malformed_schemas() {
        for (property, keyword) in [
            (r#"{"type":"string","anyOf":[]}"#, "anyOf"),
            (r#"{"type":"string","oneOf":[]}"#, "oneOf"),
            (r#"{"type":"string","allOf":[]}"#, "allOf"),
            (r##"{"$ref":"#/$defs/type"}"##, "$ref"),
            (
                r#"{"type":"object","patternProperties":{}}"#,
                "patternProperties",
            ),
            (r#"{"type":"null"}"#, "null"),
            (r#"{"type":["string","integer"]}"#, "type"),
            (
                r#"{"type":"array","items":[{"type":"string"}]}"#,
                "response_schema",
            ),
            (
                r#"{"type":"object","properties":{},"additionalProperties":true}"#,
                "additionalProperties",
            ),
            (r#"{"type":"boolean","enum":["x"]}"#, "enum"),
            (r#"{"type":"string","required":["other"]}"#, "required"),
            (
                r#"{"type":"object","properties":{"x":{"type":"string"}},"required":["unknown"]}"#,
                "required",
            ),
        ] {
            let json = format!(
                r#"{{"response_schema":{{"type":"object","properties":{{"field":{property}}}}}}}"#,
            );
            let error = result_field(TaskKind::Chat, &json).unwrap_err();
            assert!(
                matches!(error, DataFusionError::Plan(ref message) if message.contains(keyword)),
                "{error} for {property}"
            );
        }
        for schema in [
            "null",
            r#"{"type":"object","properties":{}}"#,
            r#"{"type":["object","null"],"properties":{"a":{"type":"integer"}}}"#,
            r#"{"type":"string"}"#,
        ] {
            let error = result_field(
                TaskKind::Chat,
                &format!(r#"{{"response_schema":{schema}}}"#),
            )
            .unwrap_err();
            assert!(
                matches!(error, DataFusionError::Plan(ref message) if message.contains("response_schema")),
                "{error} for {schema}"
            );
        }
    }

    #[test]
    fn embed_rejects_response_schema_at_planning() {
        let error = result_field(
            TaskKind::Embed,
            r#"{"dimension":2,"response_schema":{"type":"object","properties":{"x":{"type":"string"}}}}"#,
        )
        .unwrap_err();
        assert!(
            matches!(error, DataFusionError::Plan(message) if message.contains("embed()") && message.contains("response_schema"))
        );
    }

    #[test]
    fn response_schema_bounds_depth_and_total_properties() {
        let object = |children: &str| {
            format!(r#"{{"type":"object","properties":{{{children}}}}}"#)
        };
        let depth = (0..16).fold(r#"{"type":"string"}"#.to_owned(), |schema, _| {
            object(&format!(r#""nested":{schema}"#))
        });
        let wide = (0..513)
            .map(|i| format!(r#""f{i}":{{"type":"boolean"}}"#))
            .collect::<Vec<_>>()
            .join(",");
        for (schema, limit) in [
            (depth, "16"),
            (object(&wide), ""), // A single object at this width remains valid.
            (
                object(&format!(
                    r#""left":{},"right":{}"#,
                    object(&wide),
                    object(&wide)
                )),
                "1024",
            ),
        ] {
            let result = result_field(
                TaskKind::Chat,
                &format!(r#"{{"response_schema":{schema}}}"#),
            );
            if limit.is_empty() {
                result.unwrap();
            } else {
                assert!(
                    matches!(result, Err(DataFusionError::Plan(message)) if message.contains(limit)),
                    "expected a planning limit of {limit}"
                );
            }
        }
    }

    #[test]
    fn embedding_result_requires_an_explicit_bounded_dimension() {
        let udf = LlmMarkerUdf::new(TaskKind::Embed);
        let model = ScalarValue::from("model");
        let arg_fields = ["text", "model", "options"]
            .map(|name| Arc::new(Field::new(name, DataType::Utf8, true)));
        for json in ["{}", r#"{"dimension":0}"#, r#"{"dimension":1048577}"#] {
            let options = ScalarValue::from(json);
            let error = udf
                .return_field_from_args(ReturnFieldArgs {
                    arg_fields: &arg_fields,
                    scalar_arguments: &[None, Some(&model), Some(&options)],
                })
                .unwrap_err();
            assert!(
                matches!(error, DataFusionError::Plan(message) if message.contains("dimension"))
            );
        }
        let options = ScalarValue::from(r#"{"dimension":1024}"#);
        let field = udf
            .return_field_from_args(ReturnFieldArgs {
                arg_fields: &arg_fields,
                scalar_arguments: &[None, Some(&model), Some(&options)],
            })
            .unwrap();
        assert_eq!(
            field.data_type(),
            &DataType::FixedSizeList(
                Arc::new(Field::new_list_field(DataType::Float32, false)),
                1024,
            )
        );
    }
}
