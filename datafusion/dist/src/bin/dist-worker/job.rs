// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: See the NOTICE file distributed with this work for additional information regarding copyright ownership
// SPDX-FileContributor: Licensed to the Apache Software Foundation (ASF) under one or more contributor license agreements

use std::collections::BTreeSet;
use std::path::absolute;
use std::sync::Arc;

use datafusion::common::{Result, plan_datafusion_err, plan_err};
#[cfg(feature = "ai")]
use datafusion::execution::SessionStateBuilder;
use datafusion::execution::context::SQLOptions;
use datafusion::prelude::{ParquetReadOptions, SessionConfig, SessionContext};
#[cfg(feature = "ai")]
use datafusion_ai::llm::{LlmBackend, LlmProviderRegistry, MockLlmBackend};
#[cfg(feature = "ai-http")]
use datafusion_ai::llm::{OpenAiHttpBackend, OpenAiHttpConfig};
use datafusion_dist::{
    DistCodecs, LocalWorker, ParquetSplitSink, SinkOptions, Split, StagePlan,
    split_plan_for_distribution_with_codecs,
};
#[cfg(feature = "ai")]
use datafusion_dist::{DistExtensionRegistry, LlmExecCodec};
#[cfg(feature = "s3")]
use object_store::aws::AmazonS3Builder;
use object_store::{ObjectStore, local::LocalFileSystem, path::Path as StorePath};
use serde::{Deserialize, Serialize};
use url::{ParseError, Position, Url};

const SCHEMA: &str = "df-stage-job-v1";

pub(super) struct PlanningOptions {
    pub(super) sql: String,
    pub(super) tables: Vec<String>,
    pub(super) sink: SinkOptions,
    pub(super) output_root: Option<String>,
}

#[derive(Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub(super) struct StageJob {
    schema: String,
    stage: StagePlan,
    input_stores: BTreeSet<String>,
    options: SinkOptions,
    output_root: Option<String>,
    pub(super) splits: Vec<Split>,
    pub(super) pending_splits: Vec<PendingSplit>,
}

#[derive(Clone, Copy, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub(super) struct PendingSplit {
    pub(super) split_id: u64,
    pub(super) attempt_id: u32,
}

impl From<&Split> for PendingSplit {
    fn from(split: &Split) -> Self {
        Self {
            split_id: split.split_id,
            attempt_id: split.attempt_id,
        }
    }
}

impl StageJob {
    pub(super) async fn plan(options: PlanningOptions) -> Result<Self> {
        let output_root = options
            .output_root
            .as_deref()
            .map(Location::parse)
            .transpose()?;
        let ctx = session()?;
        let mut input_stores = BTreeSet::new();
        for table in &options.tables {
            let (name, path) = table
                .split_once('=')
                .filter(|(name, _)| !name.is_empty())
                .ok_or_else(|| plan_datafusion_err!("--table expects name=path"))?;
            let source = Location::parse(path)?;
            if source.0.scheme() == "s3"
                && input_stores.insert(source.origin().to_owned())
            {
                source.register(&ctx)?;
            }
            let path = source.input_path()?;
            ctx.register_parquet(name, path, ParquetReadOptions::default())
                .await?;
        }
        let sql_options = SQLOptions::new()
            .with_allow_ddl(false)
            .with_allow_dml(false)
            .with_allow_statements(false);
        let plan = ctx
            .sql_with_options(&options.sql, sql_options)
            .await?
            .create_physical_plan()
            .await?;
        options.sink.validate_plan(plan.as_ref())?;
        let (stage, mut splits) =
            split_plan_for_distribution_with_codecs(plan, 0, &codecs()?)?;
        splits.sort_unstable_by(|a, b| a.fingerprint.cmp(&b.fingerprint));
        for (id, split) in splits.iter_mut().enumerate() {
            split.split_id = id as u64;
        }
        let pending_splits = if let Some(root) = &output_root {
            root.sink(&stage, &options.sink)?
                .resume_filter(&splits)
                .await?
                .iter()
                .map(PendingSplit::from)
                .collect()
        } else {
            splits.iter().map(PendingSplit::from).collect()
        };
        Ok(Self {
            schema: SCHEMA.into(),
            stage,
            input_stores,
            options: options.sink,
            output_root: output_root.map(|root| root.0.into()),
            splits,
            pending_splits,
        })
    }

    pub(super) fn validate(&self) -> Result<()> {
        if self.schema != SCHEMA {
            return plan_err!("Unsupported stage job schema");
        }
        Ok(())
    }

    pub(super) fn stage(&self) -> &StagePlan {
        &self.stage
    }

    pub(super) fn split(&self, split_id: u64, attempt_id: u32) -> Result<Split> {
        let mut split = usize::try_from(split_id)
            .ok()
            .and_then(|index| self.splits.get(index))
            .filter(|split| split.split_id == split_id)
            .cloned()
            .ok_or_else(|| plan_datafusion_err!("Unknown split ID {split_id}"))?;
        split.attempt_id = attempt_id;
        Ok(split)
    }

    pub(super) fn worker(&self) -> Result<LocalWorker> {
        let ctx = session()?;
        for origin in &self.input_stores {
            let location = Location::parse(origin)?;
            if location.origin() != origin
                || location.0.scheme() != "s3"
                || !matches!(location.0.path(), "" | "/")
            {
                return plan_err!("Unsupported object store binding in stage job");
            }
            location.register(&ctx)?;
        }
        Ok(LocalWorker::with_context(ctx).with_codecs(codecs()?))
    }

    pub(super) fn sink(&self, output_root: Option<&str>) -> Result<ParquetSplitSink> {
        let job_root = self
            .output_root
            .as_deref()
            .map(Location::parse)
            .transpose()?;
        let argument = output_root.map(Location::parse).transpose()?;
        if let (Some(planned), Some(requested)) = (&job_root, &argument)
            && (planned.origin() != requested.origin()
                || planned.store_path()? != requested.store_path()?)
        {
            return plan_err!(
                "--output-root differs from the stage job's checkpoint root"
            );
        }
        argument
            .or(job_root)
            .ok_or_else(|| plan_datafusion_err!("--output-root is required"))?
            .sink(&self.stage, &self.options)
    }
}

#[derive(Clone)]
struct Location(Url);

impl Location {
    fn parse(value: &str) -> Result<Self> {
        if value.is_empty() {
            return plan_err!("An object store path cannot be empty");
        }
        let url = match Url::parse(value) {
            Ok(url) => url,
            Err(ParseError::RelativeUrlWithoutBase) => {
                Url::from_file_path(absolute(value)?)
                    .map_err(|()| plan_datafusion_err!("Invalid local file path"))?
            }
            Err(error) => return plan_err!("Invalid object store URL: {error}"),
        };
        if !url.username().is_empty()
            || url.password().is_some()
            || url.query().is_some()
            || url.fragment().is_some()
        {
            return plan_err!(
                "Object store URLs cannot contain embedded credentials, queries, or fragments"
            );
        }
        match url.scheme() {
            "file" if url.host_str().is_none() && url.to_file_path().is_ok() => {
                Ok(Self(url))
            }
            #[cfg(feature = "s3")]
            "s3" if url.host_str().is_some() && url.port().is_none() => Ok(Self(url)),
            #[cfg(not(feature = "s3"))]
            "s3" if url.host_str().is_some() && url.port().is_none() => {
                plan_err!("S3 object store URLs require the s3 feature")
            }
            scheme => {
                plan_err!("Unsupported object store URL scheme or authority: {scheme}")
            }
        }
    }

    fn origin(&self) -> &str {
        &self.0[..Position::BeforePath]
    }

    fn input_path(&self) -> Result<String> {
        if self.0.scheme() == "file" {
            let path = self
                .0
                .to_file_path()
                .map_err(|()| plan_datafusion_err!("Invalid local file URL"))?;
            path.into_os_string()
                .into_string()
                .map_err(|_| plan_datafusion_err!("Non-UTF-8 local path"))
        } else {
            Ok(self.0.to_string())
        }
    }

    fn store_path(&self) -> Result<StorePath> {
        Ok(StorePath::from_url_path(self.0.path())?)
    }

    fn register(&self, ctx: &SessionContext) -> Result<()> {
        ctx.register_object_store(&self.0, self.store()?);
        Ok(())
    }

    fn store(&self) -> Result<Arc<dyn ObjectStore>> {
        Ok(match self.0.scheme() {
            "file" => Arc::new(LocalFileSystem::new()),
            #[cfg(feature = "s3")]
            "s3" => Arc::new(
                AmazonS3Builder::from_env()
                    .with_url(self.origin())
                    .build()?,
            ),
            scheme => return plan_err!("Unsupported object store binding: {scheme}"),
        })
    }

    fn sink(&self, stage: &StagePlan, options: &SinkOptions) -> Result<ParquetSplitSink> {
        ParquetSplitSink::try_new(self.store()?, &self.store_path()?, stage, options)
    }
}

fn session() -> Result<SessionContext> {
    let mut config = SessionConfig::new().with_target_partitions(1);
    config.options_mut().optimizer.repartition_file_scans = false;
    #[cfg(feature = "ai")]
    return Ok(SessionContext::new_with_state(
        SessionStateBuilder::new_with_default_features()
            .with_config(config)
            .with_llm_provider_registry(llm_registry()?)
            .build(),
    ));
    #[cfg(not(feature = "ai"))]
    Ok(SessionContext::new_with_config(config))
}

fn codecs() -> Result<DistCodecs> {
    #[cfg(feature = "ai")]
    {
        let mut registry = DistExtensionRegistry::default();
        registry.register(Arc::new(LlmExecCodec::new()))?;
        Ok(DistCodecs::default().with_physical_codecs(registry))
    }
    #[cfg(not(feature = "ai"))]
    Ok(DistCodecs::default())
}

#[cfg(feature = "ai")]
fn llm_registry() -> Result<LlmProviderRegistry> {
    fn env(name: &str) -> Result<Option<String>> {
        match std::env::var(name) {
            Ok(value) => Ok(Some(value)),
            Err(std::env::VarError::NotPresent) => Ok(None),
            Err(std::env::VarError::NotUnicode(_)) => plan_err!("{name} must be UTF-8"),
        }
    }

    let registry = LlmProviderRegistry::default();
    let provider = env("DF_DIST_LLM_PROVIDER")?.unwrap_or_else(|| "default".into());
    let backend: Arc<dyn LlmBackend> = match env("DF_DIST_LLM_BACKEND")?.as_deref() {
        None | Some("mock") => Arc::new(MockLlmBackend),
        #[cfg(feature = "ai-http")]
        Some("openai") => Arc::new(OpenAiHttpBackend::try_new(OpenAiHttpConfig {
            base_url: env("OPENAI_BASE_URL")?
                .unwrap_or_else(|| "https://api.openai.com/v1".into()),
            api_key: env("OPENAI_API_KEY")?,
            api_key_env: None,
            ca_cert_pem: None,
        })?),
        #[cfg(not(feature = "ai-http"))]
        Some("openai") => {
            return plan_err!("DF_DIST_LLM_BACKEND=openai requires the ai-http feature");
        }
        Some(_) => return plan_err!("DF_DIST_LLM_BACKEND must be mock or openai"),
    };
    registry.register(provider, backend)?;
    Ok(registry)
}
