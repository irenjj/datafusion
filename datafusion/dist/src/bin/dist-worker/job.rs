// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: See the NOTICE file distributed with this work for additional information regarding copyright ownership
// SPDX-FileContributor: Licensed to the Apache Software Foundation (ASF) under one or more contributor license agreements

#[cfg(feature = "ai-http")]
use std::collections::BTreeMap;
use std::collections::BTreeSet;
use std::path::absolute;
use std::sync::Arc;

use datafusion::common::{Result, plan_datafusion_err, plan_err};
#[cfg(feature = "ai")]
use datafusion::execution::SessionStateBuilder;
use datafusion::execution::context::SQLOptions;
use datafusion::execution::object_store::ObjectStoreUrl;
#[cfg(feature = "ai")]
use datafusion::execution::runtime_env::RuntimeEnv;
use datafusion::prelude::{ParquetReadOptions, SessionConfig, SessionContext};
#[cfg(feature = "ai")]
use datafusion_ai::ObjectStoreMediaResolver;
#[cfg(feature = "ai")]
use datafusion_ai::llm::{LlmBackend, LlmProviderRegistry, MockLlmBackend};
#[cfg(feature = "ai-http")]
use datafusion_ai::llm::{OpenAiHttpBackend, OpenAiHttpConfig};
use datafusion_dist::{
    DistCodecs, LocalWorker, ParquetSplitSink, SinkOptions, Split, StageOutputReader,
    StagePlan, split_plan_for_distribution_with_codecs,
};
#[cfg(feature = "ai")]
use datafusion_dist::{DistExtensionRegistry, LlmExecCodec};
#[cfg(feature = "s3")]
use object_store::aws::AmazonS3Builder;
use object_store::{ObjectStore, local::LocalFileSystem, path::Path as StorePath};
use serde::{Deserialize, Serialize};
use url::{ParseError, Position, Url};

const SCHEMA: &str = "df-stage-job-v1";
const PROVIDERS_ENV: &str = "DF_DIST_LLM_PROVIDERS";

pub(super) struct PlanningOptions {
    pub(super) sql: String,
    pub(super) tables: Vec<String>,
    pub(super) stage_tables: Vec<String>,
    pub(super) stage_dlq_tables: Vec<String>,
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
            let (name, path) = table_binding(table, "--table")?;
            let source = Location::parse(path)?;
            source.register_input(&ctx, &mut input_stores)?;
            let path = source.input_path()?;
            ctx.register_parquet(name, path, ParquetReadOptions::default())
                .await?;
        }
        for (flag, tables) in [
            ("--stage-table", &options.stage_tables),
            ("--stage-dlq-table", &options.stage_dlq_tables),
        ] {
            for table in tables {
                let (name, path) = table_binding(table, flag)?;
                let source = Location::parse(path)?;
                source.register_input(&ctx, &mut input_stores)?;
                let stage = source.open_stage(&ctx).await?;
                let provider = if flag == "--stage-dlq-table" {
                    stage.dlq_table()
                } else {
                    stage.data_table()
                };
                ctx.register_table(name, provider)?;
            }
        }
        let plan = ctx
            .sql_with_options(&options.sql, read_only_sql())
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
        self.sink_with_context(output_root, None)
    }

    pub(super) fn sink_with_context(
        &self,
        output_root: Option<&str>,
        ctx: Option<&SessionContext>,
    ) -> Result<ParquetSplitSink> {
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
        let location = argument
            .or(job_root)
            .ok_or_else(|| plan_datafusion_err!("--output-root is required"))?;
        let Some(ctx) = ctx else {
            return location.sink(&self.stage, &self.options);
        };
        if location.0.scheme() == "s3" && !self.input_stores.contains(location.origin()) {
            location.register(ctx)?;
        }
        ParquetSplitSink::try_new(
            ctx.runtime_env()
                .object_store_registry
                .get_store(&location.0)?,
            &location.store_path()?,
            &self.stage,
            &self.options,
        )
    }
}

fn table_binding<'a>(table: &'a str, flag: &str) -> Result<(&'a str, &'a str)> {
    table
        .split_once('=')
        .filter(|(name, _)| !name.is_empty())
        .ok_or_else(|| plan_datafusion_err!("{flag} expects name=path"))
}

pub(super) async fn query_stage(
    root: &str,
    sql: &str,
) -> Result<datafusion::dataframe::DataFrame> {
    let ctx = SessionContext::new();
    let location = Location::parse(root)?;
    location.register_input(&ctx, &mut BTreeSet::new())?;
    let stage = location.open_stage(&ctx).await?;
    ctx.register_table("data", stage.data_table())?;
    ctx.register_table("dlq", stage.dlq_table())?;
    ctx.sql_with_options(sql, read_only_sql()).await
}

fn read_only_sql() -> SQLOptions {
    SQLOptions::new()
        .with_allow_ddl(false)
        .with_allow_dml(false)
        .with_allow_statements(false)
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

    fn register_input(
        &self,
        ctx: &SessionContext,
        input_stores: &mut BTreeSet<String>,
    ) -> Result<()> {
        if self.0.scheme() == "s3" && input_stores.insert(self.origin().to_owned()) {
            self.register(ctx)?;
        }
        Ok(())
    }

    async fn open_stage(&self, ctx: &SessionContext) -> Result<StageOutputReader> {
        StageOutputReader::open(
            ctx,
            ObjectStoreUrl::parse(self.origin())?,
            &self.store_path()?,
        )
        .await
    }

    #[cfg(feature = "s3")]
    fn s3_builder(&self) -> AmazonS3Builder {
        // The URI supplies the bucket. Connection settings come from the environment.
        AmazonS3Builder::from_env()
            .with_bucket_name(self.0.host_str().expect("S3 locations have a bucket"))
    }

    fn store(&self) -> Result<Arc<dyn ObjectStore>> {
        Ok(match self.0.scheme() {
            "file" => Arc::new(LocalFileSystem::new()),
            #[cfg(feature = "s3")]
            "s3" => Arc::new(self.s3_builder().build()?),
            scheme => return plan_err!("Unsupported object store binding: {scheme}"),
        })
    }

    fn sink(&self, stage: &StagePlan, options: &SinkOptions) -> Result<ParquetSplitSink> {
        ParquetSplitSink::try_new(self.store()?, &self.store_path()?, stage, options)
    }
}

fn session() -> Result<SessionContext> {
    #[cfg(not(feature = "ai-http"))]
    if std::env::var_os(PROVIDERS_ENV).is_some() {
        return plan_err!("{PROVIDERS_ENV} requires the ai-http feature");
    }
    let mut config = SessionConfig::new().with_target_partitions(1);
    config.options_mut().optimizer.repartition_file_scans = false;
    #[cfg(feature = "ai")]
    {
        let runtime = Arc::new(RuntimeEnv::default());
        let config = config.with_extension(Arc::new(ObjectStoreMediaResolver::new(
            Arc::clone(&runtime.object_store_registry),
        )));
        Ok(SessionContext::new_with_state(
            SessionStateBuilder::new_with_default_features()
                .with_config(config)
                .with_llm_provider_registry(llm_registry()?)
                .with_runtime_env(runtime)
                .build(),
        ))
    }
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
fn env(name: &str) -> Result<Option<String>> {
    match std::env::var(name) {
        Ok(value) => Ok(Some(value)),
        Err(std::env::VarError::NotPresent) => Ok(None),
        Err(std::env::VarError::NotUnicode(_)) => plan_err!("{name} must be UTF-8"),
    }
}

#[cfg(feature = "ai")]
fn llm_registry() -> Result<LlmProviderRegistry> {
    #[cfg(feature = "ai-http")]
    if let Some(providers) = env(PROVIDERS_ENV)? {
        return named_providers(&providers, env("DF_DIST_LLM_PROVIDER")?);
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
            ca_cert_pem: match env("OPENAI_CA_CERT_PEM")? {
                Some(pem) => Some(pem.into_bytes()),
                None => std::env::var_os("OPENAI_CA_CERT_FILE")
                    .map(std::fs::read)
                    .transpose()?,
            },
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

#[cfg(feature = "ai-http")]
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct NamedProvider {
    base_url: String,
    api_key_env: String,
    ca_cert_env: Option<String>,
}

#[cfg(feature = "ai-http")]
fn named_providers(
    json: &str,
    default_name: Option<String>,
) -> Result<LlmProviderRegistry> {
    let providers: BTreeMap<String, NamedProvider> =
        serde_json::from_str(json).map_err(|_| {
            plan_datafusion_err!("Invalid {PROVIDERS_ENV} JSON configuration")
        })?;
    if providers.is_empty() || providers.keys().any(|name| name.trim().is_empty()) {
        return plan_err!(
            "{PROVIDERS_ENV} requires at least one non-empty provider name"
        );
    }
    if let Some(name) = &default_name
        && !providers.contains_key(name)
    {
        return plan_err!("DF_DIST_LLM_PROVIDER must name a provider in {PROVIDERS_ENV}");
    }
    let registry = LlmProviderRegistry::empty_with_default(
        default_name.unwrap_or_else(|| "default".into()),
    );
    for (name, config) in providers {
        validate_env_name(&config.api_key_env, "api_key_env")?;
        if env(&config.api_key_env)?.is_none_or(|key| key.trim().is_empty()) {
            return plan_err!(
                "Missing or empty API key environment variable {}",
                config.api_key_env
            );
        }
        let ca_cert_pem = config
            .ca_cert_env
            .as_deref()
            .map(|variable| {
                validate_env_name(variable, "ca_cert_env")?;
                let read_error =
                    || plan_datafusion_err!("Could not read CA from {variable}");
                let value = std::env::var_os(variable).ok_or_else(read_error)?;
                if value
                    .as_encoded_bytes()
                    .starts_with(b"-----BEGIN CERTIFICATE-----")
                {
                    Ok(value.into_encoded_bytes())
                } else {
                    std::fs::read(value).map_err(|_| read_error())
                }
            })
            .transpose()?;
        let backend = OpenAiHttpBackend::try_new(OpenAiHttpConfig {
            base_url: config.base_url,
            api_key: None,
            api_key_env: Some(config.api_key_env),
            ca_cert_pem,
        })
        .map_err(|error| {
            plan_datafusion_err!("Invalid {PROVIDERS_ENV} configuration: {error}")
        })?;
        registry.register(name, Arc::new(backend))?;
    }
    Ok(registry)
}

#[cfg(feature = "ai-http")]
fn validate_env_name(name: &str, field: &str) -> Result<()> {
    let mut bytes = name.bytes();
    if !bytes
        .next()
        .is_some_and(|byte| byte.is_ascii_alphabetic() || byte == b'_')
        || !bytes.all(|byte| byte.is_ascii_alphanumeric() || byte == b'_')
    {
        return plan_err!("{PROVIDERS_ENV} {field} must be an environment variable name");
    }
    Ok(())
}

#[cfg(all(test, feature = "s3"))]
mod s3_tests {
    use std::process::Command;

    use object_store::{ClientConfigKey, aws::AmazonS3ConfigKey};

    use super::*;

    #[test]
    fn s3_builder_preserves_environment() -> Result<()> {
        if std::env::var_os("DF_DIST_TEST_S3_CONFIG").is_none() {
            const TEST: &str = "job::s3_tests::s3_builder_preserves_environment";
            let result = Command::new(std::env::current_exe()?)
                .args(["--exact", TEST, "--nocapture"])
                .env_clear()
                .envs([
                    ("DF_DIST_TEST_S3_CONFIG", "1"),
                    ("AWS_BUCKET", "environment-bucket"),
                    ("AWS_ENDPOINT", "http://host:80"),
                    ("AWS_ALLOW_HTTP", "true"),
                    ("AWS_VIRTUAL_HOSTED_STYLE_REQUEST", "false"),
                    ("AWS_REGION", "us-west-2"),
                    ("AWS_ACCESS_KEY_ID", "test-key"),
                    ("AWS_SECRET_ACCESS_KEY", "test-secret"),
                ])
                .output()?;
            let stdout = String::from_utf8_lossy(&result.stdout);
            assert!(
                result.status.success() && stdout.contains("1 passed"),
                "{stdout}\n{}",
                String::from_utf8_lossy(&result.stderr)
            );
            return Ok(());
        }

        use AmazonS3ConfigKey::*;
        let location = Location::parse("s3://stage-bucket/path/file.parquet")?;
        let builder = location.s3_builder();
        for (key, value) in [
            (Bucket, "stage-bucket"),
            (Endpoint, "http://host:80"),
            (Client(ClientConfigKey::AllowHttp), "true"),
            (VirtualHostedStyleRequest, "false"),
            (Region, "us-west-2"),
            (AccessKeyId, "test-key"),
            (SecretAccessKey, "test-secret"),
        ] {
            assert_eq!(builder.get_config_value(&key).as_deref(), Some(value));
        }
        builder.build()?;
        Ok(())
    }
}

#[cfg(all(test, feature = "ai"))]
mod tests {
    use datafusion_ai::{MediaRef, MediaResolver};
    use object_store::{ObjectStoreExt, memory::InMemory};

    use super::*;

    #[tokio::test]
    async fn media_resolver_sees_stores_registered_after_session_creation() -> Result<()>
    {
        let ctx = session()?;
        let resolver = ctx
            .state()
            .config()
            .get_extension::<ObjectStoreMediaResolver>()
            .unwrap();
        for uri in ["file:///asset", "s3://late-media/asset"] {
            let store = Arc::new(InMemory::new());
            store
                .put(&StorePath::from("asset"), "abcdef".into())
                .await?;
            ctx.register_object_store(&Url::parse(uri).unwrap(), store);
            let media = MediaRef {
                uri: uri.into(),
                byte_offset: Some(2),
                byte_len: Some(3),
                etag: None,
                mime: None,
            };
            assert_eq!(resolver.resolve(&media, Some(3)).await?.bytes, "cde");
        }
        Ok(())
    }
}
