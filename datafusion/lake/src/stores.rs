// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: See the NOTICE file distributed with this work for additional information regarding copyright ownership
// SPDX-FileContributor: Licensed to the Apache Software Foundation (ASF) under one or more contributor license agreements

use std::collections::HashMap;
use std::fmt;
use std::sync::Arc;

use async_trait::async_trait;
use datafusion_common::{Result, plan_datafusion_err};
use lance::dataset::builder::DatasetBuilder;
use lance::session::Session;
use lance_io::object_store::{
    DEFAULT_CLOUD_IO_PARALLELISM, ObjectStore as LanceObjectStore, ObjectStoreParams,
    ObjectStoreProvider, ObjectStoreRegistry, StorageOptions, uri_to_url,
};
use lance_table_runtime::{TableBuilder, TableSession};
use object_store::{ObjectStore, path::Path};
use url::Url;

type Store = Arc<dyn ObjectStore>;

/// Process-local object stores for Lance dataset URLs.
///
/// # Public Only for Internal Use:
///
/// This is not a public API and is for internal use only; see [API policy] for details.
/// [API policy]: https://datafusion.apache.org/contributor-guide/api-health.html#internal-public-apis
#[doc(hidden)]
#[derive(Clone, Default)]
pub struct LakeObjectStores {
    schemes: HashMap<String, Arc<SchemeStores>>,
}

impl fmt::Debug for LakeObjectStores {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("LakeObjectStores")
            .field("schemes", &self.schemes.keys())
            .finish_non_exhaustive()
    }
}

impl LakeObjectStores {
    /// Bind a store to a scheme (`s3` or `s3://`) or a credential-free URL prefix.
    /// The store receives the entire object key below the URL authority. An S3
    /// client bound to one bucket should be registered under that bucket's URL.
    /// Registering the same prefix again replaces its previous binding.
    ///
    /// # Errors
    /// Returns an error for an invalid scheme, URL, or object key prefix.
    ///
    /// # Public Only for Internal Use:
    ///
    /// This is not a public API and is for internal use only; see [API policy] for details.
    /// [API policy]: https://datafusion.apache.org/contributor-guide/api-health.html#internal-public-apis
    #[doc(hidden)]
    pub fn register(
        &mut self,
        prefix_or_url: &str,
        store: Arc<dyn ObjectStore>,
    ) -> Result<()> {
        let scheme = prefix_or_url.strip_suffix("://").unwrap_or(prefix_or_url);
        if scheme
            .as_bytes()
            .first()
            .is_some_and(u8::is_ascii_alphabetic)
            && scheme.bytes().all(|byte| {
                byte.is_ascii_alphanumeric() || matches!(byte, b'+' | b'-' | b'.')
            })
        {
            Arc::make_mut(self.schemes.entry(scheme.to_ascii_lowercase()).or_default())
                .scheme_store = Some(store);
        } else {
            let invalid_prefix = || {
                plan_datafusion_err!(
                    "Lake object store binding requires a scheme or a credential-free URL prefix with a valid object key"
                )
            };
            let url = Url::parse(prefix_or_url).map_err(|_| invalid_prefix())?;
            let prefix = ScopedPrefix {
                authority: url.authority().to_owned(),
                path: binding_path(&url).ok_or_else(invalid_prefix)?,
            };
            Arc::make_mut(self.schemes.entry(url.scheme().to_owned()).or_default())
                .prefixes
                .insert(prefix, store);
        }
        Ok(())
    }

    /// Resolve the most specific matching URL prefix, then the scheme binding.
    /// URL prefixes match whole path segments within the same authority.
    ///
    /// # Public Only for Internal Use:
    ///
    /// This is not a public API and is for internal use only; see [API policy] for details.
    /// [API policy]: https://datafusion.apache.org/contributor-guide/api-health.html#internal-public-apis
    #[doc(hidden)]
    pub fn resolve(&self, url: &Url) -> Option<Arc<dyn ObjectStore>> {
        self.schemes
            .get(url.scheme())?
            .resolve(url)
            .map(|(_, store)| Arc::clone(store))
    }

    pub(crate) fn matches_uri(&self, uri: &str) -> bool {
        !self.schemes.is_empty()
            && uri_to_url(uri).is_ok_and(|url| {
                self.schemes
                    .get(url.scheme())
                    .is_some_and(|stores| stores.resolve(&url).is_some())
            })
    }

    pub(crate) fn lance_builder(&self, uri: &str) -> DatasetBuilder {
        let builder = DatasetBuilder::from_uri(uri);
        if self.schemes.is_empty() {
            return builder;
        }
        let session = Session::default();
        self.install_providers(&session.store_registry());
        builder.with_session(Arc::new(session))
    }

    pub(crate) fn lance_table_builder(&self, uri: &Url) -> TableBuilder {
        let builder = TableBuilder::from_uri(uri.as_str());
        if self.schemes.is_empty() {
            return builder;
        }
        let session = TableSession::default();
        self.install_providers(&session.store_registry());
        builder.with_session(Arc::new(session))
    }

    fn install_providers(&self, registry: &ObjectStoreRegistry) {
        for (scheme, stores) in &self.schemes {
            let fallback = registry.get_provider(scheme);
            registry.insert(
                scheme,
                Arc::new(LanceStoreProvider {
                    stores: Arc::clone(stores),
                    fallback,
                }),
            );
        }
    }
}

#[derive(Clone, Default)]
struct SchemeStores {
    scheme_store: Option<Store>,
    prefixes: HashMap<ScopedPrefix, Store>,
}

#[derive(Clone, Hash, PartialEq, Eq)]
struct ScopedPrefix {
    authority: String,
    path: Path,
}

impl SchemeStores {
    fn resolve(&self, url: &Url) -> Option<(&str, &Store)> {
        let path = binding_path(url)?;
        self.prefixes
            .iter()
            .filter(|(prefix, _)| {
                prefix.authority == url.authority() && path.prefix_matches(&prefix.path)
            })
            .max_by_key(|(prefix, _)| prefix.path.as_ref().len())
            .map(|(prefix, store)| (prefix.path.as_ref(), store))
            .or_else(|| self.scheme_store.as_ref().map(|store| ("", store)))
    }
}

fn binding_path(url: &Url) -> Option<Path> {
    if url.cannot_be_a_base()
        || !url.username().is_empty()
        || url.password().is_some()
        || url.query().is_some()
        || url.fragment().is_some()
    {
        return None;
    }
    Path::from_url_path(url.path()).ok()
}

struct LanceStoreProvider {
    stores: Arc<SchemeStores>,
    fallback: Option<Arc<dyn ObjectStoreProvider>>,
}

impl fmt::Debug for LanceStoreProvider {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("LanceStoreProvider").finish_non_exhaustive()
    }
}

impl LanceStoreProvider {
    fn bound_cache_prefix(&self, url: &Url) -> Option<String> {
        let (path, _) = self.stores.resolve(url)?;
        // Lance caches stores by this key, including stores for additional dataset bases.
        Some(format!(
            "lake:{}://{}/{path}",
            url.scheme(),
            url.authority()
        ))
    }

    fn fallback(&self, url: &Url) -> lance::Result<&dyn ObjectStoreProvider> {
        self.fallback.as_deref().ok_or_else(|| {
            lance::Error::invalid_input(format!(
                "No Lake object store is registered for scheme {:?} and the requested URL prefix",
                url.scheme()
            ))
        })
    }
}

#[async_trait]
impl ObjectStoreProvider for LanceStoreProvider {
    async fn new_store(
        &self,
        url: Url,
        params: &ObjectStoreParams,
    ) -> lance::Result<LanceObjectStore> {
        let Some((_, store)) = self.stores.resolve(&url) else {
            return self.fallback(&url)?.new_store(url, params).await;
        };
        let options =
            StorageOptions::new(params.storage_options().cloned().unwrap_or_default());
        LanceObjectStore::try_new(
            Arc::clone(store),
            url,
            params.block_size,
            None,
            params.use_constant_size_upload_parts,
            params.list_is_lexically_ordered.unwrap_or_default(),
            DEFAULT_CLOUD_IO_PARALLELISM,
            options.download_retry_count(),
            params.storage_options(),
        )
    }

    fn extract_path(&self, url: &Url) -> lance::Result<Path> {
        if self.stores.resolve(url).is_some() {
            Path::from_url_path(url.path()).map_err(|error| {
                lance::Error::invalid_input(format!("Invalid Lake object key: {error}"))
            })
        } else {
            self.fallback(url)?.extract_path(url)
        }
    }

    fn calculate_object_store_prefix(
        &self,
        url: &Url,
        storage_options: Option<&HashMap<String, String>>,
    ) -> lance::Result<String> {
        match self.bound_cache_prefix(url) {
            Some(prefix) => Ok(prefix),
            None => self
                .fallback(url)?
                .calculate_object_store_prefix(url, storage_options),
        }
    }

    fn calculate_object_store_prefix_for_match(
        &self,
        url: &Url,
        storage_options: Option<&HashMap<String, String>>,
    ) -> lance::Result<Option<String>> {
        if let Some(prefix) = self.bound_cache_prefix(url) {
            return Ok(Some(prefix));
        }
        match &self.fallback {
            Some(provider) => {
                provider.calculate_object_store_prefix_for_match(url, storage_options)
            }
            None => Ok(None),
        }
    }
}
