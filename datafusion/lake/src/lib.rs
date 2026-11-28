// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: See the NOTICE file distributed with this work for additional information regarding copyright ownership
// SPDX-FileContributor: Licensed to the Apache Software Foundation (ASF) under one or more contributor license agreements

//! Snapshot-pinned Lance and Paimon scans for the internal DataFusion lake layer.

#[doc(hidden)]
pub mod backfill;
#[doc(hidden)]
pub mod dist;
mod filter;
mod lance_table;
mod media;
mod paimon_table;
mod scan;
#[doc(hidden)]
pub mod search;
mod split;
mod validation;

pub use lance_table::LanceReadOnlyTable;
pub use media::LakeMediaResolver;
pub use paimon_table::PaimonReadOnlyTable;
pub use search::{
    LakeSearchCatalog, PinnedLakeTable, hybrid_search_function, vector_search_function,
};
pub use split::{
    InclusiveRowRange, LAKE_SPLIT_EXT_NAME, LAKE_SPLIT_EXT_VERSION, LakeSplitExt,
    LakeSplitPayload,
};
pub use validation::{LakeTableOptions, RequiredIndexSpec};
