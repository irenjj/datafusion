// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: See the NOTICE file distributed with this work for additional information regarding copyright ownership
// SPDX-FileContributor: Licensed to the Apache Software Foundation (ASF) under one or more contributor license agreements

use std::io::{Seek, SeekFrom, Write};
use std::sync::Arc;

use arrow::array::{Array, FixedSizeListArray, RecordBatch, UInt64Array};
use arrow::datatypes::Float32Type;
use datafusion_common::DataFusionError;
use datafusion_execution::SendableRecordBatchStream;
use datafusion_lake::{
    LakeTableOptions, LanceReadOnlyTable, PaimonReadOnlyTable, RequiredIndexSpec,
};
use datafusion_physical_plan::stream::RecordBatchStreamAdapter;
use futures::stream;
use lance_core::cache::LanceCache;
use lance_core::datatypes::Schema;
use lance_core::utils::address::RowAddress;
use lance_core::{ROW_ADDR, ROW_ID};
use lance_file::version::ConcreteFileVersion;
use lance_file::versions::create_writer;
use lance_index::scalar::bitmap::BitmapIndexPlugin;
use lance_index::scalar::index_files_to_table;
use lance_index::scalar::lance_format::LanceIndexStore;
use lance_index::scalar::registry::{BasicTrainer, VALUE_COLUMN_NAME};
use lance_index::vector::DISTANCE_TYPE_KEY;
use lance_index::vector::flat::{
    index::{FlatIndex, FlatQuantizer},
    storage::FLAT_COLUMN,
};
use lance_index::vector::ivf::storage::{IVF_METADATA_KEY, IvfModel};
use lance_index::vector::quantizer::Quantization;
use lance_index::vector::shared::{
    SupportedIvfIndexType, write_unified_ivf_and_index_metadata,
};
use lance_index::vector::storage::STORAGE_METADATA_KEY;
use lance_index::vector::v3::subindex::IvfSubIndex;
use lance_index::{INDEX_AUXILIARY_FILE_NAME, INDEX_FILE_NAME, VECTOR_INDEX_VERSION, pb};
use lance_io::object_store::ObjectStore;
use lance_linalg::distance::DistanceType;
use lance_table::format::{IndexFile, IndexMetadata, Manifest};
use lance_table::io::commit::write_manifest_file_to_path;
use lance_table_runtime::write::WriteMode;
use paimon::spec::IndexManifest;
use prost::Message;
use uuid::Uuid;

use super::fixtures::{
    new_paimon, scalar_paimon_schema, scalar_rows, write_lance, write_paimon,
};

fn scalar_index(column: &str) -> LakeTableOptions {
    LakeTableOptions {
        required_indices: vec![RequiredIndexSpec::Scalar {
            column: column.into(),
        }],
    }
}

fn vector_index(dimension: u32) -> LakeTableOptions {
    LakeTableOptions {
        required_indices: vec![RequiredIndexSpec::Vector {
            column: "vector".into(),
            dimension,
        }],
    }
}

async fn attach_bitmap(
    table: &lance_table_runtime::Table,
    batch: &RecordBatch,
    fragments: usize,
) {
    let (store, _) = ObjectStore::from_uri(table.uri()).await.unwrap();
    let uuid = Uuid::from_u128(1);
    let index_store = LanceIndexStore::with_format_version(
        Arc::clone(&store),
        table.indices_dir().join(uuid.to_string()),
        Arc::new(LanceCache::with_capacity(1024 * 1024)),
        ConcreteFileVersion::V2_2,
    );
    let mut covered = Vec::new();
    let mut addresses = Vec::new();
    for fragment in table.manifest().fragments.iter().take(fragments) {
        let id = u32::try_from(fragment.id).unwrap();
        covered.push(id);
        let rows = u32::try_from(fragment.physical_rows.unwrap()).unwrap();
        addresses.extend(
            (0..rows).map(|offset| u64::from(RowAddress::new_from_parts(id, offset))),
        );
    }
    let keys = RecordBatch::try_from_iter([
        (VALUE_COLUMN_NAME, batch["id"].slice(0, addresses.len())),
        (ROW_ADDR, Arc::new(UInt64Array::from(addresses)) as _),
    ])
    .unwrap();
    let plugin = BitmapIndexPlugin;
    let request = plugin
        .new_training_request("", keys.schema().field(0))
        .unwrap();
    let input: SendableRecordBatchStream = Box::pin(RecordBatchStreamAdapter::new(
        keys.schema(),
        stream::iter([Ok(keys)]),
    ));
    let built = plugin
        .train_index(
            input,
            &index_store,
            request,
            None,
            lance_index::progress::noop_progress(),
        )
        .await
        .unwrap();
    let meta = IndexMetadata {
        uuid,
        name: "id_bitmap".into(),
        fields: vec![table.schema().field_id("id").unwrap()],
        covering_fields: Vec::new(),
        dataset_version: table.manifest().version,
        fragment_bitmap: Some(covered.into_iter().collect()),
        index_details: Some(Arc::new(built.index_details)),
        index_version: i32::try_from(built.index_version).unwrap(),
        created_at: None,
        base_id: None,
        files: Some(index_files_to_table(built.files)),
        clustering_id: None,
        metadata: Default::default(),
    };
    publish_indices(table, vec![meta]).await;
}

async fn publish_indices(
    table: &lance_table_runtime::Table,
    indices: Vec<IndexMetadata>,
) {
    let (store, base) = ObjectStore::from_uri(table.uri()).await.unwrap();
    let mut manifest = Manifest::new_from_previous(
        table.manifest(),
        table.schema().clone(),
        table.manifest().fragments.clone(),
    );
    let path = table
        .manifest_location()
        .naming_scheme
        .manifest_path(&base, manifest.version);
    write_manifest_file_to_path(&store, &mut manifest, Some(indices), &path, None)
        .await
        .unwrap();
}

#[tokio::test]
async fn lance_validates_index_kind_dimension_and_fragment_coverage() {
    for covered in [1, 2] {
        let directory = tempfile::tempdir().unwrap();
        let uri = directory.path().to_str().unwrap();
        let batch = scalar_rows(0);
        let table = write_lance(uri, batch.clone(), WriteMode::Create).await;
        attach_bitmap(&table, &batch, covered).await;
        let opened = LanceReadOnlyTable::open(uri, None, &scalar_index("id")).await;
        if covered == 2 {
            opened.unwrap();
            let missing =
                LanceReadOnlyTable::open(uri, None, &scalar_index("label")).await;
            assert!(
                matches!(missing, Err(DataFusionError::Plan(message)) if message.contains("missing"))
            );
            let wrong_kind = LakeTableOptions {
                required_indices: vec![RequiredIndexSpec::FullText {
                    column: "id".into(),
                }],
            };
            assert!(matches!(
                LanceReadOnlyTable::open(uri, None, &wrong_kind).await,
                Err(DataFusionError::Plan(_))
            ));
            let wrong_dim = LanceReadOnlyTable::open(uri, None, &vector_index(3)).await;
            assert!(
                matches!(wrong_dim, Err(DataFusionError::Plan(message)) if message.contains("dimension 3"))
            );
        } else {
            assert!(
                matches!(opened, Err(DataFusionError::Plan(message)) if message.contains("does not cover"))
            );
        }
    }
}

#[tokio::test]
async fn paimon_validates_index_publication_dimension_and_row_coverage() {
    let directory = tempfile::tempdir().unwrap();
    let schema = scalar_paimon_schema()
        .option("data-evolution.enabled", "true")
        .option("row-tracking.enabled", "true")
        .option("global-index.enabled", "true")
        .build()
        .unwrap();
    let table = new_paimon(directory.path(), schema).await;
    write_paimon(&table, &scalar_rows(0)).await;

    let missing =
        PaimonReadOnlyTable::open(table.clone(), None, &scalar_index("id")).await;
    assert!(
        matches!(missing, Err(DataFusionError::Plan(message)) if message.contains("index manifest"))
    );
    let wrong_dim =
        PaimonReadOnlyTable::open(table.clone(), None, &vector_index(3)).await;
    assert!(
        matches!(wrong_dim, Err(DataFusionError::Plan(message)) if message.contains("dimension 3"))
    );
    let unknown_col =
        PaimonReadOnlyTable::open(table.clone(), None, &scalar_index("missing")).await;
    assert!(
        matches!(unknown_col, Err(DataFusionError::Plan(message)) if message.contains("does not exist"))
    );

    let mut build = table.new_global_index_build_builder();
    let files = Box::pin(build.with_index_column("id").execute())
        .await
        .unwrap();
    assert!(files > 0);
    for (column, index_type) in [("vector", "ivf-flat"), ("label", "full-text")] {
        let mut build = table.new_global_index_build_builder();
        let files = Box::pin(
            build
                .with_index_column(column)
                .with_index_type(index_type)
                .execute(),
        )
        .await
        .unwrap();
        assert!(files > 0);
    }
    let mut required = scalar_index("id");
    required
        .required_indices
        .extend(vector_index(2).required_indices);
    required.required_indices.push(RequiredIndexSpec::FullText {
        column: "label".into(),
    });
    let indexed = PaimonReadOnlyTable::open(table.clone(), None, &required)
        .await
        .unwrap();

    write_paimon(&table, &scalar_rows(6)).await;
    PaimonReadOnlyTable::open(
        table.clone(),
        Some(indexed.snapshot_id()),
        &scalar_index("id"),
    )
    .await
    .unwrap();
    let stale = PaimonReadOnlyTable::open(table, None, &scalar_index("id")).await;
    assert!(
        matches!(stale, Err(DataFusionError::Plan(message)) if message.contains("does not cover"))
    );
}

async fn attach_vector(table: &lance_table_runtime::Table, dim: i32) {
    let (store, _) = ObjectStore::from_uri(table.uri()).await.unwrap();
    let uuid = Uuid::from_u128(2);
    let root = table.indices_dir().join(uuid.to_string());
    let addresses = table
        .manifest()
        .fragments
        .iter()
        .flat_map(|frag| {
            (0..u32::try_from(frag.physical_rows.unwrap()).unwrap()).map(|offset| {
                u64::from(RowAddress::new_from_parts(
                    u32::try_from(frag.id).unwrap(),
                    offset,
                ))
            })
        })
        .collect::<Vec<_>>();
    let rows = addresses.len();
    let vectors = FixedSizeListArray::from_iter_primitive::<Float32Type, _, _>(
        (0..rows).map(|_| Some((0..dim).map(|index| Some(index as f32)))),
        dim,
    );
    let distance = DistanceType::L2;
    let mut auxiliary_model = IvfModel::new(vectors.slice(0, 1), None);
    auxiliary_model.add_partition(u32::try_from(rows).unwrap());
    let batch = RecordBatch::try_from_iter([
        (ROW_ID, Arc::new(UInt64Array::from(addresses)) as _),
        (FLAT_COLUMN, Arc::new(vectors) as _),
    ])
    .unwrap();
    let mut auxiliary = create_writer(
        ConcreteFileVersion::V2_2,
        store
            .create(&root.clone().join(INDEX_AUXILIARY_FILE_NAME))
            .await
            .unwrap(),
        Schema::try_from(batch.schema().as_ref()).unwrap(),
        Default::default(),
    )
    .unwrap();
    auxiliary.write_batch(&batch).await.unwrap();
    let buf = pb::Ivf::try_from(&auxiliary_model).unwrap().encode_to_vec();
    let slot = auxiliary.add_global_buffer(buf.into()).await.unwrap();
    auxiliary.add_schema_metadata(IVF_METADATA_KEY, slot.to_string());
    auxiliary.add_schema_metadata(DISTANCE_TYPE_KEY, distance.to_string());
    let quantizer = FlatQuantizer::new(usize::try_from(dim).unwrap(), distance);
    let metadata = serde_json::to_string(&quantizer.metadata(None)).unwrap();
    auxiliary.add_schema_metadata(
        STORAGE_METADATA_KEY,
        serde_json::to_string(&[metadata]).unwrap(),
    );

    let mut index = create_writer(
        ConcreteFileVersion::V2_2,
        store.create(&root.join(INDEX_FILE_NAME)).await.unwrap(),
        Schema::try_from(FlatIndex::schema().as_ref()).unwrap(),
        Default::default(),
    )
    .unwrap();
    let centroids = auxiliary_model.centroids.clone().unwrap();
    let mut index_model = IvfModel::new(centroids, None);
    index_model.add_partition(0);
    write_unified_ivf_and_index_metadata(
        &mut index,
        &index_model,
        distance,
        SupportedIvfIndexType::IvfFlat,
    )
    .await
    .unwrap();
    index.add_schema_metadata(FlatIndex::metadata_key(), r#"[""]"#.to_owned());
    let details = pb::VectorIndexDetails {
        metric_type: pb::VectorMetricType::L2.into(),
        ..Default::default()
    };
    let metadata = IndexMetadata {
        uuid,
        name: "embedding".into(),
        fields: vec![table.schema().field_id("vector").unwrap()],
        covering_fields: vec![],
        dataset_version: table.manifest().version,
        fragment_bitmap: Some(
            table
                .manifest()
                .fragments
                .iter()
                .map(|f| u32::try_from(f.id).unwrap())
                .collect(),
        ),
        index_details: Some(Arc::new(prost_types::Any::from_msg(&details).unwrap())),
        index_version: i32::try_from(VECTOR_INDEX_VERSION).unwrap(),
        created_at: None,
        base_id: None,
        files: Some(vec![
            IndexFile {
                path: INDEX_FILE_NAME.into(),
                size_bytes: index.finish().await.unwrap().size_bytes,
                file_metadata_size_bytes: None,
            },
            IndexFile {
                path: INDEX_AUXILIARY_FILE_NAME.into(),
                size_bytes: auxiliary.finish().await.unwrap().size_bytes,
                file_metadata_size_bytes: None,
            },
        ]),
        clustering_id: None,
        metadata: Default::default(),
    };
    publish_indices(table, vec![metadata]).await;
}

#[tokio::test]
async fn lance_validates_persisted_vector_dimension() {
    for dimension in [2, 3] {
        let directory = tempfile::tempdir().unwrap();
        let uri = directory.path().to_str().unwrap();
        let fixture = write_lance(uri, scalar_rows(0), WriteMode::Create).await;
        attach_vector(&fixture, dimension).await;
        let opened = LanceReadOnlyTable::open(uri, None, &vector_index(2)).await;
        if dimension == 2 {
            opened.unwrap();
        } else {
            assert!(
                matches!(opened, Err(DataFusionError::Plan(message)) if message.contains("dimension 3, expected 2"))
            );
        }
    }
}

#[tokio::test]
async fn paimon_rejects_persisted_vector_dimension_mismatch() {
    let directory = tempfile::tempdir().unwrap();
    let schema = scalar_paimon_schema()
        .option("data-evolution.enabled", "true")
        .option("row-tracking.enabled", "true")
        .option("global-index.enabled", "true")
        .build()
        .unwrap();
    let table = new_paimon(directory.path(), schema).await;
    write_paimon(&table, &scalar_rows(0)).await;
    let mut build = table.new_global_index_build_builder();
    Box::pin(
        build
            .with_index_column("vector")
            .with_index_type("ivf-flat")
            .execute(),
    )
    .await
    .unwrap();
    let manager = table.snapshot_manager();
    let snapshot = manager.get_latest_snapshot().await.unwrap().unwrap();
    let manifest = manager.manifest_path(snapshot.index_manifest().unwrap());
    let entries = IndexManifest::read(table.file_io(), &manifest)
        .await
        .unwrap();
    let vector = entries
        .iter()
        .find(|entry| entry.index_file.index_type == "ivf-flat")
        .unwrap();
    let location = format!("{}/index/{}", table.location(), vector.index_file.file_name);
    let mut file = std::fs::OpenOptions::new()
        .write(true)
        .open(location)
        .unwrap();
    file.seek(SeekFrom::Start(8)).unwrap();
    file.write_all(&3u32.to_le_bytes()).unwrap();
    let opened = PaimonReadOnlyTable::open(table, None, &vector_index(2)).await;
    assert!(
        matches!(opened, Err(DataFusionError::Plan(message)) if message.contains("dimension 3, expected 2"))
    );
}
