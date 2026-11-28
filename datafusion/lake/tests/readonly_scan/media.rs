// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: See the NOTICE file distributed with this work for additional information regarding copyright ownership
// SPDX-FileContributor: Licensed to the Apache Software Foundation (ASF) under one or more contributor license agreements

use std::collections::HashMap;
use std::ops::Range;
use std::sync::Arc;

use arrow::array::{
    Array, Int32Array, LargeBinaryArray, RecordBatch, StringArray, StructArray,
    UInt64Array,
};
use arrow::datatypes::{DataType, Field, Schema};
use bytes::Bytes;
use datafusion_ai::{MediaRef, MediaRefType, MediaResolver};
use datafusion_common::DataFusionError;
use datafusion_execution::object_store::{
    DefaultObjectStoreRegistry, ObjectStoreRegistry,
};
use datafusion_lake::{
    LakeMediaResolver, LakeTableOptions, LanceReadOnlyTable, PaimonReadOnlyTable,
};
use lance::blob::{
    BlobDescriptor, BlobDescriptorArrayBuilder, BlobRange, DedicatedBlobWriter,
    PackedBlobWriter, blob_field,
};
use lance::dataset::builder::DatasetBuilder;
use lance_file::version::LanceFileVersion;
use lance_io::object_store::{ObjectStore, ObjectStoreParams, WrappingObjectStore};
use lance_table_runtime::write::{InsertBuilder, WriteParams};
use object_store::{ObjectStoreExt, memory::InMemory, path::Path};
use paimon::io::FileIO;
use paimon::spec::{
    BlobDescriptor as PaimonBlob, BlobType, DataType as PaimonType, IntType,
    Schema as PaimonSchema,
};
use url::Url;

use super::fixtures::{new_paimon, scan, write_paimon};
use super::store::{Read, Reads};

const MIME: &str = "application/octet-stream";
const INLINE_LEN: usize = 128 * 1024;

fn field_with_mime(field: Field, mime: &str) -> Field {
    let mut field = field;
    field.metadata_mut().insert("df.media.mime", mime);
    field
}

pub(super) fn read_references(batches: &[RecordBatch]) -> Vec<Option<MediaRef>> {
    batches
        .iter()
        .flat_map(|batch| {
            let schema = batch.schema();
            let field = schema.field(0);
            field.try_extension_type::<MediaRefType>().unwrap();
            let column = batch
                .column(0)
                .as_any()
                .downcast_ref::<StructArray>()
                .unwrap();
            let uri = column
                .column(0)
                .as_any()
                .downcast_ref::<StringArray>()
                .unwrap();
            let offsets = column
                .column(1)
                .as_any()
                .downcast_ref::<UInt64Array>()
                .unwrap();
            let lengths = column
                .column(2)
                .as_any()
                .downcast_ref::<UInt64Array>()
                .unwrap();
            let etags = column
                .column(3)
                .as_any()
                .downcast_ref::<StringArray>()
                .unwrap();
            let mime = column
                .column(4)
                .as_any()
                .downcast_ref::<StringArray>()
                .unwrap();
            (0..column.len())
                .map(|row| {
                    if column.is_null(row) {
                        return None;
                    }
                    Some(MediaRef {
                        uri: uri.value(row).to_owned(),
                        byte_offset: offsets.is_valid(row).then(|| offsets.value(row)),
                        byte_len: lengths.is_valid(row).then(|| lengths.value(row)),
                        etag: etags.is_valid(row).then(|| etags.value(row).to_owned()),
                        mime: mime.is_valid(row).then(|| mime.value(row).to_owned()),
                    })
                })
                .collect::<Vec<_>>()
        })
        .collect()
}

pub(super) fn observed_lance(path: &str, reads: &Reads) -> DatasetBuilder {
    DatasetBuilder::from_uri(path).with_store_params(ObjectStoreParams {
        object_store_wrapper: Some(Arc::new(reads.clone())),
        ..Default::default()
    })
}

pub(super) fn object_read_ranges<'a>(
    reads: &'a [Read],
    media: &MediaRef,
) -> impl Iterator<Item = &'a Range<u64>> {
    let uri = Url::parse(&media.uri).unwrap();
    let path = Path::from_url_path(uri.path()).unwrap();
    reads
        .iter()
        .filter(move |read| read.path == path)
        .filter_map(|read| read.range.as_ref())
}

fn assert_inline_descriptor_io(
    reads: &[Read],
    media: &MediaRef,
    block_size: u64,
    cold: bool,
) {
    let uri = Url::parse(&media.uri).unwrap();
    let size = std::fs::metadata(uri.to_file_path().unwrap())
        .unwrap()
        .len();
    let tail = size - block_size..size;
    let start = media.byte_offset.unwrap_or(0);
    assert_eq!(media.byte_len, Some(INLINE_LEN as u64));
    let end = start + INLINE_LEN as u64;
    assert!(start < tail.start);
    if cold {
        assert!(object_read_ranges(reads, media).any(|range| *range == tail));
    }
    // Only Lance's exact footer prefetch may overlap the inline payload.
    let payload_bytes_read: u64 = object_read_ranges(reads, media)
        .filter(|range| **range != tail)
        .map(|range| range.end.min(end).saturating_sub(range.start.max(start)))
        .sum();
    assert_eq!(payload_bytes_read, 0, "{media:?}, {reads:?}");
}

fn assert_separate_payloads_unread(reads: &[Read], references: &[MediaRef]) {
    for reference in references {
        let bytes: u64 = object_read_ranges(reads, reference)
            .map(|r| r.end - r.start)
            .sum();
        assert_eq!(bytes, 0, "{reference:?}, {reads:?}");
    }
}

#[tokio::test]
async fn lance_v2_descriptor_io_is_limited_to_metadata_prefetch() {
    let directory = tempfile::tempdir().unwrap();
    let dataset_path = directory.path().join("fixture.lance");
    let uri = dataset_path.to_str().unwrap();
    std::fs::create_dir(&dataset_path).unwrap();
    let external_path = directory.path().join("external payload.bin");
    let managed_path = dataset_path.join("managed payload.bin");
    std::fs::write(&external_path, b"0123456789").unwrap();
    std::fs::write(&managed_path, b"abcdefghijkl").unwrap();

    let field = field_with_mime(blob_field("payload", true), MIME);
    let mut values = BlobDescriptorArrayBuilder::for_field(&field).unwrap();
    let packed = vec![41u8; 32];
    let dedicated = vec![57u8; 96];
    let file_name = "fixture-data.lance";
    let (store, base) = ObjectStore::from_uri(uri).await.unwrap();
    let block_size = store.block_size() as u64;
    let data_file = base.join("data").join(file_name);
    let mut packed_writer =
        PackedBlobWriter::try_new(store.as_ref().clone(), data_file.clone(), 5)
            .await
            .unwrap();
    packed_writer.write_blob(b"prefix").await.unwrap();
    packed_writer.write_blob(&packed).await.unwrap();
    let mut packed_refs = packed_writer.finish().await.unwrap();
    let mut dedicated_writer =
        DedicatedBlobWriter::try_new(store.as_ref().clone(), data_file, 8)
            .await
            .unwrap();
    dedicated_writer.write(&dedicated).await.unwrap();
    let inline: Vec<_> = (0..INLINE_LEN).map(|index| (index % 251) as u8).collect();
    values.push_inline(Bytes::copy_from_slice(&inline)).unwrap();
    values.push(packed_refs.pop().unwrap()).unwrap();
    values
        .push(dedicated_writer.finish().await.unwrap())
        .unwrap();
    let external_uri = Url::from_file_path(&external_path).unwrap();
    values
        .push_external(
            external_uri.as_str(),
            Some(BlobRange { offset: 3, size: 4 }),
        )
        .unwrap();
    values.push_external(external_uri.as_str(), None).unwrap();
    values
        .push(BlobDescriptor::Managed {
            base_id: None,
            uri: "managed payload.bin".into(),
            offset: 2,
            size: 3,
        })
        .unwrap();
    values.push_inline(Bytes::new()).unwrap();
    values.push_null().unwrap();
    let (field, values) = values.finish().unwrap().into_parts();
    let schema = Schema::new_with_metadata(
        vec![field, Field::new("id", DataType::Int32, false)],
        HashMap::from([("lance.blob.managed".into(), "true".into())]),
    );
    let batch = RecordBatch::try_new(
        Arc::new(schema),
        vec![values, Arc::new(Int32Array::from_iter_values(0..8))],
    )
    .unwrap();
    InsertBuilder::new(uri)
        .with_preassigned_data_file_name(file_name)
        .with_params(
            &WriteParams::with_storage_version(LanceFileVersion::V2_2)
                .with_allow_external_blob_outside_bases(true),
        )
        .execute(vec![batch])
        .await
        .unwrap();

    let reads = Reads::new();
    let table = LanceReadOnlyTable::open_with_builder(
        observed_lance(uri, &reads),
        &LakeTableOptions::default(),
    )
    .await
    .unwrap();
    let rows = read_references(&scan(&table, Some(&[0]), &[], None).await);
    let during_projection = reads.take();
    assert!(during_projection.iter().any(|read| read.range.is_some()));
    assert_eq!(rows.len(), 8);
    assert!(rows[7].is_none());
    let references: Vec<_> = rows.into_iter().flatten().collect();
    let inline_path = dataset_path.join("data").join(file_name);
    assert_eq!(
        references[0].uri,
        Url::from_file_path(inline_path).unwrap().as_str()
    );
    for (reference, id) in [(&references[1], 5), (&references[2], 8)] {
        let path = lance_core::utils::blob::blob_path(&Path::ROOT, "fixture-data", id);
        let sidecar = dataset_path.join("data").join(path.as_ref());
        assert_eq!(
            reference.uri,
            Url::from_file_path(sidecar).unwrap().as_str()
        );
    }
    assert_eq!(references[1].byte_offset, Some(6));
    assert_eq!(references[4].byte_len, None);
    assert_eq!(references[5].byte_offset, Some(2));
    assert_eq!(references[6].byte_offset, Some(0));
    assert_eq!(references[6].byte_len, Some(0));
    let separate_payloads = &references[1..6];
    assert_inline_descriptor_io(&during_projection, &references[0], block_size, true);
    assert_separate_payloads_unread(&during_projection, separate_payloads);

    let resolver = LakeMediaResolver::default();
    let expected: [&[u8]; 7] = [
        &inline,
        &packed,
        &dedicated,
        b"3456",
        b"0123456789",
        b"cde",
        b"",
    ];
    for (reference, expected) in references.iter().zip(expected) {
        let result = resolver
            .resolve(reference, Some(expected.len() as u64))
            .await
            .unwrap();
        assert_eq!(result.bytes.as_ref(), expected);
        assert_eq!(result.mime.as_deref(), Some(MIME));
    }
    let mixed = scan(
        &table,
        Some(&[0, 1]),
        &[datafusion_expr::col("id").lt(datafusion_expr::lit(3))],
        None,
    )
    .await;
    assert_eq!(super::fixtures::ids(&mixed), [0, 1, 2]);
    let mixed_reads = reads.take();
    assert_inline_descriptor_io(&mixed_reads, &references[0], block_size, false);
    assert_separate_payloads_unread(&mixed_reads, separate_payloads);

    let scalar_only = scan(&table, Some(&[1]), &[], None).await;
    assert_eq!(
        super::fixtures::ids(&scalar_only),
        (0..8).collect::<Vec<_>>()
    );
    let scalar_reads = reads.take();
    assert_inline_descriptor_io(&scalar_reads, &references[0], block_size, false);
    assert_separate_payloads_unread(&scalar_reads, separate_payloads);
}

#[tokio::test]
async fn lance_v1_descriptors_preserve_inline_empty_and_null() {
    let directory = tempfile::tempdir().unwrap();
    let uri = directory.path().to_str().unwrap();
    let field = Field::new("payload", DataType::LargeBinary, true).with_metadata(
        HashMap::from([("lance-encoding:blob".into(), "true".into())]),
    );
    let payload: Vec<_> = (0..INLINE_LEN).map(|index| (index % 241) as u8).collect();
    let data = LargeBinaryArray::from(vec![Some(payload.as_slice()), Some(b""), None]);
    let batch =
        RecordBatch::try_new(Arc::new(Schema::new(vec![field])), vec![Arc::new(data)])
            .unwrap();
    InsertBuilder::new(uri)
        .with_params(&WriteParams::with_storage_version(LanceFileVersion::V2_0))
        .execute(vec![batch])
        .await
        .unwrap();
    let reads = Reads::new();
    let table = LanceReadOnlyTable::open_with_builder(
        observed_lance(uri, &reads),
        &LakeTableOptions::default(),
    )
    .await
    .unwrap();
    let references = read_references(&scan(&table, None, &[], None).await);
    assert_eq!(references.len(), 3);
    assert!(references[2].is_none());
    let inline = references[0].as_ref().unwrap();
    let empty = references[1].as_ref().unwrap();
    assert_eq!(empty.byte_len, Some(0));
    let (store, _) = ObjectStore::from_uri(uri).await.unwrap();
    assert_inline_descriptor_io(&reads.take(), inline, store.block_size() as u64, true);
    let resolver = LakeMediaResolver::default();
    assert_eq!(
        resolver
            .resolve(inline, Some(INLINE_LEN as u64))
            .await
            .unwrap()
            .bytes,
        payload.as_slice()
    );
    assert!(
        resolver
            .resolve(empty, Some(0))
            .await
            .unwrap()
            .bytes
            .is_empty()
    );
}

async fn paimon_blob_table(
    path: &std::path::Path,
    mime: &str,
    descriptors: &[Option<Vec<u8>>],
) -> paimon::Table {
    let schema = PaimonSchema::builder()
        .column("id", PaimonType::Int(IntType::new()))
        .column("payload", PaimonType::Blob(BlobType::new()))
        .primary_key(["id"])
        .option("bucket", "1")
        .option("file.format", "parquet")
        .option("blob-descriptor-field", "payload")
        .option("fields.payload.df.media.mime", mime)
        .build()
        .unwrap();
    let table = new_paimon(path, schema).await;
    let batch = RecordBatch::try_new(
        Arc::new(Schema::new(vec![
            Field::new("id", DataType::Int32, false),
            Field::new("payload", DataType::LargeBinary, true),
        ])),
        vec![
            Arc::new(Int32Array::from_iter_values(
                0..i32::try_from(descriptors.len()).unwrap(),
            )),
            Arc::new(LargeBinaryArray::from_iter(
                descriptors.iter().map(Option::as_deref),
            )),
        ],
    )
    .unwrap();
    Box::pin(write_paimon(&table, &batch)).await;
    table
}

#[tokio::test]
async fn paimon_descriptors_scan_with_payload_offline() {
    let directory = tempfile::tempdir().unwrap();
    let payload = directory.path().join("payload.bin");
    let offline = directory.path().join("payload.offline");
    std::fs::write(&payload, b"0123456789").unwrap();
    let uri = Url::from_file_path(&payload).unwrap().to_string();
    let zero_uri = Url::from_file_path(directory.path().join("missing.bin"))
        .unwrap()
        .to_string();
    let table = paimon_blob_table(
        directory.path(),
        MIME,
        &[
            Some(PaimonBlob::new(uri.clone(), 2, 3).serialize()),
            Some(PaimonBlob::new(uri.clone(), 7, -1).serialize()),
            Some(PaimonBlob::new(zero_uri, 999, 0).serialize()),
            None,
        ],
    )
    .await;
    std::fs::rename(&payload, &offline).unwrap();
    let table = PaimonReadOnlyTable::open(table, None, &LakeTableOptions::default())
        .await
        .unwrap();
    let refs = read_references(&scan(&table, Some(&[1]), &[], None).await);
    assert_eq!(refs.len(), 4);
    assert!(refs[3].is_none());
    assert!(
        refs.iter()
            .take(3)
            .all(|media| media.as_ref().unwrap().mime.as_deref() == Some(MIME))
    );
    assert_eq!(refs[1].as_ref().unwrap().byte_len, None);
    assert_eq!(refs[2].as_ref().unwrap().byte_len, Some(0));

    std::fs::rename(&offline, &payload).unwrap();
    let resolver = LakeMediaResolver::default();
    let mut bounded = refs[0].clone().unwrap();
    let resolved = resolver.resolve(&bounded, Some(3)).await.unwrap();
    assert_eq!(resolved.bytes, b"234".as_slice());
    bounded.etag = Some(resolved.etag.unwrap());
    assert_eq!(
        resolver.resolve(&bounded, Some(3)).await.unwrap().bytes,
        b"234".as_slice()
    );
    bounded.etag = Some("different".into());
    assert!(
        resolver
            .resolve(&bounded, Some(3))
            .await
            .unwrap_err()
            .to_string()
            .contains("ETag")
    );
    assert!(
        resolver
            .resolve(&bounded, Some(2))
            .await
            .unwrap_err()
            .to_string()
            .contains("max_bytes")
    );

    let eof = refs[1].as_ref().unwrap();
    assert_eq!(
        resolver.resolve(eof, Some(3)).await.unwrap().bytes,
        b"789".as_slice()
    );
    assert!(
        resolver
            .resolve(eof, Some(2))
            .await
            .unwrap_err()
            .to_string()
            .contains("max_bytes")
    );
    let zero = refs[2].as_ref().unwrap();
    assert!(
        resolver
            .resolve(zero, Some(0))
            .await
            .unwrap()
            .bytes
            .is_empty()
    );
}

#[tokio::test]
async fn resolver_enforces_range_limits_and_etags_through_registry() {
    let registry = Arc::new(DefaultObjectStoreRegistry::new());
    let store = Arc::new(InMemory::new());
    let path = Path::from("object");
    let result = store
        .put(&path, Bytes::from_static(b"abcdef").into())
        .await
        .unwrap();
    let reads = Reads::new();
    registry.register_store(
        &Url::parse("memory://fixture").unwrap(),
        reads.wrap("fixture", store),
    );
    let resolver = LakeMediaResolver::new(registry);
    let mut media = MediaRef {
        uri: "memory://fixture/object".into(),
        byte_offset: None,
        byte_len: None,
        etag: result.e_tag,
        mime: Some(MIME.into()),
    };
    assert!(
        resolver
            .resolve(&media, Some(5))
            .await
            .unwrap_err()
            .to_string()
            .contains("max_bytes")
    );
    let calls = reads.take();
    assert_eq!(calls.len(), 1);
    assert!(calls[0].range.is_none(), "{calls:?}");

    media.byte_offset = Some(1);
    media.byte_len = Some(3);
    assert!(
        resolver
            .resolve(&media, Some(2))
            .await
            .unwrap_err()
            .to_string()
            .contains("max_bytes")
    );
    assert!(reads.take().is_empty());

    let resolved = resolver.resolve(&media, Some(3)).await.unwrap();
    assert_eq!(resolved.bytes, b"bcd".as_slice());
    assert_eq!(resolved.etag, media.etag);
    assert_eq!(resolved.mime, media.mime);
    assert_eq!(reads.take()[0].range, Some(1..4));

    media.etag = Some("stale".into());
    assert!(
        resolver
            .resolve(&media, None)
            .await
            .unwrap_err()
            .to_string()
            .contains("ETag")
    );
    assert!(reads.take().is_empty());

    media.byte_len = Some(0);
    assert!(
        resolver
            .resolve(&media, Some(0))
            .await
            .unwrap_err()
            .to_string()
            .contains("ETag mismatch")
    );
    let calls = reads.take();
    assert_eq!(calls.len(), 1);
    assert!(calls[0].range.is_none(), "{calls:?}");

    media.etag = resolved.etag;
    let empty = resolver.resolve(&media, Some(0)).await.unwrap();
    assert!(empty.bytes.is_empty());
    assert_eq!(empty.etag, media.etag);
    assert_eq!(empty.mime, media.mime);
    let calls = reads.take();
    assert_eq!(calls.len(), 1);
    assert!(calls[0].range.is_none(), "{calls:?}");

    media.etag = None;
    media.byte_offset = Some(u64::MAX);
    assert!(
        resolver
            .resolve(&media, Some(0))
            .await
            .unwrap()
            .bytes
            .is_empty()
    );
    assert!(reads.take().is_empty());

    let invalid_uri = MediaRef {
        uri: "not a URI".into(),
        ..media.clone()
    };
    assert!(matches!(
        resolver.resolve(&invalid_uri, Some(0)).await,
        Err(DataFusionError::Plan(message)) if message.contains("Invalid media URI")
    ));
    assert!(reads.take().is_empty());

    media.byte_len = Some(1);
    assert!(matches!(
        resolver.resolve(&media, None).await,
        Err(DataFusionError::Plan(_))
    ));
}

#[tokio::test]
async fn resolver_reads_registered_paimon_memory_ranges() {
    let uri = "memory:/fixture/media";
    let io = FileIO::from_path(uri).unwrap().build().unwrap();
    io.new_output(uri)
        .unwrap()
        .write(Bytes::from_static(b"abcdef"))
        .await
        .unwrap();
    let resolver = LakeMediaResolver::default()
        .with_paimon_file_io(Url::parse("memory:/fixture").unwrap(), io);
    let media = MediaRef {
        uri: uri.into(),
        byte_offset: Some(3),
        byte_len: None,
        etag: None,
        mime: None,
    };
    assert_eq!(
        resolver.resolve(&media, Some(3)).await.unwrap().bytes,
        b"def".as_slice()
    );
    assert!(
        resolver
            .resolve(&media, Some(2))
            .await
            .unwrap_err()
            .to_string()
            .contains("max_bytes")
    );
    let empty = MediaRef {
        byte_len: Some(0),
        etag: Some("expected".into()),
        ..media
    };
    assert!(
        resolver
            .resolve(&empty, Some(0))
            .await
            .unwrap_err()
            .to_string()
            .contains("Paimon FileIO cannot verify the media ETag")
    );
}

#[tokio::test]
async fn open_rejects_blank_blob_mime() {
    let directory = tempfile::tempdir().unwrap();
    let uri = directory.path().join("fixture.lance");
    let mut blob = BlobDescriptorArrayBuilder::new("payload");
    blob.push_null().unwrap();
    let (field, array) = blob.finish().unwrap().into_parts();
    let batch = RecordBatch::try_new(
        Arc::new(Schema::new(vec![field_with_mime(field, " \t")])),
        vec![array],
    )
    .unwrap();
    InsertBuilder::new(uri.to_str().unwrap())
        .with_params(&WriteParams::with_storage_version(LanceFileVersion::V2_2))
        .execute(vec![batch])
        .await
        .unwrap();
    let result = LanceReadOnlyTable::open(
        uri.to_str().unwrap(),
        None,
        &LakeTableOptions::default(),
    )
    .await;
    assert!(
        matches!(result, Err(DataFusionError::Plan(message)) if message.contains("blank MIME"))
    );

    let table = paimon_blob_table(directory.path(), " \t", &[None]).await;
    let result =
        PaimonReadOnlyTable::open(table, None, &LakeTableOptions::default()).await;
    assert!(
        matches!(result, Err(DataFusionError::Plan(message)) if message.contains("blank MIME"))
    );
}
