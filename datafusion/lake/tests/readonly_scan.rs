// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: See the NOTICE file distributed with this work for additional information regarding copyright ownership
// SPDX-FileContributor: Licensed to the Apache Software Foundation (ASF) under one or more contributor license agreements

#[path = "readonly_scan/fixtures.rs"]
mod fixtures;
#[path = "readonly_scan/index.rs"]
mod index;
#[path = "readonly_scan/media.rs"]
mod media;
#[path = "readonly_scan/search.rs"]
mod search;
#[path = "readonly_scan/store.rs"]
mod store;

use std::sync::Arc;

use datafusion_catalog::default_table_source::provider_as_source;
use datafusion_catalog::{Session, TableProvider};
use datafusion_common::DataFusionError;
use datafusion_datasource::PartitionedFile;
use datafusion_datasource::file_groups::FileGroup;
use datafusion_datasource::file_scan_config::FileScanConfigBuilder;
use datafusion_datasource::source::DataSourceExec;
use datafusion_expr::{
    Expr, LogicalPlan, LogicalPlanBuilder, TableProviderFilterPushDown, TableType, col,
    lit,
};
use datafusion_lake::{
    InclusiveRowRange, LakeSplitExt, LakeSplitPayload, LakeTableOptions,
    LanceReadOnlyTable, PaimonReadOnlyTable,
};
use datafusion_optimizer::{Optimizer, OptimizerContext};
use datafusion_physical_plan::collect;
use lance_table_runtime::write::WriteMode;
use lance_testing::datafusion::TestSession;

use fixtures::{
    file_config, ids, new_paimon, scalar_paimon_schema, scalar_rows, scan, write_lance,
    write_paimon,
};

#[tokio::test]
async fn lance_pins_latest_and_explicit_versions() {
    let directory = tempfile::tempdir().unwrap();
    let uri = directory.path().to_str().unwrap();
    let first = write_lance(uri, scalar_rows(0), WriteMode::Create).await;
    let options = LakeTableOptions::default();
    let pinned = LanceReadOnlyTable::open(uri, None, &options).await.unwrap();
    let second = write_lance(uri, scalar_rows(6), WriteMode::Append).await;
    let latest = LanceReadOnlyTable::open(uri, None, &options).await.unwrap();
    let historical =
        LanceReadOnlyTable::open(uri, Some(first.manifest().version), &options)
            .await
            .unwrap();
    assert_eq!(pinned.dataset_version(), first.manifest().version);
    assert_eq!(latest.dataset_version(), second.manifest().version);
    assert_eq!(historical.dataset_version(), pinned.dataset_version());
    assert_eq!(
        ids(&scan(&pinned, Some(&[0]), &[], None).await),
        (0..6).collect::<Vec<_>>()
    );
    assert_eq!(
        ids(&scan(&latest, Some(&[0]), &[], None).await),
        (0..12).collect::<Vec<_>>()
    );
    assert_eq!(
        ids(&scan(&historical, Some(&[0]), &[], None).await),
        (0..6).collect::<Vec<_>>()
    );
    assert!(matches!(
        LanceReadOnlyTable::open(uri, Some(99), &options).await,
        Err(DataFusionError::Plan(_))
    ));
    let bad = directory.path().join("missing");
    assert!(matches!(
        LanceReadOnlyTable::open(bad.to_str().unwrap(), None, &options).await,
        Err(DataFusionError::Plan(_))
    ));
}

#[tokio::test]
async fn paimon_pins_latest_and_explicit_snapshots() {
    let directory = tempfile::tempdir().unwrap();
    let table =
        new_paimon(directory.path(), scalar_paimon_schema().build().unwrap()).await;
    let options = LakeTableOptions::default();
    assert!(matches!(
        PaimonReadOnlyTable::open(table.clone(), None, &options).await,
        Err(DataFusionError::Plan(_))
    ));
    write_paimon(&table, &scalar_rows(0)).await;
    let pinned = PaimonReadOnlyTable::open(table.clone(), None, &options)
        .await
        .unwrap();
    write_paimon(&table, &scalar_rows(6)).await;
    let latest = PaimonReadOnlyTable::open(table.clone(), None, &options)
        .await
        .unwrap();
    let historical =
        PaimonReadOnlyTable::open(table.clone(), Some(pinned.snapshot_id()), &options)
            .await
            .unwrap();
    assert!(latest.snapshot_id() > pinned.snapshot_id());
    assert_eq!(historical.snapshot_id(), pinned.snapshot_id());
    assert_eq!(
        ids(&scan(&pinned, Some(&[0]), &[], None).await),
        (0..6).collect::<Vec<_>>()
    );
    assert_eq!(
        ids(&scan(&latest, Some(&[0]), &[], None).await),
        (0..12).collect::<Vec<_>>()
    );
    assert_eq!(
        ids(&scan(&historical, Some(&[0]), &[], None).await),
        (0..6).collect::<Vec<_>>()
    );
    assert!(matches!(
        PaimonReadOnlyTable::open(table, Some(99), &options).await,
        Err(DataFusionError::Plan(_))
    ));
}

#[tokio::test]
async fn paimon_open_without_required_indices_defers_manifest_reads() {
    let directory = tempfile::tempdir().unwrap();
    let table =
        new_paimon(directory.path(), scalar_paimon_schema().build().unwrap()).await;
    write_paimon(&table, &scalar_rows(0)).await;
    let manager = table.snapshot_manager();
    let snapshot = manager.get_latest_snapshot().await.unwrap().unwrap();
    let manifest = manager.manifest_path(snapshot.delta_manifest_list());
    let offline = directory.path().join("offline-manifest");
    std::fs::rename(&manifest, &offline).unwrap();

    let table = PaimonReadOnlyTable::open(table, None, &LakeTableOptions::default())
        .await
        .unwrap();
    assert_eq!(table.snapshot_id(), snapshot.id());
    let session = TestSession::default();
    let error = table
        .scan(&session, Some(&[0]), &[], None)
        .await
        .unwrap_err();
    assert!(error.to_string().contains(snapshot.delta_manifest_list()));

    std::fs::rename(offline, manifest).unwrap();
    assert_eq!(
        ids(&scan(&table, Some(&[0]), &[], None).await),
        [0, 1, 2, 3, 4, 5]
    );
}

async fn check_projection_filter_and_limit(table: Arc<dyn TableProvider>) {
    assert_eq!(table.table_type(), TableType::Base);
    let rows = scan(table.as_ref(), Some(&[2, 0]), &[], None).await;
    assert_eq!(ids(&rows), (0..6).collect::<Vec<_>>());
    for batch in rows {
        assert_eq!(batch.schema().field(0).name(), "part");
        assert_eq!(batch.schema().field(1).name(), "id");
        assert_eq!(batch.num_columns(), 2);
    }
    let repeated_columns = scan(table.as_ref(), Some(&[2, 0, 2]), &[], None).await;
    assert_eq!(ids(&repeated_columns), (0..6).collect::<Vec<_>>());
    for batch in repeated_columns {
        assert_eq!(batch.column(0), batch.column(2));
    }
    let empty = scan(table.as_ref(), Some(&[]), &[], None).await;
    assert!(empty.iter().all(|batch| batch.num_columns() == 0));
    assert_eq!(
        empty
            .iter()
            .map(arrow::array::RecordBatch::num_rows)
            .sum::<usize>(),
        6
    );
    let no_rows = scan(table.as_ref(), None, &[], Some(0)).await;
    assert!(no_rows.is_empty());
    assert_eq!(
        ids(&scan(table.as_ref(), Some(&[0]), &[], Some(4)).await).len(),
        4
    );

    let id = col("id");
    let label = col("label");
    let cases: Vec<(Expr, &[i32])> = vec![
        (id.clone().eq(lit(4)), &[4]),
        (lit(4).not_eq(id.clone()), &[0, 1, 2, 3, 5]),
        (id.clone().lt(lit(2)), &[0, 1]),
        (id.clone().lt_eq(lit(2)), &[0, 1, 2]),
        (lit(3).lt(id.clone()), &[4, 5]),
        (id.clone().gt_eq(lit(3)), &[3, 4, 5]),
        (id.clone().in_list(vec![lit(3), lit(5)], false), &[3, 5]),
        (label.clone().is_null(), &[1, 4]),
        (label.clone().is_not_null(), &[0, 2, 3, 5]),
        (id.clone().gt(lit(0)).and(id.clone().lt(lit(4))), &[1, 2, 3]),
        (id.clone().eq(lit(0)).or(id.clone().eq(lit(5))), &[0, 5]),
        (Expr::Not(Box::new(label.clone().eq(lit("a")))), &[2, 3]),
        (
            Expr::Not(Box::new(
                label.clone().eq(lit("a")).or(id.clone().eq(lit(2))),
            )),
            &[3],
        ),
        (
            label.in_list(
                vec![lit("b"), lit(datafusion_common::ScalarValue::Null)],
                true,
            ),
            &[],
        ),
        (col("part").eq(lit("p1")), &[3, 4, 5]),
    ];
    for (predicate, expected) in cases {
        assert_eq!(
            table.supports_filters_pushdown(&[&predicate]).unwrap(),
            vec![TableProviderFilterPushDown::Exact],
        );
        let matching = scan(
            table.as_ref(),
            Some(&[0]),
            std::slice::from_ref(&predicate),
            None,
        )
        .await;
        assert_eq!(ids(&matching), expected, "{predicate}");
    }
    let limited = scan(
        table.as_ref(),
        Some(&[]),
        &[col("id").gt_eq(lit(4))],
        Some(1),
    )
    .await;
    assert!(limited.iter().all(|batch| batch.num_columns() == 0));
    assert_eq!(
        limited
            .iter()
            .map(arrow::array::RecordBatch::num_rows)
            .sum::<usize>(),
        1
    );

    let unsupported = (col("id") + lit(1)).eq(lit(3));
    let session = TestSession::default();
    assert!(matches!(
        table
            .scan(
                &session,
                Some(&[0]),
                std::slice::from_ref(&unsupported),
                None
            )
            .await,
        Err(DataFusionError::NotImplemented(_))
    ));
    assert!(matches!(
        table.supports_filters_pushdown(&[&unsupported]),
        Err(DataFusionError::NotImplemented(_))
    ));

    let source = provider_as_source(table);
    let logical = LogicalPlanBuilder::scan("lake", Arc::clone(&source), None)
        .unwrap()
        .filter(unsupported)
        .unwrap()
        .build()
        .unwrap();
    let err = Optimizer::new()
        .optimize(logical, &OptimizerContext::new(), |_, _| {})
        .unwrap_err();
    assert!(
        err.to_string()
            .contains("Lake cannot push down scalar filter"),
        "{err}"
    );

    let logical = LogicalPlanBuilder::scan("lake", source, None)
        .unwrap()
        .filter(col("id").eq(lit(2)))
        .unwrap()
        .build()
        .unwrap();
    let optimized = Optimizer::new()
        .optimize(logical, &OptimizerContext::new(), |_, _| {})
        .unwrap();
    let LogicalPlan::TableScan(planned) = optimized else {
        panic!("expected an exact filter in the table scan: {optimized}");
    };
    assert_eq!(planned.filters.len(), 1);
}

#[tokio::test]
async fn lance_projection_filter_limit_and_fail_closed() {
    let directory = tempfile::tempdir().unwrap();
    let uri = directory.path().to_str().unwrap();
    write_lance(uri, scalar_rows(0), WriteMode::Create).await;
    let table = LanceReadOnlyTable::open(uri, None, &LakeTableOptions::default())
        .await
        .unwrap();
    assert!(matches!(
        table.validate_prefilter(&[(col("id") * lit(2)).eq(lit(2))]),
        Err(DataFusionError::NotImplemented(_))
    ));
    check_projection_filter_and_limit(Arc::new(table)).await;
}

#[tokio::test]
async fn paimon_projection_filter_limit_and_fail_closed() {
    let directory = tempfile::tempdir().unwrap();
    let table =
        new_paimon(directory.path(), scalar_paimon_schema().build().unwrap()).await;
    write_paimon(&table, &scalar_rows(0)).await;
    let table = PaimonReadOnlyTable::open(table, None, &LakeTableOptions::default())
        .await
        .unwrap();
    assert!(matches!(
        table.validate_prefilter(&[col("vector").is_null()]),
        Err(DataFusionError::NotImplemented(_))
    ));
    check_projection_filter_and_limit(Arc::new(table)).await;
}

async fn check_split_replay(table: &dyn TableProvider) {
    let session = TestSession::default();
    let first = table.scan(&session, Some(&[0]), &[], None).await.unwrap();
    let second = table.scan(&session, Some(&[0]), &[], None).await.unwrap();
    let config = file_config(first.as_ref());
    let files: Vec<_> = config
        .file_groups
        .iter()
        .flat_map(FileGroup::files)
        .collect();
    let repeated: Vec<_> = file_config(second.as_ref())
        .file_groups
        .iter()
        .flat_map(FileGroup::files)
        .collect();
    assert_eq!(files.len(), 2);
    assert!(!config.file_source().supports_repartitioning());
    let keys: Vec<_> = files
        .iter()
        .map(
            |file| match &file.extension::<LakeSplitExt>().unwrap().payload {
                LakeSplitPayload::Lance { fragment_id, .. } => {
                    (String::new(), i64::from(*fragment_id), String::new())
                }
                LakeSplitPayload::Paimon {
                    partition_key_b64,
                    bucket,
                    data_split_json,
                    ..
                } => {
                    let split: paimon::DataSplit =
                        serde_json::from_str(data_split_json).unwrap();
                    (
                        partition_key_b64.clone(),
                        i64::from(*bucket),
                        split.data_files()[0].file_name.clone(),
                    )
                }
            },
        )
        .collect();
    assert!(keys.is_sorted());
    let mut all_rows = vec![];
    for (file, copy) in files.into_iter().zip(repeated) {
        assert_eq!(file.range, None);
        let extension = file.extension::<LakeSplitExt>().unwrap();
        assert_eq!(Some(extension), copy.extension::<LakeSplitExt>());
        let (version, payload) =
            LakeSplitExt::try_encode_from_extensions(&file.extensions)
                .unwrap()
                .unwrap();
        let mut transported =
            PartitionedFile::new(extension.split_id.clone(), file.object_meta.size);
        LakeSplitExt::try_decode_into_extensions(
            version,
            &payload,
            &mut transported.extensions,
        )
        .unwrap();
        let plan = DataSourceExec::from_data_source(
            FileScanConfigBuilder::from(config.clone())
                .with_file_groups(vec![FileGroup::new(vec![transported])])
                .build(),
        );
        let rows = ids(&collect(plan, session.task_ctx()).await.unwrap());
        assert_eq!(rows.len(), 3);
        assert!(rows == [0, 1, 2] || rows == [3, 4, 5]);
        all_rows.extend(rows);
    }
    all_rows.sort_unstable();
    assert_eq!(all_rows, (0..6).collect::<Vec<_>>());

    let missing = DataSourceExec::from_data_source(
        FileScanConfigBuilder::from(config.clone())
            .with_file_groups(vec![FileGroup::new(vec![PartitionedFile::new(
                "missing", 0,
            )])])
            .build(),
    );
    let err = collect(missing, session.task_ctx()).await.unwrap_err();
    assert!(err.to_string().contains("Missing df.lake.split"), "{err}");
}

#[tokio::test]
async fn lance_fragment_split_replay_and_range_fence() {
    let directory = tempfile::tempdir().unwrap();
    let uri = directory.path().to_str().unwrap();
    write_lance(uri, scalar_rows(0), WriteMode::Create).await;
    let table = LanceReadOnlyTable::open(uri, None, &LakeTableOptions::default())
        .await
        .unwrap();
    check_split_replay(&table).await;

    let session = TestSession::default();
    let plan = table.scan(&session, Some(&[0]), &[], None).await.unwrap();
    let config = file_config(plan.as_ref());
    let file = config.file_groups[0].files()[0].clone();
    let mut ext = file.extension::<LakeSplitExt>().unwrap().clone();
    let LakeSplitPayload::Lance { row_ranges, .. } = &mut ext.payload else {
        unreachable!()
    };
    *row_ranges = Some(vec![InclusiveRowRange { from: 1, to: 1 }]);
    let replay = |file| {
        DataSourceExec::from_data_source(
            FileScanConfigBuilder::from(config.clone())
                .with_file_groups(vec![FileGroup::new(vec![file])])
                .build(),
        )
    };
    assert_eq!(
        ids(&collect(
            replay(file.clone().with_extension(ext.clone())),
            session.task_ctx()
        )
        .await
        .unwrap()),
        [1]
    );

    let LakeSplitPayload::Lance {
        deletion_file_rev, ..
    } = &mut ext.payload
    else {
        unreachable!()
    };
    *deletion_file_rev = Some(0);
    let err = collect(replay(file.with_extension(ext)), session.task_ctx())
        .await
        .unwrap_err();
    assert!(err.to_string().contains("deletion revision"), "{err}");
}

#[tokio::test]
async fn paimon_partition_split_replay_and_snapshot_fence() {
    let directory = tempfile::tempdir().unwrap();
    let table =
        new_paimon(directory.path(), scalar_paimon_schema().build().unwrap()).await;
    write_paimon(&table, &scalar_rows(0)).await;
    let table = PaimonReadOnlyTable::open(table, None, &LakeTableOptions::default())
        .await
        .unwrap();
    check_split_replay(&table).await;
    let session = TestSession::default();
    let plan = table.scan(&session, Some(&[0]), &[], None).await.unwrap();
    let config = file_config(plan.as_ref());
    let file = config.file_groups[0].files()[0].clone();
    let mut ext = file.extension::<LakeSplitExt>().unwrap().clone();
    let LakeSplitPayload::Paimon { snapshot_id, .. } = &mut ext.payload else {
        unreachable!()
    };
    *snapshot_id += 1;
    let replay = DataSourceExec::from_data_source(
        FileScanConfigBuilder::from(config.clone())
            .with_file_groups(vec![FileGroup::new(vec![file.with_extension(ext)])])
            .build(),
    );
    let err = collect(replay, session.task_ctx()).await.unwrap_err();
    assert!(
        err.to_string()
            .contains("does not match the pinned table scan"),
        "{err}"
    );
}

#[tokio::test]
async fn lance_fragment_ranges_respect_pinned_deletions() {
    use lance_core::utils::deletion::DeletionVector;
    use lance_io::object_store::ObjectStore;
    use lance_table::feature_flags::apply_feature_flags;
    use lance_table::format::Manifest;
    use lance_table::io::commit::write_manifest_file_to_path;
    use lance_table::io::deletion::write_deletion_file;

    let directory = tempfile::tempdir().unwrap();
    let uri = directory.path().to_str().unwrap();
    let native = write_lance(uri, scalar_rows(0), WriteMode::Create).await;
    let (store, base) = ObjectStore::from_uri(uri).await.unwrap();
    let mut fragments = native.manifest().fragments.as_ref().clone();
    let version = native.manifest().version;
    let first = &mut fragments[0];
    first.deletion_file = write_deletion_file(
        &base,
        first.id,
        version,
        &DeletionVector::from_iter([1]),
        &store,
        0,
    )
    .await
    .unwrap();
    let mut manifest = Manifest::new_from_previous(
        native.manifest(),
        native.schema().clone(),
        Arc::new(fragments),
    );
    apply_feature_flags(&mut manifest, false, false).unwrap();
    let path = native
        .manifest_location()
        .naming_scheme
        .manifest_path(&base, manifest.version);
    write_manifest_file_to_path(&store, &mut manifest, None, &path, None)
        .await
        .unwrap();

    let table = LanceReadOnlyTable::open(
        uri,
        Some(manifest.version),
        &LakeTableOptions::default(),
    )
    .await
    .unwrap();
    assert_eq!(
        ids(&scan(&table, Some(&[0]), &[], None).await),
        [0, 2, 3, 4, 5]
    );

    let session = TestSession::default();
    let plan = table.scan(&session, Some(&[0]), &[], None).await.unwrap();
    let config = file_config(plan.as_ref());
    let file = config.file_groups[0].files()[0].clone();
    let mut extension = file.extension::<LakeSplitExt>().unwrap().clone();
    let LakeSplitPayload::Lance {
        row_ranges,
        deletion_file_rev,
        ..
    } = &mut extension.payload
    else {
        unreachable!()
    };
    assert_eq!(*deletion_file_rev, Some(version));
    *row_ranges = Some(vec![InclusiveRowRange { from: 1, to: 2 }]);
    let plan = DataSourceExec::from_data_source(
        FileScanConfigBuilder::from(config.clone())
            .with_file_groups(vec![FileGroup::new(vec![file.with_extension(extension)])])
            .build(),
    );
    assert_eq!(ids(&collect(plan, session.task_ctx()).await.unwrap()), [2]);
}
