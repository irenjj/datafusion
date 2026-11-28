// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: See the NOTICE file distributed with this work for additional information regarding copyright ownership
// SPDX-FileContributor: Licensed to the Apache Software Foundation (ASF) under one or more contributor license agreements

use std::fs::{self, File};
use std::path::Path;
use std::sync::Arc;

use datafusion::arrow::array::{
    AsArray, Int64Array, StringArray, StructArray, UInt64Array,
};
use datafusion::arrow::compute::{
    concat_batches, filter_record_batch, is_null, sort_to_indices, take_record_batch,
};
use datafusion::arrow::datatypes::{DataType, Field, Int64Type, Schema};
use datafusion::arrow::record_batch::RecordBatch;
use datafusion::execution::SessionStateBuilder;
use datafusion::execution::runtime_env::RuntimeEnv;
use datafusion::parquet::arrow::ArrowWriter;
use datafusion::parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;
use datafusion::prelude::{ParquetReadOptions, SessionConfig, SessionContext};
use datafusion_ai::llm::{LlmProviderRegistry, MockLlmBackend};
use datafusion_ai::{MediaRefType, ObjectStoreMediaResolver};
use serde_json::{Value, json};
use tempfile::tempdir;
use url::Url;

use super::{Result, successful, worker};

const IMAGE: &[u8] = b"P6\n1 1\n255\n\xff\x00\x00";
const PREFIX: &[u8] = b"header:";
const SQL: &str = r#"
    SELECT id,
        llm(prompt, 'chat-model', '{"on_error":"envelope"}') AS chat,
        vlm(prompt, image_uri, 'vision-model', '{"on_error":"envelope"}') AS vision,
        vlm(prompt, media, 'range-model', '{"on_error":"envelope","max_media_bytes":14}') AS ranged_vision,
        embed(prompt, 'embed-model', '{"dimension":4,"on_error":"envelope"}') AS embedding
    FROM t
"#;

fn write_inputs(input: &Path, assets: &Path) -> Result<()> {
    fs::create_dir(input)?;
    fs::create_dir(assets)?;
    let image = assets.join("pixel.ppm");
    let packed = assets.join("packed.bin");
    fs::write(&image, IMAGE)?;
    fs::write(&packed, [PREFIX, IMAGE, b":trailer"].concat())?;
    let uri = Url::from_file_path(image).unwrap();
    let missing = Url::from_file_path(assets.join("missing.ppm")).unwrap();
    let packed_uri = Url::from_file_path(packed).unwrap();

    let media = MediaRefType::field("media", false)?;
    let schema = Arc::new(Schema::new(vec![
        Field::new("id", DataType::Int64, false),
        Field::new("prompt", DataType::Utf8, false),
        Field::new("image_uri", DataType::Utf8, false),
        media,
    ]));
    let DataType::Struct(fields) = schema.field(3).data_type() else {
        unreachable!()
    };
    let references = StructArray::try_new(
        fields.clone(),
        vec![
            Arc::new(StringArray::from(vec![packed_uri.as_str(); 2])),
            Arc::new(UInt64Array::from(vec![PREFIX.len() as u64; 2])),
            Arc::new(UInt64Array::from(vec![IMAGE.len() as u64; 2])),
            Arc::new(StringArray::from(vec![None::<&str>; 2])),
            Arc::new(StringArray::from(vec!["image/x-portable-pixmap"; 2])),
        ],
        None,
    )?;
    for (file, ids, prompts, uris) in [
        ("a", vec![30, 10], vec!["three", "one"], [&uri, &uri]),
        (
            "b",
            vec![40, 20],
            vec!["missing image", "second"],
            [&missing, &uri],
        ),
    ] {
        let batch = RecordBatch::try_new(
            Arc::clone(&schema),
            vec![
                Arc::new(Int64Array::from(ids)),
                Arc::new(StringArray::from(prompts)),
                Arc::new(StringArray::from(uris.map(Url::as_str).to_vec())),
                Arc::new(references.clone()),
            ],
        )?;
        let mut writer = ArrowWriter::try_new(
            File::create(input.join(format!("{file}.parquet")))?,
            Arc::clone(&schema),
            None,
        )?;
        writer.write(&batch)?;
        writer.close()?;
    }
    Ok(())
}

fn read_parquet(path: &Path) -> Result<RecordBatch> {
    let builder = ParquetRecordBatchReaderBuilder::try_new(File::open(path)?)?;
    let schema = Arc::clone(builder.schema());
    let batches = builder
        .build()?
        .collect::<std::result::Result<Vec<_>, _>>()?;
    Ok(concat_batches(&schema, &batches)?)
}

fn sort_by_id(batch: &RecordBatch) -> Result<RecordBatch> {
    let indices = sort_to_indices(batch.column(0).as_ref(), None, None)?;
    Ok(take_record_batch(batch, &indices)?)
}

async fn single_node(input: &Path) -> Result<RecordBatch> {
    let runtime = Arc::new(RuntimeEnv::default());
    let config = SessionConfig::new()
        .with_target_partitions(1)
        .with_extension(Arc::new(ObjectStoreMediaResolver::new(Arc::clone(
            &runtime.object_store_registry,
        ))));
    let registry = LlmProviderRegistry::default();
    registry.register("default", Arc::new(MockLlmBackend))?;
    let ctx = SessionContext::new_with_state(
        SessionStateBuilder::new_with_default_features()
            .with_config(config)
            .with_llm_provider_registry(registry)
            .with_runtime_env(runtime)
            .build(),
    );
    ctx.register_parquet("t", input.to_str().unwrap(), ParquetReadOptions::default())
        .await?;
    let batches = ctx.sql(SQL).await?.collect().await?;
    sort_by_id(&concat_batches(&batches[0].schema(), &batches)?)
}

#[tokio::test]
async fn multimodal_parquet_outputs_match_single_node_and_resume() -> Result<()> {
    let dir = tempdir()?;
    let input = dir.path().join("input");
    let assets = dir.path().join("assets");
    let root = dir.path().join("output");
    write_inputs(&input, &assets)?;
    let table = format!("t={}", input.display());
    let command = |mode| {
        let mut cmd = worker();
        cmd.args([mode, "--sql", SQL, "--table", &table, "--row-key", "id"])
            .arg("--output-root")
            .arg(&root)
            .env("DF_DIST_LLM_BACKEND", "mock")
            .env_remove("DF_DIST_LLM_PROVIDER");
        cmd
    };
    let run = || -> Result<Value> {
        Ok(serde_json::from_slice(&successful(
            command("run-local").output()?,
        ))?)
    };
    let job = dir.path().join("job.json");
    let plan = || -> Result<Value> {
        successful(command("plan").arg("--out").arg(&job).output()?);
        Ok(serde_json::from_slice(&fs::read(&job)?)?)
    };
    let initial = plan()?;
    let manifest = run()?;
    assert_eq!(
        initial["stage"]["plan_fingerprint"],
        manifest["plan_fingerprint"]
    );
    let stage = root
        .join("stages")
        .join(manifest["plan_fingerprint"].as_str().unwrap());
    let splits = manifest["splits"].as_array().unwrap();
    assert_eq!(splits.len(), 2);
    assert_eq!(manifest["output_rows"], 3);
    assert_eq!(manifest["dlq_rows"], 1);
    assert_eq!(manifest["dlq_by_code"], json!({"PERMANENT_HTTP": 1}));

    let mut data = Vec::new();
    let mut dlq = Vec::new();
    for (id, split) in splits.iter().enumerate() {
        assert_eq!(split["split_id"], id);
        assert_eq!(split["winning_attempt_id"], 0);
        data.push(read_parquet(
            &stage.join(split["data_rel_path"].as_str().unwrap()),
        )?);
        if let Some(path) = split["dlq_rel_path"].as_str() {
            dlq.push(read_parquet(&stage.join(path))?);
        }
    }
    assert_eq!(dlq.len(), 1);
    let dlq = &dlq[0];
    assert_eq!(dlq.num_rows(), 1);
    assert_eq!(
        dlq.column_by_name("row_key")
            .unwrap()
            .as_primitive::<Int64Type>()
            .values(),
        &[40]
    );
    assert_eq!(
        dlq.column_by_name("failed_column")
            .unwrap()
            .as_string::<i32>()
            .value(0),
        "vision"
    );
    let code = dlq
        .column_by_name("error_code")
        .unwrap()
        .as_string::<i32>()
        .value(0);
    assert_eq!(code, "PERMANENT_HTTP");
    assert!(
        !dlq.column_by_name("retryable")
            .unwrap()
            .as_boolean()
            .value(0)
    );

    let data = sort_by_id(&concat_batches(&data[0].schema(), &data)?)?;
    assert_eq!(
        data.column(0).as_primitive::<Int64Type>().values(),
        &[10, 20, 30]
    );
    let single = single_node(&input).await?;
    let errors = single
        .column(2)
        .as_struct()
        .column_by_name("error_code")
        .unwrap();
    assert_eq!(errors.as_string::<i32>().value(3), code);
    let expected = filter_record_batch(&single, &is_null(errors.as_ref())?)?;
    assert_eq!(data.column(0), expected.column(0));
    for (index, name) in ["chat", "vision", "ranged_vision", "embedding"]
        .into_iter()
        .enumerate()
    {
        let index = index + 1;
        let schema = data.schema();
        let field = schema.field(index);
        assert_eq!(field.name(), name);
        assert!(field.extension_type_name().is_none());
        if name == "embedding" {
            let DataType::FixedSizeList(item, 4) = field.data_type() else {
                panic!("unexpected embedding type: {}", field.data_type());
            };
            assert_eq!(item.data_type(), &DataType::Float32);
        } else {
            assert_eq!(field.data_type(), &DataType::Utf8);
        }
        assert_eq!(
            data.column(index),
            expected
                .column(index)
                .as_struct()
                .column_by_name("value")
                .unwrap(),
            "{name}",
        );
    }

    let manifest_path = stage.join("STAGE_MANIFEST.json");
    fs::remove_file(&manifest_path)?;
    fs::remove_dir_all(&assets)?;
    let planned = plan()?;
    assert!(
        planned["stage"]["plan_bytes"] == initial["stage"]["plan_bytes"],
        "replanning changed the wire bytes",
    );
    assert_eq!(
        planned["stage"]["plan_fingerprint"],
        manifest["plan_fingerprint"]
    );
    assert_eq!(planned["pending_splits"], json!([]));
    let inventory = planned["splits"].as_array().unwrap();
    assert_eq!(inventory.len(), splits.len());
    for (split, committed) in inventory.iter().zip(splits) {
        assert_eq!(split["split_id"], committed["split_id"]);
        assert_eq!(split["fingerprint"], committed["split_fingerprint"]);
    }
    assert_eq!(run()?, manifest);
    assert_eq!(
        serde_json::from_slice::<Value>(&fs::read(manifest_path)?)?,
        manifest
    );
    Ok(())
}

#[cfg(feature = "ai-http")]
#[test]
fn openai_ca_cert_is_loaded_at_runtime_and_kept_out_of_the_job() -> Result<()> {
    let ca_pem = "-----BEGIN CERTIFICATE-----
MIIBjDCCATOgAwIBAgIUOm49mAs56opfNl4QYglVAoRyKBswCgYIKoZIzj0EAwIw
GzEZMBcGA1UEAwwQZGlzdC13b3JrZXItdGVzdDAgFw0yNjExMjgwNzIzMDhaGA8y
MTI2MTEwNDA3MjMwOFowGzEZMBcGA1UEAwwQZGlzdC13b3JrZXItdGVzdDBZMBMG
ByqGSM49AgEGCCqGSM49AwEHA0IABBm3DMGlXG0F/Kaf6NWXu4WgZGTLb9nEgpYM
crwvLRS2KB5OCnQ7ZlhBAqp4eBbx2U7koXewlT/4chh7R2j4v2OjUzBRMB0GA1Ud
DgQWBBTyBBm2mLM3Q5Esc+NxXuYsXAP0mzAfBgNVHSMEGDAWgBTyBBm2mLM3Q5Es
c+NxXuYsXAP0mzAPBgNVHRMBAf8EBTADAQH/MAoGCCqGSM49BAMCA0cAMEQCIHDU
mD8OCGniZ4pXZoqYUiPzAUZAj3vr9v747S/WZ7zCAiBMfTAtWWfgFsxxi1BReJMn
Nd3AeiouMgQrDh1/NJXdEw==
-----END CERTIFICATE-----
";
    let dir = tempdir()?;
    let input = dir.path().join("input");
    let job = dir.path().join("job.json");
    let ca_file = dir.path().join("ca.pem");
    super::input_files(&input)?;
    let command = || {
        let mut cmd = worker();
        cmd.args([
            "plan",
            "--sql",
            "SELECT llm(CAST(id AS VARCHAR), 'chat-model', '{}') FROM t",
            "--table",
            &format!("t={}", input.display()),
            "--out",
        ])
        .arg(&job)
        .env("DF_DIST_LLM_BACKEND", "openai")
        .env("OPENAI_BASE_URL", "https://127.0.0.1:1/v1")
        .env_remove("OPENAI_CA_CERT_FILE")
        .env_remove("OPENAI_API_KEY");
        cmd
    };
    successful(command().output()?);
    let without_ca: Value = serde_json::from_slice(&fs::read(&job)?)?;
    fs::write(&ca_file, ca_pem)?;
    successful(command().env("OPENAI_CA_CERT_FILE", &ca_file).output()?);
    let with_ca: Value = serde_json::from_slice(&fs::read(&job)?)?;
    assert_eq!(with_ca, without_ca);

    fs::write(&ca_file, "invalid CA")?;
    let output = command().env("OPENAI_CA_CERT_FILE", &ca_file).output()?;
    assert!(!output.status.success());
    let error = String::from_utf8_lossy(&output.stderr);
    assert!(error.contains("Invalid llm HTTP ca_cert_pem"), "{error}");
    Ok(())
}
