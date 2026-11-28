# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: See the NOTICE file distributed with this work for additional information regarding copyright ownership
# SPDX-FileContributor: Licensed to the Apache Software Foundation (ASF) under one or more contributor license agreements

from dataclasses import FrozenInstanceError
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import Mock, patch

from stage_sdk import StageResult, run_stage


FINGERPRINT = "f" * 64
MANIFEST = "STAGE_MANIFEST.json"
FIRST = "attempts/split-0000000000000000-attempt-0001"
SECOND = "attempts/split-0000000000000010-attempt-0002"


def manifest():
    return {
        "schema": "df-stage-manifest-v1",
        "plan_fingerprint": FINGERPRINT,
        "output_rows": 3,
        "dlq_rows": 1,
        "dlq_by_code": {"TIMEOUT": 1},
        "input_tokens": 45,
        "output_tokens": 18,
        "llm_requests": 4,
        "llm_retries": 1,
        "llm_usage_by_provider": {
            "example": {"input_tokens": 45, "output_tokens": 18, "requests": 4},
        },
        "splits": [
            {
                "schema": "df-checkpoint-v1", "plan_fingerprint": FINGERPRINT,
                "split_fingerprint": "0" * 64, "split_id": 0, "winning_attempt_id": 1,
                "data_rel_path": f"{FIRST}/data.parquet", "output_rows": 1,
                "dlq_rel_path": f"{FIRST}/dlq.parquet", "dlq_rows": 1,
            },
            {
                "schema": "df-checkpoint-v1", "plan_fingerprint": FINGERPRINT,
                "split_fingerprint": "1" * 64, "split_id": 16, "winning_attempt_id": 2,
                "data_rel_path": f"{SECOND}/data.parquet", "output_rows": 2,
                "dlq_rel_path": None, "dlq_rows": 0,
            },
        ],
    }


class S3Error(Exception):
    def __init__(self, code):
        super().__init__("transport error containing credential-value")
        self.response = {"Error": {"Code": code}}


class StageResultTests(unittest.TestCase):
    def test_read_only_metrics_and_local_paths(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            stage = root / "stages" / FINGERPRINT
            stage.mkdir(parents=True)
            (stage / MANIFEST).write_text(json.dumps(manifest()), encoding="utf-8")
            (root / "stages" / ("b" * 64)).mkdir()
            dirty = stage / "attempts/split-0000000000000000-attempt-0000"
            dirty.mkdir(parents=True)
            for name in ("data.parquet", "dlq.parquet"):
                (dirty / name).touch()

            result = StageResult.load(root)
            self.assertEqual(result, StageResult.load(stage))
            self.assertEqual(result, StageResult.from_manifest((stage / MANIFEST).as_uri()))
            self.assertEqual(result.plan_fingerprint, FINGERPRINT)
            self.assertEqual((result.output_rows, result.dlq_rows), (3, 1))
            self.assertEqual(result.dlq_by_code, {"TIMEOUT": 1})
            self.assertEqual((result.input_tokens, result.output_tokens), (45, 18))
            self.assertEqual((result.llm_requests, result.llm_retries), (4, 1))
            self.assertEqual(dict(result.llm_usage_by_provider["example"]), {
                "input_tokens": 45, "output_tokens": 18, "requests": 4,
            })
            self.assertEqual(result.data_uris(), [
                str(stage / FIRST / "data.parquet"), str(stage / SECOND / "data.parquet"),
            ])
            self.assertEqual(result.dlq_uris(), [str(stage / FIRST / "dlq.parquet")])
            result.data_uris().clear()
            self.assertEqual(len(result.data_uris()), 2)
            with self.assertRaises(FrozenInstanceError):
                result.output_rows = 99
            with self.assertRaises(TypeError):
                result.dlq_by_code["TIMEOUT"] = 99
            with self.assertRaises(TypeError):
                result.llm_usage_by_provider["example"]["input_tokens"] = 99

    def test_older_manifests_and_absent_dlq(self):
        doc = manifest()
        for name in ("input_tokens", "output_tokens", "llm_requests", "llm_retries", "llm_usage_by_provider"):
            del doc[name]
        doc.update(dlq_rows=0, dlq_by_code={})
        doc["splits"][0].update(dlq_rows=0, dlq_rel_path=None)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / MANIFEST
            path.write_text(json.dumps(doc), encoding="utf-8")
            result = StageResult.from_manifest(path)
        self.assertEqual(result.dlq_uris(), [])
        self.assertEqual((result.input_tokens, result.output_tokens, result.llm_requests, result.llm_retries), (0, 0, 0, 0))
        self.assertEqual(result.llm_usage_by_provider, {})

    def test_bad_paths_and_ambiguous_job_root(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with self.assertRaisesRegex(FileNotFoundError, "No STAGE_MANIFEST"):
                StageResult.load(root)
            stage = root / "stages" / FINGERPRINT
            stage.mkdir(parents=True)
            path = stage / MANIFEST
            for relative in ("../data.parquet", "/data.parquet", f"{SECOND}/data.parquet"):
                with self.subTest(relative=relative):
                    doc = manifest()
                    doc["splits"][0]["data_rel_path"] = relative
                    path.write_text(json.dumps(doc), encoding="utf-8")
                    with self.assertRaisesRegex(ValueError, "Invalid manifest data.parquet path"):
                        StageResult.from_manifest(path)
            path.write_text(json.dumps(manifest()), encoding="utf-8")
            second = root / "stages" / ("a" * 64)
            second.mkdir()
            (second / MANIFEST).write_bytes(path.read_bytes())
            with self.assertRaisesRegex(ValueError, "Multiple finalized stages"):
                StageResult.load(root)
            with self.assertRaisesRegex(ValueError, "does not match the manifest plan fingerprint"):
                StageResult.load(second)
            self.assertEqual(StageResult.load(stage).output_rows, 3)

    def test_s3_manifest_resolution_is_metadata_only(self):
        client = Mock()
        prefix = "job output/stages/"
        stage = prefix + FINGERPRINT
        skipped = prefix + ("b" * 64)
        client.get_paginator.return_value.paginate.return_value = [{"CommonPrefixes": [
            {"Prefix": skipped + "/"}, {"Prefix": stage + "/"},
        ]}]
        client.head_object.side_effect = [S3Error("404"), {}]
        client.get_object.side_effect = [
            S3Error("NoSuchKey"), {"Body": io.BytesIO(json.dumps(manifest()).encode())},
            {"Body": io.BytesIO(json.dumps(manifest()).encode())},
        ]
        with patch("stage_sdk._s3_client", return_value=client):
            result = StageResult.load("s3://results/job%20output/")
            self.assertEqual(result, StageResult.from_manifest(f"s3://results/{stage}/{MANIFEST}"))
        client.get_paginator.assert_called_once_with("list_objects_v2")
        client.get_paginator.return_value.paginate.assert_called_once_with(
            Bucket="results", Prefix=prefix, Delimiter="/",
        )
        self.assertEqual([call.kwargs["Key"] for call in client.get_object.call_args_list], [
            f"job output/{MANIFEST}", f"{stage}/{MANIFEST}", f"{stage}/{MANIFEST}",
        ])
        s3_stage = f"s3://results/job%20output/stages/{FINGERPRINT}"
        self.assertEqual(result.data_uris(), [f"{s3_stage}/{FIRST}/data.parquet", f"{s3_stage}/{SECOND}/data.parquet"])
        self.assertEqual(result.dlq_uris(), [f"{s3_stage}/{FIRST}/dlq.parquet"])

    def test_s3_errors_omit_transport_details(self):
        client = Mock()
        client.get_object.side_effect = S3Error("AccessDenied")
        with patch("stage_sdk._s3_client", return_value=client):
            with self.assertRaisesRegex(OSError, "check AWS settings and permissions") as error:
                StageResult.from_manifest(f"s3://results/stage/{MANIFEST}")
        self.assertNotIn("credential-value", str(error.exception))
        with self.assertRaisesRegex(ValueError, "without embedded credentials") as error:
            StageResult.from_manifest("s3://user:credential-value@results/stage/manifest")
        self.assertNotIn("credential-value", str(error.exception))


class RunStageTests(unittest.TestCase):
    def test_runs_through_driver_cli_with_temporary_job(self):
        with (tempfile.TemporaryDirectory() as directory,
              patch("stage_sdk.subprocess.run") as invoke):
            invoke.return_value = subprocess.CompletedProcess([], 0, json.dumps(manifest()), "")
            result = run_stage(
                sql="SELECT id FROM items", tables={"items": Path("input.parquet")},
                row_key="id", output_root=directory, worker=Path("/bin/dist-worker"),
                ray_address="auto", concurrency=2, timeout=5, max_attempts=2,
            )
            (args,) = invoke.call_args.args
            self.assertEqual(args[:2], [sys.executable, "-B"])
            self.assertEqual(Path(args[2]).name, "ray_driver.py")
            job_path = Path(args[args.index("--job") + 1])
            self.assertFalse(job_path.parent.exists())
            root = Path(directory).resolve()
            self.assertEqual(set(args[3:]) - {"--job", str(job_path)}, {
                "--worker=/bin/dist-worker", "--sql=SELECT id FROM items", "--table=items=input.parquet",
                f"--output-root={root}", "--row-key=id", "--ray-address=auto", "--concurrency=2",
                "--timeout=5", "--max-attempts=2",
            })
            self.assertEqual(result.data_uris()[0], str(root / "stages" / FINGERPRINT / FIRST / "data.parquet"))

    def test_driver_failure_preserves_diagnostics_and_redacts_credentials(self):
        with (tempfile.TemporaryDirectory() as directory,
              patch("stage_sdk.subprocess.run") as invoke,
              patch.dict("os.environ", {"AWS_SECRET_ACCESS_KEY": "credential-value"})):
            invoke.return_value = subprocess.CompletedProcess([], 1, "", "Access denied credential-value")
            with self.assertRaisesRegex(RuntimeError, "Access denied") as error:
                run_stage(sql="SELECT id FROM t", tables=["t=input.parquet"], output_root=directory)
            self.assertNotIn("credential-value", str(error.exception))


if __name__ == "__main__":
    unittest.main()
