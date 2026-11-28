# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: See the NOTICE file distributed with this work for additional information regarding copyright ownership
# SPDX-FileContributor: Licensed to the Apache Software Foundation (ASF) under one or more contributor license agreements

import ast
from contextlib import redirect_stderr, redirect_stdout
from io import StringIO
import json
import os
import subprocess
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, call, patch

from ray_driver import (
    RayExecutor, _download_worker, _inherited_payload, _invoke, _read_payload,
    _read_secrets_file, _s3_worker_location, drive, main,
)


class FakeWorker:
    def __init__(self, split_ids):
        self.split_ids = split_ids
        self.winners = {}
        self.plans = 0

    def plan(self):
        self.plans += 1
        return {
            "pending_splits": [
                {"split_id": split_id, "attempt_id": 0}
                for split_id in self.split_ids
                if split_id not in self.winners
            ]
        }

    def commit(self, receipt):
        artifact = receipt["artifact"]
        split_id = artifact["split_id"]
        assert split_id not in self.winners
        self.winners[split_id] = artifact["attempt_id"]

    def finalize(self):
        assert self.winners.keys() == set(self.split_ids)
        return sorted(self.winners.items())


class ScriptedExecutor:
    def __init__(self, events, failures=(), receipts=None):
        self.events = iter(events)
        self.failures = set(failures)
        self.receipts = receipts or {}
        self.now = 0.0
        self.submitted = []
        self.cancelled = []
        self.read = []

    def submit(self, split_id, attempt_id):
        ticket = (split_id, attempt_id)
        self.submitted.append(ticket)
        return ticket

    def wait(self, active, timeout):
        assert active
        assert timeout >= 0
        elapsed, ready = next(self.events)
        self.now += elapsed
        return ready

    def result(self, ticket):
        self.read.append(ticket)
        if ticket in self.failures:
            raise RuntimeError("worker lost")
        split_id, attempt_id = self.receipts.get(ticket, ticket)
        return {"artifact": {"split_id": split_id, "attempt_id": attempt_id}}

    def cancel(self, ticket):
        self.cancelled.append(ticket)


def run(worker, executor, **kwargs):
    return drive(worker, lambda: executor, clock=lambda: executor.now, **kwargs)


class DriverTests(unittest.TestCase):
    def test_timeout_retries_and_discards_late_result(self):
        worker = FakeWorker([0, 1])
        executor = ScriptedExecutor([
            (1, [(1, 0)]),
            (4, []),
            (1, [(0, 0), (0, 1)]),
        ])
        self.assertEqual(run(worker, executor, timeout=5, concurrency=2), [(0, 1), (1, 0)])
        self.assertEqual(executor.submitted, [(0, 0), (1, 0), (0, 1)])
        self.assertEqual(executor.cancelled, [(0, 0)])
        self.assertEqual(executor.read, [(1, 0), (0, 1)])

    def test_worker_failure_and_mismatched_receipt_retry(self):
        worker = FakeWorker([5])
        executor = ScriptedExecutor(
            [(0, [(5, 0)]), (0, [(5, 1)]), (0, [(5, 2)])],
            failures=[(5, 0)],
            receipts={(5, 1): (5, 0)},
        )
        self.assertEqual(run(worker, executor, timeout=5), [(5, 2)])

    def test_interruption_cancels_active_work_and_resumes_from_plan(self):
        worker = FakeWorker([5, 8])
        first = ScriptedExecutor([(0, [(5, 0)])])

        def interrupt(active, timeout):
            if worker.winners:
                raise KeyboardInterrupt
            return ScriptedExecutor.wait(first, active, timeout)

        first.wait = interrupt
        with self.assertRaises(KeyboardInterrupt):
            run(worker, first, timeout=5, concurrency=2)
        self.assertEqual(worker.winners, {5: 0})
        self.assertEqual(first.cancelled, [(8, 0)])

        second = ScriptedExecutor([(0, [(8, 0)])])
        self.assertEqual(run(worker, second, timeout=5), [(5, 0), (8, 0)])
        self.assertEqual(second.submitted, [(8, 0)])
        self.assertEqual(worker.plans, 2)

    def test_attempt_exhaustion_does_not_finalize(self):
        worker = FakeWorker([0])
        executor = ScriptedExecutor([(5, [])])
        with self.assertRaisesRegex(RuntimeError, "split 0 exhausted"):
            run(worker, executor, timeout=5, max_attempts=1)
        self.assertEqual(worker.winners, {})


class RayWorkerTests(unittest.TestCase):
    def test_worker_location(self):
        for local in ("dist-worker", "/shared/dist-worker", Path("bin/dist-worker")):
            self.assertIsNone(_s3_worker_location(local))
        self.assertEqual(
            _s3_worker_location("s3://artifacts/linux/dist%20worker"),
            ("artifacts", "linux/dist worker"),
        )
        for invalid in ("s3:///key", "s3://bucket", "s3://bucket/",
                        "s3://user:pass@bucket/key", "s3://bucket/key?x=1"):
            with self.subTest(invalid=invalid):
                with self.assertRaisesRegex(ValueError, "S3 object URI"):
                    _s3_worker_location(invalid)

    def test_stdin_payload_uses_only_object_ref_and_child_environment(self):
        public = {
            "AWS_ENDPOINT": "http://host:80",
            "DF_DIST_LLM_BACKEND": "openai",
            "DF_DIST_LLM_PROVIDERS": json.dumps({
                "qwen-openai": {
                    "base_url": "https://qwen.invalid/v1",
                    "api_key_env": "DF_LLM_KEY_QWEN_OPENAI",
                },
                "gpt-gateway": {
                    "base_url": "https://embed.invalid/v1",
                    "api_key_env": "CUSTOM_EMBED_API_KEY",
                    "ca_cert_env": "CUSTOM_EMBED_CA",
                },
            }),
        }
        secrets = {
            "AWS_ACCESS_KEY_ID": "fixture-id",
            "AWS_SECRET_ACCESS_KEY": "fixture-secret",
            "AWS_SESSION_TOKEN": "fixture-session",
            "OPENAI_API_KEY": "fixture-api-key",
            "DF_LLM_KEY_QWEN_OPENAI": "fixture-qwen",
            "CUSTOM_EMBED_API_KEY": "fixture-embed",
            "CUSTOM_EMBED_CA": "-----BEGIN CERTIFICATE-----\nfixture-ca\n",
        }
        payload = {**public, **secrets}
        job_bytes = b'{"pending_splits":[{"split_id":4,"attempt_id":2}]}'
        receipt = {"artifact": {"split_id": 4, "attempt_id": 2}}
        references = {}

        class Ref:
            pass

        def put(value):
            ref = Ref()
            references[ref] = value
            return ref

        def remote(task):
            # Ray resolves top-level ObjectRefs and preserves refs inside containers.
            def submit(*args):
                return put(task(
                    *(references[arg] if isinstance(arg, Ref) else arg for arg in args),
                ))
            task_submit.side_effect = submit
            return SimpleNamespace(remote=task_submit)

        task_submit = Mock()
        ray = Mock()
        ray.is_initialized.return_value = False
        ray.put.side_effect = put
        ray.get.side_effect = references.__getitem__
        ray.wait.side_effect = lambda handles, **_: (handles, [])
        ray.remote.return_value = remote

        def download(bucket, key, target, env):
            self.assertEqual((bucket, key), ("artifacts", "worker"))
            self.assertEqual(env["AWS_ENDPOINT"], "http://host:80")
            self.assertEqual(env["AWS_SECRET_ACCESS_KEY"], secrets["AWS_SECRET_ACCESS_KEY"])
            self.assertNotIn("CUSTOM_EMBED_API_KEY", env)
            self.assertEqual(ray.put.call_args_list[0], call(payload))
            target.write_bytes(b"worker binary")

        final_manifest = {"output_rows": 1}

        def run_worker(args, *, env, **kwargs):
            self.assertEqual({key: env[key] for key in payload}, payload)
            self.assertTrue(payload.keys().isdisjoint(os.environ))
            self.assertTrue(all(value not in " ".join(args) for value in secrets.values()))
            if args[1] == "plan":
                Path(args[args.index("--out") + 1]).write_bytes(job_bytes)
            else:
                self.assertEqual(Path(args[3]).read_bytes(), job_bytes)
            if args[1] == "run-split":
                path = Path(args[0])
                self.assertEqual(path.read_bytes(), b"worker binary")
                self.assertEqual(path.stat().st_mode & 0o777, 0o700)
                self.assertEqual(args[4:], ["--split-id", "4", "--attempt-id", "2"])
                self.assertEqual(kwargs["timeout"], 12)
                return subprocess.CompletedProcess(args, 0, json.dumps(receipt), "")
            if args[1] == "commit-split":
                self.assertEqual(json.loads(kwargs["input"]), receipt)
            if args[1] == "finalize-stage":
                return subprocess.CompletedProcess(args, 0, json.dumps(final_manifest), "")
            return subprocess.CompletedProcess(args, 0, "", "")

        stdin = Mock(wraps=StringIO(json.dumps(payload)))
        stdout, stderr = StringIO(), StringIO()
        with tempfile.TemporaryDirectory() as directory:
            job_path = Path(directory) / "job.json"
            argv = ["ray_driver.py", "--worker", "s3://artifacts/worker", "--secrets-stdin",
                    "--ray-address", "auto", "--timeout", "12", "--job", str(job_path),
                    "--sql", "SELECT id FROM t", "--table", "t=input", "--output-root", "output"]
            with (
                patch.dict(sys.modules, {"ray": ray}), patch.dict(os.environ, clear=True),
                patch.object(sys, "argv", argv), patch.object(sys, "stdin", stdin),
                patch("ray_driver._download_worker", side_effect=download),
                patch("ray_driver.subprocess.run", side_effect=run_worker) as run,
                redirect_stdout(stdout), redirect_stderr(stderr),
            ):
                main()
            self.assertFalse(Path(run.call_args_list[1].args[0][0]).exists())
            self.assertEqual(job_path.read_bytes(), job_bytes)
            self.assertEqual(sorted(p.name for p in Path(directory).iterdir()), ["job.json"])

        ray.init.assert_called_once_with()
        stdin.read.assert_called_once_with()
        self.assertEqual(ray.put.call_args_list, [call(payload), call(job_bytes)])
        worker, job_ref, (payload_ref,), split, attempt, timeout = task_submit.call_args.args
        self.assertEqual((worker, split, attempt, timeout), ("s3://artifacts/worker", 4, 2, 12))
        self.assertEqual(references[job_ref], job_bytes)
        self.assertEqual(references[payload_ref], payload)
        self.assertEqual(ray.get.call_args_list[0], call(payload_ref))
        ray.remote.assert_called_once_with(num_cpus=1, max_retries=0)
        self.assertEqual(json.loads(stdout.getvalue()), final_manifest)
        self.assertEqual(stderr.getvalue(), "")

    def test_ray_connection_modes_and_version_mismatch(self):
        ray = Mock()
        ray.is_initialized.return_value = False
        worker = SimpleNamespace(job_path=Mock(), executable="dist-worker", payload={})
        with patch.dict(sys.modules, {"ray": ray}):
            for address in ("http://dashboard:8265", "https://dashboard"):
                with self.assertRaisesRegex(ValueError, "submit a Ray Job"):
                    RayExecutor(worker, 1, address)
            ray.init.assert_not_called()
            ray.put.assert_not_called()
            RayExecutor(worker, 1, "auto")
            RayExecutor(worker, 1)
            RayExecutor(worker, 1, "ray://cluster:10001")
            self.assertEqual(ray.init.call_args_list, [
                call(), call(), call(address="ray://cluster:10001"),
            ])
            ray.init.side_effect = RuntimeError("Version mismatch: Python 3.12 vs 3.10")
            with self.assertRaisesRegex(ValueError, "Run the driver on the head"):
                RayExecutor(worker, 1, "ray://cluster:10001")
            self.assertEqual(ray.put.call_count, 3)

    def test_secrets_file_format_permissions_and_symlinks(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "secrets.json"
            path.touch(mode=0o600)
            path.write_text('{"OPENAI_API_KEY":"fixture-secret"}')
            self.assertEqual(_read_secrets_file(path), {"OPENAI_API_KEY": "fixture-secret"})
            path.chmod(0o640)
            with self.assertRaisesRegex(ValueError, "permissions 0600"):
                _read_secrets_file(path)
            path.chmod(0o600)
            link = Path(directory) / "link.json"
            link.symlink_to(path)
            with self.assertRaisesRegex(ValueError, "Could not open"):
                _read_secrets_file(link)
            path.write_text('{"OPENAI_API_KEY":12}')
            with self.assertRaisesRegex(ValueError, "JSON object of environment names and string values"):
                _read_secrets_file(path)

    def test_payload_rejects_malformed_or_multiple_values_without_echo(self):
        for value in (
            "fixture-secret",
            '{"KEY":"fixture-secret"} {}',
            '{"INVALID=KEY":"fixture-secret"}',
            '{"KEY":null}',
        ):
            with self.assertRaises(ValueError) as invalid:
                _read_payload(StringIO(value))
            self.assertNotIn("fixture-secret", str(invalid.exception))

    def test_payload_inputs_are_mutually_exclusive(self):
        argv = ["ray_driver.py", "--secrets-stdin", "--secrets-file", "/unused",
                "--sql", "SELECT 1", "--table", "t=input", "--job", "/unused-job",
                "--output-root", "output"]
        with patch.object(sys, "argv", argv), redirect_stderr(StringIO()) as stderr:
            with self.assertRaises(SystemExit):
                main()
        self.assertIn("not allowed with argument --secrets-stdin", stderr.getvalue())

    def test_default_payload_inherits_only_worker_settings(self):
        env = {
            "AWS_ENDPOINT": "http://s3.invalid",
            "OPENAI_API_KEY": "fixture-key",
            "DF_DIST_LLM_PROVIDERS": '{"provider":{"api_key_env":"NAMED_KEY"}}',
            "NAMED_KEY": "fixture-named-key",
        }
        with patch.dict(os.environ, dict(env, UNRELATED_SECRET="ignored"), clear=True):
            self.assertEqual(_inherited_payload(), env)

    def test_driver_parses_with_python_310_grammar(self):
        source = Path(__file__).with_name("ray_driver.py").read_text(encoding="utf-8")
        ast.parse(source, feature_version=(3, 10))

    def test_logged_worker_error_redacts_credentials(self):
        result = subprocess.CompletedProcess([], 1, "", "request failed: fixture-secret")
        with patch("ray_driver.subprocess.run", return_value=result):
            with self.assertRaisesRegex(RuntimeError, r"request failed: \[redacted\]"):
                _invoke("worker", ["run-split"], payload={"TEST_KEY": "fixture-secret"})

    def test_boto3_download_uses_custom_endpoint_region_and_path_style(self):
        boto3 = Mock()
        config = Mock()
        modules = {"boto3": boto3, "botocore.config": SimpleNamespace(Config=config)}
        env = {
            "AWS_ENDPOINT": "http://host:80",
            "AWS_REGION": "us-west-2",
            "AWS_VIRTUAL_HOSTED_STYLE_REQUEST": "false",
        }
        env.update(AWS_ACCESS_KEY_ID="fixture-id", AWS_SECRET_ACCESS_KEY="fixture-secret")
        with patch.dict(sys.modules, modules):
            _download_worker("artifacts", "worker", Path("/tmp/worker"), env)
        config.assert_called_once_with(s3={"addressing_style": "path"})
        boto3.client.assert_called_once_with(
            "s3", endpoint_url="http://host:80", region_name="us-west-2",
            config=config.return_value, aws_access_key_id="fixture-id",
            aws_secret_access_key="fixture-secret",
        )
        boto3.client.return_value.download_file.assert_called_once_with(
            "artifacts", "worker", "/tmp/worker",
        )

    def test_aws_cli_fallback_and_missing_downloader(self):
        env = {"AWS_ENDPOINT": "http://host:80", "AWS_REGION": "us-west-2"}
        with patch.dict(sys.modules, {"boto3": None}), patch.dict(
            os.environ, env, clear=True,
        ), patch("ray_driver.shutil.which", return_value="/bin/aws") as which, patch(
            "ray_driver.subprocess.run",
        ) as run_cli:
            _download_worker("artifacts", "worker", Path("/tmp/worker"), env)
            self.assertEqual(run_cli.call_args.args[0], [
                "/bin/aws", "s3", "cp", "s3://artifacts/worker", "/tmp/worker",
                "--only-show-errors", "--endpoint-url", "http://host:80",
                "--region", "us-west-2",
            ])
            self.assertEqual(run_cli.call_args.kwargs["env"], env)
            which.return_value = None
            with self.assertRaisesRegex(RuntimeError, "requires boto3 or the AWS CLI"):
                _download_worker("artifacts", "worker", Path("/tmp/worker"), env)
