# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: See the NOTICE file distributed with this work for additional information regarding copyright ownership
# SPDX-FileContributor: Licensed to the Apache Software Foundation (ASF) under one or more contributor license agreements

import asyncio
from contextlib import ExitStack
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from ray_driver import WorkerCli, _actor_job, _redact, drive
from test_ray_driver import FakeWorker
from worker_pool import DistWorkerActor, RayActorExecutor, ServeCoordinator


def receipt(split_id, attempt_id):
    return {"artifact": {"split_id": split_id, "attempt_id": attempt_id}}


class FakeDaemon:
    def __init__(self, *, hang=False):
        self.stdin = self.stdout = self
        self.hang = hang
        self.blocked = asyncio.Event()
        self.requests = []
        self.killed = False
        self.closed = False
        self.reaped = False

    def write(self, data):
        self.requests.append(json.loads(data))

    async def drain(self):
        pass

    async def read(self, _size):
        request = self.requests[-1]
        command = request["command"]
        reply = {"ok": True, "result": None}
        if command == "run_split":
            if self.hang:
                self.blocked.set()
                await asyncio.Future()
            if request["split_id"] == 99:
                reply = {"ok": False, "error": {"message": "bad fixture-credential"}}
            else:
                reply["result"] = receipt(request["split_id"], request["attempt_id"])
        elif command == "finalize_stage":
            reply["result"] = {"output_rows": 2}
        return json.dumps(reply).encode() + b"\n"

    def kill(self):
        self.killed = True

    def close(self):
        self.closed = True

    async def wait(self):
        self.reaped = True

    async def communicate(self):
        self.close()
        await self.wait()


class ActorTests(unittest.IsolatedAsyncioTestCase):
    async def test_revocation_reaps_daemon_and_reuses_download(self):
        old, replacement = FakeDaemon(hang=True), FakeDaemon()
        resolve = Mock(return_value="/somewhere/dist-worker")
        job = {"schema": "df-stage-job-v1"}
        payload = {"OPENAI_API_KEY": "fixture-credential"}
        with patch.dict(os.environ, clear=True), patch(
            "worker_pool.asyncio.create_subprocess_exec", side_effect=[old, replacement],
        ) as spawn:
            actor = DistWorkerActor(
                "s3://artifacts/worker", "job", job, payload, 60, resolve, _redact,
            )
            directory = Path(resolve.call_args.args[1])
            try:
                attempt = asyncio.create_task(actor.run_split(0, 7, 0))
                await old.blocked.wait()
                await actor.revoke(0)
                self.assertTrue(old.killed and old.reaped)
                with self.assertRaises(asyncio.CancelledError):
                    await attempt
                with self.assertRaisesRegex(RuntimeError, "revoked"):
                    await actor.run_split(0, 7, 0)
                self.assertEqual(await actor.run_split(1, 7, 1), receipt(7, 1))
                with self.assertRaisesRegex(RuntimeError, r"bad \[redacted\]"):
                    await actor.run_split(2, 99, 0)
                self.assertEqual(await actor.run_split(3, 8, 0), receipt(8, 0))
            finally:
                await actor.close()
            self.assertEqual(spawn.await_count, 2)
            self.assertEqual(resolve.call_count, 1)
            self.assertFalse(directory.exists())
            for call in spawn.call_args_list:
                self.assertEqual(call.args[-2:], ("/somewhere/dist-worker", "serve"))
                self.assertEqual(call.kwargs["env"], payload)
                self.assertEqual(call.kwargs["stderr"], asyncio.subprocess.DEVNULL)
                self.assertNotIn("fixture-credential", str(call.args))
        self.assertTrue(replacement.closed and replacement.reaped and not replacement.killed)
        self.assertEqual([req["command"] for req in old.requests], ["load_job", "run_split"])
        self.assertEqual(
            [req["command"] for req in replacement.requests],
            ["load_job", "run_split", "run_split", "run_split", "shutdown"],
        )
        self.assertTrue(all(
            "fixture-credential" not in json.dumps(req)
            for req in old.requests + replacement.requests
        ))

    async def test_revocation_before_dispatch_does_not_start_daemon(self):
        with patch("worker_pool.asyncio.create_subprocess_exec") as spawn:
            actor = DistWorkerActor("worker", "job", {}, {}, 60, lambda *args: args[0], _redact)
            try:
                await actor.revoke(0)
                with self.assertRaisesRegex(RuntimeError, "revoked"):
                    await actor.run_split(0, 7, 0)
            finally:
                await actor.close()
        spawn.assert_not_awaited()


class PoolTests(unittest.TestCase):
    def test_coordinator_reuses_one_daemon(self):
        worker = Mock(spec=WorkerCli)
        worker.local_executable = "/local/worker"
        worker.payload = {}
        worker.plan.return_value = {"schema": "df-stage-job-v1"}
        child = FakeDaemon()
        with patch("worker_pool.asyncio.create_subprocess_exec", return_value=child) as spawn:
            coordinator = ServeCoordinator(worker, "job")
            try:
                coordinator.plan()
                coordinator.commit(receipt(1, 0))
                coordinator.commit(receipt(2, 0))
                self.assertEqual(coordinator.finalize(), {"output_rows": 2})
            finally:
                coordinator.close()
        spawn.assert_awaited_once()
        self.assertEqual(
            [req["command"] for req in child.requests],
            ["load_job", "commit_split", "commit_split", "finalize_stage", "shutdown"],
        )

    def test_expired_receipt_is_ignored_and_retry_prefers_healthy_actor(self):
        worker = FakeWorker([7, 8])
        payload = {"OPENAI_API_KEY": "fixture-credential"}
        worker_spec = SimpleNamespace(
            job_path=Mock(), executable="s3://artifacts/worker", payload=payload,
        )
        worker_spec.job_path.read_bytes.return_value = b"{}"
        payload_ref, job_ref = object(), object()
        ray = Mock()
        ray.put.side_effect = [payload_ref, job_ref]
        ray.get.side_effect = lambda handle, **_: (
            receipt(*handle[2:]) if isinstance(handle, tuple) else None
        )
        actors = [Mock(), Mock()]
        create_actor = Mock()
        create_actor.remote.side_effect = actors
        for index, actor in enumerate(actors):
            actor.run_split.remote.side_effect = (
                lambda *attempt, index=index: (index, *attempt)
            )
        events = iter([
            (1, [(1, 1, 8, 0)]),
            (4, [(0, 0, 7, 0)]),
            (1, [(0, 0, 7, 0), (1, 2, 7, 1)]),
        ])
        now = 0

        def wait(ray_instance, handles, timeout):
            nonlocal now
            elapsed, ready = next(events)
            self.assertIs(ray_instance, ray)
            self.assertLessEqual(elapsed, timeout)
            now += elapsed
            return ready

        executor = RayActorExecutor(ray, worker_spec, 5, create_actor, wait)
        self.assertEqual(drive(worker, lambda: executor, timeout=5, concurrency=2,
                               clock=lambda: now), [(7, 1), (8, 0)])
        self.assertEqual(create_actor.remote.call_count, 2)
        for call in create_actor.remote.call_args_list:
            self.assertIs(call.args[2], job_ref)
            self.assertEqual(call.args[3], (payload_ref,))
            self.assertNotIn("fixture-credential", str(call.args))
        actors[0].revoke.remote.assert_called_once_with(0)
        actors[0].run_split.remote.assert_called_once_with(0, 7, 0)
        self.assertEqual([call.args for call in actors[1].run_split.remote.call_args_list],
                         [(1, 8, 0), (2, 7, 1)])
        self.assertFalse(any(call.args[0] == (0, 0, 7, 0) for call in ray.get.call_args_list))
        executor.close()
        for actor in actors:
            actor.close.remote.assert_called_once()
        self.assertEqual(ray.kill.call_count, 2)

    def test_actor_setup_has_no_runtime_env(self):
        ray = Mock()
        ray.cluster_resources.return_value = {"CPU": 8}
        worker = WorkerCli("worker", "SELECT id FROM t", ["t=input"], "job.json",
                           "output", payload={"OPENAI_API_KEY": "fixture-credential"})
        args = SimpleNamespace(ray_address=None, timeout=5, concurrency=4, pool_size=2)
        with patch.dict(sys.modules, {"ray": ray}), ExitStack() as stack:
            _, _, limit = _actor_job(stack, worker, args)
            self.assertEqual(limit, 2)
        ray.put.assert_called_once_with(worker.payload)
        ray.remote.assert_called_once_with(
            num_cpus=1, max_concurrency=2, max_restarts=0, max_task_retries=0,
        )

    def test_unresponsive_actor_is_removed_after_death_is_confirmed(self):
        class ActorDiedError(RuntimeError):
            pass

        ray = Mock()
        ray.exceptions.ActorDiedError = ActorDiedError
        ray.get.side_effect = [TimeoutError, ActorDiedError]
        worker = SimpleNamespace(job_path=Mock(), executable="worker", payload={})
        worker.job_path.read_bytes.return_value = b"{}"
        create_actor = Mock()
        first, replacement = Mock(), Mock()
        create_actor.remote.side_effect = [first, replacement]
        executor = RayActorExecutor(ray, worker, 5, create_actor, Mock())
        executor.cancel(executor.submit(7, 0))
        ray.kill.assert_called_once_with(first, no_restart=True)
        self.assertEqual(first.revoke.remote.call_count, 2)
        self.assertIs(executor.submit(7, 1), replacement.run_split.remote.return_value)
        replacement.run_split.remote.assert_called_once_with(1, 7, 1)


@unittest.skipUnless(os.environ.get("DIST_WORKER_E2E"), "requires a dist-worker binary and local Ray")
class ActorPoolEndToEndTests(unittest.TestCase):
    def test_ray_pool_finalizes_and_reads_four_parquet_splits(self):
        binary = Path(os.environ["DIST_WORKER_E2E"]).resolve()
        python_dir = Path(__file__).resolve().parent
        fixture = python_dir.parents[2] / "parquet-testing/data/alltypes_plain.parquet"
        env = {
            key: value for key, value in os.environ.items()
            if not key.startswith(("AWS_", "MINIO_", "OPENAI_", "DF_DIST_LLM_"))
            and key != "RAY_ADDRESS"
        }
        launcher = """
import ray
import ray_driver

ray.init(num_cpus=2, include_dashboard=False, object_store_memory=80 * 1024**2)
try:
    ray_driver.main()
finally:
    ray.shutdown()
"""
        with tempfile.TemporaryDirectory(prefix="dist-pool-e2e-") as directory:
            root = Path(directory)
            source, output, job = root / "source", root / "out", root / "job.json"
            source.mkdir()
            for index in range(4):
                shutil.copyfile(fixture, source / f"{index}.parquet")
            secret = "fixture-only-secret"
            completed = subprocess.run(
                [sys.executable, "-B", "-c", launcher, "--worker", str(binary),
                 "--sql", "SELECT id + 1 AS id FROM t", "--table", f"t={source}",
                 "--output-root", str(output), "--job", str(job), "--pool-size", "2",
                 "--max-global-inflight", "2", "--max-splits-per-minute", "120000",
                 "--secrets-stdin", "--timeout", "45"],
                cwd=python_dir, input=json.dumps({"OPENAI_API_KEY": secret}),
                env=env, capture_output=True, text=True, timeout=180,
            )
            self.assertNotIn(secret, completed.stderr)
            self.assertNotIn(secret, completed.stdout)
            self.assertEqual(completed.returncode, 0, completed.stderr)
            self.assertNotIn(secret, job.read_text())
            manifest = json.loads(completed.stdout)
            stage = output / "stages" / manifest["plan_fingerprint"]
            self.assertEqual(manifest, json.loads((stage / "STAGE_MANIFEST.json").read_text()))
            self.assertEqual(manifest["output_rows"], 32)
            self.assertEqual(len(manifest["splits"]), 4)
            readback = root / "readback"
            readback.mkdir()
            for split in manifest["splits"]:
                self.assertEqual(split["output_rows"], 8)
                self.assertEqual(split["winning_attempt_id"], 0)
                shutil.copyfile(stage / split["data_rel_path"],
                                readback / f'{split["split_id"]}.parquet')
            output = subprocess.check_output(
                [binary, "run-local", "--sql", "SELECT id FROM t",
                 "--table", f"t={readback}", "--output-root", str(root / "replayed")],
                env=env, text=True, timeout=30,
            )
            self.assertEqual(json.loads(output)["output_rows"], 32)
            print("Ray actor pool accepted 4 winning splits and read back 32 Parquet rows")
