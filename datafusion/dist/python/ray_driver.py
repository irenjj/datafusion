# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: See the NOTICE file distributed with this work for additional information regarding copyright ownership
# SPDX-FileContributor: Licensed to the Apache Software Foundation (ASF) under one or more contributor license agreements

"""Run the stateless dist-worker commands with Ray or an injected executor."""

import argparse
from collections import deque
from contextlib import suppress
from dataclasses import dataclass
import json
import math
from pathlib import Path
import subprocess
import tempfile
import time


@dataclass(frozen=True)
class _Attempt:
    split_id: int
    attempt_id: int
    tries: int
    deadline: float


def drive(worker, executor_factory, *, timeout=60.0, concurrency=4,
          max_attempts=3, clock=time.monotonic):
    """Plan, execute, commit, and finalize using a worker command adapter.

    An executor provides submit(split_id, attempt_id), wait(handles, timeout),
    result(handle), and cancel(handle). Only the driver can publish a receipt.
    """
    if not math.isfinite(timeout) or timeout <= 0 or min(concurrency, max_attempts) < 1:
        raise ValueError("timeout, concurrency, and max_attempts must be positive")

    job = worker.plan()
    pending = deque(
        (split["split_id"], split["attempt_id"], 1)
        for split in job["pending_splits"]
    )
    if not pending:
        return worker.finalize()

    executor = executor_factory()
    active = {}

    def retry(attempt, error):
        if attempt.tries >= max_attempts or attempt.attempt_id == 0xFFFFFFFF:
            raise RuntimeError(
                f"split {attempt.split_id} exhausted attempts at {attempt.attempt_id}"
            ) from error
        pending.append((attempt.split_id, attempt.attempt_id + 1, attempt.tries + 1))

    try:
        while pending or active:
            while pending and len(active) < concurrency:
                split_id, attempt_id, tries = pending.popleft()
                handle = executor.submit(split_id, attempt_id)
                active[handle] = _Attempt(split_id, attempt_id, tries, clock() + timeout)

            deadline = min(attempt.deadline for attempt in active.values())
            ready = executor.wait(tuple(active), max(0, deadline - clock()))
            now = clock()
            for handle in ready:
                attempt = active.get(handle)
                if attempt is None or attempt.deadline <= now:
                    continue
                del active[handle]
                try:
                    receipt = executor.result(handle)
                    artifact = receipt["artifact"]
                    if (artifact["split_id"], artifact["attempt_id"]) != (
                        attempt.split_id, attempt.attempt_id
                    ):
                        raise ValueError("receipt does not match the active attempt")
                except Exception as error:
                    retry(attempt, error)
                else:
                    worker.commit(receipt)

            for handle, attempt in tuple(active.items()):
                if attempt.deadline <= now:
                    del active[handle]
                    _cancel(executor, handle)
                    retry(attempt, TimeoutError(f"attempt {attempt.attempt_id} expired"))
    finally:
        for handle in active:
            _cancel(executor, handle)

    return worker.finalize()


def _cancel(executor, handle):
    with suppress(Exception):
        executor.cancel(handle)


def _invoke(executable, args, *, receipt=None, timeout=None):
    result = subprocess.run(
        [executable, *args],
        input=None if receipt is None else json.dumps(receipt),
        capture_output=True,
        text=True,
        timeout=timeout,
        check=False,
    )
    if result.returncode:
        raise RuntimeError(
            f"dist-worker {args[0]} exited {result.returncode}: {result.stderr.strip()}"
        )
    return json.loads(result.stdout) if result.stdout.strip() else None


class WorkerCli:
    """Adapter for the worker's serde JSON and Parquet artifact protocol."""

    def __init__(self, executable, sql, tables, job_path, output_root, row_key=None):
        self.executable = executable
        self.sql = sql
        self.tables = tables
        self.job_path = Path(job_path)
        self.output_root = output_root
        self.row_key = row_key

    def plan(self):
        args = ["plan", "--sql", self.sql, "--out", str(self.job_path),
                "--output-root", self.output_root]
        for table in self.tables:
            args.extend(("--table", table))
        if self.row_key is not None:
            args.extend(("--row-key", self.row_key))
        _invoke(self.executable, args)
        return json.loads(self.job_path.read_bytes())

    def commit(self, receipt):
        _invoke(self.executable, ["commit-split", "--job", str(self.job_path)], receipt=receipt)

    def finalize(self):
        return _invoke(self.executable, ["finalize-stage", "--job", str(self.job_path)])


class RayExecutor:
    """Run one child process per Ray task and return its small JSON receipt."""

    def __init__(self, worker, timeout, address=None):
        import ray

        if not ray.is_initialized():
            ray.init(address=address)
        self._ray = ray
        self._job = ray.put(worker.job_path.read_bytes())
        self._worker = worker.executable
        self._timeout = timeout

        @ray.remote(num_cpus=1, max_retries=0)
        def run_split(executable, job, split_id, attempt_id, timeout):
            with tempfile.TemporaryDirectory(prefix="dist-worker-") as directory:
                path = Path(directory) / "stage-job.json"
                path.write_bytes(job)
                return _invoke(
                    executable,
                    ["run-split", "--job", str(path), "--split-id", str(split_id),
                     "--attempt-id", str(attempt_id)],
                    timeout=timeout,
                )

        self._run_split = run_split

    def submit(self, split_id, attempt_id):
        return self._run_split.remote(
            self._worker, self._job, split_id, attempt_id, self._timeout
        )

    def wait(self, handles, timeout):
        ready, pending = self._ray.wait(list(handles), timeout=timeout)
        if ready and pending:
            also_ready, _ = self._ray.wait(pending, num_returns=len(pending), timeout=0)
            ready.extend(also_ready)
        return ready

    def result(self, handle):
        return self._ray.get(handle)

    def cancel(self, handle):
        self._ray.cancel(handle)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worker", default="dist-worker")
    parser.add_argument("--sql", required=True)
    parser.add_argument("--table", action="append", required=True)
    parser.add_argument("--job", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--row-key")
    parser.add_argument("--ray-address")
    parser.add_argument("--timeout", type=float, default=60.0)
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument("--max-attempts", type=int, default=3)
    args = parser.parse_args()
    worker = WorkerCli(args.worker, args.sql, args.table, args.job,
                       args.output_root, args.row_key)
    print(json.dumps(drive(
        worker,
        lambda: RayExecutor(worker, args.timeout, args.ray_address),
        timeout=args.timeout,
        concurrency=args.concurrency,
        max_attempts=args.max_attempts,
    )))


if __name__ == "__main__":
    main()
