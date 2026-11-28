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
import os
from pathlib import Path
import shutil
import stat
import subprocess
import sys
import tempfile
import time
from urllib.parse import unquote, urlsplit


_AWS_ENV = {
    "AWS_ENDPOINT", "AWS_ENDPOINT_URL", "AWS_ENDPOINT_URL_S3",
    "AWS_ALLOW_HTTP", "AWS_VIRTUAL_HOSTED_STYLE_REQUEST",
    "AWS_REGION", "AWS_DEFAULT_REGION", "AWS_BUCKET",
    "AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY", "AWS_SESSION_TOKEN", "AWS_SECURITY_TOKEN",
    "AWS_WEB_IDENTITY_TOKEN_FILE", "AWS_CONTAINER_AUTHORIZATION_TOKEN",
    "AWS_CONTAINER_AUTHORIZATION_TOKEN_FILE",
}
_WORKER_ENV = _AWS_ENV | {
    "OPENAI_BASE_URL", "OPENAI_API_KEY", "OPENAI_CA_CERT_FILE", "OPENAI_CA_CERT_PEM",
    "DF_DIST_LLM_PROVIDERS", "DF_DIST_LLM_PROVIDER", "DF_DIST_LLM_BACKEND",
}


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


def _invoke(executable, args, *, payload, receipt=None, timeout=None):
    try:
        result = subprocess.run(
            [executable, *args],
            env={**os.environ, **payload},
            input=None if receipt is None else json.dumps(receipt),
            capture_output=True,
            text=True,
            errors="replace",
            timeout=timeout,
            check=False,
        )
    except subprocess.TimeoutExpired:
        raise TimeoutError(f"dist-worker {args[0]} timed out") from None
    if result.returncode:
        detail = result.stderr
        for secret in sorted(set(payload.values()), key=len, reverse=True):
            if secret:
                detail = detail.replace(secret, "[redacted]")
        raise RuntimeError(f"dist-worker {args[0]} exited {result.returncode}: {detail.strip()}")
    try:
        return json.loads(result.stdout) if result.stdout.strip() else None
    except ValueError:
        raise RuntimeError(f"dist-worker {args[0]} returned invalid JSON") from None


def _s3_worker_location(worker):
    worker = os.fspath(worker)
    if not worker.startswith("s3://"):
        return None
    uri = urlsplit(worker)
    key = unquote(uri.path[1:])
    if (not uri.netloc or not key or "@" in uri.netloc or ":" in uri.netloc
            or uri.query or uri.fragment):
        raise ValueError("--worker expects an S3 object URI (s3://bucket/key)")
    return uri.netloc, key


def _download_worker(bucket, key, path, env):
    endpoint = (env.get("AWS_ENDPOINT_URL_S3")
                or env.get("AWS_ENDPOINT_URL")
                or env.get("AWS_ENDPOINT"))
    region = env.get("AWS_REGION") or env.get("AWS_DEFAULT_REGION")
    try:
        import boto3
        from botocore.config import Config
    except ImportError as error:
        aws = shutil.which("aws")
        if aws is None:
            raise RuntimeError(
                "An S3 --worker requires boto3 or the AWS CLI on the coordinator "
                "and Ray workers"
            ) from error
        args = [aws, "s3", "cp", f"s3://{bucket}/{key}", str(path), "--only-show-errors"]
        if endpoint:
            args.extend(("--endpoint-url", endpoint))
        if region:
            args.extend(("--region", region))
        try:
            subprocess.run(args, check=True, env=env, capture_output=True)
        except subprocess.CalledProcessError:
            raise RuntimeError("AWS CLI could not download the S3 worker") from None
    else:
        virtual_host = env.get("AWS_VIRTUAL_HOSTED_STYLE_REQUEST", "").lower()
        config = None
        if virtual_host in ("true", "false"):
            style = "virtual" if virtual_host == "true" else "path"
            config = Config(s3={"addressing_style": style})
        credentials = {
            option: env[var] for option, var in (
                ("aws_access_key_id", "AWS_ACCESS_KEY_ID"),
                ("aws_secret_access_key", "AWS_SECRET_ACCESS_KEY"),
                ("aws_session_token", "AWS_SESSION_TOKEN"),
            ) if var in env
        }
        try:
            client = boto3.client(
                "s3", endpoint_url=endpoint, region_name=region, config=config,
                **credentials,
            )
            client.download_file(bucket, key, str(path))
        except Exception:
            raise RuntimeError("boto3 could not download the S3 worker") from None


def _worker_executable(worker, directory, payload):
    location = _s3_worker_location(worker)
    if location is None:
        return worker
    bucket, key = location
    target = Path(directory) / "dist-worker"
    _download_worker(bucket, key, target, {
        **{name: value for name, value in os.environ.items() if name not in payload},
        **{name: value for name, value in payload.items() if name in _AWS_ENV},
    })
    target.chmod(0o700)
    return str(target)


class WorkerCli:
    """Adapter for the worker's serde JSON and Parquet artifact protocol."""

    def __init__(self, executable, sql, tables, job_path, output_root, row_key=None,
                 *, local_executable=None, payload=None):
        self.executable = executable
        self.local_executable = local_executable or executable
        self.sql = sql
        self.tables = tables
        self.job_path = Path(job_path)
        self.output_root = output_root
        self.row_key = row_key
        self.payload = payload or {}

    def plan(self):
        args = ["plan", "--sql", self.sql, "--out", str(self.job_path),
                "--output-root", self.output_root]
        for table in self.tables:
            args.extend(("--table", table))
        if self.row_key is not None:
            args.extend(("--row-key", self.row_key))
        _invoke(self.local_executable, args, payload=self.payload)
        return json.loads(self.job_path.read_bytes())

    def commit(self, receipt):
        _invoke(self.local_executable, ["commit-split", "--job", str(self.job_path)],
                receipt=receipt, payload=self.payload)

    def finalize(self):
        return _invoke(self.local_executable, ["finalize-stage", "--job", str(self.job_path)],
                       payload=self.payload)


class RayExecutor:
    """Run one child process per Ray task and return its small JSON receipt."""

    def __init__(self, worker, timeout, address=None):
        import ray

        _validate_ray_address(address)
        if not ray.is_initialized():
            if address in (None, "auto"):
                ray.init()
            else:
                try:
                    ray.init(address=address)
                except RuntimeError as error:
                    if address.startswith("ray://") and str(error).startswith("Version mismatch:"):
                        raise ValueError(
                            f"{error}\nRay Client requires matching Ray and Python major.minor "
                            "versions. Run the driver on the head with --ray-address auto."
                        ) from None
                    raise
        self._ray = ray
        self._payload = ray.put(worker.payload)
        self._job_path = worker.job_path
        self._job = None
        self._worker = worker.executable
        self._timeout = timeout
        @ray.remote(num_cpus=1, max_retries=0)
        def run_split(executable, job, payload_refs, split_id, attempt_id, timeout):
            # Ray dereferences top-level task arguments before executing the task.
            payload = ray.get(payload_refs[0])
            with tempfile.TemporaryDirectory(prefix="dist-worker-") as directory:
                path = Path(directory) / "stage-job.json"
                path.write_bytes(job)
                return _invoke(
                    _worker_executable(executable, directory, payload),
                    ["run-split", "--job", str(path), "--split-id", str(split_id),
                     "--attempt-id", str(attempt_id)],
                    timeout=timeout,
                    payload=payload,
                )

        self._run_split = run_split

    def submit(self, split_id, attempt_id):
        if self._job is None:
            self._job = self._ray.put(self._job_path.read_bytes())
        return self._run_split.remote(
            self._worker, self._job, (self._payload,), split_id, attempt_id, self._timeout
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


def _validate_ray_address(address):
    if address and address.lower().startswith(("http://", "https://")):
        raise ValueError("Use a ray:// client address or submit a Ray Job to the HTTP dashboard")


def _read_payload(stream):
    try:
        payload = json.load(stream)
    except (UnicodeError, ValueError):
        raise ValueError("Expected one JSON object of environment names and string values") from None
    if not isinstance(payload, dict) or not all(
            name and "=" not in name and "\0" not in name
            and isinstance(value, str) and "\0" not in value
            for name, value in payload.items()):
        raise ValueError("Expected one JSON object of environment names and string values")
    return payload


def _read_secrets_file(path):
    try:
        fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
        with os.fdopen(fd, encoding="utf-8") as stream:
            mode = os.fstat(stream.fileno()).st_mode
            if not stat.S_ISREG(mode) or stat.S_IMODE(mode) != 0o600:
                raise ValueError("--secrets-file must be a regular file with permissions 0600")
            return _read_payload(stream)
    except OSError:
        raise ValueError("Could not open or read --secrets-file") from None


def _inherited_payload():
    names = set(_WORKER_ENV)
    with suppress(ValueError, AttributeError):
        for config in json.loads(os.environ.get("DF_DIST_LLM_PROVIDERS", "{}")).values():
            for field in ("api_key_env", "ca_cert_env"):
                name = config.get(field)
                if isinstance(name, str):
                    names.add(name)
    return {name: os.environ[name] for name in names if name in os.environ}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--worker", default="dist-worker", help="Local executable or s3://bucket/key",
    )
    parser.add_argument("--sql", required=True)
    parser.add_argument("--table", action="append", required=True)
    parser.add_argument("--job", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--row-key")
    parser.add_argument("--ray-address")
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--secrets-stdin", action="store_true", help="Read the worker environment JSON from stdin")
    source.add_argument("--secrets-file", type=Path, help="Read the worker environment JSON from a 0600 file")
    parser.add_argument("--timeout", type=float, default=60.0)
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument("--max-attempts", type=int, default=3)
    args = parser.parse_args()
    with tempfile.TemporaryDirectory(prefix="dist-coordinator-") as directory:
        try:
            _validate_ray_address(args.ray_address)
            if args.secrets_stdin:
                payload = _read_payload(sys.stdin)
            elif args.secrets_file:
                payload = _read_secrets_file(args.secrets_file)
            else:
                payload = _inherited_payload()
            worker = WorkerCli(args.worker, args.sql, args.table, args.job,
                               args.output_root, args.row_key, payload=payload)
            executor = RayExecutor(worker, args.timeout, args.ray_address)
        except ValueError as error:
            parser.error(str(error))
        worker.local_executable = _worker_executable(args.worker, directory, payload)
        print(json.dumps(drive(
            worker,
            lambda: executor,
            timeout=args.timeout,
            concurrency=args.concurrency,
            max_attempts=args.max_attempts,
        )))


if __name__ == "__main__":
    main()
