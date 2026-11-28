# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: See the NOTICE file distributed with this work for additional information regarding copyright ownership
# SPDX-FileContributor: Licensed to the Apache Software Foundation (ASF) under one or more contributor license agreements

"""Run distributed Parquet stages with Ray workers or an injected executor."""

import argparse
from collections import deque
from collections.abc import Mapping
from contextlib import ExitStack, suppress
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
from threading import Event
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
_TABLE_FLAGS = ("--table", "--stage-table", "--stage-dlq-table", "--lance-table")


@dataclass(frozen=True)
class _Attempt:
    split_id: int
    attempt_id: int
    tries: int
    deadline: float


def drive(worker, executor_factory, *, timeout=60.0, concurrency=4,
          max_attempts=3, clock=time.monotonic, rate_limiter=None,
          wait_for_dispatch=None):
    """Plan, execute, commit, and finalize using a worker command adapter.

    An executor provides submit(split_id, attempt_id), wait(handles, timeout),
    result(handle), and cancel(handle). Under an inflight limit, cancel must confirm
    that the attempt stopped or raise. Only the driver can publish a receipt.
    """
    _validate_dispatch(timeout, concurrency, max_attempts)

    job = worker.plan()
    pending = deque(
        (split["split_id"], split["attempt_id"], 1)
        for split in job["pending_splits"]
    )
    if len({split_id for split_id, _, _ in pending}) != len(pending):
        raise ValueError("Duplicate pending split ID in stage job")
    if not pending:
        return worker.finalize()

    executor = executor_factory()
    active = {}
    if wait_for_dispatch is None:
        wait_for_dispatch = Event().wait

    def retire(handle):
        del active[handle]
        if rate_limiter is not None:
            rate_limiter.release()

    def retry(attempt, error):
        if attempt.tries >= max_attempts or attempt.attempt_id == 0xFFFFFFFF:
            raise RuntimeError(
                f"split {attempt.split_id} exhausted attempts at {attempt.attempt_id}: {error}"
            ) from error
        pending.append((attempt.split_id, attempt.attempt_id + 1, attempt.tries + 1))

    try:
        while pending or active:
            while pending and len(active) < concurrency:
                if rate_limiter is not None and not rate_limiter.try_acquire():
                    break
                split_id, attempt_id, tries = pending.popleft()
                try:
                    handle = executor.submit(split_id, attempt_id)
                except BaseException:
                    if rate_limiter is not None:
                        rate_limiter.release()
                    raise
                active[handle] = _Attempt(split_id, attempt_id, tries, clock() + timeout)

            dispatch_delay = math.inf
            if rate_limiter is not None and pending and len(active) < concurrency:
                dispatch_delay = rate_limiter.delay()
            if not active:
                wait_for_dispatch(dispatch_delay)
                continue

            deadline = min(attempt.deadline for attempt in active.values())
            ready = executor.wait(
                tuple(active), min(dispatch_delay, max(0, deadline - clock())),
            )
            now = clock()
            for handle in ready:
                attempt = active.get(handle)
                if attempt is None or attempt.deadline <= now:
                    continue
                try:
                    receipt = executor.result(handle)
                except Exception as error:
                    executor.cancel(handle)
                    retire(handle)
                    retry(attempt, error)
                    continue

                retire(handle)
                try:
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
                    executor.cancel(handle)
                    retire(handle)
                    retry(attempt, TimeoutError(f"attempt {attempt.attempt_id} expired"))
    finally:
        for handle in tuple(active):
            _cancel(executor, handle)
            retire(handle)

    return worker.finalize()


def _cancel(executor, handle):
    with suppress(Exception):
        executor.cancel(handle)


def _redact(detail, payload):
    values = set()
    for secret in payload.values():
        if secret:
            values.update((secret, json.dumps(secret)[1:-1]))
            values.update(secret.splitlines())
    for value in sorted(values - {""}, key=lambda value: (-len(value), value)):
        detail = detail.replace(value, "[redacted]")
    return detail


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
        detail = _redact(result.stderr, payload)
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

    redact = staticmethod(_redact)

    def __init__(self, executable, sql, tables, job_path, output_root, row_key=None,
                 *, local_executable=None, payload=None, stage_tables=(),
                 stage_dlq_tables=(), lance_tables=(), prepared=False):
        self.executable = executable
        self.local_executable = local_executable or executable
        self.sql = sql
        self.tables = tables
        self.stage_tables = stage_tables
        self.stage_dlq_tables = stage_dlq_tables
        self.lance_tables = lance_tables
        self.job_path = Path(job_path)
        self.output_root = output_root
        self.row_key = row_key
        self.payload = payload or {}
        self._prepared = prepared

    def plan(self):
        if not self._prepared:
            args = ["plan", "--sql", self.sql, "--out", str(self.job_path),
                    "--output-root", self.output_root]
            for flag, tables in zip(_TABLE_FLAGS, (
                    self.tables, self.stage_tables, self.stage_dlq_tables, self.lance_tables)):
                for table in tables:
                    args.extend((flag, table))
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

    def __init__(self, worker, timeout, address=None, *, confirm_stop=False):
        ray = _connect_ray(address)
        self._ray = ray
        self._confirm_stop = confirm_stop
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
        return _wait_ready(self._ray, handles, timeout)

    def result(self, handle):
        return self._ray.get(handle)

    def cancel(self, handle):
        if not self._confirm_stop:
            with suppress(Exception):
                self._ray.cancel(handle)
            return
        # The stateless task's subprocess timeout kills and reaps the child.
        # Wait for its completion before reissuing an inflight permit.
        try:
            self._ray.get(handle, timeout=self._timeout)
        except self._ray.exceptions.RayTaskError:
            pass
        except self._ray.exceptions.RayError:
            with suppress(Exception):
                self._ray.cancel(handle)
            raise RuntimeError("Could not confirm termination of expired task") from None


def _connect_ray(address):
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
    return ray


def _wait_ready(ray, handles, timeout):
    ready, pending = ray.wait(list(handles), timeout=timeout)
    if ready and pending:
        also_ready, _ = ray.wait(pending, num_returns=len(pending), timeout=0)
        ready.extend(also_ready)
    return ready


def _actor_job(stack, worker, args):
    import worker_pool

    ray = _connect_ray(args.ray_address)
    # Ship code by value so Ray nodes do not need the driver's source directory.
    ray.cloudpickle.register_pickle_by_value(worker_pool)
    ray.cloudpickle.register_pickle_by_value(sys.modules[__name__])

    @ray.remote(num_cpus=1, max_concurrency=2, max_restarts=0, max_task_retries=0)
    class WorkerActor(worker_pool.DistWorkerActor):
        def __init__(self, executable, job_id, job, payload_refs, timeout):
            super().__init__(
                executable, job_id, job, ray.get(payload_refs[0]), timeout,
                _worker_executable, _redact,
            )

    executor = worker_pool.RayActorExecutor(ray, worker, args.timeout, WorkerActor, _wait_ready)
    stack.callback(executor.close)
    coordinator = worker_pool.ServeCoordinator(worker, executor.job_id)
    stack.callback(coordinator.close)
    pool_size = args.concurrency if args.pool_size is None else args.pool_size
    capacity = int(ray.cluster_resources().get("CPU", 0))
    return coordinator, executor, min(args.concurrency, pool_size, max(1, capacity))


def _validate_ray_address(address):
    if address and address.lower().startswith(("http://", "https://")):
        raise ValueError("Use a ray:// client address or submit a Ray Job to the HTTP dashboard")


def _positive_int(value, flag):
    if type(value) is not int or value < 1:
        raise ValueError(f"{flag} must be a positive integer")


def _validate_dispatch(timeout, concurrency, max_attempts):
    if type(timeout) not in (int, float) or not 0 < timeout <= sys.float_info.max:
        raise ValueError("--timeout must be positive and finite")
    _positive_int(concurrency, "--concurrency")
    _positive_int(max_attempts, "--max-attempts")


def _validate_scheduler(execution_mode, pool_size, max_global_inflight, splits_per_minute):
    if execution_mode not in ("actor", "task"):
        raise ValueError("--execution-mode must be actor or task")
    if pool_size is not None and execution_mode != "actor":
        raise ValueError("--pool-size applies only to actor mode")
    for value, flag in (
        (pool_size, "--pool-size"),
        (max_global_inflight, "--max-global-inflight"),
        (splits_per_minute, "--max-splits-per-minute"),
    ):
        if value is not None:
            _positive_int(value, flag)
    if splits_per_minute is not None and splits_per_minute > sys.float_info.max:
        raise ValueError("--max-splits-per-minute is too large")


def _table_bindings(*groups, required=True):
    names = set()
    for flag, group in zip(_TABLE_FLAGS, groups):
        if group is None:
            continue
        try:
            if isinstance(group, Mapping):
                if any(not isinstance(name, str) or "=" in name for name in group):
                    raise ValueError(f"{flag} expects name=path bindings")
                group = [f"{name}={os.fspath(path)}" for name, path in group.items()]
            if isinstance(group, (str, bytes)):
                raise TypeError
            bindings = list(group)
        except TypeError:
            raise ValueError(f"{flag} expects a mapping or iterable of name=path bindings") from None
        for binding in bindings:
            if not isinstance(binding, str) or "\0" in binding:
                raise ValueError(f"{flag} expects name=path bindings")
            name, _, path = binding.partition("=")
            if not name.strip() or name != name.strip() or not path.strip():
                raise ValueError(f"{flag} expects name=path bindings")
            if name in names:
                raise ValueError(f"Duplicate table name in {flag}")
            names.add(name)
            yield flag, binding
    if required and not names:
        raise ValueError("Supply at least one --table, --stage-table, --stage-dlq-table or --lance-table")


def _read_payload(stream):
    try:
        payload = json.load(stream)
    except (UnicodeError, ValueError):
        raise ValueError("Expected one JSON object of environment names and string values") from None
    return _validate_payload(payload)


def _validate_payload(payload):
    if not isinstance(payload, dict) or not all(
            isinstance(name, str) and name and "=" not in name and "\0" not in name
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


def _replay_args(args):
    options = ["--base-output-root", args.base_output_root, "--output-root", args.output_root,
               "--sql", args.sql, "--row-key", args.row_key, "--batch-size", str(args.batch_size)]
    for flag, tables in (("--table", args.table), ("--stage-table", args.stage_table)):
        for table in tables:
            options.extend((flag, table))
    return options


def _run_stage(stack, args, payload, limiter):
    directory = stack.enter_context(tempfile.TemporaryDirectory(prefix="dist-coordinator-"))
    job_path = args.job or str(Path(directory) / "stage-job.json")
    replay = _replay_args(args) if args.base_output_root is not None else None
    worker = WorkerCli(args.worker, args.sql, args.table, job_path, args.output_root,
                       args.row_key, payload=payload, prepared=replay is not None,
                       stage_tables=args.stage_table, stage_dlq_tables=args.stage_dlq_table,
                       lance_tables=args.lance_table)
    if replay is not None:
        worker.local_executable = _worker_executable(args.worker, directory, payload)
        if args.execution_mode == "local":
            return _invoke(worker.local_executable, ["replay-dlq", *replay,
                                                     "--timeout-secs", str(args.timeout),
                                                     "--max-attempts", str(args.max_attempts)],
                           payload=payload)
        prepared = _invoke(worker.local_executable, ["plan-replay-dlq", *replay, "--out", job_path],
                           payload=payload)
        if "result" in prepared:
            return prepared["result"]
        attempt_id = prepared["attempt_id"]
    if args.execution_mode == "actor":
        coordinator, executor, concurrency = _actor_job(stack, worker, args)
    else:
        coordinator, concurrency = worker, args.concurrency
        executor = RayExecutor(worker, args.timeout, args.ray_address,
                               confirm_stop=args.max_global_inflight is not None)
    if replay is None:
        worker.local_executable = _worker_executable(args.worker, directory, payload)
    manifest = drive(coordinator, lambda: executor, timeout=args.timeout, concurrency=concurrency,
                     max_attempts=args.max_attempts, rate_limiter=limiter)
    if replay is not None:
        return _invoke(worker.local_executable,
                       ["finalize-replay-dlq", *replay, "--attempt-id", str(attempt_id)],
                       payload=payload)
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--worker", default="dist-worker", help="Local executable or s3://bucket/key",
    )
    parser.add_argument("--sql", required=True)
    for flag in _TABLE_FLAGS:
        parser.add_argument(flag, action="append", default=[])
    parser.add_argument("--job")
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--base-output-root", help="Replay the failed keys from this finalized stage")
    parser.add_argument("--batch-size", type=int, default=8192, help="Maximum replay source rows per split")
    parser.add_argument("--row-key")
    parser.add_argument("--ray-address")
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--secrets-stdin", action="store_true", help="Read the worker environment JSON from stdin")
    source.add_argument("--secrets-file", type=Path, help="Read the worker environment JSON from a 0600 file")
    parser.add_argument("--timeout", type=float, default=60.0)
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument("--max-attempts", type=int, default=3)
    parser.add_argument("--execution-mode", choices=("local", "actor", "task"), default="actor")
    parser.add_argument("--pool-size", type=int, help="Maximum actor count (defaults to --concurrency)")
    parser.add_argument("--max-global-inflight", type=int, help="Maximum active split attempts per job")
    parser.add_argument("--max-splits-per-minute", "--splits-per-minute", type=int,
                        help="Maximum split dispatch rate per job")
    args = parser.parse_args()
    with ExitStack() as stack:
        payload = {}
        try:
            _validate_ray_address(args.ray_address)
            _validate_dispatch(args.timeout, args.concurrency, args.max_attempts)
            if args.base_output_root is None:
                if not args.job:
                    raise ValueError("--job is required when planning a stage")
                list(_table_bindings(args.table, args.stage_table, args.stage_dlq_table, args.lance_table))
                _validate_scheduler(args.execution_mode, args.pool_size, args.max_global_inflight,
                                    args.max_splits_per_minute)
            else:
                if not args.row_key or args.batch_size < 1:
                    raise ValueError("Replay requires --row-key and a positive --batch-size")
                if args.stage_dlq_table or args.lance_table or len(args.table) + len(args.stage_table) > 1:
                    raise ValueError("Replay accepts one --table or --stage-table source")
                list(_table_bindings(args.table, args.stage_table, required=False))
                if args.pool_size is not None:
                    _positive_int(args.pool_size, "--pool-size")
            limiter = None
            if args.max_global_inflight is not None or args.max_splits_per_minute is not None:
                from rate_limiter import GlobalRateLimiter

                limiter = GlobalRateLimiter(args.max_global_inflight, args.max_splits_per_minute)
            if args.secrets_stdin:
                payload = _read_payload(sys.stdin)
            elif args.secrets_file:
                payload = _read_secrets_file(args.secrets_file)
            else:
                payload = _inherited_payload()
            manifest = _run_stage(stack, args, payload, limiter)
        except ValueError as error:
            parser.error(_redact(str(error), {**_inherited_payload(), **payload}))
        except (OSError, RuntimeError, ImportError) as error:
            parser.exit(1, f"{_redact(str(error), {**_inherited_payload(), **payload})}\n")
    print(json.dumps(manifest))


if __name__ == "__main__":
    main()
