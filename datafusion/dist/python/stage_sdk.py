# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: See the NOTICE file distributed with this work for additional information regarding copyright ownership
# SPDX-FileContributor: Licensed to the Apache Software Foundation (ASF) under one or more contributor license agreements

"""Run a distributed stage and load its winning Parquet inventory."""

from collections.abc import Mapping
from contextlib import closing
from dataclasses import dataclass, field
import json
import os
from pathlib import Path
import re
import stat
import subprocess
import sys
import tempfile
from types import MappingProxyType
from urllib.parse import quote, unquote, urlsplit

if __package__:
    from . import ray_driver
else:
    import ray_driver


_MANIFEST = "STAGE_MANIFEST.json"
_SCHEMA = "df-stage-manifest-v1"
_FINGERPRINT = re.compile(r"[a-f0-9]{64}")
_U64_MAX = (1 << 64) - 1


@dataclass(frozen=True)
class StageResult:
    """Immutable metrics and winning paths for a finalized stage."""

    plan_fingerprint: str
    output_rows: int
    dlq_rows: int
    dlq_by_code: Mapping[str, int]
    input_tokens: int
    output_tokens: int
    llm_requests: int
    llm_retries: int
    llm_usage_by_provider: Mapping[str, Mapping[str, int]]
    _data: tuple[str, ...] = field(repr=False)
    _dlq: tuple[str, ...] = field(repr=False)

    @classmethod
    def from_manifest(cls, path_or_uri):
        """Load a local manifest file or S3 object, using its directory for paths."""
        location = _location(path_or_uri)
        if isinstance(location, _S3Path):
            content = _read_s3(_s3_client(), location)
        else:
            location = location.parent.resolve() / location.name
            content = _read_local(location)
        return cls._from_document(_decode_manifest(content), location.parent)

    @classmethod
    def load(cls, output_root):
        """Load a stage directory or a job root with exactly one finalized stage."""
        root = _location(output_root)
        if isinstance(root, _S3Path):
            client = _s3_client()
            path = root.joinpath(_MANIFEST)
            content = _read_s3(client, path, missing_ok=True)
            if content is None:
                path = _one_manifest(_s3_manifests(client, root), root.uri())
                content = _read_s3(client, path)
        else:
            root = root.resolve()
            path = root / _MANIFEST
            if not _exists(path):
                path = _one_manifest(_local_manifests(root), str(root))
            content = _read_local(path)
        return cls._from_document(_decode_manifest(content), path.parent)

    def data_uris(self) -> list[str]:
        """Return winning data paths in split ID order."""
        return list(self._data)

    def dlq_uris(self) -> list[str]:
        """Return winning DLQ paths in split ID order, or an empty list."""
        return list(self._dlq)

    @classmethod
    def _from_document(cls, manifest, stage_root):
        fingerprint = _plan_fingerprint(manifest)
        if stage_root.parent.name == "stages" and stage_root.name != fingerprint:
            raise ValueError("Stage directory does not match the manifest plan fingerprint")
        splits = manifest.get("splits")
        if not isinstance(splits, list):
            raise ValueError("Stage manifest must contain a list of winning splits")

        data, dlq = [], []
        previous_id = -1
        fingerprints = set()
        data_rows = dlq_rows = 0
        for split in splits:
            if not isinstance(split, dict) or split.get("schema") != "df-checkpoint-v1":
                raise ValueError("Invalid checkpoint schema in stage manifest")
            split_id = _count(split, "split_id")
            if split_id <= previous_id:
                raise ValueError("Manifest split IDs must be strictly increasing")
            previous_id = split_id
            split_fingerprint = split.get("split_fingerprint")
            if (split.get("plan_fingerprint") != fingerprint
                    or not _is_fingerprint(split_fingerprint)
                    or split_fingerprint in fingerprints):
                raise ValueError(f"Invalid manifest checkpoint fingerprint for split {split_id}")
            fingerprints.add(split_fingerprint)

            attempt_id = _count(split, "winning_attempt_id", maximum=(1 << 32) - 1)
            attempt = f"split-{split_id:016x}-attempt-{attempt_id:04x}"
            data.append(_winner_uri(stage_root, attempt, "data.parquet", split.get("data_rel_path")))
            data_rows += _count(split, "output_rows")
            failed_rows = _count(split, "dlq_rows")
            dlq_path = split.get("dlq_rel_path")
            if (dlq_path is not None) != (failed_rows > 0):
                raise ValueError(f"Invalid manifest DLQ path or row count for split {split_id}")
            if dlq_path is not None:
                dlq.append(_winner_uri(stage_root, attempt, "dlq.parquet", dlq_path))
            dlq_rows += failed_rows

        if (data_rows, dlq_rows) != (_count(manifest, "output_rows"), _count(manifest, "dlq_rows")):
            raise ValueError("Manifest row totals do not match the winning splits")
        by_code = _object(manifest, "dlq_by_code")
        usage = _object(manifest, "llm_usage_by_provider", default={})
        return cls(
            plan_fingerprint=fingerprint,
            output_rows=data_rows,
            dlq_rows=dlq_rows,
            dlq_by_code=MappingProxyType({code: _count(by_code, code) for code in by_code}),
            input_tokens=_count(manifest, "input_tokens", default=0),
            output_tokens=_count(manifest, "output_tokens", default=0),
            llm_requests=_count(manifest, "llm_requests", default=0),
            llm_retries=_count(manifest, "llm_retries", default=0),
            llm_usage_by_provider=MappingProxyType({
                provider: MappingProxyType({
                    name: _count(_object(usage, provider), name)
                    for name in ("input_tokens", "output_tokens", "requests")
                }) for provider in usage
            }),
            _data=tuple(data),
            _dlq=tuple(dlq),
        )


def run_stage(*, sql, tables, output_root, row_key=None, worker="dist-worker",
              ray_address=None, concurrency=4, timeout=60.0, max_attempts=3) -> StageResult:
    """Plan, run and finalize a stage through the Ray driver CLI.

    ``tables`` is a mapping of names to Parquet locations or an iterable of
    ``name=path`` strings. ``timeout`` is the per-attempt timeout in seconds.
    """
    if isinstance(tables, Mapping):
        tables = [f"{name}={os.fspath(path)}" for name, path in tables.items()]
    elif isinstance(tables, str):
        raise TypeError("tables must be a mapping or an iterable of name=path strings")
    ray_driver._validate_ray_address(ray_address)
    root = _location(output_root)
    if isinstance(root, Path):
        root = root.resolve()
    with tempfile.TemporaryDirectory(prefix="dist-coordinator-") as directory:
        # Running the driver as __main__ lets Ray ship its task helpers to workers.
        command = [
            sys.executable, "-B", str(Path(ray_driver.__file__).resolve()),
            f"--worker={os.fspath(worker)}", f"--sql={sql}", f"--output-root={_uri(root)}",
            "--job", str(Path(directory) / "stage-job.json"),
            f"--concurrency={concurrency}", f"--timeout={timeout}", f"--max-attempts={max_attempts}",
        ]
        command.extend(f"--table={table}" for table in tables)
        if row_key is not None:
            command.append(f"--row-key={row_key}")
        if ray_address is not None:
            command.append(f"--ray-address={ray_address}")
        result = subprocess.run(command, capture_output=True, text=True, errors="replace", check=False)
        if result.returncode:
            detail = result.stderr
            for value in sorted(set(ray_driver._inherited_payload().values()), key=len, reverse=True):
                if value:
                    detail = detail.replace(value, "[redacted]")
            raise RuntimeError(f"Ray stage driver exited {result.returncode}: {detail.strip()}")
        manifest = _decode_manifest(result.stdout)
    stage_root = root.joinpath("stages", _plan_fingerprint(manifest))
    return StageResult._from_document(manifest, stage_root)


@dataclass(frozen=True)
class _S3Path:
    bucket: str
    key: str

    @property
    def name(self):
        return self.key.rpartition("/")[2]

    @property
    def parent(self):
        return _S3Path(self.bucket, self.key.rpartition("/")[0])

    def joinpath(self, *parts):
        return _S3Path(self.bucket, "/".join(filter(None, (self.key, *parts))))

    def uri(self):
        return f"s3://{self.bucket}/{quote(self.key)}"


def _location(value):
    value = os.fspath(value)
    if not value:
        raise ValueError("A stage location cannot be empty")
    if "://" not in value:
        return Path(value)
    try:
        uri = urlsplit(value)
    except ValueError:
        raise ValueError("Invalid stage URI") from None
    if "?" in value or "#" in value:
        raise ValueError("Stage URIs cannot contain queries or fragments")
    path = unquote(uri.path, errors="strict")
    if uri.scheme == "file" and uri.netloc in ("", "localhost") and path.startswith("/"):
        return Path(path)
    if uri.scheme != "s3" or not re.fullmatch(r"[a-zA-Z0-9._-]+", uri.netloc):
        raise ValueError("Expected a local path or an S3 URI without embedded credentials")
    key = path.strip("/")
    if key and any(part in ("", ".", "..") for part in key.split("/")):
        raise ValueError("Invalid S3 stage key")
    return _S3Path(uri.netloc, key)


def _uri(path):
    return path.uri() if isinstance(path, _S3Path) else str(path)


def _is_fingerprint(value):
    return isinstance(value, str) and bool(_FINGERPRINT.fullmatch(value))


def _plan_fingerprint(manifest):
    if not isinstance(manifest, dict) or manifest.get("schema") != _SCHEMA:
        raise ValueError("Unsupported stage manifest schema")
    fingerprint = manifest.get("plan_fingerprint")
    if not _is_fingerprint(fingerprint):
        raise ValueError("Invalid stage manifest plan fingerprint")
    return fingerprint


def _count(mapping, name, *, default=None, maximum=_U64_MAX):
    value = mapping.get(name, default)
    if type(value) is not int or not 0 <= value <= maximum:
        raise ValueError(f"Invalid integer for manifest field {name}")
    return value


def _object(mapping, name, *, default=None):
    value = mapping.get(name, default)
    if not isinstance(value, dict):
        raise ValueError(f"Invalid object for manifest field {name}")
    return value


def _winner_uri(root, attempt, filename, relative):
    if relative != f"attempts/{attempt}/{filename}":
        raise ValueError(f"Invalid manifest {filename} path for {attempt}")
    path = root.joinpath("attempts", attempt, filename)
    if isinstance(path, Path):
        _check_local_path(path)
    return _uri(path)


def _check_local_path(path):
    if path.resolve() != path:
        raise ValueError(f"Stage file must stay under its output root without symlinks: {path}")


def _read_local(path):
    _check_local_path(path)
    if not stat.S_ISREG(path.stat().st_mode):
        raise OSError(f"Expected a regular stage manifest file at {path}")
    return path.read_bytes()


def _exists(path):
    try:
        path.lstat()
        return True
    except FileNotFoundError:
        return False


def _local_manifests(root):
    stages = root / "stages"
    if not _exists(stages):
        return
    _check_local_path(stages)
    for entry in stages.iterdir():
        if _is_fingerprint(entry.name) and entry.is_dir() and _exists(entry / _MANIFEST):
            yield entry / _MANIFEST


def _one_manifest(paths, root):
    paths = iter(paths)
    path = next(paths, None)
    if path is None:
        raise FileNotFoundError(f"No {_MANIFEST} under {root}")
    if next(paths, None) is not None:
        raise ValueError(f"Multiple finalized stages under {root}; use a stage directory")
    return path


def _decode_manifest(content):
    try:
        return json.loads(content)
    except (UnicodeError, ValueError):
        raise ValueError("Invalid stage manifest JSON") from None


def _s3_client():
    try:
        import boto3
        from botocore.config import Config
    except ImportError:
        raise RuntimeError("Loading an S3 stage manifest requires boto3") from None
    endpoint = (os.getenv("AWS_ENDPOINT_URL_S3") or os.getenv("AWS_ENDPOINT_URL")
                or os.getenv("AWS_ENDPOINT"))
    region = os.getenv("AWS_REGION") or os.getenv("AWS_DEFAULT_REGION")
    virtual = os.getenv("AWS_VIRTUAL_HOSTED_STYLE_REQUEST", "").lower()
    style = {"true": "virtual", "false": "path"}.get(virtual)
    config = Config(s3={"addressing_style": style}) if style else None
    try:
        return boto3.client("s3", endpoint_url=endpoint, region_name=region, config=config)
    except Exception:
        raise RuntimeError("Could not configure boto3 for the S3 stage manifest") from None


def _s3_missing(error):
    return getattr(error, "response", {}).get("Error", {}).get("Code") in (
        "NoSuchKey", "404", "NotFound",
    )


def _read_s3(client, location, *, missing_ok=False):
    try:
        obj = client.get_object(Bucket=location.bucket, Key=location.key)
        with closing(obj["Body"]) as body:
            return body.read()
    except Exception as error:
        if _s3_missing(error):
            if missing_ok:
                return None
            raise FileNotFoundError(f"No stage manifest at {location.uri()}") from None
        raise OSError(f"Could not read S3 stage manifest at {location.uri()}; check AWS settings and permissions") from None


def _s3_manifests(client, root):
    prefix = root.joinpath("stages").key + "/"
    try:
        pages = client.get_paginator("list_objects_v2").paginate(
            Bucket=root.bucket, Prefix=prefix, Delimiter="/",
        )
        for page in pages:
            for entry in page.get("CommonPrefixes", ()):
                stage = entry["Prefix"].removeprefix(prefix).rstrip("/")
                if _is_fingerprint(stage):
                    manifest = root.joinpath("stages", stage, _MANIFEST)
                    try:
                        client.head_object(Bucket=manifest.bucket, Key=manifest.key)
                    except Exception as error:
                        if _s3_missing(error):
                            continue
                        raise
                    yield manifest
    except Exception:
        raise OSError(f"Could not discover finalized stages under {root.uri()}; check AWS settings and permissions") from None
