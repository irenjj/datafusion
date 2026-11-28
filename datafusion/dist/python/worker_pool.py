# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: See the NOTICE file distributed with this work for additional information regarding copyright ownership
# SPDX-FileContributor: Licensed to the Apache Software Foundation (ASF) under one or more contributor license agreements

"""Persistent worker transport and the job-scoped Ray actor pool."""

import asyncio
from collections import deque
from contextlib import suppress
import itertools
import json
import os
import sys
import tempfile
from uuid import uuid4


# Ray can kill the actor without running Python cleanup. Arm the signal in a fresh
# interpreter before exec, since running Python in a forked Ray worker is unsafe.
_GUARDED_EXEC = """\
import ctypes, os, signal, sys
PR_SET_PDEATHSIG = 1
if ctypes.CDLL(None).prctl(PR_SET_PDEATHSIG, signal.SIGKILL, 0, 0, 0):
    sys.exit(1)
if os.getppid() != int(sys.argv[1]):
    sys.exit(1)
os.execvp(sys.argv[2], sys.argv[2:])
"""


class _ProtocolError(RuntimeError):
    pass


class _CommandError(RuntimeError):
    pass


class ServeProcess:
    """One job per daemon, with one request outstanding at a time."""

    def __init__(self, executable, job_id, job, payload, redact):
        self._executable = os.fspath(executable)
        self._job_id = job_id
        self._job = job
        self._payload = payload
        self._redact = redact
        self._process = None

    async def _start(self):
        args = [self._executable, "serve"]
        if sys.platform == "linux":
            args[:0] = [sys.executable, "-c", _GUARDED_EXEC, str(os.getpid())]
        self._process = await asyncio.create_subprocess_exec(
            *args, env={**os.environ, **self._payload},
            stdin=asyncio.subprocess.PIPE, stdout=asyncio.subprocess.PIPE,
            # Forward structured, redacted errors only. Raw diagnostics can contain secrets.
            stderr=asyncio.subprocess.DEVNULL,
        )
        try:
            await self._exchange("load_job", job_id=self._job_id, job=self._job)
        except _CommandError:
            await self.abort()
            raise

    async def _exchange(self, command, **fields):
        request = json.dumps({"command": command, **fields}, separators=(",", ":"))
        self._process.stdin.write(request.encode("utf-8") + b"\n")
        await self._process.stdin.drain()
        line = bytearray()
        while True:
            chunk = await self._process.stdout.read(65536)
            if not chunk:
                raise _ProtocolError("daemon closed stdout before replying")
            line.extend(chunk)
            if b"\n" in chunk:
                break
        try:
            reply = json.loads(line)
        except (ValueError, UnicodeError):
            raise _ProtocolError("invalid JSON reply") from None
        if isinstance(reply, dict):
            if reply.get("ok") is True and "result" in reply:
                return reply["result"]
            if reply.get("ok") is False and isinstance(reply.get("error"), dict):
                message = reply["error"].get("message")
                if isinstance(message, str):
                    detail = self._redact(message, self._payload)
                    raise _CommandError(f"dist-worker serve {command}: {detail}")
        raise _ProtocolError("invalid reply envelope")

    async def request(self, command, *, timeout=None, **fields):
        async def exchange():
            if self._process is None:
                await self._start()
            return await self._exchange(command, job_id=self._job_id, **fields)

        try:
            return await asyncio.wait_for(exchange(), timeout)
        except asyncio.CancelledError:
            await self.abort()
            raise
        except asyncio.TimeoutError:
            await self.abort()
            raise TimeoutError(f"dist-worker serve {command} timed out") from None
        except (OSError, _ProtocolError) as error:
            await self.abort()
            detail = self._redact(str(error), self._payload)
            raise RuntimeError(f"dist-worker serve {command}: {detail}") from None

    async def abort(self):
        if self._process is not None:
            with suppress(ProcessLookupError):
                self._process.kill()
            await self._process.communicate()
            self._process = None

    async def close(self):
        async def shutdown():
            await self._exchange("shutdown")
            self._process.stdin.close()
            await self._process.wait()
            self._process = None

        if self._process is not None:
            try:
                await asyncio.wait_for(shutdown(), 5)
            except (asyncio.TimeoutError, OSError, _ProtocolError, _CommandError):
                await self.abort()


class ServeCoordinator:
    """Use one local daemon for all commits and finalization after CLI planning."""

    def __init__(self, worker, job_id):
        self._worker = worker
        self._job_id = job_id
        self._session = None
        self._loop = asyncio.new_event_loop()

    def plan(self):
        job = self._worker.plan()
        self._session = ServeProcess(
            self._worker.local_executable, self._job_id, job, self._worker.payload,
            self._worker.redact,
        )
        return job

    def _request(self, command, **fields):
        task = self._loop.create_task(self._session.request(command, **fields))
        try:
            return self._loop.run_until_complete(task)
        except KeyboardInterrupt:
            task.cancel()
            with suppress(asyncio.CancelledError):
                self._loop.run_until_complete(task)
            raise

    def commit(self, receipt):
        self._request("commit_split", receipt=receipt)

    def finalize(self):
        return self._request("finalize_stage")

    def close(self):
        try:
            if self._session is not None:
                self._loop.run_until_complete(self._session.close())
        finally:
            self._loop.close()


class DistWorkerActor:
    """Own a worker executable and restart its daemon after a revoked lease."""

    def __init__(self, executable, job_id, job, payload, timeout, resolve, redact):
        self._directory = tempfile.TemporaryDirectory(prefix="dist-worker-")
        executable = resolve(executable, self._directory.name, payload)
        self._session = ServeProcess(executable, job_id, job, payload, redact)
        self._timeout = timeout
        self._revoked = -1
        self._current = None

    async def run_split(self, ticket, split_id, attempt_id):
        if ticket <= self._revoked:
            raise RuntimeError("Split lease revoked")
        if self._current is not None:
            raise RuntimeError("Worker already executing a split")
        task = asyncio.create_task(self._session.request(
            "run_split", split_id=split_id, attempt_id=attempt_id, timeout=self._timeout,
        ))
        self._current = (ticket, task)
        try:
            return await task
        finally:
            self._current = None

    async def revoke(self, ticket):
        self._revoked = max(self._revoked, ticket)
        if self._current is not None and ticket == self._current[0]:
            _, task = self._current
            task.cancel()
            with suppress(asyncio.CancelledError, RuntimeError, TimeoutError):
                await task
            await self._session.abort()

    async def close(self):
        if self._current is not None:
            await self.revoke(self._current[0])
        await self._session.close()
        self._directory.cleanup()


class RayActorExecutor:
    """Dispatch at most one attempt per actor and confirm revocation before reuse."""

    def __init__(self, ray, worker, timeout, create_actor, wait_ready):
        self.job_id = uuid4().hex
        self._ray = ray
        self._wait_ready = wait_ready
        self._timeout = timeout
        self._worker = worker
        self._create_actor = create_actor
        self._job = None
        self._payload = ray.put(worker.payload)
        self._tickets = itertools.count()
        self._actors = set()
        self._idle = deque()
        self._active = {}

    def submit(self, split_id, attempt_id):
        if self._idle:
            actor = self._idle.popleft()
        else:
            if self._job is None:
                self._job = self._ray.put(json.loads(self._worker.job_path.read_bytes()))
            actor = self._create_actor.remote(
                self._worker.executable, self.job_id, self._job, (self._payload,), self._timeout,
            )
            self._actors.add(actor)
        ticket = next(self._tickets)
        handle = actor.run_split.remote(ticket, split_id, attempt_id)
        self._active[handle] = (actor, ticket)
        return handle

    def wait(self, handles, timeout):
        return self._wait_ready(self._ray, handles, timeout)

    def result(self, handle):
        actor, _ = self._active[handle]
        try:
            receipt = self._ray.get(handle)
        except self._ray.exceptions.ActorDiedError:
            self._actors.remove(actor)
            del self._active[handle]
            raise
        except self._ray.exceptions.RayTaskError:
            # The method returned an error after cleaning up its request.
            self._idle.append(actor)
            del self._active[handle]
            raise
        self._idle.appendleft(actor)
        del self._active[handle]
        return receipt

    def cancel(self, handle):
        if handle not in self._active:
            return
        actor, ticket = self._active[handle]
        try:
            self._ray.get(actor.revoke.remote(ticket), timeout=self._timeout)
        except self._ray.exceptions.ActorDiedError:
            self._actors.remove(actor)
        except Exception:
            self._ray.kill(actor, no_restart=True)
            try:
                self._ray.get(actor.revoke.remote(ticket), timeout=self._timeout)
            except self._ray.exceptions.ActorDiedError:
                self._actors.remove(actor)
            except Exception:
                raise RuntimeError("Could not confirm termination of expired worker") from None
            else:
                self._actors.remove(actor)
        else:
            # Prefer an already healthy daemon to this actor's next cold start.
            self._idle.append(actor)
        del self._active[handle]

    def close(self):
        try:
            refs = [actor.close.remote() for actor in self._actors]
            with suppress(self._ray.exceptions.RayError):
                if refs:
                    self._ray.get(refs, timeout=self._timeout)
        finally:
            for actor in self._actors:
                with suppress(self._ray.exceptions.RayError):
                    self._ray.kill(actor, no_restart=True)
            self._active.clear()
            self._idle.clear()
            self._actors.clear()
