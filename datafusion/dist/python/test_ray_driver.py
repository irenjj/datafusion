# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: See the NOTICE file distributed with this work for additional information regarding copyright ownership
# SPDX-FileContributor: Licensed to the Apache Software Foundation (ASF) under one or more contributor license agreements

import unittest

from ray_driver import drive


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
