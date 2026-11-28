# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: See the NOTICE file distributed with this work for additional information regarding copyright ownership
# SPDX-FileContributor: Licensed to the Apache Software Foundation (ASF) under one or more contributor license agreements

import math
import unittest

from rate_limiter import GlobalRateLimiter
from test_ray_driver import FakeWorker, ScriptedExecutor, run


class GlobalRateLimiterTests(unittest.TestCase):
    def test_refill_does_not_accumulate_burst_or_release_active_permit(self):
        now = 0
        limiter = GlobalRateLimiter(1, 2, clock=lambda: now)
        self.assertTrue(limiter.try_acquire())
        self.assertEqual(limiter.delay(), math.inf)
        now = 120
        self.assertFalse(limiter.try_acquire())
        limiter.release()
        self.assertTrue(limiter.try_acquire())
        limiter.release()
        self.assertEqual(limiter.delay(), 30)
        now = 149
        self.assertFalse(limiter.try_acquire())
        now = 150
        self.assertTrue(limiter.try_acquire())
        limiter.release()

    def test_driver_waits_for_tokens_and_reuses_permits(self):
        class Executor(ScriptedExecutor):
            def __init__(self):
                super().__init__([])
                self.dispatched_at = []
                self.wakeups = []
                self.finished_at = {}

            def submit(self, split_id, attempt_id):
                ticket = super().submit(split_id, attempt_id)
                self.dispatched_at.append(self.now)
                self.finished_at[ticket] = self.now + (45 if split_id == 1 else 1)
                return ticket

            def wait(self, active, timeout):
                self.now = min(self.now + timeout, *(self.finished_at[h] for h in active))
                return [h for h in active if self.finished_at[h] <= self.now]

            def dispatch_wait(self, delay):
                self.wakeups.append(delay)
                self.now += delay

        executor = Executor()
        limiter = GlobalRateLimiter(1, 2, clock=lambda: executor.now)
        worker = FakeWorker([0, 1, 2])
        self.assertEqual(
            run(worker, executor, concurrency=4, rate_limiter=limiter,
                wait_for_dispatch=executor.dispatch_wait),
            [(0, 0), (1, 0), (2, 0)],
        )
        self.assertEqual(executor.dispatched_at, [0, 30, 75])
        self.assertEqual(executor.wakeups, [29])
        self.assertEqual(limiter.delay(), 29)

    def test_dispatch_timer_wakes_while_another_attempt_is_running(self):
        executor = ScriptedExecutor([(30, []), (1, [(1, 0)]), (1, [(0, 0)])])
        wait = executor.wait

        def check_wait(active, timeout):
            if executor.now == 0:
                self.assertEqual(timeout, 30)
            return wait(active, timeout)

        executor.wait = check_wait
        limiter = GlobalRateLimiter(2, 2, clock=lambda: executor.now)
        self.assertEqual(run(FakeWorker([0, 1]), executor, rate_limiter=limiter),
                         [(0, 0), (1, 0)])

    def test_revocation_precedes_inflight_permit_reuse(self):
        executor = ScriptedExecutor([(5, []), (1, [(0, 0), (0, 1)])])
        limiter = GlobalRateLimiter(1, clock=lambda: executor.now)
        cancel = executor.cancel

        def confirm_reap(handle):
            self.assertFalse(limiter.try_acquire())
            cancel(handle)

        executor.cancel = confirm_reap
        self.assertEqual(run(FakeWorker([0]), executor, rate_limiter=limiter, timeout=5),
                         [(0, 1)])
        self.assertEqual(executor.cancelled, [(0, 0)])
        self.assertEqual(executor.read, [(0, 1)])
        self.assertTrue(limiter.try_acquire())
        limiter.release()

    def test_unconfirmed_revocation_stops_dispatch(self):
        executor = ScriptedExecutor([(5, [])])
        limiter = GlobalRateLimiter(1, clock=lambda: executor.now)

        def unavailable(_handle):
            raise RuntimeError("worker termination unconfirmed")

        executor.cancel = unavailable
        with self.assertRaisesRegex(RuntimeError, "termination unconfirmed"):
            run(FakeWorker([0, 1]), executor, rate_limiter=limiter, timeout=5)
        self.assertEqual(executor.submitted, [(0, 0)])
