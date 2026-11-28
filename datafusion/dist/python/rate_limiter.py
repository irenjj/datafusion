# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: See the NOTICE file distributed with this work for additional information regarding copyright ownership
# SPDX-FileContributor: Licensed to the Apache Software Foundation (ASF) under one or more contributor license agreements

"""Admission control for the single coordinator of a distributed stage."""

import math
import time


class GlobalRateLimiter:
    """Cap active attempts and refill a one-token dispatch bucket every 60 / rate seconds."""

    def __init__(self, max_inflight=None, splits_per_minute=None, *, clock=time.monotonic):
        if max_inflight is not None and max_inflight < 1:
            raise ValueError("--max-global-inflight must be positive")
        if splits_per_minute is not None and splits_per_minute < 1:
            raise ValueError("--max-splits-per-minute must be positive")
        self._max_inflight = max_inflight
        self._interval = 0 if splits_per_minute is None else 60 / splits_per_minute
        self._clock = clock
        self._next_token = -math.inf
        self._inflight = 0

    def delay(self):
        """Seconds until admission, or infinity until an attempt releases its permit."""
        return self._delay(self._clock())

    def _delay(self, now):
        if self._max_inflight is not None and self._inflight >= self._max_inflight:
            return math.inf
        return max(0, self._next_token - now)

    def try_acquire(self):
        now = self._clock()
        if self._delay(now):
            return False
        self._inflight += 1
        self._next_token = now + self._interval
        return True

    def release(self):
        if self._inflight == 0:
            raise RuntimeError("No split permit to release")
        self._inflight -= 1
