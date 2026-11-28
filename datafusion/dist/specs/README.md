<!--
SPDX-License-Identifier: Apache-2.0
SPDX-FileCopyrightText: See the NOTICE file distributed with this work for additional information regarding copyright ownership
SPDX-FileContributor: Licensed to the Apache Software Foundation (ASF) under one or more contributor license agreements
-->

# Leased split queue model

[LeasedSplitQueue.tla](LeasedSplitQueue.tla) models the atomic operations of
[`LeasedSplitQueue`](../src/scheduler.rs) under the queue mutex. TLC explores both
initial split orders, every worker choice and every issued token, including older
attempts after reassignment.

| TLA+ operator or state                         | Rust operation or state         | Mapping                                                                                                                                                     |
| ---------------------------------------------- | ------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `Init`, `pending`, `entryState`, `nextAttempt` | `new`, queue and entry fields   | Every initial permutation represents a caller's insertion order. Attempts start at zero.                                                                    |
| `now`, `AdvanceClock`                          | `MonotonicClock::now`           | Single integer ticks enumerate elapsed times up to `MaxTime`. Deadline arithmetic stays within that range plus the configured durations.                    |
| `Claim(w)`, `issuedTokens`, `fencingEpoch`     | `claim`, `LeaseToken`           | Sweep expired leases before taking the head. Allocate the next split attempt and the next queue-wide epoch.                                                 |
| `Start(tok)`, `Heartbeat(tok)`                 | `start`, `heartbeat`            | Expire the target first. A live token can renew up to `Min(now + LeaseTTL, hardDeadline)`. Start moves a leased split to running.                           |
| `ReportFailure(tok, retryable)`                | `report_failure`                | Validate the live token after checking expiry. A retry with budget joins the tail. Other failures terminate.                                                |
| `Commit(tok, val)`, `CommitResult`             | `commit`, `CommitOutcome`       | Accept an unexpired running token with matching output IDs. On replay the committed token and output determine the outcome.                                 |
| `ExpireStale`, `Expire`, `expiredAuditCount`   | `expire_stale`, `Entry::expire` | Expire strictly after the deadline. Count the intervening `Expired` audit transition inside atomic recovery and append retries in ascending split ID order. |

The hard deadline is set at claim and stays fixed during renewal. Both configurations
use a positive hard duration. The standard configuration uses `LeaseTTL = 2` and
`MaxAttemptDuration = 3`. The hard-cap configuration uses `LeaseTTL = 3` and
`MaxAttemptDuration = 2`, so even an initial claim is truncated.

The output model tracks split ID, attempt ID and an abstract payload whose equality
represents equality of the Rust batches and metrics. Calls that leave queue state
unchanged are represented by TLA+ stuttering. A rejected stale call can still
expire the target's current lease if its deadline has passed. The Rust tests cover
additional implementation details such as file and fingerprint binding, full
failure reasons, forged tokens, optional caps and duration saturation. The model
uses issued-token histories and finite epochs within the configured attempt budget.

## Properties

| Check                              | Guarantee                                                                                                         |
| ---------------------------------- | ----------------------------------------------------------------------------------------------------------------- |
| `TypeOK`                           | State fields stay in their modeled domains.                                                                       |
| `SingleActiveLeasePerSplit`        | Only leased or running splits have a current token. The pending sequence contains each queued split exactly once. |
| `AtMostOneCommitPerSplit`          | At most one publication occurs per split and the output agrees with its terminal state.                           |
| `HardDeadlineNeverExceeded`        | Claim and renewal always cap the lease deadline at the attempt's hard deadline.                                   |
| `AttemptBudgetRespected`           | Issued attempt IDs are contiguous from zero and stay below the attempt limit.                                     |
| `FencingMonotonicAndNoStaleCommit` | Epochs are globally unique and ordered. Active and published tokens are the newest for their split.               |
| `CommitReplayOutcomes`             | A matching replay is idempotent. Changed output conflicts and older tokens are stale.                             |
| `TerminalImmutability`             | Terminal states and accepted outputs persist across every subsequent step.                                        |
| `FencingEpochNeverReused`          | Token history is append-only and the queue-wide epoch advances at most once per step.                             |
| `CommitRequiresLiveLease`          | The first publication matches a running, unexpired lease and its output IDs.                                      |
| `EventuallyAllTerminal`            | Every split eventually commits or fails under `FairSpec`.                                                         |

`FairSpec` assumes weak fairness for clock advancement while the clock can tick,
claims when work is available, and sweeps when there are expired leases. At
`MaxTime` a continuously valid lease can no longer age. Weak fairness for worker
progress at that horizon allows each such split to start, commit or report failure.
This assumption applies only to liveness in the bounded model. The attempt budget
bounds the number of retries a worker can request.

TLC checks deadlocks in both configurations. `Next` includes a self-loop for
the fully terminal state and uses state-changing claim and token operations
elsewhere, so no-op calls cannot mask a nonterminal deadlock. Keep the TLC
`-deadlock` flag omitted to leave deadlock checking enabled.

## Replay with TLC

Run from the workspace root with Java and the official
[TLA+ v1.8.0 tools](https://github.com/tlaplus/tlaplus/releases/tag/v1.8.0).
Keep the JAR and TLC state directories outside the checkout.

```bash
mkdir -p /home/renjj/df-ci-data/tools
curl -fL https://github.com/tlaplus/tlaplus/releases/download/v1.8.0/tla2tools.jar \
  -o /home/renjj/df-ci-data/tools/tla2tools.jar

java -XX:+UseParallelGC -cp /home/renjj/df-ci-data/tools/tla2tools.jar tlc2.TLC \
  -workers auto -noGenerateSpecTE -metadir /tmp/tlc-states-LeasedSplitQueue \
  -config datafusion/dist/specs/LeasedSplitQueue.cfg \
  datafusion/dist/specs/LeasedSplitQueue.tla
java -XX:+UseParallelGC -cp /home/renjj/df-ci-data/tools/tla2tools.jar tlc2.TLC \
  -workers auto -noGenerateSpecTE -metadir /tmp/tlc-states-LeasedSplitQueueHardCap \
  -config datafusion/dist/specs/LeasedSplitQueueHardCap.cfg \
  datafusion/dist/specs/LeasedSplitQueue.tla
```

## Extension boundaries

- C-P3.2 can extend the state machine at the worker transport boundary with
  worker shutdown, delayed or duplicate RPCs, and coordinator-side clock
  authority. Remote work must carry the same attempt token through publication.
- C-P5 can extend the coordinator boundary with durable leases and commit
  recovery. A restart model needs persistent fencing epochs and an atomic
  relationship between accepted output and recovered scheduler state. Any
  future overlapping attempts need their own active-lease and publication
  invariants.
