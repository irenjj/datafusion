---- MODULE LeasedSplitQueue ----
\* SPDX-License-Identifier: Apache-2.0
\* SPDX-FileCopyrightText: See the NOTICE file distributed with this work for additional information regarding copyright ownership
\* SPDX-FileContributor: Licensed to the Apache Software Foundation (ASF) under one or more contributor license agreements

EXTENDS Naturals, Sequences, FiniteSets

CONSTANTS SplitIds, Workers, MaxAttempts, LeaseTTL, MaxAttemptDuration, MaxTime, Values

ASSUME /\ SplitIds \subseteq Nat
       /\ SplitIds # {} /\ Workers # {} /\ Values # {}
       /\ MaxAttempts \in Nat \ {0}
       /\ LeaseTTL \in Nat \ {0}
       /\ MaxAttemptDuration \in Nat \ {0}
       /\ MaxTime \in Nat

VARIABLES now, fencingEpoch, pending, entryState, nextAttempt, activeLease,
          committedOutput, issuedTokens, commitCount, expiredAuditCount

vars == <<now, fencingEpoch, pending, entryState, nextAttempt, activeLease,
          committedOutput, issuedTokens, commitCount, expiredAuditCount>>

ActiveStates == {"Leased", "Running"}
TerminalStates == {"Committed", "Failed"}
States == ActiveStates \cup TerminalStates \cup {"Queued"}
AttemptIds == 0..(MaxAttempts - 1)
MaxEpoch == Cardinality(SplitIds) * MaxAttempts
TokenType == [split : SplitIds, attempt : AttemptIds, worker : Workers, epoch : 1..MaxEpoch]
OutputValues == [split : SplitIds, attempt : AttemptIds, payload : Values]
LeaseType == [token : TokenType, leaseDeadline : 0..(MaxTime + LeaseTTL),
              hardDeadline : 0..(MaxTime + MaxAttemptDuration)]

NoToken == [split |-> CHOOSE s \in SplitIds : TRUE, attempt |-> 0,
            worker |-> CHOOSE w \in Workers : TRUE, epoch |-> 0]
NoLease == [token |-> NoToken, leaseDeadline |-> 0, hardDeadline |-> 0]
NoOutput == [token |-> NoToken,
             value |-> [split |-> NoToken.split, attempt |-> 0,
                         payload |-> CHOOSE v \in Values : TRUE]]

SeqSet(seq) == {seq[i] : i \in DOMAIN seq}

RECURSIVE Ascending(_)
Ascending(ids) ==
    IF ids = {} THEN <<>>
    ELSE LET first == CHOOSE s \in ids : \A t \in ids : s <= t
         IN <<first>> \o Ascending(ids \ {first})

Min(a, b) == IF a <= b THEN a ELSE b
RenewedDeadline(hard) == Min(now + LeaseTTL, hard)
IsExpired(s) == activeLease[s] # NoLease /\ now > activeLease[s].leaseDeadline
MatchesActive(tok) ==
    activeLease[tok.split] # NoLease /\ activeLease[tok.split].token = tok
FailureState(s, retryable) ==
    IF retryable /\ nextAttempt[s] < MaxAttempts THEN "Queued" ELSE "Failed"

Init ==
    /\ now = 0
    /\ fencingEpoch = 0
    /\ pending \in [1..Cardinality(SplitIds) -> SplitIds]
    /\ SeqSet(pending) = SplitIds
    /\ entryState = [s \in SplitIds |-> "Queued"]
    /\ nextAttempt = [s \in SplitIds |-> 0]
    /\ activeLease = [s \in SplitIds |-> NoLease]
    /\ committedOutput = [s \in SplitIds |-> NoOutput]
    /\ issuedTokens = {}
    /\ commitCount = [s \in SplitIds |-> 0]
    /\ expiredAuditCount = [s \in SplitIds |-> 0]

AdvanceClock ==
    /\ now < MaxTime
    /\ now' = now + 1
    /\ UNCHANGED <<fencingEpoch, pending, entryState, nextAttempt, activeLease,
                   committedOutput, issuedTokens, commitCount, expiredAuditCount>>

\* Each expired lease records Expired and its recovery in the same critical section.
AfterExpiration(ids) ==
    [ pending |-> pending \o Ascending({s \in ids : FailureState(s, TRUE) = "Queued"}),
      states |-> [s \in SplitIds |-> IF s \in ids THEN FailureState(s, TRUE)
                                     ELSE entryState[s]],
      leases |-> [s \in SplitIds |-> IF s \in ids THEN NoLease ELSE activeLease[s]],
      audit |-> [s \in SplitIds |-> expiredAuditCount[s] + (IF s \in ids THEN 1 ELSE 0)] ]

Expire(ids) ==
    LET recovered == AfterExpiration(ids)
    IN /\ pending' = recovered.pending
       /\ entryState' = recovered.states
       /\ activeLease' = recovered.leases
       /\ expiredAuditCount' = recovered.audit
       /\ UNCHANGED <<now, fencingEpoch, nextAttempt, committedOutput, issuedTokens, commitCount>>

ExpireStale ==
    LET ids == {s \in SplitIds : IsExpired(s)}
    IN ids # {} /\ Expire(ids)

Claim(w) ==
    LET ids == {s \in SplitIds : IsExpired(s)}
        recovered == AfterExpiration(ids)
    IN IF recovered.pending = <<>>
       THEN Expire(ids)
       ELSE LET s == Head(recovered.pending)
                tok == [split |-> s, attempt |-> nextAttempt[s], worker |-> w,
                        epoch |-> fencingEpoch + 1]
                hard == now + MaxAttemptDuration
                lease == [token |-> tok, leaseDeadline |-> RenewedDeadline(hard),
                          hardDeadline |-> hard]
            IN /\ pending' = Tail(recovered.pending)
               /\ entryState' = [recovered.states EXCEPT ![s] = "Leased"]
               /\ nextAttempt' = [nextAttempt EXCEPT ![s] = @ + 1]
               /\ activeLease' = [recovered.leases EXCEPT ![s] = lease]
               /\ expiredAuditCount' = recovered.audit
               /\ fencingEpoch' = tok.epoch
               /\ issuedTokens' = issuedTokens \cup {tok}
               /\ UNCHANGED <<now, committedOutput, commitCount>>

Renew(tok, start) ==
    LET s == tok.split
    IN IF IsExpired(s) THEN Expire({s})
       ELSE IF MatchesActive(tok) /\ (~start \/ entryState[s] = "Leased")
            THEN /\ activeLease' =
                        [activeLease EXCEPT ![s].leaseDeadline =
                            RenewedDeadline(activeLease[s].hardDeadline)]
                 /\ entryState' = IF start THEN [entryState EXCEPT ![s] = "Running"]
                                  ELSE entryState
                 /\ UNCHANGED <<now, fencingEpoch, pending, nextAttempt,
                                committedOutput, issuedTokens, commitCount, expiredAuditCount>>
            ELSE UNCHANGED vars

Start(tok) == Renew(tok, TRUE)
Heartbeat(tok) == Renew(tok, FALSE)

ReportFailure(tok, retryable) ==
    LET s == tok.split
        destination == FailureState(s, retryable)
    IN IF IsExpired(s) THEN Expire({s})
       ELSE IF MatchesActive(tok)
            THEN /\ pending' = IF destination = "Queued" THEN Append(pending, s)
                               ELSE pending
                 /\ entryState' = [entryState EXCEPT ![s] = destination]
                 /\ activeLease' = [activeLease EXCEPT ![s] = NoLease]
                 /\ UNCHANGED <<now, fencingEpoch, nextAttempt,
                                committedOutput, issuedTokens, commitCount, expiredAuditCount>>
            ELSE UNCHANGED vars

CommitResult(tok, val) ==
    LET s == tok.split
        published == committedOutput[s]
    IN IF published # NoOutput
       THEN IF tok # published.token THEN "StaleAttempt"
            ELSE IF val = published.value THEN "AlreadyCommitted" ELSE "ConflictingCommit"
       ELSE IF ~MatchesActive(tok) \/ IsExpired(s) \/ entryState[s] # "Running"
            THEN "RejectedLease"
            ELSE IF val.split # s \/ val.attempt # tok.attempt THEN "ConflictingCommit"
                 ELSE "Committed"

Commit(tok, val) ==
    LET s == tok.split
    IN IF IsExpired(s) THEN Expire({s})
       ELSE IF CommitResult(tok, val) = "Committed"
            THEN /\ entryState' = [entryState EXCEPT ![s] = "Committed"]
                 /\ activeLease' = [activeLease EXCEPT ![s] = NoLease]
                 /\ committedOutput' =
                        [committedOutput EXCEPT ![s] = [token |-> tok, value |-> val]]
                 /\ commitCount' = [commitCount EXCEPT ![s] = @ + 1]
                 /\ UNCHANGED <<now, fencingEpoch, pending, nextAttempt,
                                issuedTokens, expiredAuditCount>>
            ELSE UNCHANGED vars

ClaimAny == \E w \in Workers : Claim(w)
TokenOperation ==
    \E tok \in issuedTokens :
        \/ Start(tok)
        \/ Heartbeat(tok)
        \/ \E retryable \in BOOLEAN : ReportFailure(tok, retryable)
        \/ \E val \in OutputValues : Commit(tok, val)

AllTerminal == \A s \in SplitIds : entryState[s] \in TerminalStates

\* Show a deadlock at a nonterminal state with no state-changing successor.
Next ==
    \/ AdvanceClock
    \/ <<ClaimAny>>_vars
    \/ <<TokenOperation>>_vars
    \/ ExpireStale
    \/ AllTerminal /\ UNCHANGED vars

\* At the bounded time horizon, live leases can progress through a worker.
WorkerProgressAtHorizon(s) ==
    /\ now = MaxTime /\ activeLease[s] # NoLease /\ ~IsExpired(s)
    /\ LET tok == activeLease[s].token
       IN \/ Start(tok)
          \/ \E val \in OutputValues : Commit(tok, val)
          \/ \E retryable \in BOOLEAN : ReportFailure(tok, retryable)

FairSpec ==
    /\ Init /\ [][Next]_vars
    /\ WF_vars(AdvanceClock)
    /\ WF_vars(ClaimAny)
    /\ WF_vars(ExpireStale)
    /\ \A s \in SplitIds : WF_vars(WorkerProgressAtHorizon(s))

TypeOK ==
    /\ now \in 0..MaxTime
    /\ fencingEpoch \in 0..MaxEpoch
    /\ pending \in Seq(SplitIds)
    /\ entryState \in [SplitIds -> States]
    /\ nextAttempt \in [SplitIds -> 0..MaxAttempts]
    /\ activeLease \in [SplitIds -> LeaseType \cup {NoLease}]
    /\ committedOutput \in [SplitIds -> [token : TokenType, value : OutputValues] \cup {NoOutput}]
    /\ issuedTokens \subseteq TokenType
    /\ commitCount \in [SplitIds -> Nat]
    /\ expiredAuditCount \in [SplitIds -> 0..MaxAttempts]

SingleActiveLeasePerSplit ==
    /\ Len(pending) = Cardinality(SeqSet(pending))
    /\ \A s \in SplitIds :
        /\ (activeLease[s] # NoLease <=> entryState[s] \in ActiveStates)
        /\ (s \in SeqSet(pending) <=> entryState[s] = "Queued")
        /\ (activeLease[s] # NoLease => activeLease[s].token.split = s)

AtMostOneCommitPerSplit ==
    \A s \in SplitIds :
        /\ commitCount[s] <= 1
        /\ (entryState[s] = "Committed" <=> committedOutput[s] # NoOutput)
        /\ (commitCount[s] = 1 <=> committedOutput[s] # NoOutput)

HardDeadlineNeverExceeded ==
    \A s \in SplitIds :
        activeLease[s] # NoLease => activeLease[s].leaseDeadline <= activeLease[s].hardDeadline

TokensFor(s) == {tok \in issuedTokens : tok.split = s}

AttemptBudgetRespected ==
    /\ \A s \in SplitIds :
        /\ nextAttempt[s] <= MaxAttempts
        /\ expiredAuditCount[s] <= nextAttempt[s]
        /\ Cardinality(TokensFor(s)) = nextAttempt[s]
        /\ {tok.attempt : tok \in TokensFor(s)} = 0..(nextAttempt[s] - 1)
    /\ \A tok \in issuedTokens : tok.attempt < MaxAttempts

IsLatest(tok) ==
    tok \in issuedTokens /\ \A prior \in TokensFor(tok.split) : prior.epoch <= tok.epoch

FencingMonotonicAndNoStaleCommit ==
    /\ Cardinality(issuedTokens) = fencingEpoch
    /\ {tok.epoch : tok \in issuedTokens} = 1..fencingEpoch
    /\ \A a, b \in issuedTokens :
        a.split = b.split => (a.epoch < b.epoch <=> a.attempt < b.attempt)
    /\ \A s \in SplitIds :
        /\ (activeLease[s] # NoLease => IsLatest(activeLease[s].token))
        /\ (committedOutput[s] # NoOutput =>
                committedOutput[s].token.split = s /\ IsLatest(committedOutput[s].token))

CommitReplayOutcomes ==
    \A s \in SplitIds :
        LET published == committedOutput[s]
        IN published # NoOutput =>
            /\ CommitResult(published.token, published.value) = "AlreadyCommitted"
            /\ \A val \in OutputValues \ {published.value} :
                CommitResult(published.token, val) = "ConflictingCommit"
            /\ \A prior \in TokensFor(s) \ {published.token} :
                CommitResult(prior, published.value) = "StaleAttempt"

TerminalImmutability ==
    [][\A s \in SplitIds :
        entryState[s] \in TerminalStates =>
            (entryState'[s] = entryState[s] /\ committedOutput'[s] = committedOutput[s])]_vars

FencingEpochNeverReused ==
    [][/\ issuedTokens \subseteq issuedTokens'
       /\ fencingEpoch' \in fencingEpoch..(fencingEpoch + 1)]_vars

CommitRequiresLiveLease ==
    [][\A s \in SplitIds :
        (committedOutput[s] = NoOutput /\ committedOutput'[s] # NoOutput) =>
            LET published == committedOutput'[s]
            IN /\ entryState[s] = "Running" /\ ~IsExpired(s)
               /\ MatchesActive(published.token) /\ published.token.split = s
               /\ published.value.split = s /\ published.value.attempt = published.token.attempt]_vars

EventuallyAllTerminal == <>AllTerminal
====
