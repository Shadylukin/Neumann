# Formal Verification

Neumann's distributed protocols are formally specified in TLA+ and
exhaustively model-checked with the TLC model checker. The
specifications live in `specs/tla/` and cover the three critical
protocol layers in `tensor_chain`.

## What Is Model Checked

TLC explores every reachable state of a bounded model, checking
safety invariants and temporal properties at each state. Unlike
testing (which samples executions), model checking is exhaustive: if
TLC reports no errors, the properties hold for every possible
interleaving within the model bounds.

### Raft Consensus (Raft.tla)

Models leader election, log replication, and commit advancement for
the Tensor-Raft protocol implemented in `tensor_chain/src/raft.rs`.
Three configurations exercise different aspects of the protocol.

**Properties verified (14):**

| Property | Type | What It Means |
|----------|------|---------------|
| `ElectionSafety` | Invariant | At most one leader per term |
| `LogMatching` | Invariant | Same index + term implies same entry |
| `StateMachineSafety` | Invariant | No divergent committed entries |
| `LeaderCompleteness` | Invariant | Committed entries survive leader changes |
| `VoteIntegrity` | Invariant | Each node votes at most once per term |
| `PreVoteSafety` | Invariant | Pre-vote does not disrupt existing leaders |
| `ReplicationInv` | Invariant | Every committed entry exists on a quorum |
| `TermMonotonicity` | Temporal | Terms never decrease |
| `CommittedLogAppendOnlyProp` | Temporal | Committed entries never retracted |
| `MonotonicCommitIndexProp` | Temporal | commitIndex never decreases |
| `MonotonicMatchIndexProp` | Temporal | matchIndex monotonic per leader term |
| `NeverCommitEntryPrevTermsProp` | Temporal | Only current-term entries committed |
| `StateTransitionsProp` | Temporal | Valid state machine transitions |
| `PermittedLogChangesProp` | Temporal | Log changes only via valid paths |

**Result** (Raft.cfg, 3 nodes): 6,641,341 states generated,
1,338,669 distinct states, depth 42, 2 min 24s. Zero errors.

### Two-Phase Commit (TwoPhaseCommit.tla)

Models the 2PC protocol for cross-shard distributed transactions
implemented in `tensor_chain/src/distributed_tx.rs`. Includes a
fault model with message loss and participant crash/recovery.

**Properties verified (6):**

| Property | Type | What It Means |
|----------|------|---------------|
| `Atomicity` | Invariant | All participants commit or all abort |
| `NoOrphanedLocks` | Invariant | Completed transactions release locks |
| `ConsistentDecision` | Invariant | Coordinator decision matches outcomes |
| `VoteIrrevocability` | Temporal | Prepared votes cannot be retracted without coordinator |
| `DecisionStability` | Temporal | Coordinator decision never changes |

**Fault model:** `DropMessage` (network loss) and
`ParticipantRestart` (crash with WAL-backed lock recovery).

**Result**: 1,869,429,350 states generated, 190,170,601 distinct
states, depth 29, 2 hr 55 min. Zero errors. Every reachable state
under message loss and crash/recovery satisfies all properties.

### SWIM Gossip Membership (Membership.tla)

Models the SWIM-based gossip protocol for cluster membership and
failure detection implemented in `tensor_chain/src/gossip.rs` and
`tensor_chain/src/membership.rs`.

**Properties verified (3):**

| Property | Type | What It Means |
|----------|------|---------------|
| `NoFalsePositivesSafety` | Invariant | No node marked Failed above its own incarnation |
| `MonotonicEpochs` | Temporal | Lamport timestamps never decrease |
| `MonotonicIncarnations` | Temporal | Incarnation numbers never decrease |

**Result** (2-node): 136,097 states generated, 54,148 distinct
states, depth 17. Zero errors.
**Result** (3-node): 16,513 states generated, 5,992 distinct
states, depth 13. Zero errors.

## Bugs Found by Model Checking

TLC discovered real protocol bugs that would be extremely difficult
to find through testing alone:

1. **matchIndex response reporting** (Raft): Follower reported
   `matchIndex = Len(log)` instead of `prevLogIndex + Len(entries)`.
   A heartbeat response would falsely claim the full log matched
   the leader's, enabling incorrect commits. Caught by
   `ReplicationInv`.

2. **Out-of-order matchIndex regression** (Raft): Leader
   unconditionally set `matchIndex` from responses. A stale
   heartbeat response arriving after a replication response would
   regress the value. Fixed by taking the max. Caught by
   `MonotonicMatchIndexProp`.

3. **inPreVote not reset on step-down** (Raft): When stepping down
   to a higher term, the `inPreVote` flag was not cleared. A node
   could remain in pre-vote state as a Follower. Caught by
   `PreVoteSafety`.

4. **Self-message processing** (Raft): A leader could process its
   own `AppendEntries` heartbeat, truncating its own log.

5. **Heartbeat log wipe** (Raft): Empty heartbeat messages with
   `prevLogIndex = 0` computed an empty new log, destroying
   committed entries.

6. **Out-of-order AppendEntries** (Raft): Stale messages could
   overwrite entries from newer messages. Fixed with proper Raft
   Section 5.3 conflict-resolution.

7. **Gossip fairness formula** (Membership): Quantification over
   `messages` (a state variable) inside `WF_vars` is semantically
   invalid in TLA+.

## How to Run

```bash
cd specs/tla

# Fast CI check (~3 minutes total)
make ci

# All configs including extensions
make all

# Individual specs
java -XX:+UseParallelGC -Xmx4g -jar tla2tools.jar \
  -deadlock -workers auto -config Raft.cfg Raft.tla
```

The `-deadlock` flag suppresses false deadlock reports on terminal
states in bounded models. The `-workers auto` flag enables
multi-threaded checking.

## Relationship to Testing

| Technique | Coverage | Finds |
|-----------|----------|-------|
| Unit tests | Specific scenarios | Implementation bugs |
| Integration tests | Cross-crate workflows | Wiring bugs |
| Fuzz testing | Random inputs | Crash/panic bugs |
| **Model checking** | **All interleavings** | **Protocol design bugs** |

Model checking complements testing. It verifies the protocol design
is correct (no possible interleaving violates safety), while tests
verify the Rust implementation matches the design. Together they
provide high confidence that the distributed protocols behave
correctly.

## Further Reading

- [specs/tla/README.md](https://github.com/Shadylukin/Neumann/blob/main/specs/tla/README.md)
  for full specification documentation and source code mapping
- [Consensus Protocols](consensus-protocols.md) for Raft and SWIM
  protocol details
- [Distributed Transactions](distributed-transactions.md) for 2PC
  protocol details
