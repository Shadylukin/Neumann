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

**Properties verified:**

| Property | Type | What It Means |
|----------|------|---------------|
| `ElectionSafety` | Invariant | At most one leader per term |
| `LogMatching` | Invariant | Same index + term implies same entry |
| `StateMachineSafety` | Invariant | No divergent committed entries |
| `LeaderCompleteness` | Invariant | Committed entries survive leader changes |
| `VoteIntegrity` | Invariant | Each node votes at most once per term |
| `TermMonotonicity` | Temporal | Terms never decrease |

**Result**: 134,469,861 states generated, 18,268,659 distinct
states found, depth 54. Zero errors.

### Two-Phase Commit (TwoPhaseCommit.tla)

Models the 2PC protocol for cross-shard distributed transactions
implemented in `tensor_chain/src/distributed_tx.rs`.

**Properties verified:**

| Property | Type | What It Means |
|----------|------|---------------|
| `Atomicity` | Invariant | All participants commit or all abort |
| `NoOrphanedLocks` | Invariant | Completed transactions release locks |
| `ConsistentDecision` | Invariant | Coordinator decision matches outcomes |
| `VoteIrrevocability` | Invariant | Prepared votes cannot be retracted |
| `DecisionStability` | Temporal | Coordinator decision never changes |

**Result**: 7,582,773 states generated, 2,264,939 distinct states
found, depth 21. Zero errors.

### SWIM Gossip Membership (Membership.tla)

Models the SWIM-based gossip protocol for cluster membership and
failure detection implemented in `tensor_chain/src/gossip.rs` and
`tensor_chain/src/membership.rs`.

**Properties verified:**

| Property | Type | What It Means |
|----------|------|---------------|
| `NoFalsePositivesSafety` | Invariant | No node marked Failed above its own incarnation |
| `MonotonicEpochs` | Temporal | Lamport timestamps never decrease |
| `MonotonicIncarnations` | Temporal | Incarnation numbers never decrease |

**Result**: 136,097 states generated, 54,148 distinct states found,
depth 17. Zero errors.

## Bugs Found by Model Checking

TLC discovered real protocol bugs that would be extremely difficult
to find through testing alone:

1. **Self-message processing** (Raft): A leader could receive and
   process its own `AppendEntries` heartbeat, truncating its own
   log. This race condition is nearly impossible to trigger in
   integration tests because the Rust implementation uses separate
   channels, but the protocol-level vulnerability was real.

2. **Out-of-order AppendEntries** (Raft): When messages arrive out
   of order, a stale `AppendEntries` with fewer entries could
   overwrite entries from a newer message. Fixed by implementing
   Raft paper Section 5.3 conflict-resolution: only truncate on
   actual term conflicts, not unconditionally.

3. **Heartbeat log wipe** (Raft): Empty heartbeat messages with
   `prevLogIndex = 0` computed an empty new log, destroying all
   committed entries. Fixed by gating log updates on non-empty
   entry lists.

4. **Gossip fairness formula** (Membership): The temporal fairness
   property quantified over `messages` (a state variable) inside
   `WF_vars`, which is semantically invalid in TLA+. This meant
   the liveness properties were vacuously satisfied.

## How to Run

```bash
cd specs/tla

# Requires Java 11+ (tested with OpenJDK 21)
# Download tla2tools.jar from:
#   https://github.com/tlaplus/tlaplus/releases

java -XX:+UseParallelGC -Xmx4g -jar tla2tools.jar \
  -deadlock -workers auto -config Raft.cfg Raft.tla
```

The `-deadlock` flag suppresses false deadlock reports on terminal
states in bounded models. The `-workers auto` flag enables
multi-threaded checking. Full results are saved in
`specs/tla/tlc-results/`.

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

- [specs/tla/README.md](https://github.com/Shadylukin/Neumann/blob/main/specs/tla/README.md) for full
  specification documentation, model parameters, and source code
  mapping
- [Consensus Protocols](consensus-protocols.md) for Raft and SWIM
  protocol details
- [Distributed Transactions](distributed-transactions.md) for 2PC
  protocol details
