# TLA+ Formal Verification Specifications

Formal specifications for the core distributed protocols in Neumann's
`tensor_chain` crate. These specifications are designed to be
model-checked using the TLC model checker.

## Prerequisites

### Option 1: TLA+ Toolbox (GUI)

Download from https://lamport.azurewebsites.net/tla/toolbox.html

The Toolbox provides an integrated environment for editing TLA+ specs, configuring models, and running the TLC model checker.

### Option 2: Command-line TLC (tla2tools.jar)

```bash
# Download tla2tools.jar
curl -LO https://github.com/tlaplus/tlaplus/releases/latest/download/tla2tools.jar

# Or via Homebrew
brew install tlaplus
```

### Option 3: VS Code Extension

Install the "TLA+" extension by Markus Kuppe from the VS Code
marketplace. It provides syntax highlighting, parsing, and integrated
TLC model checking.

## Specifications

### Raft.tla -- Tensor-Raft Consensus

Models the Raft consensus protocol with Neumann's tensor-native extensions, as implemented in `tensor_chain/src/raft.rs`.

**What it verifies:**

| Property | Type | Description |
|----------|------|-------------|
| `ElectionSafety` | Invariant | At most one leader per term |
| `LogMatching` | Invariant | Same index + term implies same entry |
| `StateMachineSafety` | Invariant | No divergent committed entries |
| `LeaderCompleteness` | Invariant | Committed entries survive leader changes |
| `TermMonotonicity` | Temporal | Terms never decrease |
| `PreVoteSafety` | Invariant | Pre-vote does not disrupt existing leaders |
| `FastPathSafety` | Invariant | Fast-path does not violate safety |

**Tensor-Raft extensions modeled:**

- **Similarity fast-path**: modeled as a non-deterministic oracle.
  TLC explores both fast-path and full-validation paths, verifying
  safety holds regardless of which path is taken.
- **Pre-vote protocol**: fully modeled with separate
  PreVote/PreVoteResponse messages that do not increment terms.
- **Geometric tie-breaking**: modeled as non-deterministic vote
  granting for equal logs. Since TLC explores all possible choices,
  safety is verified for all possible tie-breaking outcomes.

**Model parameters** (Raft.cfg):

- 2 nodes, max term 2, max log length 2, 1 value
- PreVote and geometric tie-breaking disabled for tractable
  state space (3 nodes with PreVote exceeds 100M distinct
  states without converging)

### TwoPhaseCommit.tla -- Distributed Transaction Atomicity

Models the two-phase commit protocol for cross-shard distributed transactions, as implemented in `tensor_chain/src/distributed_tx.rs`.

**What it verifies:**

| Property | Type | Description |
|----------|------|-------------|
| `Atomicity` | Invariant | All participants commit or all abort |
| `NoOrphanedLocks` | Invariant | Completed transactions release all locks |
| `ConsistentDecision` | Invariant | Coordinator decision matches outcomes |
| `VoteIrrevocability` | Invariant | Prepared participants cannot unilaterally abort |
| `DecisionStability` | Temporal | Coordinator decision never changes |

**Model parameters** (TwoPhaseCommit.cfg):

- 2 transactions, 3 participants

### Membership.tla -- Gossip Protocol Convergence

Models the SWIM-based gossip protocol for cluster membership and failure detection, as implemented in `tensor_chain/src/gossip.rs`.

**What it verifies:**

| Property | Type | Description |
|----------|------|-------------|
| `MonotonicEpochs` | Temporal | Lamport timestamps never decrease |
| `MonotonicIncarnations` | Temporal | Node incarnation numbers never decrease |
| `NoFalsePositivesSafety` | Invariant | No node marked Failed at incarnation above its own |

**Model parameters** (Membership.cfg):

- 2 nodes, max incarnation 2, max timestamp 4
- 3 nodes with max timestamp 6 exceeds 100M states; 2 nodes
  is the tractable configuration

### Common.tla -- Shared Operators

Helper operators used across specifications:

- `Quorum(S)`: strict majority computation
- `QuorumSets(S)`: all quorum subsets
- `LogUpToDate`: log comparison for vote granting
- Sequence utilities: `Last`, `Range`, `IsPrefix`

## Model Checking Results

All three specifications have been exhaustively model-checked with
TLC. Full output is saved in `tlc-results/`.

| Spec | States Generated | Distinct States | Depth | Time |
|------|-----------------|-----------------|-------|------|
| Raft | 134,469,861 | 18,268,659 | 54 | 38 min |
| TwoPhaseCommit | 7,582,773 | 2,264,939 | 21 | 67s |
| Membership | 136,097 | 54,148 | 17 | 2s |

All runs completed with **zero errors** and **zero states left on
queue** (exhaustive exploration).

### Bugs Found and Fixed by TLC

Model checking discovered and led to fixes for real spec bugs:

**Raft.tla:**

- **Self-message processing**: A leader could process its own
  `AppendEntries` heartbeat, transitioning to Follower and
  truncating its log. Fixed by adding `m.mleaderId /= n` guard
  to `HandleAppendEntries`.
- **Heartbeat log truncation**: Empty heartbeat messages
  (`mentries = <<>>`) with `prevLogIndex = 0` computed a new
  empty log, wiping committed entries. Fixed by gating log
  updates on `Len(m.mentries) > 0`.
- **Out-of-order AppendEntries**: Stale `AppendEntries` messages
  from the same leader could truncate entries appended by newer
  messages. Fixed by implementing proper Raft Section 5.3
  conflict-resolution: only truncate on actual conflicts
  (same index, different term), and append only truly new
  entries.
- **LeaderCompleteness scope**: A stale leader at a lower term
  does not violate completeness (it will step down on
  discovering the higher term). Fixed by scoping the invariant
  to leaders with term strictly greater than the committed
  entry's term.

**Membership.tla:**

- **Forward declaration**: `Discard(m)` operator was used before
  its definition; TLA+ requires forward declaration. Moved
  definition above first use.
- **Fairness formula**: The temporal `Fairness` property
  quantified over `messages` (a variable) inside `WF_vars`,
  which is invalid. Fixed by using existential quantification
  (`\E m \in messages :`) inside `WF_vars`.
- **NoFalsePositivesSafety**: The original invariant was too
  strong for an asynchronous gossip system where failure
  detection can race with refutation messages in flight.
  Rewritten to the correct safety property: no node is ever
  marked Failed at an incarnation strictly higher than its own
  self-view incarnation.

### State Space Notes

The set-based message model (messages as a set of records) causes
combinatorial explosion with more nodes. Practical bounds:

- **Raft**: 2 nodes is tractable (~18M distinct states, 38 min).
  3 nodes with MaxTerm=2, MaxLogLen=2 exceeds 19M distinct states
  with a still-growing queue after 9 minutes.
- **TwoPhaseCommit**: 2 transactions, 3 participants is tractable
  (~2.3M distinct states, 67s).
- **Membership**: 2 nodes with MaxTimestamp=4 is tractable (~54K
  distinct states, 2s). 3 nodes with MaxTimestamp=6 exceeds
  100M states.

## Running Model Checking

### Command Line

```bash
cd specs/tla

# Check Raft (use -deadlock to suppress false deadlock
# reports on terminal states in bounded models)
java -XX:+UseParallelGC -Xmx4g -jar tla2tools.jar \
  -deadlock -workers auto -config Raft.cfg Raft.tla

# Check 2PC
java -XX:+UseParallelGC -Xmx4g -jar tla2tools.jar \
  -deadlock -workers auto \
  -config TwoPhaseCommit.cfg TwoPhaseCommit.tla

# Check Membership
java -XX:+UseParallelGC -Xmx4g -jar tla2tools.jar \
  -deadlock -workers auto \
  -config Membership.cfg Membership.tla
```

### TLA+ Toolbox

1. Open the Toolbox and create a new spec pointing to the `.tla` file
2. Create a new model (Model -> New Model)
3. Set constants as defined in the corresponding `.cfg` file
4. Add invariants and properties from the cfg file
5. Run TLC (TLC Model Checker -> Run)

### Using the Makefile

```bash
cd specs/tla

# Check all specs
make all

# Check individual specs
make raft
make tpc
make membership

# Clean TLC output
make clean
```

## Makefile

Create a `Makefile` in this directory for convenience:

<!-- markdownlint-disable MD010 -->

```makefile
TLC = java -XX:+UseParallelGC -Xmx4g -jar tla2tools.jar
TLC_OPTS = -deadlock -workers auto -cleanup

.PHONY: all raft tpc membership clean

all: raft tpc membership

raft:
	$(TLC) $(TLC_OPTS) -config Raft.cfg Raft.tla

tpc:
	$(TLC) $(TLC_OPTS) -config TwoPhaseCommit.cfg TwoPhaseCommit.tla

membership:
	$(TLC) $(TLC_OPTS) -config Membership.cfg Membership.tla

clean:
	rm -rf states/ *.dot
```

<!-- markdownlint-enable MD010 -->

## Interpreting Results

### Successful Run

```text
Model checking completed. No error has been found.
  Estimates of the probability that TLC did not check all reachable states
  because two distinct states had the same fingerprint:
  ...
```

### Invariant Violation

If TLC finds a counterexample, it prints a trace showing the sequence of states leading to the violation. For example:

```text
Error: Invariant ElectionSafety is violated.
Error: The behavior up to this point is:
State 1: ...
State 2: ...
```

This trace identifies the exact sequence of actions that violates the property, which maps directly to code paths in `tensor_chain/src/raft.rs`.

## Mapping Specs to Source Code

| TLA+ Module | Rust Source |
|-------------|-------------|
| `Raft.tla` | `tensor_chain/src/raft.rs` |
| `TwoPhaseCommit.tla` | `tensor_chain/src/distributed_tx.rs` |
| `Membership.tla` | `tensor_chain/src/gossip.rs`, `tensor_chain/src/membership.rs` |

| TLA+ Action | Rust Function |
|-------------|---------------|
| `StartElection` | `RaftNode::start_election()` |
| `HandleRequestVote` | `RaftNode::handle_request_vote()` |
| `BecomeLeader` | `RaftNode::become_leader()` |
| `AppendEntries` | Heartbeat / replication logic |
| `HandleAppendEntries` | `RaftNode::handle_append_entries()` |
| `AdvanceCommitIndex` | `RaftNode::try_advance_commit_index()` |
| `CoordinatorPrepare` | `DistributedTxCoordinator::begin_transaction()` |
| `ParticipantVoteYes` | Prepare handler, `LockManager::try_lock()` |
| `CoordinatorDecideCommit` | `DistributedTxCoordinator::record_vote()` |
| `GossipExchange` | `GossipProtocol::gossip_round()` |
| `RefuteSuspicion` | `LWWMembershipState::refute()` |
| `FailureDetection` | `LWWMembershipState::fail()` |

## Extending the Specs

When modifying the distributed protocols in `tensor_chain`, update the corresponding TLA+ spec:

1. Add new actions modeling the new behavior
2. Verify existing invariants still hold
3. Add new invariants if the change introduces new safety requirements
4. Run TLC to model-check before merging

### Adding a New Spec

1. Create `NewProtocol.tla` with the MODULE/EXTENDS/VARIABLES/Init/Next/Spec pattern
2. Create `NewProtocol.cfg` with CONSTANTS, SPECIFICATION, and INVARIANTS
3. Add a target to the Makefile
4. Document it in this README
