# TLA+ Formal Verification Specifications

Formal specifications for the core distributed protocols in Neumann's `tensor_chain` crate. These specifications are designed to be model-checked using the TLC model checker.

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

Install the "TLA+" extension by Markus Kuppe from the VS Code marketplace. It provides syntax highlighting, parsing, and integrated TLC model checking.

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

- **Similarity fast-path**: modeled as a non-deterministic oracle. TLC explores both fast-path and full-validation paths, verifying safety holds regardless of which path is taken.
- **Pre-vote protocol**: fully modeled with separate PreVote/PreVoteResponse messages that do not increment terms.
- **Geometric tie-breaking**: modeled as non-deterministic vote granting for equal logs. Since TLC explores all possible choices, safety is verified for all possible tie-breaking outcomes.

**Model parameters** (Raft.cfg):
- 3 nodes, max term 3, max log length 4, 2 values

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
| `NoFalsePositivesSafety` | Invariant | Alive nodes with higher incarnation not permanently dead |
| `EventualConvergence` | Liveness | All alive nodes converge to consistent view |
| `EventualFailureDetection` | Liveness | Crashed nodes eventually detected |

**Model parameters** (Membership.cfg):
- 3 nodes, max incarnation 3, max timestamp 6

### Common.tla -- Shared Operators

Helper operators used across specifications:
- `Quorum(S)`: strict majority computation
- `QuorumSets(S)`: all quorum subsets
- `LogUpToDate`: log comparison for vote granting
- Sequence utilities: `Last`, `Range`, `IsPrefix`

## Running Model Checking

### Command Line

```bash
cd specs/tla

# Check Raft safety properties
java -jar tla2tools.jar -config Raft.cfg Raft.tla

# Check 2PC atomicity
java -jar tla2tools.jar -config TwoPhaseCommit.cfg TwoPhaseCommit.tla

# Check Membership convergence
java -jar tla2tools.jar -config Membership.cfg Membership.tla

# With increased heap for larger models
java -Xmx4g -jar tla2tools.jar -config Raft.cfg Raft.tla

# Run with multiple worker threads
java -jar tla2tools.jar -workers auto -config Raft.cfg Raft.tla
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

```makefile
TLC = java -jar tla2tools.jar
TLC_OPTS = -workers auto -cleanup

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

## Interpreting Results

### Successful Run

```
Model checking completed. No error has been found.
  Estimates of the probability that TLC did not check all reachable states
  because two distinct states had the same fingerprint:
  ...
```

### Invariant Violation

If TLC finds a counterexample, it prints a trace showing the sequence of states leading to the violation. For example:

```
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
