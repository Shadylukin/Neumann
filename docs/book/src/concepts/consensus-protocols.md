# Consensus Protocols

tensor_chain uses Raft consensus with SWIM gossip for membership management.

## Raft Consensus

### Overview

Raft provides:

- Leader election
- Log replication
- Safety (never returns incorrect results)
- Availability (operational if majority alive)

### Node States

```mermaid
stateDiagram-v2
    [*] --> Follower
    Follower --> Candidate: election timeout
    Candidate --> Leader: wins election
    Candidate --> Follower: discovers leader
    Leader --> Follower: discovers higher term
    Candidate --> Candidate: split vote
```

### Terms

Time divided into terms with at most one leader:

```text
Term 1: [Leader A] -----> [Follower timeout]
Term 2: [Election] -> [Leader B] -----> ...
```

### Log Replication

```mermaid
sequenceDiagram
    participant C as Client
    participant L as Leader
    participant F1 as Follower 1
    participant F2 as Follower 2

    C->>L: Write request
    L->>L: Append to log
    par Replicate
        L->>F1: AppendEntries
        L->>F2: AppendEntries
    end
    F1->>L: Success
    F2->>L: Success
    L->>L: Commit (majority)
    L->>C: Success
```

### Configuration

| Parameter | Default | Description |
| --- | --- | --- |
| `election_timeout_min` | 150ms | Min election timeout |
| `election_timeout_max` | 300ms | Max election timeout |
| `heartbeat_interval` | 50ms | Leader heartbeat frequency |
| `max_entries_per_append` | 100 | Batch size for replication |

## SWIM Gossip

### Overview

Scalable Weakly-consistent Infection-style Membership:

- O(log N) failure detection
- Distributed membership view
- No single point of failure

### Protocol

```mermaid
sequenceDiagram
    participant A as Node A
    participant B as Node B (target)
    participant C as Node C

    A->>B: Ping
    Note over B: No response
    A->>C: PingReq(B)
    C->>B: Ping
    alt B responds
        B->>C: Ack
        C->>A: Ack (indirect)
    else B down
        C->>A: Nack
        A->>A: Mark B suspect
    end
```

### Node States

| State | Description | Transition |
| --- | --- | --- |
| Healthy | Responding normally | --- |
| Suspect | Failed direct ping | After timeout |
| Failed | Confirmed down | After indirect ping failure |

### LWW-CRDT Membership

Last-Writer-Wins with incarnation numbers:

```rust
// State comparison
fn supersedes(&self, other: &Self) -> bool {
    (self.incarnation, self.timestamp) > (other.incarnation, other.timestamp)
}

// Merge takes winner per node
fn merge(&mut self, other: &Self) {
    for (node_id, state) in &other.states {
        if state.supersedes(&self.states[node_id]) {
            self.states.insert(node_id.clone(), state.clone());
        }
    }
}
```

### Configuration

| Parameter | Default | Description |
| --- | --- | --- |
| `ping_interval` | 1s | Direct ping frequency |
| `ping_timeout` | 500ms | Time to wait for response |
| `suspect_timeout` | 3s | Time before marking failed |
| `indirect_ping_count` | 3 | Number of indirect pings |

## Hybrid Logical Clocks

Combine physical time with logical counters:

```rust
pub struct HybridTimestamp {
    wall_ms: u64,    // Physical time (milliseconds)
    logical: u16,    // Logical counter
}
```

### Properties

- Monotonic: Always increases
- Bounded drift: Stays close to wall clock
- Causality: If A happens-before B, then ts(A) < ts(B)

### Usage

```rust
let hlc = HybridLogicalClock::new(node_id);

// Local event
let ts = hlc.now();

// Receive message with timestamp
let ts = hlc.receive(message_ts);
```

## Formal Verification

Both protocols are formally specified in TLA+ and exhaustively
model-checked with TLC:

- **Raft.tla** verifies `ElectionSafety`, `LogMatching`,
  `StateMachineSafety`, `LeaderCompleteness`, `VoteIntegrity`,
  and `TermMonotonicity` across 18.3M distinct states.
- **Membership.tla** verifies `NoFalsePositivesSafety`,
  `MonotonicEpochs`, and `MonotonicIncarnations` across 54K
  distinct states.

Model checking found and led to fixes for protocol bugs including
out-of-order message handling in Raft log replication and an invalid
fairness formula in the gossip spec. See
[Formal Verification](formal-verification.md) for full results.

## Integration

Raft and SWIM work together:

1. **SWIM** detects node failures quickly
2. **Raft** handles leader election and log consistency
3. **HLC** provides ordering across the cluster

```mermaid
flowchart TB
    subgraph Membership Layer
        SWIM[SWIM Gossip]
    end

    subgraph Consensus Layer
        Raft[Raft Consensus]
    end

    subgraph Time Layer
        HLC[Hybrid Logical Clock]
    end

    SWIM -->|failure notifications| Raft
    HLC -->|timestamps| SWIM
    HLC -->|timestamps| Raft
```
