# Semantic Conflict Detection

Neumann uses delta embedding similarity to detect conflicts between
concurrent transactions without requiring explicit lock declarations.

## How It Works

Each transaction computes a delta embedding that encodes its write set.
When two transactions attempt to commit concurrently, the consensus manager
compares their delta embeddings using cosine similarity to determine whether
they conflict.

### Conflict Detection Flow

```mermaid
sequenceDiagram
    participant A as Transaction A
    participant B as Transaction B
    participant CM as ConsensusManager

    A->>A: compute_delta()
    Note over A: delta_a = [0.8, 0.2, 0.0, 0.0]
    B->>B: compute_delta()
    Note over B: delta_b = [0.9, 0.0, 0.1, 0.0]

    A->>CM: prepare(delta_a)
    B->>CM: prepare(delta_b)

    CM->>CM: cosine_similarity(delta_a, delta_b)
    Note over CM: similarity = 0.72 (HIGH)

    CM->>A: Vote::Yes
    CM->>B: Vote::Conflict(similarity=0.72, tx=A)

    A->>A: commit()
    B->>B: abort() + retry
```

### Classification Table

| Similarity Range | Classification | Action |
| --- | --- | --- |
| 0.0 - 0.1 | Orthogonal | Parallel commit OK |
| 0.1 - 0.5 | Low overlap | Merge possible |
| 0.5 - 0.9 | Conflicting | Serialize execution |
| 0.9 - 1.0 | Parallel | Abort one |

## Deadlock Detection

When two transactions wait on each other's locks, a cycle forms in the
wait-for graph.

### Wait-For Graph

```mermaid
flowchart LR
    T1((T1)) -->|waits for key_B| T2((T2))
    T2 -->|waits for key_A| T1
```

### Detection Flow

```mermaid
sequenceDiagram
    participant T1 as Transaction 1
    participant T2 as Transaction 2
    participant LM as LockManager
    participant WG as WaitForGraph
    participant DD as DeadlockDetector

    T1->>LM: try_lock(key_A)
    LM->>T1: Ok(handle_1)

    T2->>LM: try_lock(key_B)
    LM->>T2: Ok(handle_2)

    T1->>LM: try_lock(key_B)
    LM->>WG: add_wait(T1, T2)
    LM->>T1: Err(blocked by T2)

    T2->>LM: try_lock(key_A)
    LM->>WG: add_wait(T2, T1)
    LM->>T2: Err(blocked by T1)

    DD->>WG: detect_cycle()
    WG->>DD: Some([T1, T2])

    DD->>DD: select_victim(T2)
    DD->>T2: abort()
    DD->>LM: release(T2)
    DD->>WG: remove(T2)

    T1->>LM: try_lock(key_B)
    LM->>T1: Ok(handle_3)
```

### Victim Selection

| Criterion | Weight | Description |
| --- | --- | --- |
| Lock count | 0.3 | Fewer locks = preferred victim |
| Transaction age | 0.3 | Younger = preferred victim |
| Priority | 0.4 | Lower priority = preferred victim |

## Orthogonal Transaction Merging

Transactions with orthogonal delta embeddings (similarity near 0.0) can be
committed in parallel by merging into a single block.

### Parallel Commit

```mermaid
sequenceDiagram
    participant A as Transaction A
    participant B as Transaction B
    participant CM as ConsensusManager
    participant C as Chain

    par Prepare Phase
        A->>CM: prepare(delta_a)
        B->>CM: prepare(delta_b)
    end

    CM->>CM: similarity = 0.0 (ORTHOGONAL)

    par Commit Phase
        CM->>A: Vote::Yes
        CM->>B: Vote::Yes
        A->>C: append(block_a)
        B->>C: append(block_b)
    end

    Note over C: Both blocks committed
```

### Orthogonality Analysis

| Transaction A | Transaction B | Overlap | Similarity | Can Merge? |
| --- | --- | --- | --- | --- |
| user:1:prefs | product:42:stock | None | 0.00 | Yes |
| user:1:balance | user:2:balance | None | 0.15 | Yes |
| user:1:balance | user:1:prefs | user:1 | 0.30 | Maybe |
| account:1 | account:1 | Full | 0.95 | No |

## Summary

| Scenario | Detection Method | Resolution |
| --- | --- | --- |
| Conflict | Delta similarity > 0.5 | Serialize, retry loser |
| Deadlock | Wait-for graph cycle | Abort victim, retry |
| Orthogonal | Delta similarity < 0.1 | Parallel commit/merge |

## Further Reading

- [Embedding State Machine](embedding-state.md)
- [Transaction Workspace](transaction-workspace.md)
- [Distributed Transactions](distributed-transactions.md)
