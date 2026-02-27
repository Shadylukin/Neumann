# Handle Transaction Conflicts

## Goal

Detect and recover from transaction conflicts using semantic conflict
detection, deadlock resolution, and retry logic.

## Conflict Retry with Exponential Backoff

When a transaction is aborted due to a semantic conflict, retry with
exponential backoff:

```rust
// Retry with exponential backoff
let mut attempt = 0;
let max_attempts = 5;

loop {
    let workspace = manager.begin(&store)?;

    // Re-read current state
    let account = store.get("account:1")?;

    // Apply changes
    workspace.add_operation(Transaction::Put {
        key: "account:1".to_string(),
        data: serialize(&Account {
            balance: account.balance - 200,
        }),
    })?;

    // Try to commit
    match commit_with_conflict_check(&workspace, &manager) {
        Ok(()) => break,
        Err(ConflictError { similarity, .. }) => {
            attempt += 1;
            if attempt >= max_attempts {
                return Err("max retries exceeded");
            }

            // Exponential backoff with jitter
            let backoff = (100 * 2u64.pow(attempt))
                + rand::random::<u64>() % 50;
            std::thread::sleep(Duration::from_millis(backoff));
        }
    }
}
```

## Configure Deadlock Detection

```toml
[deadlock]
detection_interval_ms = 100
max_cycle_length = 10
victim_selection = "youngest"  # or "lowest_priority", "fewest_locks"
```

## Find and Apply Merge Candidates

For orthogonal transactions (non-overlapping writes), merge into a single
block for parallel commit:

```rust
// Find merge candidates
let candidates = manager.find_merge_candidates(
    &tx_a,
    0.1,      // orthogonal threshold
    60_000,   // merge window (60s)
);

if !candidates.is_empty() {
    // Create merged block with multiple transactions
    let mut merged_ops = tx_a.operations();
    for candidate in &candidates {
        merged_ops.extend(candidate.workspace.operations());
    }

    // Compute merged delta
    let merged_delta = tx_a.to_delta_vector();
    for candidate in &candidates {
        merged_delta = merged_delta.add(&candidate.delta);
    }

    // Single block contains both transactions
    let block = Block::new(header, merged_ops);
    chain.append(block)?;

    // Mark all as committed
    tx_a.mark_committed();
    for candidate in candidates {
        candidate.workspace.mark_committed();
    }
}
```

## Further Reading

- [Semantic Conflict Detection](../explanation/semantic-conflict-detection.md)
  -- how the detection algorithm works
- [Distributed Transactions](../explanation/distributed-transactions.md)
  -- 2PC protocol details
