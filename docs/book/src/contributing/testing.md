# Testing

## Test Philosophy

- Test the public API, not implementation details
- Include edge cases: empty inputs, boundaries, error conditions
- Performance tests for operations that must scale (10k+ entities)
- Concurrent tests for thread-safe code

## Running Tests

```bash
# All tests
cargo test

# Specific crate
cargo test -p tensor_chain

# Specific test
cargo test test_raft_election

# With output
cargo test -- --nocapture

# Run ignored tests (slow/integration)
cargo test -- --ignored
```

## Test Organization

Unit tests live in the same file:

```rust
pub fn process(data: &str) -> Result<Output> {
    // implementation
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_process_valid_input() {
        let result = process("valid").unwrap();
        assert_eq!(result.status, "ok");
    }

    #[test]
    fn test_process_empty_input() {
        let result = process("");
        assert!(result.is_err());
    }
}
```

## Test Naming

Use the pattern: `test_<function>_<scenario>_<expected>`

```rust
#[test]
fn test_insert_duplicate_key_returns_error() { }

#[test]
fn test_search_empty_index_returns_empty() { }

#[test]
fn test_commit_after_abort_fails() { }
```

## Concurrent Tests

For thread-safe code:

```rust
#[test]
fn test_store_concurrent_writes() {
    let store = Arc::new(TensorStore::new());
    let handles: Vec<_> = (0..10)
        .map(|i| {
            let store = Arc::clone(&store);
            std::thread::spawn(move || {
                for j in 0..1000 {
                    store.put(format!("key_{i}_{j}"), data.clone());
                }
            })
        })
        .collect();

    for h in handles {
        h.join().unwrap();
    }

    assert_eq!(store.len(), 10000);
}
```

## Performance Tests

Mark slow tests with `#[ignore]`:

```rust
#[test]
#[ignore]
fn test_hnsw_search_10k_vectors() {
    let mut index = HNSWIndex::new(config);
    for i in 0..10_000 {
        index.insert(format!("vec_{i}"), random_vector(128));
    }

    let start = Instant::now();
    for _ in 0..100 {
        index.search(&query, 10);
    }
    let elapsed = start.elapsed();

    assert!(elapsed < Duration::from_secs(1));
}
```

Run with: `cargo test -- --ignored`

## Integration Tests

Located in `integration_tests/`:

```bash
cargo test -p integration_tests
```

These test cross-crate behavior and full workflows.

## Coverage

Check coverage with cargo-llvm-cov:

```bash
cargo install cargo-llvm-cov
cargo llvm-cov --workspace --html
open target/llvm-cov/html/index.html
```

Target coverage thresholds:

- shell: 88%
- parser: 91%
- blob: 91%
- router: 92%
- chain: 95%

## Model Checking (TLA+)

Distributed protocol changes must be verified against the TLA+
specifications in `specs/tla/`:

```bash
cd specs/tla

# Run TLC on all three specs
java -XX:+UseParallelGC -Xmx4g -jar tla2tools.jar \
  -deadlock -workers auto -config Raft.cfg Raft.tla

java -XX:+UseParallelGC -Xmx4g -jar tla2tools.jar \
  -deadlock -workers auto \
  -config TwoPhaseCommit.cfg TwoPhaseCommit.tla

java -XX:+UseParallelGC -Xmx4g -jar tla2tools.jar \
  -deadlock -workers auto \
  -config Membership.cfg Membership.tla
```

When modifying Raft, 2PC, or gossip protocols:

1. Update the corresponding `.tla` spec
2. Run TLC and verify zero errors
3. Save output to `specs/tla/tlc-results/`

See [Formal Verification](../concepts/formal-verification.md) for
background on what model checking covers.

## Mocking

Use trait objects for dependency injection:

```rust
pub trait Transport: Send + Sync {
    fn send(&self, msg: Message) -> Result<()>;
}

// In tests
struct MockTransport {
    sent: Mutex<Vec<Message>>,
}

impl Transport for MockTransport {
    fn send(&self, msg: Message) -> Result<()> {
        self.sent.lock().unwrap().push(msg);
        Ok(())
    }
}
```
