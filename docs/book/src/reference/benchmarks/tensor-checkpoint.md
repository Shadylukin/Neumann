# tensor_checkpoint Benchmarks

The tensor_checkpoint crate provides atomic snapshot/restore with retention
policies and confirmation handling.

## Overview

Checkpoint benchmarks measure snapshot creation, rollback, listing, and
retention enforcement across varying store sizes.

<!-- BENCH:START -->
## Expected Performance Characteristics

| Operation | Complexity | Notes |
| --- | --- | --- |
| Create checkpoint | O(keys) | Serializes full store state |
| Rollback | O(keys) | Restores store from snapshot |
| List checkpoints | O(n) | Reads checkpoint metadata |
| Delete checkpoint | O(1) | Removes blob entry |
| Retention enforce | O(excess) | Deletes oldest beyond limit |
<!-- BENCH:END -->

## Benchmarking Checkpoint Operations

```bash
cargo bench --package tensor_checkpoint
```
