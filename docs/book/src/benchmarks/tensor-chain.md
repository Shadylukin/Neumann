# tensor_chain Benchmarks

The tensor_chain crate provides a tensor-native blockchain with semantic
consensus, Raft replication, 2PC distributed transactions, and sparse delta
encoding.

## Block Creation

| Configuration | Time | Per Transaction |
| --- | --- | --- |
| empty_block | 171 ns | --- |
| block_10_txns | 13.4 us | 1.34 us |
| block_100_txns | 111 us | 1.11 us |

## Transaction Commit

| Operation | Time | Throughput |
| --- | --- | --- |
| single_put | 432 us | 2.3K/s |
| multi_put_10 | 480 us | 20.8K ops/s |

## Batch Transactions

| Count | Time | Throughput |
| --- | --- | --- |
| 10 | 822 us | 12.2K/s |
| 100 | 21.5 ms | 4.7K/s |
| 1000 | 1.6 s | 607/s |

## Consensus Validation

| Operation | Time | Notes |
| --- | --- | --- |
| conflict_detection_pair | 279 ns | Hybrid cosine + Jaccard |
| cosine_similarity | 187 ns | Sparse vector |
| merge_pair | 448 ns | Orthogonal merge |
| merge_all_10 | 632 ns | Batch merge |
| find_merge_order_10 | 9 us | Optimal ordering |

## Codebook Operations

| Operation | Time | Notes |
| --- | --- | --- |
| global_quantize_128d | 854 ns | State validation |
| global_compute_residual | 925 ns | Delta compression |
| global_is_valid_state | 1.28 us | State machine check |
| local_quantize_128d | 145 ns | EMA-adaptive |
| local_quantize_and_update | 177 ns | With EMA update |
| manager_quantize_128d | 1.2 us | Full pipeline |

## Delta Vector Operations

| Operation | Time | Improvement |
| --- | --- | --- |
| cosine_similarity_128d | 196 ns | 35% faster |
| add_128d | 975 ns | 44% faster |
| scale_128d | 163 ns | 35% faster |
| weighted_average_128d | 982 ns | 26% faster |
| overlaps_with | 8.4 ns | 35% faster |
| cosine_similarity_768d | 1.96 us | 10% faster |
| add_768d | 2.6 us | 27% faster |

## Chain Query Operations

| Operation | Time | Improvement |
| --- | --- | --- |
| get_block_by_height | 1.19 us | 38% faster |
| get_tip | 1.06 us | 45% faster |
| get_genesis | 852 ns | 53% faster |
| height | 0.87 ns | 50% faster |
| tip_hash | 11.4 ns | 32% faster |
| history_key | 163 us | 15% faster |
| verify_chain_100_blocks | 276 us | --- |

## Chain Iteration

| Operation | Time | Improvement |
| --- | --- | --- |
| iterate_50_blocks | 88 us | 10% faster |
| get_blocks_range_0_25 | 35 us | 27% faster |

## K-means Codebook Training

| Configuration | Time |
| --- | --- |
| 100 vectors, 8 clusters | 123 us |
| 1000 vectors, 16 clusters | 8.4 ms |

## Sparse Vector Performance

### Conflict Detection by Sparsity Level (50 deltas, 128d)

| Sparsity | Time | Throughput | vs Dense |
| --- | --- | --- | --- |
| 10% (dense) | 389 us | 3.1M pairs/s | 1x |
| 50% | 261 us | 4.6M pairs/s | 1.5x |
| 90% | 57 us | 21.5M pairs/s | **6.8x** |
| 99% | 23 us | 52.3M pairs/s | **16.9x** |

### Individual Sparse Operations (vs previous dense implementation)

| Operation | Sparse Time | Improvement |
| --- | --- | --- |
| cosine_similarity | 16.5 ns | **76% faster** |
| angular_distance | 28.5 ns | **64% faster** |
| jaccard_index | 10.4 ns | **58% faster** |
| euclidean_distance | 13.6 ns | **71% faster** |
| overlapping_keys | 89 ns | **45% faster** |
| add | 688 ns | 19% faster |
| weighted_average | 674 ns | 12% faster |
| project_orthogonal | 624 ns | **42% faster** |
| detect_conflict_full | 53 ns | **33% faster** |

### High Dimension Sparse Performance

| Dimension | Cosine Time | Batch Detect (20 deltas) | Improvement |
| --- | --- | --- | --- |
| 128d | 10.3 ns | 8.9 us | 57% faster |
| 256d | 19 ns | 9.5 us | 55% faster |
| 512d | 41 ns | 17.2 us | **49-75% faster** |
| 768d | 62.5 ns | 24 us | **55-77% faster** |

## Real Transaction Delta Sparsity Analysis

Measurement of actual delta sparsity for different transaction patterns (128d
embeddings):

| Pattern | Avg NNZ | Sparsity | Estimated Speedup |
| --- | --- | --- | --- |
| Single Key Update | 4.0 | 96.9% | ~10x |
| Multi-Field Update | 11.3 | 91.2% | ~3x |
| New Record Insert | 29.5 | 77.0% | ~1x |
| Counter Increment | 1.0 | 99.2% | ~10x |
| Bulk Migration | 59.5 | 53.5% | ~1x |
| Graph Edge | 7.0 | 94.5% | ~3x |

**Realistic Workload Mix** (70% single-key, 20% multi-field, 10% other):

- Average NNZ: 7.1 / 128 dimensions
- Average Sparsity: **94.5%**
- Expected speedup: **3-10x** for typical workloads

## Analysis

- **Sparse advantage**: Real transaction deltas are 90-99% sparse, providing
  3-10x speedup
- **Hybrid conflict detection**: Cosine + Jaccard catches both angular and
  structural conflicts
- **Memory savings**: Sparse DeltaVector uses 8-32x less memory than dense for
  typical deltas
- **Network bandwidth**: Sparse serialization reduces replication bandwidth by
  8-10x
- **High dimension scaling**: Benefits increase with dimension (768d: 4-5x
  faster than dense)
- **Common operations optimized**: Single-key updates (most common) are 96.9%
  sparse

## Distributed Systems Benchmarks

### Raft Consensus Operations

| Operation | Time | Throughput |
| --- | --- | --- |
| raft_node_create | 545 ns | 1.8M/sec |
| raft_become_leader | 195 ns | 5.1M/sec |
| raft_heartbeat_stats_snapshot | 4.2 ns | 238M/sec |
| raft_log_length | 3.7 ns | 270M/sec |
| raft_stats_snapshot | 416 ps | 2.4B/sec |

### 2PC Distributed Transaction Operations

| Operation | Time | Throughput |
| --- | --- | --- |
| lock_manager_acquire | 256 ns | 3.9M/sec |
| lock_manager_release | 139 ns | 7.2M/sec |
| lock_manager_is_locked | 31 ns | 32M/sec |
| coordinator_create | 46 ns | 21.7M/sec |
| coordinator_stats | 418 ps | 2.4B/sec |
| participant_create | 11 ns | 91M/sec |

### Gossip Protocol Operations

| Operation | Time | Throughput |
| --- | --- | --- |
| lww_state_create | 4.2 ns | 238M/sec |
| lww_state_merge | 169 ns | 5.9M/sec |
| gossip_node_state_create | 16 ns | 62M/sec |
| gossip_message_serialize | 36 ns | 28M/sec |
| gossip_message_deserialize | 81 ns | 12M/sec |

### Snapshot Operations

| Operation | Time | Throughput |
| --- | --- | --- |
| snapshot_metadata_create | 131 ns | 7.6M/sec |
| snapshot_metadata_serialize | 76 ns | 13M/sec |
| snapshot_metadata_deserialize | 246 ns | 4.1M/sec |
| raft_membership_config_create | 102 ns | 9.8M/sec |
| raft_with_store_create | 948 ns | 1.1M/sec |

### Membership Operations

| Operation | Time | Throughput |
| --- | --- | --- |
| membership_manager_create | 526 ns | 1.9M/sec |
| membership_view | 152 ns | 6.6M/sec |
| membership_partition_status | 19 ns | 52M/sec |
| membership_node_status | 46 ns | 21.7M/sec |
| membership_stats_snapshot | 2.9 ns | 344M/sec |
| membership_peer_ids | 71 ns | 14M/sec |

### Deadlock Detection

| Operation | Time | Throughput |
| --- | --- | --- |
| wait_graph_add_edge | 372 ns | 2.7M/sec |
| wait_graph_detect_no_cycle | 374 ns | 2.7M/sec |
| wait_graph_detect_with_cycle | 302 ns | 3.3M/sec |
| deadlock_detector_detect | 392 ns | 2.6M/sec |

## Distributed Systems Analysis

- **Lock operations are fast**: Lock acquisition at 256ns and lock checks at
  31ns support high-throughput 2PC
- **Gossip is lightweight**: State creation <5ns, merges ~169ns - suitable for
  high-frequency protocol rounds
- **Stats access is near-free**: Sub-nanosecond stats snapshots (416ps) mean
  monitoring adds no overhead
- **Deadlock detection is efficient**: Cycle detection in ~300-400ns allows
  frequent checks without blocking
- **Node/manager creation is slower** (500-950ns) - expected for initialization
  with data structures
- **Snapshot deserialization at 246ns** is acceptable for fast recovery
