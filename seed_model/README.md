# Seed Model: Emergent Geometric Intelligence

Implementation of the Tensor Scaling Theory for neural language modeling.

## Core Insight

Traditional neural networks force information into fixed shapes:
- Fixed embedding dimensions
- Fixed number of layers
- Fixed attention heads
- Padding everywhere

This wastes compute on zeros and imposes arbitrary structure.

**Tensor Scaling Theory**: Let structure emerge from data.

## Principles

### 1. No Forced Zeros

Shape comes from data, not pre-allocated containers.

```
Traditional: [0.5, 0, 0, 0, 0.3, 0, 0, 0, 0, 0.2]  # 7 zeros stored
Native:      indices=[0,4,9], values=[0.5,0.3,0.2]  # Only information
```

### 2. No Arbitrary Constants

Dimensions, depths, and sizes emerge from learning.

```rust
// Wrong: magic numbers
let hidden_dim = 256;  // Why 256?
let num_layers = 12;   // Why 12?

// Right: capacities that the model uses as needed
let dim_capacity = 256;     // Upper bound
let level_capacity = 16;    // Uses what it needs
```

### 3. Adaptive Compute

Terminate on convergence, not after N steps.

```rust
// Wrong: always run N steps
for i in 0..NUM_LAYERS { x = layer(x); }

// Right: stop when sparse
while !is_sparse(residual) {
    residual = vq_level(residual);
}
```

### 4. Codebook as Knowledge

Knowledge stored as discrete geometric points, not distributed weights.

```rust
// Each codebook entry is a "concept"
// Keys: patterns the model recognizes
// Values: patterns the model expresses
// Response = softmax(-distance/temperature) @ values
```

## Architecture

```
tokens -> Embedding -> Encoder -> EmergentVQ -> GRU -> logits
                                     |
                          Level 0: keys/values (temp learned)
                          Level 1: refine residual
                          ...
                          STOP when 90% sparse
```

### What's Emergent

| Property | How It Emerges |
|----------|----------------|
| Depth | Stops when residual is 90% sparse |
| Temperature | Learned parameter per level |
| Effective codebook size | Entries that actually activate |

### What's Capacity (Not Forced)

| Property | Capacity | Used |
|----------|----------|------|
| Levels | 16 | 1-16 based on input |
| Codebook entries | 1024 | Subset activates |
| Dimensions | 256 | All (could be reduced) |

## Training Output

```
TRAINING - Watching for emergence

Ep  1 | B  100/31248 | CE 4.2 | VQ 1.8 | Depth 12.3 | Sparse 45%
Ep  1 | B  200/31248 | CE 3.9 | VQ 1.5 | Depth 10.1 | Sparse 62%
...

>> EPOCH 1
   Loss: 3.2 | VQ: 1.1
   Emergent depth: 8.4 | Sparsity: 78%
   Learned temps: L0:1.02 L1:0.89 L2:0.71 L3:0.58 ...

EMERGENCE SUMMARY

Depth distribution (emergent):
  Depth  4:   120 ( 2.1%) #
  Depth  6:   890 (15.6%) ###
  Depth  8:  2340 (41.0%) ########
  Depth 10:  1890 (33.1%) ######
  Depth 12:   460 ( 8.1%) ##

Per-level emergence:
  Level 0: temp=1.024 (learned), eff_size=847
  Level 1: temp=0.891 (learned), eff_size=612
  Level 2: temp=0.714 (learned), eff_size=389
```

## Key Files

| File | Purpose |
|------|---------|
| `model.rs` | EmergentVQ, VQLevel, SeedModel |
| `corpus.rs` | Text corpus loading (Dolly, TinyStories) |
| `storage.rs` | Bridge to Neumann codebooks |
| `main.rs` | Training loop with emergence tracking |

## Falsifiable Claims

1. **Depth correlates with complexity**: Simple inputs should use fewer levels
2. **Temperature learns meaningful values**: Not random, reflects level's role
3. **Effective codebook size < capacity**: Model doesn't use all entries

## Running

```bash
# Train (10 minute limit by default)
cargo run --release --package seed_model --bin seed_train

# Output shows emergence in real-time
```

## Theory Reference

See the Tensor Scaling Theory document for:
- Formal hypothesis statements
- Experimental protocol
- Falsifiable predictions
- Kill conditions for the theory
