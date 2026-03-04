# ADR-001: CUDA Memory Management Strategy

## Status
**Accepted** — Implemented in Phase 4

## Context
LLM inference requires managing large, dynamically-sized GPU memory allocations (primarily for KV cache). The KV cache grows with sequence length and varies across requests in a serving scenario. Naive allocation leads to:

1. **Fragmentation:** `cudaMalloc`/`cudaFree` cycles create non-contiguous free regions
2. **Over-allocation:** Reserving max sequence length per request wastes VRAM
3. **OOM:** No graceful degradation when VRAM is exhausted

## Decision
We adopt a **paged memory allocator** inspired by vLLM's PagedAttention:

### Design
- **Fixed-size pages:** All GPU memory allocated as 16KB blocks
- **Contiguous pool:** Single `cudaMalloc` carved into N pages — zero fragmentation by construction
- **Block table:** CPU-side mapping from logical blocks → physical GPU pages
- **On-demand allocation:** Pages allocated only as KV cache grows
- **CPU swap:** When GPU pages exhausted, LRU blocks swapped to CPU RAM via `cudaMemcpy`

### Implementation
- `GPUBuffer` (`cuda/memory.cu`): RAII wrapper with move semantics, typed access, ownership release
- `MemoryManager` (`cuda/memory.cu`): Paged allocator with `allocate_block()`, `free_block()`, `swap_to_cpu()`, `swap_to_gpu()`
- `KVCache` (`src/kv_cache.cpp`): Per-layer contiguous K/V buffers with `append()` and `get()` for incremental decode

### Why Paged Over Linear
| Criterion | Linear Allocator | Paged Allocator |
|---|---|---|
| Fragmentation | High after free/realloc | **Zero** (fixed-size blocks) |
| Memory waste | Allocate max seq per request | Grow on demand |
| Multi-request | Poor sharing | Blocks shared/reclaimed |
| Swap support | Complex | **Natural** (page-level granularity) |
| Complexity | Low | Medium |

### Verified
- 17 unit tests covering allocation, deallocation, pool exhaustion, swap round-trips with data integrity
- Memory benchmark (`bench_memory.cpp`) measures alloc latency: ~0.5 μs per page vs ~50 μs per `cudaMalloc`

## Consequences
- **Positive:** Near-zero fragmentation, efficient multi-request VRAM sharing, sub-microsecond page allocation
- **Negative:** Indirection overhead from block table lookup, more complex kernel code
- **Mitigation:** Block table fits in L1 cache; lookup is negligible vs compute cost

## References
- Kwon et al., "Efficient Memory Management for Large Language Model Serving with PagedAttention" (2023)
- NVIDIA, "CUDA Best Practices Guide — Memory Management"
