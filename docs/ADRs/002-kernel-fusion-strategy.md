# ADR-002: Kernel Fusion Strategy

## Status
**Accepted** — Implemented in Phase 5

## Context
The Llama-2 transformer layer consists of many small operations that are individually memory-bandwidth-bound on modern GPUs. Each operation reads from and writes to HBM (GPU global memory), but the intermediate results are only used by the immediately following operation. This creates unnecessary bandwidth pressure.

A typical transformer decode step without fusion:
```
RMSNorm:    read(4096) → write(4096)     ← 32 KB round-trip
Q proj:     read(4096) + read(W) → write(4096)
K proj:     read(4096) + read(W) → write(kv_dim)
V proj:     read(4096) + read(W) → write(kv_dim)
SiLU:       read(11008) → write(11008)   ← 88 KB round-trip (ELIMINATED)
Multiply:   read(11008×2) → write(11008) ← 132 KB (FUSED with SiLU)
```

## Decision
Fuse memory-bound adjacent operations into single kernels:

### Implemented Fusions

| Fused Kernel | Operations Merged | Memory Savings |
|---|---|---|
| `fused_silu_mul` | SiLU + element-wise multiply | Eliminates 88 KB intermediate buffer |
| `fused_rmsnorm_linear` | RMSNorm + linear projection | Eliminates normalized buffer (32 KB/token) |
| `fused_residual_rmsnorm` | Residual add + RMSNorm | Eliminates post-residual buffer |
| `quantized_matmul` | Dequantize INT4 + MatMul | 8× less weight bandwidth (in-register dequant) |

### Fusion Guidelines
1. **Only fuse bandwidth-bound ops.** Compute-bound operations (large MatMul) should NOT be fused
2. **Use shared memory** for intermediate values that span the fusion boundary
3. **Maintain numerical equivalence** — fused kernels must match unfused outputs within float tolerance
4. **Each fused kernel has a golden test** comparing fused vs unfused outputs

### Implementation
- All fused kernels in `cuda/fused_kernels.cu`
- 9 golden tests in `tests/test_quantization.cu` validating numerical equivalence

## Consequences
- **Positive:** ~2× less HBM bandwidth for MLP sub-block, reduced kernel launch overhead
- **Negative:** More complex kernel code, harder to debug
- **Mitigation:** Maintained separate unfused kernels for correctness validation

## References
- Dao et al., "FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness" (2022)
- NVIDIA, "CUDA C++ Programming Guide — Kernel Fusion"
