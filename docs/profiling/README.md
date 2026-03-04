# VULCAN Profiling Guide

## Quick Start

### 1. Build with Debug Symbols (for profiling)
```bash
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=RelWithDebInfo -DCMAKE_CUDA_ARCHITECTURES=native
cmake --build . --parallel
```

### 2. Run Benchmarks
```bash
# Kernel micro-benchmarks (outputs JSON)
./benchmarks/bench_inference benchmark_results.json

# Memory allocator profiling
./benchmarks/bench_memory

# Generate plots
python benchmarks/scripts/plot_results.py --results benchmark_results.json --output plots/
```

### 3. NVIDIA Nsight Compute (Kernel-Level)
```bash
# Profile specific kernel
ncu --set full --target-processes all ./benchmarks/bench_inference

# Profile single kernel with metrics
ncu --kernel-name "matmul_tiled_kernel" --metrics \
    sm__throughput.avg.pct_of_peak_sustained_active,\
    dram__throughput.avg.pct_of_peak_sustained_active,\
    l1tex__throughput.avg.pct_of_peak_sustained_active \
    ./benchmarks/bench_inference

# Export to interactive report
ncu -o vulcan_profile --set full ./benchmarks/bench_inference
# Open with: ncu-ui vulcan_profile.ncu-rep
```

### 4. NVIDIA Nsight Systems (System-Level)
```bash
# Full timeline
nsys profile -o vulcan_timeline --trace=cuda,nvtx ./benchmarks/bench_inference

# Open with: nsys-ui vulcan_timeline.nsys-rep
```

## Key Metrics to Optimize

### Occupancy
Target: >50% achieved occupancy per kernel.
```bash
ncu --metrics sm__warps_active.avg.pct_of_peak_sustained_active ./benchmarks/bench_inference
```

If low:
- Reduce register usage (`--maxrregcount=64`)
- Reduce shared memory per block
- Increase block size

### Memory Throughput
Target: >60% of peak HBM bandwidth for memory-bound kernels.
```bash
ncu --metrics dram__throughput.avg.pct_of_peak_sustained_active ./benchmarks/bench_inference
```

If low:
- Ensure coalesced global memory access (128-byte aligned)
- Use shared memory for reuse (already done in matmul, attention)
- Consider vectorized loads (`float4`)

### Compute Throughput
Target: >50% of peak FLOPS for compute-bound kernels (matmul).
```bash
ncu --metrics sm__throughput.avg.pct_of_peak_sustained_active ./benchmarks/bench_inference
```

### Kernel Launch Overhead
Target: <10 μs per kernel launch.
- Use fused kernels to reduce launch count (Phase 5)
- CUDA graphs for fixed-topology launch sequences (future)

## Architecture-Specific Notes

| GPU Architecture | Key Feature | Optimization |
|---|---|---|
| Volta (SM 70) | Tensor Cores | Use WMMA for FP16 MatMul |
| Ampere (SM 80) | Async memcpy | Use `cp.async` for shared memory loads |
| Hopper (SM 90) | TMA, Cluster | Use Thread Block Clusters for attention |

## Profiling Checklist

- [ ] All kernels achieve >50% occupancy
- [ ] MatMul achieves >60% compute throughput
- [ ] Memory-bound kernels achieve >60% bandwidth utilization
- [ ] No unnecessary synchronization points
- [ ] Fused kernels show measurable speedup vs unfused
- [ ] INT4 kernels show bandwidth reduction vs FP32
