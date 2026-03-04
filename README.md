# VULCAN — Custom C++/CUDA LLM Inference Engine

An inference engine for large language models, built from scratch in C++ and CUDA. Designed for maximum throughput and minimal memory footprint on NVIDIA GPUs.

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                      vulcan::Engine                             │
│  ┌──────────┐  ┌──────────────┐  ┌───────────┐  ┌───────────┐ │
│  │  Model    │  │ Transformer  │  │ KV Cache  │  │  Sampler  │ │
│  │  Loader   │→ │    Block     │→ │ (Paged)   │→ │ Top-P/K   │ │
│  └──────────┘  └──────────────┘  └───────────┘  └───────────┘ │
│                       │                                         │
│        ┌──────────────┼──────────────┐                         │
│        ▼              ▼              ▼                         │
│  ┌──────────┐  ┌───────────┐  ┌───────────┐                   │
│  │ Attention │  │ LayerNorm │  │ Activation │    CUDA Kernels  │
│  │ (Flash)   │  │ (RMSNorm) │  │ (SiLU)     │                 │
│  └──────────┘  └───────────┘  └───────────┘                   │
│        │              │              │                         │
│        └──────────────┼──────────────┘                         │
│                       ▼                                         │
│              ┌─────────────────┐                               │
│              │  Memory Manager │    Paged GPU Allocator         │
│              │  (RAII Buffers) │                                │
│              └─────────────────┘                               │
└─────────────────────────────────────────────────────────────────┘
```

## Key Features

| Feature | Description |
|---|---|
| **Custom CUDA Kernels** | 15+ hand-written kernels: GEMM, RMSNorm, RoPE, Softmax, SiLU, Attention |
| **Grouped Query Attention (GQA)** | Multi-head and grouped-query attention with causal masking |
| **KV Cache** | Pre-allocated per-layer GPU buffers for incremental decode |
| **Paged Memory Manager** | vLLM-inspired paged GPU allocator with CPU swap support |
| **INT4 Quantization** | Group-wise symmetric 4-bit weight quantization with in-register dequant |
| **Kernel Fusion** | Fused SiLU×Up, RMSNorm+Linear, Residual+RMSNorm, Quantized MatMul |
| **RAII Memory** | Zero-leak GPU memory management with `GPUBuffer` and `Tensor` wrappers |
| **Golden Tests** | Every kernel validated against CPU reference output via GoogleTest |

## Build

### Prerequisites
- CMake 3.18+
- CUDA Toolkit 12.x
- C++17-compatible compiler (GCC 9+, Clang 10+, MSVC 2019+)
- NVIDIA GPU (Volta or newer)

### Compile

```bash
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=native
cmake --build . --parallel $(nproc)
```

### Run Tests

```bash
cd build
ctest --output-on-failure
```

## Project Structure

```
vulcan-inference/
├── CMakeLists.txt              # Build configuration (C++17, CUDA 12)
├── include/vulcan/             # Public C++ API headers
├── include/cuda/               # CUDA utility and kernel declaration headers
├── src/                        # C++ implementation (engine, model, transformer, sampler)
├── cuda/                       # CUDA kernel implementations (.cu)
├── tests/                      # GoogleTest unit tests (50+ cases)
├── benchmarks/                 # Performance benchmarks (inference, memory)
├── tools/                      # Model conversion, quantization & inspection utilities
├── docs/ADRs/                  # Architecture Decision Records (memory, fusion, quantization)
├── docs/profiling/             # Kernel performance metrics and profiling results
└── third_party/                # Minimal external dependencies (pybind11)
```

## Version History

| Version | Description | Date | Status |
|---|---|---|---|
| **V0.1.0** | Initial stable release: full Llama-2 inference engine | 2026-03-04 | **Stable** ✅ |

## System Benchmarks (V0.1.0)

See [docs/profiling/README.md](docs/profiling/README.md) for detailed performance metrics across all 15+ kernels.

| Metric | VULCAN V0.1.0 |
|---|---|
| **Kernels** | 15+ Hand-written CUDA kernels |
| **Attention** | Causal MHA/GQA with KV Cache |
| **Quantization** | INT4 Group-wise Symmetric |
| **Memory** | Pre-allocated KV Cache + Paged Allocator |
| **Fusion** | 4 fused kernels (SiLU×Up, RMSNorm+Linear, Residual+RMSNorm, QuantMatMul) |
| **Interface** | C++ API + Python Bindings (pybind11) |


## License

MIT — See [LICENSE](LICENSE).
