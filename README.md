# VULCAN — Custom C++/CUDA LLM Inference Engine

A production-grade inference engine for large language models, built from scratch in C++ and CUDA. Designed for maximum throughput and minimal memory footprint on NVIDIA GPUs.

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
| **Custom CUDA Kernels** | Hand-written GEMM, FlashAttention v2, fused RMSNorm |
| **Paged KV Cache** | vLLM-inspired memory management for long contexts |
| **INT4 Quantization** | 4-bit weight dequantization in registers |
| **Kernel Fusion** | Fused Norm + MatMul + Bias to reduce kernel launch overhead |
| **RAII Memory** | Zero-leak GPU memory management with `GPUBuffer` wrappers |
| **Golden Tests** | Every kernel validated against PyTorch reference output |

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
├── include/cuda/               # CUDA utility headers
├── src/                        # C++ implementation
├── cuda/                       # CUDA kernel implementations (.cu)
├── tests/                      # GoogleTest unit tests
├── benchmarks/                 # Performance benchmarks
├── tools/                      # Model conversion & inspection utilities
├── docs/ADRs/                  # Architecture Decision Records
└── third_party/                # Minimal external dependencies
```

## Version History

| Version | Description | Date | Status |
|---|---|---|---|
| **V0.1.0** | Initial stable release: full Llama-2 inference engine | 2026-03-04 | **Stable** ✅ |

## System Benchmarks (V0.1.0)

See [docs/profiling/README.md](docs/profiling/README.md) for detailed performance metrics across all 15+ kernels.

| Metric | VULCAN V0.1.0 |
|---|---|
| **Kernels** | 15 Hand-written CUDA kernels |
| **Quantization** | INT4 Group-wise Symmetric |
| **Memory** | Paged KV Cache (vLLM style) |
| **Interface** | C++ API + Python Bindings |


## License

MIT — See [LICENSE](LICENSE).
