# ADR-003: Quantization Format

## Status
**Accepted** — Implemented in Phase 5

## Context
Llama-2-7B has ~7 billion parameters. In FP32, this requires ~28 GB of VRAM — exceeding most consumer GPUs. Even FP16 (14 GB) is tight. INT4 quantization reduces weight storage to ~3.5 GB, enabling inference on 8 GB GPUs.

## Decision
Adopt **group-wise symmetric INT4 quantization** with the following format:

### Format
- **Packing:** 2 INT4 values per `uint8_t` byte
  - `low_nibble  = byte & 0x0F` → values 0-15, mapped to signed -8 to +7
  - `high_nibble = byte >> 4` → values 0-15, mapped to signed -8 to +7
- **Scale:** One `float32` scale factor per group
  - `scale = max(|group|) / 7` (symmetric around zero)
- **Group size:** 128 elements (configurable, 64-256 supported)
- **Dequantization:** `float_val = (int4_val - 8) * scale`

### Compression Ratio
```
FP32:  4 bytes/element
INT4:  0.5 bytes/element + 4 bytes/128 elements (scale) ≈ 0.53 bytes/element
Ratio: ~7.5×
```

### Which Tensors Are Quantized
| Tensor Type | Quantized? | Reason |
|---|---|---|
| Weight matrices (QKV, O, MLP) | **Yes** | 95%+ of model parameters |
| RMSNorm weights | No | Small (dim floats per layer), sensitive |
| Embedding table | No | Lookup, not compute-bound |
| Biases | No | Very small count |

### Implementation
- **Dequant kernel** (`cuda/quantization.cu`): GPU kernel that unpacks INT4 and applies scale
- **Quantized MatMul** (`cuda/fused_kernels.cu`): Dequantizes INT4 weights in-register during MatMul — never writes FP32 weights to HBM, saving 8× weight bandwidth
- **Quantizer tool** (`tools/quantize_weights.py`): Converts FP32 model → INT4 format with per-tensor error reporting

### Verified
- 3 golden tests for INT4 dequantization (known values, full groups, multi-group)
- 1 golden test for quantized MatMul
- Python quantizer validates max and mean reconstruction error per tensor

## Consequences
- **Positive:** ~7.5× weight compression, 8× less weight bandwidth, runs on consumer GPUs
- **Negative:** ~0.5-2% accuracy degradation depending on model and task
- **Mitigation:** Group-wise quantization limits error propagation; critical layers can use larger groups

## References
- Dettmers et al., "GPTQ: Accurate Post-Training Quantization for Generative Pre-trained Transformers" (2023)
- Frantar et al., "AWQ: Activation-aware Weight Quantization" (2023)
- NVIDIA, "Tensor Core INT4 Compute" (Hopper architecture)
