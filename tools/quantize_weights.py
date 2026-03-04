#!/usr/bin/env python3
"""
VULCAN INT4 Weight Quantization Tool

Quantizes FP32/FP16 model weights to INT4 format for inference.
Uses group-wise symmetric quantization per ADR-003.

Format per group of `group_size` elements:
  - 1 float32 scale factor
  - group_size / 2 packed bytes (2 INT4 values per byte)

Usage:
    python quantize_weights.py --input model.vulcan --output model_q4.vulcan
    python quantize_weights.py --input model.vulcan --output model_q4.vulcan --group-size 64
"""

import argparse
import numpy as np
import struct
import os
import sys

def quantize_tensor_int4(tensor: np.ndarray, group_size: int = 128):
    """
    Quantize a float32 tensor to INT4 with group-wise symmetric quantization.

    Returns:
        packed:  uint8 array (N/2 bytes)
        scales:  float32 array (N/group_size scales)
    """
    flat = tensor.flatten().astype(np.float32)
    n = len(flat)

    # Pad to multiple of group_size
    padded_n = ((n + group_size - 1) // group_size) * group_size
    padded = np.zeros(padded_n, dtype=np.float32)
    padded[:n] = flat

    num_groups = padded_n // group_size
    scales = np.zeros(num_groups, dtype=np.float32)
    quantized = np.zeros(padded_n, dtype=np.int8)

    for g in range(num_groups):
        start = g * group_size
        end = start + group_size
        group = padded[start:end]

        # Symmetric: scale = max(abs(group)) / 7
        max_abs = np.max(np.abs(group))
        if max_abs < 1e-10:
            scale = 1e-10  # Avoid division by zero
        else:
            scale = max_abs / 7.0

        scales[g] = scale

        # Quantize: round(x / scale) clamped to [-8, 7], then offset to [0, 15]
        q = np.round(group / scale).astype(np.int8)
        q = np.clip(q, -8, 7)
        quantized[start:end] = q

    # Pack two INT4 values per byte
    unsigned = (quantized + 8).astype(np.uint8)  # Map [-8,7] → [0,15]
    packed_n = padded_n // 2
    packed = np.zeros(packed_n, dtype=np.uint8)
    for i in range(packed_n):
        low  = unsigned[2 * i]
        high = unsigned[2 * i + 1]
        packed[i] = (high << 4) | low

    return packed, scales, n  # Return original n for truncation


def dequantize_tensor_int4(packed: np.ndarray, scales: np.ndarray,
                           n: int, group_size: int = 128):
    """
    Dequantize INT4 packed tensor back to float32.
    Used for verification.
    """
    padded_n = len(packed) * 2
    output = np.zeros(padded_n, dtype=np.float32)

    for i in range(len(packed)):
        byte = packed[i]
        low  = int(byte & 0x0F) - 8
        high = int(byte >> 4) - 8

        idx_low  = 2 * i
        idx_high = 2 * i + 1

        group_low  = idx_low // group_size
        group_high = idx_high // group_size

        output[idx_low]  = low  * scales[group_low]
        output[idx_high] = high * scales[group_high]

    return output[:n]


def quantize_model(input_path: str, output_path: str, group_size: int = 128):
    """Quantize all weight tensors in a VULCAN binary model."""
    print(f"[QUANTIZE] Input:  {input_path}")
    print(f"[QUANTIZE] Output: {output_path}")
    print(f"[QUANTIZE] Group size: {group_size}")
    print()

    with open(input_path, 'rb') as f:
        # Read magic + version
        magic = f.read(4)
        if magic != b'VULC':
            print(f"[ERROR] Invalid magic: {magic} (expected VULC)")
            sys.exit(1)

        version = struct.unpack('<I', f.read(4))[0]
        num_tensors = struct.unpack('<I', f.read(4))[0]

        print(f"[QUANTIZE] Model: v{version}, {num_tensors} tensors")

        tensors = []
        total_orig = 0
        total_quant = 0

        for i in range(num_tensors):
            name_len = struct.unpack('<I', f.read(4))[0]
            name = f.read(name_len).decode('utf-8')
            ndims = struct.unpack('<I', f.read(4))[0]
            shape = []
            for _ in range(ndims):
                shape.append(struct.unpack('<I', f.read(4))[0])

            numel = 1
            for s in shape:
                numel *= s

            data = np.frombuffer(f.read(numel * 4), dtype=np.float32).copy()

            # Only quantize weight matrices (not biases, norms, embeddings)
            should_quantize = (
                'weight' in name and
                len(shape) >= 2 and
                'norm' not in name and
                'embeddings' not in name
            )

            if should_quantize:
                packed, scales, orig_n = quantize_tensor_int4(data, group_size)

                # Verify quantization quality
                reconstructed = dequantize_tensor_int4(packed, scales, orig_n, group_size)
                max_err = np.max(np.abs(data[:orig_n] - reconstructed))
                mean_err = np.mean(np.abs(data[:orig_n] - reconstructed))

                orig_bytes = numel * 4
                quant_bytes = len(packed) + len(scales) * 4
                ratio = orig_bytes / quant_bytes

                print(f"  Q {name}: {shape} → "
                      f"{ratio:.1f}x compression, "
                      f"max_err={max_err:.4f}, mean_err={mean_err:.6f}")

                tensors.append({
                    'name': name, 'shape': shape, 'quantized': True,
                    'packed': packed, 'scales': scales, 'group_size': group_size,
                    'orig_n': orig_n
                })
                total_orig += orig_bytes
                total_quant += quant_bytes
            else:
                print(f"  F {name}: {shape} (kept FP32)")
                tensors.append({
                    'name': name, 'shape': shape, 'quantized': False,
                    'data': data
                })
                total_orig += numel * 4
                total_quant += numel * 4

    # Write quantized model
    with open(output_path, 'wb') as f:
        f.write(b'VLQ4')  # Quantized model magic
        f.write(struct.pack('<I', 1))  # Version
        f.write(struct.pack('<I', len(tensors)))
        f.write(struct.pack('<I', group_size))

        for t in tensors:
            name_bytes = t['name'].encode('utf-8')
            f.write(struct.pack('<I', len(name_bytes)))
            f.write(name_bytes)
            f.write(struct.pack('<I', len(t['shape'])))
            for s in t['shape']:
                f.write(struct.pack('<I', s))
            f.write(struct.pack('<?', t['quantized']))

            if t['quantized']:
                f.write(struct.pack('<I', t['orig_n']))
                f.write(struct.pack('<I', len(t['packed'])))
                f.write(t['packed'].tobytes())
                f.write(struct.pack('<I', len(t['scales'])))
                f.write(t['scales'].tobytes())
            else:
                f.write(t['data'].tobytes())

    print(f"\n[QUANTIZE] Done!")
    print(f"  Original:   {total_orig / (1024*1024):.1f} MB")
    print(f"  Quantized:  {total_quant / (1024*1024):.1f} MB")
    print(f"  Reduction:  {total_orig / total_quant:.1f}x")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="VULCAN INT4 Weight Quantizer")
    parser.add_argument("--input", required=True, help="Input VULCAN model (.vulcan)")
    parser.add_argument("--output", required=True, help="Output quantized model (.vulcan)")
    parser.add_argument("--group-size", type=int, default=128,
                        help="Quantization group size (default: 128)")
    args = parser.parse_args()

    quantize_model(args.input, args.output, args.group_size)
