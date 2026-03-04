#!/usr/bin/env python3
"""
VULCAN — PyTorch Reference Generator

Generates reference outputs for VULCAN kernel validation.
Compares PyTorch outputs against VULCAN C++/CUDA kernel outputs
stored in binary test vectors.

Usage:
    python generate_golden_refs.py --output tests/golden/
    python generate_golden_refs.py --compare tests/golden/vulcan_output.bin

This script is the "source of truth" for correctness validation.
If VULCAN kernels disagree with PyTorch, VULCAN has a bug.
"""

import argparse
import numpy as np
import struct
import os
import sys

def generate_matmul_ref(M: int, K: int, N: int, seed: int = 42):
    """Generate MatMul reference: C = A @ B"""
    rng = np.random.RandomState(seed)
    A = rng.uniform(-1.0, 1.0, (M, K)).astype(np.float32)
    B = rng.uniform(-1.0, 1.0, (K, N)).astype(np.float32)
    C = A @ B
    return A, B, C

def generate_silu_ref(n: int, seed: int = 42):
    """Generate SiLU reference: y = x * sigmoid(x)"""
    rng = np.random.RandomState(seed)
    x = rng.uniform(-10.0, 10.0, n).astype(np.float32)
    y = x / (1.0 + np.exp(-x))
    return x, y

def generate_relu_ref(n: int, seed: int = 42):
    """Generate ReLU reference: y = max(0, x)"""
    rng = np.random.RandomState(seed)
    x = rng.uniform(-5.0, 5.0, n).astype(np.float32)
    y = np.maximum(0, x)
    return x, y

def generate_rmsnorm_ref(n: int, eps: float = 1e-5, seed: int = 42):
    """Generate RMSNorm reference: y = x * rsqrt(mean(x^2) + eps) * weight"""
    rng = np.random.RandomState(seed)
    x = rng.uniform(-2.0, 2.0, n).astype(np.float32)
    w = rng.uniform(0.5, 2.0, n).astype(np.float32)

    rms = np.sqrt(np.mean(x ** 2) + eps)
    y = (x / rms) * w
    return x, w, y

def save_tensor(path: str, tensor: np.ndarray):
    """Save tensor as raw binary float32."""
    tensor.astype(np.float32).tofile(path)

def compare_outputs(vulcan_path: str, ref: np.ndarray, name: str, tol: float):
    """Compare VULCAN output against PyTorch reference."""
    vulcan = np.fromfile(vulcan_path, dtype=np.float32)
    if vulcan.shape != ref.shape:
        print(f"  ✗ {name}: Shape mismatch — vulcan={vulcan.shape} ref={ref.shape}")
        return False

    max_err = np.max(np.abs(vulcan - ref))
    mean_err = np.mean(np.abs(vulcan - ref))

    if max_err < tol:
        print(f"  ✓ {name}: PASS  max_err={max_err:.2e}  mean_err={mean_err:.2e}  tol={tol:.0e}")
        return True
    else:
        print(f"  ✗ {name}: FAIL  max_err={max_err:.2e}  mean_err={mean_err:.2e}  tol={tol:.0e}")
        # Show worst indices
        worst = np.argsort(np.abs(vulcan - ref))[-5:]
        for idx in worst:
            print(f"      idx={idx}: vulcan={vulcan[idx]:.6f} ref={ref[idx]:.6f} "
                  f"diff={abs(vulcan[idx]-ref[idx]):.2e}")
        return False

def main():
    parser = argparse.ArgumentParser(description="VULCAN Golden Reference Generator")
    parser.add_argument("--output", type=str, default="tests/golden/",
                        help="Output directory for reference tensors")
    parser.add_argument("--compare", type=str, default=None,
                        help="Directory with VULCAN output tensors to compare")
    args = parser.parse_args()

    print("╔══════════════════════════════════════════════╗")
    print("║  VULCAN — PyTorch Golden Reference Gen       ║")
    print("╚══════════════════════════════════════════════╝")
    print()

    os.makedirs(args.output, exist_ok=True)

    # ─── Generate References ─────────────────────────────────────────
    print("[GEN] Generating reference outputs...\n")

    # MatMul: 64x64
    A, B, C = generate_matmul_ref(64, 64, 64, seed=100)
    save_tensor(f"{args.output}/matmul_64_A.bin", A)
    save_tensor(f"{args.output}/matmul_64_B.bin", B)
    save_tensor(f"{args.output}/matmul_64_C_ref.bin", C)
    print(f"  MatMul 64x64x64: A={A.shape} B={B.shape} C={C.shape}")

    # MatMul: 128x64x32 (non-square)
    A2, B2, C2 = generate_matmul_ref(128, 64, 32, seed=200)
    save_tensor(f"{args.output}/matmul_128x64x32_A.bin", A2)
    save_tensor(f"{args.output}/matmul_128x64x32_B.bin", B2)
    save_tensor(f"{args.output}/matmul_128x64x32_C_ref.bin", C2)
    print(f"  MatMul 128x64x32: A={A2.shape} B={B2.shape} C={C2.shape}")

    # SiLU: 4096 elements
    x_s, y_s = generate_silu_ref(4096, seed=300)
    save_tensor(f"{args.output}/silu_4096_input.bin", x_s)
    save_tensor(f"{args.output}/silu_4096_ref.bin", y_s)
    print(f"  SiLU: n={x_s.shape[0]}")

    # ReLU: 4096 elements
    x_r, y_r = generate_relu_ref(4096, seed=400)
    save_tensor(f"{args.output}/relu_4096_input.bin", x_r)
    save_tensor(f"{args.output}/relu_4096_ref.bin", y_r)
    print(f"  ReLU: n={x_r.shape[0]}")

    # RMSNorm: 4096 (Llama hidden dim)
    x_n, w_n, y_n = generate_rmsnorm_ref(4096, eps=1e-5, seed=500)
    save_tensor(f"{args.output}/rmsnorm_4096_input.bin", x_n)
    save_tensor(f"{args.output}/rmsnorm_4096_weight.bin", w_n)
    save_tensor(f"{args.output}/rmsnorm_4096_ref.bin", y_n)
    print(f"  RMSNorm: n={x_n.shape[0]}")

    print(f"\n[GEN] References saved to: {args.output}")

    # ─── Compare (if requested) ──────────────────────────────────────
    if args.compare:
        print(f"\n[CMP] Comparing VULCAN outputs from: {args.compare}\n")

        all_pass = True
        tests = [
            (f"{args.compare}/matmul_64_C.bin", C.flatten(), "MatMul 64x64", 1e-3),
            (f"{args.compare}/matmul_128x64x32_C.bin", C2.flatten(), "MatMul 128x64x32", 1e-3),
            (f"{args.compare}/silu_4096_output.bin", y_s, "SiLU", 1e-5),
            (f"{args.compare}/relu_4096_output.bin", y_r, "ReLU", 1e-6),
            (f"{args.compare}/rmsnorm_4096_output.bin", y_n, "RMSNorm", 1e-4),
        ]

        for path, ref, name, tol in tests:
            if os.path.exists(path):
                if not compare_outputs(path, ref, name, tol):
                    all_pass = False
            else:
                print(f"  ? {name}: Output file not found: {path}")

        print()
        if all_pass:
            print("═══ ALL TESTS PASSED ═══")
        else:
            print("═══ SOME TESTS FAILED ═══")
            sys.exit(1)

if __name__ == "__main__":
    main()
