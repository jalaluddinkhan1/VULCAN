#!/usr/bin/env python3
"""
VULCAN Model Converter

Converts HuggingFace Llama-2 model weights to VULCAN binary format.

Binary Format (v1):
    Header:
        [4 bytes]  magic number: 0x56554C43 ("VULC")
        [4 bytes]  format version: 1
        [4 bytes]  num_weights

    Per weight:
        [4 bytes]  name_length
        [N bytes]  name (ASCII string)
        [4 bytes]  num_dims
        [4*D bytes] shape (D int32 values)
        [4*numel bytes] data (float32 values)

Usage:
    python convert_model.py \\
        --model meta-llama/Llama-2-7b-hf \\
        --output model.vulcan \\
        --dtype float32

Requirements:
    pip install torch transformers safetensors
"""

import argparse
import struct
import sys
import os
from pathlib import Path

# VULCAN binary format constants
VULCAN_MAGIC = 0x56554C43  # "VULC"
VULCAN_VERSION = 1


def convert_model(model_path: str, output_path: str, dtype: str = "float32"):
    """Convert a HuggingFace model to VULCAN binary format."""
    try:
        import torch
        from transformers import AutoModelForCausalLM
    except ImportError:
        print("[ERROR] Required packages not found.")
        print("        pip install torch transformers safetensors")
        sys.exit(1)

    print(f"[VULCAN] Loading model: {model_path}")

    # Load model
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.float32,
        device_map="cpu"
    )

    state_dict = model.state_dict()
    print(f"[VULCAN] Found {len(state_dict)} weight tensors")

    # Write binary file
    print(f"[VULCAN] Writing to: {output_path}")

    with open(output_path, "wb") as f:
        # Header
        f.write(struct.pack("<I", VULCAN_MAGIC))
        f.write(struct.pack("<I", VULCAN_VERSION))
        f.write(struct.pack("<I", len(state_dict)))

        total_params = 0
        for i, (name, tensor) in enumerate(state_dict.items()):
            tensor = tensor.contiguous().float()  # Ensure contiguous FP32
            shape = list(tensor.shape)
            numel = tensor.numel()
            total_params += numel

            # Write name
            name_bytes = name.encode("ascii")
            f.write(struct.pack("<I", len(name_bytes)))
            f.write(name_bytes)

            # Write shape
            f.write(struct.pack("<I", len(shape)))
            for dim in shape:
                f.write(struct.pack("<i", dim))

            # Write data
            data = tensor.numpy().tobytes()
            f.write(data)

            print(f"  [{i+1}/{len(state_dict)}] {name} "
                  f"shape={shape} ({numel:,} params)")

    file_size = os.path.getsize(output_path)
    print(f"\n[VULCAN] Conversion complete!")
    print(f"  Total parameters: {total_params:,}")
    print(f"  File size: {file_size / (1024**3):.2f} GB")
    print(f"  Output: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Convert HuggingFace model weights to VULCAN binary format"
    )
    parser.add_argument(
        "--model", type=str, required=True,
        help="HuggingFace model name or local path"
    )
    parser.add_argument(
        "--output", type=str, default="model.vulcan",
        help="Output file path (default: model.vulcan)"
    )
    parser.add_argument(
        "--dtype", type=str, default="float32",
        choices=["float32", "float16"],
        help="Weight precision (default: float32)"
    )
    args = parser.parse_args()

    convert_model(args.model, args.output, args.dtype)


if __name__ == "__main__":
    main()
