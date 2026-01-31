#!/usr/bin/env python3
import argparse
import os
import shutil
from transformers import AutoModelForCausalLM
import torch


def copy_aux_files(src_dir: str, dst_dir: str):
    """
    Copy *.json and *.txt files from src_dir to dst_dir
    """
    for fname in os.listdir(src_dir):
        if fname.endswith(".json") or fname.endswith(".txt"):
            src = os.path.join(src_dir, fname)
            dst = os.path.join(dst_dir, fname)
            print(f"[INFO] Copying aux file: {fname}")
            shutil.copy2(src, dst)


def main():
    parser = argparse.ArgumentParser(
        description="Convert HF .bin model to safetensors and copy aux files"
    )
    parser.add_argument(
        "--model-dir",
        type=str,
        required=True,
        help="Input model directory (HF format, with .bin weights)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Output directory for safetensors model",
    )

    args = parser.parse_args()

    model_dir = args.model_dir
    output_dir = args.output_dir

    os.makedirs(output_dir, exist_ok=True)

    print(f"[INFO] Loading model from: {model_dir}")
    model = AutoModelForCausalLM.from_pretrained(
        model_dir,
        torch_dtype=torch.float16,
        device_map="cpu",  # avoid GPU memory usage during conversion
    )

    print(f"[INFO] Saving model to safetensors: {output_dir}")
    model.save_pretrained(
        output_dir,
        safe_serialization=True,
    )

    print("[INFO] Copying auxiliary files (*.json, *.txt)")
    copy_aux_files(model_dir, output_dir)

    print("[DONE] Conversion completed successfully.")


if __name__ == "__main__":
    main()
