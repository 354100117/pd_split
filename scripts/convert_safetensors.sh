#!/bin/bash

SCRIPT_NAME=$(basename "$0")

show_help() {
  cat << EOF
Usage:
  $SCRIPT_NAME <model_dir> <output_dir>

Description:
  Convert a HuggingFace .bin model directory to safetensors format
  using trans_bin_2_safetensors.py, and copy auxiliary files (*.json, *.txt).

Arguments:
  model_dir     Path to input model directory (HF format, with .bin weights)
  output_dir    Path to output directory for safetensors model

Options:
  -h, --help    Show this help message and exit

Example:
  $SCRIPT_NAME /ssd/models/opt-2.7b /ssd/models/opt-2.7b-safetensors

Notes:
  - The conversion is performed on CPU to avoid GPU memory usage.
  - The output directory will be created if it does not exist.
  - This script is intended to be run once per model, not per node.

EOF
}

# Handle help flag
if [[ "$1" == "-h" || "$1" == "--help" ]]; then
  show_help
  exit 0
fi

# Check arguments
if [ $# -ne 2 ]; then
  echo "[ERROR] Invalid number of arguments."
  echo
  show_help
  exit 1
fi

MODEL_DIR=$1
OUT_DIR=$2

# Basic validation
if [ ! -d "$MODEL_DIR" ]; then
  echo "[ERROR] Model directory does not exist: $MODEL_DIR"
  exit 1
fi

echo "[INFO] Starting model conversion"
echo "  Input model dir : $MODEL_DIR"
echo "  Output model dir: $OUT_DIR"
echo

python3 /ssd/pd/pd_infer_pipeline/whole_process/transfer_safetensors/trans_bin_2_safetensors.py \
  --model-dir "$MODEL_DIR" \
  --output-dir "$OUT_DIR"

RET=$?

if [ $RET -ne 0 ]; then
  echo "[ERROR] Model conversion failed."
  exit $RET
fi

echo
echo "[DONE] Model successfully converted to safetensors."
