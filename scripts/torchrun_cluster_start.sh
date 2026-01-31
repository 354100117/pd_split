#!/usr/bin/env bash
set -euo pipefail

# ===== 基本配置 =====
MASTER_IP="192.168.0.10"
MASTER_PORT="29500"
USER="nvidia"
WORKDIR="/ssd/pd/pd_infer_pipeline"
MODEL_PATH="/ssd/models/opt-2.7b"
CONFIG_PATH="/ssd/pd/pd_infer_pipeline/configs/cluster_grouped.json"
EXPERIMENT_MODE="pd_split"  # pd_split | single_node | full_pipeline
BACKEND="gloo"
NODE_NAMES="node1,node2,node3,node4,node5"
DECODE_WORKERS="4"
BATCH_SIZE="4"
BATCH_TIMEOUT_MS="20"
DECODE_BATCH_SIZE="1"
DECODE_BATCH_TIMEOUT_MS="10"
LAYER_STRATEGY="mem"
PREFILL_LAYER_STRATEGY=""
DECODE_LAYER_STRATEGY=""
WARMUP="1"
WARMUP_MAX_NEW_TOKENS="16"
WARMUP_TIMEOUT_S="120"
WARMUP_PROMPTS="Explain the concept of entropy in thermodynamics and its relation to disorder.||Write a Python function to detect if a string is a palindrome, ignoring case and non-alphanumeric characters."
EMPTY_SENTINEL="__EMPTY__"

# 节点列表（按 node_rank 顺序）
NODES=(
  "192.168.0.10"
  "192.168.0.20"
  "192.168.0.30"
  "192.168.0.40"
  "192.168.0.50"
)

usage() {
  cat <<'EOF'
用法:
  bash "torchrun_cluster_start.sh" [flags]

兼容旧位置参数:
  bash "torchrun_cluster_start.sh" <mode> <model_path> <master_port> <decode_workers>

常用 flags:
  --mode <pd_split|single_node|full_pipeline>
  --model-path <path>
  --config <path>
  --master-ip <ip>
  --master-port <port>
  --backend <gloo|nccl>
  --node-names <node1,node2,...>
  --decode-workers <int>
  --batch-size <int>
  --batch-timeout-ms <int>
  --decode-batch-size <int>
  --decode-batch-timeout-ms <int>
  --layer-strategy <mem|compute|bandwidth|uniform>
  --prefill-layer-strategy <mem|compute|bandwidth|uniform|auto>
  --decode-layer-strategy <mem|compute|bandwidth|uniform|auto>
  --warmup <0|1>
  --warmup-max-new-tokens <int>
  --warmup-timeout-s <int>
  --warmup-prompts "<p1||p2||...>"

示例:
  bash "torchrun_cluster_start.sh" \
    --mode pd_split \
    --model-path "/ssd/models/opt-2.7b" \
    --decode-workers 4 \
    --decode-batch-size 4 \
    --decode-batch-timeout-ms 10 \
    --prefill-layer-strategy compute \
    --decode-layer-strategy bandwidth
EOF
}

use_flags=0
for arg in "$@"; do
  if [[ "$arg" == -* ]]; then
    use_flags=1
    break
  fi
done

if [ "$use_flags" -eq 0 ]; then
  if [ "${1:-}" != "" ]; then
    EXPERIMENT_MODE="$1"
  fi
  if [ "${2:-}" != "" ]; then
    MODEL_PATH="$2"
  fi
  if [ "${3:-}" != "" ]; then
    MASTER_PORT="$3"
  fi
  if [ "${4:-}" != "" ]; then
    DECODE_WORKERS="$4"
  fi
else
  while [ "$#" -gt 0 ]; do
    case "$1" in
      -h|--help)
        usage
        exit 0
        ;;
      --mode)
        EXPERIMENT_MODE="${2:-}"
        shift 2
        ;;
      --model-path)
        MODEL_PATH="${2:-}"
        shift 2
        ;;
      --config)
        CONFIG_PATH="${2:-}"
        shift 2
        ;;
      --master-ip)
        MASTER_IP="${2:-}"
        shift 2
        ;;
      --master-port)
        MASTER_PORT="${2:-}"
        shift 2
        ;;
      --backend)
        BACKEND="${2:-}"
        shift 2
        ;;
      --node-names)
        NODE_NAMES="${2:-}"
        shift 2
        ;;
      --decode-workers)
        DECODE_WORKERS="${2:-}"
        shift 2
        ;;
      --batch-size)
        BATCH_SIZE="${2:-}"
        shift 2
        ;;
      --batch-timeout-ms)
        BATCH_TIMEOUT_MS="${2:-}"
        shift 2
        ;;
      --decode-batch-size)
        DECODE_BATCH_SIZE="${2:-}"
        shift 2
        ;;
      --decode-batch-timeout-ms)
        DECODE_BATCH_TIMEOUT_MS="${2:-}"
        shift 2
        ;;
      --layer-strategy)
        LAYER_STRATEGY="${2:-}"
        shift 2
        ;;
      --prefill-layer-strategy)
        PREFILL_LAYER_STRATEGY="${2:-}"
        shift 2
        ;;
      --decode-layer-strategy)
        DECODE_LAYER_STRATEGY="${2:-}"
        shift 2
        ;;
      --warmup)
        WARMUP="${2:-}"
        shift 2
        ;;
      --warmup-max-new-tokens)
        WARMUP_MAX_NEW_TOKENS="${2:-}"
        shift 2
        ;;
      --warmup-timeout-s)
        WARMUP_TIMEOUT_S="${2:-}"
        shift 2
        ;;
      --warmup-prompts)
        WARMUP_PROMPTS="${2:-}"
        shift 2
        ;;
      *)
        echo "Unknown option: $1"
        usage
        exit 1
        ;;
    esac
  done
fi

# ssh 可能丢失空字符串参数，避免位置错位
if [ -z "${PREFILL_LAYER_STRATEGY}" ]; then
  PREFILL_LAYER_STRATEGY="${EMPTY_SENTINEL}"
fi
if [ -z "${DECODE_LAYER_STRATEGY}" ]; then
  DECODE_LAYER_STRATEGY="${EMPTY_SENTINEL}"
fi
if [ -z "${WARMUP_PROMPTS}" ]; then
  WARMUP_PROMPTS="${EMPTY_SENTINEL}"
fi

NUM_NODES="${#NODES[@]}"

# ===== 启动所有节点 =====
for idx in "${!NODES[@]}"; do
  ip="${NODES[$idx]}"
  rank="$idx"
  echo "==> Launching node_rank=${rank} on ${ip}"
  ssh "${USER}@${ip}" "bash -s" -- \
    "${WORKDIR}" \
    "${MASTER_IP}" \
    "${MASTER_PORT}" \
    "${NUM_NODES}" \
    "${rank}" \
    "${EXPERIMENT_MODE}" \
    "${BACKEND}" \
    "${NODE_NAMES}" \
    "${CONFIG_PATH}" \
    "${MODEL_PATH}" \
    "${DECODE_WORKERS}" \
    "${BATCH_SIZE}" \
    "${BATCH_TIMEOUT_MS}" \
    "${DECODE_BATCH_SIZE}" \
    "${DECODE_BATCH_TIMEOUT_MS}" \
    "${LAYER_STRATEGY}" \
    "${PREFILL_LAYER_STRATEGY}" \
    "${DECODE_LAYER_STRATEGY}" \
    "${WARMUP}" \
    "${WARMUP_MAX_NEW_TOKENS}" \
    "${WARMUP_TIMEOUT_S}" \
    "${WARMUP_PROMPTS}" <<'REMOTE'
set -euo pipefail

WORKDIR="$1"
MASTER_ADDR="$2"
MASTER_PORT="$3"
NNODES="$4"
NODE_RANK="$5"
EXPERIMENT_MODE="$6"
BACKEND="$7"
NODE_NAMES="$8"
CONFIG_PATH="$9"
MODEL_PATH="${10}"
DECODE_WORKERS="${11:-}"
BATCH_SIZE="${12:-}"
BATCH_TIMEOUT_MS="${13:-}"
DECODE_BATCH_SIZE="${14:-}"
DECODE_BATCH_TIMEOUT_MS="${15:-}"
LAYER_STRATEGY="${16:-}"
PREFILL_LAYER_STRATEGY="${17:-}"
DECODE_LAYER_STRATEGY="${18:-}"
WARMUP="${19:-}"
WARMUP_MAX_NEW_TOKENS="${20:-}"
WARMUP_TIMEOUT_S="${21:-}"
WARMUP_PROMPTS="${22:-}"
EMPTY_SENTINEL="__EMPTY__"

if [ "${PREFILL_LAYER_STRATEGY}" = "${EMPTY_SENTINEL}" ]; then
  PREFILL_LAYER_STRATEGY=""
fi
if [ "${DECODE_LAYER_STRATEGY}" = "${EMPTY_SENTINEL}" ]; then
  DECODE_LAYER_STRATEGY=""
fi
if [ "${WARMUP_PROMPTS}" = "${EMPTY_SENTINEL}" ]; then
  WARMUP_PROMPTS=""
fi

if [ -f "$HOME/.bashrc" ]; then
  . "$HOME/.bashrc"
fi
export PATH="$HOME/.local/bin:$PATH"

if command -v torchrun >/dev/null 2>&1 && torchrun --help 2>/dev/null | grep -q -- "--nnodes"; then
  TORCHRUN_CMD=(torchrun)
else
  TORCHRUN_CMD=(python3 -m torch.distributed.run)
fi

cd "$WORKDIR"
export MASTER_ADDR="$MASTER_ADDR"
export MASTER_PORT="$MASTER_PORT"

SERVE_ARGS=(
  --experiment_mode "$EXPERIMENT_MODE"
  --backend "$BACKEND"
  --node_names "$NODE_NAMES"
  --config "$CONFIG_PATH"
  --model_path "$MODEL_PATH"
  --decode_workers "$DECODE_WORKERS"
  --batch_size "$BATCH_SIZE"
  --batch_timeout_ms "$BATCH_TIMEOUT_MS"
  --decode_batch_size "$DECODE_BATCH_SIZE"
  --decode_batch_timeout_ms "$DECODE_BATCH_TIMEOUT_MS"
  --layer_strategy "$LAYER_STRATEGY"
  --warmup "$WARMUP"
  --warmup_max_new_tokens "$WARMUP_MAX_NEW_TOKENS"
  --warmup_timeout_s "$WARMUP_TIMEOUT_S"
)
if [ -n "$PREFILL_LAYER_STRATEGY" ]; then
  SERVE_ARGS+=(--prefill_layer_strategy "$PREFILL_LAYER_STRATEGY")
fi
if [ -n "$DECODE_LAYER_STRATEGY" ]; then
  SERVE_ARGS+=(--decode_layer_strategy "$DECODE_LAYER_STRATEGY")
fi
if [ -n "$WARMUP_PROMPTS" ]; then
  SERVE_ARGS+=(--warmup_prompts "$WARMUP_PROMPTS")
fi

"${TORCHRUN_CMD[@]}" \
  --nnodes="$NNODES" --nproc_per_node=1 --node_rank="$NODE_RANK" \
  --master_addr="$MASTER_ADDR" --master_port="$MASTER_PORT" \
  "$WORKDIR/whole_process/serve.py" \
  "${SERVE_ARGS[@]}" \
  --host "0.0.0.0" --port 8000 \
  > "/tmp/serve_gloo_${NODE_RANK}.log" 2>&1 &
REMOTE
  sleep 0.2
done

echo "==> All nodes launched. Check logs: /tmp/serve_gloo_<rank>.log"
