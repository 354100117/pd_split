# whole_process（torch.distributed + gloo）

本目录为 **torch.distributed(gloo)** 版本的分布式推理实验实现，**不再依赖 Ray**。
通过 `serve.py` 提供 Web UI + HTTP API，支持三种实验模式：
- `pd_split`
- `single_node`
- `full_pipeline`

> 说明：gloo 仅支持 **CPU 张量通信**，跨节点会产生 CPU/GPU 往返开销。

---

## 1. 目录结构

```
whole_process/
  serve.py
  backend/
  core/
  frontend/
  scripts/
```

---

## 2. 前置条件

- Python 3.8+
- torch / transformers / fastapi / uvicorn
- 模型路径可访问（各节点本地路径一致）
- 多节点推荐使用 `torchrun`

---

## 3. 模式说明

- `pd_split`：Prefill / Decode 分离（支持微批）
- `single_node`：单节点基线推理
- `full_pipeline`：全节点单流水线（Prefill/Decode 共用）

**切换模式必须重启服务**（`--experiment_mode` 为启动参数）。

---

## 4. 快速启动（单机）

```bash
python "/ssd/pd/pd_infer_pipeline/whole_process/serve.py" \
  --experiment_mode "single_node" \
  --config "/ssd/pd/pd_infer_pipeline/configs/cluster_grouped.json" \
    --model_path "/ssd/models/opt-2.7b"
```

---

## 5. 多节点启动（torchrun）

### 5.1 节点与 rank 映射

`node_names` 的顺序必须与 `torchrun --node_rank` 一致。
示例映射：
```
node_rank=0 -> node1
node_rank=1 -> node2
node_rank=2 -> node3
node_rank=3 -> node4
node_rank=4 -> node5
```
对应参数：
```
--node_names "node1,node2,node3,node4,node5"
```

### 5.2 在每个节点启动

**node_rank=0（192.168.0.10）**
```bash
torchrun \
  --nnodes=5 --nproc_per_node=1 --node_rank=0 \
  --master_addr=192.168.0.10 --master_port=29500 \
  "/ssd/pd/pd_infer_pipeline/whole_process/serve.py" \
  --experiment_mode "full_pipeline" \
  --backend "gloo" \
  --node_names "node1,node2,node3,node4,node5" \
  --config "/ssd/pd/pd_infer_pipeline/configs/cluster_grouped.json" \
  --model_path "/ssd/models/opt-2.7b" \
  --host "0.0.0.0" --port 8000
```

**node_rank=1..4（其余节点）**
仅替换 `--node_rank`：
```bash
torchrun \
  --nnodes=5 --nproc_per_node=1 --node_rank=1 \
  --master_addr=192.168.0.10 --master_port=29500 \
  "/ssd/pd/pd_infer_pipeline/whole_process/serve.py" \
  --experiment_mode "full_pipeline" \
  --backend "gloo" \
  --node_names "node1,node2,node3,node4,node5" \
  --config "/ssd/pd/pd_infer_pipeline/configs/cluster_grouped.json" \
  --model_path "/ssd/models/opt-2.7b"
```

---

## 6. 一键启动脚本（推荐）

脚本路径：
```
/ssd/pd/pd_infer_pipeline/whole_process/scripts/torchrun_cluster_start.sh
```

### 6.1 默认行为

脚本内默认配置：
- Master：`192.168.0.10:29500`
- 节点顺序：`192.168.0.10/20/30/40/50`
- `node_names`：`node1,node2,node3,node4,node5`
- 模型路径：`/ssd/models/opt-2.7b`
- 默认模式：`pd_split`
- 后端：`gloo`
- `decode_workers`：`4`（`pd_split` / `full_pipeline` 生效；`0` 表示自动）
- `decode_batch_size`：`1`（`pd_split` / `full_pipeline` 生效；解码微批次）
- `decode_batch_timeout_ms`：`10`（`pd_split` / `full_pipeline` 生效）
- `prefill_layer_strategy`：`compute/mem/bandwidth/uniform/auto`（仅 `pd_split` 生效）
- `decode_layer_strategy`：`compute/mem/bandwidth/uniform/auto`（仅 `pd_split` 生效）
- `warmup`：`0/1`，默认为 1
- `warmup-max-new-tokens`：`int`，默认 16
- `warmup-timeout-s`：`int`，默认 120
- `warmup-prompts`：`prompt_1||prompt_2`，使用 `||` 分割

启动后每个节点日志写入：
```
/tmp/serve_gloo_<rank>.log
```
脚本会自动识别 `torchrun` 是否为 PyTorch 版本；若不是则回退：
```
python3 -m torch.distributed.run
```

### 6.2 直接启动（默认参数）

```bash
bash "/ssd/pd/pd_infer_pipeline/whole_process/scripts/torchrun_cluster_start.sh"
```

查看脚本参数说明：
```bash
bash "/ssd/pd/pd_infer_pipeline/whole_process/scripts/torchrun_cluster_start.sh" --help
```

### 6.3 三个常用场景（推荐）

**场景 A：PD 分离（主实验，含微批次与预热）**
```bash
bash "/ssd/pd/pd_infer_pipeline/whole_process/scripts/torchrun_cluster_start.sh" \
  --mode pd_split \
  --model-path "/ssd/models/opt-2.7b" \
  --master-port 29500 \
  --decode-workers 4 \
  --batch-size 4 \
  --batch-timeout-ms 20 \
  --decode-batch-size 4 \
  --decode-batch-timeout-ms 10 \
  --prefill-layer-strategy compute \
  --decode-layer-strategy bandwidth \
  --warmup 1
```

**场景 B：Full Pipeline 对比（同样微批次，关闭预热）**
```bash
bash "/ssd/pd/pd_infer_pipeline/whole_process/scripts/torchrun_cluster_start.sh" \
  --mode full_pipeline \
  --model-path "/ssd/models/opt-2.7b" \
  --master-port 29500 \
  --batch-size 4 \
  --batch-timeout-ms 20 \
  --decode-batch-size 4 \
  --decode-batch-timeout-ms 10 \
  --layer-strategy mem \
  --warmup 0
```

**场景 C：Single Node 基线（固定单机，关闭预热）**
```bash
bash "/ssd/pd/pd_infer_pipeline/whole_process/scripts/torchrun_cluster_start.sh" \
  --mode single_node \
  --model-path "/ssd/models/opt-2.7b" \
  --master-port 29500 \
  --warmup 0
```

### 6.4 传参启动（新模式：flags，完整参数）

**完整参数列表（全部可选，未填写使用脚本默认值）：**

- `--mode`：`pd_split` | `single_node` | `full_pipeline`
- `--model-path`：模型路径
- `--config`：`cluster_grouped.json` 路径
- `--master-ip`：主节点 IP
- `--master-port`：主节点端口
- `--backend`：`gloo` | `nccl`
- `--node-names`：`node1,node2,...`
- `--decode-workers`：解码线程数（`pd_split` / `full_pipeline` 生效；`0` 表示自动）
- `--batch-size`：prefill 微批次大小（`pd_split` / `full_pipeline` 生效）
- `--batch-timeout-ms`：prefill 等待时间（`pd_split` / `full_pipeline` 生效）
- `--decode-batch-size`：decode 微批次大小（`pd_split` / `full_pipeline` 生效）
- `--decode-batch-timeout-ms`：decode 等待时间（`pd_split` / `full_pipeline` 生效）
- `--layer-strategy`：`mem` | `compute` | `bandwidth` | `uniform`（`full_pipeline` 生效）
- `--prefill-layer-strategy`：`mem` | `compute` | `bandwidth` | `uniform` | `auto`（仅 `pd_split` 生效）
- `--decode-layer-strategy`：`mem` | `compute` | `bandwidth` | `uniform` | `auto`（仅 `pd_split` 生效）
- `--warmup`：`0` | `1`（是否预热）
- `--warmup-max-new-tokens`：预热生成 token 数
- `--warmup-timeout-s`：预热超时（秒）
- `--warmup-prompts`：自定义预热 prompts（`||` 分隔）

示例：
```bash
bash "/ssd/pd/pd_infer_pipeline/whole_process/scripts/torchrun_cluster_start.sh" \
  --mode pd_split \
  --model-path "/ssd/models/opt-2.7b" \
  --master-port 29500 \
  --decode-workers 4 \
  --batch-size 4 \
  --batch-timeout-ms 20 \
  --decode-batch-size 4 \
  --decode-batch-timeout-ms 10 \
  --prefill-layer-strategy compute \
  --decode-layer-strategy bandwidth \
  --warmup 1
```

如需关闭预热：
```bash
bash "/ssd/pd/pd_infer_pipeline/whole_process/scripts/torchrun_cluster_start.sh" \
  --mode pd_split \
  --warmup 0
```

默认预热 prompts（可用 `--warmup-prompts` 覆盖，使用 `||` 分隔）：
- Explain the concept of entropy in thermodynamics and its relation to disorder.
- Write a Python function to detect if a string is a palindrome, ignoring case and non-alphanumeric characters.

### 6.5 传参启动（旧模式：位置参数，兼容）

**完整参数列表（仅 4 个，按顺序）：**

1) `experiment_mode`（`pd_split` | `single_node` | `full_pipeline`）  
2) `model_path`  
3) `master_port`  
4) `decode_workers`（仅 `pd_split` 生效；`0` 表示自动=decode 节点数）

示例：
```bash
bash "/ssd/pd/pd_infer_pipeline/whole_process/scripts/torchrun_cluster_start.sh" \
  "pd_split" \
  "/ssd/models/opt-2.7b" \
  "29500" \
  "4"
```

### 6.5 不使用脚本（直接 torchrun）

示例（`pd_split` 启用双策略划分与解码微批次）：
```bash
torchrun \
  --nnodes=5 --nproc_per_node=1 --node_rank=0 \
  --master_addr=192.168.0.10 --master_port=29500 \
  "/ssd/pd/pd_infer_pipeline/whole_process/serve.py" \
  --experiment_mode "pd_split" \
  --backend "gloo" \
  --node_names "node1,node2,node3,node4,node5" \
  --config "/ssd/pd/pd_infer_pipeline/configs/cluster_grouped.json" \
  --model_path "/ssd/models/opt-2.7b" \
  --decode_workers 4 \
  --decode_batch_size 4 \
  --decode_batch_timeout_ms 10 \
  --prefill_layer_strategy compute \
  --decode_layer_strategy bandwidth
```

不使用脚本时，可在 `serve.py` 启动参数中追加：
```
--decode_workers 4
```

### 6.6 修改默认配置

修改脚本顶部变量：
- `MASTER_IP` / `MASTER_PORT`
- `NODES`（顺序必须与 `NODE_NAMES` 一致）
- `NODE_NAMES`
- `MODEL_PATH`
- `EXPERIMENT_MODE`
- `DECODE_WORKERS`（仅 `pd_split` 生效）

---

## 7. 模式切换

**模式切换必须重启服务**，推荐：

```bash
bash "/ssd/pd/pd_infer_pipeline/whole_process/scripts/torchrun_cluster_start.sh" "single_node"
```

如果需要先停止旧进程（推荐）：
```bash
bash "/ssd/pd/pd_infer_pipeline/whole_process/scripts/torchrun_cluster_stop.sh"
```

---

## 8. 前端访问与 API

### 8.1 Web UI

```
http://192.168.0.10:8000
```

### 8.2 API 示例

```bash
curl -X POST "http://192.168.0.10:8000/api/run" \
  -H "Content-Type: application/json" \
  -d '{"prompt":"Hello","max_new_tokens":32}'
```

---

## 9. 服务状态检查

### 9.1 API 检查
```bash
curl -s "http://192.168.0.10:8000/api/config"
```

### 9.2 端口监听
```bash
ss -ltnp | grep ":8000" || netstat -ltnp | grep ":8000"
```

### 9.3 进程检查
```bash
for rank in 0 1 2 3 4; do
  ip="192.168.0.$((10 + rank*10))"
  ssh "nvidia@${ip}" "pgrep -af 'torchrun|serve.py' || true"
done
```

---

## 10. 日志路径

```
whole_process/logs/
  exp_pd_split/
  exp_single_node/
  exp_full_pipeline/
```

每次运行会生成 `run_YYYYMMDD_HHMMSS/`，包含：
- `request.log`
- `stage.log`
- `pipeline.log`
- `system.log`

---

## 11. 延迟定义与排队时间

### 11.1 字段定义（request.log）

- `prefill_latency_ms`：`prefill_start_ns -> prefill_end_ns` 的用时
- `decode_latency_ms`：`decode_start_ns -> decode_end_ns` 的用时
- `total_latency_ms`：`arrival_time_ns -> finish_time_ns` 的用时  
  **包含排队 + prefill + decode**，因此 **不等于** `prefill_latency_ms + decode_latency_ms`

### 11.2 自动统计排队时间

脚本路径：
```
/ssd/pd/pd_infer_pipeline/whole_process/scripts/analyze_queue_waits.py
```

用法：
```bash
python "/ssd/pd/pd_infer_pipeline/whole_process/scripts/analyze_queue_waits.py" \
  --run_dir "/ssd/pd/pd_infer_pipeline/whole_process/logs/exp_pd_split"
```

说明：
- `--run_dir` 可以传 `exp_*` 或 `run_YYYYMMDD_HHMMSS` 目录
- 若传 `exp_*`，脚本会自动选择最新的 `run_*`
- 输出包含 `prefill_queue_wait_ms`、`decode_queue_wait_ms` 以及对应 `avg/p50/p95`

## 12. 常见问题

1) **node_names 数量与 world_size 不一致**
- `--node_names` 数量必须与 `torchrun --nnodes` 相同。

2) **full_pipeline 报 num_layers less than nodes**
- 节点数大于模型层数，需要减少节点。

3) **torchrun 参数不识别（x86 节点）**
- 脚本已自动回退到 `python3 -m torch.distributed.run`。

4) **只加入了部分节点**
- 检查 `/tmp/serve_gloo_<rank>.log`，并确认节点代码已同步。

---

## 13. 备注

- 当前实现不依赖 NCCL，跨节点通信使用 gloo（CPU）。
- 若需要真实 GPU↔GPU pipeline，需要重构为 NCCL/torch.distributed 直连通信。
