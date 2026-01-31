import argparse
import json
import os
import queue
import time
import threading
from typing import Dict, List

import torch.distributed as dist
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, HTMLResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
import uvicorn

from backend.cleanup import full_cleanup
from backend.dist import DistRpc
from backend.logger import EventLogger, now_ns
from backend.monitor import SystemMonitor
from backend.modes import DistContext, FullPipelineExperiment, PDSplitExperiment, SingleNodeExperiment
from backend.state import RequestTracker
from core.common import load_cluster_grouped

BASE_DIR = os.path.abspath(os.path.dirname(__file__))
FRONTEND_DIR = os.path.join(BASE_DIR, "frontend")


class EventBus:
    def __init__(self, max_queue: int = 1000):
        self._lock = threading.Lock()
        self._subs: List[queue.Queue] = []
        self._max_queue = max_queue

    def subscribe(self) -> queue.Queue:
        q = queue.Queue(maxsize=self._max_queue)
        with self._lock:
            self._subs.append(q)
        return q

    def unsubscribe(self, q: queue.Queue) -> None:
        with self._lock:
            if q in self._subs:
                self._subs.remove(q)

    def publish(self, payload: Dict) -> None:
        with self._lock:
            subs = list(self._subs)
        for q in subs:
            try:
                q.put_nowait(payload)
            except queue.Full:
                try:
                    q.get_nowait()
                except queue.Empty:
                    pass
                try:
                    q.put_nowait(payload)
                except queue.Full:
                    pass


def _estimate_model_mem_gb(model_name: str) -> float:
    if not model_name:
        return 0.0
    if os.path.isfile(model_name):
        size = os.path.getsize(model_name)
        return size / (1024 ** 3)
    if os.path.isdir(model_name):
        groups = {".safetensors": [], ".bin": [], ".pt": [], ".pth": []}
        for root, _, files in os.walk(model_name):
            for name in files:
                for ext in groups.keys():
                    if name.endswith(ext):
                        groups[ext].append(os.path.join(root, name))
                        break
        for ext in [".safetensors", ".bin", ".pt", ".pth"]:
            files = groups.get(ext, [])
            if files:
                total = sum(os.path.getsize(p) for p in files)
                return total / (1024 ** 3)
    return 0.0


def _init_dist(backend: str):
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        if not dist.is_initialized():
            dist.init_process_group(backend=backend, init_method="env://")
        return dist.get_rank(), dist.get_world_size(), True
    return 0, 1, False


def _parse_node_names(raw: str) -> List[str]:
    return [n.strip() for n in (raw or "").split(",") if n.strip()]


class _NullTracker:
    def __init__(self):
        self.prefill_queue_len = 0
        self.decode_queue_len = 0
        self.prefill_active = 0
        self.decode_active = 0

    def state_snapshot(self) -> Dict:
        return {
            "inflight_requests": 0,
            "prefill_queue_len": 0,
            "decode_queue_len": 0,
            "prefill_active": 0,
            "decode_active": 0,
            "completed_requests_total": 0,
            "generated_tokens_total": 0,
            "system_throughput_tps": 0.0,
            "system_throughput_window_s": 0.0,
        }


def build_app(args, experiment, tracker: RequestTracker, logger: EventLogger, event_bus: EventBus, monitor: SystemMonitor, cluster_data: Dict, current_config: Dict):
    app = FastAPI()

    recent_lock = threading.Lock()
    recent_summaries: List[Dict] = []

    def sink(payload: Dict) -> None:
        event_bus.publish(payload)
        if payload.get("type") == "request_summary":
            with recent_lock:
                recent_summaries.append(payload)
                if len(recent_summaries) > 200:
                    recent_summaries.pop(0)

    tracker._sink = sink

    app.mount("/static", StaticFiles(directory=FRONTEND_DIR), name="static")

    def _node_capabilities():
        model_mem_gb = _estimate_model_mem_gb(current_config.get("model_name", ""))
        required_mem_gb = max(float(current_config.get("min_mem_gb", 0.0) or 0.0), float(model_mem_gb or 0.0))
        nodes = []
        for name, info in cluster_data.items():
            mem = float(info.get("gpu_mem_gb", 0.0))
            can_run = required_mem_gb <= 0.0 or mem >= required_mem_gb
            reason = ""
            if not can_run and required_mem_gb > 0.0:
                reason = f"gpu_mem_gb<{required_mem_gb:.2f}"
            nodes.append(
                {
                    "name": name,
                    "gpu_mem_gb": mem,
                    "can_run": can_run,
                    "reason": reason,
                }
            )
        nodes.sort(key=lambda x: x["gpu_mem_gb"], reverse=True)
        return nodes, model_mem_gb, required_mem_gb

    @app.get("/")
    def index():
        index_path = os.path.join(FRONTEND_DIR, "index.html")
        with open(index_path, "r", encoding="utf-8") as f:
            return HTMLResponse(f.read())

    @app.get("/api/config")
    def api_config():
        return {
            "experiment_mode": args.experiment_mode,
            "run_dir": logger.run_dir,
            "model_name": current_config.get("model_name", ""),
            "batch_size": current_config.get("batch_size"),
            "batch_timeout_ms": current_config.get("batch_timeout_ms"),
            "decode_batch_size": current_config.get("decode_batch_size"),
            "decode_batch_timeout_ms": current_config.get("decode_batch_timeout_ms"),
            "decode_workers": current_config.get("decode_workers"),
            "layer_strategy": current_config.get("layer_strategy"),
            "prefill_layer_strategy": current_config.get("prefill_layer_strategy"),
            "decode_layer_strategy": current_config.get("decode_layer_strategy"),
        }

    @app.get("/api/layout")
    def api_layout():
        layout = experiment.describe()
        return {"experiment_mode": args.experiment_mode, "layout": layout}

    @app.get("/api/nodes")
    def api_nodes():
        nodes, model_mem_gb, required_mem_gb = _node_capabilities()
        selected = None
        if hasattr(experiment, "get_node"):
            try:
                selected = experiment.get_node()
            except Exception:
                selected = None
        return {
            "nodes": nodes,
            "model_mem_gb": model_mem_gb,
            "required_mem_gb": required_mem_gb,
            "selected_node": selected,
        }

    @app.get("/api/full_pipeline/config")
    def api_full_pipeline_config():
        nodes, model_mem_gb, required_mem_gb = _node_capabilities()
        selected = current_config.get("full_pipeline_nodes") or [n["name"] for n in nodes if n.get("can_run")]
        layout = {}
        if args.experiment_mode == "full_pipeline":
            try:
                layout = experiment.describe()
            except Exception:
                layout = {}
        return {
            "nodes": nodes,
            "selected_nodes": selected,
            "strategy": current_config.get("layer_strategy", "mem"),
            "model_mem_gb": model_mem_gb,
            "required_mem_gb": required_mem_gb,
            "layout": layout,
        }

    @app.post("/api/full_pipeline/config")
    async def api_full_pipeline_update(request: Request):
        if args.experiment_mode != "full_pipeline":
            raise HTTPException(status_code=400, detail="not_full_pipeline_mode")
        data = await request.json()
        nodes = data.get("nodes") or []
        strategy = data.get("strategy") or current_config.get("layer_strategy", "mem")
        if not isinstance(nodes, list) or not nodes:
            raise HTTPException(status_code=400, detail="nodes_empty")
        if (
            tracker.prefill_active > 0
            or tracker.decode_active > 0
            or tracker.prefill_queue_len > 0
            or tracker.decode_queue_len > 0
        ):
            detail = (
                "pipeline_busy "
                f"prefill_active={tracker.prefill_active} "
                f"decode_active={tracker.decode_active} "
                f"prefill_queue={tracker.prefill_queue_len} "
                f"decode_queue={tracker.decode_queue_len}"
            )
            raise HTTPException(status_code=409, detail=detail)
        try:
            layout = experiment.update_full_pipeline(nodes, strategy)
        except Exception as exc:
            raise HTTPException(status_code=500, detail=f"update_failed:{exc}") from exc
        current_config["full_pipeline_nodes"] = nodes
        current_config["layer_strategy"] = strategy
        with recent_lock:
            recent_summaries.clear()
        return {"status": "ok", "layout": layout}

    @app.post("/api/run")
    async def api_run(request: Request):
        data = await request.json()
        prompt = data.get("prompt", "")
        prompts = data.get("prompts")
        if prompts is None:
            if isinstance(prompt, str) and prompt.strip():
                prompts = [prompt]
            else:
                prompts = []
        if not prompts:
            raise HTTPException(status_code=400, detail="prompt_empty")
        max_new_tokens = int(data.get("max_new_tokens", 32))
        target_node = data.get("target_node")
        if args.experiment_mode == "single_node" and target_node:
            nodes, _, _ = _node_capabilities()
            nodes = {n["name"]: n for n in nodes}
            node_info = nodes.get(target_node)
            if node_info is None:
                raise HTTPException(status_code=400, detail="invalid_target_node")
            if not node_info.get("can_run"):
                raise HTTPException(status_code=400, detail="target_node_unavailable")
        threading.Thread(target=experiment.submit, args=(prompts, max_new_tokens, target_node), daemon=True).start()
        return {"status": "accepted", "count": len(prompts)}

    @app.post("/api/pd_split/batching")
    async def api_pd_split_batching(request: Request):
        if args.experiment_mode != "pd_split":
            raise HTTPException(status_code=400, detail="not_pd_split_mode")
        data = await request.json()
        batch_size = data.get("batch_size", None)
        batch_timeout_ms = data.get("batch_timeout_ms", None)
        decode_batch_size = data.get("decode_batch_size", None)
        decode_batch_timeout_ms = data.get("decode_batch_timeout_ms", None)
        decode_workers = data.get("decode_workers", None)
        if (
            batch_size is None
            and batch_timeout_ms is None
            and decode_batch_size is None
            and decode_batch_timeout_ms is None
            and decode_workers is None
        ):
            raise HTTPException(status_code=400, detail="missing_params")
        try:
            bs = int(batch_size) if batch_size is not None else None
            tm = int(batch_timeout_ms) if batch_timeout_ms is not None else None
            dbs = int(decode_batch_size) if decode_batch_size is not None else None
            dtm = int(decode_batch_timeout_ms) if decode_batch_timeout_ms is not None else None
            dw = int(decode_workers) if decode_workers is not None else None
        except Exception:
            raise HTTPException(status_code=400, detail="invalid_params")
        if bs is not None and bs <= 0:
            raise HTTPException(status_code=400, detail="invalid_batch_size")
        if tm is not None and tm <= 0:
            raise HTTPException(status_code=400, detail="invalid_batch_timeout")
        if dbs is not None and dbs <= 0:
            raise HTTPException(status_code=400, detail="invalid_decode_batch_size")
        if dtm is not None and dtm <= 0:
            raise HTTPException(status_code=400, detail="invalid_decode_batch_timeout")
        if dw is not None and dw <= 0:
            raise HTTPException(status_code=400, detail="invalid_decode_workers")
        if not hasattr(experiment, "update_batching"):
            raise HTTPException(status_code=500, detail="batching_not_supported")
        result = experiment.update_batching(
            batch_size=bs,
            batch_timeout_ms=tm,
            decode_batch_size=dbs,
            decode_batch_timeout_ms=dtm,
            decode_workers=dw,
        )
        current_config["batch_size"] = result.get("batch_size")
        current_config["batch_timeout_ms"] = result.get("batch_timeout_ms")
        current_config["decode_batch_size"] = result.get("decode_batch_size")
        current_config["decode_batch_timeout_ms"] = result.get("decode_batch_timeout_ms")
        current_config["decode_workers"] = result.get("decode_workers")
        return {"status": "ok", **result}

    @app.post("/api/full_pipeline/batching")
    async def api_full_pipeline_batching(request: Request):
        if args.experiment_mode != "full_pipeline":
            raise HTTPException(status_code=400, detail="not_full_pipeline_mode")
        data = await request.json()
        batch_size = data.get("batch_size", None)
        batch_timeout_ms = data.get("batch_timeout_ms", None)
        decode_batch_size = data.get("decode_batch_size", None)
        decode_batch_timeout_ms = data.get("decode_batch_timeout_ms", None)
        decode_workers = data.get("decode_workers", None)
        if (
            batch_size is None
            and batch_timeout_ms is None
            and decode_batch_size is None
            and decode_batch_timeout_ms is None
            and decode_workers is None
        ):
            raise HTTPException(status_code=400, detail="missing_params")
        try:
            bs = int(batch_size) if batch_size is not None else None
            tm = int(batch_timeout_ms) if batch_timeout_ms is not None else None
            dbs = int(decode_batch_size) if decode_batch_size is not None else None
            dtm = int(decode_batch_timeout_ms) if decode_batch_timeout_ms is not None else None
            dw = int(decode_workers) if decode_workers is not None else None
        except Exception:
            raise HTTPException(status_code=400, detail="invalid_params")
        if bs is not None and bs <= 0:
            raise HTTPException(status_code=400, detail="invalid_batch_size")
        if tm is not None and tm <= 0:
            raise HTTPException(status_code=400, detail="invalid_batch_timeout")
        if dbs is not None and dbs <= 0:
            raise HTTPException(status_code=400, detail="invalid_decode_batch_size")
        if dtm is not None and dtm <= 0:
            raise HTTPException(status_code=400, detail="invalid_decode_batch_timeout")
        if dw is not None and dw <= 0:
            raise HTTPException(status_code=400, detail="invalid_decode_workers")
        if not hasattr(experiment, "update_batching"):
            raise HTTPException(status_code=500, detail="batching_not_supported")
        result = experiment.update_batching(
            batch_size=bs,
            batch_timeout_ms=tm,
            decode_batch_size=dbs,
            decode_batch_timeout_ms=dtm,
            decode_workers=dw,
        )
        current_config["batch_size"] = result.get("batch_size")
        current_config["batch_timeout_ms"] = result.get("batch_timeout_ms")
        current_config["decode_batch_size"] = result.get("decode_batch_size")
        current_config["decode_batch_timeout_ms"] = result.get("decode_batch_timeout_ms")
        current_config["decode_workers"] = result.get("decode_workers")
        return {"status": "ok", **result}

    @app.get("/api/status")
    def api_status():
        state = tracker.state_snapshot()
        state.update(
            {
                "experiment_mode": args.experiment_mode,
                "run_dir": logger.run_dir,
            }
        )
        return state

    @app.get("/api/summary")
    def api_summary():
        with recent_lock:
            return {"items": list(recent_summaries)}

    @app.get("/api/logs/path")
    def api_logs_path():
        return {"run_dir": logger.run_dir}

    @app.get("/api/logs/file")
    def api_logs_file(name: str):
        allowed = {
            "request": "request.log",
            "stage": "stage.log",
            "pipeline": "pipeline.log",
            "system": "system.log",
        }
        filename = allowed.get(name)
        if not filename:
            raise HTTPException(status_code=404, detail="invalid_log")
        path = os.path.join(logger.run_dir, filename)
        return FileResponse(path)

    @app.get("/api/stream")
    def api_stream():
        q = event_bus.subscribe()

        def event_stream():
            try:
                while True:
                    payload = q.get()
                    yield f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"
            finally:
                event_bus.unsubscribe(q)

        return StreamingResponse(event_stream(), media_type="text/event-stream")

    @app.on_event("shutdown")
    def on_shutdown():
        try:
            monitor.stop()
        finally:
            try:
                logger.close()
            finally:
                pass

    return app


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment_mode", choices=["pd_split", "single_node", "full_pipeline"], required=True)
    parser.add_argument("--backend", default="gloo")
    parser.add_argument("--node_names", default="")
    parser.add_argument("--config", required=True)
    parser.add_argument("--model_path")
    parser.add_argument("--model_name")
    parser.add_argument("--tokenizer_dir")
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--batch_timeout_ms", type=int, default=20)
    parser.add_argument("--decode_batch_size", type=int, default=1)
    parser.add_argument("--decode_batch_timeout_ms", type=int, default=10)
    parser.add_argument("--decode_workers", type=int, default=0)
    parser.add_argument("--layer_strategy", choices=["mem", "compute", "bandwidth", "uniform"], default="mem")
    parser.add_argument("--prefill_layer_strategy", choices=["mem", "compute", "bandwidth", "uniform", "auto"], default=None)
    parser.add_argument("--decode_layer_strategy", choices=["mem", "compute", "bandwidth", "uniform", "auto"], default=None)
    parser.add_argument("--min_mem_gb", type=float, default=0.0)
    parser.add_argument("--log_dir", default=os.path.join(BASE_DIR, "logs"))
    parser.add_argument("--system_tick_interval", type=float, default=1.0)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--warmup", type=int, choices=[0, 1], default=1)
    parser.add_argument("--warmup_max_new_tokens", type=int, default=16)
    parser.add_argument("--warmup_timeout_s", type=int, default=120)
    parser.add_argument("--warmup_prompts", default="")
    return parser.parse_args()


def main():
    args = parse_args()
    if args.model_path:
        args.model_name = args.model_path
        args.tokenizer_dir = args.model_path
    if not args.model_name:
        raise ValueError("model_path 或 model_name 不能为空")
    if not args.tokenizer_dir:
        args.tokenizer_dir = args.model_name

    rank, world_size, dist_ok = _init_dist(args.backend)

    cluster_data = load_cluster_grouped(args.config)
    node_names = _parse_node_names(args.node_names)
    if not node_names:
        node_names = sorted(cluster_data.keys())
    if world_size > 1 and len(node_names) != world_size:
        raise ValueError(f"node_names 数量 {len(node_names)} 与 world_size {world_size} 不一致")
    if not dist_ok and len(node_names) != 1:
        raise ValueError("非分布式模式下 node_names 必须为1，请使用 torchrun 启动多节点")
    if world_size == 1 and not node_names:
        node_names = ["node1"]

    dist_ctx = DistContext(rank=rank, world_size=world_size, node_names=node_names)
    rpc = DistRpc(rank, world_size) if dist_ok and rank == 0 else None

    current_config = {
        "cluster_path": args.config,
        "model_name": args.model_name,
        "tokenizer_dir": args.tokenizer_dir,
        "batch_size": args.batch_size,
        "batch_timeout_ms": args.batch_timeout_ms,
        "decode_batch_size": args.decode_batch_size,
        "decode_batch_timeout_ms": args.decode_batch_timeout_ms,
        "layer_strategy": args.layer_strategy,
        "prefill_layer_strategy": args.prefill_layer_strategy,
        "decode_layer_strategy": args.decode_layer_strategy,
        "min_mem_gb": args.min_mem_gb,
        "decode_workers": args.decode_workers,
        "full_pipeline_nodes": None,
    }

    full_cleanup()

    if rank != 0:
        tracker = _NullTracker()
        if args.experiment_mode == "pd_split":
            experiment = PDSplitExperiment(current_config, tracker, dist_ctx, rpc)
        elif args.experiment_mode == "single_node":
            experiment = SingleNodeExperiment(current_config, tracker, dist_ctx, rpc)
        else:
            experiment = FullPipelineExperiment(current_config, tracker, dist_ctx, rpc)
        server = experiment.worker_server()
        if server:
            server.serve_forever()
        if dist_ok:
            dist.destroy_process_group()
        return

    event_bus = EventBus()
    logger = EventLogger(args.log_dir, args.experiment_mode)
    tracker = RequestTracker(logger, event_bus.publish, args.experiment_mode)

    if args.experiment_mode == "pd_split":
        experiment = PDSplitExperiment(current_config, tracker, dist_ctx, rpc)
    elif args.experiment_mode == "single_node":
        experiment = SingleNodeExperiment(current_config, tracker, dist_ctx, rpc)
    else:
        experiment = FullPipelineExperiment(current_config, tracker, dist_ctx, rpc)

    if args.warmup:
        default_prompts = [
            "Explain the concept of entropy in thermodynamics and its relation to disorder.",
            "Write a Python function to detect if a string is a palindrome, ignoring case and non-alphanumeric characters.",
        ]
        prompts = default_prompts
        if args.warmup_prompts:
            prompts = [p.strip() for p in args.warmup_prompts.replace("\\n", "\n").split("||") if p.strip()]
        if prompts:
            req_ids = experiment.submit(prompts, args.warmup_max_new_tokens, warmup=True)
            deadline = time.time() + max(1, args.warmup_timeout_s)
            done = False
            while time.time() < deadline:
                done = True
                for req_id in req_ids:
                    try:
                        rec = tracker.record(req_id)
                    except Exception:
                        done = False
                        break
                    if not rec.finish_time_ns:
                        done = False
                        break
                if done:
                    break
                time.sleep(0.2)
            if done:
                tracker.reset()

    monitor = SystemMonitor(args.system_tick_interval, logger, args.experiment_mode, tracker.state_snapshot, rpc, rank, world_size, event_bus.publish)
    monitor.start()

    app = build_app(args, experiment, tracker, logger, event_bus, monitor, cluster_data, current_config)
    uvicorn.run(app, host=args.host, port=args.port)

    if dist_ok:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
