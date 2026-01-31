import threading
import time
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

from transformers import AutoConfig

from backend.dist import DistRpc, WorkerServer
from backend.logger import now_ns
from backend.layer_strategy import build_ranges
from backend.dist import RPC_PRIORITY_HIGH, RPC_PRIORITY_LOW
from backend.pipeline_runner import StageHandle, run_decode_logged, run_decode_logged_batch, run_decode_worker, run_prefill_logged
from backend.state import RequestTracker
from core.common import encode_prompt, get_tokenizer, load_cluster_grouped, split_groups
from core.pipeline import DecodeStage, DecodeWorker, PrefillStage


@dataclass
class DistContext:
    rank: int
    world_size: int
    node_names: List[str]

    @property
    def node_name(self) -> str:
        if 0 <= self.rank < len(self.node_names):
            return self.node_names[self.rank]
        return f"rank{self.rank}"

    @property
    def node_to_rank(self) -> Dict[str, int]:
        return {name: idx for idx, name in enumerate(self.node_names)}


def _pad_batch(ids_batch: List[List[int]], pad_id: int) -> tuple[list, list]:
    max_len = max(len(x) for x in ids_batch) if ids_batch else 0
    padded = []
    masks = []
    for ids in ids_batch:
        pad_len = max_len - len(ids)
        padded.append(ids + [pad_id] * pad_len)
        masks.append([1] * len(ids) + [0] * pad_len)
    return padded, masks


def _split_kv_cache_by_len(kv_cache, lengths: List[int]):
    if kv_cache is None:
        return [None for _ in range(len(lengths))]
    per_req = [[] for _ in range(len(lengths))]
    for layer in kv_cache:
        if layer is None:
            for i in range(len(lengths)):
                per_req[i].append(None)
            continue
        k, v = layer
        for i, seq_len in enumerate(lengths):
            per_req[i].append(
                (
                    k[i : i + 1, :, :seq_len, :].contiguous(),
                    v[i : i + 1, :, :seq_len, :].contiguous(),
                )
            )
    return per_req


def _infer_past_len(kv_cache) -> int:
    if kv_cache is None:
        return 0
    for kv in kv_cache:
        if kv is None:
            continue
        k, _ = kv
        return int(k.shape[2])
    return 0


def _filter_nodes(cluster: Dict, nodes: List[str], min_mem_gb: float) -> List[str]:
    if min_mem_gb <= 0:
        return nodes
    return [n for n in nodes if float(cluster.get(n, {}).get("gpu_mem_gb", 0.0)) >= min_mem_gb]


def _stage_meta(nodes: List[str], ranges: List[Tuple[int, int]]) -> List[Tuple[str, Tuple[int, int]]]:
    return [(nodes[i], ranges[i]) for i in range(len(nodes))]


def _decode_text(tokenizer, ids: List[int]) -> str:
    if tokenizer is None:
        return "".join(chr(min(255, x)) for x in ids)
    try:
        return tokenizer.decode(ids, skip_special_tokens=True)
    except Exception:
        return str(ids)


def _build_stage_handles(
    nodes: List[str],
    ranges: List[Tuple[int, int]],
    dist_ctx: DistContext,
    stage_type: str,
    model_name: str,
):
    handles: List[StageHandle] = []
    node_to_rank = dist_ctx.node_to_rank
    for idx, node in enumerate(nodes):
        if node not in node_to_rank:
            raise RuntimeError(f"node {node} not in node_names mapping")
        rank = node_to_rank[node]
        layer_range = ranges[idx]
        stage = None
        if rank == dist_ctx.rank:
            if stage_type == "prefill":
                stage = PrefillStage(idx, layer_range, model_name)
            elif stage_type == "decode":
                is_first = idx == 0
                is_last = idx == len(nodes) - 1
                stage = DecodeStage(idx, layer_range, model_name, is_first, is_last)
            else:
                raise RuntimeError(f"unknown stage_type {stage_type}")
        handles.append(StageHandle(rank=rank, node=node, layer_range=layer_range, stage_id=idx, stage=stage))
    return handles


class BaseExperiment:
    def __init__(self, config: Dict, tracker: RequestTracker, dist_ctx: DistContext, rpc: Optional[DistRpc]):
        self.config = config
        self.tracker = tracker
        self.dist_ctx = dist_ctx
        self.rpc = rpc
        self.tokenizer = get_tokenizer(config.get("tokenizer_dir", ""))

    def submit(self, prompts: List[str], max_new_tokens: int, target_node: str = None, warmup: bool = False):
        raise NotImplementedError

    def describe(self) -> Dict:
        return {}

    def worker_server(self) -> Optional[WorkerServer]:
        return None

    def reset(self):
        return None


class PDSplitExperiment(BaseExperiment):
    def __init__(self, config: Dict, tracker: RequestTracker, dist_ctx: DistContext, rpc: Optional[DistRpc]):
        super().__init__(config, tracker, dist_ctx, rpc)
        cluster = load_cluster_grouped(config["cluster_path"])
        prefill_nodes, decode_nodes = split_groups(cluster)
        if not prefill_nodes or not decode_nodes:
            raise RuntimeError("prefill/decode nodes empty")
        decode_nodes = _filter_nodes(cluster, decode_nodes, config.get("min_mem_gb", 0.0))
        self.prefill_nodes = sorted(prefill_nodes)
        self.decode_nodes = sorted(decode_nodes)
        self.num_layers = AutoConfig.from_pretrained(config["model_name"]).num_hidden_layers
        prefill_strategy = config.get("prefill_layer_strategy") or config["layer_strategy"]
        decode_strategy = config.get("decode_layer_strategy") or config["layer_strategy"]
        if prefill_strategy == "auto":
            prefill_strategy = "auto_prefill"
        if decode_strategy == "auto":
            decode_strategy = "auto_decode"
        prefill_ranges = build_ranges(self.num_layers, self.prefill_nodes, cluster, prefill_strategy)
        decode_ranges = build_ranges(self.num_layers, self.decode_nodes, cluster, decode_strategy)
        self.prefill_meta = _stage_meta(self.prefill_nodes, prefill_ranges)
        self.decode_meta = _stage_meta(self.decode_nodes, decode_ranges)
        self.prefill_stages = _build_stage_handles(self.prefill_nodes, prefill_ranges, dist_ctx, "prefill", config["model_name"])
        self.decode_stages = _build_stage_handles(self.decode_nodes, decode_ranges, dist_ctx, "decode", config["model_name"])
        self.batch_size = config.get("batch_size", 4)
        self.batch_timeout_ms = config.get("batch_timeout_ms", 20)
        self.decode_batch_size = int(config.get("decode_batch_size", 1) or 1)
        self.decode_batch_timeout_ms = int(config.get("decode_batch_timeout_ms", 10) or 10)
        requested_workers = int(config.get("decode_workers", 0) or 0)
        auto_workers = len(self.decode_nodes) if self.decode_nodes else 1
        self.decode_workers = max(1, requested_workers or auto_workers)
        self._decode_workers_target = self.decode_workers
        self._queue = []
        self._order = []
        self._kv_ready: Dict[str, object] = {}
        self._items: Dict[str, Dict] = {}
        self._lock = threading.Lock()
        self._cond = threading.Condition(self._lock)
        self._running = True
        self._prefill_worker = threading.Thread(target=self._prefill_loop, daemon=True)
        self._decode_workers = [
            threading.Thread(target=self._decode_loop, args=(worker_id,), daemon=True)
            for worker_id in range(self.decode_workers)
        ]
        if self.dist_ctx.rank == 0:
            self._prefill_worker.start()
            for worker in self._decode_workers:
                worker.start()

    def describe(self) -> Dict:
        def pack(meta, group):
            return [
                {
                    "node": node,
                    "group": group,
                    "layer_range": f"{rng[0]}-{rng[1]}",
                }
                for node, rng in meta
            ]

        return {
            "prefill": pack(self.prefill_meta, "prefill"),
            "decode": pack(self.decode_meta, "decode"),
            "num_layers": self.num_layers,
        }

    def update_batching(
        self,
        batch_size: Optional[int] = None,
        batch_timeout_ms: Optional[int] = None,
        decode_batch_size: Optional[int] = None,
        decode_batch_timeout_ms: Optional[int] = None,
        decode_workers: Optional[int] = None,
    ) -> Dict:
        with self._cond:
            if batch_size is not None:
                self.batch_size = max(1, int(batch_size))
            if batch_timeout_ms is not None:
                self.batch_timeout_ms = max(1, int(batch_timeout_ms))
            if decode_batch_size is not None:
                self.decode_batch_size = max(1, int(decode_batch_size))
            if decode_batch_timeout_ms is not None:
                self.decode_batch_timeout_ms = max(1, int(decode_batch_timeout_ms))
            if decode_workers is not None:
                target = max(1, int(decode_workers))
                self.decode_workers = target
                if target > len(self._decode_workers):
                    for worker_id in range(len(self._decode_workers), target):
                        t = threading.Thread(target=self._decode_loop, args=(worker_id,), daemon=True)
                        t.start()
                        self._decode_workers.append(t)
                self._decode_workers_target = target
            self._cond.notify_all()
        return {
            "batch_size": self.batch_size,
            "batch_timeout_ms": self.batch_timeout_ms,
            "decode_batch_size": self.decode_batch_size,
            "decode_batch_timeout_ms": self.decode_batch_timeout_ms,
            "decode_workers": self.decode_workers,
        }

    def _prefill_loop(self):
        while self._running:
            batch = []
            start_wait = time.time()
            with self._cond:
                while self._running and not self._queue:
                    self._cond.wait(timeout=self.batch_timeout_ms / 1000.0)
                if not self._running:
                    return
                while len(batch) < self.batch_size:
                    if self._queue:
                        batch.append(self._queue.pop(0))
                    else:
                        remain = self.batch_timeout_ms / 1000.0 - (time.time() - start_wait)
                        if remain <= 0:
                            break
                        self._cond.wait(timeout=remain)
            if not batch:
                continue

            self.tracker.prefill_queue_len = max(0, self.tracker.prefill_queue_len - len(batch))
            self.tracker.prefill_active += 1
            ids_batch = [b["input_ids"] for b in batch]
            lengths = [len(x) for x in ids_batch]
            pad_id = self.tokenizer.pad_token_id if self.tokenizer and self.tokenizer.pad_token_id is not None else 0
            padded, mask = _pad_batch(ids_batch, int(pad_id))

            for b in batch:
                rec = b["record"]
                rec.prefill_start_ns = now_ns()
                self.tracker.stage_event(rec.request_id, "prefill", self.prefill_meta[0][0], "start")

            req_ids = [b["record"].request_id for b in batch]
            kv_cache = run_prefill_logged(self.prefill_stages, padded, mask, self.tracker, req_ids, self.rpc)

            for b in batch:
                rec = b["record"]
                rec.prefill_end_ns = now_ns()
                self.tracker.stage_event(rec.request_id, "prefill", self.prefill_meta[-1][0], "end")

            kv_splits = _split_kv_cache_by_len(kv_cache, lengths)
            with self._cond:
                for b, kv in zip(batch, kv_splits):
                    req_id = b["record"].request_id
                    self._kv_ready[req_id] = kv
                    self.tracker.decode_queue_len += 1
                self._cond.notify_all()

            self.tracker.prefill_active -= 1

    def _decode_loop(self, worker_id: int):
        while self._running:
            batch_items = []
            batch_req_ids = []
            batch_kv = []
            batch_max_new = None
            with self._cond:
                while self._running and (worker_id >= self._decode_workers_target or not self._order):
                    self._cond.wait(timeout=0.5)
                if not self._running:
                    return
                if worker_id >= self._decode_workers_target:
                    continue
                start_wait = time.time()
                while self._running:
                    ready = []
                    for req_id in list(self._order):
                        if req_id not in self._kv_ready:
                            continue
                        item = self._items.get(req_id)
                        if item is None:
                            continue
                        kv_cache = self._kv_ready.get(req_id)
                        if kv_cache is None:
                            continue
                        max_new = item["max_new_tokens"]
                        if batch_max_new is None:
                            batch_max_new = max_new
                        if max_new != batch_max_new:
                            continue
                        ready.append(req_id)
                        if len(ready) >= self.decode_batch_size:
                            break
                    if not ready:
                        batch_max_new = None
                    timeout_s = self.decode_batch_timeout_ms / 1000.0
                    elapsed = time.time() - start_wait
                    if ready and (len(ready) >= self.decode_batch_size or elapsed >= timeout_s):
                        for req_id in ready:
                            self._order.remove(req_id)
                            item = self._items.get(req_id)
                            kv_cache = self._kv_ready.pop(req_id, None)
                            if item is None or kv_cache is None:
                                continue
                            batch_items.append(item)
                            batch_req_ids.append(req_id)
                            batch_kv.append(kv_cache)
                        if batch_req_ids:
                            self.tracker.decode_queue_len = max(0, self.tracker.decode_queue_len - len(batch_req_ids))
                            break
                    if elapsed >= timeout_s:
                        start_wait = time.time()
                    remain = max(0.0, timeout_s - elapsed)
                    self._cond.wait(timeout=remain if remain > 0 else 0.5)
                if not self._running:
                    return
            if not batch_items:
                continue

            self.tracker.decode_active += len(batch_items)
            for item in batch_items:
                rec = item["record"]
                rec.decode_start_ns = now_ns()
                self.tracker.stage_event(rec.request_id, "decode", self.decode_meta[0][0], "start")
            out_ids_list = run_decode_logged_batch(
                self.decode_stages,
                [item["input_ids"] for item in batch_items],
                batch_kv,
                batch_max_new,
                self.tracker,
                batch_req_ids,
                self.rpc,
            )
            for item, out_ids in zip(batch_items, out_ids_list):
                rec = item["record"]
                rec.decode_end_ns = now_ns()
                self.tracker.stage_event(rec.request_id, "decode", self.decode_meta[-1][0], "end")
                gen_tokens = max(0, len(out_ids) - len(item["input_ids"]))
                self.tracker.finalize(rec.request_id, gen_tokens)
                text = _decode_text(self.tokenizer, out_ids)
                self.tracker.emit_result(rec.request_id, text, gen_tokens)
            self.tracker.decode_active -= len(batch_items)

    def submit(self, prompts: List[str], max_new_tokens: int, target_node: str = None, warmup: bool = False):
        if self.dist_ctx.rank != 0:
            return []
        req_ids = []
        for p in prompts:
            rec = self.tracker.new_request(warmup=warmup)
            input_ids = encode_prompt(p, self.tokenizer)
            item = {
                "record": rec,
                "input_ids": input_ids,
                "max_new_tokens": max_new_tokens,
            }
            with self._cond:
                self._queue.append(item)
                self._items[rec.request_id] = item
                self._order.append(rec.request_id)
                self.tracker.prefill_queue_len += 1
                self._cond.notify_all()
            req_ids.append(rec.request_id)
        return req_ids

    def worker_server(self) -> Optional[WorkerServer]:
        if self.dist_ctx.rank == 0:
            return None
        handlers = {}
        for stage in self.prefill_stages:
            if stage.stage is not None:
                handlers["prefill_forward"] = stage.stage.forward
        for stage in self.decode_stages:
            if stage.stage is not None:
                handlers["decode_step"] = stage.stage.decode_step
                handlers["decode_step_batch"] = stage.stage.decode_step_batch
                handlers["init_kv"] = stage.stage.init_kv
                handlers["clear_kv"] = stage.stage.clear_kv
        handlers["metrics"] = _metrics_handler
        return WorkerServer(handlers)


class SingleNodeExperiment(BaseExperiment):
    def __init__(self, config: Dict, tracker: RequestTracker, dist_ctx: DistContext, rpc: Optional[DistRpc]):
        super().__init__(config, tracker, dist_ctx, rpc)
        cluster = load_cluster_grouped(config["cluster_path"])
        self.nodes = sorted(cluster.keys(), key=lambda n: float(cluster[n].get("gpu_mem_gb", 0.0)), reverse=True)
        self.node_to_rank = dist_ctx.node_to_rank
        self.node = self.nodes[0] if self.nodes else None
        self.num_layers = AutoConfig.from_pretrained(config["model_name"]).num_hidden_layers
        self.local_worker = DecodeWorker(config["model_name"]) if dist_ctx.rank == self.node_to_rank.get(self.node, -1) else None
        self._queue = []
        self._lock = threading.Lock()
        self._cond = threading.Condition(self._lock)
        self._running = True
        self._worker = threading.Thread(target=self._run_loop, daemon=True)
        if self.dist_ctx.rank == 0:
            self._worker.start()

    def _ensure_worker(self, node: str):
        if node not in self.node_to_rank:
            raise RuntimeError(f"node not found: {node}")
        self.node = node
        if self.node_to_rank.get(node) == self.dist_ctx.rank and self.local_worker is None:
            self.local_worker = DecodeWorker(self.config["model_name"])

    def get_node(self) -> str:
        return self.node

    def describe(self) -> Dict:
        if not self.node:
            return {"nodes": [], "num_layers": self.num_layers}
        return {
            "nodes": [
                {
                    "node": self.node,
                    "group": "single_node",
                    "layer_range": f"0-{self.num_layers - 1}",
                }
            ],
            "num_layers": self.num_layers,
        }

    def _run_loop(self):
        while self._running:
            item = None
            with self._cond:
                while self._running and not self._queue:
                    self._cond.wait(timeout=0.5)
                if not self._running:
                    return
                item = self._queue.pop(0)
                self.tracker.prefill_queue_len = max(0, self.tracker.prefill_queue_len - 1)
            if not item:
                continue
            rec = item["record"]
            input_ids = item["input_ids"]
            max_new_tokens = item["max_new_tokens"]
            node = item["node"]

            if node not in self.node_to_rank:
                self.tracker.finalize(rec.request_id, 0)
                self.tracker.emit_result(rec.request_id, "", 0)
                continue

            if self.node_to_rank.get(node) == self.dist_ctx.rank and self.local_worker is None:
                self.local_worker = DecodeWorker(self.config["model_name"])

            self.tracker.prefill_active += 1
            rec.prefill_start_ns = now_ns()
            self.tracker.stage_event(rec.request_id, "prefill", node, "start")
            if self.node_to_rank.get(node) == self.dist_ctx.rank:
                kv_cache = self.local_worker.prefill([input_ids], to_cpu=False)
            else:
                kv_cache = self.rpc.call(
                    self.node_to_rank[node],
                    "single_prefill",
                    {"input_ids": [input_ids]},
                    priority=RPC_PRIORITY_LOW,
                )
            rec.prefill_end_ns = now_ns()
            self.tracker.stage_event(rec.request_id, "prefill", node, "end")
            self.tracker.pipeline_stage(
                rec.request_id,
                "prefill",
                0,
                node,
                f"0-{self.num_layers - 1}",
                rec.prefill_start_ns,
                rec.prefill_end_ns,
            )
            self.tracker.prefill_active -= 1

            self.tracker.decode_active += 1
            rec.decode_start_ns = now_ns()
            self.tracker.stage_event(rec.request_id, "decode", node, "start")
            if self.node_to_rank.get(node) == self.dist_ctx.rank:
                out_ids = run_decode_worker(self.local_worker, [input_ids], kv_cache, max_new_tokens)
            else:
                out_ids = self.rpc.call(
                    self.node_to_rank[node],
                    "single_decode",
                    {"input_ids": [input_ids], "kv_cache": kv_cache, "max_new_tokens": max_new_tokens},
                    priority=RPC_PRIORITY_HIGH,
                )
            rec.decode_end_ns = now_ns()
            self.tracker.stage_event(rec.request_id, "decode", node, "end")
            self.tracker.pipeline_stage(
                rec.request_id,
                "decode",
                0,
                node,
                f"0-{self.num_layers - 1}",
                rec.decode_start_ns,
                rec.decode_end_ns,
            )
            self.tracker.decode_active -= 1

            gen_tokens = max(0, len(out_ids[0]) - len(input_ids))
            self.tracker.finalize(rec.request_id, gen_tokens)
            text = _decode_text(self.tokenizer, out_ids[0])
            self.tracker.emit_result(rec.request_id, text, gen_tokens)

    def submit(self, prompts: List[str], max_new_tokens: int, target_node: str = None, warmup: bool = False):
        if self.dist_ctx.rank != 0:
            return []
        req_ids = []
        for p in prompts:
            rec = self.tracker.new_request(warmup=warmup)
            input_ids = encode_prompt(p, self.tokenizer)
            item = {
                "record": rec,
                "input_ids": input_ids,
                "max_new_tokens": max_new_tokens,
            }
            with self._cond:
                self._queue.append(item)
                self._items[rec.request_id] = item
                self._order.append(rec.request_id)
                self.tracker.prefill_queue_len += 1
                self._cond.notify_all()
            req_ids.append(rec.request_id)
        return req_ids

    def worker_server(self) -> Optional[WorkerServer]:
        if self.dist_ctx.rank == 0:
            return None
        worker = DecodeWorker(self.config["model_name"])
        handlers = {
            "single_prefill": worker.prefill,
            "single_decode": worker.decode,
            "metrics": _metrics_handler,
        }
        return WorkerServer(handlers)


class FullPipelineExperiment(BaseExperiment):
    def __init__(self, config: Dict, tracker: RequestTracker, dist_ctx: DistContext, rpc: Optional[DistRpc]):
        super().__init__(config, tracker, dist_ctx, rpc)
        self.cluster = load_cluster_grouped(config["cluster_path"])
        self.node_to_rank = dist_ctx.node_to_rank
        self.full_nodes = self._select_nodes(config.get("full_pipeline_nodes"))
        self.num_layers = AutoConfig.from_pretrained(config["model_name"]).num_hidden_layers
        self.layer_strategy = config["layer_strategy"]
        self.prefill_stages: List[StageHandle] = []
        self.decode_stages: List[StageHandle] = []
        self._build_pipeline(self.full_nodes, self.layer_strategy)
        self.batch_size = int(config.get("batch_size", 4) or 4)
        self.batch_timeout_ms = int(config.get("batch_timeout_ms", 20) or 20)
        self.decode_batch_size = int(config.get("decode_batch_size", 1) or 1)
        self.decode_batch_timeout_ms = int(config.get("decode_batch_timeout_ms", 10) or 10)
        requested_workers = int(config.get("decode_workers", 0) or 0)
        self.decode_workers = max(1, requested_workers or 1)
        self._decode_workers_target = self.decode_workers
        self._queue = []
        self._order = []
        self._kv_ready: Dict[str, object] = {}
        self._items: Dict[str, Dict] = {}
        self._lock = threading.Lock()
        self._cond = threading.Condition(self._lock)
        self._running = True
        self._prefill_worker = threading.Thread(target=self._prefill_loop, daemon=True)
        self._decode_workers = [
            threading.Thread(target=self._decode_loop, args=(worker_id,), daemon=True)
            for worker_id in range(self.decode_workers)
        ]
        if self.dist_ctx.rank == 0:
            self._prefill_worker.start()
            for worker in self._decode_workers:
                worker.start()

    def _select_nodes(self, selected):
        if selected:
            nodes = [n for n in selected if n in self.cluster]
        else:
            nodes = sorted(self.cluster.keys())
        if not nodes:
            raise RuntimeError("full_pipeline nodes empty")
        return nodes

    def _build_pipeline(self, nodes: List[str], strategy: str):
        ranges = build_ranges(self.num_layers, nodes, self.cluster, strategy)
        if any(r[0] == r[1] for r in ranges):
            raise RuntimeError("num_layers less than nodes; reduce node count")
        self.full_nodes = nodes
        self.layer_strategy = strategy
        self.prefill_stages = _build_stage_handles(nodes, ranges, self.dist_ctx, "prefill", self.config["model_name"])
        self.decode_stages = _build_stage_handles(nodes, ranges, self.dist_ctx, "decode", self.config["model_name"])

    def update_full_pipeline(self, nodes: List[str], strategy: str) -> Dict:
        nodes = self._select_nodes(nodes)
        old_nodes = list(self.full_nodes)
        old_strategy = self.layer_strategy
        self._build_pipeline(nodes, strategy)
        errors = []
        if self.dist_ctx.rank == 0 and self.rpc is not None:
            for node in self.dist_ctx.node_names:
                rank = self.node_to_rank.get(node)
                if rank is None or rank == self.dist_ctx.rank:
                    continue
                payload = {"nodes": nodes, "strategy": strategy}
                try:
                    resp = self.rpc.send_control(rank, "rebuild_full_pipeline", payload)
                except Exception as exc:
                    errors.append(f"{node}:{exc}")
                    continue
                if not isinstance(resp, dict):
                    errors.append(f"{node}:invalid_response")
                    continue
                if resp.get("ok") is False:
                    errors.append(f"{node}:{resp.get('error', 'unknown_error')}")
        if errors:
            self._build_pipeline(old_nodes, old_strategy)
            raise RuntimeError("rebuild_full_pipeline_failed: " + "; ".join(errors))
        return self.describe()

    def describe(self) -> Dict:
        return {
            "nodes": [
                {
                    "node": node,
                    "group": "pipeline",
                    "layer_range": f"{rng[0]}-{rng[1]}",
                }
                for node, rng in _stage_meta(self.full_nodes, [h.layer_range for h in self.prefill_stages])
            ],
            "num_layers": self.num_layers,
        }

    def update_batching(
        self,
        batch_size: Optional[int] = None,
        batch_timeout_ms: Optional[int] = None,
        decode_batch_size: Optional[int] = None,
        decode_batch_timeout_ms: Optional[int] = None,
        decode_workers: Optional[int] = None,
    ) -> Dict:
        with self._cond:
            if batch_size is not None:
                self.batch_size = max(1, int(batch_size))
            if batch_timeout_ms is not None:
                self.batch_timeout_ms = max(1, int(batch_timeout_ms))
            if decode_batch_size is not None:
                self.decode_batch_size = max(1, int(decode_batch_size))
            if decode_batch_timeout_ms is not None:
                self.decode_batch_timeout_ms = max(1, int(decode_batch_timeout_ms))
            if decode_workers is not None:
                target = max(1, int(decode_workers))
                self.decode_workers = target
                if target > len(self._decode_workers):
                    for worker_id in range(len(self._decode_workers), target):
                        t = threading.Thread(target=self._decode_loop, args=(worker_id,), daemon=True)
                        t.start()
                        self._decode_workers.append(t)
                self._decode_workers_target = target
            self._cond.notify_all()
        return {
            "batch_size": self.batch_size,
            "batch_timeout_ms": self.batch_timeout_ms,
            "decode_batch_size": self.decode_batch_size,
            "decode_batch_timeout_ms": self.decode_batch_timeout_ms,
            "decode_workers": self.decode_workers,
        }

    def _prefill_loop(self):
        while self._running:
            batch = []
            start_wait = time.time()
            with self._cond:
                while self._running and not self._queue:
                    self._cond.wait(timeout=self.batch_timeout_ms / 1000.0)
                if not self._running:
                    return
                while len(batch) < self.batch_size:
                    if self._queue:
                        batch.append(self._queue.pop(0))
                    else:
                        remain = self.batch_timeout_ms / 1000.0 - (time.time() - start_wait)
                        if remain <= 0:
                            break
                        self._cond.wait(timeout=remain)
            if not batch:
                continue

            self.tracker.prefill_queue_len = max(0, self.tracker.prefill_queue_len - len(batch))
            self.tracker.prefill_active += 1
            ids_batch = [b["input_ids"] for b in batch]
            lengths = [len(x) for x in ids_batch]
            pad_id = self.tokenizer.pad_token_id if self.tokenizer and self.tokenizer.pad_token_id is not None else 0
            padded, mask = _pad_batch(ids_batch, int(pad_id))

            for b in batch:
                rec = b["record"]
                rec.prefill_start_ns = now_ns()
                self.tracker.stage_event(rec.request_id, "prefill", self.full_nodes[0], "start")

            req_ids = [b["record"].request_id for b in batch]
            kv_cache = run_prefill_logged(self.prefill_stages, padded, mask, self.tracker, req_ids, self.rpc)

            for b in batch:
                rec = b["record"]
                rec.prefill_end_ns = now_ns()
                self.tracker.stage_event(rec.request_id, "prefill", self.full_nodes[-1], "end")

            kv_splits = _split_kv_cache_by_len(kv_cache, lengths)
            with self._cond:
                for b, kv in zip(batch, kv_splits):
                    req_id = b["record"].request_id
                    self._kv_ready[req_id] = kv
                    self.tracker.decode_queue_len += 1
                self._cond.notify_all()

            self.tracker.prefill_active -= 1

    def _decode_loop(self, worker_id: int):
        while self._running:
            batch_items = []
            batch_req_ids = []
            batch_kv = []
            batch_max_new = None
            with self._cond:
                while self._running and (worker_id >= self._decode_workers_target or not self._order):
                    self._cond.wait(timeout=0.5)
                if not self._running:
                    return
                if worker_id >= self._decode_workers_target:
                    continue
                start_wait = time.time()
                while self._running:
                    ready = []
                    for req_id in list(self._order):
                        if req_id not in self._kv_ready:
                            continue
                        item = self._items.get(req_id)
                        if item is None:
                            continue
                        kv_cache = self._kv_ready.get(req_id)
                        if kv_cache is None:
                            continue
                        max_new = item["max_new_tokens"]
                        if batch_max_new is None:
                            batch_max_new = max_new
                        if max_new != batch_max_new:
                            continue
                        ready.append(req_id)
                        if len(ready) >= self.decode_batch_size:
                            break
                    if not ready:
                        batch_max_new = None
                    timeout_s = self.decode_batch_timeout_ms / 1000.0
                    elapsed = time.time() - start_wait
                    if ready and (len(ready) >= self.decode_batch_size or elapsed >= timeout_s):
                        for req_id in ready:
                            item = self._items.get(req_id)
                            if item is None:
                                continue
                            batch_items.append(item)
                            batch_req_ids.append(req_id)
                            batch_kv.append(self._kv_ready.pop(req_id, None))
                            if req_id in self._order:
                                self._order.remove(req_id)
                            self.tracker.decode_queue_len = max(0, self.tracker.decode_queue_len - 1)
                        break
                    remain = timeout_s - elapsed
                    if remain <= 0:
                        break
                    self._cond.wait(timeout=remain)

            if not batch_items:
                continue

            self.tracker.decode_active += len(batch_items)
            for item in batch_items:
                rec = item["record"]
                rec.decode_start_ns = now_ns()
                self.tracker.stage_event(rec.request_id, "decode", self.full_nodes[0], "start")
            out_ids_list = run_decode_logged_batch(
                self.decode_stages,
                [item["input_ids"] for item in batch_items],
                batch_kv,
                batch_max_new,
                self.tracker,
                batch_req_ids,
                self.rpc,
            )
            for item, out_ids in zip(batch_items, out_ids_list):
                rec = item["record"]
                rec.decode_end_ns = now_ns()
                self.tracker.stage_event(rec.request_id, "decode", self.full_nodes[-1], "end")
                gen_tokens = max(0, len(out_ids) - len(item["input_ids"]))
                self.tracker.finalize(rec.request_id, gen_tokens)
                text = _decode_text(self.tokenizer, out_ids)
                self.tracker.emit_result(rec.request_id, text, gen_tokens)
            self.tracker.decode_active -= len(batch_items)

    def submit(self, prompts: List[str], max_new_tokens: int, target_node: str = None, warmup: bool = False):
        if self.dist_ctx.rank != 0:
            return []
        req_ids = []
        for p in prompts:
            rec = self.tracker.new_request(warmup=warmup)
            input_ids = encode_prompt(p, self.tokenizer)
            item = {
                "record": rec,
                "input_ids": input_ids,
                "max_new_tokens": max_new_tokens,
            }
            with self._cond:
                self._queue.append(item)
                self._items[rec.request_id] = item
                self._order.append(rec.request_id)
                self.tracker.prefill_queue_len += 1
                self._cond.notify_all()
            req_ids.append(rec.request_id)
        return req_ids

    def worker_server(self) -> Optional[WorkerServer]:
        if self.dist_ctx.rank == 0:
            return None
        worker = FullPipelineWorker(self, self.dist_ctx)
        handlers = {
            "prefill_forward": worker.prefill_forward,
            "decode_step": worker.decode_step,
            "decode_step_batch": worker.decode_step_batch,
            "metrics": _metrics_handler,
            "init_kv": worker.init_kv,
            "clear_kv": worker.clear_kv,
        }
        controls = {"rebuild_full_pipeline": worker.rebuild_full_pipeline}
        return WorkerServer(handlers, controls)


class FullPipelineWorker:
    def __init__(self, experiment: FullPipelineExperiment, dist_ctx: DistContext):
        self.experiment = experiment
        self.dist_ctx = dist_ctx
        self.prefill_stage = None
        self.decode_stage = None
        self._build_local()

    def _build_local(self):
        node = self.dist_ctx.node_name
        if node not in self.experiment.full_nodes:
            self.prefill_stage = None
            self.decode_stage = None
            return
        idx = self.experiment.full_nodes.index(node)
        prefill_handle = self.experiment.prefill_stages[idx]
        decode_handle = self.experiment.decode_stages[idx]
        if prefill_handle.stage is None:
            prefill_handle.stage = PrefillStage(idx, prefill_handle.layer_range, self.experiment.config["model_name"])
        if decode_handle.stage is None:
            is_first = idx == 0
            is_last = idx == len(self.experiment.full_nodes) - 1
            decode_handle.stage = DecodeStage(idx, decode_handle.layer_range, self.experiment.config["model_name"], is_first, is_last)
        self.prefill_stage = prefill_handle.stage
        self.decode_stage = decode_handle.stage

    def rebuild_full_pipeline(self, nodes: List[str], strategy: str):
        self.experiment._build_pipeline(nodes, strategy)
        self._build_local()
        return {"status": "ok"}

    def prefill_forward(self, input_ids, hidden_states, kv_cache, attention_mask):
        if self.prefill_stage is None:
            raise RuntimeError("prefill stage not available")
        return self.prefill_stage.forward(input_ids, hidden_states, kv_cache, attention_mask)

    def init_kv(self, request_id, kv_cache):
        if self.decode_stage is None:
            raise RuntimeError("decode stage not available")
        return self.decode_stage.init_kv(request_id, kv_cache)

    def clear_kv(self, request_id):
        if self.decode_stage is None:
            return True
        return self.decode_stage.clear_kv(request_id)

    def decode_step(self, request_id, input_ids, hidden_states, past_len):
        if self.decode_stage is None:
            raise RuntimeError("decode stage not available")
        return self.decode_stage.decode_step(request_id, input_ids, hidden_states, past_len)

    def decode_step_batch(self, request_ids, input_ids, hidden_states, past_len):
        if self.decode_stage is None:
            raise RuntimeError("decode stage not available")
        return self.decode_stage.decode_step_batch(request_ids, input_ids, hidden_states, past_len)


def _metrics_handler():
    from backend.monitor import collect_gpu_mem
    import socket

    try:
        node_ip = socket.gethostbyname(socket.gethostname())
    except Exception:
        node_ip = ""
    return {
        "node_ip": node_ip,
        "gpus": collect_gpu_mem(),
    }
