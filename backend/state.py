import threading
from collections import deque
from dataclasses import dataclass
from typing import Dict, Optional, Callable

from backend.logger import now_ns


@dataclass
class RequestRecord:
    request_id: str
    arrival_time_ns: int
    submit_order: int = 0
    warmup: bool = False
    prefill_start_ns: Optional[int] = None
    prefill_end_ns: Optional[int] = None
    decode_start_ns: Optional[int] = None
    decode_end_ns: Optional[int] = None
    finish_time_ns: Optional[int] = None
    generated_tokens: int = 0


class RequestTracker:
    def __init__(self, logger, event_sink: Callable[[Dict], None], mode: str):
        self._lock = threading.Lock()
        self._seq = 0
        self._records: Dict[str, RequestRecord] = {}
        self._logger = logger
        self._sink = event_sink
        self.mode = mode
        self.prefill_queue_len = 0
        self.decode_queue_len = 0
        self.prefill_active = 0
        self.decode_active = 0
        self._stat_lock = threading.Lock()
        self._completed_total = 0
        self._generated_total = 0
        self._recent_done = deque()

    def new_request(self, warmup: bool = False) -> RequestRecord:
        with self._lock:
            self._seq += 1
            seq = self._seq
            req_id = f"req-{seq:06d}"
        rec = RequestRecord(request_id=req_id, arrival_time_ns=now_ns(), submit_order=seq, warmup=warmup)
        self._records[req_id] = rec
        return rec

    def record(self, req_id: str) -> RequestRecord:
        return self._records[req_id]

    def stage_event(self, req_id: str, stage: str, node: str, event: str):
        rec = self._records.get(req_id)
        if rec and rec.warmup:
            return
        payload = {
            "type": "stage_event",
            "request_id": req_id,
            "experiment_mode": self.mode,
            "stage": stage,
            "node": node,
            "event": event,
            "timestamp_ns": now_ns(),
        }
        self._logger.log_stage(payload)
        self._sink(payload)

    def pipeline_stage(
        self,
        req_id: str,
        stage: str,
        stage_id: int,
        node: str,
        layer_range: str,
        start_ns: int,
        end_ns: int,
    ):
        rec = self._records.get(req_id)
        if rec and rec.warmup:
            return
        payload = {
            "type": "pipeline_stage",
            "request_id": req_id,
            "experiment_mode": self.mode,
            "stage": stage,
            "pipeline_stage_id": stage_id,
            "node": node,
            "layer_range": layer_range,
            "start_ns": start_ns,
            "end_ns": end_ns,
            "latency_ms": (end_ns - start_ns) / 1e6,
        }
        self._logger.log_pipeline(payload)
        self._sink(payload)

    def finalize(self, req_id: str, generated_tokens: int):
        rec = self._records[req_id]
        rec.finish_time_ns = now_ns()
        rec.generated_tokens = generated_tokens
        if rec.warmup:
            return
        finish_order = 0
        with self._stat_lock:
            finish_order = self._completed_total + 1
            self._completed_total += 1
            self._generated_total += int(generated_tokens)
            if rec.finish_time_ns:
                self._recent_done.append((rec.finish_time_ns, int(generated_tokens)))
        prefill_latency_ms = 0.0
        decode_latency_ms = 0.0
        total_latency_ms = 0.0
        if rec.prefill_start_ns and rec.prefill_end_ns:
            prefill_latency_ms = (rec.prefill_end_ns - rec.prefill_start_ns) / 1e6
        if rec.decode_start_ns and rec.decode_end_ns:
            decode_latency_ms = (rec.decode_end_ns - rec.decode_start_ns) / 1e6
        if rec.finish_time_ns:
            total_latency_ms = (rec.finish_time_ns - rec.arrival_time_ns) / 1e6
        throughput_tps = 0.0
        if total_latency_ms > 0:
            throughput_tps = generated_tokens / (total_latency_ms / 1000.0)
        order_delta = rec.submit_order - finish_order if finish_order > 0 else 0
        payload = {
            "type": "request_summary",
            "request_id": rec.request_id,
            "experiment_mode": self.mode,
            "arrival_time_ns": rec.arrival_time_ns,
            "submit_order": rec.submit_order,
            "finish_order": finish_order,
            "order_delta": order_delta,
            "prefill_start_ns": rec.prefill_start_ns,
            "prefill_end_ns": rec.prefill_end_ns,
            "decode_start_ns": rec.decode_start_ns,
            "decode_end_ns": rec.decode_end_ns,
            "finish_time_ns": rec.finish_time_ns,
            "prefill_latency_ms": prefill_latency_ms,
            "decode_latency_ms": decode_latency_ms,
            "total_latency_ms": total_latency_ms,
            "generated_tokens": generated_tokens,
            "throughput_tps": throughput_tps,
        }
        self._logger.log_request(payload)
        self._sink(payload)

    def emit_result(self, req_id: str, text: str, generated_tokens: int):
        rec = self._records.get(req_id)
        if rec and rec.warmup:
            return
        payload = {
            "type": "request_result",
            "request_id": req_id,
            "experiment_mode": self.mode,
            "text": text,
            "generated_tokens": generated_tokens,
            "timestamp_ns": now_ns(),
        }
        self._sink(payload)

    def state_snapshot(self) -> Dict:
        inflight = self.prefill_active + self.decode_active
        snapshot = {
            "inflight_requests": inflight,
            "prefill_queue_len": self.prefill_queue_len,
            "decode_queue_len": self.decode_queue_len,
            "prefill_active": self.prefill_active,
            "decode_active": self.decode_active,
        }
        snapshot.update(self.throughput_snapshot(10.0))
        return snapshot

    def throughput_snapshot(self, window_s: float) -> Dict:
        window_ns = int(max(window_s, 0.0) * 1e9)
        now = now_ns()
        with self._stat_lock:
            while self._recent_done and (now - self._recent_done[0][0]) > window_ns:
                self._recent_done.popleft()
            tokens_in_window = sum(x[1] for x in self._recent_done)
            system_tps = 0.0
            if window_ns > 0:
                system_tps = tokens_in_window / (window_ns / 1e9)
            return {
                "completed_requests_total": self._completed_total,
                "generated_tokens_total": self._generated_total,
                "system_throughput_tps": system_tps,
                "system_throughput_window_s": window_s,
            }

    def reset(self) -> None:
        with self._lock:
            self._seq = 0
            self._records.clear()
        with self._stat_lock:
            self._completed_total = 0
            self._generated_total = 0
            self._recent_done.clear()
        self.prefill_queue_len = 0
        self.decode_queue_len = 0
        self.prefill_active = 0
        self.decode_active = 0
