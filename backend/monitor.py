import os
import re
import shutil
import socket
import subprocess
import threading
import time
from typing import Dict, List, Optional

from backend.dist import RPC_PRIORITY_LOW
from backend.logger import now_ns


def _get_local_ip() -> str:
    try:
        return socket.gethostbyname(socket.gethostname())
    except Exception:
        return ""


def _gpu_mem_from_nvidia_smi() -> List[Dict]:
    cmd = [
        "nvidia-smi",
        "--query-gpu=memory.total,memory.used",
        "--format=csv,noheader,nounits",
    ]
    out = subprocess.check_output(cmd, text=True).strip()
    results = []
    if not out:
        return results
    for idx, line in enumerate(out.splitlines()):
        parts = [p.strip() for p in line.split(",")]
        if len(parts) != 2:
            continue
        total, used = parts
        results.append(
            {
                "gpu_index": idx,
                "mem_total_mb": int(total),
                "mem_used_mb": int(used),
            }
        )
    return results


def _gpu_mem_from_pynvml() -> Optional[List[Dict]]:
    try:
        import pynvml  # type: ignore
    except Exception:
        return None
    try:
        pynvml.nvmlInit()
        count = pynvml.nvmlDeviceGetCount()
        results = []
        for idx in range(count):
            handle = pynvml.nvmlDeviceGetHandleByIndex(idx)
            mem = pynvml.nvmlDeviceGetMemoryInfo(handle)
            results.append(
                {
                    "gpu_index": idx,
                    "mem_total_mb": int(mem.total / 1024 / 1024),
                    "mem_used_mb": int(mem.used / 1024 / 1024),
                }
            )
        return results
    except Exception:
        return None
    finally:
        try:
            pynvml.nvmlShutdown()
        except Exception:
            pass


def _gpu_mem_from_tegrastats() -> List[Dict]:
    path = shutil.which("tegrastats")
    if not path:
        for cand in ("/usr/bin/tegrastats", "/usr/sbin/tegrastats", "/bin/tegrastats"):
            if os.path.exists(cand):
                path = cand
                break
    if not path:
        return []
    try:
        proc = subprocess.Popen(
            [path],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )
        assert proc.stdout is not None
        out = ""
        deadline = time.time() + 3.0
        while time.time() < deadline:
            line = proc.stdout.readline()
            if not line:
                time.sleep(0.05)
                continue
            out = line.strip()
            if "RAM" in out:
                break
    except Exception:
        return []
    finally:
        try:
            proc.terminate()
        except Exception:
            pass
    if not out:
        return []
    match = re.search(r"RAM\s+(\d+)/(\d+)MB", out)
    if not match:
        return []
    used, total = match.groups()
    return [
        {
            "gpu_index": 0,
            "mem_total_mb": int(total),
            "mem_used_mb": int(used),
        }
    ]


def collect_gpu_mem() -> List[Dict]:
    try:
        mem = _gpu_mem_from_pynvml()
        if mem is None:
            mem = _gpu_mem_from_nvidia_smi()
        if not mem:
            mem = _gpu_mem_from_tegrastats()
    except Exception:
        mem = _gpu_mem_from_tegrastats()
    return mem


class SystemMonitor:
    def __init__(self, interval_s: float, logger, mode: str, state_getter, rpc, rank: int, world_size: int, event_sink=None):
        self.interval_s = interval_s
        self.logger = logger
        self.mode = mode
        self.state_getter = state_getter
        self.rpc = rpc
        self.rank = rank
        self.world_size = world_size
        self.event_sink = event_sink
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._local_ip = _get_local_ip()

    def start(self):
        self._thread.start()

    def stop(self):
        self._stop.set()
        if self._thread.is_alive():
            self._thread.join(timeout=self.interval_s * 2)

    def _collect_gpu_map(self) -> Dict[str, Dict[str, int]]:
        gpu_map: Dict[str, Dict[str, int]] = {}
        local_mem = collect_gpu_mem()
        if self._local_ip:
            total_used = sum(int(g.get("mem_used_mb", 0)) for g in local_mem)
            total_mem = sum(int(g.get("mem_total_mb", 0)) for g in local_mem)
            gpu_map[self._local_ip] = {"used_mb": total_used, "total_mb": total_mem}
        if self.rpc is None:
            return gpu_map
        for r in range(self.world_size):
            if r == self.rank:
                continue
            try:
                resp = self.rpc.call(r, "metrics", {}, priority=RPC_PRIORITY_LOW)
                if not isinstance(resp, dict):
                    continue
                ip = resp.get("node_ip", "")
                gpus = resp.get("gpus", [])
                total_used = 0
                total_mem = 0
                for g in gpus:
                    total_used += int(g.get("mem_used_mb", 0))
                    total_mem += int(g.get("mem_total_mb", 0))
                if ip:
                    gpu_map[ip] = {"used_mb": total_used, "total_mb": total_mem}
            except Exception:
                continue
        return gpu_map

    def _loop(self):
        while not self._stop.is_set():
            gpu_map: Dict[str, Dict[str, int]] = {}
            try:
                gpu_map = self._collect_gpu_map()
            except Exception:
                gpu_map = {}
            gpu_used: Dict[str, int] = {}
            gpu_total: Dict[str, int] = {}
            for ip, mem in gpu_map.items():
                if not isinstance(mem, dict):
                    continue
                gpu_used[ip] = int(mem.get("used_mb", 0))
                gpu_total[ip] = int(mem.get("total_mb", 0))
            state = self.state_getter()
            payload = {
                "type": "system_tick",
                "timestamp_ns": now_ns(),
                "experiment_mode": self.mode,
                "inflight_requests": state.get("inflight_requests", 0),
                "prefill_queue_len": state.get("prefill_queue_len", 0),
                "decode_queue_len": state.get("decode_queue_len", 0),
                "prefill_active": state.get("prefill_active", 0),
                "decode_active": state.get("decode_active", 0),
                "completed_requests_total": state.get("completed_requests_total", 0),
                "generated_tokens_total": state.get("generated_tokens_total", 0),
                "system_throughput_tps": state.get("system_throughput_tps", 0.0),
                "system_throughput_window_s": state.get("system_throughput_window_s", 0.0),
                "gpu_mem_used_mb": gpu_used,
                "gpu_mem_total_mb": gpu_total,
            }
            self.logger.log_system(payload)
            if self.event_sink:
                self.event_sink(payload)
            time.sleep(self.interval_s)
