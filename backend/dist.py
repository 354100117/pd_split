import io
import pickle
import queue
import threading
from typing import Any, Dict, Optional, Tuple

import torch
import torch.distributed as dist

TAG_REQ = 100
TAG_RESP = 101

RPC_PRIORITY_HIGH = 0
RPC_PRIORITY_NORMAL = 5
RPC_PRIORITY_LOW = 10


def _serialize(obj: Any) -> bytes:
    buffer = io.BytesIO()
    try:
        torch.save(obj, buffer)
        return buffer.getvalue()
    except Exception:
        return pickle.dumps(obj)


def _deserialize(blob: bytes) -> Any:
    buffer = io.BytesIO(blob)
    try:
        try:
            return torch.load(buffer, map_location="cpu", weights_only=False)
        except TypeError:
            return torch.load(buffer, map_location="cpu")
    except Exception:
        return pickle.loads(blob)


def _bytes_to_tensor(blob: bytes) -> torch.Tensor:
    storage = torch.ByteStorage.from_buffer(blob)
    return torch.ByteTensor(storage)


def _tensor_to_bytes(tensor: torch.Tensor) -> bytes:
    try:
        return tensor.cpu().numpy().tobytes()
    except Exception:
        return bytes(bytearray(tensor.tolist()))


def send_obj(obj: Any, dst: int, tag: int) -> None:
    blob = _serialize(obj)
    length = torch.tensor([len(blob)], dtype=torch.int64)
    dist.send(length, dst=dst, tag=tag)
    if blob:
        payload = _bytes_to_tensor(blob)
        dist.send(payload, dst=dst, tag=tag)


def recv_obj(src: int, tag: int) -> Any:
    length = torch.empty(1, dtype=torch.int64)
    dist.recv(length, src=src, tag=tag)
    size = int(length.item())
    if size <= 0:
        return None
    payload = torch.empty(size, dtype=torch.uint8)
    dist.recv(payload, src=src, tag=tag)
    blob = _tensor_to_bytes(payload)
    return _deserialize(blob)


class DistRpc:
    def __init__(self, rank: int, world_size: int):
        self.rank = rank
        self.world_size = world_size
        self._lock = threading.Lock()
        self._next_id = 1
        self._send_seq = 0
        self._pending: Dict[int, "_PendingCall"] = {}
        self._send_queue: "queue.PriorityQueue[Tuple[int, int, int, Dict, int]]" = queue.PriorityQueue()
        self._running = True
        self._sender = threading.Thread(target=self._send_loop, daemon=True)
        self._sender.start()
        self._receivers = []
        for src in range(world_size):
            if src == rank:
                continue
            t = threading.Thread(target=self._recv_loop, args=(src,), daemon=True)
            t.start()
            self._receivers.append(t)

    def _alloc_id(self) -> int:
        with self._lock:
            req_id = self._next_id
            self._next_id += 1
            return req_id

    def _alloc_seq(self) -> int:
        with self._lock:
            seq = self._send_seq
            self._send_seq += 1
            return seq

    def _enqueue(self, dst: int, op: str, payload: Dict, priority: int) -> "_PendingCall":
        req_id = self._alloc_id()
        seq = self._alloc_seq()
        pending = _PendingCall()
        with self._lock:
            self._pending[req_id] = pending
        msg = {"op": op, "payload": payload, "req_id": req_id}
        self._send_queue.put((priority, seq, dst, msg, req_id))
        return pending

    def _deliver_response(self, resp: Any) -> None:
        if not isinstance(resp, dict):
            return
        req_id = resp.get("req_id")
        if req_id is None:
            return
        with self._lock:
            pending = self._pending.pop(int(req_id), None)
        if pending:
            pending.set(resp)

    def _send_loop(self) -> None:
        while self._running:
            _, _, dst, msg, req_id = self._send_queue.get()
            try:
                send_obj(msg, dst=dst, tag=TAG_REQ)
            except Exception as exc:
                self._deliver_response({"ok": False, "error": str(exc), "req_id": req_id})

    def _recv_loop(self, src: int) -> None:
        while self._running:
            resp = recv_obj(src=src, tag=TAG_RESP)
            self._deliver_response(resp)

    def call(self, dst: int, op: str, payload: Dict, priority: int = RPC_PRIORITY_NORMAL) -> Any:
        pending = self._enqueue(dst, op, payload, priority)
        resp = pending.wait()
        if isinstance(resp, dict) and resp.get("ok"):
            return resp.get("result")
        error = None
        if isinstance(resp, dict):
            error = resp.get("error")
        raise RuntimeError(error or f"rpc error from rank {dst}")

    def send_control(self, dst: int, op: str, payload: Dict, priority: int = RPC_PRIORITY_HIGH) -> Any:
        pending = self._enqueue(dst, op, payload, priority)
        return pending.wait()


class _PendingCall:
    def __init__(self):
        self._event = threading.Event()
        self.response: Any = None

    def set(self, resp: Any) -> None:
        self.response = resp
        self._event.set()

    def wait(self) -> Any:
        self._event.wait()
        return self.response


class WorkerServer:
    def __init__(self, handlers: Dict[str, Any], control_handlers: Optional[Dict[str, Any]] = None):
        self.handlers = handlers
        self.control_handlers = control_handlers or {}

    def serve_forever(self, src_rank: int = 0) -> None:
        while True:
            msg = recv_obj(src=src_rank, tag=TAG_REQ)
            if not isinstance(msg, dict):
                send_obj({"ok": False, "error": "invalid_message", "req_id": None}, dst=src_rank, tag=TAG_RESP)
                continue
            op = msg.get("op")
            payload = msg.get("payload") or {}
            req_id = msg.get("req_id")
            if op == "shutdown":
                send_obj({"ok": True, "result": "bye", "req_id": req_id}, dst=src_rank, tag=TAG_RESP)
                break
            handler = None
            if op in self.control_handlers:
                handler = self.control_handlers[op]
            elif op in self.handlers:
                handler = self.handlers[op]
            if handler is None:
                send_obj({"ok": False, "error": f"unknown_op:{op}", "req_id": req_id}, dst=src_rank, tag=TAG_RESP)
                continue
            try:
                result = handler(**payload)
                send_obj({"ok": True, "result": result, "req_id": req_id}, dst=src_rank, tag=TAG_RESP)
            except Exception as exc:
                send_obj({"ok": False, "error": str(exc), "req_id": req_id}, dst=src_rank, tag=TAG_RESP)
