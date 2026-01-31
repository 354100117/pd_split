from dataclasses import dataclass
from typing import List, Tuple, Union, Optional

import torch

from backend.dist import RPC_PRIORITY_HIGH, RPC_PRIORITY_LOW
from backend.logger import now_ns


@dataclass
class StageHandle:
    rank: int
    node: str
    layer_range: Tuple[int, int]
    stage_id: int
    stage: Optional[object] = None

    def is_local(self) -> bool:
        return self.stage is not None


def _iter_req_ids(req_ids: Union[str, List[str]]) -> List[str]:
    if isinstance(req_ids, str):
        return [req_ids]
    return list(req_ids)


def _call_prefill(stage: StageHandle, rpc, input_ids, hidden_states, kv_cache, attention_mask):
    if stage.stage is not None:
        return stage.stage.forward(input_ids, hidden_states, kv_cache, attention_mask)
    return rpc.call(
        stage.rank,
        "prefill_forward",
        {
            "input_ids": input_ids,
            "hidden_states": hidden_states,
            "kv_cache": kv_cache,
            "attention_mask": attention_mask,
        },
        priority=RPC_PRIORITY_LOW,
    )


def _call_decode(stage: StageHandle, rpc, request_id, input_ids, hidden_states, past_len):
    if stage.stage is not None:
        return stage.stage.decode_step(request_id, input_ids, hidden_states, past_len)
    return rpc.call(
        stage.rank,
        "decode_step",
        {
            "request_id": request_id,
            "input_ids": input_ids,
            "hidden_states": hidden_states,
            "past_len": past_len,
        },
        priority=RPC_PRIORITY_HIGH,
    )


def _call_decode_batch(stage: StageHandle, rpc, request_ids, input_ids, hidden_states, past_len):
    if stage.stage is not None:
        return stage.stage.decode_step_batch(request_ids, input_ids, hidden_states, past_len)
    return rpc.call(
        stage.rank,
        "decode_step_batch",
        {
            "request_ids": request_ids,
            "input_ids": input_ids,
            "hidden_states": hidden_states,
            "past_len": past_len,
        },
        priority=RPC_PRIORITY_HIGH,
    )


def _call_init_kv(stage: StageHandle, rpc, request_id, kv_cache):
    if stage.stage is not None:
        return stage.stage.init_kv(request_id, kv_cache)
    return rpc.call(
        stage.rank,
        "init_kv",
        {
            "request_id": request_id,
            "kv_cache": kv_cache,
        },
        priority=RPC_PRIORITY_HIGH,
    )


def _call_clear_kv(stage: StageHandle, rpc, request_id):
    if stage.stage is not None:
        return stage.stage.clear_kv(request_id)
    return rpc.call(
        stage.rank,
        "clear_kv",
        {
            "request_id": request_id,
        },
        priority=RPC_PRIORITY_HIGH,
    )


def _slice_kv(kv_cache, layer_range):
    if kv_cache is None:
        return None
    start, end = layer_range
    return kv_cache[start:end]


def run_prefill_logged(stages: List[StageHandle], input_ids, attention_mask, tracker, req_ids: Union[str, List[str]], rpc):
    hidden = None
    kv_cache = None
    attn = attention_mask
    req_list = _iter_req_ids(req_ids)
    for stage in stages:
        start_ns = now_ns()
        hidden, kv_cache, attn = _call_prefill(stage, rpc, input_ids, hidden, kv_cache, attn)
        end_ns = now_ns()
        layer_range = f"{stage.layer_range[0]}-{stage.layer_range[1]}"
        for req_id in req_list:
            tracker.pipeline_stage(
                req_id=req_id,
                stage="prefill",
                stage_id=stage.stage_id,
                node=stage.node,
                layer_range=layer_range,
                start_ns=start_ns,
                end_ns=end_ns,
            )
    return kv_cache


def run_decode_logged(stages: List[StageHandle], input_ids, kv_cache, max_new_tokens: int, tracker, req_id: str, rpc):
    if not stages:
        return []
    generated = torch.tensor(input_ids)
    if kv_cache is not None:
        for stage in stages:
            kv_part = _slice_kv(kv_cache, stage.layer_range)
            _call_init_kv(stage, rpc, req_id, kv_part)
    else:
        for stage in stages:
            _call_init_kv(stage, rpc, req_id, None)
    past_len = 0
    if kv_cache is not None:
        for kv in kv_cache:
            if kv is None:
                continue
            k, _ = kv
            past_len = int(k.shape[2])
            break
    for _ in range(max_new_tokens):
        hidden = None
        for stage in stages:
            start_ns = now_ns()
            hidden = _call_decode(stage, rpc, req_id, generated[:, -1:].tolist(), hidden, past_len)
            end_ns = now_ns()
            layer_range = f"{stage.layer_range[0]}-{stage.layer_range[1]}"
            tracker.pipeline_stage(
                req_id=req_id,
                stage="decode",
                stage_id=stage.stage_id,
                node=stage.node,
                layer_range=layer_range,
                start_ns=start_ns,
                end_ns=end_ns,
            )
        logits = hidden if isinstance(hidden, torch.Tensor) else torch.tensor(hidden)
        if logits.dim() == 2:
            logits = logits.unsqueeze(1)
        if logits.shape[1] == 0:
            raise RuntimeError("decode logits seq length is 0; check input_ids encoding and prompt content")
        next_token = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)
        generated = torch.cat([generated, next_token], dim=-1)
        past_len += 1
    for stage in stages:
        _call_clear_kv(stage, rpc, req_id)
    return generated.tolist()


def run_decode_logged_batch(
    stages: List[StageHandle],
    input_ids_list,
    kv_cache_list,
    max_new_tokens: int,
    tracker,
    req_ids: List[str],
    rpc,
):
    if not stages or not req_ids:
        return []
    for stage in stages:
        for req_id, kv_cache in zip(req_ids, kv_cache_list):
            kv_part = _slice_kv(kv_cache, stage.layer_range)
            _call_init_kv(stage, rpc, req_id, kv_part)
    generated = [list(ids) for ids in input_ids_list]
    past_len = 0
    first_kv = kv_cache_list[0] if kv_cache_list else None
    if first_kv is not None:
        for kv in first_kv:
            if kv is None:
                continue
            k, _ = kv
            past_len = int(k.shape[2])
            break
    for _ in range(max_new_tokens):
        hidden = None
        token_batch = [[seq[-1]] for seq in generated]
        for stage in stages:
            start_ns = now_ns()
            hidden = _call_decode_batch(stage, rpc, req_ids, token_batch, hidden, past_len)
            end_ns = now_ns()
            layer_range = f"{stage.layer_range[0]}-{stage.layer_range[1]}"
            for req_id in req_ids:
                tracker.pipeline_stage(
                    req_id=req_id,
                    stage="decode",
                    stage_id=stage.stage_id,
                    node=stage.node,
                    layer_range=layer_range,
                    start_ns=start_ns,
                    end_ns=end_ns,
                )
        logits = hidden if isinstance(hidden, torch.Tensor) else torch.tensor(hidden)
        if logits.dim() == 2:
            logits = logits.unsqueeze(1)
        if logits.shape[1] == 0:
            raise RuntimeError("decode logits seq length is 0; check input_ids encoding and prompt content")
        next_tokens = torch.argmax(logits[:, -1, :], dim=-1).tolist()
        for i, tok in enumerate(next_tokens):
            generated[i].append(int(tok))
        past_len += 1
    for stage in stages:
        for req_id in req_ids:
            _call_clear_kv(stage, rpc, req_id)
    return generated


def run_decode_worker(worker, input_ids, kv_cache, max_new_tokens):
    if worker is None:
        return []
    return worker.decode(input_ids, kv_cache, max_new_tokens)
