import importlib.abc
import importlib.machinery
import json
import math
import os
import sys
import threading
import types
from collections import defaultdict

import torch
from transformers import AutoConfig, AutoModelForCausalLM
from transformers.modeling_utils import init_empty_weights

try:
    from safetensors import safe_open as _safe_open
except Exception:
    _safe_open = None

os.environ.setdefault("TORCHDYNAMO_DISABLE", "1")
os.environ.setdefault("TORCH_DISABLE_DYNAMO", "1")
os.environ.setdefault("TORCH_COMPILE_DISABLE", "1")
os.environ.setdefault("TORCHINDUCTOR_DISABLE", "1")
os.environ.setdefault("TRANSFORMERS_NO_TORCHDYNAMO", "1")
os.environ.setdefault("DISABLE_TRITON", "1")

_TRITON_STUB_INSTALLED = False


def _install_stub_module(name, is_pkg=False, attrs=None):
    mod = sys.modules.get(name)
    if mod is None:
        mod = types.ModuleType(name)
        sys.modules[name] = mod
    if getattr(mod, "__spec__", None) is None:
        mod.__spec__ = importlib.machinery.ModuleSpec(name, loader=None, is_package=is_pkg)
        if is_pkg:
            mod.__spec__.submodule_search_locations = []
    if is_pkg and not hasattr(mod, "__path__"):
        mod.__path__ = []
    if attrs:
        for key, value in attrs.items():
            if not hasattr(mod, key):
                setattr(mod, key, value)
    if "." in name:
        parent_name, child_name = name.rsplit(".", 1)
        parent = sys.modules.get(parent_name)
        if parent is not None and not hasattr(parent, child_name):
            setattr(parent, child_name, mod)
    return mod


class _TritonStubFinder(importlib.abc.MetaPathFinder, importlib.abc.Loader):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == "triton" or fullname.startswith("triton."):
            return importlib.machinery.ModuleSpec(fullname, self, is_package=True)
        return None

    def create_module(self, spec):
        return None

    def exec_module(self, module):
        module.__spec__ = module.__spec__ or importlib.machinery.ModuleSpec(module.__name__, self, is_package=True)
        module.__spec__.submodule_search_locations = []
        module.__path__ = []
        if module.__name__ == "triton" and not hasattr(module, "language"):
            module.language = types.SimpleNamespace(dtype=object)


def _ensure_triton():
    os.environ.setdefault("TORCHDYNAMO_DISABLE", "1")
    os.environ.setdefault("TORCH_DISABLE_DYNAMO", "1")
    os.environ.setdefault("TORCH_COMPILE_DISABLE", "1")
    os.environ.setdefault("TORCHINDUCTOR_DISABLE", "1")
    os.environ.setdefault("TRANSFORMERS_NO_TORCHDYNAMO", "1")
    os.environ.setdefault("DISABLE_TRITON", "1")
    global _TRITON_STUB_INSTALLED
    if not _TRITON_STUB_INSTALLED:
        sys.meta_path.insert(0, _TritonStubFinder())
        _TRITON_STUB_INSTALLED = True
    _install_stub_module("triton", is_pkg=True, attrs={"language": types.SimpleNamespace(dtype=object)})
    _install_stub_module("triton.backends", is_pkg=True)
    _install_stub_module("triton.backends.compiler", is_pkg=False)
    _install_stub_module("triton.compiler", is_pkg=False)
    _install_stub_module("triton.compiler.compiler", is_pkg=False)
    _install_stub_module("triton.language", is_pkg=False, attrs={"dtype": object})
    _install_stub_module("triton.runtime", is_pkg=True)
    _install_stub_module("triton.runtime.jit", is_pkg=False)


_ensure_triton()
try:
    import torch.utils._triton as _torch_triton

    _torch_triton.has_triton_package = lambda: False
    _torch_triton.has_triton = lambda: False
    _torch_triton.has_triton_tma = lambda: False
    _torch_triton.has_triton_experimental_host_tma = lambda: False
    _torch_triton.has_triton_tensor_descriptor_host_tma = lambda: False
    _torch_triton.has_triton_tma_device = lambda: False
    _torch_triton.has_triton_stable_tma_api = lambda: False
except Exception:
    pass


def _device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _find_safetensors_index(model_dir: str):
    try:
        files = os.listdir(model_dir)
    except Exception:
        return None
    if "model.safetensors.index.json" in files:
        return os.path.join(model_dir, "model.safetensors.index.json")
    candidates = sorted([f for f in files if f.endswith(".safetensors.index.json")])
    if candidates:
        return os.path.join(model_dir, candidates[0])
    return None


def _find_single_safetensors(model_dir: str):
    try:
        files = os.listdir(model_dir)
    except Exception:
        return None
    if "model.safetensors" in files:
        return os.path.join(model_dir, "model.safetensors")
    candidates = [f for f in files if f.endswith(".safetensors")]
    if len(candidates) == 1:
        return os.path.join(model_dir, candidates[0])
    return None


def _build_index_prefixes(layer_range, load_embed: bool, load_lm_head: bool):
    start, end = layer_range
    prefixes = []
    for base in ("model.decoder.", "decoder."):
        prefixes.extend([f"{base}layers.{i}." for i in range(start, end)])
        if load_embed:
            prefixes.extend([f"{base}embed_tokens.", f"{base}embed_positions."])
    if load_lm_head:
        prefixes.extend(["lm_head.", "model.lm_head."])
    return prefixes


def _build_model_prefixes(model, layer_range, load_embed: bool, load_lm_head: bool):
    keys = list(model.state_dict().keys())
    has_model_decoder = any(k.startswith("model.decoder.") for k in keys)
    decoder_prefix = "model.decoder." if has_model_decoder else "decoder."
    has_model_lm_head = any(k.startswith("model.lm_head.") for k in keys)
    lm_head_prefix = "model.lm_head." if has_model_lm_head else "lm_head."
    start, end = layer_range
    prefixes = [f"{decoder_prefix}layers.{i}." for i in range(start, end)]
    if load_embed:
        prefixes.extend([f"{decoder_prefix}embed_tokens.", f"{decoder_prefix}embed_positions."])
    if load_lm_head:
        prefixes.append(lm_head_prefix)
    return prefixes


def _module_has_meta(module) -> bool:
    for param in module.parameters(recurse=True):
        if getattr(param, "is_meta", False):
            return True
    for buf in module.buffers(recurse=True):
        if getattr(buf, "is_meta", False):
            return True
    return False


def _partial_model_ready(model, layer_range, load_embed: bool, load_lm_head: bool) -> bool:
    decoder = model.model.decoder
    start, end = layer_range
    if load_embed:
        if _module_has_meta(decoder.embed_tokens):
            return False
        if _module_has_meta(decoder.embed_positions):
            return False
    for i in range(start, end):
        if _module_has_meta(decoder.layers[i]):
            return False
    if load_lm_head and _module_has_meta(model.lm_head):
        return False
    return True


def _load_safetensors_subset(model_dir: str, prefixes):
    if _safe_open is None:
        return None
    index_path = _find_safetensors_index(model_dir)
    state_dict = {}
    if index_path:
        try:
            with open(index_path, "r", encoding="utf-8") as f:
                index = json.load(f)
        except Exception:
            return None
        weight_map = index.get("weight_map", {})
        file_to_keys = defaultdict(list)
        for key, filename in weight_map.items():
            for prefix in prefixes:
                if key.startswith(prefix):
                    file_to_keys[filename].append(key)
                    break
        if not file_to_keys:
            return None
        for filename, keys in file_to_keys.items():
            path = os.path.join(model_dir, filename)
            with _safe_open(path, framework="pt", device="cpu") as f:
                for key in keys:
                    state_dict[key] = f.get_tensor(key)
        return state_dict
    single_path = _find_single_safetensors(model_dir)
    if not single_path:
        return None
    with _safe_open(single_path, framework="pt", device="cpu") as f:
        for key in f.keys():
            for prefix in prefixes:
                if key.startswith(prefix):
                    state_dict[key] = f.get_tensor(key)
                    break
    return state_dict if state_dict else None


def _remap_state_dict_keys(state_dict, model):
    if not state_dict:
        return state_dict
    keys = list(model.state_dict().keys())
    has_model_decoder = any(k.startswith("model.decoder.") for k in keys)
    has_decoder = any(k.startswith("decoder.") for k in keys)
    has_model_lm_head = any(k.startswith("model.lm_head.") for k in keys)
    remapped = {}
    for key, value in state_dict.items():
        new_key = key
        if has_model_decoder and key.startswith("decoder."):
            new_key = "model." + key
        elif has_decoder and key.startswith("model.decoder."):
            new_key = key[len("model.") :]
        if has_model_lm_head and new_key.startswith("lm_head."):
            new_key = "model." + new_key
        elif not has_model_lm_head and new_key.startswith("model.lm_head."):
            new_key = new_key[len("model.") :]
        remapped[new_key] = value
    return remapped


def _try_load_partial_model(model_name: str, layer_range, load_embed: bool, load_lm_head: bool):
    if _safe_open is None:
        return None
    if not model_name or not os.path.isdir(model_name):
        return None
    index_prefixes = _build_index_prefixes(layer_range, load_embed, load_lm_head)
    state_dict = _load_safetensors_subset(model_name, index_prefixes)
    if not state_dict:
        return None
    config = AutoConfig.from_pretrained(model_name)
    with init_empty_weights():
        model = AutoModelForCausalLM.from_config(config)
    state_dict = _remap_state_dict_keys(state_dict, model)
    prefixes = _build_model_prefixes(model, layer_range, load_embed, load_lm_head)
    try:
        incompatible = model.load_state_dict(state_dict, strict=False, assign=True)
    except TypeError:
        if hasattr(model, "to_empty"):
            model = model.to_empty(device="cpu")
        incompatible = model.load_state_dict(state_dict, strict=False)
    missing = getattr(incompatible, "missing_keys", []) or []
    critical_missing = []
    for key in missing:
        if any(key.startswith(prefix) for prefix in prefixes):
            if load_lm_head and (key.startswith("lm_head.") or key.startswith("model.lm_head.")):
                continue
            critical_missing.append(key)
    if critical_missing:
        return None
    if load_lm_head and hasattr(model, "tie_weights"):
        try:
            model.tie_weights()
        except Exception:
            pass
    if not _partial_model_ready(model, layer_range, load_embed, load_lm_head):
        return None
    return model


def build_layer_ranges(num_layers, num_stages):
    if num_stages <= 0:
        return []
    base = num_layers // num_stages
    rem = num_layers % num_stages
    counts = [base + (1 if i < rem else 0) for i in range(num_stages)]
    ranges = []
    start = 0
    for c in counts:
        end = start + c
        ranges.append((start, end))
        start = end
    return ranges


class PrefillStage:
    def __init__(self, stage_id, layer_range, model_name):
        self.stage_id = stage_id
        self.layer_range = layer_range
        self.model_name = model_name
        self.device = _device()
        self.model = None
        self.layers = None
        self.embed_tokens = None
        self.embed_positions = None
        self.num_layers = None
        self.pad_token_id = None

    def load(self):
        _ensure_triton()
        start, end = self.layer_range
        load_embed = start == 0
        self.model = _try_load_partial_model(self.model_name, self.layer_range, load_embed, False)
        load_mode = "partial" if self.model is not None else "full"
        if self.model is None:
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                torch_dtype=torch.float16,
                low_cpu_mem_usage=True,
            )
        print(f"[load] PrefillStage {self.stage_id} layers {start}-{end} mode={load_mode}")
        self.pad_token_id = self.model.config.pad_token_id
        decoder = self.model.model.decoder
        self.num_layers = len(decoder.layers)
        if start == 0:
            self.embed_tokens = decoder.embed_tokens.to(self.device)
            self.embed_positions = decoder.embed_positions.to(self.device)
        self.layers = [decoder.layers[i].to(self.device) for i in range(start, end)]
        self.model = None
        torch.cuda.empty_cache()

    def forward(self, input_ids, hidden_states, kv_cache, attention_mask):
        if self.layers is None:
            self.load()
        if hidden_states is None:
            input_ids = torch.tensor(input_ids, device=self.device)
            bs, seq = input_ids.shape
            pos_mask_2d = None
            if attention_mask is not None:
                if isinstance(attention_mask, torch.Tensor):
                    pos_mask_2d = attention_mask.to(self.device)
                else:
                    pos_mask_2d = torch.tensor(attention_mask, device=self.device)
                if pos_mask_2d.dim() == 2:
                    pos_mask_2d = pos_mask_2d.long()
            if hasattr(self.embed_positions, "offset"):
                if pos_mask_2d is not None:
                    pos_embeds = self.embed_positions(pos_mask_2d)
                else:
                    if self.pad_token_id is None:
                        pos_attention_mask = torch.ones((bs, seq), device=self.device, dtype=torch.long)
                    else:
                        pos_attention_mask = (input_ids != self.pad_token_id).long()
                    pos_embeds = self.embed_positions(pos_attention_mask)
            else:
                positions = torch.arange(seq, device=self.device).unsqueeze(0).expand(bs, seq)
                pos_embeds = self.embed_positions(positions)
            hidden_states = self.embed_tokens(input_ids) + pos_embeds
            causal = torch.triu(
                torch.full(
                    (bs, 1, seq, seq),
                    float("-inf"),
                    device=self.device,
                    dtype=hidden_states.dtype,
                ),
                diagonal=1,
            )
            if pos_mask_2d is not None:
                pad_mask = pos_mask_2d.unsqueeze(1).unsqueeze(2)
                attention_mask = causal.masked_fill(pad_mask == 0, float("-inf"))
            else:
                attention_mask = causal
        else:
            if isinstance(hidden_states, torch.Tensor):
                hidden_states = hidden_states.to(self.device)
            else:
                hidden_states = torch.tensor(hidden_states, device=self.device)
            if attention_mask is not None:
                if isinstance(attention_mask, torch.Tensor):
                    attention_mask = attention_mask.to(self.device)
                else:
                    attention_mask = torch.tensor(attention_mask, device=self.device)

        if kv_cache is None:
            kv_cache = [None for _ in range(self.num_layers)]

        start, end = self.layer_range
        for idx, layer in enumerate(self.layers, start=start):
            out = layer(hidden_states, attention_mask=attention_mask, use_cache=True)
            hidden_states = out[0]
            present = None
            if len(out) >= 3 and out[2] is not None:
                present = out[2]
            elif len(out) >= 2 and out[1] is not None:
                present = out[1]
            if present is not None:
                kv_cache[idx] = tuple(x.detach().cpu() for x in present)

        attn_out = attention_mask.detach().cpu() if attention_mask is not None else None
        return hidden_states.detach().cpu(), kv_cache, attn_out


class DecodeStage:
    def __init__(self, stage_id, layer_range, model_name, is_first, is_last):
        self.stage_id = stage_id
        self.layer_range = layer_range
        self.model_name = model_name
        self.is_first = is_first
        self.is_last = is_last
        self.device = _device()
        self.model = None
        self.layers = None
        self.embed_tokens = None
        self.embed_positions = None
        self.lm_head = None
        self.num_layers = None
        self.pad_token_id = None
        self._kv_cache = {}
        self._cache_lock = threading.Lock()

    def load(self):
        _ensure_triton()
        self.model = _try_load_partial_model(self.model_name, self.layer_range, self.is_first, self.is_last)
        load_mode = "partial" if self.model is not None else "full"
        if self.model is None:
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                torch_dtype=torch.float16,
                low_cpu_mem_usage=True,
            )
        start, end = self.layer_range
        print(
            f"[load] DecodeStage {self.stage_id} layers {start}-{end} first={self.is_first} last={self.is_last} mode={load_mode}"
        )
        self.pad_token_id = self.model.config.pad_token_id
        decoder = self.model.model.decoder
        self.num_layers = len(decoder.layers)
        start, end = self.layer_range
        if self.is_first:
            self.embed_tokens = decoder.embed_tokens.to(self.device)
            self.embed_positions = decoder.embed_positions.to(self.device)
        if self.is_last:
            self.lm_head = self.model.lm_head.to(self.device)
        self.layers = [decoder.layers[i].to(self.device) for i in range(start, end)]
        self.model = None
        torch.cuda.empty_cache()

    def init_kv(self, request_id, kv_cache):
        if self.layers is None:
            self.load()
        local_len = len(self.layers) if self.layers is not None else 0
        cache = [None for _ in range(local_len)]
        if kv_cache:
            for idx in range(min(local_len, len(kv_cache))):
                kv = kv_cache[idx]
                if kv is None:
                    cache[idx] = None
                    continue
                k, v = kv
                cache[idx] = (k.to(self.device), v.to(self.device))
        with self._cache_lock:
            self._kv_cache[request_id] = cache
        return True

    def clear_kv(self, request_id):
        with self._cache_lock:
            if request_id in self._kv_cache:
                del self._kv_cache[request_id]
        return True

    def decode_step(self, request_id, input_ids, hidden_states, past_len):
        if self.layers is None:
            self.load()
        with self._cache_lock:
            cache = self._kv_cache.get(request_id)
        if cache is None:
            cache = [None for _ in range(len(self.layers))]
            with self._cache_lock:
                self._kv_cache[request_id] = cache
        if hidden_states is None:
            input_ids = torch.tensor(input_ids, device=self.device)
            bs, seq = input_ids.shape
            if hasattr(self.embed_positions, "offset"):
                total_len = int(past_len) + seq
                if self.pad_token_id is None:
                    pos_attention_mask = torch.ones((bs, total_len), device=self.device, dtype=torch.long)
                else:
                    pos_attention_mask = torch.ones((bs, total_len), device=self.device, dtype=torch.long)
                    pos_attention_mask[:, -seq:] = (input_ids != self.pad_token_id).long()
                pos_embeds = self.embed_positions(pos_attention_mask, past_key_values_length=int(past_len))
            else:
                positions = torch.arange(
                    int(past_len),
                    int(past_len) + seq,
                    device=self.device,
                    dtype=torch.long,
                ).unsqueeze(0).expand(bs, seq)
                pos_embeds = self.embed_positions(positions)
            hidden_states = self.embed_tokens(input_ids) + pos_embeds
        else:
            if isinstance(hidden_states, torch.Tensor):
                hidden_states = hidden_states.to(self.device)
            else:
                hidden_states = torch.tensor(hidden_states, device=self.device)

        for local_idx, layer in enumerate(self.layers):
            past = cache[local_idx]
            out = layer(hidden_states, attention_mask=None, use_cache=True, past_key_value=past)
            hidden_states = out[0]
            present = None
            if len(out) >= 3 and out[2] is not None:
                present = out[2]
            elif len(out) >= 2 and out[1] is not None:
                present = out[1]
            if present is not None:
                cache[local_idx] = tuple(x.detach() for x in present)

        if self.lm_head is not None:
            logits = self.lm_head(hidden_states)
            return logits.detach().cpu()
        return hidden_states.detach().cpu()

    def decode_step_batch(self, request_ids, input_ids, hidden_states, past_len):
        if self.layers is None:
            self.load()
        if not request_ids:
            return None
        with self._cache_lock:
            caches = []
            for req_id in request_ids:
                cache = self._kv_cache.get(req_id)
                if cache is None:
                    cache = [None for _ in range(len(self.layers))]
                    self._kv_cache[req_id] = cache
                caches.append(cache)

        past_lens = []
        for cache in caches:
            past_len_i = 0
            for kv in cache:
                if kv is None:
                    continue
                k, _ = kv
                past_len_i = int(k.shape[2])
                break
            past_lens.append(past_len_i)
        max_past_len = max(past_lens) if past_lens else int(past_len)
        need_pad = any(pl != max_past_len for pl in past_lens)

        seq = None
        if hidden_states is None:
            input_ids = torch.tensor(input_ids, device=self.device)
            bs, seq = input_ids.shape
            if hasattr(self.embed_positions, "offset"):
                total_len = int(max_past_len) + seq
                pos_attention_mask = torch.zeros((bs, total_len), device=self.device, dtype=torch.long)
                for i, pl in enumerate(past_lens):
                    if pl > 0:
                        pos_attention_mask[i, :pl] = 1
                if self.pad_token_id is None:
                    pos_attention_mask[:, -seq:] = 1
                else:
                    pos_attention_mask[:, -seq:] = (input_ids != self.pad_token_id).long()
                pos_embeds = self.embed_positions(pos_attention_mask, past_key_values_length=int(max_past_len))
            else:
                base = torch.tensor(past_lens, device=self.device, dtype=torch.long).unsqueeze(1)
                positions = torch.arange(0, seq, device=self.device, dtype=torch.long).unsqueeze(0) + base
                pos_embeds = self.embed_positions(positions)
            hidden_states = self.embed_tokens(input_ids) + pos_embeds
        else:
            if isinstance(hidden_states, torch.Tensor):
                hidden_states = hidden_states.to(self.device)
            else:
                hidden_states = torch.tensor(hidden_states, device=self.device)
            seq = int(hidden_states.shape[1])

        attn_mask = None
        if need_pad and max_past_len > 0:
            dtype = hidden_states.dtype if isinstance(hidden_states, torch.Tensor) else self.embed_tokens.weight.dtype
            mask_value = torch.finfo(dtype).min
            attn_mask = torch.zeros(
                (len(caches), 1, seq, int(max_past_len) + seq),
                device=self.device,
                dtype=dtype,
            )
            for i, pl in enumerate(past_lens):
                if pl < max_past_len:
                    attn_mask[i, :, :, pl:max_past_len] = mask_value

        for local_idx, layer in enumerate(self.layers):
            batch_k = []
            batch_v = []
            for cache in caches:
                kv = cache[local_idx]
                if kv is None:
                    batch_k = []
                    batch_v = []
                    break
                k, v = kv
                batch_k.append(k)
                batch_v.append(v)
            past = None
            if batch_k:
                if need_pad:
                    for i, (k, v) in enumerate(zip(batch_k, batch_v)):
                        cur_len = int(k.shape[2])
                        if cur_len < max_past_len:
                            pad_len = int(max_past_len) - cur_len
                            pad_shape = (k.shape[0], k.shape[1], pad_len, k.shape[3])
                            pad_k = k.new_zeros(pad_shape)
                            pad_v = v.new_zeros(pad_shape)
                            batch_k[i] = torch.cat([k, pad_k], dim=2)
                            batch_v[i] = torch.cat([v, pad_v], dim=2)
                past = (torch.cat(batch_k, dim=0), torch.cat(batch_v, dim=0))
            out = layer(hidden_states, attention_mask=attn_mask, use_cache=True, past_key_value=past)
            hidden_states = out[0]
            present = None
            if len(out) >= 3 and out[2] is not None:
                present = out[2]
            elif len(out) >= 2 and out[1] is not None:
                present = out[1]
            if present is not None:
                pk, pv = present
                for i, cache in enumerate(caches):
                    if need_pad:
                        new_len = past_lens[i] + seq
                        cache[local_idx] = (
                            pk[i : i + 1, :, :new_len, :].detach(),
                            pv[i : i + 1, :, :new_len, :].detach(),
                        )
                    else:
                        cache[local_idx] = (pk[i : i + 1].detach(), pv[i : i + 1].detach())

        with self._cache_lock:
            for req_id, cache in zip(request_ids, caches):
                self._kv_cache[req_id] = cache

        if self.lm_head is not None:
            logits = self.lm_head(hidden_states)
            return logits.detach().cpu()
        return hidden_states.detach().cpu()


class DecodeWorker:
    def __init__(self, model_name):
        self.model_name = model_name
        self.device = _device()
        self.model = None

    def load(self):
        _ensure_triton()
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True,
        ).to(self.device)

    def prefill(self, input_ids, to_cpu: bool = True):
        if self.model is None:
            self.load()
        if not isinstance(input_ids, torch.Tensor):
            input_ids = torch.tensor(input_ids, device=self.device)
        attention_mask = torch.ones_like(input_ids, device=self.device)
        with torch.no_grad():
            outputs = self.model(input_ids, attention_mask=attention_mask, use_cache=True)
        past = outputs.past_key_values
        if past is None:
            return None
        kv_cache = []
        for k, v in past:
            if to_cpu:
                kv_cache.append((k.detach().cpu(), v.detach().cpu()))
            else:
                kv_cache.append((k.detach(), v.detach()))
        return kv_cache

    def decode(self, input_ids, kv_cache, max_new_tokens):
        if self.model is None:
            self.load()
        input_ids = torch.tensor(input_ids, device=self.device)
        past = None
        if kv_cache is not None:
            past = []
            for kv in kv_cache:
                if kv is None:
                    past.append(None)
                    continue
                k, v = kv
                past.append((k.to(self.device), v.to(self.device)))
            past = tuple(past)

        generated = input_ids
        for _ in range(max_new_tokens):
            outputs = self.model(generated[:, -1:], past_key_values=past, use_cache=True)
            logits = outputs.logits[:, -1, :]
            next_token = torch.argmax(logits, dim=-1, keepdim=True)
            generated = torch.cat([generated, next_token], dim=-1)
            past = outputs.past_key_values
        return generated.detach().cpu().tolist()
