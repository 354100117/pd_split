import json
import socket
import time

DEFAULT_TEST_REQUESTS = [
    {"request_id": "req-1", "prompt": "Hello", "max_new_tokens": 32},
    {"request_id": "req-2", "prompt": "Explain edge AI.", "max_new_tokens": 64},
]


def load_cluster_grouped(path):
    with open(path, "r") as f:
        return json.load(f)


def split_groups(cluster_data):
    prefill = [n for n, v in cluster_data.items() if v.get("group") == "prefill"]
    decode = [n for n, v in cluster_data.items() if v.get("group") == "decode"]
    return prefill, decode


def log(msg):
    ts = time.strftime("%H:%M:%S")
    print(f"{ts} - {msg}", flush=True)


def resolve_host_ip(host):
    try:
        return socket.gethostbyname(host)
    except Exception:
        return host


def get_tokenizer(tokenizer_dir):
    if not tokenizer_dir:
        return None
    try:
        from transformers import AutoTokenizer
    except Exception:
        return None
    return AutoTokenizer.from_pretrained(tokenizer_dir)


def encode_prompt(prompt, tokenizer):
    if tokenizer is None:
        ids = [min(ord(c), 255) for c in prompt][:128]
        return ids if ids else [0]
    ids = tokenizer.encode(prompt)
    if not ids:
        fallback = tokenizer.bos_token_id
        if fallback is None:
            fallback = tokenizer.eos_token_id
        ids = [fallback if fallback is not None else 0]
    return ids
