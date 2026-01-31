import argparse
import json
import os
from pathlib import Path


def load_jsonl(path: Path):
    items = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            items.append(json.loads(line))
    return items


def latest_run_dir(exp_dir: Path) -> Path:
    runs = [p for p in exp_dir.iterdir() if p.is_dir() and p.name.startswith("run_")]
    if not runs:
        raise FileNotFoundError(f"no run_* under {exp_dir}")
    return max(runs, key=lambda p: p.stat().st_mtime)


def stats(values):
    if not values:
        return {"n": 0, "avg": 0.0, "p50": 0.0, "p95": 0.0}
    values = sorted(values)
    n = len(values)
    avg = sum(values) / n
    p50 = values[int(0.5 * (n - 1))]
    p95 = values[int(0.95 * (n - 1))]
    return {"n": n, "avg": avg, "p50": p50, "p95": p95}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_dir", required=True, help="run_YYYYMMDD_HHMMSS 目录或 exp_* 目录")
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    if run_dir.is_dir() and run_dir.name.startswith("exp_"):
        run_dir = latest_run_dir(run_dir)

    request_log = run_dir / "request.log"
    if not request_log.exists():
        raise FileNotFoundError(f"request.log not found: {request_log}")

    reqs = load_jsonl(request_log)

    prefill_wait = []
    decode_wait = []
    prefill_latency = []
    decode_latency = []
    total_latency = []

    for r in reqs:
        if r.get("prefill_start_ns") and r.get("arrival_time_ns"):
            prefill_wait.append((r["prefill_start_ns"] - r["arrival_time_ns"]) / 1e6)
        if r.get("decode_start_ns") and r.get("prefill_end_ns"):
            decode_wait.append((r["decode_start_ns"] - r["prefill_end_ns"]) / 1e6)
        if r.get("prefill_latency_ms") is not None:
            prefill_latency.append(float(r.get("prefill_latency_ms", 0.0)))
        if r.get("decode_latency_ms") is not None:
            decode_latency.append(float(r.get("decode_latency_ms", 0.0)))
        if r.get("total_latency_ms") is not None:
            total_latency.append(float(r.get("total_latency_ms", 0.0)))

    print(f"run_dir: {run_dir}")
    print("prefill_queue_wait_ms", stats(prefill_wait))
    print("decode_queue_wait_ms", stats(decode_wait))
    print("prefill_latency_ms", stats(prefill_latency))
    print("decode_latency_ms", stats(decode_latency))
    print("total_latency_ms", stats(total_latency))


if __name__ == "__main__":
    main()
