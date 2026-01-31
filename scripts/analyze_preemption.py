#!/usr/bin/env python3
"""
Analyze preemption between later prefill and earlier decode, and plot timeline.

Example:
  python whole_process/scripts/analyze_preemption.py \
    --request-log whole_process/logs/exp_full_pipeline/run_xxx/request.log \
    --pipeline-log whole_process/logs/exp_full_pipeline/run_xxx/pipeline.log \
    --first-n 4 \
    --second-n 8 \
    --out-dir whole_process/logs/exp_full_pipeline/run_xxx
"""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Tuple


def _read_json_lines(path: Path) -> Iterable[Dict]:
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except Exception:
                continue


def _merge_intervals(intervals: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
    if not intervals:
        return []
    intervals.sort()
    merged: List[Tuple[int, int]] = []
    cur_s, cur_e = intervals[0]
    for s, e in intervals[1:]:
        if s > cur_e:
            merged.append((cur_s, cur_e))
            cur_s, cur_e = s, e
        else:
            if e > cur_e:
                cur_e = e
    merged.append((cur_s, cur_e))
    return merged


def _overlap_len(s: int, e: int, intervals: List[Tuple[int, int]]) -> int:
    total = 0
    for a, b in intervals:
        if s < b and e > a:
            total += max(0, min(e, b) - max(s, a))
    return total


def _percentile(values: List[float], p: float) -> float:
    if not values:
        return 0.0
    values = sorted(values)
    k = (len(values) - 1) * p
    f = math.floor(k)
    c = math.ceil(k)
    if f == c:
        return values[int(k)]
    return values[f] + (values[c] - values[f]) * (k - f)


def main() -> int:
    parser = argparse.ArgumentParser(description="Analyze prefill/decoder preemption.")
    parser.add_argument("--request-log", required=True, type=Path)
    parser.add_argument("--pipeline-log", required=True, type=Path)
    parser.add_argument("--first-n", type=int, default=4)
    parser.add_argument("--second-n", type=int, default=8)
    parser.add_argument("--group-by", choices=["submit", "arrival"], default="submit")
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--out-prefix", default="preemption")
    args = parser.parse_args()

    try:
        import matplotlib.pyplot as plt
    except Exception as exc:  # pragma: no cover
        print("Missing dependency: matplotlib is required for plotting.")
        print(str(exc))
        return 2

    request_summaries: List[Dict] = []
    for obj in _read_json_lines(args.request_log):
        if obj.get("type") == "request_summary":
            request_summaries.append(obj)

    if not request_summaries:
        print("No request_summary entries found.")
        return 1

    if args.group_by == "submit":
        request_summaries.sort(key=lambda r: (r.get("submit_order", 0), r.get("arrival_time_ns", 0)))
    else:
        request_summaries.sort(key=lambda r: (r.get("arrival_time_ns", 0), r.get("submit_order", 0)))

    first_ids = [r["request_id"] for r in request_summaries[: args.first_n]]
    second_ids = [r["request_id"] for r in request_summaries[args.first_n : args.first_n + args.second_n]]

    pipeline = list(_read_json_lines(args.pipeline_log))

    prefill_intervals: List[Tuple[int, int]] = []
    prefill_spans: Dict[str, List[int]] = defaultdict(list)
    decode_events: Dict[str, List[Dict]] = defaultdict(list)

    for obj in pipeline:
        if obj.get("type") != "pipeline_stage":
            continue
        rid = obj.get("request_id")
        stage = obj.get("stage")
        if stage == "prefill":
            if rid in second_ids:
                prefill_intervals.append((obj["start_ns"], obj["end_ns"]))
            prefill_spans[rid].append(obj["start_ns"])
            prefill_spans[rid].append(obj["end_ns"])
        elif stage == "decode":
            if rid in first_ids:
                decode_events[rid].append(obj)

    merged_prefill = _merge_intervals(prefill_intervals)

    # Decode overlap stats
    inside_lat = []
    outside_lat = []
    overlap_decode_ns = 0
    total_decode_ns = 0
    overlap_event_count = 0
    total_event_count = 0

    for rid, events in decode_events.items():
        events.sort(key=lambda e: e["start_ns"])
        for e in events:
            s, e_ns = e["start_ns"], e["end_ns"]
            dur = e_ns - s
            total_decode_ns += dur
            total_event_count += 1
            ov = _overlap_len(s, e_ns, merged_prefill)
            if ov > 0:
                overlap_decode_ns += ov
                overlap_event_count += 1
                inside_lat.append(e.get("latency_ms", 0.0))
            else:
                outside_lat.append(e.get("latency_ms", 0.0))

    def _avg(vals: List[float]) -> float:
        return sum(vals) / len(vals) if vals else 0.0

    # Text summary
    print("first_ids:", first_ids)
    print("second_ids:", second_ids)
    print("prefill_merged_intervals:", len(merged_prefill))
    print("decode_events_total:", total_event_count, "overlap_events:", overlap_event_count)
    if total_decode_ns:
        print("decode_overlap_ratio_time:", overlap_decode_ns / total_decode_ns)
    if total_event_count:
        print("decode_overlap_ratio_events:", overlap_event_count / total_event_count)
    print("decode_latency_inside_avg_ms:", _avg(inside_lat), "p95:", _percentile(inside_lat, 0.95))
    print("decode_latency_outside_avg_ms:", _avg(outside_lat), "p95:", _percentile(outside_lat, 0.95))

    # Plot
    args.out_dir.mkdir(parents=True, exist_ok=True)
    fig, (ax_top, ax_bottom) = plt.subplots(2, 1, figsize=(14, 8), sharex=False)

    # Top: decode latency scatter with prefill shading
    base_ns = None
    for rid in first_ids:
        for e in decode_events.get(rid, []):
            if base_ns is None or e["start_ns"] < base_ns:
                base_ns = e["start_ns"]
    if base_ns is None:
        base_ns = merged_prefill[0][0] if merged_prefill else 0

    for (s, e) in merged_prefill:
        ax_top.axvspan((s - base_ns) / 1e6, (e - base_ns) / 1e6, color="#ffcccc", alpha=0.4, linewidth=0)

    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b"]
    for idx, rid in enumerate(first_ids):
        events = decode_events.get(rid, [])
        xs = [(e["start_ns"] - base_ns) / 1e6 for e in events]
        ys = [e.get("latency_ms", 0.0) for e in events]
        ax_top.scatter(xs, ys, s=6, alpha=0.4, color=colors[idx % len(colors)], label=rid)

    ax_top.set_title("Decode latency (first group) with later prefill windows shaded")
    ax_top.set_xlabel("Time from first decode start (ms)")
    ax_top.set_ylabel("Decode stage latency (ms)")
    ax_top.legend(loc="upper right", fontsize=8, ncol=2)

    # Bottom: request-level spans (prefill+decode)
    y = 0
    yticks = []
    ylabels = []

    def _span(name: str, start_ns: int, end_ns: int, color: str):
        nonlocal y
        ax_bottom.broken_barh(
            [((start_ns - base_ns) / 1e6, (end_ns - start_ns) / 1e6)],
            (y - 0.4, 0.8),
            facecolors=color,
        )
        yticks.append(y)
        ylabels.append(name)
        y += 1

    # Prefill spans for second group and decode spans for first group
    req_map = {r["request_id"]: r for r in request_summaries}

    for rid in first_ids:
        r = req_map.get(rid, {})
        ds = r.get("decode_start_ns")
        de = r.get("decode_end_ns")
        if ds and de:
            _span(f"{rid} decode", ds, de, "#4c78a8")

    for rid in second_ids:
        spans = prefill_spans.get(rid)
        if spans:
            s = min(spans)
            e = max(spans)
            _span(f"{rid} prefill", s, e, "#f58518")

    ax_bottom.set_title("Request spans (decode for first group, prefill for second group)")
    ax_bottom.set_xlabel("Time from first decode start (ms)")
    ax_bottom.set_yticks(yticks)
    ax_bottom.set_yticklabels(ylabels, fontsize=8)

    fig.tight_layout()
    out_path = args.out_dir / f"{args.out_prefix}.png"
    fig.savefig(out_path, dpi=150)
    print(f"Saved plot: {out_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
