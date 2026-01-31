import math
from typing import Dict, List, Tuple


def _weighted_ranges(num_layers: int, nodes: List[str], weights: List[float]) -> List[Tuple[int, int]]:
    if num_layers <= 0 or not nodes:
        return []
    safe = [max(float(w), 1e-6) for w in weights]
    total = sum(safe)
    raw = [num_layers * w / total for w in safe]
    counts = [max(1, int(math.floor(x))) for x in raw]
    if num_layers < len(counts):
        counts = [1] * num_layers + [0] * (len(counts) - num_layers)
    diff = num_layers - sum(counts)
    if diff != 0:
        frac = [r - math.floor(r) for r in raw]
        order = sorted(range(len(counts)), key=lambda i: frac[i], reverse=(diff > 0))
        idx = 0
        while diff != 0 and idx < len(order):
            i = order[idx]
            if diff > 0:
                counts[i] += 1
                diff -= 1
            else:
                if counts[i] > 1:
                    counts[i] -= 1
                    diff += 1
            idx += 1
    ranges = []
    start = 0
    for c in counts:
        end = start + c
        ranges.append((start, end))
        start = end
    return ranges


def build_ranges(num_layers: int, nodes: List[str], cluster: Dict, strategy: str) -> List[Tuple[int, int]]:
    if strategy == "auto_prefill":
        weights = [
            0.6 * float(cluster.get(n, {}).get("compute_gpu", 0.0))
            + 0.25 * float(cluster.get(n, {}).get("gpu_bw_gb_s", 0.0))
            + 0.15 * float(cluster.get(n, {}).get("gpu_mem_gb", 0.0))
            for n in nodes
        ]
        return _weighted_ranges(num_layers, nodes, weights)
    if strategy == "auto_decode":
        weights = [
            0.45 * float(cluster.get(n, {}).get("gpu_bw_gb_s", 0.0))
            + 0.35 * float(cluster.get(n, {}).get("gpu_mem_gb", 0.0))
            + 0.2 * float(cluster.get(n, {}).get("compute_gpu", 0.0))
            for n in nodes
        ]
        return _weighted_ranges(num_layers, nodes, weights)
    if strategy == "compute":
        weights = [float(cluster.get(n, {}).get("compute_gpu", 0.0)) for n in nodes]
        return _weighted_ranges(num_layers, nodes, weights)
    if strategy == "mem":
        weights = [float(cluster.get(n, {}).get("gpu_mem_gb", 0.0)) for n in nodes]
        return _weighted_ranges(num_layers, nodes, weights)
    if strategy == "bandwidth":
        weights = [float(cluster.get(n, {}).get("gpu_bw_gb_s", 0.0)) for n in nodes]
        return _weighted_ranges(num_layers, nodes, weights)
    return _weighted_ranges(num_layers, nodes, [1.0 for _ in nodes])
