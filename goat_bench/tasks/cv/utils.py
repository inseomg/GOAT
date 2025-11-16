# goat_bench/tasks/cv/utils.py
from __future__ import annotations

import csv
import math
import random
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence

import torch
from torch.utils.data import Subset


def set_seed(seed: int):
    """Set RNG seeds across Python, NumPy (if installed), and PyTorch."""
    random.seed(seed)
    try:
        import numpy as np  # optional dependency

        np.random.seed(seed)
    except Exception:
        pass
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def percentile(values: Sequence[float], q: float) -> float:
    if not values:
        return float("nan")
    ys = sorted(values)
    k = (len(ys) - 1) * (q / 100.0)
    f = math.floor(k)
    c = math.ceil(k)
    if f == c:
        return ys[int(k)]
    return ys[f] + (ys[c] - ys[f]) * (k - f)


def subset_dataset(dataset, frac: float, seed: int = 42):
    """Return a deterministic subset view (used for quick smoke tests)."""
    if frac >= 1.0:
        return dataset
    idx = list(range(len(dataset)))
    rnd = random.Random(seed)
    rnd.shuffle(idx)
    k = max(1, int(len(idx) * max(min(frac, 1.0), 0.0001)))
    return Subset(dataset, idx[:k])


def write_csv_row(path: Path | None, row: Dict[str, Any], header_first: bool = True):
    if path is None:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    write_header = header_first and (not path.exists())
    with path.open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def human_mb(x_bytes: int) -> float:
    return round(x_bytes / (1024.0 * 1024.0), 2)
