# goat_bench/utils/checkpointing.py
from __future__ import annotations
from pathlib import Path
from typing import Any, Dict

import torch

from .helpers import ensure_dir


def save_checkpoint(path: Path, state: Dict[str, Any]):
    ensure_dir(path.parent)
    torch.save(state, path)


def load_checkpoint(path: Path) -> Dict[str, Any]:
    return torch.load(path, map_location="cpu")