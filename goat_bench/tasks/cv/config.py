# goat_bench/tasks/cv/config.py
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Optional


TaskName = Literal["cls", "det", "seg"]


@dataclass
class CVTaskConfig:
    """Unified configuration used by the CV runners."""

    task: TaskName
    data_dir: Path
    optimizer: str

    # Shared knobs
    dataset: str = "imagenet"
    model: str = "resnet50"
    lr: Optional[float] = None
    weight_decay: float = 1e-2
    epochs: Optional[int] = None
    batch_size: int = 128
    batch_override: bool = False
    workers: int = 8
    amp: Literal["none", "fp16", "bf16"] = "none"
    warmup_epochs: int = 5
    subset_frac: float = 1.0
    seed: int = 42

    # Reporting / instrumentation
    ttt_target: Optional[float] = None
    log_csv: Optional[Path] = None
    log_json: Optional[Path] = None

    # Optimizer-specific knobs
    rico_bk_beta: float = 0.9
    rico_k_cap: float = 0.08
    rico_g_floor: float = 1e-3
    rico_sync_every: int = 20
    lion_beta1: float = 0.9
    lion_beta2: float = 0.99
    soap_args: Optional[str] = None

    def with_updates(self, **kwargs) -> "CVTaskConfig":
        """Convenience helper to clone configs while overriding fields."""
        data = {**self.__dict__, **kwargs}
        return CVTaskConfig(**data)
