# goat_bench/tasks/cv/cli.py
from __future__ import annotations

import argparse
from pathlib import Path

from .config import CVTaskConfig


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser("GOAT CV benchmark (classification/detection/segmentation)")
    parser.add_argument("--task", type=str, required=True, choices=["cls", "det", "seg"])
    parser.add_argument("--dataset", type=str, default="imagenet", help="[cls] cifar100|tinyimagenet|imagenet")
    parser.add_argument(
        "--model",
        type=str,
        default="resnet50",
        choices=["resnet18", "resnet34", "resnet50", "resnet101", "resnet152"],
        help="[cls] backbone",
    )
    parser.add_argument("--data", type=str, required=True, help="dataset root directory")
    parser.add_argument("--optimizer", type=str, required=True, choices=["rico", "adamw", "lion", "soap"])
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--wd", type=float, default=1e-2, help="weight decay")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--amp", type=str, default="none", choices=["none", "fp16", "bf16"])
    parser.add_argument("--warmup-epochs", type=int, default=5, help="[cls] only")
    parser.add_argument("--subset-frac", type=float, default=1.0, help="0<frac<=1.0: subset of the data")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--ttt-target", type=float, default=None, help="metric threshold for time-to-target")
    parser.add_argument("--log-csv", type=str, default=None, help="per-epoch CSV path")
    parser.add_argument("--log-json", type=str, default=None, help="final JSON summary path")

    parser.add_argument("--rico_bk_beta", type=float, default=0.9)
    parser.add_argument("--rico_k_cap", type=float, default=0.08)
    parser.add_argument("--rico_g_floor", type=float, default=1e-3)
    parser.add_argument("--rico_sync_every", type=int, default=20)

    parser.add_argument("--lion_beta1", type=float, default=0.9)
    parser.add_argument("--lion_beta2", type=float, default=0.99)
    parser.add_argument("--soap_args", type=str, default=None, help="JSON string for additional SOAP kwargs")
    return parser


def args_to_config(args: argparse.Namespace) -> CVTaskConfig:
    return CVTaskConfig(
        task=args.task.lower(),
        data_dir=Path(args.data),
        optimizer=args.optimizer.lower(),
        dataset=args.dataset.lower(),
        model=args.model.lower(),
        lr=args.lr,
        weight_decay=args.wd,
        epochs=args.epochs,
        batch_size=args.batch_size,
        workers=args.workers,
        amp=args.amp,
        warmup_epochs=args.warmup_epochs,
        subset_frac=args.subset_frac,
        seed=args.seed,
        ttt_target=args.ttt_target,
        log_csv=Path(args.log_csv) if args.log_csv else None,
        log_json=Path(args.log_json) if args.log_json else None,
        rico_bk_beta=args.rico_bk_beta,
        rico_k_cap=args.rico_k_cap,
        rico_g_floor=args.rico_g_floor,
        rico_sync_every=args.rico_sync_every,
        lion_beta1=args.lion_beta1,
        lion_beta2=args.lion_beta2,
        soap_args=args.soap_args,
    )


def main(argv: list[str] | None = None):
    parser = build_parser()
    args = parser.parse_args(argv)
    cfg = args_to_config(args)
    from .runners import run_task

    run_task(cfg)


if __name__ == "__main__":
    main()
