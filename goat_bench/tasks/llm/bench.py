from __future__ import annotations

"""
GOAT LLM Benchmark (L1/L2/L3)
- L1: Small GPT-style pretrain (WikiText-103)
- L2: 1B SFT (default alpaca-style dataset)
- L3: 7-8B SFT (same pipeline as L2)

Ported from user-supplied bench script, adapted for GOAT repo layout.
"""

import argparse
import json
import math
import os
import random
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn  # noqa: F401  # kept for completeness / potential extensions
import torch.optim as optim
from torch.utils.data import DataLoader

try:
    from datasets import load_dataset
except Exception as exc:
    raise RuntimeError("`pip install datasets` 패키지가 필요합니다.") from exc

try:
    from transformers import (
        AutoConfig,
        AutoModelForCausalLM,
        AutoTokenizer,
    )
except Exception as exc:
    raise RuntimeError("`pip install transformers` 패키지가 필요합니다.") from exc

try:
    from torch.utils.data import Dataset  # type: ignore
except Exception:
    pass

LionOpt = None
SOAPOpt = None
try:
    from lion_pytorch import Lion as LionOpt
except Exception:
    pass
try:
    import pytorch_optimizer as _pyo

    SOAPOpt = getattr(_pyo, "SOAP", None)
except Exception:
    pass

try:
    from rico import RICO, rico_layerwise_groups
except Exception:
    RICO, rico_layerwise_groups = None, None
    avail = ["adamw"]
    if LionOpt is not None:
        avail.append("lion")
    if SOAPOpt is not None:
        avail.append("soap")
    print(f"[warn] rico.py 미발견 → --optimizer {', '.join(avail)} 만 사용 가능")


def set_seed(seed: int):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def percentile(xs: List[float], q: float) -> float:
    if not xs:
        return float("nan")
    ys = sorted(xs)
    k = (len(ys) - 1) * (q / 100.0)
    f = math.floor(k)
    c = math.ceil(k)
    if f == c:
        return ys[int(k)]
    return ys[f] + (ys[c] - ys[f]) * (k - f)


def human_mb(x_bytes: int) -> float:
    return round(x_bytes / (1024.0 * 1024.0), 2)


def write_csv_row(path: Optional[Path], row: Dict[str, Any], header_first: bool = True):
    if path is None:
        return
    import csv

    path.parent.mkdir(parents=True, exist_ok=True)
    write_header = header_first and (not path.exists())
    with path.open("a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(row.keys()))
        if write_header:
            w.writeheader()
        w.writerow(row)


class LMTrainer:
    def __init__(
        self,
        model,
        optimizer,
        scheduler,
        device: str,
        amp: str = "none",
        task_name: str = "L1",
        ttt_target: Optional[float] = None,
    ):
        self.model = model
        self.opt = optimizer
        self.sch = scheduler
        self.device = device
        self.amp = amp if torch.cuda.is_available() and device.startswith("cuda") else "none"
        self.task_name = task_name
        self.autocast_dtype = torch.bfloat16 if self.amp == "bf16" else torch.float16
        self.scaler = torch.cuda.amp.GradScaler(enabled=(self.amp == "fp16"))
        self.step_times: List[float] = []
        self.losses_epoch: List[float] = []
        self.tokens_epoch: int = 0
        self.global_tokens: int = 0
        self.peak_mem_bytes: int = 0
        self.best_val_metric: float = float("inf")
        self.ttt_target = ttt_target
        self.ttt_sec: Optional[float] = None
        self._wall0 = time.perf_counter()

    def _move_batch_to_device(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        if self.device.startswith("cuda"):
            return {k: v.to(self.device, non_blocking=True) for k, v in batch.items()}
        return batch

    def _token_count(self, batch: Dict[str, torch.Tensor]) -> int:
        target = batch.get("labels")
        if target is None:
            target = batch.get("input_ids")
        if target is None:
            return 0
        if isinstance(target, torch.Tensor):
            return int((target != -100).sum().item())
        return 0

    def train_one_epoch(self, train_loader, epoch_idx: int, total_epochs: int):
        self.model.train()
        self.losses_epoch.clear()
        self.step_times.clear()
        self.tokens_epoch = 0

        if torch.cuda.is_available() and self.device.startswith("cuda"):
            torch.cuda.reset_peak_memory_stats()

        for step_index, batch in enumerate(train_loader, start=1):
            t0 = time.perf_counter()
            batch = self._move_batch_to_device(batch)
            self.opt.zero_grad(set_to_none=True)

            if self.amp in ("fp16", "bf16"):
                with torch.autocast(device_type="cuda", dtype=self.autocast_dtype):
                    out = self.model(**batch)
                    loss = out["loss"] if isinstance(out, dict) else out.loss
                if self.amp == "fp16":
                    self.scaler.scale(loss).backward()
                    self.scaler.step(self.opt)
                    self.scaler.update()
                else:
                    loss.backward()
                    self.opt.step()
            else:
                out = self.model(**batch)
                loss = out["loss"] if isinstance(out, dict) else out.loss
                loss.backward()
                self.opt.step()

            if self.sch is not None:
                self.sch.step()

            dt = time.perf_counter() - t0
            self.step_times.append(dt)
            self.losses_epoch.append(float(loss.detach().cpu()))
            tk = self._token_count(batch)
            self.tokens_epoch += tk
            self.global_tokens += tk

        if torch.cuda.is_available() and self.device.startswith("cuda"):
            self.peak_mem_bytes = max(self.peak_mem_bytes, torch.cuda.max_memory_allocated())

    @torch.no_grad()
    def evaluate_one_epoch(self, val_loader) -> Dict[str, float]:
        self.model.eval()
        total_loss, total_tok = 0.0, 0
        for batch in val_loader:
            batch = self._move_batch_to_device(batch)
            if self.amp in ("fp16", "bf16"):
                with torch.autocast(device_type="cuda", dtype=self.autocast_dtype):
                    out = self.model(**batch)
                    loss = out["loss"] if isinstance(out, dict) else out.loss
            else:
                out = self.model(**batch)
                loss = out["loss"] if isinstance(out, dict) else out.loss
            btk = self._token_count(batch)
            total_loss += float(loss.detach().cpu()) * max(btk, 1)
            total_tok += max(btk, 1)
        avg_loss = total_loss / max(total_tok, 1)
        ppl = float(math.exp(min(avg_loss, 20.0)))
        self.best_val_metric = min(self.best_val_metric, avg_loss)
        if self.ttt_target is not None and self.ttt_sec is None and ppl <= self.ttt_target:
            self.ttt_sec = time.perf_counter() - self._wall0
        return {"val_loss": avg_loss, "val_ppl": ppl}

    def summarize_train_epoch(self) -> Dict[str, float]:
        step_p50 = percentile(self.step_times, 50)
        step_p90 = percentile(self.step_times, 90)
        step_p99 = percentile(self.step_times, 99)
        loss_avg = float(sum(self.losses_epoch) / max(len(self.losses_epoch), 1))
        throughput_tok_per_sec = float(self.tokens_epoch / max(sum(self.step_times), 1e-9))
        return {
            "train_loss": loss_avg,
            "step_p50_sec": step_p50,
            "step_p90_sec": step_p90,
            "step_p99_sec": step_p99,
            "throughput_tok_per_sec": throughput_tok_per_sec,
        }


def split_decay_params(model, wd: float):
    decay, nodecay = [], []
    no_decay_keys = ("bias", "norm.weight", "LayerNorm.weight", "layer_norm.weight")
    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if (len(p.shape) == 1) or any(k in n for k in no_decay_keys):
            nodecay.append(p)
        else:
            decay.append(p)
    return [{"params": decay, "weight_decay": wd}, {"params": nodecay, "weight_decay": 0.0}]


def make_rico_optimizer(model, lr: float, wd: float, args) -> optim.Optimizer:
    assert RICO is not None and rico_layerwise_groups is not None, "RICO 사용을 위해 rico.py 필요"
    pg = rico_layerwise_groups(model, weight_decay=wd)
    rico_args = dict(
        bk_beta_target=args.rico_bk_beta,
        k_cap=args.rico_k_cap,
        g_rms_floor=args.rico_g_floor,
        sync_every=args.rico_sync_every,
    )
    try:
        opt = RICO(pg, lr=lr, ft_mode=False, weight_decay=wd, wd_mode="decoupled", **rico_args)
    except TypeError as exc:
        print(f"[warn] RICO extra args 적용 실패 → 최소 인자로 폴백 ({exc})")
        opt = RICO(pg, lr=lr, ft_mode=False, weight_decay=wd, wd_mode="decoupled")
    return opt


def build_optimizer_generic(model, name: str, lr: float, wd: float, args):
    name = name.lower()
    if name == "rico":
        return make_rico_optimizer(model, lr, wd, args)
    pg = split_decay_params(model, wd)
    if name == "adamw":
        return optim.AdamW(pg, lr=lr)
    if name == "lion":
        if LionOpt is None:
            raise RuntimeError("Lion 사용을 위해 `pip install lion-pytorch` 필요")
        return LionOpt(pg, lr=lr, betas=(args.lion_beta1, args.lion_beta2), weight_decay=wd)
    if name == "soap":
        if SOAPOpt is None:
            raise RuntimeError("SOAP 사용을 위해 `pip install pytorch-optimizer` 필요 (SOAP 포함)")
        extra = {}
        if args.soap_args:
            try:
                extra = json.loads(args.soap_args)
            except Exception as exc:
                print(f"[warn] --soap_args JSON 파싱 실패: {exc} → 빈 kwargs")
        return SOAPOpt(pg, lr=lr, weight_decay=wd, **extra)
    raise ValueError(f"Unknown optimizer: {name}")


def build_l1_datasets(tokenizer_name: str, max_length: int):
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token or tokenizer.unk_token
    raw = load_dataset("wikitext", "wikitext-103-raw-v1")

    def tokenize(batch):
        return tokenizer(batch["text"], return_attention_mask=True, truncation=True, max_length=max_length)

    tokenized = raw.map(tokenize, batched=True, remove_columns=["text"])

    def group_fn(batch):
        input_ids = sum(batch["input_ids"], [])
        total_len = (len(input_ids) // max_length) * max_length
        input_ids = input_ids[:total_len]
        if len(input_ids) == 0:
            return {"input_ids": [], "attention_mask": [], "labels": []}
        blocks = [
            input_ids[i : i + max_length] for i in range(0, len(input_ids), max_length)
        ]
        attn = [[1] * max_length for _ in blocks]
        labels = [block.copy() for block in blocks]
        return {"input_ids": blocks, "attention_mask": attn, "labels": labels}

    grouped = tokenized.map(
        group_fn,
        batched=True,
        batch_size=1000,
        remove_columns=tokenized["train"].column_names,
    )
    columns = ["input_ids", "attention_mask", "labels"]
    train_ds = grouped["train"].with_format("torch", columns=columns)
    val_ds = grouped["validation"].with_format("torch", columns=columns)
    return train_ds, val_ds


def build_sft_dataset(tokenizer_name: str, dataset_name: str, text_field: str, max_length: int, train_split: str, val_split: str):
    tok = AutoTokenizer.from_pretrained(tokenizer_name, use_fast=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token or tok.unk_token
    raw = load_dataset(dataset_name)

    def format_example(ex):
        txt = ex.get(text_field)
        if txt is None:
            txt = json.dumps(ex, ensure_ascii=False)
        return {"text": txt}

    raw = raw.map(format_example)

    def tokenize(batch):
        out = tok(batch["text"], truncation=True, max_length=max_length, padding="max_length", return_attention_mask=True)
        out["labels"] = out["input_ids"].copy()
        return out

    tokenized = raw.map(tokenize, batched=True, remove_columns=raw[train_split].column_names)
    columns = ["input_ids", "attention_mask", "labels"]
    train_ds = tokenized[train_split].with_format("torch", columns=columns)
    val_ds = tokenized[val_split].with_format("torch", columns=columns)
    return train_ds, val_ds


def run_l1(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer_name = args.tokenizer or args.model_name
    train_ds, val_ds = build_l1_datasets(tokenizer_name, args.max_seq_len)
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, use_fast=True)

    if args.from_scratch:
        cfg = AutoConfig.from_pretrained(args.model_name)
        cfg.n_embd = args.d_model
        cfg.n_layer = args.n_layer
        cfg.n_head = args.n_head
        cfg.vocab_size = len(tokenizer)
        model = AutoModelForCausalLM.from_config(cfg)
    else:
        model = AutoModelForCausalLM.from_pretrained(args.model_name, torch_dtype=torch.float32)

    model.to(device)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)

    base_lr_rico = 3e-4
    base_lr_adamw = 6e-4
    lr = args.lr or (base_lr_rico if args.optimizer == "rico" else base_lr_adamw)
    opt = build_optimizer_generic(model, args.optimizer, lr, args.wd, args)

    total_steps = args.epochs * len(train_loader)
    if args.warmup_epochs > 0:
        warmup_steps = args.warmup_epochs * len(train_loader)
        warm = optim.lr_scheduler.LinearLR(opt, start_factor=1e-3, total_iters=warmup_steps)
        cosine = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(total_steps - warmup_steps, 1))
        sch = optim.lr_scheduler.SequentialLR(opt, [warm, cosine], milestones=[warmup_steps])
    else:
        sch = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(total_steps, 1))

    trainer = LMTrainer(model, opt, sch, device, amp=args.amp, task_name="L1", ttt_target=args.ttt_target)
    csv_path = Path(args.log_csv) if args.log_csv else None
    best = float("inf")
    wall0 = time.perf_counter()

    for ep in range(1, args.epochs + 1):
        if torch.cuda.is_available() and device.startswith("cuda"):
            torch.cuda.reset_peak_memory_stats()
        trainer.train_one_epoch(train_loader, ep, args.epochs)
        train_summary = trainer.summarize_train_epoch()
        val_summary = trainer.evaluate_one_epoch(val_loader)
        best = min(best, val_summary["val_loss"])
        peak = torch.cuda.max_memory_allocated() if torch.cuda.is_available() and device.startswith("cuda") else 0

        row = {
            "epoch": ep,
            "lr": opt.param_groups[0]["lr"],
            **train_summary,
            **val_summary,
            "best_val_loss": best,
            "peak_mem_mb": human_mb(peak),
        }
        write_csv_row(csv_path, row, header_first=(ep == 1))
        print(
            f"[L1][Epoch {ep}/{args.epochs}] loss {val_summary['val_loss']:.4f} | ppl {val_summary['val_ppl']:.2f} | "
            f"best_loss {best:.4f} | p99 {train_summary['step_p99_sec']:.3f}s | "
            f"thru {train_summary['throughput_tok_per_sec']:.1f} tok/s | LR {row['lr']:.6e}"
        )

    total_sec = time.perf_counter() - wall0
    peak_all = trainer.peak_mem_bytes
    loss_var = float(torch.tensor(trainer.losses_epoch).var().item() if trainer.losses_epoch else 0.0)
    summary = {
        "track": "L1_PRETRAIN",
        "optimizer": args.optimizer,
        "model_name": args.model_name,
        "tokenizer": tokenizer_name,
        "ttt_sec": trainer.ttt_sec,
        "perf": best,
        "speed": percentile(trainer.step_times, 50),
        "task_score": best,
        "throughput_tok_per_sec": trainer.summarize_train_epoch()["throughput_tok_per_sec"],
        "hp_sensitivity_sd": None,
        "loss_var": loss_var,
        "peak_mem_mb": human_mb(peak_all),
        "total_time_sec": total_sec,
    }
    if args.log_json:
        Path(args.log_json).parent.mkdir(parents=True, exist_ok=True)
        with open(args.log_json, "w") as f:
            json.dump(summary, f, indent=2)
    print("[L1][SUMMARY]", json.dumps(summary, indent=2))


def run_sft(args, track: str):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer_name = args.tokenizer or args.model_name
    train_ds, val_ds = build_sft_dataset(
        tokenizer_name,
        args.sft_dataset,
        args.sft_text_field,
        args.max_seq_len,
        args.sft_split_train,
        args.sft_split_val,
    )
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        torch_dtype=torch.bfloat16 if (args.amp == "bf16" and torch.cuda.is_available()) else torch.float32,
    )
    if args.device_map and torch.cuda.is_available():
        model = AutoModelForCausalLM.from_pretrained(
            args.model_name,
            torch_dtype=torch.bfloat16 if args.amp == "bf16" else torch.float16,
            device_map=args.device_map,
        )
        device_used = "cuda"
    else:
        model.to(device)
        device_used = device

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)

    base_lr_rico = 2e-5
    base_lr_adamw = 3e-5
    lr = args.lr or (base_lr_rico if args.optimizer == "rico" else base_lr_adamw)
    opt = build_optimizer_generic(model, args.optimizer, lr, args.wd, args)

    total_steps = args.epochs * len(train_loader)
    if args.warmup_epochs > 0:
        warmup_steps = args.warmup_epochs * len(train_loader)
        warm = optim.lr_scheduler.LinearLR(opt, start_factor=1e-3, total_iters=warmup_steps)
        cosine = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(total_steps - warmup_steps, 1))
        sch = optim.lr_scheduler.SequentialLR(opt, [warm, cosine], milestones=[warmup_steps])
    else:
        sch = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(total_steps, 1))

    trainer = LMTrainer(model, opt, sch, device_used, amp=args.amp, task_name=track, ttt_target=args.ttt_target)
    csv_path = Path(args.log_csv) if args.log_csv else None
    best = float("inf")
    wall0 = time.perf_counter()

    for ep in range(1, args.epochs + 1):
        if torch.cuda.is_available() and device_used.startswith("cuda"):
            torch.cuda.reset_peak_memory_stats()
        trainer.train_one_epoch(train_loader, ep, args.epochs)
        train_summary = trainer.summarize_train_epoch()
        val_summary = trainer.evaluate_one_epoch(val_loader)
        best = min(best, val_summary["val_loss"])
        peak = torch.cuda.max_memory_allocated() if torch.cuda.is_available() and device_used.startswith("cuda") else 0

        row = {
            "epoch": ep,
            "lr": opt.param_groups[0]["lr"],
            **train_summary,
            **val_summary,
            "best_val_loss": best,
            "peak_mem_mb": human_mb(peak),
        }
        write_csv_row(csv_path, row, header_first=(ep == 1))
        print(
            f"[{track}][Epoch {ep}/{args.epochs}] loss {val_summary['val_loss']:.4f} | ppl {val_summary['val_ppl']:.2f} | "
            f"best_loss {best:.4f} | p99 {train_summary['step_p99_sec']:.3f}s | "
            f"thru {train_summary['throughput_tok_per_sec']:.1f} tok/s | LR {row['lr']:.6e}"
        )

    total_sec = time.perf_counter() - wall0
    peak_all = trainer.peak_mem_bytes
    loss_var = float(torch.tensor(trainer.losses_epoch).var().item() if trainer.losses_epoch else 0.0)
    summary = {
        "track": track,
        "optimizer": args.optimizer,
        "model_name": args.model_name,
        "tokenizer": tokenizer_name,
        "ttt_sec": trainer.ttt_sec,
        "perf": best,
        "speed": percentile(trainer.step_times, 50),
        "task_score": best,
        "throughput_tok_per_sec": trainer.summarize_train_epoch()["throughput_tok_per_sec"],
        "hp_sensitivity_sd": None,
        "loss_var": loss_var,
        "peak_mem_mb": human_mb(peak_all),
        "total_time_sec": total_sec,
    }
    if args.log_json:
        Path(args.log_json).parent.mkdir(parents=True, exist_ok=True)
        with open(args.log_json, "w") as f:
            json.dump(summary, f, indent=2)
    print(f"[{track}][SUMMARY]", json.dumps(summary, indent=2))


def build_arg_parser():
    ap = argparse.ArgumentParser("GOAT LLM bench — L1/L2/L3")
    ap.add_argument("--task", type=str, required=True, choices=["l1_pretrain", "l2_sft", "l3_llama"])
    ap.add_argument("--model-name", type=str, default=None)
    ap.add_argument("--tokenizer", type=str, default=None)
    ap.add_argument("--sft-dataset", type=str, default="tatsu-lab/alpaca")
    ap.add_argument("--sft-text-field", type=str, default="text")
    ap.add_argument("--sft-split-train", type=str, default="train")
    ap.add_argument("--sft-split-val", type=str, default="validation")
    ap.add_argument("--optimizer", type=str, required=True, choices=["rico", "adamw", "lion", "soap"])
    ap.add_argument("--lr", type=float, default=None)
    ap.add_argument("--wd", type=float, default=0.01)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--max-seq-len", type=int, default=512)
    ap.add_argument("--amp", type=str, default="bf16", choices=["none", "fp16", "bf16"])
    ap.add_argument("--warmup-epochs", type=int, default=1)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--from-scratch", action="store_true")
    ap.add_argument("--d-model", type=int, default=768)
    ap.add_argument("--n-layer", type=int, default=12)
    ap.add_argument("--n-head", type=int, default=12)
    ap.add_argument("--device-map", type=str, default=None)
    ap.add_argument("--ttt-target", type=float, default=None)
    ap.add_argument("--log-csv", type=str, default=None)
    ap.add_argument("--log-json", type=str, default=None)
    ap.add_argument("--rico_bk_beta", type=float, default=0.9)
    ap.add_argument("--rico_k_cap", type=float, default=0.08)
    ap.add_argument("--rico_g_floor", type=float, default=1e-3)
    ap.add_argument("--rico_sync_every", type=int, default=20)
    ap.add_argument("--lion_beta1", type=float, default=0.9)
    ap.add_argument("--lion_beta2", type=float, default=0.99)
    ap.add_argument("--soap_args", type=str, default=None)
    return ap


def main(argv: Optional[List[str]] = None):
    ap = build_arg_parser()
    args = ap.parse_args(argv)
    set_seed(args.seed)
    if args.model_name is None:
        if args.task == "l1_pretrain":
            args.model_name = "gpt2"
        elif args.task == "l2_sft":
            args.model_name = "allenai/OLMo-2-1124-1B"
        else:
            args.model_name = "meta-llama/Meta-Llama-3-8B"

    if args.task == "l1_pretrain":
        print("== L1: Small GPT-style Pretrain ==")
        run_l1(args)
    elif args.task == "l2_sft":
        print("== L2: 1B-class LLM SFT ==")
        run_sft(args, track="L2_SFT")
    else:
        print("== L3: 7–8B-class LLM SFT ==")
        run_sft(args, track="L3_LLAMA")


if __name__ == "__main__":
    main()
