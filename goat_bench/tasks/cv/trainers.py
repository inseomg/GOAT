# goat_bench/tasks/cv/trainers.py
from __future__ import annotations

import time
from typing import Callable, List, Optional

import torch
import torch.nn as nn
from tqdm import tqdm

from .utils import percentile


class TrainerBase:
    """Common training loop shared by classification/detection/segmentation tasks."""

    def __init__(
        self,
        model,
        optimizer,
        scheduler,
        device: str,
        *,
        amp: str = "none",
        task_name: str = "task",
        criterion=None,
        ttt_target=None,
    ):
        self.model = model
        self.opt = optimizer
        self.sch = scheduler
        self.device = device
        self.crit = criterion
        self.amp = amp if torch.cuda.is_available() else "none"
        self.task_name = task_name
        self.autocast_dtype = torch.bfloat16 if self.amp == "bf16" else torch.float16
        self.scaler = torch.amp.GradScaler("cuda", enabled=(self.amp == "fp16"))
        self.step_times: List[float] = []
        self.losses_epoch: List[float] = []
        self.global_seen = 0
        self.epoch_seen = 0
        self.peak_mem_bytes = 0
        self.best_val_metric = -float("inf")
        self.ttt_target = ttt_target
        self.ttt_sec = None
        self._wall0 = time.perf_counter()

    def prepare_batch(self, batch):
        raise NotImplementedError

    def forward_train_loss(self, *batch_tensors):
        raise NotImplementedError

    @torch.no_grad()
    def evaluate_one_epoch(self, val_loader):
        raise NotImplementedError

    def train_one_epoch(self, train_loader, epoch_idx: int, total_epochs: int):
        self.model.train()
        self.losses_epoch.clear()
        self.step_times.clear()
        self.epoch_seen = 0

        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()

        pbar = tqdm(train_loader, desc=f"[{self.task_name}][Train {epoch_idx}/{total_epochs}]")
        for batch in pbar:
            t0 = time.perf_counter()
            tensors = self.prepare_batch(batch)
            self.opt.zero_grad(set_to_none=True)

            if self.amp in ("fp16", "bf16"):
                with torch.autocast(device_type="cuda", dtype=self.autocast_dtype):
                    loss = self.forward_train_loss(*tensors)
                if self.amp == "fp16":
                    self.scaler.scale(loss).backward()
                    self.scaler.step(self.opt)
                    self.scaler.update()
                else:
                    loss.backward()
                    self.opt.step()
            else:
                loss = self.forward_train_loss(*tensors)
                loss.backward()
                self.opt.step()

            dt = time.perf_counter() - t0
            self.step_times.append(dt)
            self.losses_epoch.append(float(loss.detach().cpu()))
            self.add_seen(tensors)

            pbar.set_postfix(
                loss=f"{sum(self.losses_epoch) / len(self.losses_epoch):.4f}",
                p50=f"{percentile(self.step_times, 50):.3f}s",
            )

        if torch.cuda.is_available():
            self.peak_mem_bytes = max(self.peak_mem_bytes, torch.cuda.max_memory_allocated())

    def add_seen(self, tensors):
        x0 = tensors[0] if isinstance(tensors, (tuple, list)) else tensors
        if torch.is_tensor(x0):
            bsz = x0.size(0)
            self.global_seen += bsz
            self.epoch_seen += bsz

    def summarize_train_epoch(self):
        step_p50 = percentile(self.step_times, 50)
        step_p90 = percentile(self.step_times, 90)
        step_p99 = percentile(self.step_times, 99)
        loss_avg = float(sum(self.losses_epoch) / max(len(self.losses_epoch), 1))
        throughput = float(self.epoch_seen / max(sum(self.step_times), 1e-9))
        return {
            "train_loss": loss_avg,
            "step_p50_sec": step_p50,
            "step_p90_sec": step_p90,
            "step_p99_sec": step_p99,
            "throughput": throughput,
        }

    def maybe_mark_ttt(self, now_metric: float):
        if (self.ttt_target is not None) and (self.ttt_sec is None):
            if now_metric >= self.ttt_target:
                self.ttt_sec = time.perf_counter() - self._wall0


def topk_acc(output, target, topk=(1,)):
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)
        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target[None, :].expand_as(pred))
        res = []
        for k in topk:
            res.append(correct[:k].reshape(-1).float().sum().mul_(100.0 / batch_size).item())
        return res


class ClsTrainer(TrainerBase):
    def __init__(self, *args, label_smoothing: float = 0.0, **kwargs):
        super().__init__(*args, **kwargs)
        self.crit = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
        self._train_top1_last = None

    def prepare_batch(self, batch):
        x, y = batch
        return x.to(self.device, non_blocking=True), y.to(self.device, non_blocking=True)

    def forward_train_loss(self, x, y):
        out = self.model(x)
        loss = self.crit(out, y)
        top1 = topk_acc(out, y, (1,))[0]
        self._train_top1_last = top1
        return loss

    @torch.no_grad()
    def evaluate_one_epoch(self, val_loader):
        self.model.eval()
        vloss, v1, v5, n = 0.0, 0.0, 0.0, 0
        for x, y in tqdm(val_loader, desc=f"[{self.task_name}][Val]"):
            x, y = x.to(self.device), y.to(self.device)
            if self.amp in ("fp16", "bf16"):
                with torch.autocast(device_type="cuda", dtype=self.autocast_dtype):
                    out = self.model(x)
                    loss = self.crit(out, y)
            else:
                out = self.model(x)
                loss = self.crit(out, y)
            bsz = x.size(0)
            t1, t5 = topk_acc(out, y, (1, 5))
            vloss += float(loss) * bsz
            v1 += t1 * bsz / 100.0
            v5 += t5 * bsz / 100.0
            n += bsz
        val_loss = vloss / max(n, 1)
        top1 = 100.0 * v1 / max(n, 1)
        top5 = 100.0 * v5 / max(n, 1)
        self.best_val_metric = max(self.best_val_metric, top1)
        self.maybe_mark_ttt(top1)
        gen_gap = (self._train_top1_last - top1) if (self._train_top1_last is not None) else None
        return {"val_loss": val_loss, "val_top1": top1, "val_top5": top5, "gen_gap": gen_gap}


def seg_miou(pred: torch.Tensor, target: torch.Tensor, num_classes: int = 150, ignore_index: int = 255):
    ious = []
    for cls in range(num_classes):
        if cls == ignore_index:
            continue
        pred_i = pred == cls
        tgt_i = target == cls
        inter = (pred_i & tgt_i).sum().item()
        union = (pred_i | tgt_i).sum().item()
        if union > 0:
            ious.append(inter / union)
    return 100.0 * sum(ious) / max(len(ious), 1)


class SegTrainer(TrainerBase):
    def __init__(self, *args, num_classes: int = 150, **kwargs):
        super().__init__(*args, **kwargs)
        self.crit = nn.CrossEntropyLoss(ignore_index=255)
        self.num_classes = num_classes

    def prepare_batch(self, batch):
        x, y = batch
        return x.to(self.device, non_blocking=True), y.to(self.device, non_blocking=True)

    def forward_train_loss(self, x, y):
        out = self.model(x)["out"]
        return self.crit(out, y)

    @torch.no_grad()
    def evaluate_one_epoch(self, val_loader):
        self.model.eval()
        miou_total, n = 0.0, 0
        for x, y in tqdm(val_loader, desc=f"[{self.task_name}][Val]"):
            x, y = x.to(self.device), y.to(self.device)
            if self.amp in ("fp16", "bf16"):
                with torch.autocast(device_type="cuda", dtype=self.autocast_dtype):
                    logits = self.model(x)["out"]
            else:
                logits = self.model(x)["out"]
            pred = logits.argmax(1)
            miou_total += seg_miou(pred.cpu(), y.cpu(), num_classes=self.num_classes, ignore_index=255)
            n += 1
        miou = miou_total / max(n, 1)
        self.best_val_metric = max(self.best_val_metric, miou)
        self.maybe_mark_ttt(miou)
        return {"val_mIoU": miou}


class DetTrainer(TrainerBase):
    def __init__(self, *args, eval_fn: Optional[Callable] = None, eval_kwargs: Optional[dict] = None, **kwargs):
        super().__init__(*args, **kwargs)
        self.eval_fn = eval_fn
        self.eval_kwargs = eval_kwargs or {}

    def prepare_batch(self, batch):
        images, targets = batch
        images = [img.to(self.device, non_blocking=True) for img in images]
        targets = [
            {k: (v.to(self.device, non_blocking=True) if torch.is_tensor(v) else v) for k, v in t.items()}
            for t in targets
        ]
        return images, targets

    def forward_train_loss(self, images, targets):
        losses = self.model(images, targets)
        return sum(losses.values())

    def add_seen(self, tensors):
        images, _ = tensors
        batch = len(images)
        self.global_seen += batch
        self.epoch_seen += batch

    @torch.no_grad()
    def evaluate_one_epoch(self, val_loader):
        metrics = {}
        if self.eval_fn is not None:
            metrics = self.eval_fn(
                self.model,
                val_loader,
                amp=self.amp,
                autocast_dtype=self.autocast_dtype,
                **self.eval_kwargs,
            )
        mAP = float(metrics.get("mAP", 0.0)) * 100.0
        self.best_val_metric = max(self.best_val_metric, mAP)
        self.maybe_mark_ttt(mAP)
        return {"val_mAP": mAP}
