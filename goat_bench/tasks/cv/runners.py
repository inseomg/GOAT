# goat_bench/tasks/cv/runners.py
from __future__ import annotations

import json
import time
from pathlib import Path
from datetime import datetime
from typing import Any, Dict, List

import torch
import torch.optim as optim
import torchvision.models as models
import torchvision.models.detection as detection_models
import torchvision.models.segmentation as seg_models
from torch.utils.data import DataLoader, Subset

from goat_bench.optimizers.builder import build_optimizer
from goat_bench.utils.checkpointing import save_checkpoint
from goat_bench.utils.helpers import ensure_dir, get_hw_profile

from .config import CVTaskConfig
from .datasets import ADE20K, CocoDet, collate_det, get_cls_datasets
from .trainers import ClsTrainer, DetTrainer, SegTrainer
from .utils import human_mb, percentile, set_seed, subset_dataset, write_csv_row


_cpu_warned = False
_CKPT_ROOT = Path(__file__).resolve().parents[2] / "results" / "checkpoints"
_CLS_DATA_DIR_ALIASES = {"imagenet": "imagenet1k"}
_HPS_CV_CACHE: Dict[str, Any] | None = None


def _load_cv_hps() -> Dict[str, Any]:
    global _HPS_CV_CACHE
    if _HPS_CV_CACHE is None:
        path = Path(__file__).resolve().parents[2] / "configs" / "hps_cv.json"
        try:
            _HPS_CV_CACHE = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            _HPS_CV_CACHE = {}
    return _HPS_CV_CACHE


def _cv_hp_for(dataset: str, optimizer: str) -> Dict[str, Any] | None:
    cfg = _load_cv_hps()
    opt_block = cfg.get(optimizer.lower())
    if not opt_block:
        return None
    return opt_block.get(dataset.lower())


def _apply_hp_overrides(cfg: CVTaskConfig):
    hp = _cv_hp_for(cfg.dataset, cfg.optimizer)
    if not hp:
        return
    batch_hint = hp.get("batch_size")
    if batch_hint and not cfg.batch_override:
        cfg.batch_size = int(batch_hint)
    if hp.get("weight_decay") is not None:
        cfg.weight_decay = float(hp["weight_decay"])
    if cfg.lr is None and hp.get("lr") is not None:
        cfg.lr = float(hp["lr"])


def _apply_profile_overrides(cfg: CVTaskConfig):
    profile = get_hw_profile()
    if profile == "auto":
        profile = "gpu" if torch.cuda.is_available() else "cpu"
    if profile == "cpu":
        if not cfg.batch_override:
            cfg.batch_size = min(cfg.batch_size, 64 if cfg.task == "cls" else 8)
        cfg.workers = min(cfg.workers, 2)
        if hasattr(cfg, "model") and cfg.task == "cls" and cfg.dataset.lower() != "imagenet":
            # keep modest backbone on CPU
            pass
    elif profile == "gpu":
        if not cfg.batch_override:
            if cfg.task == "cls":
                cfg.batch_size = max(cfg.batch_size, 128)
            else:
                cfg.batch_size = max(cfg.batch_size, 16)
        cfg.workers = max(cfg.workers, 4)
        if hasattr(cfg, "model") and cfg.task == "cls":
            if cfg.model.startswith("resnet"):
                # keep user choice
                pass
    elif profile == "gpu_high":
        if not cfg.batch_override:
            if cfg.task == "cls":
                cfg.batch_size = max(cfg.batch_size, 256)
            elif cfg.task == "det":
                cfg.batch_size = max(cfg.batch_size, 32)
            else:
                cfg.batch_size = max(cfg.batch_size, 32)
        cfg.workers = max(cfg.workers, 8)
        if hasattr(cfg, "model") and cfg.task == "cls" and cfg.model.startswith("resnet"):
            # ensure at least resnet50
            order = ["resnet18", "resnet34", "resnet50", "resnet101", "resnet152"]
            try:
                idx = order.index(cfg.model)
                if idx < 2:
                    cfg.model = "resnet50"
            except ValueError:
                pass


def _warn_if_cpu_only():
    global _cpu_warned
    if torch.cuda.is_available() or _cpu_warned:
        return
    print("[WARN] CUDA 디바이스를 찾을 수 없어 CPU 모드로 실행합니다. 속도가 매우 느릴 수 있습니다.")
    _cpu_warned = True


def get_cls_model(num_classes: int, input_size: int, model_name: str = "resnet50"):
    factories = {
        "resnet18": models.resnet18,
        "resnet34": models.resnet34,
        "resnet50": models.resnet50,
        "resnet101": models.resnet101,
        "resnet152": models.resnet152,
        "densenet121": models.densenet121,
        "densenet169": models.densenet169,
    }
    if model_name not in factories:
        raise ValueError(f"unknown model: {model_name}")
    model = factories[model_name](weights=None, num_classes=num_classes)
    if input_size < 112:
        if hasattr(model, "conv1") and hasattr(model, "maxpool"):
            model.conv1 = torch.nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
            model.maxpool = torch.nn.Identity()
        elif hasattr(model, "features") and hasattr(model.features, "conv0"):
            model.features.conv0 = torch.nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
            if hasattr(model.features, "pool0"):
                model.features.pool0 = torch.nn.Identity()
    return model


def get_det_model(num_classes: int):
    return detection_models.fasterrcnn_resnet50_fpn_v2(weights=None, weights_backbone=None, num_classes=num_classes)


def get_seg_model(num_classes: int):
    return seg_models.deeplabv3_resnet50(weights=None, num_classes=num_classes, aux_loss=None)


def coco_evaluate(model, loader, dataset, *, amp: str, autocast_dtype, device: str = "cuda"):
    try:
        from pycocotools.cocoeval import COCOeval
    except Exception as exc:
        raise RuntimeError("pycocotools is required for COCO evaluation") from exc

    model.eval()
    base_ds = dataset.dataset if isinstance(dataset, Subset) else dataset
    results: List[Dict[str, Any]] = []
    use_cuda = device.startswith("cuda") and torch.cuda.is_available()
    for images, targets in loader:
        images = [img.to(device, non_blocking=True) for img in images]
        if amp in ("fp16", "bf16") and use_cuda:
            with torch.autocast(device_type="cuda", dtype=autocast_dtype):
                outputs = model(images)
        else:
            outputs = model(images)

        for out, tgt in zip(outputs, targets):
            img_id = int(tgt["image_id"])
            boxes = out["boxes"].detach().cpu().tolist()
            scores = out["scores"].detach().cpu().tolist()
            labels = out["labels"].detach().cpu().tolist()
            for bb, score, label in zip(boxes, scores, labels):
                x1, y1, x2, y2 = bb
                w, h = x2 - x1, y2 - y1
                if w <= 0 or h <= 0:
                    continue
                results.append(
                    {
                        "image_id": img_id,
                        "category_id": base_ds.contig2cat[label],
                        "bbox": [x1, y1, w, h],
                        "score": float(score),
                    }
                )

    coco_gt = base_ds.coco
    if len(results) == 0:
        print("[warn] no detections produced during evaluation")
        return {"mAP": 0.0}
    coco_dt = coco_gt.loadRes(results)
    coco_eval = COCOeval(coco_gt, coco_dt, iouType="bbox")
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()
    return {"mAP": float(coco_eval.stats[0])}


def _finalize_summary(summary: Dict[str, Any], log_json: Path | None):
    if log_json is None:
        return
    log_json.parent.mkdir(parents=True, exist_ok=True)
    with log_json.open("w") as f:
        json.dump(summary, f, indent=2)


def _resolve_cls_data_root(base: Path, dataset: str) -> Path:
    dataset_lower = dataset.lower()
    alias = _CLS_DATA_DIR_ALIASES.get(dataset_lower, dataset_lower)
    names = [dataset_lower]
    if alias not in names:
        names.append(alias)

    if base.name.lower() in names:
        return base

    for name in names:
        candidate = base / name
        if candidate.exists():
            return candidate
    return base / names[-1]


def run_classification(cfg: CVTaskConfig) -> Dict[str, Any]:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    _apply_profile_overrides(cfg)
    _apply_hp_overrides(cfg)
    data_root = _resolve_cls_data_root(Path(cfg.data_dir), cfg.dataset)
    train_set, val_set, num_classes, input_size = get_cls_datasets(cfg.dataset, data_root)
    train_set = subset_dataset(train_set, cfg.subset_frac, cfg.seed)
    val_set = subset_dataset(val_set, max(cfg.subset_frac, 0.2), cfg.seed)

    train_loader = DataLoader(
        train_set,
        cfg.batch_size,
        shuffle=True,
        num_workers=cfg.workers,
        pin_memory=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_set,
        cfg.batch_size * 2,
        shuffle=False,
        num_workers=cfg.workers,
        pin_memory=True,
    )

    model = get_cls_model(num_classes, input_size, cfg.model).to(device)
    dataset_lower = cfg.dataset.lower()
    if dataset_lower == "imagenet":
        base_epochs, base_rico, base_adamw, scale, ls, default_ttt = 90, 0.6, 3e-4, cfg.batch_size / 256, 0.1, 76.0
    elif dataset_lower == "tinyimagenet":
        base_epochs, base_rico, base_adamw, scale, ls, default_ttt = 100, 0.8, 1e-3, cfg.batch_size / 256, 0.1, 50.0
    else:
        base_epochs, base_rico, base_adamw, scale, ls, default_ttt = 100, 1.0, 1e-3, cfg.batch_size / 128, 0.0, 45.0

    epochs = cfg.epochs if cfg.epochs is not None else base_epochs
    base_lr = base_rico if cfg.optimizer.lower() == "rico" else base_adamw
    lr = cfg.lr if cfg.lr is not None else (base_lr * max(scale, 1e-3))

    opt = build_optimizer(model, cfg.optimizer, lr, cfg.weight_decay, cfg)

    if cfg.warmup_epochs > 0:
        warm = optim.lr_scheduler.LinearLR(opt, start_factor=1e-3, total_iters=cfg.warmup_epochs)
        cosine = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(epochs - cfg.warmup_epochs, 1))
        sch = optim.lr_scheduler.SequentialLR(opt, [warm, cosine], milestones=[cfg.warmup_epochs])
    else:
        sch = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)

    ttt_target = cfg.ttt_target if cfg.ttt_target is not None else default_ttt
    trainer = ClsTrainer(
        model,
        opt,
        sch,
        device,
        amp=cfg.amp,
        task_name="CLS",
        label_smoothing=ls,
        ttt_target=ttt_target,
    )

    csv_path = Path(cfg.log_csv) if cfg.log_csv else None
    best = -1e9
    wall0 = time.perf_counter()
    val_summary: Dict[str, Any] = {}
    last_val_summary: Dict[str, Any] = {}
    interrupted_ckpt = None
    current_epoch = 0
    try:
        for epoch in range(1, epochs + 1):
            current_epoch = epoch
            if torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats()
            trainer.train_one_epoch(train_loader, epoch, epochs)
            train_summary = trainer.summarize_train_epoch()
            val_summary = trainer.evaluate_one_epoch(val_loader)
            last_val_summary = val_summary
            sch.step()
            best = max(best, val_summary["val_top1"])
            peak = torch.cuda.max_memory_allocated() if torch.cuda.is_available() else 0
            row = {
                "epoch": epoch,
                "lr": sch.get_last_lr()[0],
                **train_summary,
                **val_summary,
                "best_val": best,
                "peak_mem_mb": human_mb(peak),
            }
            write_csv_row(csv_path, row, header_first=(epoch == 1))
            print(
                "[CLS][Epoch {epoch}/{epochs}] top1 {top1:.2f} | best {best:.2f} | p99 {p99:.3f}s | thru {thru:.1f} img/s | LR {lr:.6f}".format(
                    epoch=epoch,
                    epochs=epochs,
                    top1=val_summary["val_top1"],
                    best=best,
                    p99=train_summary["step_p99_sec"],
                    thru=train_summary["throughput"],
                    lr=row["lr"],
                )
            )
    except KeyboardInterrupt:
        interrupted_ckpt = _save_cv_checkpoint(
            "cls",
            cfg,
            model,
            opt,
            sch,
            current_epoch,
            best,
            {"val_summary": last_val_summary},
        )
        print(f"[CLS] 'exit' 신호를 감지했습니다. 체크포인트 저장: {interrupted_ckpt}")

    total_sec = time.perf_counter() - wall0
    peak_all = trainer.peak_mem_bytes
    loss_var = (
        float(torch.tensor(trainer.losses_epoch).var().item()) if trainer.losses_epoch else 0.0
    )
    summary = {
        "task": "classification",
        "dataset": cfg.dataset,
        "optimizer": cfg.optimizer,
        "ttt_sec": trainer.ttt_sec,
        "perf": best,
        "speed": percentile(trainer.step_times, 50),
        "task_score": best,
        "throughput": trainer.summarize_train_epoch()["throughput"],
        "hp_sensitivity_sd": None,
        "gen_gap": val_summary.get("gen_gap"),
        "loss_var": loss_var,
        "peak_mem_mb": round(peak_all / (1024 * 1024), 2),
        "total_time_sec": total_sec,
    }
    summary["interrupted"] = interrupted_ckpt is not None
    if interrupted_ckpt:
        summary["checkpoint"] = str(interrupted_ckpt)
    log_json = Path(cfg.log_json) if cfg.log_json else None
    _finalize_summary(summary, log_json)
    print("[CLS][SUMMARY]", json.dumps(summary, indent=2))
    return summary


def run_detection(cfg: CVTaskConfig) -> Dict[str, Any]:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    _apply_profile_overrides(cfg)
    _apply_hp_overrides(cfg)
    root = Path(cfg.data_dir)
    train_set_full = CocoDet(root / "train2017", root / "annotations" / "instances_train2017.json", train=True)
    val_set_full = CocoDet(root / "val2017", root / "annotations" / "instances_val2017.json", train=False)

    train_set = subset_dataset(train_set_full, cfg.subset_frac, cfg.seed)
    val_set = subset_dataset(val_set_full, max(cfg.subset_frac, 0.2), cfg.seed)

    train_loader = DataLoader(
        train_set,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.workers,
        pin_memory=True,
        collate_fn=collate_det,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_set,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.workers,
        pin_memory=True,
        collate_fn=collate_det,
    )

    num_classes = len(train_set_full.cat2contig) + 1
    model = get_det_model(num_classes).to(device)

    base_lr_rico, base_lr_adamw = 0.02, 1e-4
    scale = max(cfg.batch_size / 16, 1e-3)
    base_lr = base_lr_rico if cfg.optimizer.lower() == "rico" else base_lr_adamw
    lr = cfg.lr if cfg.lr is not None else (base_lr * scale)

    opt = build_optimizer(model, cfg.optimizer, lr, cfg.weight_decay, cfg)
    epochs = cfg.epochs if cfg.epochs is not None else 12
    sch = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)

    ttt_target = cfg.ttt_target if cfg.ttt_target is not None else 38.0
    trainer = DetTrainer(
        model,
        opt,
        sch,
        device,
        amp=cfg.amp,
        task_name="DET",
        ttt_target=ttt_target,
        eval_fn=coco_evaluate,
        eval_kwargs={"dataset": val_loader.dataset, "device": device},
    )

    csv_path = Path(cfg.log_csv) if cfg.log_csv else None
    best = -1e9
    wall0 = time.perf_counter()
    val_summary: Dict[str, Any] = {}
    last_val_summary: Dict[str, Any] = {}
    interrupted_ckpt = None
    current_epoch = 0
    try:
        for epoch in range(1, epochs + 1):
            current_epoch = epoch
            if torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats()
            trainer.train_one_epoch(train_loader, epoch, epochs)
            train_summary = trainer.summarize_train_epoch()
            val_summary = trainer.evaluate_one_epoch(val_loader)
            last_val_summary = val_summary
            sch.step()
            best = max(best, val_summary["val_mAP"])
            peak = torch.cuda.max_memory_allocated() if torch.cuda.is_available() else 0
            row = {
                "epoch": epoch,
                "lr": sch.get_last_lr()[0],
                **train_summary,
                **val_summary,
                "best_val": best,
                "peak_mem_mb": human_mb(peak),
            }
            write_csv_row(csv_path, row, header_first=(epoch == 1))
            print(
                "[DET][Epoch {epoch}/{epochs}] mAP {map:.2f} | best {best:.2f} | p99 {p99:.3f}s | thru {thru:.1f} img/s | LR {lr:.6f}".format(
                    epoch=epoch,
                    epochs=epochs,
                    map=val_summary["val_mAP"],
                    best=best,
                    p99=train_summary["step_p99_sec"],
                    thru=train_summary["throughput"],
                    lr=row["lr"],
                )
            )
    except KeyboardInterrupt:
        interrupted_ckpt = _save_cv_checkpoint(
            "det",
            cfg,
            model,
            opt,
            sch,
            current_epoch,
            best,
            {"val_summary": last_val_summary},
        )
        print(f"[DET] 'exit' 신호를 감지했습니다. 체크포인트 저장: {interrupted_ckpt}")

    total_sec = time.perf_counter() - wall0
    peak_all = trainer.peak_mem_bytes
    loss_var = float(torch.tensor(trainer.losses_epoch).var().item()) if trainer.losses_epoch else 0.0
    summary = {
        "task": "detection",
        "dataset": "coco2017",
        "optimizer": cfg.optimizer,
        "ttt_sec": trainer.ttt_sec,
        "perf": best,
        "speed": percentile(trainer.step_times, 50),
        "task_score": best,
        "throughput": trainer.summarize_train_epoch()["throughput"],
        "hp_sensitivity_sd": None,
        "gen_gap": None,
        "loss_var": loss_var,
        "peak_mem_mb": round(peak_all / (1024 * 1024), 2),
        "total_time_sec": total_sec,
    }
    summary["interrupted"] = interrupted_ckpt is not None
    if interrupted_ckpt:
        summary["checkpoint"] = str(interrupted_ckpt)
    log_json = Path(cfg.log_json) if cfg.log_json else None
    _finalize_summary(summary, log_json)
    print("[DET][SUMMARY]", json.dumps(summary, indent=2))
    return summary


def run_segmentation(cfg: CVTaskConfig) -> Dict[str, Any]:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    _apply_profile_overrides(cfg)
    _apply_hp_overrides(cfg)
    root = Path(cfg.data_dir)
    train_set_full = ADE20K(root, "train", crop=(512, 512), train=True)
    val_set_full = ADE20K(root, "val", crop=(512, 512), train=False)

    train_set = subset_dataset(train_set_full, cfg.subset_frac, cfg.seed)
    val_set = subset_dataset(val_set_full, max(cfg.subset_frac, 0.2), cfg.seed)

    effective_batch = max(4, cfg.batch_size // 4)
    train_loader = DataLoader(
        train_set,
        batch_size=effective_batch,
        shuffle=True,
        num_workers=cfg.workers,
        pin_memory=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_set,
        batch_size=effective_batch,
        shuffle=False,
        num_workers=cfg.workers,
        pin_memory=True,
    )

    model = get_seg_model(num_classes=150).to(device)

    base_lr_rico, base_lr_adamw = 0.02, 3e-4
    scale = max(cfg.batch_size / 16, 1e-3)
    base_lr = base_lr_rico if cfg.optimizer.lower() == "rico" else base_lr_adamw
    lr = cfg.lr if cfg.lr is not None else (base_lr * scale)

    opt = build_optimizer(model, cfg.optimizer, lr, cfg.weight_decay, cfg)
    epochs = cfg.epochs if cfg.epochs is not None else 80
    sch = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)

    ttt_target = cfg.ttt_target if cfg.ttt_target is not None else 40.0
    trainer = SegTrainer(model, opt, sch, device, amp=cfg.amp, task_name="SEG", ttt_target=ttt_target)

    csv_path = Path(cfg.log_csv) if cfg.log_csv else None
    best = -1e9
    wall0 = time.perf_counter()
    val_summary: Dict[str, Any] = {}
    last_val_summary: Dict[str, Any] = {}
    interrupted_ckpt = None
    current_epoch = 0
    try:
        for epoch in range(1, epochs + 1):
            current_epoch = epoch
            if torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats()
            trainer.train_one_epoch(train_loader, epoch, epochs)
            train_summary = trainer.summarize_train_epoch()
            val_summary = trainer.evaluate_one_epoch(val_loader)
            last_val_summary = val_summary
            sch.step()
            best = max(best, val_summary["val_mIoU"])
            peak = torch.cuda.max_memory_allocated() if torch.cuda.is_available() else 0
            row = {
                "epoch": epoch,
                "lr": sch.get_last_lr()[0],
                **train_summary,
                **val_summary,
                "best_val": best,
                "peak_mem_mb": human_mb(peak),
            }
            write_csv_row(csv_path, row, header_first=(epoch == 1))
            print(
                "[SEG][Epoch {epoch}/{epochs}] mIoU {miou:.2f} | best {best:.2f} | p99 {p99:.3f}s | thru {thru:.1f} img/s | LR {lr:.6f}".format(
                    epoch=epoch,
                    epochs=epochs,
                    miou=val_summary["val_mIoU"],
                    best=best,
                    p99=train_summary["step_p99_sec"],
                    thru=train_summary["throughput"],
                    lr=row["lr"],
                )
            )
    except KeyboardInterrupt:
        interrupted_ckpt = _save_cv_checkpoint(
            "seg",
            cfg,
            model,
            opt,
            sch,
            current_epoch,
            best,
            {"val_summary": last_val_summary},
        )
        print(f"[SEG] 'exit' 신호를 감지했습니다. 체크포인트 저장: {interrupted_ckpt}")

    total_sec = time.perf_counter() - wall0
    peak_all = trainer.peak_mem_bytes
    loss_var = float(torch.tensor(trainer.losses_epoch).var().item()) if trainer.losses_epoch else 0.0
    summary = {
        "task": "segmentation",
        "dataset": "ADE20K",
        "optimizer": cfg.optimizer,
        "ttt_sec": trainer.ttt_sec,
        "perf": best,
        "speed": percentile(trainer.step_times, 50),
        "task_score": best,
        "throughput": trainer.summarize_train_epoch()["throughput"],
        "hp_sensitivity_sd": None,
        "gen_gap": None,
        "loss_var": loss_var,
        "peak_mem_mb": round(peak_all / (1024 * 1024), 2),
        "total_time_sec": total_sec,
    }
    summary["interrupted"] = interrupted_ckpt is not None
    if interrupted_ckpt:
        summary["checkpoint"] = str(interrupted_ckpt)
    log_json = Path(cfg.log_json) if cfg.log_json else None
    _finalize_summary(summary, log_json)
    print("[SEG][SUMMARY]", json.dumps(summary, indent=2))
    return summary


def run_task(cfg: CVTaskConfig) -> Dict[str, Any]:
    _warn_if_cpu_only()
    set_seed(cfg.seed)
    task = cfg.task.lower()
    if task == "cls":
        return run_classification(cfg)
    if task == "det":
        return run_detection(cfg)
    if task == "seg":
        return run_segmentation(cfg)
    raise ValueError(f"Unknown CV task: {cfg.task}")
def _save_cv_checkpoint(tag: str, cfg: CVTaskConfig, model, optimizer, scheduler, epoch: int, best: float, extra: Dict[str, Any]):
    ensure_dir(_CKPT_ROOT)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    dataset = getattr(cfg, "dataset", tag)
    path = _CKPT_ROOT / f"{tag}_{dataset}_epoch{epoch}_{stamp}.pt"
    state = {
        "task": tag,
        "dataset": dataset,
        "epoch": epoch,
        "best_metric": best,
        "config": cfg.__dict__,
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "scheduler_state": scheduler.state_dict(),
        **extra,
    }
    save_checkpoint(path, state)
    return path
