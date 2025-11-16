# goat_bench/tasks/registry.py
from __future__ import annotations

from pathlib import Path
from typing import Optional

from goat_bench.utils.helpers import clear_screen, print_header
from setup.prepare_data import has_dataset

from .cv.config import CVTaskConfig


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATA_DIR = ROOT / "data"
SMOKE_DATASET = "cifar100"


def _prompt(text: str, default: Optional[str] = None) -> str:
    suffix = f" [{default}]" if default else ""
    val = input(f"{text}{suffix} >> ").strip()
    return val or (default or "")


def _prompt_int(text: str, default: Optional[int] = None) -> Optional[int]:
    raw = _prompt(text, str(default) if default is not None else "")
    if raw == "":
        return None
    try:
        return int(raw)
    except ValueError:
        print("숫자로 입력되지 않아 기본값을 사용합니다.")
        return default


def _prompt_float(text: str, default: float) -> float:
    raw = _prompt(text, str(default))
    if raw == "":
        return default
    try:
        return float(raw)
    except ValueError:
        print("숫자로 입력되지 않아 기본값을 사용합니다.")
        return default


def _select_from(options: list[str], default_idx: int, label: str) -> str:
    print(f"{label}:")
    for idx, opt in enumerate(options, start=1):
        flag = "(기본)" if (idx - 1) == default_idx else ""
        print(f"  [{idx}] {opt} {flag}")
    choice = input(" 번호 선택 >> ").strip()
    if not choice:
        return options[default_idx]
    try:
        idx = int(choice) - 1
        if 0 <= idx < len(options):
            return options[idx]
    except ValueError:
        pass
    print("잘못된 입력이라 기본값을 사용합니다.")
    return options[default_idx]


def _build_cv_config(task: str) -> CVTaskConfig:
    default_data = _prompt("데이터 디렉터리", str(DEFAULT_DATA_DIR))
    data_dir = Path(default_data).expanduser()
    optimizer = _select_from(["rico", "adamw", "lion", "soap"], 1, "Optimizer 선택")
    amp = _select_from(["none", "fp16", "bf16"], 0, "AMP 모드")
    subset = _prompt_float("데이터 사용 비율 (0<frac<=1)", 1.0)
    epochs = _prompt_int("Epoch 수 (빈칸=자동)", None)

    if task == "cls":
        dataset = _select_from(["cifar100", "tinyimagenet", "imagenet"], 0, "분류 Dataset")
        model = _select_from(["resnet18", "resnet34", "resnet50", "resnet101", "resnet152"], 2, "Backbone")
        batch = _prompt_int("배치 사이즈", 128) or 128
        return CVTaskConfig(
            task="cls",
            data_dir=data_dir,
            optimizer=optimizer,
            dataset=dataset,
            model=model,
            batch_size=batch,
            subset_frac=subset,
            epochs=epochs,
            amp=amp,
        )
    if task == "det":
        batch = _prompt_int("배치 사이즈", 16) or 16
        return CVTaskConfig(
            task="det",
            data_dir=data_dir,
            optimizer=optimizer,
            batch_size=batch,
            subset_frac=subset,
            epochs=epochs,
            amp=amp,
        )
    batch = _prompt_int("배치 사이즈", 16) or 16
    return CVTaskConfig(
        task="seg",
        data_dir=data_dir,
        optimizer=optimizer,
        batch_size=batch,
        subset_frac=subset,
        epochs=epochs,
        amp=amp,
    )


def _run_quick_cv_smoke():
    target_root = DEFAULT_DATA_DIR / SMOKE_DATASET
    if not has_dataset(SMOKE_DATASET, DEFAULT_DATA_DIR):
        print(f"[WARN] {SMOKE_DATASET} 데이터가 아직 준비되지 않았습니다.")
        print("  - main 메뉴에서 '데이터셋 관리' → CIFAR100 → 다운로드 를 먼저 실행하세요.")
        return

    cfg = CVTaskConfig(
        task="cls",
        data_dir=target_root,
        optimizer="adamw",
        dataset=SMOKE_DATASET,
        model="resnet18",
        batch_size=32,
        subset_frac=0.05,
        epochs=1,
        amp="none",
        workers=2,
    )
    print("[SMOKE] CIFAR100 1-epoch sanity run (AdamW, ResNet18, subset=5%)")
    try:
        from .cv.runners import run_task

        run_task(cfg)
    except Exception as exc:
        print(f"[ERROR] 스모크 테스트 실행 실패: {exc}")


def run_benchmark_menu():
    while True:
        clear_screen()
        print_header("벤치마크 실행")
        print(" [1] CV - Classification (ResNet on CIFAR/TinyIN/ImageNet)")
        print(" [2] CV - Detection (FasterRCNN on COCO)")
        print(" [3] CV - Segmentation (DeepLabV3 on ADE20K)")
        print(" [4] CV - QUICK CHECK (CIFAR100 1-epoch smoke test)")
        print(" [0] 돌아가기")
        print("--------------------------------------------------")
        choice = input(" 선택 >> ").strip()
        if choice == "0":
            break
        if choice == "4":
            _run_quick_cv_smoke()
            input("엔터를 눌러 계속하세요...")
            continue
        task_map = {"1": "cls", "2": "det", "3": "seg"}
        task = task_map.get(choice)
        if not task:
            print("잘못된 선택입니다.")
            input("엔터를 눌러 계속하세요...")
            continue
        cfg = _build_cv_config(task)
        try:
            from .cv.runners import run_task

            run_task(cfg)
        except Exception as exc:
            print(f"[ERROR] 벤치마크 실행 중 오류: {exc}")
        input("엔터를 눌러 계속하세요...")
