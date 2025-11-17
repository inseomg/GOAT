# goat_bench/tasks/registry.py
from __future__ import annotations

from pathlib import Path
from typing import Optional

from goat_bench.utils.helpers import clear_screen, print_header, get_hw_profile, set_hw_profile, profile_name
from goat_bench.tasks.nlp import run_nlp_menu, run_nlp_suite, run_nlp_smoke_menu
from goat_bench.tasks.llm import run_llm_menu, run_llm_quick, run_llm_smoke_menu
from setup.prepare_data import has_dataset, dataset_root

from .cv.config import CVTaskConfig


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATA_DIR = ROOT / "data"
DATA_DIR_ALIASES = {"imagenet": "imagenet1k"}


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


def _confirm(text: str, default: bool = False) -> bool:
    suffix = " [Y/n]" if default else " [y/N]"
    raw = input(f"{text}{suffix} >> ").strip().lower()
    if not raw:
        return default
    return raw in ("y", "yes")


def _default_data_dir_for(dataset: str) -> Path:
    alias = DATA_DIR_ALIASES.get(dataset.lower(), dataset.lower())
    return dataset_root(DEFAULT_DATA_DIR, alias)


def _default_cv_config(task: str) -> CVTaskConfig:
    if task == "cls":
        dataset = "cifar100"
        return CVTaskConfig(
            task="cls",
            data_dir=_default_data_dir_for(dataset),
            optimizer="adamw",
            dataset=dataset,
            model="resnet50",
            batch_size=128,
            batch_override=False,
            epochs=2,
            amp="fp16",
            subset_frac=1.0,
        )
    if task == "det":
        dataset = "coco2017"
        return CVTaskConfig(
            task="det",
            data_dir=_default_data_dir_for(dataset),
            optimizer="adamw",
            dataset=dataset,
            batch_size=16,
            batch_override=False,
            epochs=12,
            amp="fp16",
        )
    dataset = "ade20k"
    return CVTaskConfig(
        task="seg",
        data_dir=_default_data_dir_for(dataset),
        optimizer="adamw",
        dataset=dataset,
        batch_size=16,
        batch_override=False,
        epochs=40,
        amp="fp16",
    )


CV_SMOKE_SCENARIOS = [
    {
        "label": "Classification - CIFAR100 (ResNet18, subset=10%, 1 epoch)",
        "dataset": "cifar100",
        "builder": lambda: CVTaskConfig(
            task="cls",
            data_dir=_default_data_dir_for("cifar100"),
            optimizer="adamw",
            dataset="cifar100",
            model="resnet18",
            batch_size=64,
            batch_override=True,
            subset_frac=0.1,
            epochs=1,
            amp="fp16",
            workers=2,
        ),
    },
    {
        "label": "Classification - TinyImageNet (ResNet34, subset=2%, 1 epoch)",
        "dataset": "tinyimagenet",
        "builder": lambda: CVTaskConfig(
            task="cls",
            data_dir=_default_data_dir_for("tinyimagenet"),
            optimizer="adamw",
            dataset="tinyimagenet",
            model="resnet34",
            batch_size=48,
            batch_override=True,
            subset_frac=0.02,
            epochs=1,
            amp="fp16",
            workers=4,
        ),
    },
    {
        "label": "Detection - COCO2017 (subset=2%, 1 epoch)",
        "dataset": "coco2017",
        "builder": lambda: CVTaskConfig(
            task="det",
            data_dir=_default_data_dir_for("coco2017"),
            optimizer="adamw",
            dataset="coco2017",
            batch_size=4,
            batch_override=True,
            subset_frac=0.02,
            epochs=1,
            amp="fp16",
            workers=2,
        ),
    },
    {
        "label": "Segmentation - ADE20K (subset=2%, 1 epoch)",
        "dataset": "ade20k",
        "builder": lambda: CVTaskConfig(
            task="seg",
            data_dir=_default_data_dir_for("ade20k"),
            optimizer="adamw",
            dataset="ade20k",
            batch_size=8,
            batch_override=True,
            subset_frac=0.02,
            epochs=1,
            amp="fp16",
            workers=2,
        ),
    },
]


def _build_cv_config(task: str) -> CVTaskConfig:
    mode = _select_from(["빠른 실행 (기본 설정)", "상세 설정"], 0, "실행 모드")
    if mode.startswith("빠른"):
        return _default_cv_config(task)

    optimizer = "adamw"
    amp = "none"
    subset = 1.0
    epochs: Optional[int] = None

    if task == "cls":
        dataset = _select_from(["cifar100", "tinyimagenet", "imagenet"], 0, "분류 Dataset")
        model = _select_from(
            ["resnet18", "resnet34", "resnet50", "resnet101", "resnet152", "densenet121", "densenet169"],
            2,
            "Backbone",
        )
        batch_input = _prompt_int("배치 사이즈", 128)
        batch = batch_input or 128
        batch_override = batch_input is not None
        data_dir_default = _default_data_dir_for(dataset)
        data_dir = Path(_prompt("데이터 디렉터리", str(data_dir_default))).expanduser()

        if _confirm("Optimizer/AMP/데이터 비율 등 고급 옵션을 조정하시겠습니까?", False):
            optimizer = _select_from(["adamw", "lion", "soap"], 0, "Optimizer 선택")
            amp = _select_from(["none", "fp16", "bf16"], 1, "AMP 모드")
            subset = _prompt_float("데이터 사용 비율 (0<frac<=1)", 1.0)
            epochs = _prompt_int("Epoch 수 (빈칸=자동)", None)
        return CVTaskConfig(
            task="cls",
            data_dir=data_dir,
            optimizer=optimizer,
            dataset=dataset,
            model=model,
            batch_size=batch,
            batch_override=batch_override,
            subset_frac=subset,
            epochs=epochs,
            amp=amp,
        )
    default_name = "coco2017" if task == "det" else "ade20k"
    data_dir_default = _default_data_dir_for(default_name)
    data_dir = Path(_prompt("데이터 디렉터리", str(data_dir_default))).expanduser()
    batch_default = 16
    batch_input = _prompt_int("배치 사이즈", batch_default)
    batch = batch_input or batch_default
    batch_override = batch_input is not None
    if _confirm("Optimizer/AMP/데이터 비율 등 고급 옵션을 조정하시겠습니까?", False):
        optimizer = _select_from(["adamw", "lion", "soap"], 0, "Optimizer 선택")
        amp = _select_from(["none", "fp16", "bf16"], 1, "AMP 모드")
        subset = _prompt_float("데이터 사용 비율 (0<frac<=1)", 1.0)
        epochs = _prompt_int("Epoch 수 (빈칸=자동)", None)
    return CVTaskConfig(
        task=task,
        data_dir=data_dir,
        optimizer=optimizer,
        dataset=default_name,
        batch_size=batch,
        batch_override=batch_override,
        subset_frac=subset,
        epochs=epochs,
        amp=amp,
    )


def _run_cv_smoke_menu():
    while True:
        clear_screen()
        print_header("CV Smoke Tests")
        for idx, scenario in enumerate(CV_SMOKE_SCENARIOS, start=1):
            status = "✅" if has_dataset(scenario["dataset"], DEFAULT_DATA_DIR) else "❌"
            print(f" [{idx}] {scenario['label']} [{status}]")
        print(" [A] 전체 스모크 일괄 실행")
        print(" [0] 돌아가기")
        choice = input(" 선택 >> ").strip()
        if choice == "0":
            return
        if choice.lower() == "a":
            for scenario in CV_SMOKE_SCENARIOS:
                if not has_dataset(scenario["dataset"], DEFAULT_DATA_DIR):
                    print(f"[SKIP] {scenario['label']} (데이터셋 미준비)")
                    continue
                cfg = scenario["builder"]()
                print(f"[SMOKE-ALL] {scenario['label']}")
                try:
                    from .cv.runners import run_task
                    run_task(cfg)
                except Exception as exc:
                    print(f"[ERROR] 스모크 테스트 실행 실패: {exc}")
            input("엔터를 눌러 계속하세요...")
            continue
        try:
            scenario = CV_SMOKE_SCENARIOS[int(choice) - 1]
        except Exception:
            print("잘못된 선택입니다.")
            input("엔터를 눌러 계속...")
            continue
        if not has_dataset(scenario["dataset"], DEFAULT_DATA_DIR):
            print(f"[WARN] '{scenario['dataset']}' 데이터셋이 준비되지 않았습니다. 데이터셋 관리 메뉴에서 먼저 다운로드하세요.")
            input("엔터를 눌러 계속...")
            continue
        cfg = scenario["builder"]()
        print(f"[SMOKE] {scenario['label']}")
        try:
            from .cv.runners import run_task

            run_task(cfg)
        except Exception as exc:
            print(f"[ERROR] 스모크 테스트 실행 실패: {exc}")
        input("엔터를 눌러 계속하세요...")


def run_benchmark_menu():
    while True:
        clear_screen()
        print_header(f"벤치마크 실행 (Profile: {profile_name()})")
        print(" [TIP] 학습 중 터미널에 'exit' 입력(또는 .goat_exit 파일 생성)하면 안전하게 중단되고 체크포인트가 저장됩니다.")
        print(" [1] CV - Classification (ResNet on CIFAR/TinyIN/ImageNet)")
        print(" [2] CV - Detection (FasterRCNN on COCO)")
        print(" [3] CV - Segmentation (DeepLabV3 on ADE20K)")
        print(" [4] CV - SMOKE TESTS (선택 실행)")
        print(" [5] NLP - Suite EASY (자동)")
        print(" [6] NLP - Suite MEDIUM (자동)")
        print(" [7] NLP - Suite HARD (자동)")
        print(" [8] NLP - 전체 작업 선택")
        print(" [9] NLP - SMOKE TESTS (선택 실행)")
        print(" [10] LLM - L1 QUICK (wikitext-103)")
        print(" [11] LLM - L2 QUICK (1B SFT)")
        print(" [12] LLM - L3 QUICK (8B SFT)")
        print(" [13] LLM - 전체 작업 선택")
        print(" [14] LLM - SMOKE TESTS")
        print(" [P] 프로파일 설정 (auto/cpu/gpu/gpu_high)")
        print(" [0] 돌아가기")
        print("--------------------------------------------------")
        choice = input(" 선택 >> ").strip()
        if choice == "0":
            break
        if choice.lower() == "p":
            print("프로파일 선택: [1] AUTO (기본) [2] CPU [3] GPU [4] GPU-HIGH")
            pick = input(" 번호 >> ").strip()
            mapping = {"1": "auto", "2": "cpu", "3": "gpu", "4": "gpu_high"}
            set_hw_profile(mapping.get(pick, "auto"))
            continue
        if choice == "4":
            _run_cv_smoke_menu()
            continue
        if choice in {"5", "6", "7"}:
            suite_map = {"5": "suite:easy", "6": "suite:medium", "7": "suite:hard"}
            run_nlp_suite(DEFAULT_DATA_DIR, suite_map[choice])
            continue
        if choice == "8":
            run_nlp_menu(DEFAULT_DATA_DIR)
            continue
        if choice == "9":
            run_nlp_smoke_menu(DEFAULT_DATA_DIR)
            continue
        if choice == "10":
            run_llm_quick("l1_pretrain")
            continue
        if choice == "11":
            run_llm_quick("l2_sft")
            continue
        if choice == "12":
            run_llm_quick("l3_llama")
            continue
        if choice == "13":
            run_llm_menu()
            continue
        if choice == "14":
            run_llm_smoke_menu()
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
