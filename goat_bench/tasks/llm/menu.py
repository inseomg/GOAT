from __future__ import annotations

import os
import shlex
import subprocess
import sys
from pathlib import Path
from typing import Dict, List

from goat_bench.utils.helpers import (
    clear_screen,
    print_header,
    ensure_dir,
    configure_hf_cache,
    ConsoleSpinner,
)
from setup.prepare_data import has_dataset, download_dataset
import torch

SCRIPT_PATH = Path(__file__).resolve().with_name("bench.py")
ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data"


class LLMTask:
    def __init__(self, key: str, label: str, description: str):
        self.key = key
        self.label = label
        self.description = description


TASKS: Dict[str, LLMTask] = {
    "l1_pretrain": LLMTask(
        "l1_pretrain",
        "L1 - Small GPT Pretrain (WikiText-103)",
        "소규모 GPT 형태의 사전학습 (RICO vs AdamW vs Lion vs SOAP)",
    ),
    "l2_sft": LLMTask(
        "l2_sft",
        "L2 - 1B급 SFT (OLMo-2 1B 등)",
        "1B 파라미터급 LLM SFT (alpaca/지정 데이터셋)",
    ),
    "l3_llama": LLMTask(
        "l3_llama",
        "L3 - 7~8B급 SFT (Llama-3 등)",
        "대형 체크포인트 SFT, 라이선스 주의 필요",
    ),
}

DEFAULT_QUICK_AMP = "bf16"
LLM_QUICK_PRESETS: Dict[str, List[str]] = {
    "l1_pretrain": ["--epochs", "1", "--batch-size", "4", "--amp", DEFAULT_QUICK_AMP],
    "l2_sft": ["--epochs", "1", "--batch-size", "2", "--amp", DEFAULT_QUICK_AMP],
    "l3_llama": ["--epochs", "1", "--batch-size", "1", "--amp", DEFAULT_QUICK_AMP],
}

LLM_SMOKE_SCENARIOS = [
    {
        "label": "L1 - GPT2 (seq=256, 1 epoch, small batch)",
        "task": "l1_pretrain",
        "optimizer": "adamw",
        "extra": ["--epochs", "1", "--batch-size", "2", "--max-seq-len", "256", "--amp", DEFAULT_QUICK_AMP],
    },
    {
        "label": "L2 - 1B SFT (128 seq, 1 epoch)",
        "task": "l2_sft",
        "optimizer": "adamw",
        "extra": ["--epochs", "1", "--batch-size", "1", "--max-seq-len", "256", "--amp", DEFAULT_QUICK_AMP],
    },
    {
        "label": "L3 - Llama-3 8B (bf16, 0.5 epoch)",
        "task": "l3_llama",
        "optimizer": "adamw",
        "extra": ["--epochs", "1", "--batch-size", "1", "--amp", DEFAULT_QUICK_AMP, "--warmup-epochs", "0"],
    },
]


_LLM_DATASET_MAP: Dict[str, str] = {
    "l1_pretrain": "wikitext103",
    "l2_sft": "alpaca",
    "l3_llama": "alpaca",
}

_LLM_MODEL_DEFAULTS: Dict[str, str] = {
    "l1_pretrain": "gpt2",
    "l2_sft": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    "l3_llama": "mistralai/Mistral-7B-v0.1",
}


def _llm_dataset_ready(task_key: str) -> bool:
    ds = _LLM_DATASET_MAP.get(task_key)
    if not ds:
        return True
    try:
        return has_dataset(ds, DATA_DIR)
    except Exception:
        return False


def _run_llm_task(task_key: str, optimizer: str, extra: List[str]):
    ds = _LLM_DATASET_MAP.get(task_key)
    if ds and not _llm_dataset_ready(task_key):
        print(f"[INFO] '{ds}' 데이터셋이 준비되어 있지 않습니다.")
        ans = input(" 자동으로 다운로드할까요? [y/N] >> ").strip().lower()
        if ans in ("y", "yes"):
            download_dataset(ds, DATA_DIR, pause=False)
        else:
            print("취소되었습니다.")
            return
    # ensure model is specified; fall back to defaults per task
    flags = {extra[i]: extra[i + 1] if i + 1 < len(extra) else None for i in range(0, len(extra) - 1, 2) if extra[i].startswith("--")}
    if "--model-name" not in flags and "--model" not in flags:
        default_model = _LLM_MODEL_DEFAULTS.get(task_key)
        if default_model:
            extra = ["--model-name", default_model, *extra]
    # CPU 환경에서는 amp를 none으로 강제
    if not torch.cuda.is_available():
        cleaned: List[str] = []
        skip = False
        for i, tok in enumerate(extra):
            if skip:
                skip = False
                continue
            if tok == "--amp":
                skip = True
                continue
            cleaned.append(tok)
        extra = ["--amp", "none", *cleaned]
    env = os.environ.copy()
    env["PYTHONPATH"] = os.pathsep.join([str(ROOT), env.get("PYTHONPATH", "")])
    cmd = [sys.executable, str(SCRIPT_PATH), "--task", task_key, "--optimizer", optimizer, *extra]
    print(" [TIP] 학습 중 터미널에 'exit' 를 입력하면 중단 및 체크포인트 저장이 이루어집니다.")
    spinner = ConsoleSpinner(f"[LLM] {task_key} 실행 중")
    spinner.start()
    try:
        result = subprocess.run(cmd, check=False, env=env)
        if result.returncode != 0:
            print(f"[ERROR] LLM 작업이 비정상 종료(returncode={result.returncode}). 로그를 확인하세요.")
    finally:
        spinner.stop()


def run_llm_quick(task_key: str):
    info = TASKS[task_key]
    configure_hf_cache(DATA_DIR)
    print(f"[LLM-QUICK] {info.label}")
    preset = LLM_QUICK_PRESETS.get(task_key, [])
    _run_llm_task(task_key, "adamw", preset)
    input("엔터를 눌러 계속...")


def run_llm_menu():
    configure_hf_cache(DATA_DIR)
    ensure_dir(DATA_DIR)
    while True:
        clear_screen()
        print_header("LLM Benchmarks (Custom)")
        for idx, task in enumerate(TASKS.values(), start=1):
            print(f" [{idx}] {task.label}")
            print(f"      - {task.description}")
        print(" [0] 돌아가기")
        print("--------------------------------------------------")
        choice = input(" 선택 >> ").strip()
        if choice == "0":
            return
        try:
            task = list(TASKS.values())[int(choice) - 1]
        except Exception:
            print("잘못된 선택입니다.")
            input("엔터를 눌러 계속...")
            continue
        optimizer = input(" Optimizer [adamw] >> ").strip() or "adamw"
        args = input(" 추가 CLI 인자 (예: --epochs 1 --batch-size 2) >> ").strip()
        extra: List[str] = []
        if args:
            try:
                extra = shlex.split(args)
            except ValueError:
                print("인자 파싱 실패. 기본 옵션으로 실행합니다.")
                extra = []
        print(" [0] 취소 / [Enter] 계속 실행")
        confirm = input(" 실행하시겠습니까? [Enter=yes/0=no] >> ").strip()
        if confirm == "0":
            continue
        _run_llm_task(task.key, optimizer, extra)
        input("엔터를 눌러 계속...")


def run_llm_smoke_menu():
    configure_hf_cache(DATA_DIR)
    ensure_dir(DATA_DIR)
    while True:
        clear_screen()
        print_header("LLM Smoke Tests")
        for idx, scenario in enumerate(LLM_SMOKE_SCENARIOS, start=1):
            ready = "✅" if _llm_dataset_ready(scenario["task"]) else "❌"
            print(f" [{idx}] {scenario['label']} | 데이터셋 {ready}")
        print(" [A] 전체 스모크 일괄 실행")
        print(" [0] 돌아가기")
        choice = input(" 선택 >> ").strip()
        if choice == "0":
            return
        if choice.lower() == "a":
            for scenario in LLM_SMOKE_SCENARIOS:
                status = "READY" if _llm_dataset_ready(scenario["task"]) else "MISS"
                print(f"[LLM-SMOKE-ALL] {scenario['label']} ({status})")
                _run_llm_task(scenario["task"], scenario["optimizer"], scenario["extra"])
            input("엔터를 눌러 계속...")
            continue
        try:
            scenario = LLM_SMOKE_SCENARIOS[int(choice) - 1]
        except Exception:
            print("잘못된 선택입니다.")
            input("엔터를 눌러 계속...")
            continue
        print(f"[LLM-SMOKE] {scenario['label']}")
        _run_llm_task(scenario["task"], scenario["optimizer"], scenario["extra"])
        input("엔터를 눌러 계속...")
