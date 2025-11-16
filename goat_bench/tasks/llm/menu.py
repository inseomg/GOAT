from __future__ import annotations

import shlex
import subprocess
import sys
from pathlib import Path
from typing import Dict, List

from goat_bench.utils.helpers import clear_screen, print_header, ensure_dir, ConsoleSpinner

SCRIPT_PATH = Path(__file__).resolve().with_name("bench.py")


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

LLM_QUICK_PRESETS: Dict[str, List[str]] = {
    "l1_pretrain": ["--epochs", "1", "--batch-size", "4"],
    "l2_sft": ["--epochs", "1", "--batch-size", "2"],
    "l3_llama": ["--epochs", "1", "--batch-size", "1"],
}


def _run_llm_task(task_key: str, optimizer: str, extra: List[str]):
    cmd = [sys.executable, str(SCRIPT_PATH), "--task", task_key, "--optimizer", optimizer, *extra]
    spinner = ConsoleSpinner(f"[LLM] {task_key} 실행 중")
    spinner.start()
    try:
        subprocess.run(cmd, check=False)
    finally:
        spinner.stop()


def run_llm_quick(task_key: str):
    info = TASKS[task_key]
    print(f"[LLM-QUICK] {info.label}")
    preset = LLM_QUICK_PRESETS.get(task_key, [])
    _run_llm_task(task_key, "adamw", preset)
    input("엔터를 눌러 계속...")


def run_llm_menu():
    ensure_dir(Path("data"))
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
        _run_llm_task(task.key, optimizer, extra)
        input("엔터를 눌러 계속...")
