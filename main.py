# main.py
from __future__ import annotations
import sys
from pathlib import Path
import time

# 로컬 모듈
from setup.env_check import run_env_setup
from setup.prepare_data import run_dataset_manager
from goat_bench.utils.helpers import clear_screen, print_header

# NOTE: tasks/ 쪽은 아직 미구현이라 임시 stub 사용
try:
    from goat_bench.tasks.registry import run_benchmark_menu
except Exception:  # 아직 없을 수 있으니
    def run_benchmark_menu():
        print("[WARN] tasks/registry.py 가 아직 구현되지 않았습니다.")
        input("엔터를 눌러 계속하세요...")

# 결과 보기용 (나중에 확장)
from goat_bench.utils.results import show_results_menu


ROOT = Path(__file__).resolve().parent
ASCII_GOAT = [
    r"  /\_/\\ ",
    r" ( o.o )",
    r"  > ^ < ",
]
MENU_FRAMES = ["◈", "◆", "◇", "◆"]


def main():
    while True:
        clear_screen()
        frame = MENU_FRAMES[int(time.time() * 3) % len(MENU_FRAMES)]
        print_header(f"{frame} GOAT Benchmark {frame}")
        for line in ASCII_GOAT:
            print(f"        {line}")
        print()

        print(" [1] 환경 / 의존성 체크 (requirements)")
        print(" [2] 데이터셋 관리 (다운로드 / 상태 확인)")
        print(" [3] 벤치마크 실행 (task / model / optimizer)")
        print(" [4] 결과 보기 (JSON 요약)")
        print(" [5] 종료")
        print("--------------------------------------------------")

        choice = input(" 선택 >> ").strip()

        if choice == "1":
            run_env_setup(ROOT / "requirements.txt")
        elif choice == "2":
            run_dataset_manager(ROOT / "data")
        elif choice == "3":
            run_benchmark_menu()
        elif choice == "4":
            show_results_menu(ROOT / "results")
        elif choice == "5":
            print("종료합니다.")
            sys.exit(0)
        else:
            print("잘못된 입력입니다.")
            input("엔터를 눌러 계속하세요...")


if __name__ == "__main__":
    main()
