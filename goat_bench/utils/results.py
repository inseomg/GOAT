# goat_bench/utils/results.py
from __future__ import annotations
from pathlib import Path
from typing import Any, Dict, List

from .helpers import clear_screen, print_header, load_json


def _list_summary_files(results_root: Path) -> List[Path]:
    summaries = results_root / "summaries"
    if not summaries.exists():
        return []
    return sorted(p for p in summaries.glob("*.json"))


def show_results_menu(results_root: Path):
    clear_screen()
    print_header("결과 보기 (JSON 요약)")
    files = _list_summary_files(results_root)

    if not files:
        print("아직 요약 JSON 파일이 없습니다.")
        input("엔터를 눌러 계속...")
        return

    for idx, p in enumerate(files, start=1):
        print(f" [{idx}] {p.name}")
    print(" [0] 돌아가기")
    choice = input(" 선택 >> ").strip()
    if choice == "0":
        return

    try:
        idx = int(choice) - 1
        path = files[idx]
    except Exception:
        print("잘못된 번호입니다.")
        input("엔터를 눌러 계속...")
        return

    data: Dict[str, Any] = load_json(path)
    clear_screen()
    print_header(f"요약: {path.name}")
    # 아주 간단한 pretty print (나중에 포맷 마음껏 바꾸면 됨)
    from pprint import pprint
    pprint(data)
    input("\n엔터를 눌러 계속...")