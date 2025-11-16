# setup/env_check.py
from __future__ import annotations
import subprocess, sys
from pathlib import Path
from typing import List
import importlib


def _parse_requirements(path: Path) -> List[str]:
    pkgs = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        # 'torch==2.3.0' -> 'torch'
        pkgs.append(line.split()[0].split("==")[0])
    return pkgs


def _is_installed(pkg: str) -> bool:
    try:
        importlib.import_module(pkg)
        return True
    except Exception:
        return False


def check_requirements(req_path: Path) -> list[str]:
    if not req_path.exists():
        print(f"[WARN] requirements.txt 가 없습니다: {req_path}")
        return []

    needed = _parse_requirements(req_path)
    missing = [p for p in needed if not _is_installed(p)]
    return missing


def install_requirements(req_path: Path):
    """단순 래퍼. 실제로는 사용자가 직접 실행하도록 안내해도 됨."""
    cmd = [sys.executable, "-m", "pip", "install", "-r", str(req_path)]
    print("실행 중:", " ".join(cmd))
    subprocess.run(cmd, check=False)


def run_env_setup(req_path: Path):
    print("=== 환경 / 의존성 체크 ===")
    missing = check_requirements(req_path)

    if not missing:
        print("✅ 모든 requirements 충족")
        input("엔터를 눌러 계속...")
        return

    print("❌ 부족한 패키지:")
    for m in missing:
        print("  -", m)

    print("\n[1] 자동으로 pip install 실행")
    print("[2] 설치 명령어만 출력")
    print("[기타] 돌아가기")
    choice = input("선택 >> ").strip()

    if choice == "1":
        install_requirements(req_path)
        input("엔터를 눌러 계속...")
    elif choice == "2":
        print("\n아래 명령을 수동으로 실행하세요:")
        print(f"  {sys.executable} -m pip install -r {req_path}")
        input("엔터를 눌러 계속...")
    else:
        return