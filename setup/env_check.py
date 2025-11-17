# setup/env_check.py
from __future__ import annotations
import subprocess, sys
from pathlib import Path
from typing import List, Dict
import importlib
from packaging import version


REQ_IMPORT_MAP: Dict[str, str] = {
    "pillow": "PIL",
    "lion-pytorch": "lion_pytorch",
    "pytorch-optimizer": "pytorch_optimizer",
    "sentencepiece": "sentencepiece",
    "rouge-score": "rouge_score",
    "absl-py": "absl",
}


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
    module_name = REQ_IMPORT_MAP.get(pkg.lower(), pkg)
    try:
        importlib.import_module(module_name)
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


def _conflict_warnings() -> list[str]:
    """
    Surface known compatibility pitfalls early (numpy/pillow too new, etc.).
    """
    warns: list[str] = []
    try:
        import numpy as np

        if version.parse(np.__version__) >= version.parse("2.2.0"):
            warns.append(
                f"numpy {np.__version__} 감지됨 → TensorFlow/numba/opencv와 충돌합니다. "
                "다음 명령으로 재설치하세요: pip install 'numpy>=1.26,<2.2' 'scipy>=1.10,<1.13' --upgrade --force-reinstall"
            )
    except Exception:
        pass

    try:
        import PIL

        if hasattr(PIL, "__version__") and version.parse(PIL.__version__) >= version.parse("12.0.0"):
            warns.append(
                f"Pillow {PIL.__version__} 감지됨 → gradio 5.x 와 충돌합니다. "
                "다음 명령으로 재설치하세요: pip install 'pillow>=9,<12' --upgrade --force-reinstall"
            )
    except Exception:
        pass

    return warns


def install_requirements(req_path: Path):
    """단순 래퍼. 실제로는 사용자가 직접 실행하도록 안내해도 됨."""
    cmd = [
        sys.executable,
        "-m",
        "pip",
        "install",
        "--upgrade",
        "--no-cache-dir",
        "-r",
        str(req_path),
    ]
    print("실행 중:", " ".join(cmd))
    subprocess.run(cmd, check=False)


def run_env_setup(req_path: Path):
    print("=== 환경 / 의존성 체크 ===")
    missing = check_requirements(req_path)
    conflicts = _conflict_warnings()

    if not missing:
        print("✅ 모든 requirements 충족")
        if conflicts:
            print("\n⚠️ 버전 충돌 경고:")
            for msg in conflicts:
                print(" -", msg)
        input("엔터를 눌러 계속...")
        return

    print("❌ 부족한 패키지:")
    for m in missing:
        print("  -", m)

    if conflicts:
        print("\n⚠️ 추가 주의사항:")
        for msg in conflicts:
            print(" -", msg)

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
