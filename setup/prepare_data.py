# setup/prepare_data.py
from __future__ import annotations
import os
from pathlib import Path
from typing import Callable, Dict, Optional

from .dataset_registry import list_datasets, get_dataset
from goat_bench.utils.helpers import clear_screen, print_header, ensure_dir


def _dataset_root(data_root: Path, ds_name: str) -> Path:
    return data_root / ds_name


def has_dataset(ds_name: str, data_root: Path) -> bool:
    path = _dataset_root(data_root, ds_name)
    return path.exists() and any(path.iterdir())


def _require_torchvision():
    try:
        import torchvision  # type: ignore
    except ImportError as exc:
        raise RuntimeError("torchvision이 설치되어 있어야 자동 다운로드가 가능합니다.") from exc
    return torchvision


def _download_cifar(ds_class: str):
    def _runner(root: Path):
        tv = _require_torchvision()
        dataset_cls = getattr(tv.datasets, ds_class)
        for train in (True, False):
            dataset_cls(root=str(root), train=train, download=True)

    return _runner


def _download_tinyimagenet(root: Path):
    from goat_bench.tasks.cv.datasets import ensure_tinyimagenet

    ensure_tinyimagenet(root)


def _manual_instructions(ds_name: str, root: Path):
    print(f"[MANUAL] '{ds_name}' 는 수동 다운로드가 필요합니다.")
    print(f"  - {root} 경로에 데이터셋을 배치하세요.")
    print("  - 자세한 방법은 README_data.md 를 참고하세요.")


DOWNLOADERS: Dict[str, Callable[[Path], None]] = {
    "cifar10": _download_cifar("CIFAR10"),
    "cifar100": _download_cifar("CIFAR100"),
    "tinyimagenet": _download_tinyimagenet,
}


def download_dataset(ds_name: str, data_root: Path):
    info = get_dataset(ds_name)
    root = _dataset_root(data_root, ds_name)
    ensure_dir(root)

    downloader = DOWNLOADERS.get(ds_name)
    if info.requires_manual_download or downloader is None:
        _manual_instructions(ds_name, root)
    else:
        print(f"[AUTO] '{ds_name}' 자동 다운로드/준비를 시작합니다...")
        try:
            downloader(root)
            print(f"[DONE] '{ds_name}' 다운로드/준비 완료: {root}")
        except RuntimeError as exc:
            print(f"[ERROR] 자동 다운로드에 실패했습니다: {exc}")
            print("torch / torchvision 설치 및 네트워크 상태를 확인하세요.")
    input("엔터를 눌러 계속...")


def run_dataset_manager(data_root: Path):
    ensure_dir(data_root)
    while True:
        clear_screen()
        print_header("데이터셋 관리")

        ds_list = list_datasets()
        for idx, info in enumerate(ds_list, start=1):
            status = "✅" if has_dataset(info.name, data_root) else "❌"
            tag_str = ",".join(info.tags)
            print(f" [{idx}] {info.name:<15} ({info.domain}) {status}")
            print(f"      ~ {info.description}")
            print(f"      ~ size≈{info.approx_size_gb}GB, tags={tag_str}")
        print(" [0] 돌아가기")
        print("--------------------------------------------------")

        choice = input(" 선택 (번호) >> ").strip()
        if choice == "0":
            break

        try:
            idx = int(choice) - 1
            info = ds_list[idx]
        except Exception:
            print("잘못된 번호입니다.")
            input("엔터를 눌러 계속...")
            continue

        clear_screen()
        print_header(f"데이터셋: {info.name}")
        status = "✅ 있음" if has_dataset(info.name, data_root) else "❌ 없음"
        print(" 상태 :", status)
        print(" 설명 :", info.description)
        print(" 용량 :", f"≈{info.approx_size_gb}GB")
        print(" 태그 :", ", ".join(info.tags))
        print("\n [1] 다운로드/준비")
        print(" [기타] 뒤로가기")

        sub = input(" 선택 >> ").strip()
        if sub == "1":
            download_dataset(info.name, data_root)
