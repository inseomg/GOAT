# setup/prepare_data.py
from __future__ import annotations
import os
import shutil
import inspect
from pathlib import Path
from typing import Callable, Dict, Optional

from .dataset_registry import DatasetInfo, list_datasets, get_dataset
from goat_bench.utils.helpers import clear_screen, print_header, ensure_dir, ConsoleSpinner, configure_hf_cache



_HF_NLP_DATASETS = {
    "sst2",
    "mrpc",
    "stsb",
    "qqp",
    "ag_news",
    "boolq",
    "rte",
    "cb",
    "anli",
    "hellaswag",
    "piqa",
    "copa",
    "winogrande",
    "alpha_nli",
    "squad_v2",
    "wmt14_en_de",
    "wmt14_de_en",
    "xsum",
    "wikitext2",
    "gsm8k",
    "multirc",
    "record",
    "hotpotqa",
}

_DATASET_NAME_ALIASES = {
    "imagenet": "imagenet1k",
}


def _tinyimagenet_ready(root: Path) -> bool:
    return (root / "tiny-imagenet-200" / "train").exists()


def _cifar_ready(root: Path, marker: str) -> bool:
    return (root / marker).exists()


_DATASET_READY_CHECKS: Dict[str, Callable[[Path], bool]] = {
    "tinyimagenet": _tinyimagenet_ready,
    "cifar10": lambda root: _cifar_ready(root, "cifar-10-batches-py"),
    "cifar100": lambda root: _cifar_ready(root, "cifar-100-python"),
}


def dataset_root(data_root: Path, ds_name: str) -> Path:
    """
    Compute the canonical storage directory for a dataset under the shared data root.
    """
    name = ds_name.lower()
    if name in _HF_NLP_DATASETS:
        return Path(data_root) / "hf-cache" / name
    alias = _DATASET_NAME_ALIASES.get(name, name)
    return Path(data_root) / alias


_dataset_root = dataset_root  # backward compatibility for older imports


def _purge_hf_cache(ds_root: Path, patterns: list[str]):
    """
    Remove stale HF cache entries (e.g., legacy scripts) for the given dataset patterns.
    """
    # remove dataset-specific root (keeps parent directories)
    if ds_root.exists():
        shutil.rmtree(ds_root, ignore_errors=True)
    hf_root = ds_root.parent  # typically data/hf-cache
    hub_dir = hf_root / "hub"
    if hub_dir.exists():
        for pat in patterns:
            # HF hub uses datasets--<namespace>--<name>
            for sub in hub_dir.glob(f"datasets--*{pat}*"):
                shutil.rmtree(sub, ignore_errors=True)


def has_dataset(ds_name: str, data_root: Path) -> bool:
    name = ds_name.lower()
    path = dataset_root(data_root, ds_name)
    check = _DATASET_READY_CHECKS.get(name)
    if check:
        try:
            return check(path)
        except Exception:
            return False
    if name in _HF_NLP_DATASETS:
        if path.exists() and any(path.iterdir()):
            return True
        # fallback: check shared HF cache (when cache_dir wasn't dataset-specific)
        base = Path(data_root) / "hf-cache"
        candidates = [
            base / "datasets",
            base / "hub",
        ]
        tokens = {name, name.replace("_", "-")}
        for cand in candidates:
            if not cand.exists():
                continue
            for sub in cand.glob("*"):
                if any(tok in sub.name for tok in tokens):
                    return True
        return False
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


_TRUST_REMOTE_CODE_SUPPORTED: bool | None = None


def _supports_trust_remote_code(load_dataset) -> bool:
    global _TRUST_REMOTE_CODE_SUPPORTED
    if _TRUST_REMOTE_CODE_SUPPORTED is None:
        try:
            sig = inspect.signature(load_dataset)
            _TRUST_REMOTE_CODE_SUPPORTED = "trust_remote_code" in sig.parameters
        except Exception:
            _TRUST_REMOTE_CODE_SUPPORTED = False
    return bool(_TRUST_REMOTE_CODE_SUPPORTED)


def _load_dataset_compat(hf_name: str, subset: Optional[str], cache_dir: Path, *, trust_remote_code: bool, revision: Optional[str] = None):
    """
    Wrapper for datasets.load_dataset that copes with:
      - newer versions deprecating trust_remote_code
      - community datasets that still require trust_remote_code=True
      - numpy/Scipy ABI errors (e.g., numpy 2.3.x + scipy) with actionable guidance
    """
    from datasets import load_dataset  # type: ignore

    supports_trc = _supports_trust_remote_code(load_dataset)
    kwargs = {"cache_dir": str(cache_dir)}
    if revision:
        kwargs["revision"] = revision

    def _call(use_trust: bool):
        opts = dict(kwargs)
        if use_trust and supports_trc:
            opts["trust_remote_code"] = True
        return load_dataset(hf_name, subset, **opts)

    try:
        return _call(trust_remote_code)
    except TypeError as exc:
        if "trust_remote_code" in str(exc):
            return _call(False)
        raise
    except ValueError as exc:
        msg = str(exc)
        if "trust_remote_code is not supported anymore" in msg:
            return _call(False)
        if "trust_remote_code=True" in msg and supports_trc and not trust_remote_code:
            return _call(True)
        raise
    except ImportError as exc:
        if "_center" in str(exc) and "numpy" in str(exc):
            raise RuntimeError(
                "데이터셋 로딩 중 numpy/Scipy ABI 오류가 발생했습니다. "
                "requirements.txt 버전에 맞춰 `pip install 'numpy>=1.26,<2.2' 'scipy>=1.10,<1.13' --upgrade --force-reinstall` 를 실행하세요."
            ) from exc
        raise


def _download_hf_dataset(dataset_key: str, hf_name: str, subset: Optional[str] = None, *, trust_remote_code: bool = False):
    def _runner(root: Path):
        try:
            from datasets import load_dataset  # type: ignore
        except ImportError as exc:
            raise RuntimeError("`datasets` 패키지가 필요합니다. pip install datasets 로 설치하세요.") from exc
        ensure_dir(root)
        base_cache = root.parent if root.parent.name == "hf-cache" else configure_hf_cache(root)
        print(f"[HF] {hf_name}{f'/{subset}' if subset else ''} 다운로드를 시작합니다 (cache_dir={base_cache})")
        _load_dataset_compat(hf_name, subset, base_cache, trust_remote_code=trust_remote_code)
        # marker so has_dataset() reports ready even though cache is shared
        (root / ".ready").write_text("ready", encoding="utf-8")
        print("[HF] 다운로드 완료")

    return _runner


def _download_xsum_dataset(root: Path):
    ensure_dir(root)
    # Clean up legacy scripts that break datasets>=3.0
    _purge_hf_cache(root, ["xsum", "GEM--xsum", "EdinburghNLP--xsum"])
    attempts = [
        ("xsum", None),
        ("xsum", None, "refs/convert/parquet"),
        ("GEM/xsum", None),
        ("EdinburghNLP/xsum", None),
    ]
    errors = []
    for entry in attempts:
        if len(entry) == 3:
            hf_name, subset, revision = entry
        else:
            hf_name, subset = entry
            revision = None
        try:
            _purge_hf_cache(root, ["xsum", "GEM--xsum", "EdinburghNLP--xsum"])
            print(f"[HF] {hf_name}{f'/{subset}' if subset else ''} 다운로드를 시도합니다 (cache_dir={root})")
            base_cache = root.parent if root.parent.name == "hf-cache" else configure_hf_cache(root)
            _load_dataset_compat(hf_name, subset, base_cache, trust_remote_code=True, revision=revision)
            (root / ".ready_xsum").write_text("ready", encoding="utf-8")
            print("[HF] XSum 다운로드 완료")
            return
        except Exception as exc:
            errors.append((hf_name, subset, str(exc)))
            print(f"[HF] {hf_name} 실패: {exc}")
    raise RuntimeError(f"XSum 데이터를 다운로드할 수 없습니다. 시도: {errors}")


def _download_alpha_nli(root: Path):
    ensure_dir(root)
    _purge_hf_cache(root, ["alpha_nli", "abductive_nli", "abductive-nli", "Rowan--abductive_nli", "XiangRong--abductive_nli", "allenai--art"])
    candidates = [
        ("alpha_nli", None),
        ("abductive_nli", None),
        ("abductive-nli", None),
        ("Rowan/abductive_nli", None),
        ("XiangRong/abductive_nli", None),
        ("allenai/art", "anli"),
    ]
    tried = []
    for candidate, subset in candidates:
        tried.append(candidate if subset is None else f"{candidate}/{subset}")
        try:
            print(f"[HF] {candidate}{f'/{subset}' if subset else ''} 다운로드를 시작합니다 (cache_dir={root})")
            base_cache = root.parent if root.parent.name == "hf-cache" else configure_hf_cache(root)
            _load_dataset_compat(candidate, subset, base_cache, trust_remote_code=True)
            marker = root / ".ready_alpha_nli"
            marker.write_text("ready", encoding="utf-8")
            print("[HF] 다운로드 완료")
            return
        except Exception as exc:
            print(f"[HF] {candidate} 실패: {exc}")
    raise RuntimeError(f"alpha_nli 데이터셋을 다운로드할 수 없습니다. 시도: {tried}")


def _download_wmt(direction: str):
    def _runner(root: Path):
        ensure_dir(root)

        possible = [direction, "-".join(reversed(direction.split("-")))]
        base_configs = ["cs-en", "de-en", "fr-en", "hi-en", "ru-en"]
        for cfg in base_configs:
            if cfg not in possible:
                possible.append(cfg)
        errors = []
        for cfg in possible:
            try:
                print(f"[HF] wmt14/{cfg} 다운로드를 시작합니다 (cache_dir={root})")
                base_cache = root.parent if root.parent.name == "hf-cache" else configure_hf_cache(root)
                _load_dataset_compat("wmt14", cfg, base_cache, trust_remote_code=True)
                marker = root / f".ready_wmt14_{direction}"
                marker.write_text("ready", encoding="utf-8")
                print("[HF] 다운로드 완료")
                return
            except Exception as exc:
                errors.append((cfg, str(exc)))
        raise RuntimeError(f"WMT14 {direction} 다운로드 실패: {errors}")

    return _runner


def _manual_instructions(ds_name: str, root: Path):
    print(f"[MANUAL] '{ds_name}' 는 수동 다운로드가 필요합니다.")
    print(f"  - {root} 경로에 데이터셋을 배치하세요.")
    if ds_name.lower() == "ade20k":
        print("  - MIT Scene Parsing Challenge 페이지에서 ADEChallengeData2016.zip 을 내려받아 압축 해제하세요.")
        print("  - https://data.csail.mit.edu/places/ADEchallenge/ADEChallengeData2016.zip (계정 필요)")
        print("  - 압축을 풀면 images/와 annotations/ 디렉터리가 생성되어야 합니다.")
    else:
        print("  - 자세한 방법은 README_data.md 를 참고하세요.")


DOWNLOADERS: Dict[str, Callable[[Path], None]] = {
    "cifar10": _download_cifar("CIFAR10"),
    "cifar100": _download_cifar("CIFAR100"),
    "tinyimagenet": _download_tinyimagenet,
    # Hugging Face datasets
    "sst2": _download_hf_dataset("sst2", "glue", "sst2"),
    "mrpc": _download_hf_dataset("mrpc", "glue", "mrpc"),
    "stsb": _download_hf_dataset("stsb", "glue", "stsb"),
    "qqp": _download_hf_dataset("qqp", "glue", "qqp"),
    "ag_news": _download_hf_dataset("ag_news", "ag_news", None),
    "boolq": _download_hf_dataset("boolq", "super_glue", "boolq"),
    "rte": _download_hf_dataset("rte", "super_glue", "rte"),
    "cb": _download_hf_dataset("cb", "super_glue", "cb"),
    "anli": _download_hf_dataset("anli", "anli"),
    "hellaswag": _download_hf_dataset("hellaswag", "Rowan/hellaswag"),
    "piqa": _download_hf_dataset("piqa", "lighteval/piqa"),
    "copa": _download_hf_dataset("copa", "super_glue", "copa"),
    "winogrande": _download_hf_dataset("winogrande", "winogrande", "winogrande_xl"),
    "alpha_nli": lambda root: _download_alpha_nli(root),
    "squad_v2": _download_hf_dataset("squad_v2", "squad_v2"),
    "wmt14_en_de": _download_wmt("en-de"),
    "wmt14_de_en": _download_wmt("de-en"),
    "xsum": _download_xsum_dataset,
    "wikitext2": _download_hf_dataset("wikitext2", "wikitext", "wikitext-2-raw-v1"),
    "wikitext103": _download_hf_dataset("wikitext103", "wikitext", "wikitext-103-raw-v1"),
    "gsm8k": _download_hf_dataset("gsm8k", "gsm8k", "main"),
    "multirc": _download_hf_dataset("multirc", "super_glue", "multirc"),
    "record": _download_hf_dataset("record", "super_glue", "record"),
    "hotpotqa": _download_hf_dataset("hotpotqa", "hotpot_qa", "distractor"),
    "alpaca": _download_hf_dataset("alpaca", "tatsu-lab/alpaca"),
}


def download_dataset(ds_name: str, data_root: Path, pause: bool = True):
    configure_hf_cache(data_root)
    info = get_dataset(ds_name)
    root = dataset_root(data_root, ds_name)
    ensure_dir(root)
    if has_dataset(ds_name, data_root):
        print(f"[SKIP] '{ds_name}' 은 이미 준비되어 있습니다. 필요하면 데이터를 삭제 후 다시 실행하세요.")
        if pause:
            input("엔터를 눌러 계속...")
        return

    if "heavy" in info.tags:
        ans = input(f"[경고] '{ds_name}' 은 대용량(~{info.approx_size_gb}GB) 데이터입니다. 다운로드 하시겠습니까? [y/N] >> ").strip().lower()
        if ans not in ("y", "yes"):
            print("다운로드를 취소했습니다.")
            input("엔터를 눌러 계속...")
            return

    downloader = DOWNLOADERS.get(ds_name)
    if info.requires_manual_download or downloader is None:
        if "license_warning" in info.tags:
            print("⚠️ 라이선스/약관 동의가 필요한 데이터셋입니다.")
        _manual_instructions(ds_name, root)
    else:
        spinner = ConsoleSpinner(f"[AUTO] '{ds_name}' 다운로드 중")
        spinner.start()
        try:
            downloader(root)
            spinner.stop()
            print(f"[DONE] '{ds_name}' 다운로드/준비 완료: {root}")
        except RuntimeError as exc:
            spinner.stop()
            print(f"[ERROR] 자동 다운로드에 실패했습니다: {exc}")
            print("torch / torchvision 설치 및 네트워크 상태를 확인하세요.")
    if pause:
        input("엔터를 눌러 계속...")


def _bulk_targets(min_size_gb: float, max_size_gb: float | None) -> list[DatasetInfo]:
    targets = []
    for info in list_datasets():
        if info.requires_manual_download:
            continue
        size = info.approx_size_gb
        if size < min_size_gb:
            continue
        if max_size_gb is not None and size > max_size_gb:
            continue
        targets.append(info)
    return targets


def download_by_size_range(
    data_root: Path,
    *,
    min_size_gb: float = 0,
    max_size_gb: float | None = None,
    label: str = "",
):
    targets = _bulk_targets(min_size_gb, max_size_gb)
    if not targets:
        max_label = f"~{max_size_gb}GB" if max_size_gb is not None else "이상"
        print(f"[INFO] {min_size_gb}GB {max_label} 자동 다운로드 대상이 없습니다.")
        input("엔터를 눌러 계속...")
        return

    if not label:
        label = f"{min_size_gb}~{f'{max_size_gb}' if max_size_gb is not None else '∞'}GB"
    print(f"[BULK] {label} 구간의 {len(targets)}개 데이터셋을 순차적으로 다운로드합니다.")
    for info in targets:
        if has_dataset(info.name, data_root):
            print(f"[SKIP] {info.name} 은 이미 준비되어 있습니다.")
            continue
        print(f"[BULK] '{info.name}' 다운로드를 시작합니다...")
        download_dataset(info.name, data_root, pause=False)
    print("[BULK] 일괄 다운로드가 완료되었습니다.")
    input("엔터를 눌러 계속...")


def run_dataset_manager(data_root: Path):
    ensure_dir(data_root)
    while True:
        clear_screen()
        print_header("데이터셋 관리")

        ds_list = list_datasets()
        print(" 번호 | 데이터셋        | 상태 | 도메인 | 태그")
        print("------+----------------+------+--------+---------------------------")
        for idx, info in enumerate(ds_list, start=1):
            status = "READY" if has_dataset(info.name, data_root) else "NONE "
            tag_str = ",".join(info.tags)
            print(
                f" [{idx}] {info.name:<14} {status} | {info.domain:<6} | tags: {tag_str}"
            )
            print(f"      설명: {info.description}")
        print(" [A] 0~5GB 자동 다운로드")
        print(" [B] 5~25GB 자동 다운로드")
        print(" [C] 25GB 이상 자동 다운로드")
        print(" [0] 돌아가기")
        print("--------------------------------------------------")

        choice = input(" 선택 (번호) >> ").strip()
        if choice.lower() == "a":
            download_by_size_range(data_root, min_size_gb=0, max_size_gb=5, label="0~5GB")
            continue
        if choice.lower() == "b":
            download_by_size_range(data_root, min_size_gb=5, max_size_gb=25, label="5~25GB")
            continue
        if choice.lower() == "c":
            download_by_size_range(data_root, min_size_gb=25, max_size_gb=None, label="25GB 이상")
            continue
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
        print(" 위치 :", dataset_root(data_root, info.name))
        if "heavy" in info.tags:
            print(" ⚠️ 경고: 대용량 데이터셋입니다. 충분한 디스크 공간과 네트워크를 확보하세요.")
        if "license_warning" in info.tags:
            print(" ⚠️ 라이선스 확인 필요: 공식 배포처의 약관을 반드시 읽고 동의해야 합니다.")
        print("\n [1] 다운로드/준비")
        print(" [기타] 뒤로가기")

        sub = input(" 선택 >> ").strip()
        if sub == "1":
            download_dataset(info.name, data_root)
