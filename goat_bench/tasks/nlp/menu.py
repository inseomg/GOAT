from __future__ import annotations

import json
import os
import shlex
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple

from goat_bench.utils.helpers import clear_screen, print_header, ensure_dir
from setup.prepare_data import has_dataset, dataset_root


@dataclass
class NLPTaskInfo:
    key: str
    label: str
    script: str  # "part1" or "part2"
    description: str
    default_model: str | None = None
    default_amp: str = "bf16"
    heavy_warning: bool = False


PART1_TASKS = [
    NLPTaskInfo("suite:easy", "Suite: EASY (GLUE/MCQ + LM)", "part1", "sst2/mrpc/hellaswag/piqa/ag_news/lm 연속 실행"),
    NLPTaskInfo("suite:medium", "Suite: MEDIUM (GLUE/QA)", "part1", "boolq/rte/cb/copa/anli/squad_v2/xsum 연속 실행"),
    NLPTaskInfo("suite:hard", "Suite: HARD (WMT14 + GSM8K)", "part1", "wmt14_en_de, gsm8k 하드 태스크", None, "bf16", heavy_warning=True),
    NLPTaskInfo("sst2", "GLUE - SST-2 Sentiment", "part1", "Binary sentiment classification", "roberta-base"),
    NLPTaskInfo("mrpc", "GLUE - MRPC Paraphrase", "part1", "Paraphrase detection", "roberta-base"),
    NLPTaskInfo("stsb", "GLUE - STS-B Regression", "part1", "Semantic textual similarity", "roberta-base"),
    NLPTaskInfo("qqp", "GLUE - QQP Paraphrase", "part1", "Quora duplicate questions", "bert-base-uncased"),
    NLPTaskInfo("ag_news", "AG News Classification", "part1", "News topic classification", "bert-base-uncased"),
    NLPTaskInfo("boolq", "SuperGLUE - BoolQ", "part1", "Binary QA classification", "roberta-base"),
    NLPTaskInfo("rte", "SuperGLUE - RTE", "part1", "Recognizing textual entailment", "roberta-base"),
    NLPTaskInfo("cb", "SuperGLUE - CB", "part1", "CommitmentBank entailment", "roberta-base"),
    NLPTaskInfo("anli", "ANLI (round-mix)", "part1", "Adversarial NLI", "roberta-large"),
    NLPTaskInfo("hellaswag", "HellaSwag MCQ", "part1", "Commonsense completion (4-way MCQ)", "roberta-base"),
    NLPTaskInfo("piqa", "PIQA MCQ", "part1", "Physical commonsense MCQ", "roberta-base"),
    NLPTaskInfo("copa", "COPA MCQ", "part1", "Choice of plausible alternative", "roberta-base"),
    NLPTaskInfo("winogrande", "WinoGrande MCQ", "part1", "Pronoun resolution MCQ", "roberta-base"),
    NLPTaskInfo("alpha_nli", "AlphaNLI (abductive)", "part1", "Abductive reasoning MCQ", "roberta-base"),
    NLPTaskInfo("squad_v2", "QA - SQuAD v2", "part1", "Question answering with no-answer option", "bert-base-uncased"),
    NLPTaskInfo("wmt14_en_de", "MT - WMT14 EN→DE", "part1", "Machine translation (English→German)", "Helsinki-NLP/opus-mt-en-de", "bf16", heavy_warning=True),
    NLPTaskInfo("wmt14_de_en", "MT - WMT14 DE→EN", "part1", "Machine translation (German→English)", "Helsinki-NLP/opus-mt-de-en", "bf16", heavy_warning=True),
    NLPTaskInfo("xsum", "Summarization - XSum", "part1", "Abstractive summarization", "facebook/bart-base"),
    NLPTaskInfo("lm", "Language Modeling - WikiText-2", "part1", "GPT-style causal LM fine-tune", "distilgpt2"),
    NLPTaskInfo("gsm8k", "Math - GSM8K", "part1", "Math word problems (auto-regressive)", "gpt2", "none"),
]

PART2_TASKS = [
    NLPTaskInfo("multirc", "SuperGLUE - MultiRC", "part2", "Multi-sentence reading comprehension", "roberta-base"),
    NLPTaskInfo("record", "SuperGLUE - ReCoRD", "part2", "Entity cloze MCQ", "roberta-base"),
    NLPTaskInfo("hotpotqa", "HotpotQA (distractor)", "part2", "Multi-hop QA with spans", "bert-base-uncased", "bf16", heavy_warning=True),
]

TASK_INFO: Dict[str, NLPTaskInfo] = {info.key: info for info in PART1_TASKS + PART2_TASKS}

TASK_GROUPS: List[Tuple[str, List[str]]] = [
    ("Suites", ["suite:easy", "suite:medium", "suite:hard"]),
    ("GLUE / SuperGLUE Classification", ["sst2", "mrpc", "stsb", "qqp", "ag_news", "boolq", "rte", "cb", "anli"]),
    ("Multiple Choice / Reasoning", ["hellaswag", "piqa", "copa", "winogrande", "alpha_nli"]),
    ("Question Answering / Seq2Seq / LM", ["squad_v2", "wmt14_en_de", "wmt14_de_en", "xsum", "lm", "gsm8k"]),
    ("Heavy Benchmarks", ["multirc", "record", "hotpotqa"]),
]

SUPPORTED_OPTIMIZERS = ["adamw", "lion", "soap"]
NLP_SMOKE_DEFAULTS: Dict[str, List[str]] = {
    "suite:easy": ["--epochs", "1", "--batch-size", "32"],
    "suite:medium": ["--epochs", "1", "--batch-size", "32"],
    "suite:hard": ["--epochs", "1", "--batch-size", "32"],
    "sst2": ["--epochs", "1", "--batch-size", "32"],
    "mrpc": ["--epochs", "1", "--batch-size", "32"],
    "stsb": ["--epochs", "1", "--batch-size", "32"],
    "qqp": ["--epochs", "1", "--batch-size", "32"],
    "ag_news": ["--epochs", "1", "--batch-size", "64"],
    "boolq": ["--epochs", "1", "--batch-size", "32"],
    "rte": ["--epochs", "1", "--batch-size", "32"],
    "cb": ["--epochs", "1", "--batch-size", "32"],
    "anli": ["--epochs", "1", "--batch-size", "32"],
    "hellaswag": ["--epochs", "1", "--batch-size", "32"],
    "piqa": ["--epochs", "1", "--batch-size", "32"],
    "copa": ["--epochs", "1", "--batch-size", "32"],
    "winogrande": ["--epochs", "1", "--batch-size", "32"],
    "alpha_nli": ["--epochs", "1", "--batch-size", "32"],
    "squad_v2": ["--epochs", "1", "--batch-size", "32"],
    "wmt14_en_de": ["--epochs", "1", "--batch-size", "32"],
    "wmt14_de_en": ["--epochs", "1", "--batch-size", "32"],
    "xsum": ["--epochs", "1", "--batch-size", "32"],
    "lm": ["--epochs", "1", "--batch-size", "32", "--block-size", "512"],
    "gsm8k": ["--epochs", "1", "--batch-size", "32"],
    "multirc": ["--epochs", "1", "--batch-size", "32"],
    "record": ["--epochs", "1", "--batch-size", "32"],
    "hotpotqa": ["--epochs", "1", "--batch-size", "32"],
}
NLP_SMOKE_SCENARIOS: List[Dict[str, Any]] = [
    {
        "key": info.key,
        "label": info.label,
        "script": info.script,
        "group": "Part1" if info.script == "part1" else "Part2",
        "prefill": NLP_SMOKE_DEFAULTS.get(info.key, ["--epochs", "1", "--batch-size", "32"]),
    }
    for info in (PART1_TASKS + PART2_TASKS)
]

ROOT = Path(__file__).resolve().parents[2]
HPS_NLP_PATH = ROOT / "configs" / "hps_nlp.json"
HP_TASK_ALIASES = {
    "lm": "lm_wikitext",
}


def _load_hp_presets() -> Dict[str, Dict[str, Dict[str, float]]]:
    if not HPS_NLP_PATH.exists():
        return {}
    try:
        data = json.loads(HPS_NLP_PATH.read_text(encoding="utf-8"))
        return data
    except Exception:
        return {}


HPS_CACHE = _load_hp_presets()


def _hp_for_task(optimizer: str, task_key: str) -> Dict[str, float] | None:
    opt_block = HPS_CACHE.get(optimizer.lower())
    if not opt_block:
        return None
    alias = HP_TASK_ALIASES.get(task_key, task_key)
    hp = opt_block.get(alias)
    if hp:
        return hp
    return opt_block.get(task_key.replace("suite:", ""))


def _maybe_apply_hp(args: List[str], optimizer: str, task_key: str):
    hp = _hp_for_task(optimizer, task_key)
    if not hp:
        return
    if "--lr" not in args and "lr" in hp:
        args += ["--lr", str(hp["lr"])]
    wd = hp.get("weight_decay") or hp.get("wd")
    if wd is not None and "--wd" not in args and "--weight-decay" not in args:
        args += ["--wd", str(wd)]


def _prepare_env_and_scripts(data_root: Path) -> Tuple[Dict[str, str], Dict[str, Path]]:
    hf_cache = ensure_dir(Path(data_root) / "hf-cache")
    env = os.environ.copy()
    env.setdefault("HF_HOME", str(hf_cache))
    env.setdefault("HF_DATASETS_CACHE", str(hf_cache))
    env.pop("TRANSFORMERS_CACHE", None)
    root = Path(__file__).resolve().parents[3]
    existing_pp = env.get("PYTHONPATH", "")
    paths = [str(root)] + ([existing_pp] if existing_pp else [])
    env["PYTHONPATH"] = os.pathsep.join([p for p in paths if p])
    scripts_dir = Path(__file__).resolve().parent
    scripts = {"part1": scripts_dir / "bench_part1.py", "part2": scripts_dir / "bench_part2.py"}
    return env, scripts


def _dataset_ready(data_root: Path, task_key: str) -> bool:
    # Suites: ready when all constituent tasks are ready
    suites = {
        "suite:easy": ["sst2", "mrpc", "hellaswag", "piqa", "ag_news", "lm"],
        "suite:medium": ["boolq", "rte", "cb", "copa", "anli", "squad_v2", "xsum"],
        "suite:hard": ["wmt14_en_de", "gsm8k"],
    }
    if task_key in suites:
        return all(_dataset_ready(data_root, t) for t in suites[task_key])
    ds_map = {
        "lm": "wikitext2",  # task key -> dataset key in registry
    }
    ds_name = ds_map.get(task_key, task_key)
    try:
        return has_dataset(ds_name, data_root)
    except KeyError:
        # fallback: check presence under hf-cache
        path = dataset_root(data_root, ds_name)
        return path.exists() and any(path.iterdir())


def run_nlp_menu(data_root: Path):
    env, scripts = _prepare_env_and_scripts(data_root)
    """
    Interactive menu wrapper that shells out to bench_part1/bench_part2 with sane defaults.
    """

    while True:
        clear_screen()
        print_header("NLP Benchmarks")
        print(" Hugging Face datasets/weights는 data/hf-cache 경로에 저장됩니다.")
        print(" 대형 태스크는 시간이 오래 걸릴 수 있습니다.")
        print("--------------------------------------------------")

        options: List[str] = []
        idx = 1
        for group, keys in TASK_GROUPS:
            print(f" {group}")
            for key in keys:
                info = TASK_INFO[key]
                print(f"   [{idx}] {info.label}")
                options.append(key)
                idx += 1
        print("   [0] 돌아가기")
        choice = input(" 선택 >> ").strip()
        if choice == "0":
            return
        try:
            key = options[int(choice) - 1]
        except Exception:
            print("잘못된 선택입니다.")
            input("엔터를 눌러 계속...")
            continue

        info = TASK_INFO[key]
        _run_task_via_cli(info, scripts[info.script], env)


def run_nlp_suite(data_root: Path, suite_key: str):
    env, scripts = _prepare_env_and_scripts(data_root)
    info = TASK_INFO[suite_key]
    _run_task_via_cli(info, scripts[info.script], env)


def run_nlp_smoke_menu(data_root: Path):
    env, scripts = _prepare_env_and_scripts(data_root)
    filter_group = "all"
    while True:
        clear_screen()
        header = f"NLP Smoke Tests (Filter: {filter_group.upper()})"
        print_header(header)
        print(" [F1] 전체 보기")
        print(" [F2] Part 1 전용")
        print(" [F3] Part 2 전용")
        print(" [A] 전체 스모크 일괄 실행 (현재 필터 대상만)")
        print("--------------------------------------------------")
        scenarios = _filtered_smoke_scenarios(filter_group)
        for idx, sc in enumerate(scenarios, start=1):
            info = TASK_INFO[sc["key"]]
            bsz = sc["prefill"][sc["prefill"].index("--batch-size") + 1] if "--batch-size" in sc["prefill"] else "32"
            ready = "✅" if _dataset_ready(data_root, sc["key"]) else "❌"
            print(f" [{idx}] {info.label:<35} | {sc['group']} | batch {bsz} | {ready}")
        print(" [0] 돌아가기")
        choice = input(" 선택 >> ").strip().lower()
        if choice == "0":
            return
        if choice in {"f1", "f", "all"}:
            filter_group = "all"; continue
        if choice in {"f2", "p1", "part1"}:
            filter_group = "part1"; continue
        if choice in {"f3", "p2", "part2"}:
            filter_group = "part2"; continue
        if choice == "a":
            for sc in scenarios:
                info = TASK_INFO[sc["key"]]
                if not _dataset_ready(data_root, sc["key"]):
                    print(f"[SKIP] {info.label} (데이터셋 미준비)")
                    continue
                print(f"[SMOKE-ALL] {info.label}")
                _run_task_via_cli(info, scripts[info.script], env, prefill_args=sc["prefill"], allow_custom=False)
            input("엔터를 눌러 계속...")
            continue
        try:
            scenario = scenarios[int(choice) - 1]
        except Exception:
            print("잘못된 선택입니다.")
            input("엔터를 눌러 계속...")
            continue
        if not _dataset_ready(data_root, scenario["key"]):
            print("[WARN] 데이터셋이 준비되지 않았습니다. 데이터셋 관리에서 먼저 다운로드하세요.")
            input("엔터를 눌러 계속...")
            continue
        info = TASK_INFO[scenario["key"]]
        print(f"[SMOKE] {info.label}")
        _run_task_via_cli(info, scripts[info.script], env, prefill_args=scenario["prefill"], allow_custom=False)


def _filtered_smoke_scenarios(group: str) -> List[Dict[str, Any]]:
    if group == "part1":
        return [sc for sc in NLP_SMOKE_SCENARIOS if sc["group"].lower() == "part1"]
    if group == "part2":
        return [sc for sc in NLP_SMOKE_SCENARIOS if sc["group"].lower() == "part2"]
    return NLP_SMOKE_SCENARIOS


def _run_task_via_cli(
    info: NLPTaskInfo,
    script_path: Path,
    env: Dict[str, str],
    prefill_args: List[str] | None = None,
    allow_custom: bool = True,
):
    clear_screen()
    print_header(f"NLP Task: {info.label}")
    print(info.description)
    print(f" 기본 모델: {info.default_model or 'script 기본값'}")
    print(f" 스크립트: {script_path.name}")
    default_cache = Path(__file__).resolve().parents[3] / "data" / "hf-cache"
    cache_dir = Path(env.get("HF_DATASETS_CACHE", default_cache))
    data_root = cache_dir.parent  # expected: .../data
    ready = _dataset_ready(data_root, info.key)
    if info.heavy_warning and not ready:
        print(" ⚠️ 해당 태스크는 대용량 데이터(Hugging Face) 다운로드가 필요합니다.")
        ans = input(" 계속 진행하시겠습니까? [y/N] >> ").strip().lower()
        if ans not in ("y", "yes"):
            print("실행을 취소했습니다.")
            input("엔터를 눌러 계속...")
            return

    base_args = ["--task", info.key]

    if not allow_custom:
        args = base_args + ["--optimizer", "adamw"]
        if info.default_amp and info.default_amp != "none":
            args += ["--amp", info.default_amp]
        args += prefill_args or []
        _execute_cli(script_path, env, args)
        return

    print("--------------------------------------------------")
    print(" [1] 빠른 실행 (AdamW, 기본 epoch/batch)")
    print(" [2] 고급 설정 (optimizer/model/lr 등 세부 조정)")
    print(" [0] 취소/뒤로가기")
    mode = input(" 선택 >> ").strip() or "1"
    if mode == "0":
        return

    optimizer = "adamw"
    model = info.default_model or ""
    extra_args: List[str] = []
    amp = info.default_amp
    epochs = ""
    batch = ""
    lr = ""
    max_len = ""
    block_size = ""

    if mode != "1":
        optimizer = _prompt_choice("Optimizer", SUPPORTED_OPTIMIZERS, "adamw")
        model = input(f" 모델 이름 [{model or '스크립트 기본'}] >> ").strip() or model
        amp = _prompt_choice("AMP (혼합정밀)", ["none", "fp16", "bf16"], amp or "none")
        epochs = input(" Epoch 수 (빈칸=기본) >> ").strip()
        batch = input(" 배치 사이즈 (빈칸=기본) >> ").strip()
        lr = input(" 학습률 lr (빈칸=기본) >> ").strip()
        if info.key not in ("lm",) and info.script == "part1":
            max_len = input(" max_length (빈칸=기본) >> ").strip()
        if info.key == "lm":
            block_size = input(" block_size (빈칸=기본=512) >> ").strip()
        extras = input(" 추가 CLI 인자 (--warmup_ratio 0.05 등) >> ").strip()
        if extras:
            try:
                extra_args.extend(shlex.split(extras))
            except ValueError:
                print("추가 인자 파싱에 실패했습니다. 그대로 전달하지 않습니다.")
    else:
        print(" 빠른 실행: optimizer=adamw, 기본 하이퍼파라미터 사용.")

    args = base_args + ["--optimizer", optimizer]
    if model:
        args += ["--model-name", model]
    if amp and amp != "none":
        args += ["--amp", amp]
    if epochs:
        args += ["--epochs", epochs]
    if batch:
        args += ["--batch-size", batch]
    if lr:
        args += ["--lr", lr]
    if max_len:
        args += ["--max-length", max_len]
    if block_size:
        args += ["--block-size", block_size]
    args.extend(extra_args)
    if prefill_args:
        args.extend(prefill_args)
    _maybe_apply_hp(args, optimizer, info.key)
    _execute_cli(script_path, env, args)


def _execute_cli(script_path: Path, env: Dict[str, str], args: List[str]):
    print("\n실행 명령:")
    print(f"  python {script_path.name} {' '.join(args)}")
    print("--------------------------------------------------")
    print(" [TIP] 학습 중 터미널에 'exit' 입력하면 안전하게 중단되며 체크포인트가 저장됩니다.")
    input("엔터를 누르면 실행합니다...")

    cmd = [sys.executable, str(script_path), *args]
    try:
        subprocess.run(cmd, env=env, check=False)
    except KeyboardInterrupt:
        print("\n[중단] 사용자가 실행을 취소했습니다.")
    input("엔터를 눌러 메뉴로...")


def _prompt_choice(label: str, options: List[str], default: str) -> str:
    print(f"{label}:")
    for idx, opt in enumerate(options, start=1):
        flag = "(기본)" if opt == default else ""
        print(f"  [{idx}] {opt} {flag}")
    raw = input(" 번호 선택 >> ").strip()
    if not raw:
        return default
    try:
        idx = int(raw) - 1
        if 0 <= idx < len(options):
            return options[idx]
    except ValueError:
        pass
    print("잘못된 입력이라 기본값을 사용합니다.")
    return default
