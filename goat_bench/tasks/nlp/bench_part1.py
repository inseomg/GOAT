from __future__ import annotations

# bench_nlp_part1.py — GOAT-style unified NLP benchmark (extended, patched)
# =========================================================
# 환경/로그 설정 (imports보다 앞)
import os as _os, warnings as _warnings, multiprocessing as _mp
_os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
_os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
_os.environ.setdefault("TRANSFORMERS_NO_ADVISORY_WARNINGS", "1")
_os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
_os.environ.setdefault("TRANSFORMERS_NO_TF", "1")    # transformers가 TF 백엔드 로딩 금지
_os.environ.setdefault("TRANSFORMERS_NO_FLAX", "1")  # Flax 백엔드도 금지

# ---- TensorFlow silent stub v3 (for transformers + datasets) ----
import os as _os, sys as _sys, types as _types
import importlib.machinery as _mach

# 완전 비활성화 기본값(환경변수로 켜고 끌 수 있음)
if _os.environ.get("DISABLE_TF_IMPORT", "1") == "1":
    def _mkpkg(name: str):
        m = _types.ModuleType(name)
        m.__file__ = "<stub>"
        m.__path__ = []  # 패키지처럼 보이게
        m.__spec__ = _mach.ModuleSpec(name, loader=None, is_package=True)
        return m

    # 메인 패키지
    tf = _sys.modules.get("tensorflow") or _mkpkg("tensorflow")
    # datasets가 타입체크에 쓰는 최소 속성만 제공
    class _TF_Tensor: ...
    class _TF_RaggedTensor: ...
    class _TF_SparseTensor: ...
    tf.Tensor = _TF_Tensor
    tf.RaggedTensor = _TF_RaggedTensor
    tf.SparseTensor = _TF_SparseTensor
    tf.__version__ = "0.0.0-stub"
    _sys.modules["tensorflow"] = tf

    # 흔히 참조되는 서브패키지들 스텁
    for sub in (
        "tensorflow.keras", "tensorflow.python", "tensorflow.compat", "tensorflow._api",
        "tensorflow.experimental", "tensorflow.experimental.numpy", "tensorflow._api.v2"
    ):
        if sub not in _sys.modules:
            _sys.modules[sub] = _mkpkg(sub)

    # 일부 라이브러리가 확인하는 compat.v1 존재시켜주기
    if "tensorflow.compat.v1" not in _sys.modules:
        _sys.modules["tensorflow.compat.v1"] = _mkpkg("tensorflow.compat.v1")

# (선택) datasets가 TF를 아예 무시하도록 힌트 — 있어도 문제없음
_os.environ.setdefault("HF_DATASETS_DISABLE_TF", "1")
# ---------------------------------------------------------------

# absl 로그 레벨 (TF 내부 로그 소거)
try:
    import absl.logging as _absl_log
    _absl_log.set_verbosity(_absl_log.ERROR)
except Exception:
    pass
# tokenizers 병렬 비활성(포크 충돌/경고 회피)
try:
    import tokenizers as _toks
    _toks.util.disable_parallelism()
except Exception:
    pass
# DataLoader fork→spawn (Colab 등에서 안전)
try:
    if _mp.get_start_method(allow_none=True) != "spawn":
        _mp.set_start_method("spawn", force=True)
except Exception:
    pass
# =========================================================

# ---- top-level datasets/collates for multiprocessing workers ----
import torch
from torch.utils.data import Dataset

class SquadValDataset(Dataset):
    """SQuAD v2 검증용: features + id->answers/context 매핑을 유지"""
    def __init__(self, features, id_to_answers, id_to_context):
        self.features = features
        self.id_to_answers = id_to_answers
        self.id_to_context = id_to_context
    def __len__(self): return len(self.features)
    def __getitem__(self, i):
        f = self.features[i]  # HF Dataset row (dict-like)
        ex_id = f["example_id"]
        return {
            "input_ids": torch.tensor(f["input_ids"]),
            "attention_mask": torch.tensor(f["attention_mask"]),
            "offset_mapping": f["offset_mapping"],   # list[(s,e)|None]
            "example_id": ex_id,
            "answers": self.id_to_answers.get(ex_id, []),
            "context": self.id_to_context.get(ex_id, ""),
        }

def squad_val_collate(batch):
    return {
        "input_ids": torch.stack([b["input_ids"] for b in batch]),
        "attention_mask": torch.stack([b["attention_mask"] for b in batch]),
        "offset_mapping": [b["offset_mapping"] for b in batch],
        "example_id": [b["example_id"] for b in batch],
        "answers": [b["answers"] for b in batch],
        "contexts": [b["context"] for b in batch],
    }

class GSM8KEvalDS(Dataset):
    """GSM8K 검증용 (spawn-safe)"""
    def __init__(self, hf_dataset):
        self.hf = hf_dataset
    def __len__(self): return len(self.hf)
    def __getitem__(self, i):
        item = {
            "input_ids": torch.tensor(self.hf[i]["input_ids"]),
            "attention_mask": torch.tensor(self.hf[i]["attention_mask"]),
        }
        item["answers"] = self.hf[i]["answers"]
        return item


# === GSM8K helpers (top-level; not inside a function) ===
class GSM8KTrainDS(Dataset):
    def __init__(self, hf_table): self.hf = hf_table
    def __len__(self): return len(self.hf)
    def __getitem__(self, i):
        r = self.hf[i]
        return {
            "input_ids":      torch.tensor(r["input_ids"],      dtype=torch.long),
            "attention_mask": torch.tensor(r["attention_mask"], dtype=torch.long),
            "labels":         torch.tensor(r["labels"],         dtype=torch.long),
        }

class GSM8KValDS(Dataset):
    def __init__(self, hf_table): self.hf = hf_table
    def __len__(self): return len(self.hf)
    def __getitem__(self, i):
        r = self.hf[i]
        return {
            "input_ids":      torch.tensor(r["input_ids"],      dtype=torch.long),
            "attention_mask": torch.tensor(r["attention_mask"], dtype=torch.long),
            "labels":         torch.tensor(r["labels"],         dtype=torch.long),
            "answers":        r["gold"],
        }

def gsm8k_train_collate(b):
    return {
        "input_ids":      torch.stack([x["input_ids"] for x in b]),
        "attention_mask": torch.stack([x["attention_mask"] for x in b]),
        "labels":         torch.stack([x["labels"] for x in b]),
    }

def gsm8k_val_collate(b):
    return {
        "input_ids":      torch.stack([x["input_ids"] for x in b]),
        "attention_mask": torch.stack([x["attention_mask"] for x in b]),
        "labels":         torch.stack([x["labels"] for x in b]),
        "answers":        [x["answers"] for x in b],
    }




WMT14_VALID = {"cs-en","de-en","fr-en","hi-en","ru-en"}
WMT16_VALID = {"cs-en","de-en","fi-en","ro-en","ru-en","tr-en"}

def _load_wmt_with_direction(direction: str):
    """
    Returns: (dataset_dict, src_lang, tgt_lang, used_name)
    - 입력 방향(en-de/de-en 등)을 그대로 src/tgt로 **되돌려줍니다**.
    - 데이터셋은 사용 가능한 설정으로 로드하되, 샘플 내용을 스왑하지 않습니다.
      (전처리에서 칼럼 이름으로 바로 뽑아 쓰면 됨)
    """
    src, tgt = direction.split("-")

    if direction in WMT14_VALID:
        ds = _hf_load("wmt14", direction)
        used = ("wmt14", direction)
    elif f"{tgt}-{src}" in WMT14_VALID:
        # 반대 설정으로 로드하지만, src/tgt는 "요청 방향"을 그대로 유지
        ds = _hf_load("wmt14", f"{tgt}-{src}")
        used = ("wmt14", f"{tgt}-{src}")
    elif f"{tgt}-{src}" in WMT16_VALID:
        ds = _hf_load("wmt16", f"{tgt}-{src}")
        used = ("wmt16", f"{tgt}-{src}")
    else:
        raise ValueError(
            f"Unsupported WMT direction '{direction}'. "
            f"Available (wmt14): {sorted(WMT14_VALID)}; (wmt16): {sorted(WMT16_VALID)}"
        )
    # 핵심: 여기서 **스왑(map)** 하지 않음
    return ds, src, tgt, used

# 필수/옵션 라이브러리
import argparse, time, json, math, os, random, statistics, re, shutil
from pathlib import Path
import sys
from typing import Dict, Any, List, Tuple, Optional
import sys

import torch, torch.nn as nn, torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import DefaultDataCollator, DataCollatorWithPadding, DataCollatorForSeq2Seq

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
DATA_DIR = ROOT / "data"
HF_CACHE = DATA_DIR / "hf-cache"
_os.environ.setdefault("HF_HOME", str(HF_CACHE))
_os.environ.setdefault("HF_DATASETS_CACHE", str(HF_CACHE))
_os.environ.pop("TRANSFORMERS_CACHE", None)
HF_CACHE.mkdir(parents=True, exist_ok=True)

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
DATA_DIR = ROOT / "data"
HF_CACHE = DATA_DIR / "hf-cache"
os.environ.setdefault("HF_HOME", str(HF_CACHE))
os.environ.setdefault("HF_DATASETS_CACHE", str(HF_CACHE))
os.environ.pop("TRANSFORMERS_CACHE", None)
# datasets
try:
    from datasets import load_dataset
except Exception as e:
    raise RuntimeError("`pip install datasets` 필요") from e
import inspect

# transformers 모델
try:
    from transformers import (
        AutoTokenizer, AutoModelForSequenceClassification,
        AutoModelForMultipleChoice,
        AutoModelForQuestionAnswering,
        AutoModelForSeq2SeqLM, AutoModelForCausalLM
    )
except Exception as e:
    raise RuntimeError("`pip install transformers` 필요") from e

# evaluate / sacrebleu (둘 중 없으면 내부 fallback 사용)
_evaluate = None
_sacrebleu = None
try:
    import evaluate as _evaluate
except Exception:
    pass
try:
    import sacrebleu as _sacrebleu
except Exception:
    pass


def _purge_cache_entry(names: list[str]):
    """Remove broken HF cache entries (legacy scripts) for the given dataset names."""
    for nm in names:
        ds_root = HF_CACHE / nm
        if ds_root.exists():
            shutil.rmtree(ds_root, ignore_errors=True)
    hub = HF_CACHE / "hub"
    if hub.exists():
        for nm in names:
            pat = nm.replace("/", "--")
            for sub in hub.glob(f"datasets--*{pat}*"):
                shutil.rmtree(sub, ignore_errors=True)


_TRUST_REMOTE_CODE_SUPPORTED: bool | None = None


def _supports_trust_remote_code() -> bool:
    global _TRUST_REMOTE_CODE_SUPPORTED
    if _TRUST_REMOTE_CODE_SUPPORTED is None:
        try:
            sig = inspect.signature(load_dataset)
            _TRUST_REMOTE_CODE_SUPPORTED = "trust_remote_code" in sig.parameters
        except Exception:
            _TRUST_REMOTE_CODE_SUPPORTED = False
    return bool(_TRUST_REMOTE_CODE_SUPPORTED)


def _load_dataset_compat(name: str, subset: str | None, *, allow_remote: bool):
    """
    Load a dataset while handling:
      - new datasets versions that reject trust_remote_code
      - remote/community datasets that still require trust_remote_code=True
      - numpy/scipy ABI mismatches (gives actionable hint)
    """
    kwargs = {"cache_dir": str(HF_CACHE)}
    supports_trc = _supports_trust_remote_code()

    def _call(trust_remote: bool):
        opts = dict(kwargs)
        if trust_remote and supports_trc:
            opts["trust_remote_code"] = True
        return load_dataset(name, subset, **opts)

    try:
        return _call(allow_remote)
    except TypeError as exc:
        # Older/newer datasets may not accept the argument
        if "trust_remote_code" in str(exc):
            return _call(False)
        raise
    except ValueError as exc:
        msg = str(exc)
        # Newer datasets: "trust_remote_code is not supported anymore"
        if "trust_remote_code is not supported anymore" in msg:
            return _call(False)
        # Community datasets that insist on trust_remote_code=True
        if "trust_remote_code=True" in msg and supports_trc and not allow_remote:
            return _call(True)
        raise
    except ImportError as exc:
        # numpy 2.3.x + scipy wheels can break with `_center` import errors
        if "_center" in str(exc) and "numpy" in str(exc):
            raise RuntimeError(
                "HF 데이터셋 로딩 중 numpy/Scipy ABI 오류가 발생했습니다. "
                "requirements.txt 버전에 맞춰 `pip install 'numpy>=1.26,<2.2' 'scipy>=1.10,<1.13' --upgrade --force-reinstall` 를 실행하세요."
            ) from exc
        raise


def _hf_load(name: str, subset: str | None = None):
    allow_remote = "/" in name or name in {"abductive_nli", "abductive-nli", "allenai/art", "Rowan/hellaswag"}
    try:
        return _load_dataset_compat(name, subset, allow_remote=allow_remote)
    except Exception as exc:
        # Legacy script error: purge cache and try once more
        msg = str(exc)
        if "xsum.py" in msg or "script" in msg:
            patterns = [name, name.replace("/", "--")]
            _purge_cache_entry(patterns)
            return _load_dataset_compat(name, subset, allow_remote=allow_remote)
        raise

from datetime import datetime
from goat_bench.optimizers.builder import build_optimizer, build_optimizer_generic
from goat_bench.utils.helpers import exit_requested
from goat_bench.utils.checkpointing import save_checkpoint

# scipy (선택) — STS-B spearman 정확판정
_scipy_spearmanr = None
try:
    from scipy.stats import spearmanr as _scipy_spearmanr
except Exception:
    _scipy_spearmanr = None

# 외부 옵티마이저 (선택)
LionOpt = None
SOAPOpt = None
try:
    from lion_pytorch import Lion as LionOpt
except Exception:
    pass
try:
    import pytorch_optimizer as _pyo
    SOAPOpt = getattr(_pyo, "SOAP", None)
except Exception:
    pass

# RICO (선택)
try:
    from rico import RICO, rico_layerwise_groups
except Exception:
    RICO, rico_layerwise_groups = None, None

# ===== 공통 유틸 =====
_CKPT_ROOT = Path(__file__).resolve().parents[2] / "results" / "checkpoints"


def _save_nlp_checkpoint(
    tag: str,
    dataset: str,
    args,
    model,
    optimizer,
    scheduler,
    epoch: int,
    best_metric: float,
    extra: Optional[Dict[str, Any]] = None,
):
    _CKPT_ROOT.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_ds = dataset.replace("/", "_")
    path = _CKPT_ROOT / f"{tag}_{safe_ds}_epoch{epoch}_{stamp}.pt"
    state: Dict[str, Any] = {
        "tag": tag,
        "dataset": dataset,
        "epoch": epoch,
        "best_metric": best_metric,
        "args": vars(args),
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "scheduler_state": scheduler.state_dict() if scheduler is not None else None,
    }
    if extra:
        state.update(extra)
    save_checkpoint(path, state)
    return path


def _attach_interrupt_meta(summary: Dict[str, Any], checkpoint_path: Optional[Path]):
    summary["interrupted"] = checkpoint_path is not None
    if checkpoint_path:
        summary["checkpoint"] = str(checkpoint_path)
    return summary
def set_seed(seed:int):
    random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)

def percentile(xs: List[float], q: float) -> float:
    if not xs: return float("nan")
    ys = sorted(xs); k = (len(ys)-1)*(q/100.0)
    f, c = math.floor(k), math.ceil(k)
    return ys[int(k)] if f==c else ys[f] + (ys[c]-ys[f])*(k-f)

def write_csv(path: Optional[Path], row: Dict[str,Any], header_keys: Optional[List[str]] = None):
    if path is None: return
    import csv
    path.parent.mkdir(parents=True, exist_ok=True)
    new = not path.exists()
    keys = header_keys or list(row.keys())
    with path.open("a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        if new: w.writeheader()
        w.writerow({k: row.get(k) for k in keys})

def human_mb(x:int) -> float: return round(x/(1024*1024), 2)

def simple_rouge_l(preds: List[str], refs: List[str]) -> float:
    # 매우 단순한 ROUGE-L fallback (공식 점수와 차이 가능)
    def lcs(a,b):
        dp = [[0]*(len(b)+1) for _ in range(len(a)+1)]
        for i in range(1,len(a)+1):
            for j in range(1,len(b)+1):
                dp[i][j] = dp[i-1][j-1]+1 if a[i-1]==b[j-1] else max(dp[i-1][j], dp[i][j-1])
        return dp[-1][-1]
    scores=[]
    for p,r in zip(preds,refs):
        ap, ar = p.split(), r.split()
        if not ar: continue
        scores.append(100.0*lcs(ap,ar)/len(ar))
    return sum(scores)/max(len(scores),1)

def norm_answer(s:str) -> str:
    s = s.lower().strip()
    s = re.sub(r"[^0-9a-z가-힣\.\- ]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s

def extract_last_number(s:str) -> Optional[str]:
    m = re.findall(r"[-+]?\d*\.?\d+(?:/\d+)?", s.replace(",", ""))
    return m[-1] if m else None

def _spearman_pct(x: List[float], y: List[float]) -> float:
    # Spearman ρ × 100 (scipy 없으면 tie-aware 수동 구현)
    import numpy as _np
    a = _np.array(x, dtype=float); b = _np.array(y, dtype=float)
    if a.size==0 or b.size==0 or a.size!=b.size: return 0.0
    if _scipy_spearmanr is not None:
        rho, _ = _scipy_spearmanr(a, b)
        if rho is None or math.isnan(rho): return 0.0
        return float(rho * 100.0)
    def _avg_rank(v):
        order = v.argsort(kind="mergesort")
        ranks = _np.empty_like(order, dtype=float)
        ranks[order] = _np.arange(len(v), dtype=float)
        uniq, inv, cnt = _np.unique(v, return_inverse=True, return_counts=True)
        cum = _np.cumsum(cnt); starts = cum - cnt
        avg = (starts + cum - 1) / 2.0
        return avg[inv]
    ra, rb = _avg_rank(a), _avg_rank(b)
    ra = (ra - ra.mean()) / (ra.std() + 1e-12)
    rb = (rb - rb.mean()) / (rb.std() + 1e-12)
    return float((ra * rb).mean() * 100.0)

# ===== 옵티마이저 빌더 =====
def split_decay_params(model, wd: float):
    decay, nodecay = [], []
    no_decay_keys = ("bias","LayerNorm.weight","layer_norm.weight","ln.weight","ln_1.weight","ln_2.weight","norm.weight","bn.weight","bn.bias")
    for n,p in model.named_parameters():
        if not p.requires_grad: continue
        if len(p.shape)==1 or any(k in n for k in no_decay_keys):
            nodecay.append(p)
        else:
            decay.append(p)
    return [{"params":decay,"weight_decay":wd},{"params":nodecay,"weight_decay":0.0}]

def make_rico_opt(model, lr, wd, args):
    assert RICO is not None, "rico.py 필요"
    try:
        pg = rico_layerwise_groups(model, weight_decay=wd)
    except Exception:
        pg = split_decay_params(model, wd)
    kw = dict(
        bk_beta_target=args.rico_bk_beta,
        k_cap=args.rico_k_cap,
        g_rms_floor=args.rico_g_floor,
        sync_every=args.rico_sync_every
    )
    ft = getattr(args, "rico_ft", True)  # HF 사전학습 모델: FT 모드 권장
    try:
        return RICO(pg, lr=lr, ft_mode=ft, weight_decay=wd, wd_mode="decoupled", **kw)
    except TypeError:
        return RICO(pg, lr=lr, ft_mode=ft, weight_decay=wd, wd_mode="decoupled")

def build_optimizer(model, name:str, lr:float, wd:float, args):
    name = name.lower()
    if name=="rico":
        if RICO is None: raise RuntimeError("RICO 사용을 위해 rico.py 필요")
        return make_rico_opt(model, lr, wd, args)
    pg = split_decay_params(model, wd)
    if name=="adamw":
        return optim.AdamW(pg, lr=lr)
    if name=="lion":
        if LionOpt is None: raise RuntimeError("`pip install lion-pytorch` 필요")
        return LionOpt(pg, lr=lr, betas=(args.lion_beta1, args.lion_beta2), weight_decay=wd)
    if name=="soap":
        if SOAPOpt is None: raise RuntimeError("`pip install pytorch-optimizer` 및 SOAP 버전 확인")
        extra = {}
        if args.soap_args:
            try: extra = json.loads(args.soap_args)
            except Exception as e: print(f"[warn] --soap_args 파싱 실패: {e}")
        return SOAPOpt(pg, lr=lr, weight_decay=wd, **extra)
    raise ValueError(f"unknown optimizer {name}")

# === MCQ: collator & field unify ===
class DataCollatorForMultipleChoice:
    """각 문항 선택지 길이를 패드해 [B, C, L] 텐서로 묶는다."""
    def __init__(self, tokenizer, pad_to_multiple_of=None):
        self.tokenizer = tokenizer
        self.pad_to_multiple_of = pad_to_multiple_of
    def __call__(self, features):
        import torch
        labels = torch.tensor([f["labels"] for f in features], dtype=torch.long)
        bsz = len(features); num_choices = len(features[0]["input_ids"])
        flat_input_ids, flat_attn = [], []
        for f in features:
            for i in range(num_choices):
                flat_input_ids.append(f["input_ids"][i])
                flat_attn.append(f["attention_mask"][i])
        batch = self.tokenizer.pad(
            {"input_ids": flat_input_ids, "attention_mask": flat_attn},
            return_tensors="pt",
            pad_to_multiple_of=self.pad_to_multiple_of,
        )
        batch["input_ids"] = batch["input_ids"].view(bsz, num_choices, -1)
        batch["attention_mask"] = batch["attention_mask"].view(bsz, num_choices, -1)
        batch["labels"] = labels
        return batch

def _mcq_unify_fields(dataset_name: str, ex: dict):
    """(context, choices(list[str]), label, fill_blank) 형태로 통일."""
    name = dataset_name.lower()

    if name == "hellaswag":
        ctx_a = ex.get("ctx_a", ""); ctx_b = ex.get("ctx_b", "")
        ctx = (ctx_a + " " + ctx_b).strip()
        endings = ex.get("endings") or ex.get("endings_randomized") or ex.get("choices")
        label = int(ex["label"])
        return ctx, endings, label, False

    if name == "piqa":
        ctx = ex["goal"]; endings = [ex["sol1"], ex["sol2"]]
        label = int(ex["label"])
        return ctx, endings, label, False

    if name == "copa":
        premise = ex["premise"]; question = ex["question"]
        conn = " because " if str(question).strip() == "cause" else " so "
        ctx = (premise + conn).strip()
        endings = [ex["choice1"], ex["choice2"]]
        label = int(ex["label"])
        return ctx, endings, label, False

    if name == "winogrande":
        sentence = ex["sentence"]; endings = [ex["option1"], ex["option2"]]
        ans = ex.get("label", ex.get("answer"))
        label = 0
        if ans is not None:
            try:
                iv = int(str(ans).strip())
                label = iv - 1 if iv in (1,2) else max(0, iv)
            except Exception:
                label = 0
        return sentence, endings, label, True  # fill_blank=True

    if name in ("alpha_nli", "abductive_nli", "abductive-nli"):
        obs1, obs2 = ex.get("obs1", ""), ex.get("obs2", "")
        ctx = (obs1 + " " + obs2).strip()
        endings = [ex["hypothesis1"], ex["hypothesis2"]]
        label = int(ex["label"])
        return ctx, endings, label, False

    raise ValueError(f"Unsupported MCQ dataset: {dataset_name}")

# ===== 공통 Trainer 베이스 (TTT/throughput/step 통계) =====
class TrainerBase:
    def __init__(self, model, optimizer, scheduler, device, amp="none",
                 higher_is_better=True, ttt_target=None, clip_grad_norm=None):
        self.model, self.opt, self.sch = model, optimizer, scheduler
        self.device = device
        self.amp = amp if torch.cuda.is_available() else "none"
        self.autocast_dtype = torch.bfloat16 if self.amp=="bf16" else torch.float16
        self.scaler = torch.amp.GradScaler("cuda", enabled=(self.amp=="fp16"))
        self.step_times: List[float] = []
        self.losses_epoch: List[float] = []
        self.seen_tokens_epoch = 0
        self.higher_is_better = higher_is_better
        self.ttt_target = ttt_target
        self.ttt_sec = None
        self.clip_grad_norm = clip_grad_norm
        self.best_metric = (-1e18 if higher_is_better else 1e18)
        self.wall0 = time.perf_counter()

    def _mark_seen_tokens(self, n:int): self.seen_tokens_epoch += int(n)
    def _check_stop(self):
        if exit_requested():
            raise KeyboardInterrupt
    def _maybe_ttt(self, metric: float):
        if self.ttt_target is None or self.ttt_sec is not None: return
        ok = (metric >= self.ttt_target) if self.higher_is_better else (metric <= self.ttt_target)
        if ok: self.ttt_sec = time.perf_counter() - self.wall0
    def _update_best(self, metric: float):
        if self.higher_is_better: self.best_metric = max(self.best_metric, metric)
        else: self.best_metric = min(self.best_metric, metric)
    def _throughput(self) -> float:
        total = sum(self.step_times) if self.step_times else 0.0
        return float(self.seen_tokens_epoch / max(total, 1e-9))

# ===== 분류(글루/슈글루) =====
def load_cls_dataset(name:str):
    n = name.lower()
    if n=="sst2":              return _hf_load("glue","sst2"), ("sentence",None), 2, "acc"
    if n=="mrpc":              return _hf_load("glue","mrpc"), ("sentence1","sentence2"), 2, "f1_acc"
    if n=="stsb":              return _hf_load("glue","stsb"), ("sentence1","sentence2"), 1, "pearson_spearman"
    if n=="qqp":               return _hf_load("glue","qqp"), ("question1","question2"), 2, "f1_acc"
    if n=="ag_news":           return _hf_load("ag_news"), ("text",None), 4, "acc"
    if n=="boolq":             return _hf_load("super_glue","boolq"), ("question","passage"), 2, "acc"
    if n=="rte":               return _hf_load("super_glue","rte"), ("premise","hypothesis"), 2, "acc"
    if n=="cb":                return _hf_load("super_glue","cb"), ("premise","hypothesis"), 3, "macro_f1"
    if n=="anli":              return _hf_load("anli"), ("premise","hypothesis"), 3, "acc"
    # MCQ 계열은 별도 로더 사용
    raise ValueError(f"cls dataset 미지원: {name}")

class NLPClsRunner(TrainerBase):
    def __init__(self, model, tokenizer, collator, optimizer, scheduler, device,
                 task_name:str, metric_kind:str, amp="none", ttt_target=None, clip_grad_norm=None):
        super().__init__(model, optimizer, scheduler, device, amp, higher_is_better=True, ttt_target=ttt_target, clip_grad_norm=clip_grad_norm)
        self.tok = tokenizer; self.col = collator
        self.task_name = task_name; self.metric_kind = metric_kind

    def _metrics(self, gold, pred, prob=None) -> Dict[str,float]:
        if self.metric_kind=="acc":
            return {"acc": 100.0*sum(int(a==b) for a,b in zip(gold,pred))/max(len(gold),1)}
        if self.metric_kind=="macro_f1":
            labels = sorted(set(gold)); f1s=[]
            for c in labels:
                tp = sum(1 for g,p in zip(gold,pred) if g==c and p==c)
                fp = sum(1 for g,p in zip(gold,pred) if g!=c and p==c)
                fn = sum(1 for g,p in zip(gold,pred) if g==c and p!=c)
                prec = tp/max(tp+fp,1); rec = tp/max(tp+fn,1)
                f1s.append(0.0 if prec+rec==0 else 200*prec*rec/max(prec+rec,1e-12))
            return {"macro_f1": sum(f1s)/max(len(f1s),1)}
        if self.metric_kind=="f1_acc":
            acc = 100.0*sum(int(a==b) for a,b in zip(gold,pred))/max(len(gold),1)
            tp = sum(1 for g,p in zip(gold,pred) if g==1 and p==1)
            fp = sum(1 for g,p in zip(gold,pred) if g==0 and p==1)
            fn = sum(1 for g,p in zip(gold,pred) if g==1 and p==0)
            prec = tp/max(tp+fp,1); rec = tp/max(tp+fn,1)
            f1 = 0.0 if prec+rec==0 else 200*prec*rec/max(prec+rec,1e-12)
            return {"f1":f1, "acc":acc}
        if self.metric_kind=="pearson_spearman":
            import numpy as np
            g = np.array(gold, dtype=float); p = np.array(pred, dtype=float)
            def corr(a,b):
                if a.std()==0 or b.std()==0: return 0.0
                return float(np.corrcoef(a,b)[0,1]*100.0)
            return {"pearson": corr(g,p), "spearman": _spearman_pct(g.tolist(), p.tolist())}
        return {"acc": 0.0}

    def train_epoch(self, loader):
        self.model.train()
        self.step_times.clear(); self.losses_epoch.clear(); self.seen_tokens_epoch = 0
        pbar = tqdm(loader, desc=f"[CLS:{self.task_name}][Train]")
        for batch in pbar:
            t0 = time.perf_counter()
            batch = {k: v.to(self.device) for k,v in batch.items()}
            self.opt.zero_grad(set_to_none=True)
            if self.amp in ("fp16","bf16"):
                with torch.autocast(device_type="cuda", dtype=self.autocast_dtype):
                    out = self.model(**batch); loss = out.loss
                if self.amp=="fp16":
                    self.scaler.scale(loss).backward()
                    if self.clip_grad_norm:
                        self.scaler.unscale_(self.opt); torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip_grad_norm)
                    self.scaler.step(self.opt); self.scaler.update()
                else:
                    loss.backward()
                    if self.clip_grad_norm:
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip_grad_norm)
                    self.opt.step()
            else:
                out = self.model(**batch); loss = out.loss
                loss.backward()
                if self.clip_grad_norm:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip_grad_norm)
                self.opt.step()
            dt = time.perf_counter()-t0
            self.step_times.append(dt)
            self.losses_epoch.append(float(loss.detach().cpu()))
            non_pad = (batch["input_ids"] != self.model.config.pad_token_id).sum().item()
            self._mark_seen_tokens(non_pad)
            pbar.set_postfix(loss=f"{sum(self.losses_epoch)/len(self.losses_epoch):.4f}",
                             p50=f"{percentile(self.step_times,50):.3f}s")
            self._check_stop()

    @torch.no_grad()
    def validate(self, loader):
        self.model.eval()
        gold, pred, reg_pred, vloss, total = [], [], [], 0.0, 0
        for batch in tqdm(loader, desc=f"[CLS:{self.task_name}][Val]"):
            batch = {k: v.to(self.device) for k,v in batch.items()}
            if self.amp in ("fp16","bf16"):
                with torch.autocast(device_type="cuda", dtype=self.autocast_dtype):
                    out = self.model(**batch)
            else:
                out = self.model(**batch)
            vloss += float(out.loss.detach().cpu()) * batch["input_ids"].size(0)
            total += batch["input_ids"].size(0)
            if self.model.num_labels==1:  # STS-B 회귀
                reg_pred.extend(out.logits.squeeze(-1).detach().cpu().tolist())
                gold.extend(batch["labels"].detach().cpu().tolist())
            else:
                logits = out.logits
                p = logits.argmax(-1)
                pred.extend(p.detach().cpu().tolist())
                gold.extend(batch["labels"].detach().cpu().tolist())
            self._check_stop()
        if self.model.num_labels==1:
            m = self._metrics(gold, reg_pred); score = m.get("pearson", 0.0)
        else:
            m = self._metrics(gold, pred); score = list(m.values())[0]
        self._maybe_ttt(score); self._update_best(score)
        return {"val_score": score, "val_loss": vloss/max(total,1), **m}

# ===== MCQ (HellaSwag/PIQA/COPA/WinoGrande/AlphaNLI) =====
def build_mcq_loaders(name: str, model_name: str, max_length: int, batch_size: int, workers: int, amp: str):
    n = name.lower()

    # 📦 데이터셋 소스 고정 (ybisk 제거)
    if n == "winogrande":
        ds = _hf_load("winogrande", "winogrande_xl")
    elif n in ("alpha_nli", "abductive_nli", "abductive-nli"):
        _purge_cache_entry(["alpha_nli", "abductive_nli", "abductive-nli", "Rowan--abductive_nli", "XiangRong--abductive_nli"])
        try:
            ds = _hf_load("abductive_nli")
        except Exception:
            ds = _hf_load("abductive-nli")
    elif n == "piqa":
        ds = _hf_load("lighteval/piqa")
    elif n == "copa":
        ds = _hf_load("super_glue", "copa")
    elif n == "hellaswag":
        ds = _hf_load("Rowan/hellaswag")
    else:
        ds = _hf_load(n)  # 기타 MCQ

    tok = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token or tok.sep_token

    # HellaSwag는 문맥 길이가 길어지는 편이라 기본 길이를 살짝 키움
    eff_max_len = max_length if n != "hellaswag" else max(256, max_length)

    def preprocess(batch):
        out_ids, out_mask, out_lab = [], [], []
        first_key = next(iter(batch))
        for i in range(len(batch[first_key])):
            ex = {k: batch[k][i] for k in batch}
            ctx, choices, label, fill_blank = _mcq_unify_fields(n, ex)
            if fill_blank:
                filled = [ctx.replace("_", c) for c in choices]
                enc = tok(filled, truncation=True, max_length=eff_max_len)
            else:
                first = [ctx] * len(choices)
                enc = tok(first, choices, truncation=True, max_length=eff_max_len)
            out_ids.append(enc["input_ids"])
            out_mask.append(enc["attention_mask"])
            out_lab.append(int(label))
        return {"input_ids": out_ids, "attention_mask": out_mask, "labels": out_lab}

    tr_split = "train"
    if "validation" in ds:
        va_split = "validation"
    elif "test" in ds:
        va_split = "test"
    else:
        va_split = list(ds.keys())[0]

    ds_tr = ds[tr_split].map(preprocess, batched=True, remove_columns=ds[tr_split].column_names, desc=f"Tokenizing {n} train")
    ds_va = ds[va_split].map(preprocess, batched=True, remove_columns=ds[va_split].column_names, desc=f"Tokenizing {n} val")

    col = DataCollatorForMultipleChoice(tokenizer=tok, pad_to_multiple_of=(8 if amp in ("fp16","bf16") else None))
    pw = workers > 0
    tr = DataLoader(ds_tr, batch_size=batch_size, shuffle=True,  collate_fn=col, num_workers=workers, pin_memory=True, persistent_workers=pw)
    va = DataLoader(ds_va, batch_size=batch_size, shuffle=False, collate_fn=col, num_workers=workers, pin_memory=True, persistent_workers=pw)
    return tok, tr, va

class MCQRunner(TrainerBase):
    def __init__(self, model, optimizer, scheduler, device, task_name:str,
                 amp="none", ttt_target=None, clip_grad_norm=None):
        super().__init__(model, optimizer, scheduler, device, amp, True, ttt_target, clip_grad_norm)
        self.task_name = task_name

    def train_epoch(self, loader):
        import torch
        self.model.train()
        self.step_times.clear(); self.losses_epoch.clear(); self.seen_tokens_epoch = 0
        pbar = tqdm(loader, desc=f"[MCQ:{self.task_name}][Train]")
        for batch in pbar:
            t0 = time.perf_counter()
            batch = {k: v.to(self.device) for k, v in batch.items()}
            self.opt.zero_grad(set_to_none=True)
            if self.amp in ("fp16","bf16"):
                with torch.autocast(device_type="cuda", dtype=self.autocast_dtype):
                    out = self.model(**batch); loss = out.loss
                if self.amp == "fp16":
                    self.scaler.scale(loss).backward()
                    if self.clip_grad_norm:
                        self.scaler.unscale_(self.opt)
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip_grad_norm)
                    self.scaler.step(self.opt); self.scaler.update()
                else:
                    loss.backward()
                    if self.clip_grad_norm:
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip_grad_norm)
                    self.opt.step()
            else:
                out = self.model(**batch); loss = out.loss
                loss.backward()
                if self.clip_grad_norm:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip_grad_norm)
                self.opt.step()
            dt = time.perf_counter() - t0
            self.step_times.append(dt)
            self.losses_epoch.append(float(loss.detach().cpu()))
            non_pad = (batch["attention_mask"] > 0).sum().item()
            self._mark_seen_tokens(non_pad)
            pbar.set_postfix(loss=f"{sum(self.losses_epoch)/len(self.losses_epoch):.4f}",
                             p50=f"{percentile(self.step_times,50):.3f}s")
            self._check_stop()

    @torch.no_grad()
    def validate(self, loader):
        import torch
        self.model.eval()
        correct, total, vloss = 0, 0, 0.0
        for batch in tqdm(loader, desc=f"[MCQ:{self.task_name}][Val]"):
            batch = {k: v.to(self.device) for k, v in batch.items()}
            if self.amp in ("fp16","bf16"):
                with torch.autocast(device_type="cuda", dtype=self.autocast_dtype):
                    out = self.model(**batch)
            else:
                out = self.model(**batch)
            logits = out.logits  # [B, C]
            vloss += float(out.loss.detach().cpu()) * logits.size(0)
            preds = logits.argmax(-1)
            correct += (preds == batch["labels"]).sum().item()
            total   += batch["labels"].numel()
            self._check_stop()
        acc = 100.0 * correct / max(total, 1)
        self._maybe_ttt(acc); self._update_best(acc)
        return {"val_score": acc, "acc": acc, "val_loss": vloss/max(total,1)}

def _maybe_rescue_mcq(ep:int, runner:MCQRunner, opt:torch.optim.Optimizer, args, threshold=60.0):
    """
    MCQ 정체 구제: 1 epoch 이상 진행했고, 손실 분산이 낮고 점수가 낮으면 LR ×0.1
    """
    if not getattr(args, "rescue_lr", True): return
    if ep < 2: return
    import statistics as _stats
    loss_var = _stats.pvariance(runner.losses_epoch) if runner.losses_epoch else 0.0
    if runner.best_metric < threshold and loss_var < 1e-4:
        for g in opt.param_groups:
            g["lr"] = max(g["lr"] * 0.1, 1e-7)
        print(f"[RESCUE] MCQ stagnation detected → lr *= 0.1 (new lr: {opt.param_groups[0]['lr']:.2e})")


# ===================== PATCH 1: QARunner.validate 교체 =====================
class QARunner(TrainerBase):
    def __init__(self, model, tokenizer, optimizer, scheduler, device, amp="none", ttt_target=None, clip_grad_norm=None):
        super().__init__(model, optimizer, scheduler, device, amp, True, ttt_target, clip_grad_norm)
        self.tok = tokenizer

    # QARunner.train_epoch — 교체본
    def train_epoch(self, loader):
        self.model.train()
        self.step_times.clear(); self.losses_epoch.clear(); self.seen_tokens_epoch = 0
        pbar = tqdm(loader, desc="[QA][Train]")
        for batch in pbar:
            t0 = time.perf_counter()
            # 텐서만 디바이스로 이동 (train 로더엔 start/end_positions가 포함됨)
            batch = {k: (v.to(self.device) if torch.is_tensor(v) else v) for k, v in batch.items()}

            # ✅ forward kwargs: 검증 로더엔 start/end가 없으므로 조건부로 추가
            kwargs = {
                "input_ids": batch["input_ids"],
                "attention_mask": batch["attention_mask"],
            }
            if "start_positions" in batch and "end_positions" in batch:
                kwargs["start_positions"] = batch["start_positions"]
                kwargs["end_positions"]   = batch["end_positions"]

            self.opt.zero_grad(set_to_none=True)

            if self.amp in ("fp16", "bf16"):
                with torch.autocast(device_type="cuda", dtype=self.autocast_dtype):
                    out = self.model(**kwargs)
                    # train 에서는 loss가 반드시 존재하지만, 방어적으로 fallback 유지
                    loss = out.loss if getattr(out, "loss", None) is not None else \
                          (out.start_logits.mean()*0 + out.end_logits.mean()*0)
                if self.amp == "fp16":
                    self.scaler.scale(loss).backward()
                    if self.clip_grad_norm:
                        self.scaler.unscale_(self.opt)
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip_grad_norm)
                    self.scaler.step(self.opt); self.scaler.update()
                else:
                    loss.backward()
                    if self.clip_grad_norm:
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip_grad_norm)
                    self.opt.step()
            else:
                out = self.model(**kwargs)
                loss = out.loss if getattr(out, "loss", None) is not None else \
                      (out.start_logits.mean()*0 + out.end_logits.mean()*0)
                loss.backward()
                if self.clip_grad_norm:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip_grad_norm)
                self.opt.step()

            dt = time.perf_counter() - t0
            self.step_times.append(dt)
            self.losses_epoch.append(float(loss.detach().cpu()))
            # 토큰 수 집계 (패딩 제외)
            self._mark_seen_tokens(batch["input_ids"].ne(self.tok.pad_token_id).sum().item())
            pbar.set_postfix(loss=f"{sum(self.losses_epoch)/len(self.losses_epoch):.4f}",
                            p50=f"{percentile(self.step_times,50):.3f}s")
            self._check_stop()

    @torch.no_grad()
    def validate(self, loader):
        """
        공식 post-processing 스타일:
          - overflow된 feature들을 example_id로 모아 최고 점수 span 선택
          - offset_mapping으로 원문 context에서 정확한 문자 구간 추출
          - no-answer: CLS 점수(시작/끝 로짓의 CLS 위치 합)와 비교해 임계값으로 결정
        """
        self.model.eval()
        null_threshold = 0.0      # 필요시 CLI로 노출 가능
        max_answer_len = 30
        n_best_start = 20
        n_best_end   = 20

        # example_id → { "best_non_null":(score,text), "best_null":score, "gold":[...]} 누적
        ex_best = {}
        for batch in tqdm(loader, desc="[QA][Val]"):
            # 텐서만 GPU로, 나머지는 그대로 유지
            input_ids = batch["input_ids"].to(self.device)
            attn      = batch["attention_mask"].to(self.device)
            example_ids   = batch["example_id"]        # list[str]
            offset_maps   = batch["offset_mapping"]    # list[list[Tuple[int,int] or None]]
            contexts      = batch["contexts"]          # list[str]
            golds_per_feat= batch["answers"]           # list[list[str]]

            if self.amp in ("fp16","bf16"):
                with torch.autocast(device_type="cuda", dtype=self.autocast_dtype):
                    out = self.model(input_ids=input_ids, attention_mask=attn)
            else:
                out = self.model(input_ids=input_ids, attention_mask=attn)

            start_logits = out.start_logits.detach().cpu()
            end_logits   = out.end_logits.detach().cpu()
            input_ids_cpu = input_ids.detach().cpu()

            B, L = start_logits.size(0), start_logits.size(1)

            for i in range(B):
                ex_id   = example_ids[i]
                offsets = offset_maps[i]      # length L
                context = contexts[i]
                s_logit = start_logits[i]
                e_logit = end_logits[i]
                ids_row = input_ids_cpu[i]

                # CLS index (없으면 0)
                cls_idx = (ids_row == self.tok.cls_token_id).nonzero(as_tuple=False)
                cls_idx = int(cls_idx[0].item()) if len(cls_idx) else 0
                null_score = float(s_logit[cls_idx] + e_logit[cls_idx])

                # gold answers (예시마다 동일하므로 최초만 세팅)
                if ex_id not in ex_best:
                    ex_best[ex_id] = {
                        "best_non_null": (-1e18, ""),
                        "best_null": null_score,
                        "gold": golds_per_feat[i]
                    }
                else:
                    ex_best[ex_id]["gold"] = ex_best[ex_id]["gold"] or golds_per_feat[i]
                    ex_best[ex_id]["best_null"] = max(ex_best[ex_id]["best_null"], null_score)

                # 상위 k 시작/끝 후보
                s_topk_score, s_topk = torch.topk(s_logit, k=min(n_best_start, L))
                e_topk_score, e_topk = torch.topk(e_logit, k=min(n_best_end,  L))

                # 후보 span들 스코어링
                best_score = ex_best[ex_id]["best_non_null"][0]
                best_text  = ex_best[ex_id]["best_non_null"][1]
                for si, s_idx in enumerate(s_topk.tolist()):
                    for ei, e_idx in enumerate(e_topk.tolist()):
                        if s_idx > e_idx:
                            continue
                        if (e_idx - s_idx + 1) > max_answer_len:
                            continue
                        # context 토큰이 아닌 위치 제거 (prepare_val_features에서 non-context는 None 처리)
                        if offsets[s_idx] is None or offsets[e_idx] is None:
                            continue
                        char_s, char_e = offsets[s_idx][0], offsets[e_idx][1]
                        if char_s is None or char_e is None or char_e <= char_s:
                            continue
                        span_text = context[char_s:char_e].strip()
                        if len(span_text) == 0:
                            continue
                        score = float(s_logit[s_idx] + e_logit[e_idx])
                        if score > best_score:
                            best_score = score
                            best_text  = span_text
                ex_best[ex_id]["best_non_null"] = (best_score, best_text)
            self._check_stop()

        # 최종 선택 (null vs non-null)
        preds = {}
        golds = {}
        for ex_id, rec in ex_best.items():
            best_non_null_score, best_text = rec["best_non_null"]
            best_null_score = rec["best_null"]
            pred_text = "" if (best_null_score > best_non_null_score + null_threshold) else best_text
            preds[ex_id] = pred_text
            golds[ex_id] = rec["gold"]

        # EM/F1 계산
        def _f1(a,b):
            wa, wb = norm_answer(a).split(), norm_answer(b).split()
            if not wa and not wb: return 1.0
            common = len(set(wa) & set(wb))
            if common == 0: return 0.0
            prec = common / max(len(wa),1); rec = common / max(len(wb),1)
            return 2*prec*rec / max(prec+rec,1e-12)

        ems, f1s = [], []
        for ex_id in preds.keys():
            pred = preds[ex_id]
            gold_list = golds.get(ex_id, [])
            if not gold_list:  # 방어
                ems.append(0.0); f1s.append(0.0); continue
            em = 1.0 if any(norm_answer(pred)==norm_answer(g) for g in gold_list) else 0.0
            f1 = max([_f1(pred, g) for g in gold_list]) if gold_list else 0.0
            ems.append(em); f1s.append(f1)

        em = 100.0 * (sum(ems) / max(len(ems),1))
        f1 = 100.0 * (sum(f1s) / max(len(f1s),1))
        score = 0.5*(em+f1)
        self._maybe_ttt(score); self._update_best(score)
        return {"val_score": score, "em": em, "f1": f1}
# =================== /PATCH 1 ===================

# ===================== PATCH 2: build_squad2_loaders 교체 =====================
def build_squad2_loaders(model_name:str, batch:int, workers:int,
                         max_len:int=384, doc_stride:int=128):
    # ✔ use_auth_token 제거
    tok = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token or tok.sep_token
    ds = _hf_load("squad_v2")

    # ---- Train features (표준 처리) ----
    def prepare_train_features(examples):
        tokenized = tok(
            examples["question"], examples["context"],
            truncation="only_second", max_length=max_len, stride=doc_stride,
            return_overflowing_tokens=True, return_offsets_mapping=True, padding="max_length",
        )
        sample_map = tokenized.pop("overflow_to_sample_mapping")
        offsets = tokenized.pop("offset_mapping")
        start_positions, end_positions = [], []
        for i, off in enumerate(offsets):
            input_ids = tokenized["input_ids"][i]
            cls_index = input_ids.index(tok.cls_token_id) if tok.cls_token_id in input_ids else 0
            sample_idx = sample_map[i]
            answers = examples["answers"][sample_idx]
            if len(answers["answer_start"]) == 0:
                start_positions.append(cls_index); end_positions.append(cls_index); continue
            start_char = answers["answer_start"][0]
            end_char = start_char + len(answers["text"][0])
            sequence_ids = tokenized.sequence_ids(i)
            # context 구간 찾기
            idx = 0
            while idx < len(sequence_ids) and sequence_ids[idx] != 1: idx += 1
            context_start = idx
            while idx < len(sequence_ids) and sequence_ids[idx] == 1: idx += 1
            context_end = idx - 1
            if not (off[context_start][0] <= start_char and off[context_end][1] >= end_char):
                start_positions.append(cls_index); end_positions.append(cls_index); continue
            start_token = context_start
            while start_token <= context_end and off[start_token][0] <= start_char:
                start_token += 1
            start_token -= 1
            end_token = start_token
            while end_token <= context_end and off[end_token][1] < end_char:
                end_token += 1
            start_positions.append(start_token)
            end_positions.append(end_token if end_token<=context_end else context_end)
        tokenized["start_positions"] = start_positions
        tokenized["end_positions"] = end_positions
        return tokenized

    # ---- Val features: 공식 post-processing을 위한 구성 ----
    def prepare_val_features(examples):
        tokenized = tok(
            examples["question"], examples["context"],
            truncation="only_second", max_length=max_len, stride=doc_stride,
            return_overflowing_tokens=True, return_offsets_mapping=True, padding="max_length",
        )
        sample_map = tokenized.pop("overflow_to_sample_mapping")
        # example_id 기록
        tokenized["example_id"] = []
        # non-context 토큰은 offset을 None으로 마스킹 (후처리에 필요)
        offsets = tokenized["offset_mapping"]
        for i in range(len(tokenized["input_ids"])):
            sample_idx = sample_map[i]
            tokenized["example_id"].append(examples["id"][sample_idx])
            seq_ids = tokenized.sequence_ids(i)
            offsets[i] = [o if seq_ids[k]==1 else None for k,o in enumerate(offsets[i])]
        tokenized["offset_mapping"] = offsets
        return tokenized

    train_set = ds["train"].map(
        prepare_train_features, batched=True, remove_columns=ds["train"].column_names, desc="Tokenizing train"
    )
    val_examples = ds["validation"]  # 원본 텍스트/정답 보유
    val_set = val_examples.map(
        prepare_val_features, batched=True, remove_columns=val_examples.column_names, desc="Tokenizing val"
    )

    # ---- id → gold answers / context 매핑 ----
    ans_map = {ex_id: val_examples[i]["answers"]["text"] for i, ex_id in enumerate(val_examples["id"])}
    ctx_map = {ex_id: val_examples[i]["context"]        for i, ex_id in enumerate(val_examples["id"])}

    col = DefaultDataCollator()
    pw = workers > 0
    tr = DataLoader(
        train_set, batch_size=batch, shuffle=True, collate_fn=col,
        num_workers=workers, pin_memory=True, persistent_workers=pw
    )
    va = DataLoader(
        SquadValDataset(val_set, ans_map, ctx_map), batch_size=batch, shuffle=False,
        collate_fn=squad_val_collate, num_workers=workers, pin_memory=True, persistent_workers=pw
    )
    return tok, tr, va
# =================== /PATCH 2 =====================



# ===== Seq2Seq (WMT14 En-De / XSum) =====
class S2SRunner(TrainerBase):
    def __init__(self, model, tokenizer, optimizer, scheduler, device, task_name:str, amp="none", ttt_target=None, clip_grad_norm=None, max_gen_len=128):
        super().__init__(model, optimizer, scheduler, device, amp, True, ttt_target, clip_grad_norm)
        self.tok = tokenizer; self.task_name = task_name; self.max_gen_len = max_gen_len

    def train_epoch(self, loader):
        self.model.train()
        self.step_times.clear(); self.losses_epoch.clear(); self.seen_tokens_epoch=0
        pbar = tqdm(loader, desc=f"[S2S:{self.task_name}][Train]")
        for batch in pbar:
            t0 = time.perf_counter()
            batch = {k: v.to(self.device) for k,v in batch.items()}
            self.opt.zero_grad(set_to_none=True)
            if self.amp in ("fp16","bf16"):
                with torch.autocast(device_type="cuda", dtype=self.autocast_dtype):
                    out = self.model(**batch); loss = out.loss
                if self.amp=="fp16":
                    self.scaler.scale(loss).backward()
                    if self.clip_grad_norm:
                        self.scaler.unscale_(self.opt); torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip_grad_norm)
                    self.scaler.step(self.opt); self.scaler.update()
                else:
                    loss.backward()
                    if self.clip_grad_norm:
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip_grad_norm)
                    self.opt.step()
            else:
                out = self.model(**batch); loss = out.loss
                loss.backward()
                if self.clip_grad_norm:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip_grad_norm)
                self.opt.step()
            dt = time.perf_counter()-t0
            self.step_times.append(dt); self.losses_epoch.append(float(loss.detach().cpu()))
            self._mark_seen_tokens(batch["input_ids"].ne(self.tok.pad_token_id).sum().item() + batch["labels"].ne(-100).sum().item())
            pbar.set_postfix(loss=f"{sum(self.losses_epoch)/len(self.losses_epoch):.4f}",
                             p50=f"{percentile(self.step_times,50):.3f}s")
            self._check_stop()

    @torch.no_grad()
    def validate(self, loader, metric:str):
        self.model.eval()
        preds, refs = [], []
        for batch in tqdm(loader, desc=f"[S2S:{self.task_name}][Val]"):
            in_ids = batch["input_ids"].to(self.device)
            attn   = batch["attention_mask"].to(self.device)
            gen = self.model.generate(input_ids=in_ids, attention_mask=attn, max_length=self.max_gen_len, num_beams=4)
            preds.extend(self.tok.batch_decode(gen, skip_special_tokens=True))
            y = batch["labels"].clone()
            y[y==-100] = self.tok.pad_token_id
            refs.extend(self.tok.batch_decode(y, skip_special_tokens=True))
            self._check_stop()
        if metric=="bleu":
            if _sacrebleu:
                bleu = _sacrebleu.corpus_bleu(preds, [refs]).score
            else:
                score=[]
                for p,r in zip(preds,refs):
                    wp=set(p.split()); wr=set(r.split()); score.append(100.0*len(wp&wr)/max(len(wp),1))
                bleu=sum(score)/max(len(score),1)
            score=bleu
        else:  # rouge-l
            if _evaluate:
                rouge = _evaluate.load("rouge")
                res = rouge.compute(predictions=preds, references=refs, use_stemmer=True)
                score = 100.0*res["rougeL"]
            else:
                score = simple_rouge_l(preds, refs)
        self._maybe_ttt(score); self._update_best(score)
        return {"val_score": score}

def build_wmt14_loaders(direction: str, tok, batch_size: int, workers: int):
    ds, src, tgt, used = _load_wmt_with_direction(direction)

    def preprocess(batch):
        # ① translation dict 스키마
        if "translation" in batch:
            translations = batch["translation"]
            sources = [ex[src] for ex in translations]
            targets = [ex[tgt] for ex in translations]
        # ② 언어별 칼럼 스키마 (일반적인 wmt14 de-en: 'de', 'en')
        elif src in batch and tgt in batch:
            sources = batch[src]
            targets = batch[tgt]
        else:
            raise KeyError(f"WMT14 columns not recognized. keys={list(batch.keys())}, "
                           f"expected 'translation' or '{src}','{tgt}'")

        model_inputs = tok(sources, max_length=128, truncation=True)
        labels = tok(text_target=targets, max_length=128, truncation=True)
        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

    cols = ds["train"].column_names
    ds_proc = ds.map(preprocess, batched=True, remove_columns=cols)

    # ✅ 라벨 패딩을 -100으로 처리 (loss에서 무시)
    col = DataCollatorForSeq2Seq(
        tokenizer=tok,
        label_pad_token_id=-100,
        pad_to_multiple_of=(8 if torch.cuda.is_available() else None),
    )

    pw = workers > 0
    tr = torch.utils.data.DataLoader(
        ds_proc["train"], batch_size=batch_size, shuffle=True,
        collate_fn=col, num_workers=workers, pin_memory=True, persistent_workers=pw
    )
    va_key = "validation" if "validation" in ds_proc else "test"
    va = torch.utils.data.DataLoader(
        ds_proc[va_key], batch_size=batch_size, shuffle=False,
        collate_fn=col, num_workers=workers, pin_memory=True, persistent_workers=pw
    )
    return tr, va

from datasets import load_dataset
from transformers import DataCollatorForSeq2Seq
from torch.utils.data import DataLoader

# ===================== PATCH 3 (refined) =====================
def build_xsum_loaders(tok, batch_size: int, workers: int):
    """
    Robust XSum loader:
    1) Try official hub script (will likely fail on new datasets)
    2) Fallback to GEM parquet mirror (refs/convert/parquet)
    """
    _purge_cache_entry(["xsum", "GEM--xsum", "EdinburghNLP--xsum"])
    try:
        ds = _hf_load("EdinburghNLP/xsum")  # columns: document, summary
        src_key, tgt_key = "document", "summary"
    except Exception as e1:
        try:
            print(f"[XSum Load] Official script failed ({e1}). Falling back to GEM/gem parquet mirror...")
            base = "https://huggingface.co/datasets/GEM/gem/resolve/refs/convert/parquet/xsum"
            data_files = {
                "train": f"{base}/gem-train.parquet",
                "validation": f"{base}/gem-validation.parquet",
                "test": f"{base}/gem-test.parquet",
            }
            ds = load_dataset("parquet", data_files=data_files, cache_dir=str(HF_CACHE))
            cols = set(ds["train"].column_names)

            if {"document", "summary"}.issubset(cols):
                src_key, tgt_key = "document", "summary"
            elif {"source", "target"}.issubset(cols):
                src_key, tgt_key = "source", "target"
            elif "source" in cols and "references" in cols:
                src_key, tgt_key = "source", "references"
            elif "document" in cols and "target" in cols:
                src_key, tgt_key = "document", "target"
            elif "document" in cols and "references" in cols:
                src_key, tgt_key = "document", "references"
            else:
                raise ValueError(f"Unexpected XSum columns from mirror: {cols}")

            print(f"[XSum Load] Using mirror columns: src='{src_key}', tgt='{tgt_key}'")

        except Exception as e2:
            raise RuntimeError(
                "XSum download failed:\n"
                f"- EdinburghNLP/xsum error: {e1}\n"
                f"- GEM/gem parquet error: {e2}"
            )

    all_cols = ds["train"].column_names

    def preprocess(batch):
        if tgt_key == "references":
            srcs = batch[src_key]
            refs = batch.get("references")
            # guard: None / non-list entries
            tgts = []
            for r in refs:
                if isinstance(r, list) and len(r) > 0:
                    tgts.append(r[0] if isinstance(r[0], str) else "")
                else:
                    tgts.append("")
        else:
            srcs = batch[src_key]
            tgts = batch[tgt_key]

        model_inputs = tok(srcs, max_length=1024, truncation=True)
        labels = tok(text_target=tgts, max_length=128, truncation=True)
        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

    ds_tok = ds.map(preprocess, batched=True, remove_columns=all_cols, desc="Tokenizing XSum")

    col = DataCollatorForSeq2Seq(
        tokenizer=tok,
        label_pad_token_id=-100,
        pad_to_multiple_of=(8 if torch.cuda.is_available() else None),
    )
    pw = workers > 0
    train_loader = DataLoader(
        ds_tok["train"], batch_size=batch_size, shuffle=True,
        num_workers=workers, pin_memory=True, persistent_workers=pw, collate_fn=col
    )
    val_loader = DataLoader(
        ds_tok["validation"], batch_size=batch_size, shuffle=False,
        num_workers=workers, pin_memory=True, persistent_workers=pw, collate_fn=col
    )
    return train_loader, val_loader
# =================== /PATCH 3 =====================




# ===== LM (WikiText-2; causal LM) =====
def pack_blocks(token_ids: List[int], block_size:int) -> List[List[int]]:
    total = (len(token_ids)//block_size)*block_size
    token_ids = token_ids[:total]
    return [token_ids[i:i+block_size] for i in range(0,total,block_size)]

class LMLoader(torch.utils.data.Dataset):
    def __init__(self, tokens: List[int], block_size:int):
        self.blocks = pack_blocks(tokens, block_size)
    def __len__(self): return len(self.blocks)
    def __getitem__(self, idx):
        ids = self.blocks[idx]
        x = torch.tensor(ids[:-1], dtype=torch.long)
        y = torch.tensor(ids[1:], dtype=torch.long)
        return {"input_ids": x, "labels": y}

def lm_collate(batch: List[Dict[str,torch.Tensor]]):
    return {k: torch.stack([b[k] for b in batch], dim=0) for k in batch[0].keys()}

class LMRunner(TrainerBase):
    def __init__(self, model, optimizer, scheduler, device, amp="none", ttt_target=None, clip_grad_norm=None):
        super().__init__(model, optimizer, scheduler, device, amp, False, ttt_target, clip_grad_norm)

    def train_epoch(self, loader):
        self.model.train()
        self.step_times.clear(); self.losses_epoch.clear(); self.seen_tokens_epoch=0
        pbar = tqdm(loader, desc="[LM][Train]")
        for batch in pbar:
            t0 = time.perf_counter()
            x = batch["input_ids"].to(self.device); y = batch["labels"].to(self.device)
            self.opt.zero_grad(set_to_none=True)
            if self.amp in ("fp16","bf16"):
                with torch.autocast(device_type="cuda", dtype=self.autocast_dtype):
                    out = self.model(input_ids=x, labels=y); loss = out.loss
                if self.amp=="fp16":
                    self.scaler.scale(loss).backward()
                    if self.clip_grad_norm:
                        self.scaler.unscale_(self.opt); torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip_grad_norm)
                    self.scaler.step(self.opt); self.scaler.update()
                else:
                    loss.backward()
                    if self.clip_grad_norm:
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip_grad_norm)
                    self.opt.step()
            else:
                out = self.model(input_ids=x, labels=y); loss = out.loss
                loss.backward()
                if self.clip_grad_norm:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip_grad_norm)
                self.opt.step()
            dt = time.perf_counter()
            self.step_times.append(dt - t0)
            self.losses_epoch.append(float(loss.detach().cpu()))
            self._mark_seen_tokens(x.numel())
            pbar.set_postfix(loss=f"{sum(self.losses_epoch)/len(self.losses_epoch):.4f}",
                             p50=f"{percentile(self.step_times,50):.3f}s")
            self._check_stop()

    @torch.no_grad()
    def validate(self, loader):
        self.model.eval()
        tot_loss, tot_tok = 0.0, 0
        for batch in tqdm(loader, desc="[LM][Val]"):
            x = batch["input_ids"].to(self.device); y = batch["labels"].to(self.device)
            if self.amp in ("fp16","bf16"):
                with torch.autocast(device_type="cuda", dtype=self.autocast_dtype):
                    out = self.model(input_ids=x, labels=y); loss = out.loss
            else:
                out = self.model(input_ids=x, labels=y); loss = out.loss
            tokens = x.numel()
            tot_loss += float(loss.detach().cpu()) * tokens
            tot_tok  += tokens
            self._check_stop()
        mean_loss = tot_loss/max(tot_tok,1)
        ppl = math.exp(mean_loss) if mean_loss < 20 else float("inf")
        self._maybe_ttt(ppl); self._update_best(ppl)
        return {"val_ppl": ppl, "val_loss_tok": mean_loss}

# ===== Math (GSM8K) =====
class MathRunner(TrainerBase):
    def __init__(self, model, tokenizer, optimizer, scheduler, device, amp="none", ttt_target=None, clip_grad_norm=None, max_new_tokens=128):
        super().__init__(model, optimizer, scheduler, device, amp, True, ttt_target, clip_grad_norm)
        self.tok=tokenizer; self.max_new_tokens=max_new_tokens

    def train_epoch(self, loader):
        self.model.train()
        self.step_times.clear(); self.losses_epoch.clear(); self.seen_tokens_epoch=0
        pbar = tqdm(loader, desc="[GSM8K][Train](optional)")
        for batch in pbar:
            t0=time.perf_counter()
            inp=batch["input_ids"].to(self.device)
            labels=batch["labels"].to(self.device)
            self.opt.zero_grad(set_to_none=True)
            out=self.model(input_ids=inp, labels=labels); loss=out.loss
            loss.backward();
            if self.clip_grad_norm: torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip_grad_norm)
            self.opt.step()
            dt=time.perf_counter()-t0
            self.step_times.append(dt); self.losses_epoch.append(float(loss.detach().cpu())); self._mark_seen_tokens(inp.numel())
            self._check_stop()

    @torch.no_grad()
    def validate(self, loader):
        self.model.eval()
        golds, preds=[], []
        for batch in tqdm(loader, desc="[GSM8K][Eval]"):
            inp = batch["input_ids"].to(self.device)
            gen = self.model.generate(inp, max_new_tokens=self.max_new_tokens)
            out = self.tok.batch_decode(gen, skip_special_tokens=True)
            preds += [extract_last_number(o) or "" for o in out]
            golds += [extract_last_number(g) or "" for g in batch["answers"]]
            self._check_stop()
        acc = 100.0*sum(int(a==b) for a,b in zip(golds,preds))/max(len(golds),1)
        self._maybe_ttt(acc); self._update_best(acc)
        return {"val_score": acc, "acc": acc}


def build_gsm8k_loaders(model_name: str, batch_size: int, workers: int, max_len: int = 512):
    # 1) tokenizer: pad 토큰 확보 + 왼쪽 패딩
    tok = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token or tok.unk_token
    tok.padding_side = "left"  # ★ decoder-only는 left padding 권장

    # 2) 데이터 로드
    ds = _hf_load("gsm8k", "main")

    def _prepare(ex, max_len=max_len):
        q = ex["question"]
        a = ex["answer"]
        # GSM8K 정답 항에서 최종 숫자 추출(형식: '#### 123') — 학습 라벨용 gold
        final = (a.split("####")[-1].strip() if isinstance(a, str) and "####" in a else (a or "").strip())

        prompt = f"Question: {q}\nAnswer:"
        ans = " " + final

        p_ids = tok(prompt, add_special_tokens=False).input_ids
        a_ids = tok(ans, add_special_tokens=False).input_ids + [tok.eos_token_id]

        # 답변은 최대한 보존, 프롬프트는 남는 길이만
        keep_p = max_len - len(a_ids)
        if keep_p <= 0:
            p_ids = []
            a_ids = a_ids[-max_len:]
        else:
            p_ids = p_ids[-keep_p:]

        content = p_ids + a_ids
        content_len = len(content)
        pad_n = max(0, max_len - content_len)

        # ★ 왼쪽 패딩 + mask/labels 구성
        input_ids      = ([tok.pad_token_id] * pad_n) + content
        attention_mask = ([0] * pad_n) + ([1] * content_len)
        labels         = ([-100] * pad_n) + ([-100] * len(p_ids)) + a_ids[:]

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
            "gold": final,
        }

    # 3) 토크나이즈
    tr_tok = ds["train"].map(_prepare, remove_columns=ds["train"].column_names)
    te_tok = ds["test"].map(_prepare,  remove_columns=ds["test"].column_names)

    # 4) PyTorch Dataset/DataLoader
    tr_ds = GSM8KTrainDS(tr_tok)
    te_ds = GSM8KValDS(te_tok)

    tr_loader = DataLoader(
        tr_ds, batch_size=batch_size, shuffle=True,
        num_workers=workers, pin_memory=True, collate_fn=gsm8k_train_collate
    )
    te_loader = DataLoader(
        te_ds, batch_size=batch_size, shuffle=False,
        num_workers=workers, pin_memory=True, collate_fn=gsm8k_val_collate
    )
    return tok, tr_loader, te_loader

# ===== /PATCH =====

# ===== 태스크 실행기 =====
def run_glue_superglue(args, task:str):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    ds, ds_keys, num_labels, metric_kind = load_cls_dataset(task)
    if task == "anli":
        from datasets import DatasetDict, concatenate_datasets
        if "train" not in ds:
            ds = DatasetDict({
                "train": concatenate_datasets([ds[k] for k in ("train_r1","train_r2","train_r3") if k in ds]),
                "validation": concatenate_datasets([ds[k] for k in ("dev_r1","dev_r2","dev_r3") if k in ds]),
            })

    mdl_name = args.model_name or ("roberta-base" if task in ("rte","cb","boolq","anli") else "bert-base-uncased")
    tok = AutoTokenizer.from_pretrained(mdl_name, use_fast=True)
    model = AutoModelForSequenceClassification.from_pretrained(mdl_name, num_labels=num_labels).to(device)
    if model.config.pad_token_id is None:
        tok.pad_token = tok.eos_token or tok.sep_token
        model.config.pad_token_id = tok.pad_token_id

    if isinstance(ds_keys, tuple) and len(ds_keys)==2 and isinstance(ds_keys[0], tuple):
        k1,k2 = ds_keys
        def preprocess(ex):
            r = tok(ex[k1[0]], ex[k1[1]], truncation=True, max_length=args.max_length)
            r["labels"]=ex["label"]; return r
    elif isinstance(ds_keys, tuple) and len(ds_keys)==2 and isinstance(ds_keys[0], str) and ds_keys[1] is not None:
        def preprocess(ex):
            r=tok(ex[ds_keys[0]], ex[ds_keys[1]], truncation=True, max_length=args.max_length); r["labels"]=ex["label"]; return r
    else:
        def preprocess(ex):
            r=tok(ex[ds_keys[0]], truncation=True, max_length=args.max_length); r["labels"]=ex["label"]; return r

    split_val = "validation" if "validation" in ds else ("validation_matched" if "validation_matched" in ds else "test")
    ds_tr = ds["train"].map(preprocess, batched=True, remove_columns=ds["train"].column_names)
    ds_va = ds[split_val].map(preprocess, batched=True, remove_columns=ds[split_val].column_names)

    col = DataCollatorWithPadding(tokenizer=tok, pad_to_multiple_of=8 if args.amp in ("fp16","bf16") else None)
    trldr = DataLoader(ds_tr, batch_size=args.batch_size, shuffle=True, collate_fn=col, num_workers=args.workers)
    valdr = DataLoader(ds_va, batch_size=args.batch_size*2, shuffle=False, collate_fn=col, num_workers=args.workers)

    base_rico, base_other = 5e-4, 3e-5
    lr = args.lr or (base_rico if args.optimizer=="rico" else base_other)
    opt = build_optimizer(model, args.optimizer, lr, args.wd, args)
    sch = make_epoch_scheduler(opt, args.epochs, args.warmup_ratio)
    ttt_default = args.ttt_target or (90.0 if task=="sst2" else 75.0)
    runner = NLPClsRunner(model, tok, col, opt, sch, device, task, metric_kind, amp=args.amp, ttt_target=ttt_default, clip_grad_norm=args.clip_grad_norm)

    # 🔑 피크 메모리 집계를 "런 전체"로 (에포크 밖에서 reset)
    if torch.cuda.is_available(): torch.cuda.reset_peak_memory_stats()

    csv = Path(args.log_csv) if args.log_csv else None
    csv_keys = ["epoch","lr","train_loss","p50","p90","p99","throughput_tok_s","val_score","best_val","peak_mem_mb"]
    best=-1e9; wall0=time.perf_counter()
    interrupted_ckpt = None
    current_epoch = 0
    last_val = {}
    try:
        for ep in range(1, args.epochs+1):
            current_epoch = ep
            runner.train_epoch(trldr)
            tr_sum={"train_loss": sum(runner.losses_epoch)/max(len(runner.losses_epoch),1),
                    "p50": percentile(runner.step_times,50), "p90":percentile(runner.step_times,90),
                    "p99":percentile(runner.step_times,99), "throughput_tok_s": runner._throughput()}
            val=runner.validate(valdr); last_val = val
            sch.step(); best=max(best, val["val_score"])
            peak = torch.cuda.max_memory_allocated() if torch.cuda.is_available() else 0
            row={"epoch":ep,"lr":sch.get_last_lr()[0],**tr_sum,**val,"best_val":best,"peak_mem_mb":human_mb(peak)}
            write_csv(csv, row, header_keys=csv_keys)
            print(f"[CLS:{task}] ep {ep}/{args.epochs} | score {val['val_score']:.2f} | best {best:.2f} | p99 {tr_sum['p99']:.3f}s")
    except KeyboardInterrupt:
        interrupted_ckpt = _save_nlp_checkpoint("nlp_cls", task, args, model, opt, sch, current_epoch, best, {"val_summary": last_val})
        print(f"[CLS:{task}] 'exit' 신호 감지 → 체크포인트 저장: {interrupted_ckpt}")
    total_sec=time.perf_counter()-wall0
    loss_var=float(torch.tensor(runner.losses_epoch).var().item() if runner.losses_epoch else 0.0)
    summary = {
        "task":"cls", "dataset":task, "model":mdl_name, "optimizer":args.optimizer,
        "ttt_sec": runner.ttt_sec, "perf": best, "task_score": best,
        "speed": percentile(runner.step_times,50), "throughput": runner._throughput(),
        "hp_sensitivity_sd": None, "gen_gap": None, "loss_var": loss_var,
        "peak_mem_mb": human_mb(torch.cuda.max_memory_allocated() if torch.cuda.is_available() else 0),
        "total_time_sec": total_sec
    }
    return _attach_interrupt_meta(summary, interrupted_ckpt)

def run_mcq(args, task: str):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    mdl_name = args.model_name or "roberta-base"
    tok, tr, va = build_mcq_loaders(task, mdl_name, args.max_length, args.batch_size, args.workers, args.amp)
    model = AutoModelForMultipleChoice.from_pretrained(mdl_name).to(device)

    base_rico, base_other = 5e-4, 3e-5
    lr = args.lr or (base_rico if args.optimizer == "rico" else base_other)
    opt = build_optimizer(model, args.optimizer, lr, args.wd, args)
    sch = make_epoch_scheduler(opt, args.epochs, args.warmup_ratio)

    default_ttt = args.ttt_target
    if default_ttt is None:
        if task == "hellaswag": default_ttt = 80.0
        elif task == "piqa":    default_ttt = 78.0
        elif task in ("winogrande", "copa"): default_ttt = 65.0
        elif task in ("alpha_nli", "abductive_nli", "abductive-nli"): default_ttt = 70.0
        else: default_ttt = 70.0
    runner = MCQRunner(model, opt, sch, device, task_name=task, amp=args.amp, ttt_target=default_ttt, clip_grad_norm=args.clip_grad_norm)

    if torch.cuda.is_available(): torch.cuda.reset_peak_memory_stats()

    best=-1e9; wall0=time.perf_counter()
    interrupted_ckpt = None
    current_epoch = 0
    last_val = {}
    try:
        for ep in range(1, args.epochs+1):
            current_epoch = ep
            runner.train_epoch(tr)
            val = runner.validate(va); last_val = val
            _maybe_rescue_mcq(ep, runner, opt, args)
            sch.step()
            best = max(best, val["val_score"])
            peak = torch.cuda.max_memory_allocated() if torch.cuda.is_available() else 0
            print(f"[MCQ:{task}] ep {ep}/{args.epochs} | acc {val['val_score']:.2f} | best {best:.2f} | p99 {percentile(runner.step_times,99):.3f}s | peak {human_mb(peak)} MB")
    except KeyboardInterrupt:
        interrupted_ckpt = _save_nlp_checkpoint("nlp_mcq", task, args, model, opt, sch, current_epoch, best, {"val_summary": last_val})
        print(f"[MCQ:{task}] 'exit' 신호 감지 → 체크포인트 저장: {interrupted_ckpt}")

    total_sec = time.perf_counter() - wall0
    loss_var = float(torch.tensor(runner.losses_epoch).var().item() if runner.losses_epoch else 0.0)
    summary = {
        "task":"mcq", "dataset":task, "model":mdl_name, "optimizer":args.optimizer,
        "ttt_sec": runner.ttt_sec, "perf": best, "task_score": best,
        "speed": percentile(runner.step_times,50), "throughput": runner._throughput(),
        "hp_sensitivity_sd": None, "gen_gap": None, "loss_var": loss_var,
        "peak_mem_mb": human_mb(torch.cuda.max_memory_allocated() if torch.cuda.is_available() else 0),
        "total_time_sec": total_sec
    }
    return _attach_interrupt_meta(summary, interrupted_ckpt)

def run_qa_squad2(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    mdl_name = args.model_name or "bert-base-uncased"
    tok, tr, va = build_squad2_loaders(mdl_name, args.batch_size, args.workers)
    model = AutoModelForQuestionAnswering.from_pretrained(mdl_name).to(device)
    base_rico, base_other = 5e-3, 3e-5
    lr = args.lr or (base_rico if args.optimizer=="rico" else base_other)
    opt=build_optimizer(model,args.optimizer,lr,args.wd,args); sch=make_epoch_scheduler(opt, args.epochs, args.warmup_ratio)
    runner = QARunner(model, tok, opt, sch, device, amp=args.amp, ttt_target=args.ttt_target, clip_grad_norm=args.clip_grad_norm)

    if torch.cuda.is_available(): torch.cuda.reset_peak_memory_stats()

    best=-1e9; wall0=time.perf_counter()
    interrupted_ckpt = None
    current_epoch = 0
    last_val = {}
    try:
        for ep in range(1,args.epochs+1):
            current_epoch = ep
            runner.train_epoch(tr); val=runner.validate(va); last_val = val
            sch.step(); best=max(best, val["val_score"])
            print(f"[QA:SQuADv2] ep {ep}/{args.epochs} | score {val['val_score']:.2f} | best {best:.2f}")
    except KeyboardInterrupt:
        interrupted_ckpt = _save_nlp_checkpoint("nlp_qa", "squad_v2", args, model, opt, sch, current_epoch, best, {"val_summary": last_val})
        print(f"[QA:SQuADv2] 'exit' 신호 감지 → 체크포인트 저장: {interrupted_ckpt}")
    total_sec=time.perf_counter()-wall0
    loss_var=float(torch.tensor(runner.losses_epoch).var().item() if runner.losses_epoch else 0.0)
    summary = {
        "task":"qa", "dataset":"squad_v2", "model":mdl_name, "optimizer":args.optimizer,
        "ttt_sec": runner.ttt_sec, "perf": best, "task_score": best,
        "speed": percentile(runner.step_times,50), "throughput": runner._throughput(),
        "hp_sensitivity_sd": None, "gen_gap": None, "loss_var": loss_var,
        "peak_mem_mb": human_mb(torch.cuda.max_memory_allocated() if torch.cuda.is_available() else 0),
        "total_time_sec": total_sec
    }
    return _attach_interrupt_meta(summary, interrupted_ckpt)

def run_wmt14(args, direction="en-de"):
    device="cuda" if torch.cuda.is_available() else "cpu"
    mdl_name = args.model_name or ("Helsinki-NLP/opus-mt-en-de" if direction=="en-de" else "Helsinki-NLP/opus-mt-de-en")
    tok = AutoTokenizer.from_pretrained(mdl_name, use_fast=True)
    model = AutoModelForSeq2SeqLM.from_pretrained(mdl_name).to(device)
    tr, va = build_wmt14_loaders(direction, tok, args.batch_size, args.workers)
    base_rico, base_other = 5e-3, 3e-5
    lr=args.lr or (base_rico if args.optimizer=="rico" else base_other)
    opt=build_optimizer(model,args.optimizer,lr,args.wd,args); sch=make_epoch_scheduler(opt, args.epochs, args.warmup_ratio)
    runner=S2SRunner(model,tok,opt,sch,device,task_name=f"wmt14-{direction}",amp=args.amp,ttt_target=args.ttt_target,clip_grad_norm=args.clip_grad_norm,max_gen_len=128)

    if torch.cuda.is_available(): torch.cuda.reset_peak_memory_stats()

    best=-1e9; wall0=time.perf_counter()
    interrupted_ckpt = None
    current_epoch = 0
    last_val = {}
    try:
        for ep in range(1,args.epochs+1):
            current_epoch = ep
            runner.train_epoch(tr); val=runner.validate(va, metric="bleu"); last_val = val
            sch.step(); best=max(best, val["val_score"])
            print(f"[WMT14 {direction}] ep {ep}/{args.epochs} | BLEU {val['val_score']:.2f} | best {best:.2f}")
    except KeyboardInterrupt:
        dataset_name = f"wmt14-{direction}"
        interrupted_ckpt = _save_nlp_checkpoint("nlp_mt", dataset_name, args, model, opt, sch, current_epoch, best, {"val_summary": last_val})
        print(f"[WMT14 {direction}] 'exit' 신호 감지 → 체크포인트 저장: {interrupted_ckpt}")
    total_sec=time.perf_counter()-wall0
    loss_var=float(torch.tensor(runner.losses_epoch).var().item() if runner.losses_epoch else 0.0)
    summary = {
        "task":"mt", "dataset":f"wmt14-{direction}", "model":mdl_name, "optimizer":args.optimizer,
        "ttt_sec": runner.ttt_sec, "perf": best, "task_score": best,
        "speed": percentile(runner.step_times,50), "throughput": runner._throughput(),
        "hp_sensitivity_sd": None, "gen_gap": None, "loss_var": loss_var,
        "peak_mem_mb": human_mb(torch.cuda.max_memory_allocated() if torch.cuda.is_available() else 0),
        "total_time_sec": total_sec
    }
    return _attach_interrupt_meta(summary, interrupted_ckpt)

def run_xsum(args):
    device="cuda" if torch.cuda.is_available() else "cpu"
    mdl_name = args.model_name or "facebook/bart-base"
    tok=AutoTokenizer.from_pretrained(mdl_name, use_fast=True)
    model=AutoModelForSeq2SeqLM.from_pretrained(mdl_name).to(device)
    tr, va = build_xsum_loaders(tok, args.batch_size, args.workers)
    base_rico, base_other = 5e-3, 3e-5
    lr=args.lr or (base_rico if args.optimizer=="rico" else base_other)
    opt=build_optimizer(model,args.optimizer,lr,args.wd,args); sch=make_epoch_scheduler(opt, args.epochs, args.warmup_ratio)
    runner=S2SRunner(model,tok,opt,sch,device,task_name="xsum",amp=args.amp,ttt_target=args.ttt_target,clip_grad_norm=args.clip_grad_norm,max_gen_len=128)

    if torch.cuda.is_available(): torch.cuda.reset_peak_memory_stats()

    best=-1e9; wall0=time.perf_counter()
    interrupted_ckpt = None
    current_epoch = 0
    last_val = {}
    try:
        for ep in range(1,args.epochs+1):
            current_epoch = ep
            runner.train_epoch(tr); val=runner.validate(va, metric="rouge"); last_val = val
            sch.step(); best=max(best, val["val_score"])
            print(f"[XSum] ep {ep}/{args.epochs} | ROUGE-L {val['val_score']:.2f} | best {best:.2f}")
    except KeyboardInterrupt:
        interrupted_ckpt = _save_nlp_checkpoint("nlp_summarization", "xsum", args, model, opt, sch, current_epoch, best, {"val_summary": last_val})
        print(f"[XSum] 'exit' 신호 감지 → 체크포인트 저장: {interrupted_ckpt}")
    total_sec=time.perf_counter()-wall0
    loss_var=float(torch.tensor(runner.losses_epoch).var().item() if runner.losses_epoch else 0.0)
    summary = {
        "task":"summ", "dataset":"xsum", "model":mdl_name, "optimizer":args.optimizer,
        "ttt_sec": runner.ttt_sec, "perf": best, "task_score": best,
        "speed": percentile(runner.step_times,50), "throughput": runner._throughput(),
        "hp_sensitivity_sd": None, "gen_gap": None, "loss_var": loss_var,
        "peak_mem_mb": human_mb(torch.cuda.max_memory_allocated() if torch.cuda.is_available() else 0),
        "total_time_sec": total_sec
    }
    return _attach_interrupt_meta(summary, interrupted_ckpt)

def run_lm(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    ds = _hf_load("wikitext", "wikitext-2-raw-v1")
    mdl_name = args.model_name or "distilgpt2"
    tok = AutoTokenizer.from_pretrained(mdl_name, use_fast=True)
    if tok.pad_token is None: tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(mdl_name).to(device)
    model.resize_token_embeddings(len(tok))

    def to_ids(split): return tok.encode("\n\n".join(t["text"] for t in ds[split]), add_special_tokens=False)
    tr_ids = to_ids("train"); va_ids = to_ids("validation")

    block = args.block_size
    trset = LMLoader(tr_ids, block); vaset = LMLoader(va_ids, block)
    trldr = DataLoader(trset, batch_size=args.batch_size, shuffle=True,  collate_fn=lm_collate, num_workers=args.workers, drop_last=True)
    valdr = DataLoader(vaset, batch_size=args.batch_size, shuffle=False, collate_fn=lm_collate, num_workers=args.workers)

    base_rico, base_other = 5e-4, 5e-5
    lr = args.lr or (base_rico if args.optimizer=="rico" else base_other)
    opt = build_optimizer(model, args.optimizer, lr, args.wd, args)
    sch = make_epoch_scheduler(opt, args.epochs, args.warmup_ratio)
    ttt_default = args.ttt_target if args.ttt_target is not None else 35.0
    runner = LMRunner(model,opt,sch,device,amp=args.amp,ttt_target=ttt_default,clip_grad_norm=args.clip_grad_norm)

    if torch.cuda.is_available(): torch.cuda.reset_peak_memory_stats()

    best=float("inf"); wall0=time.perf_counter()
    interrupted_ckpt = None
    current_epoch = 0
    last_val = {}
    try:
        for ep in range(1,args.epochs+1):
            current_epoch = ep
            runner.train_epoch(trldr)
            tr_sum={"train_loss": sum(runner.losses_epoch)/max(len(runner.losses_epoch),1),
                    "p50": percentile(runner.step_times,50), "p90":percentile(runner.step_times,90),
                    "p99":percentile(runner.step_times,99), "throughput_tok_s": runner._throughput()}
            val = runner.validate(valdr); last_val = val
            sch.step(); best=min(best, val["val_ppl"])
            peak = torch.cuda.max_memory_allocated() if torch.cuda.is_available() else 0
            print(f"[LM][WT2] ep {ep}/{args.epochs} | ppl {val['val_ppl']:.2f} | best {best:.2f} | p99 {tr_sum['p99']:.3f}s (peak {human_mb(peak)}MB)")
    except KeyboardInterrupt:
        interrupted_ckpt = _save_nlp_checkpoint("nlp_lm", "wikitext-2-raw", args, model, opt, sch, current_epoch, best, {"val_summary": last_val})
        print(f"[LM][WT2] 'exit' 신호 감지 → 체크포인트 저장: {interrupted_ckpt}")
    total_sec=time.perf_counter()-wall0
    loss_var=float(torch.tensor(runner.losses_epoch).var().item() if runner.losses_epoch else 0.0)
    summary = {
        "task":"lm", "dataset":"wikitext-2-raw", "model":mdl_name, "optimizer":args.optimizer,
        "ttt_sec": runner.ttt_sec, "perf": best, "task_score": best,
        "speed": percentile(runner.step_times,50), "throughput": runner._throughput(),
        "hp_sensitivity_sd": None, "gen_gap": None, "loss_var": loss_var,
        "peak_mem_mb": human_mb(torch.cuda.max_memory_allocated() if torch.cuda.is_available() else 0),
        "total_time_sec": total_sec
    }
    return _attach_interrupt_meta(summary, interrupted_ckpt)

def run_gsm8k(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    mdl_name = args.model_name or "gpt2"

    tok = AutoTokenizer.from_pretrained(mdl_name, use_fast=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token or tok.unk_token
    model = AutoModelForCausalLM.from_pretrained(mdl_name).to(device)
    model.config.pad_token_id = tok.pad_token_id
    model.resize_token_embeddings(len(tok))

    tr, va = build_gsm8k_loaders(mdl_name, args.batch_size, args.workers)
    base_rico, base_other = 1e-3, 3e-5
    lr = args.lr or (base_rico if args.optimizer == "rico" else base_other)
    opt = build_optimizer(model, args.optimizer, lr, args.wd, args)
    sch = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    runner = MathRunner(model, tok, opt, sch, device,
                        amp=args.amp, ttt_target=args.ttt_target,
                        clip_grad_norm=args.clip_grad_norm)

    if torch.cuda.is_available(): torch.cuda.reset_peak_memory_stats()

    best = -1e9; wall0 = time.perf_counter()
    interrupted_ckpt = None
    current_epoch = 0
    last_val = {}
    try:
        for ep in range(1, args.epochs+1):
            current_epoch = ep
            runner.train_epoch(tr)
            val = runner.validate(va); last_val = val
            sch.step(); best = max(best, val["val_score"])
            print(f"[GSM8K] ep {ep}/{args.epochs} | acc {val['val_score']:.2f} | best {best:.2f}")
    except KeyboardInterrupt:
        interrupted_ckpt = _save_nlp_checkpoint("nlp_math", "gsm8k", args, model, opt, sch, current_epoch, best, {"val_summary": last_val})
        print(f"[GSM8K] 'exit' 신호 감지 → 체크포인트 저장: {interrupted_ckpt}")
    total_sec = time.perf_counter() - wall0
    loss_var = float(torch.tensor(runner.losses_epoch).var().item() if runner.losses_epoch else 0.0)
    summary = {
        "task":"math", "dataset":"gsm8k", "model":mdl_name, "optimizer":args.optimizer,
        "ttt_sec": runner.ttt_sec, "perf": best, "task_score": best,
        "speed": percentile(runner.step_times,50), "throughput": runner._throughput(),
        "hp_sensitivity_sd": None, "gen_gap": None, "loss_var": loss_var,
        "peak_mem_mb": human_mb(torch.cuda.max_memory_allocated() if torch.cuda.is_available() else 0),
        "total_time_sec": total_sec
    }
    return _attach_interrupt_meta(summary, interrupted_ckpt)

# ===== HP 감도 스윕 (3x3) =====
def hp_sweep(run_fn, args, base_lr:float, base_wd:float) -> float:
    lmults=[0.5,1.0,2.0]; wmults=[0.5,1.0,2.0]; scores=[]
    for lm in lmults:
        for wm in wmults:
            a = argparse.Namespace(**vars(args))
            a.lr = base_lr*lm if base_lr is not None else None
            a.wd = base_wd*wm
            print(f"[HP-SWEEP] lr={a.lr} wd={a.wd}")
            summ = run_fn(a)
            scores.append(summ["perf"])
    return float(statistics.pstdev(scores)) if len(scores)>1 else None

# ===== CLI =====
def main():
    ap = argparse.ArgumentParser("GOAT-style NLP bench (RICO/AdamW/Lion/SOAP)")
    ap.add_argument("--task", type=str, required=False,
                    help="Tasks: sst2|mrpc|stsb|qqp|ag_news|boolq|rte|cb|anli|hellaswag|piqa|copa|winogrande|alpha_nli|squad_v2|wmt14_en_de|wmt14_de_en|xsum|lm|gsm8k|suite:easy|suite:medium|suite:hard")
    ap.add_argument("--model-name", type=str, default=None)
    ap.add_argument("--optimizer", type=str, required=True, choices=["rico","adamw","lion","soap"])
    ap.add_argument("--lr", type=float, default=None)
    ap.add_argument("--wd", type=float, default=1e-2)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--amp", type=str, default="none", choices=["none","fp16","bf16"])
    ap.add_argument("--max-length", type=int, default=128)
    ap.add_argument("--block-size", type=int, default=512)
    ap.add_argument("--ttt-target", type=float, default=None)
    ap.add_argument("--clip-grad-norm", type=float, default=1.0)
    ap.add_argument("--log-csv", type=str, default=None)
    ap.add_argument("--log-json", type=str, default=None)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--hp-sweep", action="store_true", help="3x3 lr/wd sweep로 hp_sensitivity_sd 산출")

    # RICO knobs
    ap.add_argument("--rico_bk_beta", type=float, default=0.9)
    ap.add_argument("--rico_k_cap", type=float, default=0.08)
    ap.add_argument("--rico_g_floor", type=float, default=1e-5)  # 작은 grad도 움직이도록
    ap.add_argument("--rico_sync_every", type=int, default=20)    # 미세 과업은 매 step 동기화 권장
    # FT 모드 토글 (기본 ON)
    ap.add_argument("--rico_ft", dest="rico_ft", action="store_true")
    ap.add_argument("--no-rico_ft", dest="rico_ft", action="store_false")
    ap.set_defaults(rico_ft=True)

    # epoch-warmup 비율 & 정체구제
    ap.add_argument("--warmup_ratio", type=float, default=0.1)
    ap.add_argument("--rescue_lr", action="store_true", default=True)

    # Lion/SOAP
    ap.add_argument("--lion_beta1", type=float, default=0.9)
    ap.add_argument("--lion_beta2", type=float, default=0.99)
    ap.add_argument("--soap_args", type=str, default=None)

    args = ap.parse_args()
    set_seed(args.seed)

    suites = {
        "suite:easy":   ["sst2","mrpc","hellaswag","piqa","ag_news","lm"],
        "suite:medium": ["boolq","rte","cb","copa","anli","squad_v2","xsum"],
        "suite:hard":   ["wmt14_en_de","gsm8k"]
    }

    tasks = [args.task] if args.task and not args.task.startswith("suite:") else suites.get(args.task, [])
    if not tasks:
        raise SystemExit("반드시 --task (또는 --task suite:easy|suite:medium|suite:hard) 지정 바랍니다.")

    all_summaries=[]

    def _dispatch(a, t):
        t=t.lower()
        if t in ("sst2","mrpc","stsb","qqp","ag_news","boolq","rte","cb","anli"):
            return run_glue_superglue(a, t)
        if t in ("hellaswag","piqa","copa","winogrande","alpha_nli"):
            return run_mcq(a, t)
        if t=="squad_v2":
            return run_qa_squad2(a)
        if t=="wmt14_en_de":
            return run_wmt14(a, "en-de")
        if t=="wmt14_de_en":
            return run_wmt14(a, "de-en")
        if t=="xsum":
            return run_xsum(a)
        if t=="lm":
            return run_lm(a)
        if t=="gsm8k":
            return run_gsm8k(a)
        raise ValueError(f"지원하지 않는 task: {t}")

    for t in tasks:
        base_lr = args.lr
        base_wd = args.wd
        if args.hp_sweep:
            sd = hp_sweep(lambda aa: _dispatch(aa, t), args, base_lr, base_wd)
            print(f"[HP-SWEEP RESULT] {t}: hp_sensitivity_sd={sd}")
        summ = _dispatch(args, t); summ["hp_sensitivity_sd"] = summ.get("hp_sensitivity_sd")
        all_summaries.append(summ)

        # 🔧 suite 실행 시 메모리 누수 방지 (각 태스크 사이)
        import gc
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    if args.log_json:
        out = {"runs": all_summaries}
        Path(args.log_json).parent.mkdir(parents=True, exist_ok=True)
        with open(args.log_json, "w") as f: f.write(json.dumps(out, indent=2))
    print("[SUMMARY][ALL]", json.dumps({"runs": all_summaries}, indent=2))

# ---- epoch warmup + cosine (epoch 단위) ----
def make_epoch_scheduler(opt, epochs:int, warmup_ratio:float):
    import math as _m
    if warmup_ratio <= 0:
        return optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    w = max(1, int(_m.ceil(epochs * warmup_ratio)))
    def _lambda(ep_idx: int):
        # ep_idx: 0,1,2,... (Trainer에서 epoch 끝에 step() 호출)
        if ep_idx < w:
            return float(ep_idx + 1) / float(w)
        t = float(ep_idx - w + 1) / max(epochs - w, 1)
        return 0.5 * (1.0 + _m.cos(_m.pi * t))
    return optim.lr_scheduler.LambdaLR(opt, lr_lambda=_lambda)

if __name__ == "__main__":
    main()
