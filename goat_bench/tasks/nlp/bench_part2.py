from __future__ import annotations

# bench_nlp_part2.py — Heavy/Complex NLP benchmark (MultiRC, ReCoRD, HotpotQA; plugs: NQ-open, BBH)
# pip install datasets transformers evaluate sacrebleu rouge-score
import argparse, time, json, math, random
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional
import torch, torch.nn as nn, torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from datasets import load_dataset
from transformers import (
    AutoTokenizer, AutoModelForSequenceClassification,
    AutoModelForMultipleChoice, AutoModelForQuestionAnswering,
    DataCollatorWithPadding
)

# ----- optional metrics -----
try:
    import evaluate as _evaluate
except Exception:
    _evaluate = None

# ----- third-party optimizers (optional) -----
LionOpt = None; SOAPOpt = None
try:
    from lion_pytorch import Lion as LionOpt
except Exception:
    pass
try:
    import pytorch_optimizer as _pyo
    SOAPOpt = getattr(_pyo, "SOAP", None)
except Exception:
    pass

# ----- RICO (optional) -----
try:
    from rico import RICO, rico_layerwise_groups
except Exception:
    RICO, rico_layerwise_groups = None, None

# ===== utils & optimizer builder (match Part-1) =====
def set_seed(seed:int):
    random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)

def percentile(xs: List[float], q: float) -> float:
    if not xs: return float("nan")
    ys = sorted(xs); k = (len(ys)-1)*(q/100.0)
    f, c = math.floor(k), math.ceil(k)
    return ys[int(k)] if f==c else ys[f] + (ys[c]-ys[f])*(k-f)

def human_mb(x:int) -> float: return round(x/(1024*1024), 2)

def split_decay_params(model, wd: float):
    decay, nodecay = [], []
    no_decay_keys = ("bias","LayerNorm.weight","layer_norm.weight","ln.weight","norm.weight","bn.weight","bn.bias")
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
    kw = dict(bk_beta_target=args.rico_bk_beta, k_cap=args.rico_k_cap, g_rms_floor=args.rico_g_floor, sync_every=args.rico_sync_every)
    try:
        return RICO(pg, lr=lr, ft_mode=False, weight_decay=wd, wd_mode="decoupled", **kw)
    except TypeError:
        return RICO(pg, lr=lr, ft_mode=False, weight_decay=wd, wd_mode="decoupled")

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
            import json as _json
            try: extra = _json.loads(args.soap_args)
            except Exception as e: print(f"[warn] --soap_args 파싱 실패: {e}")
        return SOAPOpt(pg, lr=lr, weight_decay=wd, **extra)
    raise ValueError(f"unknown optimizer {name}")

# ===== Base Trainer (GOAT hooks) =====
class TrainerBase:
    def __init__(self, model, optimizer, scheduler, device, amp="none",
                 higher_is_better=True, ttt_target=None, clip_grad_norm=None):
        self.model, self.opt, self.sch = model, optimizer, scheduler
        self.device = device
        self.amp = amp if torch.cuda.is_available() else "none"
        self.autocast_dtype = torch.bfloat16 if self.amp=="bf16" else torch.float16
        self.scaler = torch.cuda.amp.GradScaler(enabled=(self.amp=="fp16"))
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

# ===== MultiRC (SuperGLUE) =====
# 약식: 정식 metric은 evaluate(super_glue, "multirc")가 있으면 사용. 없으면 answer-level macro F1로 근사.
class MultiRCRunner(TrainerBase):
    def __init__(self, model, tok, optimizer, scheduler, device, amp="none", ttt_target=None, clip_grad_norm=None):
        super().__init__(model, optimizer, scheduler, device, amp, True, ttt_target, clip_grad_norm)
        self.tok = tok
        self.metric = _evaluate.load("super_glue","multirc") if _evaluate else None

    def train_epoch(self, loader):
        self.model.train()
        self.step_times.clear(); self.losses_epoch.clear(); self.seen_tokens_epoch=0
        pbar = tqdm(loader, desc="[MultiRC][Train]")
        for batch in pbar:
            t0=time.perf_counter()
            batch = {k: v.to(self.device) for k,v in batch.items() if k not in ("qid","aid")}
            self.opt.zero_grad(set_to_none=True)
            if self.amp in ("fp16","bf16"):
                with torch.autocast(device_type="cuda", dtype=self.autocast_dtype):
                    out=self.model(**batch); loss=out.loss
                if self.amp=="fp16":
                    self.scaler.scale(loss).backward()
                    if self.clip_grad_norm:
                        self.scaler.unscale_(self.opt); torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip_grad_norm)
                    self.scaler.step(self.opt); self.scaler.update()
                else:
                    loss.backward()
                    if self.clip_grad_norm: torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip_grad_norm)
                    self.opt.step()
            else:
                out=self.model(**batch); loss=out.loss
                loss.backward()
                if self.clip_grad_norm: torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip_grad_norm)
                self.opt.step()
            dt=time.perf_counter()-t0
            self.step_times.append(dt); self.losses_epoch.append(float(loss.detach().cpu()))
            self._mark_seen_tokens(batch["input_ids"].ne(self.tok.pad_token_id).sum().item())

    @torch.no_grad()
    def validate(self, loader):
        self.model.eval()
        preds, refs = [], []
        gold, hat = [], []
        for batch in tqdm(loader, desc="[MultiRC][Val]"):
            qid = batch["qid"]; aid = batch["aid"]
            feat = {k: v.to(self.device) for k,v in batch.items() if k not in ("qid","aid")}
            with torch.autocast(device_type="cuda", dtype=self.autocast_dtype) if self.amp in ("fp16","bf16") else torch.no_grad():
                out = self.model(**feat)
            p = (out.logits.squeeze(-1)>0).long().cpu().tolist()
            y = feat["labels"].cpu().tolist()
            # for approx macro-F1
            gold += y; hat += p
            # for official evaluate
            for qi,ai,yi,pi in zip(qid,aid,y,p):
                preds.append({"idx": {"question": int(qi), "answer": int(ai)}, "label": int(pi)})
                refs.append({"idx": {"question": int(qi), "answer": int(ai)}, "label": int(yi)})

        if self.metric:
            res = self.metric.compute(predictions=preds, references=refs)
            score = 100.0*res.get("f1_a", res.get("f1", 0.0))
        else:
            # macro F1 over answers (approximate)
            tp = sum(1 for g,p in zip(gold,hat) if g==1 and p==1)
            fp = sum(1 for g,p in zip(gold,hat) if g==0 and p==1)
            fn = sum(1 for g,p in zip(gold,hat) if g==1 and p==0)
            prec = tp/max(tp+fp,1); rec = tp/max(tp+fn,1)
            f1 = 0.0 if prec+rec==0 else 200*prec*rec/max(prec+rec,1e-12)
            score = f1
        self._maybe_ttt(score); self._update_best(score)
        return {"val_score": score}

# ===================== PATCH A: MultiRC (robust loader) =====================
def build_multirc_loaders(mdl_name:str, batch:int, workers:int, max_len:int=256):
    """
    - batched=False 로 flatten (정석)
    - 각 보기(answers[*])를 이진 분류 샘플로 전개
    """
    ds = load_dataset("super_glue","multirc")
    tok = AutoTokenizer.from_pretrained(mdl_name, use_fast=True)

    def _encode_single(ex):
        out = []
        qid = int(ex["idx"]["question"])
        para = ex["paragraph"]
        ques = ex["question"]
        for aid, ans in enumerate(ex["answers"]):
            text_ans = ans.get("text","")
            label = int(ans.get("label", 0))
            enc = tok(ques + " " + text_ans, para, truncation=True, max_length=max_len)
            enc["labels"] = label
            enc["qid"] = qid
            enc["aid"] = int(aid)
            out.append(enc)
        return out

    tr = ds["train"].map(_encode_single, batched=False, remove_columns=ds["train"].column_names)
    va = ds["validation"].map(_encode_single, batched=False, remove_columns=ds["validation"].column_names)

    col = DataCollatorWithPadding(tokenizer=tok)
    class DS(torch.utils.data.Dataset):
        def __init__(self, hf): self.hf=hf
        def __len__(self): return len(self.hf)
        def __getitem__(self,i):
            item = {k: torch.tensor(self.hf[i][k]) for k in ("input_ids","attention_mask","labels")}
            item["qid"] = int(self.hf[i]["qid"])
            item["aid"] = int(self.hf[i]["aid"])
            return item

    return tok, DataLoader(DS(tr), batch_size=batch, shuffle=True,  num_workers=workers,
                           collate_fn=col, pin_memory=True, persistent_workers=(workers>0)), \
               DataLoader(DS(va), batch_size=batch*2, shuffle=False, num_workers=workers,
                           collate_fn=col, pin_memory=True, persistent_workers=(workers>0))

# ===== ReCoRD (SuperGLUE): entity cloze → MultipleChoice로 처리 =====
class ReCORDRunner(TrainerBase):
    def __init__(self, model, tok, optimizer, scheduler, device, amp="none", ttt_target=None, clip_grad_norm=None):
        super().__init__(model, optimizer, scheduler, device, amp, True, ttt_target, clip_grad_norm)
        self.tok = tok
        self.metric = _evaluate.load("super_glue","record") if _evaluate else None

    def train_epoch(self, loader):
        self.model.train()
        self.step_times.clear(); self.losses_epoch.clear(); self.seen_tokens_epoch=0
        pbar=tqdm(loader, desc="[ReCoRD][Train]")
        for stems, choices, labels in pbar:
            t0=time.perf_counter()
            self.opt.zero_grad(set_to_none=True)
            # pack MCQ
            B = len(labels); C = max(len(c) for c in choices)
            ids=[]; attn=[]
            for i in range(B):
                enc=[]
                for c in choices[i]:
                    enc.append(self.tok(stems[i].replace("@placeholder", c), truncation=True, max_length=256))
                # pad to C
                while len(enc)<C: enc.append(enc[-1])
                L = max(len(e["input_ids"]) for e in enc)
                ids.append(torch.stack([torch.tensor(e["input_ids"] + [self.tok.pad_token_id]*(L-len(e["input_ids"]))) for e in enc],0))
                attn.append(torch.stack([torch.tensor(e["attention_mask"] + [0]*(L-len(e["attention_mask"]))) for e in enc],0))
            input_ids = torch.stack(ids,0).long().to(self.device)
            attention_mask = torch.stack(attn,0).long().to(self.device)
            labels = labels.to(self.device)
            if self.amp in ("fp16","bf16"):
                with torch.autocast(device_type="cuda", dtype=self.autocast_dtype):
                    out = self.model(input_ids=input_ids, attention_mask=attention_mask, labels=labels); loss=out.loss
                if self.amp=="fp16":
                    self.scaler.scale(loss).backward()
                    if self.clip_grad_norm:
                        self.scaler.unscale_(self.opt); torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip_grad_norm)
                    self.scaler.step(self.opt); self.scaler.update()
                else:
                    loss.backward()
                    if self.clip_grad_norm: torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip_grad_norm)
                    self.opt.step()
            else:
                out = self.model(input_ids=input_ids, attention_mask=attention_mask, labels=labels); loss=out.loss
                loss.backward()
                if self.clip_grad_norm: torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip_grad_norm)
                self.opt.step()
            dt=time.perf_counter()-t0
            self.step_times.append(dt); self.losses_epoch.append(float(loss.detach().cpu()))
            self._mark_seen_tokens(attention_mask.sum().item())

    @torch.no_grad()
    def validate(self, loader):
        self.model.eval()
        accs=[]
        for stems, choices, labels in tqdm(loader, desc="[ReCoRD][Val]"):
            # build same batch
            B = len(labels); C = max(len(c) for c in choices)
            ids=[]; attn=[]
            for i in range(B):
                enc=[]
                for c in choices[i]:
                    enc.append(self.tok(stems[i].replace("@placeholder", c), truncation=True, max_length=256))
                while len(enc)<C: enc.append(enc[-1])
                L = max(len(e["input_ids"]) for e in enc)
                ids.append(torch.stack([torch.tensor(e["input_ids"] + [self.tok.pad_token_id]*(L-len(e["input_ids"]))) for e in enc],0))
                attn.append(torch.stack([torch.tensor(e["attention_mask"] + [0]*(L-len(e["attention_mask"]))) for e in enc],0))
            input_ids = torch.stack(ids,0).long().to(self.device)
            attention_mask = torch.stack(attn,0).long().to(self.device)
            with torch.autocast(device_type="cuda", dtype=self.autocast_dtype) if self.amp in ("fp16","bf16") else torch.no_grad():
                out = self.model(input_ids=input_ids, attention_mask=attention_mask)
            pred = out.logits.argmax(-1).cpu()
            accs += (pred==labels).int().tolist()
        score = 100.0*sum(accs)/max(len(accs),1)
        self._maybe_ttt(score); self._update_best(score)
        return {"val_score": score}


def build_record_loaders(batch:int, workers:int):
    """
    - SuperGLUE ReCoRD를 '단일 정답이 엔티티 목록에 존재'하는 샘플로 필터
    - stem = query + ' [SEP] ' + passage (런타임에서 @placeholder 대체)
    - 라벨은 정답 텍스트의 첫 매칭 엔티티 인덱스
    """
    ds = load_dataset("super_glue","record")

    def _prepare(split):
        d = ds[split]
        stems, choices, labels = [], [], []
        cols = d.column_names
        for i in range(len(d)):
            passage = d[i]["passage"]
            query   = d[i]["query"] if "query" in cols else d[i].get("query","")
            ents    = d[i]["entities"]  # list[str]
            # answers: dict(text=[...], ... ) 혹은 list[str]
            raw_ans = d[i]["answers"]
            ans_texts = raw_ans.get("text", []) if isinstance(raw_ans, dict) else raw_ans
            ans_texts = ans_texts if isinstance(ans_texts, list) else [ans_texts]

            # 엔티티 목록에 존재하는 첫 정답 하나만 사용 (여러개면 첫 매칭)
            lab = -1
            for t in ans_texts:
                if t in ents:
                    lab = ents.index(t); break
            if lab < 0:  # 매칭 실패시 스킵
                continue

            stems.append((query or "@placeholder").strip() + " [SEP] " + passage)
            choices.append(list(ents))
            labels.append(lab)

        return stems, choices, torch.tensor(labels, dtype=torch.long)

    class DS(torch.utils.data.Dataset):
        def __init__(self, split):
            self.stems, self.choices, self.labels = _prepare(split)
        def __len__(self): return len(self.labels)
        def __getitem__(self, i): return self.stems[i], self.choices[i], self.labels[i]

    def _collate(batch):
        stems, choices, labels = zip(*batch)
        return list(stems), list(choices), torch.stack(labels, 0)

    tr = DataLoader(DS("train"),      batch_size=batch, shuffle=True,  num_workers=workers,
                    pin_memory=True, collate_fn=_collate, persistent_workers=(workers>0))
    va = DataLoader(DS("validation"), batch_size=batch, shuffle=False, num_workers=workers,
                    pin_memory=True, collate_fn=_collate, persistent_workers=(workers>0))
    return tr, va

# ===== HotpotQA (distractor) span-QA with official offsets =====
class HotpotRunner(TrainerBase):
    def __init__(self, model, tok, optimizer, scheduler, device, amp="none", ttt_target=None, clip_grad_norm=None, max_len=384, stride=128):
        super().__init__(model, optimizer, scheduler, device, amp, True, ttt_target, clip_grad_norm)
        self.tok=tok; self.max_len=max_len; self.stride=stride

    def train_epoch(self, loader):
        self.model.train()
        self.step_times.clear(); self.losses_epoch.clear(); self.seen_tokens_epoch=0
        pbar=tqdm(loader, desc="[HotpotQA][Train]")
        for batch in pbar:
            t0=time.perf_counter()
            feat={k: v.to(self.device) for k,v in batch.items()}
            self.opt.zero_grad(set_to_none=True)
            if self.amp in ("fp16","bf16"):
                with torch.autocast(device_type="cuda", dtype=self.autocast_dtype):
                    out=self.model(**feat); loss=out.loss
                if self.amp=="fp16":
                    self.scaler.scale(loss).backward()
                    if self.clip_grad_norm:
                        self.scaler.unscale_(self.opt); torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip_grad_norm)
                    self.scaler.step(self.opt); self.scaler.update()
                else:
                    loss.backward()
                    if self.clip_grad_norm: torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip_grad_norm)
                    self.opt.step()
            else:
                out=self.model(**feat); loss=out.loss
                loss.backward()
                if self.clip_grad_norm: torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip_grad_norm)
                self.opt.step()
            dt=time.perf_counter()-t0
            self.step_times.append(dt); self.losses_epoch.append(float(loss.detach().cpu()))
            self._mark_seen_tokens(feat["attention_mask"].sum().item())

    @torch.no_grad()
    def validate(self, loader):
        self.model.eval()
        ems=[]; f1s=[]
        def _norm(s): return " ".join(s.lower().split())
        for batch in tqdm(loader, desc="[HotpotQA][Val]"):
            feat={k: v.to(self.device) for k,v in batch.items() if k not in ("answers_text",)}
            with torch.autocast(device_type="cuda", dtype=self.autocast_dtype) if self.amp in ("fp16","bf16") else torch.no_grad():
                out=self.model(**feat)
            s=out.start_logits.argmax(-1); e=out.end_logits.argmax(-1)
            for i in range(s.size(0)):
                ids = batch["input_ids"][i]
                si, ei = int(min(s[i], e[i])), int(max(s[i], e[i]))
                pred = self.tok.decode(ids[si:ei+1], skip_special_tokens=True)
                golds = batch["answers_text"][i]
                em = 1.0 if any(_norm(pred)==_norm(g) for g in golds) else 0.0
                def f1(a,b):
                    wa=_norm(a).split(); wb=_norm(b).split()
                    if not wa and not wb: return 1.0
                    common = len(set(wa)&set(wb))
                    if common==0: return 0.0
                    prec = common/max(len(wa),1); rec = common/max(len(wb),1);
                    return 2*prec*rec/max(prec+rec,1e-12)
                f1s.append(max([f1(pred,g) for g in golds]) if golds else 0.0); ems.append(em)
        em = 100.0*sum(ems)/max(len(ems),1); f1 = 100.0*sum(f1s)/max(len(f1s),1)
        score = 0.5*(em+f1)
        self._maybe_ttt(score); self._update_best(score)
        return {"val_score": score, "em": em, "f1": f1}

def build_hotpot_loaders(mdl_name:str, batch:int, workers:int, max_len:int=384, doc_stride:int=128):
    """
    - context 스키마가 리스트( [title, sentences] )/dict 모두 오케이
    - 정답 문자열이 문서에 없으면 start/end=0으로 대체 (train '그냥 돌아가게')
    - collator 커스텀: answers_text 유지
    """
    tok = AutoTokenizer.from_pretrained(mdl_name, use_fast=True)
    ds = load_dataset("hotpot_qa","distractor")

    def _concat_context(ctx):
        # ctx: list[[title, sentences], ...]  또는 dict(sentences=[...]) 등
        if isinstance(ctx, dict) and "sentences" in ctx:
            return " ".join(ctx.get("sentences", []))
        parts = []
        if isinstance(ctx, list):
            for it in ctx:
                if isinstance(it, dict) and "sentences" in it:
                    parts.append(" ".join(it.get("sentences", [])))
                elif isinstance(it, (list, tuple)) and len(it) >= 2:
                    sents = it[1] if isinstance(it[1], list) else []
                    parts.append(" ".join(sents))
        return " ".join(parts)

    def _encode_single(ex):
        question = ex["question"]
        ctx_text = _concat_context(ex["context"])
        answer   = ex.get("answer", "")

        enc = tok(
            question, ctx_text,
            truncation="only_second", max_length=max_len, stride=doc_stride,
            return_offsets_mapping=True
        )
        # fast tokenizer 전제
        enc0 = enc.encodings[0]
        offsets = enc0.offsets
        seq_ids = enc0.sequence_ids
        start_pos = end_pos = 0
        if isinstance(answer, str) and len(answer) > 0:
            # char-span 찾기 (case-insensitive)
            a_low = answer.lower()
            c_low = ctx_text.lower()
            start_char = c_low.find(a_low)
            if start_char >= 0:
                end_char = start_char + len(answer)
                # 두 번째 시퀀스(=context) 토큰 중 겹치는 첫/마지막 토큰 찾기
                token_start = token_end = None
                for i, (sid, (s, e)) in enumerate(zip(seq_ids, offsets)):
                    if sid != 1:  # 1 == context
                        continue
                    if s is None or e is None:  # special tokens
                        continue
                    # overlap?
                    if token_start is None and s <= start_char < e:
                        token_start = i
                    if token_start is not None and s < end_char <= e:
                        token_end = i
                        break
                if token_start is not None and token_end is not None:
                    start_pos, end_pos = token_start, token_end

        # offset_mapping은 저장 안 함 (collator 충돌 방지)
        feat = {k: v for k, v in enc.items() if k != "offset_mapping"}
        feat["start_positions"] = start_pos
        feat["end_positions"] = end_pos
        feat["answers_text"] = [answer] if isinstance(answer, str) else []
        return feat

    tr = ds["train"].map(_encode_single, batched=False, remove_columns=ds["train"].column_names)
    va = ds["validation"].map(_encode_single, batched=False, remove_columns=ds["validation"].column_names)

    def _collate_hotpot(features):
        # 표준 패딩 + answers_text 유지
        keys = ("input_ids","attention_mask","start_positions","end_positions")
        batch = {}
        # pad-able
        pad_feats = [{k:v for k,v in f.items() if k in ("input_ids","attention_mask")} for f in features]
        pad_out = DataCollatorWithPadding(tokenizer=tok)(pad_feats)
        batch.update(pad_out)
        # positions
        batch["start_positions"] = torch.tensor([f["start_positions"] for f in features], dtype=torch.long)
        batch["end_positions"]   = torch.tensor([f["end_positions"]   for f in features], dtype=torch.long)
        # keep answers
        batch["answers_text"] = [f.get("answers_text", []) for f in features]
        return batch

    class DS(torch.utils.data.Dataset):
        def __init__(self, hf): self.hf=hf
        def __len__(self): return len(self.hf)
        def __getitem__(self, i): return dict(self.hf[i])

    return tok, DataLoader(DS(tr), batch_size=batch, shuffle=True,  num_workers=workers,
                           collate_fn=_collate_hotpot, pin_memory=True, persistent_workers=(workers>0)), \
               DataLoader(DS(va), batch_size=batch, shuffle=False, num_workers=workers,
                           collate_fn=_collate_hotpot, pin_memory=True, persistent_workers=(workers>0))
# ===== Dispatchers =====
def run_multirc(args):
    device="cuda" if torch.cuda.is_available() else "cpu"
    mdl = args.model_name or "roberta-base"
    tok, tr, va = build_multirc_loaders(mdl, args.batch_size, args.workers, max_len=args.max_length)
    model = AutoModelForSequenceClassification.from_pretrained(mdl, num_labels=1, problem_type="single_label_classification").to(device)
    base_rico, base_other = 1e-2, 3e-5
    lr = args.lr or (base_rico if args.optimizer=="rico" else base_other)
    opt=build_optimizer(model,args.optimizer,lr,args.wd,args); sch=optim.lr_scheduler.CosineAnnealingLR(opt,T_max=args.epochs)
    runner=MultiRCRunner(model,tok,opt,sch,device,amp=args.amp,ttt_target=args.ttt_target,clip_grad_norm=args.clip_grad_norm)
    best=-1e9; wall0=time.perf_counter()
    for ep in range(1,args.epochs+1):
        runner.train_epoch(tr); val=runner.validate(va); sch.step(); best=max(best, val["val_score"])
        peak=torch.cuda.max_memory_allocated() if torch.cuda.is_available() else 0
        print(f"[MultiRC] ep {ep}/{args.epochs} | score {val['val_score']:.2f} | best {best:.2f} | peak {human_mb(peak)}MB")
    return {"task":"multirc","perf":best,"ttt_sec":runner.ttt_sec}

def run_record(args):
    device="cuda" if torch.cuda.is_available() else "cpu"
    mdl = args.model_name or "roberta-base"
    tok = AutoTokenizer.from_pretrained(mdl, use_fast=True)
    model = AutoModelForMultipleChoice.from_pretrained(mdl).to(device)
    tr, va = build_record_loaders(args.batch_size, args.workers)
    base_rico, base_other = 5e-3, 3e-5
    lr=args.lr or (base_rico if args.optimizer=="rico" else base_other)
    opt=build_optimizer(model,args.optimizer,lr,args.wd,args); sch=optim.lr_scheduler.CosineAnnealingLR(opt,T_max=args.epochs)
    runner=ReCORDRunner(model,tok,opt,sch,device,amp=args.amp,ttt_target=args.ttt_target,clip_grad_norm=args.clip_grad_norm)
    best=-1e9
    for ep in range(1,args.epochs+1):
        runner.train_epoch(tr); val=runner.validate(va); sch.step(); best=max(best, val["val_score"])
        print(f"[ReCoRD] ep {ep}/{args.epochs} | acc {val['val_score']:.2f} | best {best:.2f}")
    return {"task":"record","perf":best,"ttt_sec":runner.ttt_sec}

def run_hotpot(args):
    device="cuda" if torch.cuda.is_available() else "cpu"
    mdl = args.model_name or "bert-base-uncased"
    tok, tr, va = build_hotpot_loaders(mdl, args.batch_size, args.workers, max_len=args.max_length, doc_stride=128)
    model = AutoModelForQuestionAnswering.from_pretrained(mdl).to(device)
    base_rico, base_other = 5e-3, 3e-5
    lr=args.lr or (base_rico if args.optimizer=="rico" else base_other)
    opt=build_optimizer(model,args.optimizer,lr,args.wd,args); sch=optim.lr_scheduler.CosineAnnealingLR(opt,T_max=args.epochs)
    runner=HotpotRunner(model,tok,opt,sch,device,amp=args.amp,ttt_target=args.ttt_target,clip_grad_norm=args.clip_grad_norm)
    best=-1e9
    for ep in range(1,args.epochs+1):
        runner.train_epoch(tr); val=runner.validate(va); sch.step(); best=max(best, val["val_score"])
        print(f"[HotpotQA] ep {ep}/{args.epochs} | score {val['val_score']:.2f} | best {best:.2f}")
    return {"task":"hotpotqa","perf":best,"ttt_sec":runner.ttt_sec}

# ---- NQ-open / BBH plugs (stub) ----
def run_nq_open(args):
    raise SystemExit("NQ-open: 리트리벌 인덱스가 필요합니다. 다음 단계에서 BM25/FAISS 플러그인을 붙입니다.")

def run_bbh(args):
    raise SystemExit("BBH: 로컬 JSONL 경로를 받아 정확도(EM) 평가 하니스를 다음 단계에서 연결합니다.")

# ===== CLI =====
def main():
    ap = argparse.ArgumentParser("NLP Heavy bench (MultiRC/ReCoRD/Hotpot; NQ-open/BBH plugs)")
    ap.add_argument("--task", type=str, required=True, choices=["multirc","record","hotpotqa","nq_open","bbh"])
    ap.add_argument("--model-name", type=str, default=None)
    ap.add_argument("--optimizer", type=str, required=True, choices=["rico","adamw","lion","soap"])
    ap.add_argument("--lr", type=float, default=None)
    ap.add_argument("--wd", type=float, default=1e-2)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--amp", type=str, default="bf16", choices=["none","fp16","bf16"])
    ap.add_argument("--max-length", type=int, default=256)
    ap.add_argument("--ttt-target", type=float, default=None)
    ap.add_argument("--clip-grad-norm", type=float, default=1.0)
    # RICO knobs
    ap.add_argument("--rico_bk_beta", type=float, default=0.9)
    ap.add_argument("--rico_k_cap", type=float, default=0.08)
    ap.add_argument("--rico_g_floor", type=float, default=1e-3)
    ap.add_argument("--rico_sync_every", type=int, default=20)
    # Lion/SOAP
    ap.add_argument("--lion_beta1", type=float, default=0.9)
    ap.add_argument("--lion_beta2", type=float, default=0.99)
    ap.add_argument("--soap_args", type=str, default=None)

    args = ap.parse_args(); set_seed(42)

    dispatch = {
        "multirc": run_multirc,
        "record": run_record,
        "hotpotqa": run_hotpot,
        "nq_open": run_nq_open,
        "bbh": run_bbh,
    }
    summ = dispatch[args.task](args)
    print("[SUMMARY]", json.dumps(summ, indent=2))

if __name__ == "__main__":
    main()
