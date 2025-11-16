# goat_bench/optimizers/builder.py
from __future__ import annotations

import json
from typing import Any, Dict
import torch.optim as optim

try:
    from .rico import RICO, rico_layerwise_groups
except Exception:
    RICO, rico_layerwise_groups = None, None

try:
    from lion_pytorch import Lion as LionOpt
except Exception:
    LionOpt = None

try:
    import pytorch_optimizer as _pyo

    SOAPOpt = getattr(_pyo, "SOAP", None)
except Exception:
    SOAPOpt = None


def _opt_attr(options: Any, name: str, default):
    return getattr(options, name, default) if options is not None else default


def split_decay_params(model, weight_decay: float):
    decay, nodecay = [], []
    no_decay_keys = (
        "bias",
        "norm.weight",
        "bn.weight",
        "bn.bias",
        "layer_norm.weight",
        "LayerNorm.weight",
    )
    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if (len(p.shape) == 1) or any(k in n for k in no_decay_keys):
            nodecay.append(p)
        else:
            decay.append(p)
    return [
        {"params": decay, "weight_decay": weight_decay},
        {"params": nodecay, "weight_decay": 0.0},
    ]


def make_rico_optimizer(model, lr: float, weight_decay: float, rico_args: Dict[str, Any]):
    if RICO is None or rico_layerwise_groups is None:
        raise RuntimeError("RICO optimizer requested but `goat_bench.optimizers.rico` is unavailable.")
    param_groups = rico_layerwise_groups(model, weight_decay=weight_decay)
    try:
        opt = RICO(
            param_groups,
            lr=lr,
            ft_mode=False,
            weight_decay=weight_decay,
            wd_mode="decoupled",
            **rico_args,
        )
    except TypeError as exc:
        print(f"[warn] RICO extra args failed ({exc}) → falling back to minimal args.")
        opt = RICO(param_groups, lr=lr, ft_mode=False, weight_decay=weight_decay, wd_mode="decoupled")
    return opt


def build_optimizer(model, name: str, lr: float, weight_decay: float, options: Any):
    name = name.lower()
    if name == "rico":
        rico_args = dict(
            bk_beta_target=_opt_attr(options, "rico_bk_beta", 0.9),
            k_cap=_opt_attr(options, "rico_k_cap", 0.08),
            g_rms_floor=_opt_attr(options, "rico_g_floor", 1e-3),
            sync_every=_opt_attr(options, "rico_sync_every", 20),
        )
        return make_rico_optimizer(model, lr, weight_decay, rico_args)

    param_groups = split_decay_params(model, weight_decay)

    if name == "adamw":
        return optim.AdamW(param_groups, lr=lr)

    if name == "lion":
        if LionOpt is None:
            raise RuntimeError("Lion optimizer requested but `lion-pytorch` is not installed.")
        beta1 = _opt_attr(options, "lion_beta1", 0.9)
        beta2 = _opt_attr(options, "lion_beta2", 0.99)
        return LionOpt(param_groups, lr=lr, betas=(beta1, beta2), weight_decay=weight_decay)

    if name == "soap":
        if SOAPOpt is None:
            raise RuntimeError("SOAP optimizer requested but `pytorch-optimizer` with SOAP is not installed.")
        extra_args = {}
        soap_raw = _opt_attr(options, "soap_args", None)
        if soap_raw:
            try:
                extra_args = json.loads(soap_raw)
            except Exception as exc:
                print(f"[warn] --soap_args JSON parse failed: {exc} → ignoring extra kwargs")
        return SOAPOpt(param_groups, lr=lr, weight_decay=weight_decay, **extra_args)

    raise ValueError(f"Unknown optimizer: {name}")
