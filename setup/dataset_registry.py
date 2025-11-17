# setup/dataset_registry.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List


@dataclass
class DatasetInfo:
    name: str
    domain: str                # "cv" / "nlp" / "llm" / "structure" ...
    approx_size_gb: float
    requires_manual_download: bool
    description: str
    tags: List[str]


# 최소 예시 (나중에 계속 추가)
DATASETS: Dict[str, DatasetInfo] = {
    "cifar10": DatasetInfo(
        name="cifar10",
        domain="cv",
        approx_size_gb=0.2,
        requires_manual_download=False,
        description="CIFAR-10 (torchvision으로 자동 다운로드)",
        tags=["small", "toy"],
    ),
    "cifar100": DatasetInfo(
        name="cifar100",
        domain="cv",
        approx_size_gb=0.3,
        requires_manual_download=False,
        description="CIFAR-100 (torchvision 자동 다운로드)",
        tags=["small", "toy"],
    ),
    "tinyimagenet": DatasetInfo(
        name="tinyimagenet",
        domain="cv",
        approx_size_gb=1.0,
        requires_manual_download=False,
        description="TinyImageNet-200",
        tags=["small", "toy"],
    ),
    "wmt14_en_de": DatasetInfo(
        name="wmt14_en_de",
        domain="nlp",
        approx_size_gb=40.0,
        requires_manual_download=False,
        description="WMT14 English-German translation corpus",
        tags=["heavy"],
    ),
    "coco2017": DatasetInfo(
        name="coco2017",
        domain="cv",
        approx_size_gb=25.0,
        requires_manual_download=True,
        description="COCO 2017 detection dataset (수동 다운로드 추천)",
        tags=["cv", "heavy"],
    ),
    "ade20k": DatasetInfo(
        name="ade20k",
        domain="cv",
        approx_size_gb=3.5,
        requires_manual_download=True,
        description="ADE20K segmentation dataset (MIT Scene Parsing Challenge, manual download required)",
        tags=["cv", "segmentation"],
    ),
    "imagenet1k": DatasetInfo(
        name="imagenet1k",
        domain="cv",
        approx_size_gb=150.0,
        requires_manual_download=True,
        description="ImageNet-1k (수동 다운로드 및 라이선스 동의 필요)",
        tags=["license_warning", "heavy"],
    ),
    # === NLP / HF-managed datasets ===
    "sst2": DatasetInfo(
        name="sst2",
        domain="nlp",
        approx_size_gb=3.0,
        requires_manual_download=False,
        description="GLUE - SST-2 감성 분류 (Hugging Face)",
        tags=["nlp", "hf", "small"],
    ),
    "mrpc": DatasetInfo(
        name="mrpc",
        domain="nlp",
        approx_size_gb=0.5,
        requires_manual_download=False,
        description="GLUE - MRPC 패러프레이즈 (Hugging Face)",
        tags=["nlp", "hf", "small"],
    ),
    "stsb": DatasetInfo(
        name="stsb",
        domain="nlp",
        approx_size_gb=0.3,
        requires_manual_download=False,
        description="GLUE - STS-B 유사도 회귀",
        tags=["nlp", "hf", "small"],
    ),
    "qqp": DatasetInfo(
        name="qqp",
        domain="nlp",
        approx_size_gb=2.0,
        requires_manual_download=False,
        description="GLUE - QQP 중복 질문 판별",
        tags=["nlp", "hf", "medium"],
    ),
    "ag_news": DatasetInfo(
        name="ag_news",
        domain="nlp",
        approx_size_gb=1.0,
        requires_manual_download=False,
        description="AG News 4-way 분류",
        tags=["nlp", "hf", "small"],
    ),
    "boolq": DatasetInfo(
        name="boolq",
        domain="nlp",
        approx_size_gb=1.0,
        requires_manual_download=False,
        description="SuperGLUE - BoolQ (HF)",
        tags=["nlp", "hf", "small"],
    ),
    "rte": DatasetInfo(
        name="rte",
        domain="nlp",
        approx_size_gb=0.5,
        requires_manual_download=False,
        description="SuperGLUE - Recognizing Textual Entailment",
        tags=["nlp", "hf", "small"],
    ),
    "cb": DatasetInfo(
        name="cb",
        domain="nlp",
        approx_size_gb=0.2,
        requires_manual_download=False,
        description="SuperGLUE - CommitmentBank",
        tags=["nlp", "hf", "tiny"],
    ),
    "anli": DatasetInfo(
        name="anli",
        domain="nlp",
        approx_size_gb=5.0,
        requires_manual_download=False,
        description="ANLI (r1-r3) NLI 데이터",
        tags=["nlp", "hf", "medium"],
    ),
    "hellaswag": DatasetInfo(
        name="hellaswag",
        domain="nlp",
        approx_size_gb=10.0,
        requires_manual_download=False,
        description="HellaSwag commonsense MCQ",
        tags=["nlp", "hf", "medium"],
    ),
    "piqa": DatasetInfo(
        name="piqa",
        domain="nlp",
        approx_size_gb=1.5,
        requires_manual_download=False,
        description="PIQA physical commonsense MCQ",
        tags=["nlp", "hf", "small"],
    ),
    "copa": DatasetInfo(
        name="copa",
        domain="nlp",
        approx_size_gb=0.5,
        requires_manual_download=False,
        description="SuperGLUE - COPA choice-of-plausible-alternative",
        tags=["nlp", "hf", "small"],
    ),
    "winogrande": DatasetInfo(
        name="winogrande",
        domain="nlp",
        approx_size_gb=15.0,
        requires_manual_download=False,
        description="WinoGrande XL pronoun resolution",
        tags=["nlp", "hf", "medium"],
    ),
    "alpha_nli": DatasetInfo(
        name="alpha_nli",
        domain="nlp",
        approx_size_gb=5.0,
        requires_manual_download=False,
        description="Abductive NLI (Rowan/abductive_nli)",
        tags=["nlp", "hf", "medium"],
    ),
    "squad_v2": DatasetInfo(
        name="squad_v2",
        domain="nlp",
        approx_size_gb=3.0,
        requires_manual_download=False,
        description="SQuAD v2 QA (Hugging Face)",
        tags=["nlp", "hf", "small"],
    ),
    "wmt14_en_de": DatasetInfo(
        name="wmt14_en_de",
        domain="nlp",
        approx_size_gb=15.0,
        requires_manual_download=False,
        description="WMT14 En↔De translation (Hugging Face)",
        tags=["nlp", "hf", "heavy"],
    ),
    "wmt14_de_en": DatasetInfo(
        name="wmt14_de_en",
        domain="nlp",
        approx_size_gb=15.0,
        requires_manual_download=False,
        description="WMT14 De↔En translation (Hugging Face)",
        tags=["nlp", "hf", "heavy"],
    ),
    "xsum": DatasetInfo(
        name="xsum",
        domain="nlp",
        approx_size_gb=9.0,
        requires_manual_download=False,
        description="XSum abstractive summarization",
        tags=["nlp", "hf", "medium"],
    ),
    "wikitext2": DatasetInfo(
        name="wikitext2",
        domain="nlp",
        approx_size_gb=1.0,
        requires_manual_download=False,
        description="WikiText-2 language modeling corpus",
        tags=["nlp", "hf", "small"],
    ),
    "wikitext103": DatasetInfo(
        name="wikitext103",
        domain="nlp",
        approx_size_gb=4.0,
        requires_manual_download=False,
        description="WikiText-103 raw corpus (대용량 버전)",
        tags=["nlp", "hf", "medium", "license_warning"],
    ),
    "gsm8k": DatasetInfo(
        name="gsm8k",
        domain="nlp",
        approx_size_gb=2.0,
        requires_manual_download=False,
        description="GSM8K math word problems (HF)",
        tags=["nlp", "hf", "medium"],
    ),
    "multirc": DatasetInfo(
        name="multirc",
        domain="nlp",
        approx_size_gb=2.0,
        requires_manual_download=False,
        description="SuperGLUE - MultiRC",
        tags=["nlp", "hf", "medium"],
    ),
    "record": DatasetInfo(
        name="record",
        domain="nlp",
        approx_size_gb=6.0,
        requires_manual_download=False,
        description="SuperGLUE - ReCoRD",
        tags=["nlp", "hf", "medium"],
    ),
    "hotpotqa": DatasetInfo(
        name="hotpotqa",
        domain="nlp",
        approx_size_gb=18.0,
        requires_manual_download=False,
        description="HotpotQA distractor split",
        tags=["nlp", "hf", "heavy"],
    ),
    "alpaca": DatasetInfo(
        name="alpaca",
        domain="nlp",
        approx_size_gb=0.5,
        requires_manual_download=False,
        description="Instruction-tuning dataset (tatsu-lab/alpaca)",
        tags=["nlp", "hf", "small", "license_warning"],
    ),
}


def list_datasets() -> List[DatasetInfo]:
    return list(DATASETS.values())


def get_dataset(name: str) -> DatasetInfo:
    if name not in DATASETS:
        raise KeyError(f"Unknown dataset: {name}")
    return DATASETS[name]
