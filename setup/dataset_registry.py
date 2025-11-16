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
        description="ADE20K segmentation dataset",
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
    "alphafold_pdb": DatasetInfo(
        name="alphafold_pdb",
        domain="structure",
        approx_size_gb=255.0,
        requires_manual_download=True,
        description="AlphaFold/Protein 구조 데이터 (PDB 등)",
        tags=["license_warning", "heavy"],
    ),
}


def list_datasets() -> List[DatasetInfo]:
    return list(DATASETS.values())


def get_dataset(name: str) -> DatasetInfo:
    if name not in DATASETS:
        raise KeyError(f"Unknown dataset: {name}")
    return DATASETS[name]
