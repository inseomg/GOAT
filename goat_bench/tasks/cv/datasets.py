# goat_bench/tasks/cv/datasets.py
from __future__ import annotations

import shutil
from pathlib import Path
from typing import Tuple

import numpy as np
import torch
import torchvision
import torchvision.transforms as T
from torchvision.transforms import functional as TF


IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
CIFAR100_MEAN = [0.5071, 0.4867, 0.4408]
CIFAR100_STD = [0.2675, 0.2565, 0.2761]


def ensure_tinyimagenet(root: Path):
    from urllib.request import urlretrieve
    import zipfile

    root = Path(root)
    ti_dir = root / "tiny-imagenet-200"
    if not ti_dir.exists():
        root.mkdir(parents=True, exist_ok=True)
        url = "http://cs231n.stanford.edu/tiny-imagenet-200.zip"
        zip_path = root / "tiny-imagenet-200.zip"
        print(f"[tinyimagenet] downloading → {zip_path}")
        urlretrieve(url, zip_path)
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(root)
        print("[tinyimagenet] extracted")

    # reorganize validation split (val/images -> val/<wnid>/)
    val_dir = ti_dir / "val"
    img_dir = val_dir / "images"
    ann = val_dir / "val_annotations.txt"
    if img_dir.exists() and ann.exists():
        print("[tinyimagenet] reorganizing validation folders…")
        with ann.open("r") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) < 2:
                    continue
                fname, wnid = parts[0], parts[1]
                (val_dir / wnid).mkdir(exist_ok=True)
                src = img_dir / fname
                dst = val_dir / wnid / fname
                if src.exists():
                    shutil.move(str(src), str(dst))
        shutil.rmtree(img_dir, ignore_errors=True)
        print("[tinyimagenet] validation reorganized")


def get_cls_datasets(dataset: str, root: Path):
    d = dataset.lower()
    if d == "cifar100":
        mean, std = CIFAR100_MEAN, CIFAR100_STD
        tr = T.Compose(
            [
                T.RandomCrop(32, padding=4),
                T.RandomHorizontalFlip(),
                T.ToTensor(),
                T.Normalize(mean, std),
            ]
        )
        va = T.Compose([T.ToTensor(), T.Normalize(mean, std)])
        trset = torchvision.datasets.CIFAR100(root=root, train=True, download=True, transform=tr)
        vaset = torchvision.datasets.CIFAR100(root=root, train=False, download=True, transform=va)
        num_classes, in_size = 100, 32
    elif d == "tinyimagenet":
        ensure_tinyimagenet(root)
        mean, std = IMAGENET_MEAN, IMAGENET_STD
        tr = T.Compose(
            [
                T.RandomResizedCrop(64, scale=(0.6, 1.0)),
                T.RandomHorizontalFlip(),
                T.ToTensor(),
                T.Normalize(mean, std),
            ]
        )
        va = T.Compose(
            [
                T.Resize(64),
                T.CenterCrop(64),
                T.ToTensor(),
                T.Normalize(mean, std),
            ]
        )
        trset = torchvision.datasets.ImageFolder(root / "tiny-imagenet-200" / "train", transform=tr)
        vaset = torchvision.datasets.ImageFolder(root / "tiny-imagenet-200" / "val", transform=va)
        num_classes, in_size = 200, 64
    elif d == "imagenet":
        mean, std = IMAGENET_MEAN, IMAGENET_STD
        tr = T.Compose(
            [
                T.RandomResizedCrop(224),
                T.RandomHorizontalFlip(),
                T.ToTensor(),
                T.Normalize(mean, std),
            ]
        )
        va = T.Compose(
            [
                T.Resize(256),
                T.CenterCrop(224),
                T.ToTensor(),
                T.Normalize(mean, std),
            ]
        )
        trset = torchvision.datasets.ImageFolder(root / "train", transform=tr)
        vaset = torchvision.datasets.ImageFolder(root / "val", transform=va)
        num_classes, in_size = 1000, 224
    else:
        raise ValueError(f"unknown cls dataset: {dataset}")
    return trset, vaset, num_classes, in_size


class RandomHFlipDet:
    def __init__(self, p: float = 0.5):
        self.p = p

    def __call__(self, img, target):
        if torch.rand(1).item() < self.p and target["boxes"].numel() > 0:
            img = T.functional.hflip(img)
            w = img.size(2)
            boxes = target["boxes"]
            x1 = w - boxes[:, 2]
            x2 = w - boxes[:, 0]
            boxes[:, 0], boxes[:, 2] = x1, x2
            target["boxes"] = boxes
        return img, target


class CocoDet(torchvision.datasets.CocoDetection):
    def __init__(self, img_folder, ann_file, train: bool = True):
        super().__init__(img_folder, ann_file)
        self.train = train
        cats = sorted(self.coco.getCatIds())
        self.cat2contig = {c: i + 1 for i, c in enumerate(cats)}
        self.contig2cat = {v: k for k, v in self.cat2contig.items()}
        self.to_tensor = T.ToTensor()
        self.flip = RandomHFlipDet(p=0.5 if train else 0.0)

    def __getitem__(self, idx):
        img, anns = super().__getitem__(idx)
        w, h = img.size
        boxes, labels, areas, iscrowd = [], [], [], []
        for a in anns:
            if a.get("iscrowd", 0) == 1:
                continue
            x, y, bw, bh = a["bbox"]
            if bw <= 1 or bh <= 1:
                continue
            x2, y2 = x + bw, y + bh
            boxes.append([x, y, x2, y2])
            labels.append(self.cat2contig[a["category_id"]])
            areas.append(a["area"])
            iscrowd.append(0)
        if len(boxes) == 0:
            boxes = torch.zeros((0, 4), dtype=torch.float32)
            labels = torch.zeros((0,), dtype=torch.int64)
            areas = torch.zeros((0,), dtype=torch.float32)
            iscrowd = torch.zeros((0,), dtype=torch.int64)
        else:
            boxes = torch.tensor(boxes, dtype=torch.float32)
            labels = torch.tensor(labels, dtype=torch.int64)
            areas = torch.tensor(areas, dtype=torch.float32)
            iscrowd = torch.tensor(iscrowd, dtype=torch.int64)

        img = self.to_tensor(img)
        target = {
            "boxes": boxes,
            "labels": labels,
            "image_id": torch.tensor(self.ids[idx], dtype=torch.int64),
            "area": areas,
            "iscrowd": iscrowd,
            "orig_size": torch.tensor([int(h), int(w)]),
        }
        img, target = self.flip(img, target)
        return img, target


def collate_det(batch):
    return tuple(zip(*batch))


class ADE20K(torch.utils.data.Dataset):
    def __init__(self, root: Path, split: str, crop: Tuple[int, int] = (512, 512), train: bool = True):
        self.img_dir = Path(root) / "images" / ("training" if split == "train" else "validation")
        self.ann_dir = Path(root) / "annotations" / ("training" if split == "train" else "validation")
        self.ids = sorted([p.stem for p in self.img_dir.glob("*.jpg")])
        self.train = train
        self.crop_size = crop
        self.to_tensor = T.ToTensor()

    def __len__(self):
        return len(self.ids)

    def _load(self, idx):
        from PIL import Image

        img = Image.open(self.img_dir / f"{self.ids[idx]}.jpg").convert("RGB")
        mask = Image.open(self.ann_dir / f"{self.ids[idx]}.png")
        return img, mask

    def __getitem__(self, idx):
        img, mask = self._load(idx)
        if self.train:
            i, j, h, w = T.RandomResizedCrop.get_params(img, scale=(0.5, 1.0), ratio=(1.0, 1.0))
            img = TF.resized_crop(img, i, j, h, w, self.crop_size, interpolation=T.InterpolationMode.BILINEAR)
            mask = TF.resized_crop(mask, i, j, h, w, self.crop_size, interpolation=T.InterpolationMode.NEAREST)
            if torch.rand(1).item() < 0.5:
                img = TF.hflip(img)
                mask = TF.hflip(mask)
        else:
            img = TF.resize(img, 520, interpolation=T.InterpolationMode.BILINEAR)
            mask = TF.resize(mask, 520, interpolation=T.InterpolationMode.NEAREST)
            img = TF.center_crop(img, self.crop_size)
            mask = TF.center_crop(mask, self.crop_size)

        img = self.to_tensor(img)
        mask = torch.from_numpy(np.array(mask, dtype=np.uint8)).long()
        return img, mask
