"""数据集与图像预处理。

沿用兄弟项目 image_classifier 验证过的两个关键选择，本数据集同样适用：

1. **letterbox（等比缩放 + 白边补齐）而非 resize+centercrop**。
   本任务的判别信号是图里的**文字与排版**，裁剪会直接切掉文字。实测本库缩到 336 后
   最短边 min=16.7px、中位数 213px，没有任何一张被压塌，所以 letterbox 是安全的。
2. **温和增强，不做水平翻转 / 大裁剪**。翻转会破坏文字方向，大裁剪会切掉内容。
   只做小幅尺变/平移/旋转 + 轻微颜色抖动。

新增 ManifestDataset：直接读 prepare_data.py 产出的 splits.jsonl，
让改划分不必重新抽特征（特征缓存与划分解耦）。
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Callable, Sequence

import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler
from torchvision import transforms
from torchvision.datasets import ImageFolder

# 各 backbone 的归一化统计值。
# 新增 backbone 时必须在这里登记，否则 get_image_processor 会直接报错，
# 避免用错归一化导致指标莫名下降。
_BACKBONE_NORMALIZE = {
    "clip": ([0.48145466, 0.4578275, 0.40821073], [0.26862954, 0.26130258, 0.27577711]),
    "dinov3": ([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    "dinov2": ([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    "siglip": ([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
    "resnet": ([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
}

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff", ".gif"}

try:
    _BILINEAR = Image.Resampling.BILINEAR
except AttributeError:  # Pillow < 9.1
    _BILINEAR = Image.BILINEAR


class LetterboxSquare:
    """等比缩放后用背景色居中补齐为正方形。

    不压扁、不裁剪，完整保留整张图——尤其是图表里的文字与排版，这是本数据集的判别关键。
    默认白边：本库多数图表本身就是白底，白边能自然融合。
    """

    def __init__(self, resolution: int, fill: int = 255):
        self.resolution = resolution
        self.fill = fill

    def __call__(self, img: Image.Image) -> Image.Image:
        w, h = img.size
        scale = self.resolution / max(w, h)
        new_w = max(1, round(w * scale))
        new_h = max(1, round(h * scale))
        img = img.resize((new_w, new_h), resample=_BILINEAR)
        canvas = Image.new("RGB", (self.resolution, self.resolution), (self.fill,) * 3)
        canvas.paste(img, ((self.resolution - new_w) // 2, (self.resolution - new_h) // 2))
        return canvas


def _mild_augment_transforms() -> list:
    """温和增强：小幅尺变/平移/旋转 + 轻微颜色抖动。

    刻意不做水平翻转与大裁剪——本任务的信号是文字方向与排版，翻转/大裁剪会破坏语义。
    仅在训练分支启用；验证集始终用原图，保证指标干净可比。
    """
    return [
        transforms.RandomAffine(degrees=2, translate=(0.03, 0.03), scale=(0.95, 1.05), fill=255),
        transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.15),
    ]


def _make_letterbox_compose(backbone: str, resolution: int, augment: bool = False):
    mean, std = _BACKBONE_NORMALIZE[backbone]
    ops: list = [LetterboxSquare(resolution)]
    if augment:
        ops += _mild_augment_transforms()
    ops += [transforms.ToTensor(), transforms.Normalize(mean=mean, std=std)]
    return transforms.Compose(ops)


def _make_resize_crop_compose(backbone: str, resolution: int, augment: bool = False):
    """旧策略：短边缩放到 resolution 后中心裁出正方形。仅用于与 letterbox 做消融对比。

    会裁掉长图两端的文字/排版内容——本库 31.4% 的图长宽比 >=2，损失明显。
    """
    mean, std = _BACKBONE_NORMALIZE[backbone]
    ops: list = [transforms.Resize(resolution), transforms.CenterCrop(resolution)]
    if augment:
        ops += _mild_augment_transforms()
    ops += [transforms.ToTensor(), transforms.Normalize(mean=mean, std=std)]
    return transforms.Compose(ops)


def _read_local_size(model_name: str) -> int | None:
    """从本地模型目录的 preprocessor_config.json / config.json 读出原生分辨率。"""
    path = Path(model_name)
    if not path.is_dir():
        return None

    pre = path / "preprocessor_config.json"
    if pre.exists():
        try:
            cfg = json.loads(pre.read_text(encoding="utf-8"))
        except Exception:
            cfg = {}
        size = cfg.get("size")
        if isinstance(size, int):
            return size
        if isinstance(size, dict):
            if isinstance(size.get("height"), int):
                return size["height"]
            if isinstance(size.get("shortest_edge"), int):
                return size["shortest_edge"]
        crop = cfg.get("crop_size")
        if isinstance(crop, int):
            return crop
        if isinstance(crop, dict) and isinstance(crop.get("height"), int):
            return crop["height"]

    con = path / "config.json"
    if con.exists():
        try:
            cfg = json.loads(con.read_text(encoding="utf-8"))
        except Exception:
            cfg = {}
        vision = cfg.get("vision_config") or {}
        for src in (vision, cfg):
            size = src.get("image_size")
            if isinstance(size, int):
                return size
    return None


def native_resolution(backbone: str, model_name: str) -> int:
    """推断 backbone 的原生输入分辨率。

    优先级：本地模型目录里的真实配置 > 名字里的三位数 > 224。
    优先读真实配置，是因为像 `dinov3-vitl16-pretrain` 这种名字里没有分辨率信息。
    """
    resolved = _read_local_size(model_name)
    if resolved:
        return resolved

    if backbone == "clip":
        # clip-vit-large-patch14-336 -> 336；clip-vit-base-patch32 -> 224
        m = re.search(r"-(\d{3})(?:$|[-.])", model_name)
        if m:
            return int(m.group(1))
    return 224


def get_image_processor(
    backbone: str,
    model_name: str,
    resolution: int | None = None,
    preprocess: str = "letterbox",
    augment: bool = False,
):
    """返回统一的图像预处理器，输出 (3, R, R) tensor。

    统一用 torchvision Compose，因此 collate 时直接 torch.stack 即可，
    不依赖各 HF AutoImageProcessor 各自的默认裁剪/缩放行为（那些行为不一致且会静默改变输入）。
    """
    backbone = backbone.lower()
    if backbone not in _BACKBONE_NORMALIZE:
        raise ValueError(
            f"未知 backbone：{backbone}。支持 {list(_BACKBONE_NORMALIZE)}。"
            "新增 backbone 时请同时登记它的归一化统计值。"
        )

    if resolution is None:
        resolution = native_resolution(backbone, model_name)

    preprocess = preprocess.lower()
    if preprocess == "letterbox":
        return _make_letterbox_compose(backbone, resolution, augment)
    if preprocess == "crop":
        return _make_resize_crop_compose(backbone, resolution, augment)
    raise ValueError(f"未知 preprocess：{preprocess}。支持 letterbox / crop")


def make_collate(proc: Callable):
    """把 processor 包成 collate_fn（先应用 processor 再 stack）。"""

    def collate_fn(batch):
        images, labels = zip(*batch)
        return torch.stack([proc(img) for img in images]), torch.tensor(labels, dtype=torch.long)

    return collate_fn


class ManifestDataset(Dataset):
    """读 prepare_data.py 产出的 splits.jsonl。

    与 ImageFolder 相比的好处：划分是显式的、可审计的文件，改划分不必重抽特征，
    且能直接带上 dup_group 等元信息。
    """

    def __init__(
        self,
        splits_file: str | Path,
        split: str,
        transform: Callable | None = None,
        class_names: Sequence[str] | None = None,
    ):
        self.splits_file = Path(splits_file)
        self.split = split
        self.transform = transform
        if not self.splits_file.exists():
            raise FileNotFoundError(
                f"划分文件不存在：{self.splits_file}。请先运行 scripts/prepare_data.py"
            )

        self.records: list[dict] = []
        with self.splits_file.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                if rec["split"] == split:
                    self.records.append(rec)

        if not self.records:
            raise ValueError(f"{self.splits_file} 中没有任何 split={split!r} 的记录")

        ordered = list(class_names) if class_names is not None else None
        if ordered is not None:
            self.class_names = ordered
        else:  # 没给类别清单时按 label_index 还原
            n = max(r["label_index"] for r in self.records) + 1
            self.class_names = [None] * n
            for rec in self.records:
                self.class_names[rec["label_index"]] = rec["label"]

        self.targets = [int(r["label_index"]) for r in self.records]
        self.paths = [r["path"] for r in self.records]

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int):
        rec = self.records[idx]
        with Image.open(rec["path"]) as im:
            img = im.convert("RGB")
        if self.transform is not None:
            img = self.transform(img)
        return img, int(rec["label_index"])


def build_dataset(
    kind: str,
    split: str,
    data_dir: str | Path,
    splits_file: str | Path | None,
    transform: Callable,
    class_names: Sequence[str] | None = None,
) -> Dataset:
    """按 kind 构建数据集：manifest（读 splits.jsonl）或 imagefolder（读 data/{split}/{类}/）。"""
    if kind == "manifest":
        return ManifestDataset(splits_file, split, transform, class_names)
    if kind == "imagefolder":
        root = Path(data_dir) / split
        if not root.exists():
            raise FileNotFoundError(
                f"{root} 不存在。请先运行 scripts/prepare_data.py 物化划分目录"
            )
        return ImageFolder(str(root), transform=transform)
    raise ValueError(f"未知数据集类型：{kind!r}。支持 manifest / imagefolder")


def get_dataloaders(
    train_dataset: Dataset | None,
    val_dataset: Dataset,
    batch_size: int,
    train_collate: Callable,
    val_collate: Callable,
    num_workers: int = 8,
    pin_memory: bool = True,
    world_size: int = 1,
    rank: int = 0,
) -> tuple[DataLoader | None, DataLoader]:
    """构建 DataLoader。

    DDP 的关键点：训练集**必须**用 DistributedSampler 切分，否则每个 rank 都会遍历
    全量数据。加上各 rank 的种子相同，打乱顺序与增强也会逐样本一致 —— 于是 N 个 rank
    算出来的是 N 份完全相同的梯度，被 DDP 平均掉后等价于单卡，只是慢了 N 倍。

    验证集**不切分**：每个 rank 跑全量。val 没有反向传播，20% 的数据冗余算一遍很便宜，
    换来的是每个 rank 都持有一份完整的混淆矩阵，直接算指标即可，不需要跨卡归约
    （跨卡 SUM 会把 N 份相同的矩阵叠加成 N 倍，逐类 support 会打印成 N 倍）。
    """
    train_loader = None
    if train_dataset is not None:
        sampler = None
        if world_size > 1:
            sampler = DistributedSampler(
                train_dataset, num_replicas=world_size, rank=rank, shuffle=True, drop_last=True
            )
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=sampler is None,  # 与 sampler 互斥
            sampler=sampler,
            num_workers=num_workers,
            pin_memory=pin_memory,
            collate_fn=train_collate,
            drop_last=True,
            persistent_workers=num_workers > 0,
        )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=val_collate,
        drop_last=False,
        persistent_workers=num_workers > 0,
    )
    return train_loader, val_loader


def class_counts(dataset: Dataset, num_classes: int | None = None) -> torch.Tensor:
    """统计各类样本数，按 label_index 索引。"""
    targets = getattr(dataset, "targets", None)
    if targets is None:
        raise ValueError("dataset 没有 targets 属性，无法统计类别分布")
    if num_classes is None:
        num_classes = max(int(t) for t in targets) + 1
    counts = torch.zeros(num_classes, dtype=torch.long)
    for t in targets:
        counts[int(t)] += 1
    return counts
