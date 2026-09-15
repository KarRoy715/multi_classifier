"""跨脚本共享的纯工具函数。

把原本散落在 extract_features.py / evaluate.py 里、被其他主脚本 import 的函数
集中到这里，让主脚本不再互相 import，边界更清晰。
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from model import ClassifierHead, load_backbone_from_checkpoint  # noqa: E402


def load_records(splits_file: Path) -> list[dict]:
    """读 splits.jsonl，返回固定排序的记录列表。"""
    records = []
    with splits_file.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    records.sort(key=lambda r: r["path"])
    return records


def cache_paths(out_dir: Path, name: str) -> dict[str, Path]:
    """返回一个候选特征缓存的四个标准路径。"""
    return {
        "npy": out_dir / f"{name}.npy",
        "index": out_dir / f"{name}.index.jsonl",
        "meta": out_dir / f"{name}.meta.json",
        "done": out_dir / f"{name}.done.json",
    }


def load_checkpoint(path: Path, device: Any = None) -> dict:
    """加载 train.py 保存的 checkpoint，校验必要字段。

    device 参数保留以兼容旧调用，实际 map_location 固定为 cpu。
    """
    if not path.exists():
        raise FileNotFoundError(f"checkpoint 不存在：{path}")
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    for key in ("head_state_dict", "backbone", "num_classes"):
        if key not in ckpt:
            raise KeyError(
                f"checkpoint 缺少必需字段 {key!r}。"
                "若是旧版本 train.py 存的，请重新训练。"
            )
    return ckpt


def build_from_checkpoint(ckpt: dict, cfg, device):
    """按 checkpoint 的元信息重建 backbone + head。配置只做兜底。"""
    models_root = cfg["runtime"]["models_root"]
    backbone = load_backbone_from_checkpoint(ckpt, models_root)
    head = ClassifierHead(
        input_dim=int(ckpt["input_dim"]),
        hidden_dim=int(ckpt.get("hidden_dim", 256)),
        num_classes=int(ckpt["num_classes"]),
        dropout=float(ckpt.get("dropout", 0.3)),
    )
    head.load_state_dict(ckpt["head_state_dict"])
    if "backbone_state_dict" in ckpt:
        backbone.load_state_dict(ckpt["backbone_state_dict"])
    backbone.to(device).eval()
    head.to(device).eval()
    return backbone, head
