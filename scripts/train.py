"""15 类微调训练。

参数全部来自 YAML（configs/base.yaml + configs/exp/*.yaml），CLI 只保留
`--config` / `--override` / `--no-wandb` 这类与实验无关的开关。

兄弟项目 image_classifier 的三条硬结论直接继承到默认值里：
  1. **过拟合而非欠拟合**（train recall 0.97 / val 0.93）。对应的杠杆是
     weight_decay 1e-3、温和增强、只解冻 2 个 block、label_smoothing 0.1，
     **不是**提高学习率。
  2. **分辨率不是杠杆**（448 ≈ 336），所以默认用 backbone 原生分辨率。
  3. **标签噪声才是天花板**，所以选模型用 macro-F1 而不是 loss。

相比兄弟项目修掉的地方：
  - 所有 `num_classes = 2` / `minlength=2` / 「手动权重必须两个值」的硬编码
  - checkpoint 补存 `model_name_or_path` / `resolution` / `preprocess` /
    `class_names`——evaluate.py 与 predict.py 靠它们复现预处理，
    缺一个就会出现「训练 letterbox@336、推理 crop@224」的静默掉点
  - 单机直接 `python scripts/train.py` 也能跑（不强依赖 torchrun）
  - 逐类指标与两个口径的 macro-F1（15 类全量 / 10 个可靠类）
  - DDP 训练集用 DistributedSampler 真正切分（此前各 rank 遍历全量、梯度完全相同，
    多卡等于单卡还慢 N 倍）

用法：
  # 单卡
  python scripts/train.py --config configs/base.yaml --config configs/exp/exp001_clip_l336.yaml
  # 多卡（batch_size 是全局值，会按卡数均分）
  torchrun --nproc_per_node=4 scripts/train.py --config configs/base.yaml --config configs/exp/exp001_clip_l336.yaml
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import (  # noqa: E402
    add_config_args,
    config_from_args,
    describe,
    load_class_config,
    resolve_model_path,
    resolve_resolution,
    save_resolved,
    setup_hf_env,
)
from dataset import build_dataset, get_dataloaders, get_image_processor, make_collate  # noqa: E402
from model import build_model  # noqa: E402


def setup_distributed() -> tuple[int, int, int]:
    """返回 (rank, local_rank, world_size)。

    没走 torchrun 时（环境里没有 RANK/WORLD_SIZE）自动退化成单进程，
    这样 `python scripts/train.py` 直接就能跑，不必强行包一层 torchrun。
    """
    has_cuda = torch.cuda.is_available()
    if "RANK" not in os.environ or "WORLD_SIZE" not in os.environ:
        if has_cuda:
            torch.cuda.set_device(0)
        return 0, 0, 1

    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    if has_cuda:
        torch.cuda.set_device(local_rank)
    # GPU 上用 nccl，CPU 上退回 gloo
    dist.init_process_group(backend="nccl" if has_cuda else "gloo")
    return rank, local_rank, world_size


def cleanup_distributed(world_size: int) -> None:
    if world_size > 1 and dist.is_initialized():
        dist.destroy_process_group()


def _unwrap(module):
    """拿到被 DDP 包裹前的原始模块。"""
    return module.module if hasattr(module, "module") else module


def compute_metrics(conf_mat: torch.Tensor, reliable_indices: list[int]) -> dict:
    """从混淆矩阵算逐类 P/R/F1，以及两个口径的 macro-F1。

    口径一定要分开报：本库有 5 类样本极少（最小的类 val 只有 2 张），
    它们进 macro 平均会把结论抖得没法看。所以
      macro_f1            —— 15 类全量，与论文/常规口径可比
      macro_f1_reliable   —— 只统计样本充足的类，**选模型与早停用这个**
    """
    tp = conf_mat.diag()
    fp = conf_mat.sum(dim=0) - tp
    fn = conf_mat.sum(dim=1) - tp
    precision = tp / (tp + fp).clamp_min(1e-8)
    recall = tp / (tp + fn).clamp_min(1e-8)
    f1 = 2 * precision * recall / (precision + recall).clamp_min(1e-8)
    support = conf_mat.sum(dim=1)

    def _mean(idx: list[int]) -> float:
        if not idx:
            return 0.0
        return float(f1[idx].mean().item())

    return {
        "precision": precision.tolist(),
        "recall": recall.tolist(),
        "f1": f1.tolist(),
        "support": support.tolist(),
        "macro_precision": float(precision.mean().item()),
        "macro_recall": float(recall.mean().item()),
        "macro_f1": float(f1.mean().item()),
        "macro_f1_reliable": _mean(reliable_indices),
        "accuracy": float(tp.sum().item() / max(conf_mat.sum().item(), 1e-8)),
    }


def _conf_mat_from_preds(labels: torch.Tensor, preds: torch.Tensor, num_classes: int) -> torch.Tensor:
    """向量化累加混淆矩阵。比 Python 逐样本循环快一到两个数量级。"""
    idx = labels * num_classes + preds
    counts = torch.bincount(idx, minlength=num_classes * num_classes)
    return counts.reshape(num_classes, num_classes).float()


@torch.no_grad()
def evaluate(backbone, head, loader, device, num_classes: int, criterion, use_amp: bool, amp_dtype):
    head.eval()
    backbone.eval()
    conf_mat = torch.zeros(num_classes, num_classes, device=device)
    loss_sum, total = 0.0, 0

    for pixel_values, labels in loader:
        pixel_values = pixel_values.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        if use_amp:
            with torch.autocast(device_type="cuda", dtype=amp_dtype):
                logits = head(backbone(pixel_values))
            loss = criterion(logits.float(), labels)
        else:
            logits = head(backbone(pixel_values))
            loss = criterion(logits, labels)
        loss_sum += loss.item() * labels.size(0)
        total += labels.size(0)
        conf_mat += _conf_mat_from_preds(labels, logits.argmax(dim=-1), num_classes)

    return loss_sum / max(total, 1), conf_mat


def build_class_weights(mode: str, counts: torch.Tensor, device) -> torch.Tensor | None:
    """类别权重。101:1 的不均衡下三种口径差别很大，必须能选。

    none —— 标准交叉熵，让模型学先验，多数类表现好、macro-F1 差
    sqrt —— 逆频率**开方**（默认）。全逆频率会把梯度过度倾斜到 10 张的类上，
             开方是常用的折中，在极度不均衡下更稳
    auto —— 全逆频率，少数类召回拉满但误报会明显变多
    """
    if mode == "none":
        return None
    if mode not in ("sqrt", "auto"):
        raise ValueError(f"未知 class_weight：{mode}。支持 none / sqrt / auto")

    counts = counts.float().clamp_min(1.0)
    weights = 1.0 / counts
    if mode == "sqrt":
        weights = weights.sqrt()
    # 归一化到最小权重为 1，使权重尺度与 lr 无关，换类别数时不用重调 lr
    weights = weights / weights.min()
    return weights.to(device)


def save_checkpoint(path: Path, backbone, head, args_meta: dict, metrics: dict) -> None:
    """保存 head（以及可选的 backbone 权重）+ 复现推理所需的全部元信息。

    元信息清单不是可选项：`model_name_or_path` / `resolution` / `preprocess` /
    `class_names` 少任何一个，evaluate.py 与 predict.py 就可能用错预处理或错位标签，
    且不会报错、只会掉点。
    """
    ckpt = {
        "head_state_dict": _unwrap(head).state_dict(),
        "num_classes": args_meta["num_classes"],
        "class_names": args_meta["class_names"],
        "input_dim": args_meta["input_dim"],
        "hidden_dim": args_meta["hidden_dim"],
        "dropout": args_meta["dropout"],
        # ---- 复现推理所必需 ----
        "backbone": args_meta["backbone"],
        "model_name_or_path": args_meta["model_name_or_path"],
        "resolution": args_meta["resolution"],
        "preprocess": args_meta["preprocess"],
        "unfreeze_blocks": args_meta["unfreeze_blocks"],
        # ---- 指标 ----
        "metrics": metrics,
    }
    if _unwrap(backbone).has_trainable_params:
        ckpt["backbone_state_dict"] = _unwrap(backbone).state_dict()
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(ckpt, path)


def format_report(metrics: dict, class_names: list[str], unreliable: set[str]) -> str:
    lines = [
        f"{'类别':<24}{'P':>8}{'R':>8}{'F1':>8}{'support':>9}   备注",
        "-" * 78,
    ]
    for i, name in enumerate(class_names):
        flag = "unreliable" if name in unreliable else ""
        lines.append(
            f"{name:<24}{metrics['precision'][i]:>8.3f}{metrics['recall'][i]:>8.3f}"
            f"{metrics['f1'][i]:>8.3f}{int(metrics['support'][i]):>9}   {flag}"
        )
    lines.append("-" * 78)
    lines.append(
        f"acc={metrics['accuracy']:.4f}  "
        f"macro-F1(15类)={metrics['macro_f1']:.4f}  "
        f"macro-F1(可靠类)={metrics['macro_f1_reliable']:.4f}"
    )
    return "\n".join(lines)


def main() -> int:
    parser = add_config_args(argparse.ArgumentParser(description="15 类图片分类微调"))
    parser.add_argument("--no-wandb", action="store_true", help="禁用 wandb（覆盖配置）")
    args = parser.parse_args()

    cfg = config_from_args(args)
    setup_hf_env(cfg)
    class_spec = load_class_config(cfg)
    class_names = class_spec["class_names"]
    unreliable_names = {c["name"] for c in class_spec["classes"] if c.get("unreliable")}
    reliable_indices = [
        c["index"] for c in class_spec["classes"] if not c.get("unreliable")
    ]
    num_classes = len(class_names)

    rank, local_rank, world_size = setup_distributed()
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
    is_main = rank == 0

    seed = int(cfg["train"]["seed"])
    torch.manual_seed(seed)
    np.random.seed(seed)

    ckpt_dir = Path(cfg["train"]["checkpoint_dir"])
    if is_main:
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        print(describe(cfg))
        print(f"\n设备：{device}  世界大小：{world_size}")
        print(
            f"类别数：{num_classes}（{len(unreliable_names)} 类标记 unreliable，"
            f"不进入选模型的主口径）"
        )

    # ---- 模型 ----
    model_name = resolve_model_path(cfg)
    resolution = resolve_resolution(cfg)
    if resolution is None:
        from dataset import native_resolution

        resolution = native_resolution(cfg["model"]["backbone"], model_name)
    backbone, head = build_model(
        backbone=cfg["model"]["backbone"],
        model_name=model_name,
        num_classes=num_classes,
        unfreeze_blocks=int(cfg["model"]["unfreeze_blocks"]),
        resolution=resolution,
        hidden_dim=int(cfg["model"]["hidden_dim"]),
        dropout=float(cfg["model"]["dropout"]),
    )
    backbone.to(device)
    head.to(device)

    # ---- 数据 ----
    splits_file = Path(cfg["data"]["splits_out"])
    if not splits_file.exists():
        print(
            f"[error] 划分文件不存在：{splits_file}\n请先运行 scripts/prepare_data.py",
            file=sys.stderr,
        )
        return 1

    preprocess = cfg["preprocess"]["mode"]
    augment = cfg["preprocess"]["augment"] == "mild"
    val_proc = get_image_processor(cfg["model"]["backbone"], model_name, resolution, preprocess)
    train_proc = (
        get_image_processor(cfg["model"]["backbone"], model_name, resolution, preprocess, augment=True)
        if augment
        else val_proc
    )
    # 验证集始终用不含增强的 processor，保证 epoch 之间指标干净可比。
    # 注意：transform 交给 collate_fn 施加（dataset 侧传 None），
    # 否则 ManifestDataset.__getitem__ 与 collate 会各做一次 → 双重归一化。
    train_ds = build_dataset("manifest", "train", cfg["data"]["materialize_dir"],
                             splits_file, None, class_names)
    val_ds = build_dataset("manifest", "val", cfg["data"]["materialize_dir"],
                           splits_file, None, class_names)

    # batch_size 记的是**全局**大小，多卡时按卡数均分。
    # 这样 YAML 里那套已经调好的 lr / weight_decay 在多卡下依然成立，
    # 加卡只是更快，不会把配方变成另一个配方（per-rank 语义会让有效 batch 变成 N 倍，
    # 等于悄悄换了学习率尺度）。
    bs_global = int(cfg["train"]["batch_size"])
    if world_size > 1 and bs_global % world_size != 0:
        print(
            f"[warn] 全局 batch_size={bs_global} 不能被卡数 {world_size} 整除，"
            f"每卡取 {max(1, bs_global // world_size)}",
            file=sys.stderr,
        )
    bs = max(1, bs_global // world_size)
    nw = int(cfg["train"]["num_workers"])

    # 各 rank 用不同的增强随机序列。模型初始化在上面已经用统一种子做完了，
    # 这里只影响 dataloader 的增强，让同一张图在不同卡上的抖动不一样。
    if world_size > 1:
        torch.manual_seed(seed + rank)

    train_loader, val_loader = get_dataloaders(
        train_ds, val_ds, bs,
        make_collate(train_proc), make_collate(val_proc),
        num_workers=nw, world_size=world_size, rank=rank,
    )
    if is_main:
        print(f"train {len(train_ds)} 张 / val {len(val_ds)} 张，batch={bs}，增强={'mild' if augment else 'none'}")
        if world_size > 1:
            print(
                f"多卡：{world_size} 张卡，每卡 batch={bs}，"
                f"每卡每 epoch 见 {len(train_ds) // world_size} 张（全局 batch={bs * world_size}）"
            )

    # ---- 损失 ----
    class_weights = build_class_weights(
        cfg["train"]["class_weight"],
        torch.bincount(torch.tensor(train_ds.targets), minlength=num_classes),
        device,
    )
    label_smoothing = float(cfg["train"]["label_smoothing"])
    criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=label_smoothing)
    if is_main:
        print(f"class_weight={cfg['train']['class_weight']}  label_smoothing={label_smoothing}")
        if class_weights is not None:
            shown = ", ".join(f"{n}:{w:.2f}" for n, w in zip(class_names, class_weights.tolist()))
            print(f"  权重：{shown}")

    # ---- DDP ----
    # device_ids 只在 CUDA 下合法：CPU module 传了会直接抛 ValueError。
    # 所以 CPU + gloo 这条降级路径不能带 device_ids（否则多进程 CPU 直接起不来）。
    if world_size > 1:
        ddp_kwargs = {"device_ids": [local_rank]} if torch.cuda.is_available() else {}
        head = DDP(head, **ddp_kwargs)
        if _unwrap(backbone).has_trainable_params:
            backbone = DDP(backbone, **ddp_kwargs)

    # ---- 分层学习率优化器 ----
    head_lr = float(cfg["train"]["lr"])
    bb_mult = float(cfg["train"]["backbone_lr_multiplier"])
    param_groups = [{"params": _unwrap(head).parameters(), "lr": head_lr, "name": "head"}]
    if _unwrap(backbone).has_trainable_params:
        bb_params = [p for p in _unwrap(backbone).parameters() if p.requires_grad]
        if bb_params:
            param_groups.append({"params": bb_params, "lr": head_lr * bb_mult, "name": "backbone"})
    optimizer = AdamW(param_groups, weight_decay=float(cfg["train"]["weight_decay"]))
    epochs = int(cfg["train"]["epochs"])
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs)
    grad_clip = float(cfg["train"]["grad_clip"])

    amp = cfg["train"]["amp"]
    amp_dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "none": None}[amp]
    use_amp = amp_dtype is not None and device.type == "cuda"

    if is_main:
        n_trainable = sum(p.numel() for p in _unwrap(head).parameters() if p.requires_grad)
        n_bb = sum(p.numel() for p in _unwrap(backbone).parameters() if p.requires_grad)
        print(f"可训练参数：head {n_trainable:,} / backbone {n_bb:,}（解冻 {cfg['model']['unfreeze_blocks']} 个 block）")
        print(f"AMP：{amp}  grad_clip={grad_clip}  select_by={cfg['train']['select_by']}")

    # ---- wandb ----
    use_wandb = (not args.no_wandb) and bool(cfg["train"]["wandb"]["enabled"]) and is_main
    if use_wandb:
        try:
            import wandb

            wandb.init(
                project=cfg["train"]["wandb"]["project"],
                name=cfg["train"]["wandb"]["run_name"],
                config=cfg.to_dict() if hasattr(cfg, "to_dict") else dict(cfg),
            )
        except Exception as exc:
            print(f"[warn] wandb 初始化失败，改为不记录：{exc}", file=sys.stderr)
            use_wandb = False

    # 复现所必需的一整套元信息，checkpoint 与 resolved_config 共用
    args_meta = {
        "num_classes": num_classes,
        "class_names": class_names,
        "input_dim": _unwrap(backbone).output_dim,
        "hidden_dim": int(cfg["model"]["hidden_dim"]),
        "dropout": float(cfg["model"]["dropout"]),
        "backbone": cfg["model"]["backbone"],
        "model_name_or_path": model_name,
        "resolution": resolution,
        "preprocess": preprocess,
        "unfreeze_blocks": int(cfg["model"]["unfreeze_blocks"]),
    }

    select_by = cfg["train"]["select_by"]
    if select_by not in ("macro-f1", "macro-f1-all", "loss"):
        print(f"[error] 未知 select_by：{select_by}。支持 macro-f1 / macro-f1-all / loss", file=sys.stderr)
        return 1

    best_score = -float("inf") if select_by != "loss" else float("inf")
    best_epoch = 0
    best_metrics: dict | None = None
    patience_counter = 0
    history: list[dict] = []

    for epoch in range(1, epochs + 1):
        head.train()
        if _unwrap(backbone).has_trainable_params:
            backbone.train()

        # 让 DistributedSampler 每个 epoch 重新洗牌，否则每个 epoch 的分片完全相同
        if world_size > 1 and hasattr(train_loader, "sampler"):
            train_loader.sampler.set_epoch(epoch)

        running_loss, seen = 0.0, 0
        iterator = train_loader
        if is_main:
            try:
                from tqdm import tqdm

                iterator = tqdm(train_loader, desc=f"Epoch {epoch}/{epochs}")
            except ImportError:
                pass

        for pixel_values, labels in iterator:
            pixel_values = pixel_values.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            if _unwrap(backbone).has_trainable_params:
                features = backbone(pixel_values)
            else:
                with torch.no_grad():
                    features = backbone(pixel_values)

            optimizer.zero_grad(set_to_none=True)
            if use_amp:
                with torch.autocast(device_type="cuda", dtype=amp_dtype):
                    logits = head(features)
                loss = criterion(logits.float(), labels)
            else:
                logits = head(features)
                loss = criterion(logits, labels)

            loss.backward()
            if grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(
                    [p for g in param_groups for p in g["params"]], grad_clip
                )
            optimizer.step()

            running_loss += loss.item() * labels.size(0)
            seen += labels.size(0)
            if is_main and hasattr(iterator, "set_postfix"):
                iterator.set_postfix({"loss": f"{loss.item():.4f}"})

        scheduler.step()
        train_loss = running_loss / max(seen, 1)

        val_loss, conf_mat = evaluate(
            backbone, head, val_loader, device, num_classes, criterion, use_amp, amp_dtype
        )
        # 验证集不切分：每个 rank 跑的都是全量，conf_mat 已经是完整的。
        # 这里**不能** all_reduce(SUM)——那会把 N 份相同的矩阵叠加成 N 倍，
        # macro-F1 是比值所以看不出来，但逐类 support 会打印成 N 倍。

        metrics = compute_metrics(conf_mat, reliable_indices)
        metrics["val_loss"] = val_loss
        metrics["train_loss"] = train_loss
        metrics["epoch"] = epoch

        if is_main:
            print(
                f"\nEpoch {epoch}/{epochs}  train_loss={train_loss:.4f}  val_loss={val_loss:.4f}  "
                f"acc={metrics['accuracy']:.4f}  macroF1={metrics['macro_f1']:.4f}  "
                f"macroF1(可靠类)={metrics['macro_f1_reliable']:.4f}"
            )

            if select_by == "loss":
                better = val_loss < best_score
                score = val_loss
            elif select_by == "macro-f1-all":
                better = metrics["macro_f1"] > best_score
                score = metrics["macro_f1"]
            else:
                better = metrics["macro_f1_reliable"] > best_score
                score = metrics["macro_f1_reliable"]

            if better:
                best_score, best_epoch, best_metrics = score, epoch, dict(metrics)
                patience_counter = 0
                save_checkpoint(ckpt_dir / "best.pt", backbone, head, args_meta, metrics)
                print(f"  ✓ 新最优（{select_by}={score:.4f}），已保存 {ckpt_dir / 'best.pt'}")
                print(format_report(metrics, class_names, unreliable_names))
            else:
                patience_counter += 1
                print(f"  未提升，patience {patience_counter}/{cfg['train']['patience']}")

            history.append({k: v for k, v in metrics.items() if k not in ("precision", "recall", "f1", "support")})
            (ckpt_dir / "history.json").write_text(
                json.dumps(history, ensure_ascii=False, indent=2), encoding="utf-8"
            )
            if use_wandb:
                wandb.log({k: v for k, v in metrics.items() if isinstance(v, (int, float))})
            save_checkpoint(ckpt_dir / "last.pt", backbone, head, args_meta, metrics)

        should_stop = torch.tensor(
            [1 if (is_main and patience_counter >= int(cfg["train"]["patience"])) else 0],
            dtype=torch.int32, device=device,
        )
        if world_size > 1:
            dist.broadcast(should_stop, src=0)
        if should_stop.item() == 1:
            if is_main:
                print(f"\n早停：连续 {patience_counter} 个 epoch 未提升。")
            break

    if is_main:
        save_resolved(cfg, ckpt_dir, "resolved_config.yaml")
        print("\n" + "=" * 78)
        if best_metrics is not None:
            print(f"最优 epoch {best_epoch}：")
            print(format_report(best_metrics, class_names, unreliable_names))
        print(f"\n产物：{ckpt_dir}/best.pt  {ckpt_dir}/last.pt  {ckpt_dir}/history.json  "
              f"{ckpt_dir}/resolved_config.yaml")
        print(f"下一步：python scripts/evaluate.py --config configs/base.yaml "
              f"--override eval.checkpoint={ckpt_dir / 'best.pt'}")
    if use_wandb:
        wandb.finish()
    cleanup_distributed(world_size)
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
