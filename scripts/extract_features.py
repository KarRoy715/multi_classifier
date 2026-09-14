"""抽取冻结 backbone 的图像特征并缓存。

为什么先抽特征
--------------
backbone 冻结时特征与训练无关，抽一次可以反复用：
  - 线性探针大比拼（选 backbone）——秒级完成，不碰 GPU
  - 微调时的 head 初始化参考
  - 换模型结构/超参重训 head 时无需重跑 backbone
这让「先比 backbone 再微调」的成本降到几乎为零。

缓存格式
--------
  features/{name}.npy          float16 memmap，形状 (N, D)，行序与 index.jsonl 对齐
  features/{name}.index.jsonl  每行 {path, label, label_index, split, dup_group}
  features/{name}.meta.json    维度/分辨率/模型等元信息，用于校验缓存是否匹配
  features/{name}.done.json    已完成的样本数（断点续传用）

断点续传：预分配 memmap，只跑未完成的样本，每 flush_every 个 batch 落盘一次。
中断后重跑会自动接着跑剩下的，不会从头再来。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import (  # noqa: E402
    add_config_args,
    config_from_args,
    load_class_config,
    resolve_model_path,
    resolve_resolution,
    save_resolved,
    setup_hf_env,
)
from dataset import get_image_processor  # noqa: E402
from model import BackboneWrapper  # noqa: E402
from PIL import Image  # noqa: E402

Image.MAX_IMAGE_PIXELS = None


def load_records(splits_file: Path) -> list[dict]:
    records = []
    with splits_file.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    # 固定顺序，保证多次运行时行序一致
    records.sort(key=lambda r: r["path"])
    return records


def cache_paths(out_dir: Path, name: str) -> dict[str, Path]:
    return {
        "npy": out_dir / f"{name}.npy",
        "index": out_dir / f"{name}.index.jsonl",
        "meta": out_dir / f"{name}.meta.json",
        "done": out_dir / f"{name}.done.json",
    }


def read_done(path: Path) -> int:
    if path.exists():
        try:
            return int(json.loads(path.read_text(encoding="utf-8"))["n_done"])
        except Exception:
            return 0
    return 0


def write_done(path: Path, n_done: int) -> None:
    path.write_text(json.dumps({"n_done": n_done}), encoding="utf-8")


def extract_one(
    candidate: dict,
    records: list[dict],
    cfg,
    out_dir: Path,
    device: torch.device,
    force: bool = False,
) -> dict:
    name = candidate["name"]
    paths = cache_paths(out_dir, name)
    n = len(records)
    model_path = resolve_model_path(cfg, candidate)
    resolution = resolve_resolution(cfg, candidate)
    backbone_type = candidate["backbone"]

    # 已完整就跳过（除非 --force）
    if paths["meta"].exists() and not force:
        meta = json.loads(paths["meta"].read_text(encoding="utf-8"))
        if (
            meta.get("n_images") == n
            and results_ok(meta, paths, n)
            and read_done(paths["done"]) >= n
        ):
            print(f"[skip] {name}：缓存已完整（{n} 张，D={meta['feat_dim']}）")
            return meta
        print(f"[redo] {name}：缓存不完整，继续抽剩余部分")

    print(f"\n{'=' * 70}\n[{name}] backbone={backbone_type} res={resolution} model={Path(model_path).name}")
    backbone = BackboneWrapper(backbone_type, model_path, unfreeze_blocks=0, resolution=resolution)
    backbone.to(device).eval()
    dim = backbone.output_dim

    # 预分配 memmap（float16 存一半体积：11.5k × 1152 × 2B ≈ 27MB）
    if paths["npy"].exists() and force:
        paths["npy"].unlink()
    if not paths["npy"].exists():
        np.lib.format.open_memmap(paths["npy"], mode="w+", dtype=np.float16, shape=(n, dim))
        write_done(paths["done"], 0)

    arr = np.lib.format.open_memmap(paths["npy"], mode="r+")
    if arr.shape != (n, dim):
        raise ValueError(
            f"缓存维度不符：{paths['npy']} 是 {arr.shape}，期望 {(n, dim)}。"
            "请删除该缓存或加 --force 重抽。"
        )

    n_done = read_done(paths["done"])
    if n_done >= n:
        print(f"[done] {name}：已全部完成")
    else:
        proc = get_image_processor(backbone_type, model_path, resolution, cfg["preprocess"]["mode"])
        amp_dtype = {
            "bf16": torch.bfloat16,
            "fp16": torch.float16,
            "none": None,
        }[cfg["features"].get("amp", "bf16")]

        batch_size = int(cfg["features"]["batch_size"])
        flush_every = int(cfg["features"]["flush_every"])
        started = time.time()

        # 只跑未完成的样本。注意这里按行号直接循环，不 shuffle，保证行序与 records 对齐。
        with torch.no_grad():
            i = n_done
            while i < n:
                batch_records = records[i : i + batch_size]
                tensors = []
                for rec in batch_records:
                    with Image.open(rec["path"]) as im:
                        tensors.append(proc(im.convert("RGB")))
                pixel_values = torch.stack(tensors).to(device, non_blocking=True)

                if amp_dtype is not None and device.type == "cuda":
                    with torch.autocast(device_type="cuda", dtype=amp_dtype):
                        feats = backbone(pixel_values)
                else:
                    feats = backbone(pixel_values)

                arr[i : i + len(batch_records)] = feats.detach().float().cpu().numpy().astype(np.float16)
                i += len(batch_records)

                if (i // batch_size) % flush_every == 0 or i >= n:
                    arr.flush()
                    write_done(paths["done"], i)
                    elapsed = time.time() - started
                    rate = (i - n_done) / elapsed if elapsed > 0 else 0
                    eta = (n - i) / rate if rate > 0 else 0
                    print(
                        f"    {i}/{n}  {rate:6.1f} img/s  "
                        f"已用 {elapsed / 60:.1f} min  剩余约 {eta / 60:.1f} min",
                        flush=True,
                    )
        del arr

    # 写完 index 与 meta
    with paths["index"].open("w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    meta = {
        "name": name,
        "backbone": backbone_type,
        "model_name_or_path": model_path,
        "resolution": resolution,
        "preprocess": cfg["preprocess"]["mode"],
        "feat_dim": dim,
        "n_images": n,
        "dtype": "float16",
        "amp": cfg["features"].get("amp"),
        "torch": torch.__version__,
        "splits_file": str(cfg["data"]["splits_out"]),
    }
    paths["meta"].write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[ok] {name}: {paths['npy'].name}  shape=({n}, {dim})")
    return meta


def results_ok(meta: dict, paths: dict[str, Path], n: int) -> bool:
    """快速校验缓存文件形状是否与 meta 一致。"""
    try:
        arr = np.lib.format.open_memmap(paths["npy"], mode="r")
        return arr.shape == (n, int(meta["feat_dim"]))
    except Exception:
        return False


def candidate_tag(candidates: list[dict]) -> str:
    """给 resolved_config 的文件名生成一个能区分「这次跑了哪些候选」的 tag。

    为什么需要它：本脚本最常见的并行用法是**一个候选一个进程**（见 README 的单机多卡
    抽取）。如果 resolved_config 用固定文件名，这些进程会同时往同一个路径写，
    两次 write_text 撞在一起可能写出**截断的 YAML**，而且事后无法分辨是哪个进程写的。

    单候选直接用候选名（并发的常见形态，名字本身可读）；
    多候选时名字加起来太长，退化成「数量 + 名字集合的短哈希」——
    同一组候选必然算出同一个 tag，所以重复跑全量不会堆出一堆文件。
    """
    names = [c["name"] for c in candidates]
    if len(names) == 1:
        return names[0]
    joined = "-".join(names)
    if len(joined) <= 48:
        return joined
    return f"{len(names)}cands-{hashlib.sha1(joined.encode()).hexdigest()[:8]}"


def build_candidates(cfg, args) -> list[dict]:
    candidates = list(cfg.get("candidates") or [])
    if not candidates:
        raise ValueError(
            "配置里没有 candidates。请同时加载 configs/bakeoff.yaml，"
            "或在 YAML 里用 model.backbone / model.model_name_or_path 指定单个模型。"
        )

    # 分辨率消融：用 RoPE 的 dinov3 复核「分辨率不是杠杆」这一结论
    if getattr(args, "include_ablation", False):
        ab = cfg.get("resolution_ablation") or {}
        base = next((c for c in candidates if c["name"] == ab.get("base_candidate")), None)
        if base is None:
            raise ValueError(
                f"resolution_ablation.base_candidate="
                f"{ab.get('base_candidate')!r} 不在 candidates 里"
            )
        extra = dict(base)
        extra["name"] = base["name"] + ab.get("name_suffix", "_r448")
        extra["resolution"] = ab["resolution"]
        candidates.append(extra)

    if args.only:
        wanted = set(args.only)
        candidates = [c for c in candidates if c["name"] in wanted]
        missing = wanted - {c["name"] for c in candidates}
        if missing:
            raise ValueError(f"--only 指定的候选不存在：{sorted(missing)}")
    return candidates


def main() -> int:
    parser = add_config_args(argparse.ArgumentParser(description="抽取并缓存冻结 backbone 特征"))
    parser.add_argument("--only", nargs="*", default=None, help="只跑这些候选（按 name）")
    parser.add_argument("--include-ablation", action="store_true", help="额外跑分辨率消融候选")
    parser.add_argument("--force", action="store_true", help="忽略已有缓存，重抽")
    args = parser.parse_args()

    cfg = config_from_args(args)
    setup_hf_env(cfg)
    load_class_config(cfg)  # 提前校验类别配置可用

    splits_file = Path(cfg["data"]["splits_out"])
    if not splits_file.exists():
        print(f"[error] 划分文件不存在：{splits_file}\n请先运行 scripts/prepare_data.py", file=sys.stderr)
        return 1

    records = load_records(splits_file)
    out_dir = Path(cfg["features"]["out_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(cfg["runtime"]["device"] if torch.cuda.is_available() else "cpu")
    print(f"设备：{device}")
    print(f"待抽特征图片：{len(records)} 张")
    print(f"缓存目录：{out_dir}")

    candidates = build_candidates(cfg, args)
    save_resolved(cfg, out_dir, f"resolved_config.extract_features.{candidate_tag(candidates)}.yaml")

    metas = []
    for candidate in candidates:
        try:
            metas.append(extract_one(candidate, records, cfg, out_dir, device, force=args.force))
        except Exception as exc:
            print(f"[fail] {candidate['name']}：{type(exc).__name__}: {exc}", file=sys.stderr)
            if len(candidates) == 1:
                raise

    print(f"\n{'=' * 70}\n完成 {len(metas)}/{len(candidates)} 个候选")
    for meta in metas:
        print(f"  {meta['name']:<22} D={meta['feat_dim']:<5} n={meta['n_images']}")
    print("\n下一步：python scripts/linear_probe.py --config configs/base.yaml --config configs/bakeoff.yaml")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
