"""15 类推理。支持单张图与目录批处理。

**不做拒识**：按约定 `不相关` 这一类从数据划分开始就被排除，本项目只做 15 类
单级分类，无关图片的过滤由项目外的环节负责。因此这里对所有输入都会给出一个
十五选一的答案，并附带置信度与 top-k，由使用方自行决定是否采信。

预处理严格按 checkpoint 里记录的 `backbone` / `resolution` / `preprocess` 走，
与 evaluate.py 同一套逻辑——训练与推理的预处理不一致会静默掉点，且不报错。

用法：
  python scripts/predict.py --config configs/base.yaml --input 某张图.jpg
  python scripts/predict.py --config configs/base.yaml --input 某个目录/ --out preds.csv
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import build_from_checkpoint, load_checkpoint  # noqa: E402
from config import (  # noqa: E402
    add_config_args,
    config_from_args,
    load_class_config,
    resolve_model_path,
    setup_hf_env,
)
from dataset import IMAGE_SUFFIXES, get_image_processor  # noqa: E402
from PIL import Image  # noqa: E402

Image.MAX_IMAGE_PIXELS = None


def collect_inputs(target: Path) -> list[Path]:
    """单文件直接用；目录则递归收集所有图片。"""
    if target.is_file():
        return [target]
    if target.is_dir():
        files = sorted(
            p for p in target.rglob("*")
            if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES
        )
        if not files:
            raise FileNotFoundError(f"目录里没有找到图片：{target}")
        return files
    raise FileNotFoundError(f"输入路径不存在：{target}")


@torch.no_grad()
def predict_paths(
    paths: list[Path],
    backbone,
    head,
    proc,
    device,
    class_names: list[str],
    top_k: int,
    batch_size: int,
    amp: bool,
    amp_dtype,
    want_all_probs: bool = False,
) -> list[dict]:
    """逐批推理。单张读图失败只跳过该张并告警，不让整批挂掉。

    want_all_probs 为真时，每张图额外带上全部类别的概率（批量模式导出完整
    CSV 用），否则只保留 top-k，避免结果集无谓膨胀。
    """
    results: list[dict] = []
    k = max(1, min(top_k, len(class_names)))

    for start in range(0, len(paths), batch_size):
        chunk = paths[start : start + batch_size]
        tensors, kept = [], []
        for p in chunk:
            try:
                with Image.open(p) as im:
                    tensors.append(proc(im.convert("RGB")))
                kept.append(p)
            except Exception as exc:
                print(f"  [warn] 无法读取 {p}：{exc}", file=sys.stderr)

        if not tensors:
            continue

        pixel_values = torch.stack(tensors).to(device, non_blocking=True)
        if amp:
            with torch.autocast(device_type="cuda", dtype=amp_dtype):
                logits = head(backbone(pixel_values))
        else:
            logits = head(backbone(pixel_values))
        probs = torch.softmax(logits.float(), dim=-1).cpu()

        top_p, top_i = probs.topk(k, dim=-1)
        for row, p in enumerate(kept):
            record = {
                "path": str(p),
                "pred": class_names[int(top_i[row, 0])],
                "prob": float(top_p[row, 0]),
                "topk": [
                    {"label": class_names[int(top_i[row, j])], "prob": float(top_p[row, j])}
                    for j in range(k)
                ],
            }
            if want_all_probs:
                record["all_probs"] = probs[row].tolist()
            results.append(record)
    return results


def main() -> int:
    parser = add_config_args(argparse.ArgumentParser(description="15 类图片推理"))
    parser.add_argument("--input", required=True, help="单张图片或目录")
    parser.add_argument("--checkpoint", default=None, help="覆盖 eval.checkpoint")
    parser.add_argument("--out", default=None, help="结果写到该 CSV；不给则打到 stdout")
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--show_all_probs", action="store_true",
                        help="单图模式下列出全部 15 类的概率")
    args = parser.parse_args()

    cfg = config_from_args(args)
    setup_hf_env(cfg)
    class_spec = load_class_config(cfg)
    class_names = class_spec["class_names"]

    ckpt_path = Path(args.checkpoint or cfg["eval"].get("checkpoint") or "checkpoints/best.pt")
    device = torch.device(cfg["runtime"]["device"] if torch.cuda.is_available() else "cpu")

    ckpt = load_checkpoint(ckpt_path, device)
    # 类序以 checkpoint 为准，否则标签会整体错位
    if ckpt.get("class_names") and list(ckpt["class_names"]) != class_names:
        print(
            "[warn] checkpoint 的 class_names 与 classes.yaml 不一致，按 checkpoint 的类序输出。",
            file=sys.stderr,
        )
        class_names = list(ckpt["class_names"])

    backbone, head = build_from_checkpoint(ckpt, cfg, device)
    model_name = ckpt.get("model_name_or_path") or resolve_model_path(cfg)
    proc = get_image_processor(
        ckpt["backbone"], str(model_name), ckpt.get("resolution"),
        ckpt.get("preprocess", cfg["preprocess"]["mode"]),
    )

    paths = collect_inputs(Path(args.input))
    top_k = int(cfg["predict"]["top_k"])
    batch_size = int(args.batch_size or cfg["predict"]["batch_size"])
    amp = cfg["train"]["amp"]
    amp_dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "none": None}[amp]

    print(f"checkpoint：{ckpt_path}")
    print(f"  backbone={ckpt['backbone']}  resolution={ckpt.get('resolution')}  "
          f"preprocess={ckpt.get('preprocess')}")
    print(f"待推理 {len(paths)} 张，输出 top-{top_k}\n")

    results = predict_paths(
        paths, backbone, head, proc, device, class_names, top_k, batch_size,
        amp_dtype is not None and device.type == "cuda", amp_dtype,
        want_all_probs=args.show_all_probs,
    )

    # ---- 单图：打成可读的表 ----
    if len(paths) == 1 and len(results) == 1:
        r = results[0]
        print(f"{r['path']}\n")
        for j, item in enumerate(r["topk"]):
            bar = "█" * int(round(item["prob"] * 40))
            print(f"  {j + 1}. {item['label']:<24} {item['prob']:>7.2%}  {bar}")
        if args.show_all_probs:
            print("\n全部类别概率（按概率降序）：")
            probs = r.get("all_probs")
            if probs is None:
                # 兜底：旧路径或 want_all_probs 未传时再补一次前向
                with torch.no_grad():
                    with Image.open(r["path"]) as im:
                        x = proc(im.convert("RGB")).unsqueeze(0).to(device)
                    logits = head(backbone(x))
                    probs = torch.softmax(logits.float(), dim=-1)[0].cpu().tolist()
            for name, prob in sorted(zip(class_names, probs), key=lambda t: -t[1]):
                print(f"  {name:<24} {prob:>7.2%}")

    # ---- 批量：CSV ----
    else:
        out_path = Path(args.out) if args.out else None
        header = ["path", "pred", "prob"] + [
            f"top{j + 1}_{s}" for j in range(top_k) for s in ("label", "prob")
        ]
        # --show_all_probs 在批量模式下展开成 15 列完整概率，供下游做阈值分析
        if args.show_all_probs:
            header += [f"p_{name}" for name in class_names]
        lines = [",".join(header)]
        for r in results:
            cells = [r["path"], r["pred"], f"{r['prob']:.4f}"]
            for item in r["topk"]:
                cells += [item["label"], f"{item['prob']:.4f}"]
            if args.show_all_probs:
                cells += [f"{p:.4f}" for p in r.get("all_probs", [])]
            lines.append(",".join(f'"{c}"' if "," in c else c for c in cells))
        text = "\n".join(lines) + "\n"
        if out_path:
            out_path.parent.mkdir(parents=True, exist_ok=True)
            out_path.write_text(text, encoding="utf-8")
            note = "（含全部 15 类概率）" if args.show_all_probs else ""
            print(f"已写出 {out_path}（{len(results)} 行）{note}")
        else:
            print(text)

        # 分布概览：一眼看出是不是全被预测成同一个类
        counts: dict[str, int] = {}
        for r in results:
            counts[r["pred"]] = counts.get(r["pred"], 0) + 1
        print("预测分布：")
        for name, n in sorted(counts.items(), key=lambda t: -t[1]):
            print(f"  {name:<24}{n:>6}  {n / len(results):>6.1%}")

    mean_conf = sum(r["prob"] for r in results) / max(len(results), 1)
    print(f"\n平均置信度 {mean_conf:.4f}（top-1 概率）")
    if mean_conf < 0.5:
        print("  提示：平均置信度偏低，说明模型对这批图不自信，结果需人工复核。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
