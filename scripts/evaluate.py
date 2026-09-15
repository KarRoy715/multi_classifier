"""在验证/测试集上评估 checkpoint，产出可读的逐类报告与混淆矩阵。

核心约定：**重建模型与预处理一律以 checkpoint 里记录的元信息为准**，
配置里写的值只在 checkpoint 缺字段时兜底。否则很容易出现「训练用
letterbox@336、评估却按配置的 crop@224」这种不报错、只掉点的错配。

指标双口径（本库有 5 类样本极少，必须分开看）：
  macro_f1            —— 15 类全量，与常规口径可比
  macro_f1_reliable   —— 只统计样本充足的 10 类，**判断模型好坏看这个**

产出（默认 outputs/eval/）：
  report.txt            逐类 P/R/F1 + 汇总，人读
  metrics.json          机器读，含完整混淆矩阵
  predictions.csv       每张图的 true/pred/置信度，用来人工抽查错例
  confusion_matrix.png  原始计数 + 行归一化，并排
  top_confusions.md     最易混类对排行（含占比），指导后续补数据
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import build_from_checkpoint, load_checkpoint  # noqa: E402
from config import (  # noqa: E402
    add_config_args,
    config_from_args,
    describe,
    load_class_config,
    resolve_model_path,
    save_resolved,
    setup_hf_env,
)
from dataset import build_dataset, get_dataloaders, get_image_processor, make_collate  # noqa: E402

# 中文字体候选。找不到就退回用类别序号当标签——总比渲染成一堆方框强。
_CJK_FONTS = [
    "Noto Sans CJK SC", "Noto Sans CJK JP", "Source Han Sans CN",
    "WenQuanYi Zen Hei", "WenQuanYi Micro Hei", "Droid Sans Fallback",
    "SimHei", "Microsoft YaHei",
]


def setup_cjk_font() -> bool:
    from matplotlib import font_manager

    available = {f.name for f in font_manager.fontManager.ttflist}
    for name in _CJK_FONTS:
        if name in available:
            import matplotlib

            matplotlib.rcParams["font.sans-serif"] = [name]
            matplotlib.rcParams["axes.unicode_minus"] = False
            return True
    print(
        "[warn] 未找到中文字体，混淆矩阵将用类别序号标注。"
        "安装 Noto Sans CJK 后重跑可获得中文标签。",
        file=sys.stderr,
    )
    return False


@torch.no_grad()
def infer(backbone, head, loader, device, num_classes: int, amp: bool, amp_dtype):
    """跑一遍推理，返回 (真实标签, 预测标签, 各类概率)。"""
    all_probs, all_true, all_pred = [], [], []
    for pixel_values, labels in loader:
        pixel_values = pixel_values.to(device, non_blocking=True)
        if amp:
            with torch.autocast(device_type="cuda", dtype=amp_dtype):
                logits = head(backbone(pixel_values))
        else:
            logits = head(backbone(pixel_values))
        probs = torch.softmax(logits.float(), dim=-1)
        all_probs.append(probs.cpu().numpy())
        all_true.append(labels.numpy())
        all_pred.append(probs.argmax(dim=-1).cpu().numpy())
    return (
        np.concatenate(all_true),
        np.concatenate(all_pred),
        np.concatenate(all_probs),
    )


def confusion_matrix_np(y_true: np.ndarray, y_pred: np.ndarray, n: int) -> np.ndarray:
    cm = np.zeros((n, n), dtype=np.int64)
    np.add.at(cm, (y_true, y_pred), 1)
    return cm


def per_class_metrics(cm: np.ndarray) -> dict:
    """从混淆矩阵算逐类 P/R/F1。zero_division 显式处理，避免 nan 传染。"""
    tp = np.diag(cm).astype(np.float64)
    fp = cm.sum(axis=0) - np.diag(cm)
    fn = cm.sum(axis=1) - np.diag(cm)
    support = cm.sum(axis=1).astype(np.int64)

    precision = np.divide(tp, tp + fp, out=np.zeros_like(tp), where=(tp + fp) > 0)
    recall = np.divide(tp, tp + fn, out=np.zeros_like(tp), where=(tp + fn) > 0)
    f1 = np.divide(2 * precision * recall, precision + recall,
                   out=np.zeros_like(tp), where=(precision + recall) > 0)
    return {"precision": precision, "recall": recall, "f1": f1, "support": support}


def summarise(cm: np.ndarray, class_names: list[str], reliable_indices: list[int]) -> dict:
    pc = per_class_metrics(cm)
    total = cm.sum()
    weights = pc["support"] / max(total, 1)

    def _macro(key: str, idx: list[int] | None = None) -> float:
        vals = pc[key] if idx is None else pc[key][idx]
        return float(vals.mean()) if len(vals) else 0.0

    return {
        "n_samples": int(total),
        "accuracy": float(np.diag(cm).sum() / max(total, 1)),
        "macro_precision": _macro("precision"),
        "macro_recall": _macro("recall"),
        "macro_f1": _macro("f1"),
        "weighted_f1": float((pc["f1"] * weights).sum()),
        "macro_f1_reliable": _macro("f1", reliable_indices),
        "macro_precision_reliable": _macro("precision", reliable_indices),
        "macro_recall_reliable": _macro("recall", reliable_indices),
        "per_class": {
            name: {
                "precision": float(pc["precision"][i]),
                "recall": float(pc["recall"][i]),
                "f1": float(pc["f1"][i]),
                "support": int(pc["support"][i]),
            }
            for i, name in enumerate(class_names)
        },
        "confusion_matrix": cm.tolist(),
    }


def format_report(summary: dict, class_names: list[str], unreliable: set[str]) -> str:
    lines = [
        f"样本数 {summary['n_samples']}    准确率 {summary['accuracy']:.4f}",
        "",
        f"{'类别':<24}{'P':>8}{'R':>8}{'F1':>8}{'support':>9}   备注",
        "-" * 78,
    ]
    for name in class_names:
        m = summary["per_class"][name]
        flags = []
        if name in unreliable:
            flags.append("unreliable")
        if m["support"] and m["support"] < 5:
            flags.append(f"仅{m['support']}张")
        lines.append(
            f"{name:<24}{m['precision']:>8.3f}{m['recall']:>8.3f}{m['f1']:>8.3f}"
            f"{m['support']:>9}   {' / '.join(flags)}"
        )
    lines += [
        "-" * 78,
        "",
        f"macro-F1 (15 类全量)      {summary['macro_f1']:.4f}",
        f"macro-F1 (仅可靠类)        {summary['macro_f1_reliable']:.4f}   <-- 判断模型好坏看这个",
        f"weighted-F1               {summary['weighted_f1']:.4f}",
        f"macro-P / macro-R         {summary['macro_precision']:.4f} / {summary['macro_recall']:.4f}",
    ]
    if unreliable:
        lines += [
            "",
            f"注：{len(unreliable)} 个类标为 unreliable（样本过少或类内一致性无保证）：",
            f"    {'、'.join(sorted(unreliable))}",
            "    它们的逐类指标波动极大，不要据此下结论。",
        ]
    return "\n".join(lines)


def top_confusions(cm: np.ndarray, class_names: list[str], top_n: int) -> list[dict]:
    """最易混类对。按「占真实类样本的比例」排序，而不是绝对计数——
    否则 4626 张的大类会霸榜，把小类的问题盖掉。"""
    rows = []
    for i in range(len(class_names)):
        support = cm[i].sum()
        if support == 0:
            continue
        for j in range(len(class_names)):
            if i == j or cm[i, j] == 0:
                continue
            rows.append({
                "true": class_names[i],
                "pred": class_names[j],
                "count": int(cm[i, j]),
                "rate_of_true": float(cm[i, j] / support),
                "true_support": int(support),
            })
    rows.sort(key=lambda r: (-r["rate_of_true"], -r["count"]))
    return rows[:top_n]


def format_confusions(rows: list[dict], class_names: list[str]) -> str:
    lines = [
        "最易混类对（按「占真实类样本比例」排序，绝对计数会偏向大 class）",
        "",
        f"{'真实类别':<24}{'误判为':<24}{'张数':>7}{'占该类':>9}",
        "-" * 78,
    ]
    for r in rows:
        lines.append(
            f"{r['true']:<24}{r['pred']:<24}{r['count']:>7}{r['rate_of_true']:>8.1%}"
        )
    lines.append("")
    lines.append("这些类对优先考虑：补该类样本 / 检查两者标注边界是否一致。")
    return "\n".join(lines)


def plot_confusion(
    cm: np.ndarray,
    class_names: list[str],
    out_path: Path,
    use_cjk: bool,
    title_suffix: str = "",
) -> None:
    """并排画原始计数与行归一化两张混淆矩阵。

    行归一化那张更重要：不均衡数据下原始计数会被大类的数量淹没，
    看不出小类到底错在哪。
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n = len(class_names)
    labels = class_names if use_cjk else [str(i) for i in range(n)]
    row_norm = cm / np.clip(cm.sum(axis=1, keepdims=True), 1, None)

    fig, axes = plt.subplots(1, 2, figsize=(max(18, n * 1.1), max(9, n * 0.62)))
    for ax, data, title, fmt, vmax in (
        (axes[0], cm, "混淆矩阵（原始计数）", "d", None),
        (axes[1], row_norm, "混淆矩阵（行归一化）", ".2f", 1.0),
    ):
        im = ax.imshow(data, cmap="Blues", vmax=vmax)
        ax.set_xticks(range(n), labels, rotation=90, fontsize=8)
        ax.set_yticks(range(n), labels, fontsize=8)
        ax.set_xlabel("预测", fontsize=10)
        ax.set_ylabel("真实", fontsize=10)
        ax.set_title(title + title_suffix, fontsize=11)
        # 数值标注：类别多的时候只标非零项，否则一片糊
        threshold = data.max() * 0.5 if vmax is None else 0.5
        for i in range(n):
            for j in range(n):
                if data[i, j] == 0:
                    continue
                ax.text(
                    j, i, format(data[i, j], fmt), ha="center", va="center",
                    fontsize=7 if n > 12 else 8,
                    color="white" if data[i, j] > threshold else "black",
                )
        fig.colorbar(im, ax=ax, fraction=0.046)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    parser = add_config_args(argparse.ArgumentParser(description="评估 checkpoint"))
    parser.add_argument("--split", default="val", choices=["val", "train"],
                        help="在哪个划分上评估。train 用来看是否过拟合")
    parser.add_argument("--checkpoint", default=None, help="覆盖 eval.checkpoint")
    parser.add_argument("--out_dir", default=None, help="覆盖 eval.out_dir")
    parser.add_argument("--no_plot", action="store_true", help="不画混淆矩阵")
    args = parser.parse_args()

    cfg = config_from_args(args)
    setup_hf_env(cfg)
    class_spec = load_class_config(cfg)
    class_names = class_spec["class_names"]
    unreliable = {c["name"] for c in class_spec["classes"] if c.get("unreliable")}
    reliable_indices = [c["index"] for c in class_spec["classes"] if not c.get("unreliable")]

    ckpt_path = Path(args.checkpoint or cfg["eval"].get("checkpoint") or "checkpoints/best.pt")
    out_dir = Path(args.out_dir or cfg["eval"]["out_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(cfg["runtime"]["device"] if torch.cuda.is_available() else "cpu")

    print(describe(cfg))
    ckpt = load_checkpoint(ckpt_path, device)

    # 用 checkpoint 的元信息覆盖配置：这几项决定预处理，错一项就静默掉点
    bb_type = ckpt["backbone"]
    resolution = ckpt.get("resolution")
    preprocess = ckpt.get("preprocess", cfg["preprocess"]["mode"])
    ckpt_classes = ckpt.get("class_names")
    if ckpt_classes is not None and list(ckpt_classes) != class_names:
        print(
            "[warn] checkpoint 里的 class_names 与当前 classes.yaml 不一致！\n"
            f"  checkpoint: {list(ckpt_classes)}\n"
            f"  classes.yaml: {class_names}\n"
            "  继续按 checkpoint 的类序评估（否则标签会整体错位）。",
            file=sys.stderr,
        )
        class_names = list(ckpt_classes)
        unreliable = {n for n in unreliable if n in class_names}
        reliable_indices = [i for i, n in enumerate(class_names) if n not in unreliable]

    num_classes = len(class_names)
    if int(ckpt["num_classes"]) != num_classes:
        raise ValueError(
            f"checkpoint 的 num_classes={ckpt['num_classes']} 与类别清单的 {num_classes} 不一致"
        )

    print(f"\ncheckpoint：{ckpt_path}")
    print(f"  最优 epoch={ckpt.get('metrics', {}).get('epoch', '?')}  "
          f"backbone={bb_type}  resolution={resolution}  preprocess={preprocess}")
    print(f"  评估划分：{args.split}")

    backbone, head = build_from_checkpoint(ckpt, cfg, device)

    # 预处理严格按 checkpoint 走：先解析模型路径，再用 ckpt 的 backbone/resolution
    model_name = ckpt.get("model_name_or_path") or resolve_model_path(cfg)
    proc = get_image_processor(bb_type, str(model_name), resolution, preprocess)

    splits_file = Path(cfg["data"]["splits_out"])
    ds = build_dataset(
        "manifest", args.split, cfg["data"]["materialize_dir"],
        splits_file, None, class_names,
    )
    _, loader = get_dataloaders(
        None, ds, int(cfg["eval"]["batch_size"]), None, make_collate(proc),
        num_workers=int(cfg["train"]["num_workers"]),
    )
    print(f"样本数：{len(ds)}（{args.split}）\n")

    amp = cfg.get_path("eval.amp")
    if amp is None:
        amp = cfg["train"]["amp"]
    amp_dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "none": None}[amp]
    y_true, y_pred, probs = infer(
        backbone, head, loader, device, num_classes,
        amp_dtype is not None and device.type == "cuda", amp_dtype,
    )

    cm = confusion_matrix_np(y_true, y_pred, num_classes)
    summary = summarise(cm, class_names, reliable_indices)
    confusions = top_confusions(cm, class_names, int(cfg["eval"]["top_confusions"]))

    print(format_report(summary, class_names, unreliable))
    print()
    print(format_confusions(confusions, class_names))

    # ---- 落盘 ----
    (out_dir / "report.txt").write_text(
        f"checkpoint: {ckpt_path}\nsplit: {args.split}\n\n"
        + format_report(summary, class_names, unreliable) + "\n\n"
        + format_confusions(confusions, class_names) + "\n",
        encoding="utf-8",
    )
    summary["checkpoint"] = str(ckpt_path)
    summary["split"] = args.split
    summary["top_confusions"] = confusions
    (out_dir / "metrics.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (out_dir / "top_confusions.md").write_text(
        format_confusions(confusions, class_names) + "\n", encoding="utf-8"
    )

    # 逐样本预测，用于人工抽查错例
    paths = list(getattr(ds, "paths", []))
    with (out_dir / "predictions.csv").open("w", encoding="utf-8") as f:
        f.write("path,true,pred,correct,confidence\n")
        for i in range(len(y_true)):
            conf = float(probs[i, y_pred[i]])
            p = paths[i] if i < len(paths) else ""
            # 路径里可能有逗号，做最小转义
            p = f'"{p}"' if "," in str(p) else str(p)
            f.write(
                f"{p},{class_names[y_true[i]]},{class_names[y_pred[i]]},"
                f"{int(y_true[i] == y_pred[i])},{conf:.4f}\n"
            )

    if not args.no_plot and cfg["eval"]["confusion_matrix_png"]:
        # 类多时用序号当标签更快能看清，但中文可读性更好——优先中文
        use_cjk = setup_cjk_font()
        png = out_dir / "confusion_matrix.png"
        plot_confusion(cm, class_names, png, use_cjk, f"（{args.split}）")
        print(f"\n混淆矩阵：{png}")

    save_resolved(cfg, out_dir, "resolved_config.evaluate.yaml")

    # ---- 错例诊断 ----
    wrong = y_true != y_pred
    if wrong.any():
        print(f"\n错例 {wrong.sum()}/{len(y_true)}（{wrong.mean():.1%}）；"
              f"置信度最高的错例通常最值得看：")
        # argsort 给出的是**样本下标**，因此要先用 order 重排 wrong 掩码，
        # 再筛出判错的那些，否则掩码与下标错位、会挑出判对的样本。
        conf = probs[range(len(y_true)), y_pred]
        order = np.argsort(-conf)
        for i in order[wrong[order]][:5]:
            print(f"  {conf[i]:.3f}  真={class_names[y_true[i]]}  "
                  f"预测={class_names[y_pred[i]]}  {paths[i] if i < len(paths) else ''}")

    print(f"\n产物目录：{out_dir}/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
