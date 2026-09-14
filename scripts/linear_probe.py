"""在冻结特征缓存上跑线性探针，横向比较各 backbone —— 本项目的第一步实验。

为什么要先做这一步
------------------
特征已缓存，探针训练是秒级的，不碰 backbone。因此「哪个 backbone 适合本任务」这个问题
可以用极低成本回答，再决定把微调预算花在谁身上。这比直接微调 6 个模型省下大量算力。

三种探针各有用途
----------------
- LogisticRegression：标准线性探针，噪声鲁棒，是最公平的 backbone 比较基准
- MLP：结构等同于微调时的分类头，给出「微调起点大概能到多少」的参考上界
- kNN：零训练成本，反映特征空间本身的类聚性；若 kNN 就很强，说明特征本身已线性可分

指标口径
--------
本库 15 类严重不均衡，且 5 类样本极少（质量类仅 10 张、val 仅 2 张）。因此同时报告：
  - macro-F1（15 类全量）
  - macro-F1（仅 reliable 类）——**主排序依据**，避免 2 张 val 的类把结论带偏
"""

from __future__ import annotations

import argparse
import json
import sys
import warnings
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

# sklearn 1.9 起部分参数被废弃，会刷大量 FutureWarning 把关键输出淹掉。
# 这里只屏蔽 sklearn 自身的告警，其余照常抛出。
warnings.filterwarnings("ignore", category=FutureWarning, module="sklearn")

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import add_config_args, config_from_args, load_class_config, save_resolved  # noqa: E402
from extract_features import cache_paths, load_records  # noqa: E402

try:
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import (
        accuracy_score,
        classification_report,
        confusion_matrix,
        f1_score,
    )
    from sklearn.neighbors import KNeighborsClassifier
except ImportError:  # pragma: no cover
    print("[error] 需要 scikit-learn：pip install scikit-learn", file=sys.stderr)
    raise


def normalize_features(x: np.ndarray, mode: str) -> np.ndarray:
    """特征归一化。对 CLIP/DINO 这类模型，L2 归一化通常显著优于原始特征。"""
    if mode == "l2":
        norm = np.linalg.norm(x, axis=1, keepdims=True)
        return x / np.clip(norm, 1e-8, None)
    if mode == "standard":
        mean = x.mean(axis=0, keepdims=True)
        std = x.std(axis=0, keepdims=True)
        return (x - mean) / np.clip(std, 1e-8, None)
    if mode == "none":
        return x
    raise ValueError(f"未知的归一化方式：{mode}。支持 l2 / standard / none")


def load_cache(cache_dir: Path, name: str, splits_file: Path):
    """读特征缓存，按 split 切成 train/val。

    用 index.jsonl 里的 split 字段切分，而不是靠行号猜——保证与 splits.jsonl 一致。
    """
    paths = cache_paths(cache_dir, name)
    if not paths["npy"].exists() or not paths["meta"].exists():
        return None

    meta = json.loads(paths["meta"].read_text(encoding="utf-8"))
    feats = np.asarray(np.lib.format.open_memmap(paths["npy"], mode="r"), dtype=np.float32)

    records = []
    with paths["index"].open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))

    if len(records) != feats.shape[0]:
        raise ValueError(
            f"{name} 的 index 行数（{len(records)}）与特征行数（{feats.shape[0]}）不一致"
        )

    # 与当前划分文件对照：若划分已变，缓存的 split 字段会过时
    current = {r["path"]: r for r in load_records(splits_file)}
    stale = 0
    for rec in records:
        cur = current.get(rec["path"])
        if cur is None or cur["split"] != rec["split"] or cur["label"] != rec["label"]:
            stale += 1
    if stale:
        print(
            f"  [warn] {name}：{stale}/{len(records)} 条记录的划分或标签与当前 "
            f"splits.jsonl 不一致，缓存可能过时。建议重跑 extract_features.py。"
        )
        # 以当前划分文件为准（标签与 split 都重新采），特征本身仍可用
        for rec in records:
            cur = current.get(rec["path"])
            if cur is not None:
                rec["split"] = cur["split"]
                rec["label_index"] = cur["label_index"]
                rec["label"] = cur["label"]

    idx_tr = [i for i, r in enumerate(records) if r["split"] == "train"]
    idx_va = [i for i, r in enumerate(records) if r["split"] == "val"]
    y_tr = np.array([records[i]["label_index"] for i in idx_tr])
    y_va = np.array([records[i]["label_index"] for i in idx_va])
    return feats[idx_tr], y_tr, feats[idx_va], y_va, meta


def reliable_macro_f1(y_true, y_pred, reliable_indices: list[int]) -> float:
    """只在这些类别上算 macro-F1（主口径）。类别缺失时按实际出现的类计算。"""
    present = [c for c in reliable_indices if (y_true == c).any() or (y_pred == c).any()]
    if not present:
        return float("nan")
    return float(f1_score(y_true, y_pred, labels=present, average="macro", zero_division=0))


def evaluate_predictions(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: list[str],
    reliable_indices: list[int],
) -> dict:
    labels = list(range(len(class_names)))
    report = classification_report(
        y_true, y_pred, labels=labels, target_names=class_names,
        output_dict=True, zero_division=0,
    )
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y_true, y_pred, labels=labels, average="weighted", zero_division=0)),
        "macro_f1_reliable": reliable_macro_f1(y_true, y_pred, reliable_indices),
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=labels).tolist(),
        "per_class": {
            name: {
                "precision": report[name]["precision"],
                "recall": report[name]["recall"],
                "f1": report[name]["f1-score"],
                "support": report[name]["support"],
            }
            for name in class_names
        },
    }


def probe_logreg(
    x_tr, y_tr, x_va, y_va, cs: list[float], max_iter: int, reliable_indices: list[int]
):
    """线性探针，按 C 扫描取验证集上主口径（可靠类 macro-F1）最优的一档。

    选 C 与最终报告用**同一个指标**，避免「按 A 选、按 B 报」的口径错位。
    注意这是在全 val 上选超参，属于轻度的 val 过拟合；因此本表只用来说明
    「哪个 backbone 的特征更线性可分」，不是最终性能估计——最终数字以
    train.py 微调后在 val 上的表现为准。
    """
    best = None
    for c in cs:
        clf = LogisticRegression(C=c, max_iter=max_iter, class_weight="balanced")
        clf.fit(x_tr, y_tr)
        pred = clf.predict(x_va)
        score = reliable_macro_f1(y_va, pred, reliable_indices)
        if best is None or score > best[1]:
            best = (c, float(score), pred)
    return best[0], best[2]


def probe_knn(x_tr, y_tr, x_va, k: int):
    k = max(1, min(k, len(x_tr)))
    clf = KNeighborsClassifier(n_neighbors=k, weights="distance", n_jobs=-1)
    clf.fit(x_tr, y_tr)
    return clf.predict(x_va)


def probe_mlp(
    x_tr, y_tr, x_va, num_classes: int, cfg_probe: dict, device: torch.device, seed: int
):
    """小 MLP 探针，结构与 ClassifierHead 一致（微调起点参考）。"""
    torch.manual_seed(seed)
    hidden = int(cfg_probe.get("hidden", 256))
    model = nn.Sequential(
        nn.Dropout(float(cfg_probe.get("dropout", 0.3))),
        nn.Linear(x_tr.shape[1], hidden),
        nn.ReLU(inplace=True),
        nn.Dropout(float(cfg_probe.get("dropout", 0.3))),
        nn.Linear(hidden, num_classes),
    ).to(device)

    xt = torch.from_numpy(x_tr).to(device)
    yt = torch.from_numpy(y_tr).long().to(device)
    xv = torch.from_numpy(x_va).to(device)

    counts = np.bincount(y_tr, minlength=num_classes).astype(np.float64)
    counts[counts == 0] = 1.0
    weights = 1.0 / np.sqrt(counts)  # 逆频率开方，与 train.py 的 sqrt 口径一致
    weights = weights / weights.min()
    criterion = nn.CrossEntropyLoss(weight=torch.tensor(weights, dtype=torch.float32, device=device))

    epochs = int(cfg_probe.get("epochs", 40))
    opt = torch.optim.AdamW(model.parameters(), lr=float(cfg_probe.get("lr", 1e-3)),
                            weight_decay=float(cfg_probe.get("weight_decay", 1e-3)))
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    batch = int(cfg_probe.get("batch_size", 256))
    amp = cfg_probe.get("amp", "bf16")
    amp_dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "none": None}[amp]

    n = len(xt)
    for _ in range(epochs):
        model.train()
        perm = torch.randperm(n, device=device)
        for i in range(0, n, batch):
            sel = perm[i : i + batch]
            opt.zero_grad(set_to_none=True)
            if amp_dtype is not None and device.type == "cuda":
                with torch.autocast(device_type="cuda", dtype=amp_dtype):
                    loss = criterion(model(xt[sel]), yt[sel])
            else:
                loss = criterion(model(xt[sel]), yt[sel])
            loss.backward()
            opt.step()
        sched.step()

    model.eval()
    with torch.no_grad():
        preds = []
        for i in range(0, len(xv), 1024):
            preds.append(model(xv[i : i + 1024]).argmax(dim=-1))
        return torch.cat(preds).cpu().numpy()


def run_candidate(
    name: str,
    cache_dir: Path,
    splits_file: Path,
    cfg,
    class_names: list[str],
    reliable_indices: list[int],
    device: torch.device,
    probes: set[str],
) -> dict | None:
    loaded = load_cache(cache_dir, name, splits_file)
    if loaded is None:
        return None
    x_tr, y_tr, x_va, y_va, meta = loaded

    norm = cfg["probe"].get("normalize", "l2")
    x_tr = normalize_features(x_tr, norm)
    x_va = normalize_features(x_va, norm)

    num_classes = len(class_names)
    results: dict = {
        "name": name,
        "backbone": meta["backbone"],
        "resolution": meta["resolution"],
        "feat_dim": meta["feat_dim"],
        "n_train": int(len(y_tr)),
        "n_val": int(len(y_va)),
        "probes": {},
    }

    if "logreg" in probes:
        best_c, pred = probe_logreg(
            x_tr, y_tr, x_va, y_va, list(cfg["probe"]["cs"]),
            int(cfg["probe"]["max_iter"]), reliable_indices,
        )
        ev = evaluate_predictions(y_va, pred, class_names, reliable_indices)
        ev["best_C"] = best_c
        results["probes"]["logreg"] = ev
        print(
            f"    logreg  C={best_c:<6g} acc={ev['accuracy']:.4f} "
            f"macroF1={ev['macro_f1']:.4f} macroF1(可靠类)={ev['macro_f1_reliable']:.4f}"
        )

    if "knn" in probes:
        pred = probe_knn(x_tr, y_tr, x_va, int(cfg["probe"]["knn_k"]))
        ev = evaluate_predictions(y_va, pred, class_names, reliable_indices)
        results["probes"]["knn"] = ev
        print(
            f"    knn     k={cfg['probe']['knn_k']:<4d} acc={ev['accuracy']:.4f} "
            f"macroF1={ev['macro_f1']:.4f} macroF1(可靠类)={ev['macro_f1_reliable']:.4f}"
        )

    if "mlp" in probes:
        pred = probe_mlp(
            x_tr, y_tr, x_va, num_classes, cfg["probe"]["mlp"], device,
            int(cfg["probe"]["mlp"].get("seed", 42)),
        )
        ev = evaluate_predictions(y_va, pred, class_names, reliable_indices)
        results["probes"]["mlp"] = ev
        print(
            f"    mlp     {cfg['probe']['mlp']['epochs']}ep    acc={ev['accuracy']:.4f} "
            f"macroF1={ev['macro_f1']:.4f} macroF1(可靠类)={ev['macro_f1_reliable']:.4f}"
        )

    return results


def write_report(results: list[dict], class_names: list[str], out_dir: Path) -> None:
    """写出横向对比表（CSV + markdown）与逐类明细。"""
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for res in results:
        for probe_name, ev in res["probes"].items():
            rows.append({
                "candidate": res["name"],
                "backbone": res["backbone"],
                "resolution": res["resolution"],
                "feat_dim": res["feat_dim"],
                "probe": probe_name,
                "accuracy": round(ev["accuracy"], 4),
                "macro_f1": round(ev["macro_f1"], 4),
                "macro_f1_reliable": round(ev["macro_f1_reliable"], 4),
                "weighted_f1": round(ev["weighted_f1"], 4),
            })

    # 主排序：可靠类 macro-F1
    rows.sort(key=lambda r: (-r["macro_f1_reliable"], r["candidate"], r["probe"]))

    header = ["candidate", "backbone", "resolution", "feat_dim", "probe",
              "accuracy", "macro_f1", "macro_f1_reliable", "weighted_f1"]
    csv_path = out_dir / "bakeoff_summary.csv"
    with csv_path.open("w", encoding="utf-8") as f:
        f.write(",".join(header) + "\n")
        for row in rows:
            f.write(",".join(str(row[h]) for h in header) + "\n")

    md_path = out_dir / "bakeoff_summary.md"
    with md_path.open("w", encoding="utf-8") as f:
        f.write("# Backbone 大比拼结果\n\n")
        f.write("按 `macro_f1_reliable`（只统计样本充足的类别）降序排列。\n\n")
        f.write("| " + " | ".join(header) + " |\n")
        f.write("|" + "---|" * len(header) + "\n")
        for row in rows:
            f.write("| " + " | ".join(str(row[h]) for h in header) + " |\n")

        # 逐类明细
        f.write("\n## 逐类 F1（各候选 × 探针）\n\n")
        for res in results:
            for probe_name, ev in res["probes"].items():
                f.write(f"\n### {res['name']} / {probe_name}\n\n")
                f.write("| 类别 | precision | recall | f1 | support |\n|---|---|---|---|---|\n")
                for cname in class_names:
                    pc = ev["per_class"][cname]
                    f.write(
                        f"| {cname} | {pc['precision']:.3f} | {pc['recall']:.3f} "
                        f"| {pc['f1']:.3f} | {int(pc['support'])} |\n"
                    )

    json_path = out_dir / "bakeoff_full.json"
    json_path.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"\n已写出：\n  {csv_path}\n  {md_path}\n  {json_path}")


def main() -> int:
    parser = add_config_args(argparse.ArgumentParser(description="冻结特征线性探针大比拼"))
    parser.add_argument("--only", nargs="*", default=None, help="只评估这些候选")
    parser.add_argument("--probes", nargs="*", default=["logreg", "mlp", "knn"],
                        choices=["logreg", "mlp", "knn"])
    parser.add_argument("--out_dir", default="outputs/bakeoff")
    args = parser.parse_args()

    cfg = config_from_args(args)
    class_spec = load_class_config(cfg)
    class_names = class_spec["class_names"]
    reliable_indices = [
        c["index"] for c in class_spec["classes"] if not c.get("unreliable", False)
    ]

    splits_file = Path(cfg["data"]["splits_out"])
    cache_dir = Path(cfg["features"]["out_dir"])
    if not splits_file.exists():
        print(f"[error] 划分文件不存在：{splits_file}\n请先运行 scripts/prepare_data.py",
              file=sys.stderr)
        return 1

    candidates = list(cfg.get("candidates") or [])
    if not candidates:
        candidates = [{"name": "default", "backbone": cfg["model"]["backbone"]}]
    if args.only:
        wanted = set(args.only)
        candidates = [c for c in candidates if c["name"] in wanted]
        if not candidates:
            print(f"[error] --only 未匹配到任何候选：{sorted(wanted)}\n"
                  f"可用候选：{[c['name'] for c in (cfg.get('candidates') or [])]}",
                  file=sys.stderr)
            return 1

    device = torch.device(cfg["runtime"]["device"] if torch.cuda.is_available() else "cpu")
    out_dir = Path(args.out_dir)

    print(f"探针：{args.probes}")
    print(f"特征归一化：{cfg['probe'].get('normalize', 'l2')}")
    print(
        f"类别 {len(class_names)} 个；主口径 macro-F1 只统计 {len(reliable_indices)} 个可靠类，"
        f"另有 {len(class_names) - len(reliable_indices)} 类标记 unreliable（样本过少，指标不可信）"
    )
    print("=" * 78)

    results = []
    for candidate in candidates:
        name = candidate["name"]
        if not cache_paths(cache_dir, name)["npy"].exists():
            print(f"[skip] {name}：没有特征缓存，请先跑 extract_features.py")
            continue
        print(f"\n[{name}]")
        try:
            res = run_candidate(
                name, cache_dir, splits_file, cfg, class_names,
                reliable_indices, device, set(args.probes),
            )
            if res:
                results.append(res)
        except Exception as exc:
            print(f"[fail] {name}：{type(exc).__name__}: {exc}", file=sys.stderr)

    if not results:
        print("[error] 没有任何候选完成评估", file=sys.stderr)
        return 1

    write_report(results, class_names, out_dir)
    save_resolved(cfg, out_dir, "resolved_config.linear_probe.yaml")

    # 结论：挑出主口径最优的候选
    best = max(
        ((res, probe, ev) for res in results for probe, ev in res["probes"].items()),
        key=lambda t: t[2]["macro_f1_reliable"],
    )
    print(
        f"\n>>> 主口径最优：{best[0]['name']} / {best[1]}  "
        f"macro-F1(可靠类)={best[2]['macro_f1_reliable']:.4f}  "
        f"acc={best[2]['accuracy']:.4f}"
    )
    print(f">>> 建议用它作为微调起点：在 configs/exp/ 的 YAML 里设置 model.backbone 与 model.model_name_or_path")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
