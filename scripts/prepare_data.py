"""构建 15 类分层 train/val 划分，并做近重复防泄漏分组。

为什么需要近重复分组
--------------------
本数据集 md5 层面 0 重复，但 dHash 层面实测存在 **48 对完全相同的感知哈希**
（同一张图的不同压缩/编码），汉明距离 <=4 的近重复对估计约 1900 对。随机划分会把
这些同图变体劈到 train/val 两侧，验证指标被直接虚高。因此这里把近重复图片并成
「组」，**组整体分配**，保证同组不跨越 train/val。

产出的东西
----------
1. `data/splits.jsonl` —— 每行 {path, label, label_index, split, dup_group}
   这是特征缓存与线性探针用的事实来源（改划分不必重抽特征）。
2. `data/train/<类>/` 与 `data/val/<类>/` —— 物化目录树（默认符号链接，不占空间），
   供 train.py 沿用兄弟项目验证过的 torchvision ImageFolder 路径。

`不相关` 这一类从划分开始就被排除，既不计入统计也不物化。
"""

from __future__ import annotations

import argparse
import json
import os
import random
import shutil
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import (  # noqa: E402
    add_config_args,
    config_from_args,
    describe,
    load_class_config,
    save_resolved,
)

# 超大图不做 DecompressionBomb 拦截：本库有 3298px 的扫描件，且我们只读缩略图
Image.MAX_IMAGE_PIXELS = None

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff", ".gif"}


def compute_dhash(path: Path, hash_size: int = 8) -> np.ndarray | None:
    """64-bit 差值哈希。

    把图转灰度缩到 (hash_size+1, hash_size)，比较水平相邻像素大小关系。
    只用 PIL + numpy，不引入 imagehash 依赖（本机未装）。
    """
    try:
        with Image.open(path) as im:
            gray = im.convert("L").resize((hash_size + 1, hash_size), Image.BILINEAR)
        arr = np.asarray(gray, dtype=np.int16)
    except Exception as exc:  # 损坏文件不应让整个流程挂掉
        print(f"  [warn] 无法读取 {path}: {exc}", file=sys.stderr)
        return None
    return (arr[:, 1:] > arr[:, :-1]).flatten()


def pack_hashes(hashes: list[np.ndarray]) -> np.ndarray:
    """把布尔哈希打包成 uint8，便于用 np.bitwise_count 快速算汉明距离。"""
    return np.packbits(np.asarray(hashes, dtype=np.uint8), axis=1)


class UnionFind:
    def __init__(self, n: int):
        self.parent = list(range(n))
        self.rank = [0] * n

    def find(self, x: int) -> int:
        root = x
        while self.parent[root] != root:
            root = self.parent[root]
        while self.parent[x] != root:  # 路径压缩
            self.parent[x], x = root, self.parent[x]
        return root

    def union(self, a: int, b: int) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return
        if self.rank[ra] < self.rank[rb]:
            ra, rb = rb, ra
        self.parent[rb] = ra
        if self.rank[ra] == self.rank[rb]:
            self.rank[ra] += 1


def _popcount(x: np.ndarray) -> np.ndarray:
    """按字节统计置位数。优先用 numpy>=2.0 的 bitwise_count，否则退回查表。"""
    if hasattr(np, "bitwise_count"):
        return np.bitwise_count(x)
    table = np.array([bin(i).count("1") for i in range(256)], dtype=np.uint8)
    return table[x]


def group_near_duplicates(
    packed: np.ndarray, max_dist: int, labels: list[str]
) -> tuple[np.ndarray, int, int]:
    """返回 (每张图的组 id, 组数, 跨类冲突组数)。

    跨类近重复意味着同一张图在图库里有**两个不同标签**——那本身就是标签噪声。
    这里仍然把它们并成一组（避免同图跨 split），但单独统计出来告警。
    """
    n = len(packed)
    uf = UnionFind(n)
    if max_dist <= 0:
        return np.arange(n), n, 0

    # 逐行与全库比较。打包成 8 字节后，一行 XOR 全库只需 n*8 次字节运算，
    # popcount 用查表/bitwise_count，11.5k 张图秒级完成。
    for i in range(n):
        dists = _popcount(packed ^ packed[i]).sum(axis=1)
        dists[i] = 255  # 排除自身
        for j in np.nonzero(dists <= max_dist)[0]:
            uf.union(i, int(j))

    roots = np.array([uf.find(i) for i in range(n)])
    _, group_ids = np.unique(roots, return_inverse=True)

    label_by_group: dict[int, set[str]] = defaultdict(set)
    for gid, lab in zip(group_ids, labels):
        label_by_group[int(gid)].add(lab)
    conflicting = sum(1 for labs in label_by_group.values() if len(labs) > 1)
    return group_ids, int(group_ids.max()) + 1, conflicting


def assign_splits(
    labels: list[str],
    group_ids: np.ndarray,
    class_names: list[str],
    train_ratio: float,
    seed: int,
) -> tuple[list[str], list[str], int]:
    """按类分层、以「组」为单位分配 train/val。

    关键点：**组是全局单位**。一个近重复组可能横跨两个类别（本库实测有 87 个这样的组，
    本身就是标签噪声），若按类各处理一次，同一组会被分到两侧造成泄漏。因此这里对每个组
    只按其**多数标签**归属到一个类，然后整组（跨类的成员一起）分配。

    返回 (每张图的 split, 无 val 样本的类名, 因跨类归属而落在非本类的 val 图片数)。
    """
    rng = random.Random(seed)
    assignments = ["train"] * len(labels)
    val_empty: list[str] = []

    # 组 → 成员；组 → 多数标签
    members_of: dict[int, list[int]] = defaultdict(list)
    for i, gid in enumerate(group_ids):
        members_of[int(gid)].append(i)

    class_rank = {name: i for i, name in enumerate(class_names)}
    primary_of: dict[int, str] = {}
    for gid, members in members_of.items():
        counts = Counter(labels[i] for i in members)
        # 出现次数相同时按类名清单顺序取，保证确定性
        primary_of[gid] = max(counts.items(), key=lambda kv: (kv[1], -class_rank[kv[0]]))[0]

    for cls in class_names:
        cls_groups = [(gid, mem) for gid, mem in members_of.items() if primary_of[gid] == cls]
        if not cls_groups:
            continue

        # 先按 (大小, 组id) 排序再打乱，保证同种子下结果可复现
        ordered = sorted(cls_groups, key=lambda kv: (len(kv[1]), kv[0]))
        rng.shuffle(ordered)

        if len(ordered) == 1:
            # 整类就是一个近重复组：拆开必然泄漏，故整组留 train 并报告
            val_empty.append(cls)
            continue

        n_cls = sum(len(mem) for _, mem in ordered)
        target_train = train_ratio * n_cls
        n_train = 0
        for gid, members in ordered:
            # 第一组无条件进 train，避免 target_train 很小时 train 为空
            if n_train == 0 or n_train + len(members) <= target_train:
                n_train += len(members)
                continue
            for i in members:
                assignments[i] = "val"

        # 兜底：贪心可能一张都没分到 val（例如最大的组就超过 target_train），
        # 这时把最小的组挪去 val，保证每类都有可评估的 val 样本。
        if all(assignments[i] == "train" for gid, mem in ordered for i in mem):
            for i in ordered[0][1]:
                assignments[i] = "val"

    # 统计跨类归属的副作用：某张图进了 val，但它的标签不是该组的多数标签
    cross_class = sum(
        1
        for i, split in enumerate(assignments)
        if split == "val" and labels[i] != primary_of[int(group_ids[i])]
    )
    return assignments, val_empty, cross_class


def materialize(
    records: list[dict],
    out_root: Path,
    mode: str,
    class_names: list[str],
) -> None:
    """按 split/类 物化目录树。默认用**相对**符号链接，整棵树跟着数据一起搬也不会断。"""
    for split in ("train", "val"):
        for cls in class_names:
            (out_root / split / cls).mkdir(parents=True, exist_ok=True)

    for rec in records:
        src = Path(rec["path"])
        dest = out_root / rec["split"] / rec["label"] / src.name
        if dest.is_symlink() or dest.exists():
            dest.unlink()
        if mode == "symlink":
            # 相对链接：out_root/split/cls/ 指向 src
            rel = os.path.relpath(src, dest.parent)
            os.symlink(rel, dest)
        elif mode == "copy":
            shutil.copy2(src, dest)
        else:
            raise ValueError(f"未知的 materialize_mode：{mode}")


def collect_images(
    root: Path, class_names: list[str], aliases: dict[str, str], exclude: list[str]
) -> tuple[list[Path], list[str]]:
    """扫描图片根目录，返回 (路径, 类别名)。

    只收 class_names 里列出的类别；`不相关` 等被 exclude 的目录直接跳过。
    目录名命中 aliases 时自动映射到规范名（处理「器件资料类」→「器件资料」）。
    """
    paths: list[Path] = []
    labels: list[str] = []
    skipped: Counter[str] = Counter()

    for entry in sorted(root.iterdir()):
        if not entry.is_dir():
            continue
        name = entry.name
        if name in exclude:
            skipped[name] = len(list(entry.iterdir()))
            continue
        canonical = aliases.get(name, name)
        if canonical not in class_names:
            skipped[name] = len(list(entry.iterdir()))
            continue
        for f in sorted(entry.iterdir()):
            if f.is_file() and f.suffix.lower() in IMAGE_SUFFIXES:
                paths.append(f)
                labels.append(canonical)

    if skipped:
        print("[跳过] 以下目录不在类别清单内（含「不相关」）：")
        for name, count in sorted(skipped.items()):
            print(f"    {name}: {count} 张")
    return paths, labels


def main() -> int:
    parser = add_config_args(argparse.ArgumentParser(description="构建 15 类 train/val 划分"))
    parser.add_argument("--force", action="store_true", help="覆盖已存在的物化目录")
    args = parser.parse_args()

    cfg = config_from_args(args)
    class_spec = load_class_config(cfg)
    class_names = class_spec["class_names"]

    root = Path(cfg["data"]["root"])
    if not root.exists():
        print(f"[error] 图片根目录不存在：{root}", file=sys.stderr)
        return 1

    print(f"配置：{describe(cfg)}")
    print(f"图片根目录：{root}")
    print(f"类别数：{len(class_names)}")

    paths, labels = collect_images(
        root, class_names, class_spec["aliases"], class_spec["exclude"]
    )
    if not paths:
        print("[error] 没有收集到任何图片", file=sys.stderr)
        return 1

    # ---- 1. dHash + 近重复分组 ----
    max_dist = int(cfg["data"]["phash_max_dist"])
    print(f"\n[1/4] 计算 dHash（{len(paths)} 张）...")
    hashes: list[np.ndarray] = []
    keep_idx: list[int] = []
    for i, p in enumerate(paths):
        h = compute_dhash(p)
        if h is not None:
            hashes.append(h)
            keep_idx.append(i)
        if (i + 1) % 2000 == 0:
            print(f"    {i + 1}/{len(paths)}")

    if len(keep_idx) != len(paths):
        print(f"  [warn] {len(paths) - len(keep_idx)} 张图无法读取，已排除")
    paths = [paths[i] for i in keep_idx]
    labels = [labels[i] for i in keep_idx]

    print(f"[2/4] 近重复分组（汉明距离 <= {max_dist}）...")
    packed = pack_hashes(hashes)
    group_ids, n_groups, n_conflicting = group_near_duplicates(packed, max_dist, labels)
    grouped = len(paths) - n_groups
    print(f"    图片 {len(paths)} 张 → {n_groups} 组（{grouped} 张与他图构成近重复）")
    if n_conflicting:
        print(
            f"  [warn] 有 {n_conflicting} 个组包含**不同标签**的近重复图片，"
            f"说明图库中存在同图不同标的标签噪声（这些组会被整体划到同一侧）"
        )

    # ---- 2. 分配 split ----
    print("[3/4] 分层分配 train/val...")
    assignments, val_empty, cross_class = assign_splits(
        labels, group_ids, class_names, float(cfg["data"]["train_ratio"]), int(cfg["data"]["seed"])
    )
    if val_empty:
        print(
            f"  [warn] 以下类别的全部图片构成单个近重复组，整组留在 train，"
            f"因此没有 val 样本（硬拆会造成泄漏，故不拆）：{', '.join(val_empty)}"
        )
    if cross_class:
        print(
            f"    {cross_class} 张跨类近重复图片随组归属落到了非本类的 val（组是全局单位，"
            f"这是避免泄漏的必要代价）"
        )

    # 自检：任何近重复组都不能跨越 train/val
    by_group: dict[int, set[str]] = defaultdict(set)
    for gid, split in zip(group_ids, assignments):
        by_group[int(gid)].add(split)
    leaked = [g for g, s in by_group.items() if len(s) > 1]
    assert not leaked, f"近重复组跨越了 train/val：{leaked[:5]}"

    # ---- 3. 写出 splits.jsonl ----
    records = []
    for path, label, gid, split in zip(paths, labels, group_ids, assignments):
        records.append({
            "path": str(path),
            "label": label,
            "label_index": class_names.index(label),
            "split": split,
            "dup_group": int(gid),
        })
    records.sort(key=lambda r: (r["split"], r["label_index"], r["path"]))

    splits_out = Path(cfg["data"]["splits_out"])
    splits_out.parent.mkdir(parents=True, exist_ok=True)
    with splits_out.open("w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    print(f"    写出 {splits_out} （{len(records)} 行）")

    # ---- 4. 物化目录树 ----
    mat_root = Path(cfg["data"]["materialize_dir"])
    mode = cfg["data"]["materialize_mode"]
    if mat_root.exists() and args.force:
        shutil.rmtree(mat_root / "train", ignore_errors=True)
        shutil.rmtree(mat_root / "val", ignore_errors=True)
    print(f"[4/4] 物化 {mat_root}/{{train,val}}/ （mode={mode}）...")
    materialize(records, mat_root, mode, class_names)

    # ---- 报告 ----
    unreliable = {c["name"] for c in class_spec["classes"] if c.get("unreliable")}
    train_counts = Counter(r["label"] for r in records if r["split"] == "train")
    val_counts = Counter(r["label"] for r in records if r["split"] == "val")

    print("\n" + "=" * 78)
    print(f"{'类别':<24}{'train':>8}{'val':>7}{'合计':>8}   备注")
    print("-" * 78)
    for name in class_names:
        tr, va = train_counts[name], val_counts[name]
        flags = []
        if name in unreliable:
            flags.append("unreliable")
        if va and va < 5:
            flags.append(f"val仅{va}张,指标不可信")
        print(f"{name:<24}{tr:>8}{va:>7}{tr + va:>8}   {' / '.join(flags)}")
    total_tr, total_va = sum(train_counts.values()), sum(val_counts.values())
    print("-" * 78)
    print(f"{'合计':<24}{total_tr:>8}{total_va:>7}{total_tr + total_va:>8}")

    n_reliable = len([n for n in class_names if n not in unreliable])
    print(
        f"\n类别数 {len(class_names)}（其中 {len(unreliable)} 类标记 unreliable、"
        f"{n_reliable} 类进入主口径 macro-F1）"
    )
    print(f"近重复组 {n_groups} 个，跨 split 泄漏 0 组（已断言）")
    print(f"「不相关」已按配置排除，未出现在划分中：{'不相关' not in set(labels)}")

    save_resolved(cfg, mat_root, "resolved_config.prepare_data.yaml")
    print(f"\n完成。下一步：python scripts/extract_features.py --config configs/base.yaml --config configs/bakeoff.yaml")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
