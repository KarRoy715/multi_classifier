#!/usr/bin/env python3
"""按 suggested_label 把 *_correct.jsonl / *_errors.jsonl 里的图片拷贝到目标目录。

- 读 label_review 下所有 *_correct.jsonl 与 *_errors.jsonl（JSONL，每行一条）。
- 以每条记录的 suggested_label 作为子文件夹名，把 image_path 指向的图片拷入。
- 源文件缺失则跳过并记入清单；目标同名冲突跳过并记入清单。

用法（可传参，通用）:
    python3 label_review/copy_rejudged.py [src_dir] [dst_dir]
默认: src_dir=label_review  dst_dir=multi_data/data0910/rejudged
"""
import glob
import json
import os
import shutil
import sys
import collections


def main():
    src_dir = sys.argv[1] if len(sys.argv) > 1 else "label_review"
    dst_dir = sys.argv[2] if len(sys.argv) > 2 else "multi_data/data0910/rejudged"

    files = sorted(glob.glob(os.path.join(src_dir, "*_correct.jsonl"))
                   + glob.glob(os.path.join(src_dir, "*_errors.jsonl")))
    if not files:
        print(f"未在 {src_dir} 下找到 *_correct.jsonl / *_errors.jsonl")
        return

    label_count = collections.Counter()
    copied = 0
    missing = []      # (src, src_file, label)
    collisions = []   # (src, dst)

    for f in files:
        with open(f) as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    d = json.loads(line)
                except json.JSONDecodeError:
                    continue

                label = d.get("suggested_label")
                src = d.get("image_path")
                if not label or not src:
                    continue

                label_count[label] += 1
                if not os.path.exists(src):
                    missing.append((src, os.path.basename(f), label))
                    continue

                dst = os.path.join(dst_dir, label, os.path.basename(src))
                if os.path.exists(dst):
                    collisions.append((src, dst))
                    continue
                os.makedirs(os.path.dirname(dst), exist_ok=True)
                shutil.copy2(src, dst)
                copied += 1

    print(f"输入文件数: {len(files)}")
    print(f"成功拷贝: {copied}")
    print(f"源文件缺失(跳过): {len(missing)}")
    print(f"目标同名冲突(跳过): {len(collisions)}")
    print(f"子文件夹数: {len(label_count)}")
    for k, v in sorted(label_count.items(), key=lambda x: -x[1]):
        print(f"  {v:5d}  {k}")

    if missing or collisions:
        manifest = os.path.join(dst_dir, "_copy_manifest.jsonl")
        with open(manifest, "w") as fh:
            for src, sf, label in missing:
                fh.write(json.dumps(
                    {"image_path": src, "src_file": sf,
                     "suggested_label": label, "status": "missing"},
                    ensure_ascii=False) + "\n")
            for src, dst in collisions:
                fh.write(json.dumps(
                    {"image_path": src, "dst": dst, "status": "collision"},
                    ensure_ascii=False) + "\n")
        print(f"\n缺失/冲突清单已写入: {manifest}")


if __name__ == "__main__":
    main()
