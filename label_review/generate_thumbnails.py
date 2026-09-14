#!/usr/bin/env python3
"""Generate thumbnails for all images referenced in label_review JSONL files."""
import json, os, sys
from pathlib import Path
from PIL import Image

PROJ_ROOT = Path('/tcfs/fs_dev/zhaoaohui/Project/VLM/image_classifier')
LABEL_REVIEW = PROJ_ROOT / 'label_review'
THUMB_DIR = LABEL_REVIEW / 'thumbnails'
THUMB_SIZE = (300, 300)

def main():
    # Collect unique image paths from all JSONL files
    image_paths = set()
    for jsonl_file in sorted(LABEL_REVIEW.glob('*.jsonl')):
        if jsonl_file.name.startswith('all_'):
            continue
        with open(jsonl_file, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                    image_paths.add(obj['image_path'])
                except Exception:
                    pass

    total = len(image_paths)
    print(f"Found {total} unique images. Generating thumbnails to {THUMB_DIR} ...")

    THUMB_DIR.mkdir(parents=True, exist_ok=True)

    success = 0
    skipped = 0
    failed = 0
    failed_list = []

    for i, rel_path in enumerate(sorted(image_paths), 1):
        src = PROJ_ROOT / rel_path
        thumb_path = THUMB_DIR / rel_path

        if thumb_path.exists():
            skipped += 1
            continue

        if not src.exists():
            failed += 1
            failed_list.append(rel_path)
            continue

        try:
            thumb_path.parent.mkdir(parents=True, exist_ok=True)
            with Image.open(src) as im:
                im.thumbnail(THUMB_SIZE, Image.LANCZOS)
                # Convert RGBA to RGB for JPEG if needed
                if im.mode in ('RGBA', 'P'):
                    im = im.convert('RGB')
                im.save(thumb_path, 'JPEG', quality=85)
            success += 1
        except Exception as e:
            failed += 1
            failed_list.append(f"{rel_path}: {e}")

        if i % 500 == 0 or i == total:
            print(f"  Progress: {i}/{total}  success={success} skipped={skipped} failed={failed}")

    print(f"\nDone. success={success}, skipped={skipped}, failed={failed}")
    if failed_list:
        print("Failed items (first 10):")
        for f in failed_list[:10]:
            print(f"  - {f}")

if __name__ == '__main__':
    main()
