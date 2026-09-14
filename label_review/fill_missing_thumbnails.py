#!/usr/bin/env python3
"""Fill missing thumbnails only. Much faster than full regeneration."""
import json, os
from pathlib import Path
from PIL import Image

PROJ_ROOT = Path('/tcfs/fs_dev/zhaoaohui/Project/VLM/image_classifier')
LABEL_REVIEW = PROJ_ROOT / 'label_review'
THUMB_DIR = LABEL_REVIEW / 'thumbnails'
THUMB_SIZE = (300, 300)

def main():
    # Collect all image paths that need thumbnails
    todo = []
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
                    p = obj['image_path']
                    # Strip absolute path prefix if present
                    if p.startswith('/tcfs/fs_dev/zhaoaohui/Project/VLM/image_classifier/'):
                        p = p[len('/tcfs/fs_dev/zhaoaohui/Project/VLM/image_classifier/'):]
                    src = PROJ_ROOT / p
                    thumb = THUMB_DIR / p
                    if not thumb.exists() and src.exists():
                        todo.append((src, thumb, p))
                except Exception:
                    pass

    total = len(todo)
    print(f"Missing thumbnails to generate: {total}")
    if total == 0:
        return

    success = 0
    failed = 0
    for i, (src, thumb, rel_path) in enumerate(todo, 1):
        try:
            thumb.parent.mkdir(parents=True, exist_ok=True)
            with Image.open(src) as im:
                im.thumbnail(THUMB_SIZE, Image.LANCZOS)
                if im.mode in ('RGBA', 'P'):
                    im = im.convert('RGB')
                im.save(thumb, 'JPEG', quality=85)
            success += 1
        except Exception as e:
            failed += 1
            if failed <= 5:
                print(f"  FAIL: {rel_path}: {e}")

        if i % 500 == 0 or i == total:
            print(f"  Progress: {i}/{total}  success={success} failed={failed}")

    print(f"\nDone. success={success}, failed={failed}")

if __name__ == '__main__':
    main()
