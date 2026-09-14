#!/usr/bin/env python3
"""Unified append script for audit results. Reads JSON array from stdin."""
import json, os, sys

proj = '/tcfs/fs_dev/zhaoaohui/Project/VLM/image_classifier'

def main():
    if len(sys.argv) < 2:
        print("Usage: python append_audit.py <category>", file=sys.stderr)
        sys.exit(1)
    category = sys.argv[1]
    data = sys.stdin.read().strip()
    if not data:
        print("No data on stdin.")
        return
    batch = json.loads(data)
    errors = [e for e in batch if e.get('suggested_label') != category]
    correct = [c for c in batch if c.get('suggested_label') == category]

    existing = set()
    for fname in [f'label_review/{category}_errors.jsonl', f'label_review/{category}_correct.jsonl']:
        path = os.path.join(proj, fname)
        if os.path.exists(path):
            with open(path, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        existing.add(json.loads(line)['image_path'])
                    except:
                        pass

    new_errors = [e for e in errors if e['image_path'] not in existing]
    new_correct = [c for c in correct if c['image_path'] not in existing]

    with open(os.path.join(proj, f'label_review/{category}_errors.jsonl'), 'a', encoding='utf-8') as f:
        for e in new_errors:
            f.write(json.dumps(e, ensure_ascii=False) + '\n')

    with open(os.path.join(proj, f'label_review/{category}_correct.jsonl'), 'a', encoding='utf-8') as f:
        for c in new_correct:
            f.write(json.dumps(c, ensure_ascii=False) + '\n')

    print(f"Added {len(new_errors)} errors, {len(new_correct)} correct.")
    if len(errors)-len(new_errors) > 0 or len(correct)-len(new_correct) > 0:
        print(f"Skipped {len(errors)-len(new_errors)} dup errors, {len(correct)-len(new_correct)} dup correct.")

if __name__ == '__main__':
    main()
