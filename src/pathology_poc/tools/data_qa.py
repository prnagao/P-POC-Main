import argparse, hashlib
from pathlib import Path
from PIL import Image

def sha1(p: Path) -> str:
    h = hashlib.sha1()
    with p.open('rb') as f:
        for chunk in iter(lambda: f.read(8192), b''):
            h.update(chunk)
    return h.hexdigest()

def main():
    ap = argparse.ArgumentParser(description="Basic dataset QA: duplicates, sizes, naive FOV leakage.")
    ap.add_argument("--data_root", default="dataset", help="Root with train/val/test subfolders.")
    args = ap.parse_args()

    root = Path(args.data_root)
    splits = ["train", "val", "test"]
    problems = []

    records = []  # (split, cls, path, size, group)
    for split in splits:
        split_dir = root / split
        if not split_dir.exists():
            continue
        for cls_dir in sorted([d for d in split_dir.iterdir() if d.is_dir()]):
            for img in cls_dir.rglob("*.png"):
                try:
                    with Image.open(img) as im:
                        size = im.size
                except Exception as e:
                    size = None
                    problems.append(f"Unreadable image: {img} ({e})")
                group_id = img.stem.split('_')[0]  # tweak if you use a different FOV naming
                records.append((split, cls_dir.name, img, size, group_id))

    # Hash-based duplicate detection across splits/classes
    by_hash = {}
    for split, cls, p, size, group in records:
        h = sha1(p)
        by_hash.setdefault(h, []).append((split, cls, p))

    for h, rows in by_hash.items():
        splits_here = {r[0] for r in rows}
        classes_here = {r[1] for r in rows}
        if len(rows) > 1 and (len(splits_here) > 1 or len(classes_here) > 1):
            problems.append(f"IDENTICAL FILE REUSED across splits/classes: {rows}")

    # Size sanity
    for _, _, p, size, _ in records:
        if size and not (size[0] == size[1] and 200 <= size[0] <= 6000):
            problems.append(f"Odd crop size {size} at {p}")

    # Naive FOV leakage check (same group_id across splits)
    groups = {}
    for split, _, _, _, g in records:
        groups.setdefault(g, set()).add(split)
    for g, splits_set in groups.items():
        if len(splits_set) > 1:
            problems.append(f"Potential FOV leakage: group '{g}' appears in multiple splits: {splits_set}")

    # Summary
    counts = {}
    for split, cls, _, _, _ in records:
        counts[(split, cls)] = counts.get((split, cls), 0) + 1
    print("Counts per split/class:")
    for (split, cls), n in sorted(counts.items()):
        print(f"  {split:5s} {cls:12s} : {n}")

    if problems:
        print("\n== Issues ==")
        for p in problems:
            print("-", p)
    else:
        print("\nNo issues found.")

if __name__ == "__main__":
    main()
