import os
import argparse
import shutil
import random
from pathlib import Path
from typing import List, Tuple

CLASS_NAMES = ["Parasitized", "Uninfected"]


def collect_images_by_class(raw_dir: Path) -> List[Tuple[str, int]]:
    samples = []
    for idx, cls in enumerate(CLASS_NAMES):
        class_dir = raw_dir / cls
        if not class_dir.exists():
            raise FileNotFoundError(f"Expected class folder not found: {class_dir}")
        files = []
        for ext in ("*.png", "*.jpg", "*.jpeg"):
            files.extend(class_dir.glob(ext))
        files = [str(p) for p in files]
        files.sort()
        samples.append((files, idx))
    return samples


def make_split_dirs(base_out: Path):
    for split in ["train", "val", "test"]:
        for cls in CLASS_NAMES:
            (base_out / split / cls).mkdir(parents=True, exist_ok=True)


def split_list(items: List[str], val_frac: float, test_frac: float, seed: int):
    rnd = random.Random(seed)
    items = list(items)
    rnd.shuffle(items)
    n = len(items)
    n_test = int(round(n * test_frac))
    n_val = int(round(n * val_frac))
    test = items[:n_test]
    val = items[n_test:n_test + n_val]
    train = items[n_test + n_val:]
    return train, val, test


def copy_into(pairs: List[Tuple[str, int]], dest_root: Path):
    for src, label in pairs:
        cls = CLASS_NAMES[label]
        dst = dest_root / cls / os.path.basename(src)
        if not dst.exists():
            shutil.copy2(src, dst)


def main():
    ap = argparse.ArgumentParser(description="Create stratified splits without sklearn (per-class fractional splits)")
    ap.add_argument("--raw_dir", required=True)
    ap.add_argument("--out_dir", default="data/splits")
    ap.add_argument("--val_size", type=float, default=0.15)
    ap.add_argument("--test_size", type=float, default=0.15)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    raw = Path(args.raw_dir)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    per_class = collect_images_by_class(raw)

    make_split_dirs(out)

    total_train = total_val = total_test = 0
    for files, label in per_class:
        train, val, test = split_list(files, args.val_size, args.test_size, args.seed)
        copy_into([(p, label) for p in train], out / "train")
        copy_into([(p, label) for p in val], out / "val")
        copy_into([(p, label) for p in test], out / "test")
        total_train += len(train)
        total_val += len(val)
        total_test += len(test)

    print(f"Done. Counts -> train: {total_train}, val: {total_val}, test: {total_test}")
    print(f"Splits saved under: {out}")


if __name__ == "__main__":
    main()
