import os
import argparse
import shutil
from pathlib import Path
from typing import Tuple, List
from sklearn.model_selection import train_test_split

CLASS_NAMES = ["Parasitized", "Uninfected"]


def collect_image_paths(raw_dir: Path) -> List[Tuple[str, int]]:
    samples = []
    for idx, cls in enumerate(CLASS_NAMES):
        class_dir = raw_dir / cls
        if not class_dir.exists():
            raise FileNotFoundError(f"Expected class folder not found: {class_dir}")
        for p in class_dir.glob("*.png"):
            samples.append((str(p), idx))
        for p in class_dir.glob("*.jpg"):
            samples.append((str(p), idx))
        for p in class_dir.glob("*.jpeg"):
            samples.append((str(p), idx))
    if not samples:
        raise RuntimeError("No images found. Ensure raw_dir has Parasitized/ and Uninfected/ with images.")
    return samples


def make_split_dirs(base_out: Path):
    for split in ["train", "val", "test"]:
        for cls in CLASS_NAMES:
            d = base_out / split / cls
            d.mkdir(parents=True, exist_ok=True)


def copy_samples(pairs: List[Tuple[str, int]], dest_root: Path):
    for src, label in pairs:
        cls = CLASS_NAMES[label]
        dst = dest_root / cls / os.path.basename(src)
        if not dst.exists():
            shutil.copy2(src, dst)


def main():
    parser = argparse.ArgumentParser(description="Create stratified train/val/test splits from raw malaria cell images.")
    parser.add_argument("--raw_dir", type=str, required=True, help="Path to raw dataset folder containing Parasitized/ and Uninfected/")
    parser.add_argument("--out_dir", type=str, default="data/splits", help="Output directory for splits")
    parser.add_argument("--val_size", type=float, default=0.15, help="Validation fraction")
    parser.add_argument("--test_size", type=float, default=0.15, help="Test fraction")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    raw_dir = Path(args.raw_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    samples = collect_image_paths(raw_dir)
    X = [p for p, _ in samples]
    y = [c for _, c in samples]

    # First split off test
    X_tmp, X_test, y_tmp, y_test = train_test_split(
        X, y, test_size=args.test_size, random_state=args.seed, stratify=y
    )
    # Then split tmp into train and val such that val is args.val_size of total
    val_fraction_of_tmp = args.val_size / (1.0 - args.test_size)
    X_train, X_val, y_train, y_val = train_test_split(
        X_tmp, y_tmp, test_size=val_fraction_of_tmp, random_state=args.seed, stratify=y_tmp
    )

    make_split_dirs(out_dir)
    copy_samples(list(zip(X_train, y_train)), out_dir / "train")
    copy_samples(list(zip(X_val, y_val)), out_dir / "val")
    copy_samples(list(zip(X_test, y_test)), out_dir / "test")

    print(f"Done. Counts -> train: {len(X_train)}, val: {len(X_val)}, test: {len(X_test)}")
    print(f"Splits saved under: {out_dir}")


if __name__ == "__main__":
    main()
