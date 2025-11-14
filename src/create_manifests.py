import argparse
from pathlib import Path
import random
import csv

CLASS_NAMES = ["Parasitized", "Uninfected"]


def collect(raw_dir: Path):
    data = []
    for label, cls in enumerate(CLASS_NAMES):
        cls_dir = raw_dir / cls
        if not cls_dir.exists():
            raise FileNotFoundError(f"Missing class folder: {cls_dir}")
        files = []
        for ext in ("*.png", "*.jpg", "*.jpeg"):
            files.extend(cls_dir.glob(ext))
        for p in files:
            data.append((str(p), label))
    return data


def split_pairs(pairs, val_frac, test_frac, seed):
    rnd = random.Random(seed)
    rnd.shuffle(pairs)
    n = len(pairs)
    n_test = int(round(n * test_frac))
    n_val = int(round(n * val_frac))
    test = pairs[:n_test]
    val = pairs[n_test:n_test+n_val]
    train = pairs[n_test+n_val:]
    return train, val, test


def write_csv(rows, out_path: Path):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(["path", "label"])  # header
        for p, y in rows:
            w.writerow([p, y])


def main():
    ap = argparse.ArgumentParser(description="Create CSV manifests (train/val/test) listing image paths and labels.")
    ap.add_argument("--raw_dir", required=True, help="Folder containing Parasitized/ and Uninfected/")
    ap.add_argument("--out_dir", default="data/manifests", help="Where to write train.csv/val.csv/test.csv")
    ap.add_argument("--val_size", type=float, default=0.15)
    ap.add_argument("--test_size", type=float, default=0.15)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    pairs = collect(Path(args.raw_dir))
    train, val, test = split_pairs(pairs, args.val_size, args.test_size, args.seed)

    out = Path(args.out_dir)
    write_csv(train, out / "train.csv")
    write_csv(val, out / "val.csv")
    write_csv(test, out / "test.csv")

    print(f"Wrote manifests to {out}")
    print(f"Counts -> train: {len(train)}, val: {len(val)}, test: {len(test)}")


if __name__ == "__main__":
    main()
