"""Periksa pasangan gambar dan label dataset YOLO."""

import argparse
from pathlib import Path


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def files_by_stem(directory, extensions=None):
    if not directory.exists():
        return {}
    return {
        path.stem: path
        for path in directory.iterdir()
        if path.is_file() and (extensions is None or path.suffix.lower() in extensions)
    }


def check_split(name, image_dir, label_dir):
    images = files_by_stem(image_dir, IMAGE_EXTENSIONS)
    labels = files_by_stem(label_dir, {".txt"})
    missing = sorted(set(images) - set(labels))
    orphaned = sorted(set(labels) - set(images))
    empty = sorted(stem for stem, path in labels.items() if not path.read_text(encoding="utf-8").strip())

    print(f"{name}:")
    print(f"  images : {len(images)} ({image_dir})")
    print(f"  labels : {len(labels)} ({label_dir})")
    print(f"  kosong : {len(empty)}")
    if missing:
        print("  label hilang:", ", ".join(missing))
    if orphaned:
        print("  label tanpa gambar:", ", ".join(orphaned))
    return not missing and not orphaned


def main():
    parser = argparse.ArgumentParser(description="Cek dataset YOLO train/validation")
    parser.add_argument("--root", default="data/dataset", help="Folder root dataset")
    args = parser.parse_args()
    root = Path(args.root)
    train_ok = check_split("TRAIN", root / "train/images", root / "train/labels")
    val_ok = check_split("VALIDATION", root / "validation/images", root / "validation/labels")
    if not train_ok or not val_ok:
        raise SystemExit("\nDataset belum siap: lengkapi pasangan gambar dan label.")
    print("\nDataset siap digunakan untuk training.")


if __name__ == "__main__":
    main()
