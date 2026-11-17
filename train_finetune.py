"""Fine-tune model lama (models/best.pt) dengan dataset yang sudah ditambah.

Bukan resume: training dimulai baru dengan bobot best.pt sebagai titik awal,
sehingga foto/label baru di data/dataset ikut dibaca.

Contoh:
    python train_finetune.py
    python train_finetune.py --imgsz 1280 --epochs 150 --batch 4
"""

import argparse
import os
from pathlib import Path

from ultralytics import YOLO


ROOT = Path(__file__).resolve().parent
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def parse_args():
    parser = argparse.ArgumentParser(description="Fine-tune YOLO dari models/best.pt")
    parser.add_argument("--model", default="models/best.pt", help="Model awal (hasil training sebelumnya)")
    parser.add_argument("--data", default="data/data.yaml", help="Konfigurasi dataset")
    parser.add_argument("--imgsz", type=int, default=640, help="Samakan dengan training sebelumnya")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--device", default="0", help="0 = GPU pertama, cpu = tanpa GPU")
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--patience", type=int, default=30, help="Berhenti jika tidak membaik selama N epoch")
    parser.add_argument("--name", default="finetune", help="Nama folder di runs/detect/")
    return parser.parse_args()


def check_dataset(root: Path) -> None:
    """Pastikan setiap gambar punya label sebelum training dimulai."""
    for split in ("train", "validation"):
        image_dir = root / split / "images"
        label_dir = root / split / "labels"
        images = {p.stem for p in image_dir.iterdir() if p.suffix.lower() in IMAGE_EXTENSIONS}
        labels = {p.stem for p in label_dir.glob("*.txt")}
        missing = sorted(images - labels)
        print(f"{split:<10}: {len(images)} gambar, {len(labels)} label")
        if missing:
            raise SystemExit(
                f"{len(missing)} gambar {split} belum punya label, contoh: {', '.join(missing[:5])}"
            )


def main():
    args = parse_args()
    os.chdir(ROOT)  # data.yaml memakai path relatif ./data/dataset

    model_path = Path(args.model)
    if not model_path.exists():
        raise SystemExit(f"Model awal tidak ditemukan: {model_path}")

    check_dataset(ROOT / "data" / "dataset")

    print("\n=== Skor model LAMA ===")
    old_metrics = YOLO(str(model_path)).val(
        data=args.data, imgsz=args.imgsz, device=args.device, verbose=False
    )

    model = YOLO(str(model_path))
    model.train(
        data=args.data,
        imgsz=args.imgsz,
        epochs=args.epochs,
        batch=args.batch,
        device=args.device,
        workers=args.workers,
        patience=args.patience,
        name=args.name,
        exist_ok=False,  # run baru otomatis jadi finetune2, finetune3, ...
    )

    new_best = Path(model.trainer.save_dir) / "weights" / "best.pt"
    print("\n=== Skor model BARU ===")
    new_metrics = YOLO(str(new_best)).val(
        data=args.data, imgsz=args.imgsz, device=args.device, verbose=False
    )

    old_map, new_map = old_metrics.box.map, new_metrics.box.map
    print("\n" + "=" * 50)
    print(f"mAP50-95 lama : {old_map:.4f}   (mAP50 {old_metrics.box.map50:.4f})")
    print(f"mAP50-95 baru : {new_map:.4f}   (mAP50 {new_metrics.box.map50:.4f})")
    print(f"Model baru    : {new_best}")
    if new_map > old_map:
        print("\nModel baru LEBIH BAIK. Pasang dengan:")
        print(f"  cp {model_path} {model_path.with_name('best_old.pt')}")
        print(f"  cp {new_best} {model_path}")
        print("  lalu restart server uvicorn.")
    else:
        print("\nModel baru TIDAK lebih baik. Tetap pakai model lama; cek ulang label foto baru.")


if __name__ == "__main__":
    main()
