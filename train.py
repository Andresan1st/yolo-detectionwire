"""
Training Script untuk Wire Branch Detection
Jalankan: python train.py
"""

import os
import sys
from pathlib import Path

def main():
    print("=" * 60)
    print("🚀 Wire Branch Detection - Training")
    print("=" * 60)

    # Konfigurasi
    DATA_YAML = "data/data.yaml"
    MODEL_BASE = os.getenv("MODEL_BASE", "yolo11l.pt")
    EPOCHS = int(os.getenv("EPOCHS", "150"))
    IMG_SIZE = int(os.getenv("IMG_SIZE", "640"))
    BATCH_SIZE = int(os.getenv("BATCH_SIZE", "4"))
    DEVICE = os.getenv("DEVICE", "")  # kosong = auto, "0" = GPU pertama, "cpu" = CPU

    # Cek file konfigurasi
    if not Path(DATA_YAML).exists():
        print(f"❌ File konfigurasi tidak ditemukan: {DATA_YAML}")
        print("   Pastikan data.yaml ada di root project")
        sys.exit(1)

    # Cek model base
    if not Path(MODEL_BASE).exists():
        print(f"⚠️  Model base tidak ditemukan: {MODEL_BASE}")
        print("   Akan didownload otomatis oleh Ultralytics")
        MODEL_ARG = MODEL_BASE
    else:
        MODEL_ARG = MODEL_BASE

    print(f"\n📋 Konfigurasi:")
    print(f"   Dataset:        {DATA_YAML}")
    print(f"   Model Base:    {MODEL_BASE}")
    print(f"   Epochs:        {EPOCHS}")
    print(f"   Image Size:    {IMG_SIZE}")
    print(f"   Batch Size:    {BATCH_SIZE}")
    print(f"   Device:        {DEVICE or 'auto (GPU优先)'}")

    # Count images according to data/data.yaml
    train_dir = Path("data/dataset/train/images")
    val_dir = Path("data/dataset/validation/images")
    image_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    train_images = [path for path in train_dir.iterdir() if path.suffix.lower() in image_extensions] if train_dir.exists() else []
    val_images = [path for path in val_dir.iterdir() if path.suffix.lower() in image_extensions] if val_dir.exists() else []

    train_labels = {path.stem for path in (train_dir.parent / "labels").glob("*.txt")} if train_dir.exists() else set()
    val_labels = {path.stem for path in (val_dir.parent / "labels").glob("*.txt")} if val_dir.exists() else set()
    train_stems = {path.stem for path in train_images}
    val_stems = {path.stem for path in val_images}

    print(f"\n📊 Dataset:")
    print(f"   Train images:  {len(train_images)}")
    print(f"   Val images:    {len(val_images)}")
    print(f"   Train labels:  {len(train_labels)}")
    print(f"   Val labels:    {len(val_labels)}")

    if len(train_images) == 0:
        print("\n❌ Tidak ada gambar training!")
        print("   Tambahkan gambar ke data/dataset/train/images")
        sys.exit(1)

    missing_train = sorted(train_stems - train_labels)
    missing_val = sorted(val_stems - val_labels)
    if missing_train or missing_val or not val_images:
        print("\n❌ Dataset belum siap:")
        if missing_train:
            print(f"   Label train hilang ({len(missing_train)}): {', '.join(missing_train[:10])}")
        if missing_val:
            print(f"   Label validation hilang ({len(missing_val)}): {', '.join(missing_val[:10])}")
        if not val_images:
            print("   Validation belum memiliki gambar.")
        print("   Jalankan: python3 check_dataset.py")
        sys.exit(1)

    # Build command
    cmd = [
        "yolo",
        "detect",
        "train",
        f"data={DATA_YAML}",
        f"model={MODEL_ARG}",
        f"epochs={EPOCHS}",
        f"imgsz={IMG_SIZE}",
        f"batch={BATCH_SIZE}",
        "project=runs/detect",
        "name=train_yolo11l",
        "exist_ok=True",
        "plots=True",
        "save=True",
        "verbose=True",
    ]

    if DEVICE:
        cmd.append(f"device={DEVICE}")

    print(f"\n🔧 Command:")
    print(f"   {' '.join(cmd)}")

    # Run training
    print("\n" + "=" * 60)
    print("🏋️  Training dimulai...")
    print("=" * 60 + "\n")

    import subprocess
    result = subprocess.run(cmd)

    if result.returncode == 0:
        print("\n" + "=" * 60)
        print("✅ Training selesai!")
        print("=" * 60)

        # Copy best model
        best_model = Path("runs/detect/train_yolo11l/weights/best.pt")
        if best_model.exists():
            target = Path("models/best.pt")
            target.parent.mkdir(parents=True, exist_ok=True)
            import shutil
            shutil.copy2(best_model, target)
            print(f"\n📦 Model baru disalin ke: {target}")
            print(f"   Size: {best_model.stat().st_size / 1024 / 1024:.1f} MB")
        else:
            print(f"\n❌ Training selesai tetapi model tidak ditemukan: {best_model}")
            sys.exit(1)

        print("\n📈 Untuk melihat hasil training:")
        print("   1. Buka runs/detect/train_yolo11l/ untuk melihat metrics")
        print("   2. Jalankan server: uvicorn app.main:app --host 0.0.0.0 --port 8000")
    else:
        print("\n❌ Training gagal!")
        sys.exit(1)


if __name__ == "__main__":
    main()
