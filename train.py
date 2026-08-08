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
    MODEL_BASE = "yolo11n.pt"  # atau yolo11s.pt, yolo11m.pt untuk akurasi lebih
    EPOCHS = int(os.getenv("EPOCHS", "100"))
    IMG_SIZE = int(os.getenv("IMG_SIZE", "640"))
    BATCH_SIZE = int(os.getenv("BATCH_SIZE", "16"))
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

    # Count images
    train_images = list(Path("data/dataset/images/augmented").glob("*.*")) if Path("data/dataset/images/augmented").exists() else []
    val_images = list(Path("data/dataset/images/val").glob("*.*")) if Path("data/dataset/images/val").exists() else []

    print(f"\n📊 Dataset:")
    print(f"   Train images:  {len(train_images)}")
    print(f"   Val images:    {len(val_images)}")

    if len(train_images) == 0:
        print("\n❌ Tidak ada gambar training!")
        print("   Jalankan augment.py dulu untuk generate data")
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
        "name=train",
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
        best_model = Path("runs/detect/train/weights/best.pt")
        if best_model.exists():
            target = Path("models/best.pt")
            import shutil
            shutil.copy(best_model, target)
            print(f"\n📦 Model baru disalin ke: {target}")
            print(f"   Size: {best_model.stat().st_size / 1024 / 1024:.1f} MB")
        else:
            print("\n⚠️  Model training tidak ditemukan")

        print("\n📈 Untuk melihat hasil training:")
        print("   1. Buka runs/detect/train/ untuk melihat metrics")
        print("   2. Jalankan server: uvicorn app.main:app --host 0.0.0.0 --port 8000")
    else:
        print("\n❌ Training gagal!")
        sys.exit(1)


if __name__ == "__main__":
    main()
