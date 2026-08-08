"""
Script Diagnostik untuk Wire Detection Model
Cek apakah model siap dan bisa deteksi dengan benar
"""

import os
import sys
from pathlib import Path

def print_header(text):
    print("\n" + "=" * 60)
    print(f"  {text}")
    print("=" * 60)

def print_status(ok, text):
    status = "✅" if ok else "❌"
    print(f"{status} {text}")

def check_files():
    """Cek semua file yang diperlukan"""
    print_header("1. CEK FILE & FOLDER")

    items = [
        ("models/best.pt", "Model YOLO"),
        ("data/data.yaml", "Konfigurasi Dataset"),
        ("data/dataset/images/augmented", "Folder Train Images"),
        ("data/dataset/images/val", "Folder Val Images"),
        ("data/dataset/labels/augmented", "Folder Train Labels"),
    ]

    all_ok = True
    for path, desc in items:
        p = Path(path)
        exists = p.exists()
        print_status(exists, f"{desc}: {path}")

        if p.is_file():
            size_mb = p.stat().st_size / 1024 / 1024
            print(f"      Size: {size_mb:.1f} MB")

        if p.is_dir():
            count = len(list(p.glob("*")))
            print(f"      Files: {count}")

        if not exists:
            all_ok = False

    return all_ok

def check_training_results():
    """Cek hasil training"""
    print_header("2. CEK HASIL TRAINING")

    results_csv = Path("runs/detect/train/results.csv")
    if not results_csv.exists():
        print("❌ Folder runs/detect/train/ belum ada")
        print("   → Training belum dijalankan atau gagal")
        print("   → Jalankan: yolo detect train data=data/data.yaml model=yolo11n.pt epochs=100")
        return False

    print("✅ Hasil training ditemukan!")

    # Baca semua hasil
    lines = results_csv.read_text().strip().split("\n")
    if len(lines) < 2:
        print("❌ File results.csv kosong atau corrupt")
        return False

    # Parse header dan data terakhir
    header = lines[0].split(",")
    last_row = lines[-1].split(",")

    # Cari index kolom
    try:
        epoch_idx = header.index("epoch")
        precision_idx = header.index("metrics/precision(B)")
        recall_idx = header.index("metrics/recall(B)")
        map50_idx = header.index("metrics/mAP50(B)")
        map50_95_idx = header.index("metrics/mAP50-95(B)")
    except ValueError as e:
        print(f"❌ Format results.csv tidak sesuai: {e}")
        return False

    epoch = last_row[epoch_idx]
    precision = float(last_row[precision_idx]) if last_row[precision_idx] else 0
    recall = float(last_row[recall_idx]) if last_row[recall_idx] else 0
    map50 = float(last_row[map50_idx]) if last_row[map50_idx] else 0
    map50_95 = float(last_row[map50_95_idx]) if last_row[map50_95_idx] else 0

    print(f"\n📊 METRIK TRAINING (Epoch {epoch}):")
    print(f"   Precision:  {precision*100:.1f}%")
    print(f"   Recall:     {recall*100:.1f}%")
    print(f"   mAP@0.5:    {map50*100:.1f}%")
    print(f"   mAP@0.5-95: {map50_95*100:.1f}%")

    # Saran berdasarkan metrics
    print("\n📝 ANALISIS:")
    if map50 < 0.3:
        print("   ⚠️  mAP@0.5 masih rendah (<30%)")
        print("   → Model belum cukup akurat")
        print("   → Perlu training lebih banyak epochs (150-300)")
        print("   → Atau tambah dataset dengan variasi lebih")
    elif map50 < 0.5:
        print("   ⚠️  mAP@0.5 moderate (30-50%)")
        print("   → Model bisa deteksi tapi belum optimal")
        print("   → Turunkan CONFIDENCE di .env ke 0.15-0.20")
    elif map50 < 0.7:
        print("   ✅ mAP@0.5 cukup baik (50-70%)")
        print("   → Model sudah bisa digunakan")
        print("   → CONFIDENCE 0.25-0.35 sudah sesuai")
    else:
        print("   ✅ mAP@0.5 sangat baik (>70%)")
        print("   → Model sudah optimal")

    return True

def test_detection():
    """Test deteksi dengan gambar sample"""
    print_header("3. TEST DETEKSI")

    try:
        from ultralytics import YOLO
        import cv2
    except ImportError as e:
        print(f"❌ Gagal import library: {e}")
        print("   → Jalankan: pip install ultralytics opencv-python-headless")
        return False

    # Load model
    model_path = Path("models/best.pt")
    if not model_path.exists():
        print(f"❌ Model tidak ditemukan: {model_path}")
        return False

    try:
        print(f"📦 Loading model: {model_path}")
        model = YOLO(str(model_path))
        print(f"✅ Model loaded: {model.names}")

        # Cari gambar test
        test_images = list(Path("data/dataset/images/val").glob("*.*"))
        if not test_images:
            test_images = list(Path("data/dataset/images/augmented").glob("*.jpg"))[:3]

        if not test_images:
            print("❌ Tidak ada gambar untuk test")
            return False

        print(f"\n🖼️  Test pada {len(test_images)} gambar:")

        total_detections = 0
        for img_path in test_images:
            # Deteksi
            results = model.predict(
                str(img_path),
                conf=0.28,  # Sesuai .env
                iou=0.45,
                verbose=False
            )

            num_dets = len(results[0].boxes) if results[0].boxes is not None else 0
            total_detections += num_dets

            # Detail deteksi
            if num_dets > 0:
                print(f"\n   📷 {img_path.name}:")
                print(f"      Deteksi: {num_dets} objek")
                for i, box in enumerate(results[0].boxes[:5]):  # Max 5 first
                    conf = float(box.conf[0])
                    cls = int(box.cls[0])
                    label = model.names.get(cls, cls)
                    print(f"      {i+1}. {label} - {conf*100:.1f}%")
            else:
                print(f"\n   📷 {img_path.name}: ❌ 0 deteksi")

        print(f"\n📊 TOTAL: {total_detections} deteksi dari {len(test_images)} gambar")

        if total_detections == 0:
            print("\n⚠️  TIDAK ADA DETEKSI!")
            print("   Kemungkinan penyebab:")
            print("   1. Model belum di-training dengan benar")
            print("   2. Confidence threshold terlalu tinggi")
            print("   3. Gambar test tidak cocok dengan training data")
            print("\n   Solusi:")
            print("   → Cek runs/detect/train/results.csv untuk lihat metrics")
            print("   → Turunkan CONFIDENCE di .env ke 0.10-0.15")
            print("   → Training ulang dengan epochs lebih banyak")
            return False
        else:
            print("\n✅ Model bisa mendeteksi objek!")
            return True

    except Exception as e:
        print(f"❌ Error saat testing: {e}")
        import traceback
        traceback.print_exc()
        return False

def check_env_config():
    """Cek konfigurasi .env"""
    print_header("4. CEK KONFIGURASI (.env)")

    from dotenv import load_dotenv
    load_dotenv()

    configs = {
        "MODEL_PATH": os.getenv("MODEL_PATH", "models/best.pt"),
        "CONFIDENCE": os.getenv("CONFIDENCE", "0.35"),
        "IOU": os.getenv("IOU", "0.45"),
        "DEVICE": os.getenv("DEVICE", "cpu"),
    }

    for key, value in configs.items():
        print(f"   {key}: {value}")

    conf = float(configs["CONFIDENCE"])
    if conf > 0.4:
        print(f"\n⚠️  CONFIDENCE={conf} cukup tinggi")
        print("   → Coba turunkan ke 0.15-0.25 untuk deteksi lebih banyak")

    return True

def main():
    print("\n" + "=" * 60)
    print("  WIRE DETECTION MODEL - DIAGNOSTIC TOOL")
    print("=" * 60)

    # Run all checks
    files_ok = check_files()
    check_env_config()
    training_ok = check_training_results()
    detection_ok = test_detection()

    # Summary
    print_header("📋 RINGKASAN")

    if not files_ok:
        print("❌ File/folder ada yang missing")
        print("   → Pastikan semua file ada sebelum lanjut")
    elif not training_ok:
        print("⚠️  Training belum pernah dijalankan atau gagal")
        print("   → Jalankan: yolo detect train data=data/data.yaml model=yolo11n.pt epochs=100")
        print("   → Copy hasil: cp runs/detect/train/weights/best.pt models/best.pt")
    elif not detection_ok:
        print("⚠️  Model tidak bisa mendeteksi dengan baik")
        print("   → Cek hasil training di runs/detect/train/results.csv")
        print("   → Turunkan CONFIDENCE di .env")
        print("   → Pertimbangkan training ulang")
    else:
        print("✅ SEMUA CHECK PASSED!")
        print("   Model siap digunakan.")
        print("")
        print("   Jalankan server:")
        print("   uvicorn app.main:app --host 0.0.0.0 --port 8000")

if __name__ == "__main__":
    main()
