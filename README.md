# Copper Wire Branch Counter

Aplikasi webcam untuk mendeteksi dan menghitung objek kawat tembaga bercabang menggunakan YOLOv8/YOLO11 custom. Webcam dibuka di **PC user**, sedangkan inferensi dijalankan di server.

## 1. Menjalankan server

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
```

Letakkan model hasil training Ultralytics di `models/best.pt`, lalu jalankan:

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

Buka `http://IP_SERVER:8000` dari browser PC user. Kamera memerlukan HTTPS jika browser mengakses server bukan dari `localhost`; gunakan reverse proxy HTTPS (Nginx/Caddy) untuk production.

## 2. Dataset sekitar 30 gambar

Tiga puluh gambar cukup untuk prototipe, tetapi biasanya belum cukup untuk kondisi produksi. Labeli **satu bounding box per kawat bercabang** dengan class `split_wire`, lalu pisahkan data menjadi train/val. Contoh struktur dan YAML ada di `data/data.yaml.example`.

Training dari model pretrained:

```bash
yolo detect train data=data/data.yaml model=yolo11n.pt epochs=100 imgsz=640
cp runs/detect/train/weights/best.pt models/best.pt
```

Jika satu foto berisi tiga kawat bercabang, beri tiga bounding box; aplikasi akan menghitung tiga deteksi. Jangan memberi satu box besar untuk seluruh foto.

## 3. Konfigurasi

Salin `.env.example` menjadi `.env`. `CONFIDENCE` mengatur ambang deteksi, `DEVICE=cpu` untuk server biasa, atau `DEVICE=0` untuk GPU NVIDIA. `MODEL_PATH` dapat diarahkan ke lokasi model lain.

## Catatan akurasi

Ambil variasi pencahayaan, jarak, sudut, latar belakang, dan jumlah cabang. Tambahkan gambar negatif (tanpa kawat) dan uji pada foto yang tidak dipakai saat training. Jumlah yang tampil adalah jumlah bounding box, bukan jumlah ujung/cabang geometris dalam satu kawat.
