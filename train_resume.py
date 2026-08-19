from ultralytics import YOLO

# Load model dari checkpoint terakhir train-17
model = YOLO('runs/detect/train-17/weights/last.pt')

# Training ulang dari checkpoint terakhir
results = model.train(
    data='data.yaml',          # path ke dataset config
    epochs=50,                 # tambahan epoch (sesuaikan)
    imgsz=640,                 # ukuran gambar
    resume=True,               # ⬅️ INI PENTING! Lanjutkan dari checkpoint
    device=0                   # GPU yang digunakan
)
