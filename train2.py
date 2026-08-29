from ultralytics import YOLO

# Load model terakhir yang sudah ditrain, BUKAN yolo11n.pt
model = YOLO("runs/detect/train-9/weights/last.pt")

model.train(
    data="data/data.yaml",
    imgsz=640,
    batch=8,
    epochs=200,
    workers=1,
    device=0,
    patience=50,
    resume=True,      # ← WAJIB! lanjut dari checkpoint
    
    # Augmentasi yang lebih agresif
    augment=True,
    hsv_h=0.015,
    hsv_s=0.7,
    hsv_v=0.4,
    degrees=15.0,
    translate=0.1,
    scale=0.5,
    shear=2.0,
    flipud=0.5,
    fliplr=0.5,
    mosaic=1.0,
    mixup=0.1,
)
