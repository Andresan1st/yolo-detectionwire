from ultralytics import YOLO

# Ganti dari yolo11n.pt ke yolo11s.pt atau yolo11m.pt
model = YOLO("yolo11s.pt")  # ← SMALL (lebih bagus dari nano)

model.train(
    data="data/data.yaml",
    imgsz=640,
    batch=8,
    epochs=400,          # ← 300 epochs
    device=0,
    patience=50,
    resume=True,
    # AUGMENTASI YANG BENER
    augment=True
)
