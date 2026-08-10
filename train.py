from ultralytics import YOLO

model = YOLO("yolo11n.pt")

model.train(
    data="data/data.yaml",
    imgsz=640,
    batch=8,
    epochs=200,       # naikkan epochs untuk compensate model kecil
    workers=1,
    device=0,
    patience=50,
    augment=True,
)