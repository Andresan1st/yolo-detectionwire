from ultralytics import YOLO

# Model
model = YOLO("yolo11s.pt")

results = model.train(
    data="data/data.yaml",

    # Image
    imgsz=1280,

    # Training
    epochs=200,
    batch=8,
    device=0,
    workers=4,

    # Early stopping
    patience=40,



)
