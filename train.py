from ultralytics import YOLO

model = YOLO("yolo11l.pt")

model.train(
    data="data/data.yaml",
    imgsz=640,
    batch=8,
    epochs=200,        # ← naikkan dari 100
    workers=1,
    device=0,
    patience=50,      # ← early stopping
    augment=True,      # ← tambah augmentasi
    flipud=0.5,        # ← flip vertical
    mosaic=1.0,        # ← mosaic augmentation
    copy_paste=0.1,    # ← copy-paste augmentation
)
