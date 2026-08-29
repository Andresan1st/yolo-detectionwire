from ultralytics import YOLO

# Load dari checkpoint train-17
model = YOLO('runs/detect/train-19/weights/last.pt')

# Training lanjut + 50 epoch
results = model.train(
    data='data/data.yaml',
    epochs=100,           # total epoch yang diinginkan
    imgsz=640,
    resume=True           # ⬅️ kunci utamanya
)
