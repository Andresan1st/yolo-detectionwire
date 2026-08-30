"""
Wire Branch Counter - Server-Side Inference
Inferens di server, browser hanya tampilkan hasil
"""

import os
import base64
import sqlite3
from pathlib import Path
from datetime import datetime
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel

import numpy as np
import onnxruntime as ort
import cv2


# ==================== CONFIG ====================
BASE_DIR = Path(__file__).resolve().parent
TEMPLATES_DIR = BASE_DIR / "templates"
MODEL_PATH = BASE_DIR / "best.onnx"
DB_PATH = os.getenv("DB_PATH", str(BASE_DIR / "detections.db"))
DB_TABLE = "machine_detection_copper"

INPUT_SIZE = 1280  # Ukuran model
CONFIDENCE = 0.35
IOU_THRESH = 0.45
CLASS_NAMES = ["element_copper"]


# ==================== ONNX SESSION ====================
session = None

def load_session():
    global session
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model tidak ditemukan: {MODEL_PATH}")

    print(f"Loading: {MODEL_PATH} ({MODEL_PATH.stat().st_size / 1024 / 1024:.1f} MB)")
    session = ort.InferenceSession(
        str(MODEL_PATH),
        providers=['CUDAExecutionProvider', 'CPUExecutionProvider']
    )
    print(f"Providers: {session.get_providers()}")
    print("Model loaded!")


# ==================== INFERENCE ====================
def letterbox_resize(image, target_size=1280):
    """Resize image dengan letterbox, return coords untuk scaling balik"""
    h, w = image.shape[:2]
    scale = min(target_size / h, target_size / w)
    new_w, new_h = int(w * scale), int(h * scale)

    resized = cv2.resize(image, (new_w, new_h))

    # Create canvas with gray
    canvas = np.full((target_size, target_size, 3), 114, dtype=np.uint8)
    x_offset = (target_size - new_w) // 2
    y_offset = (target_size - new_h) // 2
    canvas[y_offset:y_offset + new_h, x_offset:x_offset + new_w] = resized

    return canvas, scale, (x_offset, y_offset)


def preprocess(image_bytes):
    """Preprocess gambar untuk inference"""
    nparr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    orig_h, orig_w = img.shape[:2]

    # Letterbox resize
    canvas, scale, pad = letterbox_resize(img, INPUT_SIZE)

    # Convert BGR -> RGB -> CHW
    rgb = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)
    blob = rgb.astype(np.float32) / 255.0
    blob = blob.transpose(2, 0, 1)  # HWC -> CHW
    blob = np.expand_dims(blob, axis=0)

    return blob, scale, pad, orig_w, orig_h


def detect(image_bytes):
    """Jalankan deteksi"""
    global session

    # Preprocess
    tensor, scale, pad, orig_w, orig_h = preprocess(image_bytes)

    # Run inference
    input_name = session.get_inputs()[0].name
    output = session.run(None, {input_name: tensor})[0]

    print(f"DEBUG: output shape = {output.shape}")

    detections = []

    # Detect format: [batch, num_predictions, 4+num_classes]
    # or [batch, 4+num_classes, num_predictions]
    output = output.squeeze()  # Remove batch dimension

    if len(output.shape) == 2:
        # Format: [num_predictions, 4+num_classes]
        predictions = output
    else:
        # Format: [4+num_classes, num_predictions] -> transpose
        predictions = output.T

    num_values = predictions.shape[1]  # 4 + num_classes
    num_classes = num_values - 4

    print(f"DEBUG: predictions shape = {predictions.shape}, num_classes = {num_classes}")

    for pred in predictions:
        x, y, w, h = pred[:4]
        conf = float(pred[4])

        if conf < CONFIDENCE:
            continue

        # Class (for single class model, it's just the confidence)
        if num_classes == 1:
            class_id = 0
        else:
            class_scores = pred[5:5+num_classes]
            if len(class_scores) == 0:
                continue
            class_id = int(np.argmax(class_scores))

        # Convert center -> corner
        x1 = x - w / 2
        y1 = y - h / 2
        x2 = x + w / 2
        y2 = y + h / 2

        # Remove padding & scale back to original
        pad_x, pad_y = pad
        x1 = (x1 - pad_x) / scale
        y1 = (y1 - pad_y) / scale
        x2 = (x2 - pad_x) / scale
        y2 = (y2 - pad_y) / scale

        # Clip
        x1 = max(0, min(orig_w, x1))
        y1 = max(0, min(orig_h, y1))
        x2 = max(0, min(orig_w, x2))
        y2 = max(0, min(orig_h, y2))

        if x2 > x1 and y2 > y1:
            detections.append({
                "bbox": [float(x1), float(y1), float(x2), float(y2)],
                "confidence": conf,
                "class_id": class_id,
                "label": CLASS_NAMES[class_id] if class_id < len(CLASS_NAMES) else f"class_{class_id}"
            })

    # NMS
    detections = nms(detections, IOU_THRESH)

    return detections, orig_w, orig_h


def nms(detections, iou_thresh):
    """Non-Maximum Suppression"""
    if not detections:
        return []

    # Sort by confidence
    detections = sorted(detections, key=lambda x: x["confidence"], reverse=True)
    keep = []

    for det in detections:
        skip = False
        for kept in keep:
            if det["class_id"] != kept["class_id"]:
                continue
            iou = calc_iou(det["bbox"], kept["bbox"])
            if iou > iou_thresh:
                skip = True
                break
        if not skip:
            keep.append(det)

    return keep


def calc_iou(box1, box2):
    """Calculate IoU"""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    inter = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])

    return inter / (area1 + area2 - inter) if (area1 + area2 - inter) > 0 else 0


# ==================== DATABASE ====================
def init_db():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute(f"""CREATE TABLE IF NOT EXISTS {DB_TABLE} (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        countc INTEGER NOT NULL,
        dt_ins TEXT NOT NULL,
        seq_time TEXT NOT NULL,
        area TEXT NOT NULL,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )""")
    conn.commit()
    conn.close()


# ==================== LIFESPAN ====================
@asynccontextmanager
async def lifespan(app: FastAPI):
    print("=" * 50)
    print("🔌 Wire Branch Counter")
    print("=" * 50)
    init_db()
    print("✅ DB ready")
    try:
        load_session()
    except Exception as e:
        print(f"❌ Model error: {e}")
    print("=" * 50 + "\n")
    yield
    print("👋 Shutdown")


# ==================== APP ====================
app = FastAPI(title="Wire Branch Counter", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ==================== ROUTES ====================
@app.get("/")
async def root():
    return FileResponse(str(TEMPLATES_DIR / "index.html"))


@app.get("/health")
async def health():
    return {
        "status": "ok",
        "model": "onnx",
        "inference": "server_side"
    }


class DetectRequest(BaseModel):
    image: str  # base64


@app.post("/api/detect")
async def detect_api(req: DetectRequest):
    try:
        image_bytes = base64.b64decode(req.image)
        detections, width, height = detect(image_bytes)
        return {
            "success": True,
            "count": len(detections),
            "detections": detections,
            "image_width": width,
            "image_height": height
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/save")
async def save_api(area: str, countc: int, dt_ins: str, seq_time: str):
    try:
        conn = sqlite3.connect(DB_PATH)
        c = conn.cursor()
        c.execute(f"INSERT INTO {DB_TABLE} (countc, dt_ins, seq_time, area) VALUES (?,?,?,?)",
                  (countc, dt_ins, seq_time, area))
        conn.commit()
        id = c.lastrowid
        conn.close()
        return {"success": True, "id": id}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/history")
async def history(limit: int = 100):
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    c = conn.cursor()
    c.execute(f"SELECT * FROM {DB_TABLE} ORDER BY id DESC LIMIT ?", (limit,))
    rows = c.fetchall()
    conn.close()
    return {"success": True, "data": [dict(r) for r in rows]}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
