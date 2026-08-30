"""
FastAPI Server - Wire Branch Counter
YOLO ONNX Runtime Server-Side Inference

Inferens di server (Python), browser hanya kirim gambar & dapat hasil.
"""

import os
import base64
import sqlite3
from pathlib import Path
from datetime import datetime
from contextlib import asynccontextmanager
from typing import Optional, List
from io import BytesIO

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel

import numpy as np
import onnxruntime as ort
import cv2
from PIL import Image


# =========================================================
# CONFIG
# =========================================================

BASE_DIR = Path(__file__).resolve().parent
STATIC_DIR = BASE_DIR / "static"
TEMPLATES_DIR = BASE_DIR / "templates"
MODEL_PATH = BASE_DIR / "best.onnx"

INPUT_SIZE = 640  # Sesuaikan dengan model
CONFIDENCE_THRESHOLD = 0.35
IOU_THRESHOLD = 0.45
CLASS_NAMES = ['element_copper']

DB_PATH = os.getenv("DB_PATH", str(BASE_DIR / "detections.db"))
DB_TABLE = os.getenv("DB_TABLE", "machine_detection_copper")


# =========================================================
# ONNX SESSION
# =========================================================

session = None

def load_model():
    """Load ONNX model once at startup"""
    global session

    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model tidak ditemukan: {MODEL_PATH}")

    print(f"Loading model: {MODEL_PATH}")
    print(f"Model size: {MODEL_PATH.stat().st_size / 1024 / 1024:.2f} MB")

    # Create ONNX Runtime session
    session = ort.InferenceSession(
        str(MODEL_PATH),
        providers=['CUDAExecutionProvider', 'CPUExecutionProvider']
    )

    print(f"Available providers: {session.get_providers()}")
    print("Model loaded successfully!")


# =========================================================
# INFERENCE FUNCTIONS
# =========================================================

def letterbox(img, new_shape=(640, 640), color=(114, 114, 114)):
    """Resize and pad image for inference"""
    shape = img.shape[:2]  # current shape [height, width]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    # Scale ratio
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])

    # Compute padding
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]

    dw /= 2
    dh /= 2

    if shape[::-1] != new_unpad:
        img = np.ascontiguousarray(
            np.array(Image.fromarray(img).resize(new_unpad, Image.BILINEAR))
        )

    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    img = np.array(Image.fromarray(img).resize(new_unpad, Image.BILINEAR))
    img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)

    return img, r, (dw, dh)


def preprocess_image(image_bytes: bytes) -> tuple:
    """Preprocess image for YOLO"""
    # Read image
    nparr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    orig_h, orig_w = img.shape[:2]

    # Letterbox resize
    img_resized, ratio, pad = letterbox(img, (INPUT_SIZE, INPUT_SIZE))
    img_resized = img_resized.transpose(2, 0, 1)[::-1]  # HWC to CHW
    img_resized = np.ascontiguousarray(img_resized)
    img_resized = img_resized.astype(np.float32) / 255.0
    img_resized = np.expand_dims(img_resized, axis=0)

    return img_resized, ratio, pad, orig_w, orig_h


def postprocess_yolo(output, ratio, pad, orig_w, orig_h):
    """Postprocess YOLO output"""
    predictions = output[0]

    detections = []

    # Transpose: (batch, channels, num_predictions) -> (num_predictions, channels)
    predictions = predictions.T

    for pred in predictions:
        # Get box coordinates
        x, y, w, h, *scores = pred
        scores = np.array(scores)

        # Get max class and score
        max_score = scores.max()
        max_class = scores.argmax()

        if max_score < CONFIDENCE_THRESHOLD:
            continue

        # Convert from center to corner
        x1 = x - w / 2
        y1 = y - h / 2
        x2 = x + w / 2
        y2 = y + h / 2

        # Remove padding and scale
        x1 = (x1 - pad[0]) / ratio
        y1 = (y1 - pad[1]) / ratio
        x2 = (x2 - pad[0]) / ratio
        y2 = (y2 - pad[1]) / ratio

        # Clip to image bounds
        x1 = max(0, min(orig_w, x1))
        y1 = max(0, min(orig_h, y1))
        x2 = max(0, min(orig_w, x2))
        y2 = max(0, min(orig_h, y2))

        if x2 > x1 and y2 > y1:
            detections.append({
                'class': int(max_class),
                'label': CLASS_NAMES[max_class] if max_class < len(CLASS_NAMES) else f'class_{max_class}',
                'confidence': float(max_score),
                'bbox': [float(x1), float(y1), float(x2), float(y2)]
            })

    # Apply NMS
    detections = apply_nms(detections, IOU_THRESHOLD)

    return detections


def apply_nms(detections, iou_threshold):
    """Apply Non-Maximum Suppression"""
    if not detections:
        return []

    # Sort by confidence
    detections = sorted(detections, key=lambda x: x['confidence'], reverse=True)

    keep = []

    for det in detections:
        should_keep = True
        for kept in keep:
            if det['class'] != kept['class']:
                continue
            iou = calculate_iou(det['bbox'], kept['bbox'])
            if iou > iou_threshold:
                should_keep = False
                break

        if should_keep:
            keep.append(det)

    return keep


def calculate_iou(box1, box2):
    """Calculate IoU between two boxes"""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    inter_area = max(0, x2 - x1) * max(0, y2 - y1)

    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])

    union_area = box1_area + box2_area - inter_area

    return inter_area / union_area if union_area > 0 else 0


def run_inference(image_bytes: bytes) -> List[dict]:
    """Run inference on image bytes"""
    if session is None:
        raise RuntimeError("Model belum diload")

    # Preprocess
    img_tensor, ratio, pad, orig_w, orig_h = preprocess_image(image_bytes)

    # Run inference
    input_name = session.get_inputs()[0].name
    output = session.run(None, {input_name: img_tensor})[0]

    # Postprocess
    detections = postprocess_yolo(output, ratio, pad, orig_w, orig_h)

    return detections


# =========================================================
# DATABASE
# =========================================================

def init_db():
    """Initialize SQLite database"""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    cursor.execute(f"""
        CREATE TABLE IF NOT EXISTS {DB_TABLE} (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            countc INTEGER NOT NULL,
            dt_ins TEXT NOT NULL,
            seq_time TEXT NOT NULL,
            area TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)

    conn.commit()
    conn.close()


# =========================================================
# LIFESPAN
# =========================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    print("\n" + "=" * 60)
    print("🔌 Wire Branch Counter (Server-Side)")
    print("=" * 60)
    print(f"📁 Base directory : {BASE_DIR}")
    print(f"🗄️ Database       : {DB_PATH}")
    print(f"📋 Table          : {DB_TABLE}")

    init_db()
    print("✅ Database initialized")

    try:
        load_model()
    except Exception as e:
        print(f"⚠️ Model error: {e}")

    print("=" * 60 + "\n")

    yield

    print("👋 Server shutdown")


# =========================================================
# FASTAPI APP
# =========================================================

app = FastAPI(
    title="Wire Branch Counter",
    description="YOLO Wire Counter with Server-Side Inference",
    version="2.0.0",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# =========================================================
# MODELS
# =========================================================

class DetectionRequest(BaseModel):
    image_base64: str  # Base64 encoded image


class SaveDetectionRequest(BaseModel):
    countc: int
    dt_ins: str
    seq_time: str
    area: str


# =========================================================
# ROUTES
# =========================================================

@app.get("/")
async def root():
    """Serve main page"""
    index_path = TEMPLATES_DIR / "index.html"
    if not index_path.exists():
        raise HTTPException(status_code=404, detail="index.html tidak ditemukan")
    return FileResponse(str(index_path))


@app.get("/health")
async def health():
    """Health check"""
    return {
        "status": "ok",
        "model": "onnx_runtime",
        "inference_mode": "server_side",
        "device": "cpu" if session is None else str(session.get_providers()),
        "database": DB_PATH
    }


@app.post("/api/detect")
async def detect(request: DetectionRequest):
    """Detect objects in image (server-side inference)"""
    if session is None:
        raise HTTPException(status_code=500, detail="Model belum diload")

    try:
        # Decode base64 image
        image_bytes = base64.b64decode(request.image_base64)

        # Run inference
        detections = run_inference(image_bytes)

        return {
            "success": True,
            "count": len(detections),
            "detections": detections
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference error: {str(e)}")


@app.post("/api/save-detection")
async def save_detection(req: SaveDetectionRequest):
    """Save detection result to database"""
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()

        cursor.execute(f"""
            INSERT INTO {DB_TABLE} (countc, dt_ins, seq_time, area)
            VALUES (?, ?, ?, ?)
        """, (req.countc, req.dt_ins, req.seq_time, req.area))

        conn.commit()
        inserted_id = cursor.lastrowid
        conn.close()

        return {
            "success": True,
            "id": inserted_id,
            "count": req.countc,
            "area": req.area,
            "dt_ins": req.dt_ins,
            "seq_time": req.seq_time
        }

    except sqlite3.Error as e:
        raise HTTPException(status_code=500, detail=f"Database error: {e}")


@app.get("/api/detections")
async def get_detections(limit: int = 100, area: Optional[str] = None):
    """Get all detections"""
    try:
        conn = sqlite3.connect(DB_PATH)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()

        if area:
            cursor.execute(f"""
                SELECT * FROM {DB_TABLE}
                WHERE area = ?
                ORDER BY id DESC
                LIMIT ?
            """, (area, limit))
        else:
            cursor.execute(f"""
                SELECT * FROM {DB_TABLE}
                ORDER BY id DESC
                LIMIT ?
            """, (limit,))

        rows = cursor.fetchall()
        conn.close()

        return {
            "success": True,
            "count": len(rows),
            "data": [dict(row) for row in rows]
        }

    except sqlite3.Error as e:
        raise HTTPException(status_code=500, detail=f"Database error: {e}")


# =========================================================
# STATIC FILES
# =========================================================

if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


# =========================================================
# MAIN
# =========================================================

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
