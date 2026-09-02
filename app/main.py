"""
Backend API untuk Copper Wire Branch Detection
Dapat menerima gambar dari Laravel via base64 atau multipart
"""

import base64
import json
import os
import time
from datetime import datetime, timezone, timedelta
from pathlib import Path
from threading import Lock
from typing import Optional

# Asia/Jakarta timezone (UTC+7)
JAKARTA_TZ = timezone(timedelta(hours=7))

import cv2
import numpy as np
from dotenv import load_dotenv
from fastapi import FastAPI, File, HTTPException, UploadFile, Body, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from ultralytics import YOLO

# Try to import database drivers
PYODBC_AVAILABLE = False
PYMSSQL_AVAILABLE = False

try:
    import pymssql
    PYMSSQL_AVAILABLE = True
except ImportError:
    pymssql = None

try:
    import pyodbc
    PYODBC_AVAILABLE = True
except ImportError:
    pyodbc = None

load_dotenv()

ROOT = Path(__file__).resolve().parent.parent
TEMPLATE_DIR = Path(__file__).resolve().parent / "templates"
STATIC_DIR = Path(__file__).resolve().parent / "static"
CAPTURES_DIR = STATIC_DIR / "captures"
MODEL_PATH = Path(os.getenv("MODEL_PATH", "models/best.pt"))
if not MODEL_PATH.is_absolute():
    MODEL_PATH = ROOT / MODEL_PATH
CONFIDENCE = float(os.getenv("CONFIDENCE", "0.35"))
IOU = float(os.getenv("IOU", "0.45"))
DEVICE = os.getenv("DEVICE", "cpu")
MAX_FRAME_WIDTH = int(os.getenv("MAX_FRAME_WIDTH", "416"))  # YOLO optimal = FAST!
WEBSOCKET_INTERVAL = float(os.getenv("WEBSOCKET_INTERVAL", "0.15"))  # 150ms!

# Create captures directory if not exists
CAPTURES_DIR.mkdir(parents=True, exist_ok=True)

# Database Configuration
DB_HOST = os.getenv("DB_HOST", "localhost")
DB_USER = os.getenv("DB_USER", "sa")
DB_PASSWORD = os.getenv("DB_PASSWORD", "")
DB_NAME = os.getenv("DB_NAME", "master")
DB_TABLE = os.getenv("DB_TABLE", "machine_detection_copper")

app = FastAPI(
    title="Wire Branch Detection API",
    version="1.0.0",
    description="API untuk deteksi kawat tembaga bercabang"
)
origins = [item.strip() for item in os.getenv("ALLOWED_ORIGINS", "*").split(",")]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=False,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)

# Serve captured images as static files
app.mount("/captures", StaticFiles(directory=str(CAPTURES_DIR)), name="captures")

_model = None
_model_lock = Lock()


def get_model() -> YOLO:
    global _model
    if _model is not None:
        return _model
    if not MODEL_PATH.exists():
        raise HTTPException(
            status_code=503,
            detail=f"Model belum ditemukan: {MODEL_PATH}. Letakkan model YOLO custom di lokasi tersebut.",
        )
    with _model_lock:
        if _model is None:
            _model = YOLO(str(MODEL_PATH))
    return _model


def decode_image(image_data: bytes) -> Optional[np.ndarray]:
    """Decode image dari bytes ke numpy array"""
    frame = cv2.imdecode(np.frombuffer(image_data, dtype=np.uint8), cv2.IMREAD_COLOR)
    return frame


def save_capture(image_data: bytes, detections: list) -> str:
    """
    Simpan capture gambar ke folder captures.
    Returns: filename yang disimpan
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    count_suffix = f"_count{len(detections)}" if detections else "_nocount"
    filename = f"capture_{timestamp}{count_suffix}.jpg"
    filepath = CAPTURES_DIR / filename

    # Decode and save image
    frame = decode_image(image_data)
    if frame is not None:
        cv2.imwrite(str(filepath), frame)
        return filename
    return ""


def detect_upload_fast(frame: np.ndarray) -> tuple[list[dict], str]:
    """
    FAST detection for uploaded images.
    - Resize to 1280px max (matching frontend max size)
    - Skip annotation (return empty string)
    """
    height, width = frame.shape[:2]

    # Resize ke 1280px max (optimal for YOLO, same as frontend)
    MAX_UPLOAD_SIZE = 1280
    if max(width, height) > MAX_UPLOAD_SIZE:
        scale = MAX_UPLOAD_SIZE / max(width, height)
        new_width = int(width * scale)
        new_height = int(height * scale)
        frame = cv2.resize(frame, (new_width, new_height))

    # Deteksi dengan YOLO
    model = get_model()
    result = model.predict(frame, conf=CONFIDENCE, iou=IOU, device=DEVICE, verbose=False)[0]

    # Parse hasil deteksi
    detections = []
    names = result.names or {}
    boxes = result.boxes

    if boxes is not None:
        for box in boxes:
            coordinates = box.xyxy[0].cpu().numpy().astype(int).tolist()
            class_id = int(box.cls[0].item())
            confidence = float(box.conf[0].item())
            label = str(names.get(class_id, class_id))

            x1, y1, x2, y2 = coordinates

            detections.append({
                "class_id": class_id,
                "label": label,
                "confidence": round(confidence, 3),
                "bbox": {
                    "x1": x1,
                    "y1": y1,
                    "x2": x2,
                    "y2": y2,
                    "width": x2 - x1,
                    "height": y2 - y1
                }
            })

    # Skip annotation untuk speed
    return detections, ""


def detect_fast(frame: np.ndarray) -> list[dict]:
    """
    FAST detection - NO annotation, returns only detections.
    Optimized for real-time WebSocket streaming.
    Keeps original resolution for accurate bounding boxes.
    """
    # Deteksi dengan YOLO (no resize for WebSocket)
    model = get_model()
    result = model.predict(frame, conf=CONFIDENCE, iou=IOU, device=DEVICE, verbose=False)[0]

    # Parse hasil deteksi
    detections = []
    names = result.names or {}
    boxes = result.boxes

    if boxes is not None:
        for box in boxes:
            coordinates = box.xyxy[0].cpu().numpy().astype(int).tolist()
            class_id = int(box.cls[0].item())
            confidence = float(box.conf[0].item())
            label = str(names.get(class_id, class_id))

            x1, y1, x2, y2 = coordinates

            detections.append({
                "class_id": class_id,
                "label": label,
                "confidence": round(confidence, 3),
                "bbox": {
                    "x1": x1,
                    "y1": y1,
                    "x2": x2,
                    "y2": y2,
                    "width": x2 - x1,
                    "height": y2 - y1
                }
            })

    return detections


def detect_in_image(frame: np.ndarray) -> tuple[list[dict], str]:
    """
    Deteksi objek dalam gambar.
    Returns: (detections, annotated_image_base64)
    """
    height, width = frame.shape[:2]

    # Resize jika terlalu besar
    if width > MAX_FRAME_WIDTH:
        scale = MAX_FRAME_WIDTH / width
        frame = cv2.resize(frame, (MAX_FRAME_WIDTH, int(height * scale)))

    # Deteksi dengan YOLO
    model = get_model()
    result = model.predict(frame, conf=CONFIDENCE, iou=IOU, device=DEVICE, verbose=False)[0]

    # Parse hasil deteksi
    detections = []
    names = result.names or {}
    boxes = result.boxes

    if boxes is not None:
        for box in boxes:
            coordinates = box.xyxy[0].cpu().numpy().astype(int).tolist()
            class_id = int(box.cls[0].item())
            confidence = float(box.conf[0].item())
            label = str(names.get(class_id, class_id))

            x1, y1, x2, y2 = coordinates

            detections.append({
                "class_id": class_id,
                "label": label,
                "confidence": round(confidence, 3),
                "bbox": {
                    "x1": x1,
                    "y1": y1,
                    "x2": x2,
                    "y2": y2,
                    "width": x2 - x1,
                    "height": y2 - y1
                }
            })

            # Gambar bounding box
            cv2.rectangle(frame, (x1, y1), (x2, y2), (40, 220, 120), 2)
            cv2.putText(
                frame,
                f"{label} {confidence:.0%}",
                (x1, max(24, y1 - 8)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (40, 220, 120),
                2,
                cv2.LINE_AA,
            )

    # Encode annotated image ke base64
    success, encoded = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 82])
    if not success:
        return detections, ""

    annotated_base64 = base64.b64encode(encoded.tobytes()).decode("ascii")
    return detections, annotated_base64


# ==================== REQUEST/RESPONSE MODELS ====================

class DetectionRequest(BaseModel):
    """Request body untuk deteksi via JSON/base64"""
    image: str  # Base64 encoded image (tanpa prefix data:image/jpeg;base64,)
    include_annotated: bool = True  # Apakah perlu mengembalikan gambar berannotated


class DetectionResponse(BaseModel):
    """Response untuk deteksi"""
    success: bool
    count: int
    detections: list[dict]
    annotated_image: Optional[str] = None  # Base64
    saved_file: Optional[str] = None  # Filename yang disimpan


class HealthResponse(BaseModel):
    """Response untuk health check"""
    status: str
    model: str
    model_loaded: bool
    device: str
    confidence: float
    iou: float


# ==================== API ENDPOINTS ====================

@app.get("/")
def root():
    """Serve frontend HTML page"""
    return FileResponse(TEMPLATE_DIR / "index.html")


@app.get("/ui")
def root_ui():
    """Serve frontend HTML page (alias)"""
    return FileResponse(TEMPLATE_DIR / "index.html")


@app.get("/health", response_model=HealthResponse)
def health():
    """Health check endpoint"""
    return HealthResponse(
        status="ok",
        model=str(MODEL_PATH),
        model_loaded=_model is not None,
        device=DEVICE,
        confidence=CONFIDENCE,
        iou=IOU
    )


@app.get("/{area}")
def root_with_area(area: str):
    """Serve frontend HTML page with area identifier (e.g., /labqc, /produksi)"""
    # Skip API routes
    if area in ["api", "health", "info", "ui", "ws", "docs", "openapi.json", "redoc"]:
        raise HTTPException(status_code=404, detail="Not found")
    return FileResponse(TEMPLATE_DIR / "index.html")


@app.get("/info")
def info():
    """Info tentang model dan konfigurasi"""
    return {
        "model_path": str(MODEL_PATH),
        "model_exists": MODEL_PATH.exists(),
        "model_loaded": _model is not None,
        "confidence_threshold": CONFIDENCE,
        "iou_threshold": IOU,
        "device": DEVICE,
        "max_frame_width": MAX_FRAME_WIDTH
    }


# ==================== DETECTION ENDPOINTS ====================

@app.post("/api/detect", response_model=DetectionResponse)
async def detect_upload(file: UploadFile = File(...)):
    """
    Deteksi dengan upload file (multipart/form-data).
    Compatible dengan Laravel/frontend lama.
    """
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(
            status_code=415,
            detail="File harus berupa gambar."
        )

    raw = await file.read()
    frame = decode_image(raw)
    if frame is None:
        raise HTTPException(
            status_code=400,
            detail="Frame tidak dapat dibaca."
        )

    # Simpan capture ke folder
    saved_filename = save_capture(raw, [])

    # Detection dengan annotate (asli, sebelum perubahan)
    detections, annotated_base64 = detect_in_image(frame)

    return DetectionResponse(
        success=True,
        count=len(detections),
        detections=detections,
        annotated_image=annotated_base64 if annotated_base64 else None,
        saved_file=saved_filename if saved_filename else None
    )


@app.post("/api/detect/base64", response_model=DetectionResponse)
async def detect_base64(request: DetectionRequest):
    """
    Deteksi dengan gambar base64 (JSON body).
    Cocok untuk integrasi dengan Laravel.

    Body JSON:
    {
        "image": "base64_encoded_image_without_prefix",
        "include_annotated": true
    }
    """
    try:
        # Hapus prefix jika ada
        image_data = request.image
        if "," in image_data:
            image_data = image_data.split(",")[-1]

        # Decode base64 ke bytes
        raw = base64.b64decode(image_data)
        frame = decode_image(raw)

        if frame is None:
            raise HTTPException(
                status_code=400,
                detail="Gambar tidak dapat dibaca."
            )

        # FAST detection - optimized for speed
        detections, _ = detect_upload_fast(frame)

        return DetectionResponse(
            success=True,
            count=len(detections),
            detections=detections,
            annotated_image=None
        )

    except base64.binascii.Error:
        raise HTTPException(
            status_code=400,
            detail="Format base64 tidak valid."
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error processing image: {str(e)}"
        )


@app.post("/api/detect/raw")
async def detect_raw(image_data: bytes = Body(...)):
    """
    Deteksi dengan raw bytes image (tanpa wrapper JSON).
    Paling cepat untuk streaming data.

    Headers: Content-Type: image/jpeg (atau image/png)
    """
    frame = decode_image(image_data)
    if frame is None:
        raise HTTPException(
            status_code=400,
            detail="Gambar tidak dapat dibaca."
        )

    detections, annotated_base64 = detect_in_image(frame)

    return JSONResponse({
        "success": True,
        "count": len(detections),
        "detections": detections,
        "annotated_image": annotated_base64 if annotated_base64 else None
    })


@app.websocket("/ws/detect")
async def websocket_detect(websocket: WebSocket):
    """Real-time webcam detection over WebSocket - FAST MODE."""
    await websocket.accept()
    try:
        while True:
            raw = await websocket.receive_bytes()
            frame = decode_image(raw)
            if frame is None:
                await websocket.send_json({
                    "success": False,
                    "error": "Frame tidak dapat dibaca."
                })
                continue

            # FAST detection - no resize, no annotation
            detections = detect_fast(frame)
            await websocket.send_json({
                "success": True,
                "count": len(detections),
                "detections": detections
            })
    except WebSocketDisconnect:
        return
    except Exception as e:
        try:
            await websocket.send_json({
                "success": False,
                "error": str(e)
            })
        except RuntimeError:
            pass


# ==================== BATCH DETECTION ====================

class BatchDetectionRequest(BaseModel):
    """Request untuk deteksi banyak gambar sekaligus"""
    images: list[str]  # Array of base64 images
    include_annotated: bool = True


@app.post("/api/detect/batch", response_model=dict)
async def detect_batch(request: BatchDetectionRequest):
    """
    Deteksi banyak gambar sekaligus.
    Cocok untuk proses batch dari Laravel.

    Body JSON:
    {
        "images": ["base64_image_1", "base64_image_2", ...],
        "include_annotated": true
    }
    """
    results = []

    for i, img_base64 in enumerate(request.images):
        try:
            # Hapus prefix jika ada
            image_data = img_base64
            if "," in image_data:
                image_data = image_data.split(",")[-1]

            raw = base64.b64decode(image_data)
            frame = decode_image(raw)

            if frame is None:
                results.append({
                    "index": i,
                    "success": False,
                    "error": "Cannot decode image"
                })
                continue

            detections, annotated_base64 = detect_in_image(frame)

            results.append({
                "index": i,
                "success": True,
                "count": len(detections),
                "detections": detections,
                "annotated_image": annotated_base64 if request.include_annotated else None
            })

        except Exception as e:
            results.append({
                "index": i,
                "success": False,
                "error": str(e)
            })

    return {
        "total": len(request.images),
        "processed": len([r for r in results if r.get("success")]),
        "results": results
    }


# ==================== DATABASE ENDPOINTS ====================

class SaveDetectionRequest(BaseModel):
    """Request untuk menyimpan hasil deteksi ke database"""
    countc: int
    dt_ins: Optional[str] = None
    seq_time: Optional[str] = None
    area: Optional[str] = None


class SaveDetectionResponse(BaseModel):
    """Response untuk save detection"""
    success: bool
    id: Optional[int] = None
    countc: int
    area: str
    message: str
  


def get_db_connection():
    """Create database connection"""
    if PYMSSQL_AVAILABLE:
        # Use pymssql (recommended - easier to install)
        try:
            conn = pymssql.connect(
                server=DB_HOST,
                user=DB_USER,
                password=DB_PASSWORD,
                database=DB_NAME
            )
            return conn
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Database connection failed (pymssql): {str(e)}")

    elif PYODBC_AVAILABLE:
        # Fallback to pyodbc
        connection_string = (
            f"DRIVER={{ODBC Driver 17 for SQL Server}};"
            f"SERVER={DB_HOST};"
            f"DATABASE={DB_NAME};"
            f"UID={DB_USER};"
            f"PWD={DB_PASSWORD};"
        )
        try:
            conn = pyodbc.connect(connection_string)
            return conn
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Database connection failed (pyodbc): {str(e)}")

    else:
        raise HTTPException(
            status_code=500,
            detail="No database driver available. Install pymssql: pip install pymssql"
        )


@app.post("/api/save-detection", response_model=SaveDetectionResponse)
async def save_detection(request: SaveDetectionRequest):
    """
    Simpan hasil deteksi ke SQL Server database.
    """
    if not PYMSSQL_AVAILABLE and not PYODBC_AVAILABLE:
        raise HTTPException(
            status_code=500,
            detail="Database driver not installed. Run: pip install pymssql"
        )

    try:
        conn = get_db_connection()
        cursor = conn.cursor()

        # Get timestamp - use system LOCAL time (Asia/Jakarta WIB)
        # time.localtime() returns local time based on system timezone
        local_time = time.localtime()

        if request.dt_ins:
            dt_ins = request.dt_ins
        else:
            dt_ins = time.strftime("%Y-%m-%d", local_time)

        seq_time = time.strftime("%Y-%m-%d %H:%M:%S", local_time)

        area = request.area if request.area else "default"

        # Insert query - pymssql uses %s, pyodbc uses ?
        if PYMSSQL_AVAILABLE:
            insert_query = f"""
                INSERT INTO {DB_TABLE} (countc, dt_ins, seq_time, area)
                VALUES (%s, %s, %s, %s)
            """
        else:
            insert_query = f"""
                INSERT INTO {DB_TABLE} (countc, dt_ins, seq_time, area)
                OUTPUT INSERTED.id
                VALUES (?, ?, ?, ?)
            """

        if PYMSSQL_AVAILABLE:
            cursor.execute(insert_query, (request.countc, dt_ins, seq_time, area))
            cursor.execute("SELECT SCOPE_IDENTITY()")
            row = cursor.fetchone()
            inserted_id = row[0] if row else None
        else:
            cursor.execute(insert_query, (request.countc, dt_ins, seq_time, area))
            row = cursor.fetchone()
            inserted_id = row[0] if row else None

        conn.commit()
        cursor.close()
        conn.close()

        return SaveDetectionResponse(
            success=True,
            id=inserted_id,
            countc=request.countc,
            area=area,
            message=f"Saved successfully at {seq_time}, Area: {area}"
        )

        conn.commit()
        cursor.close()
        conn.close()

        return SaveDetectionResponse(
            success=True,
            id=inserted_id,
            countc=request.countc,
            message=f"Saved successfully at {seq_time}"
        )

    except pymssql.Error as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(
            status_code=500,
            detail=f"Database error (pymssql): {str(e)}"
        )
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(
            status_code=500,
            detail=f"Error: {str(e)}"
        )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)