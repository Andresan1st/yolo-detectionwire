"""
FastAPI Application - Copper Detection WebRTC
"""
import os
import base64
import uuid
import cv2
import numpy as np
from datetime import datetime
from io import BytesIO
from typing import List, Optional
from contextlib import asynccontextmanager

from fastapi import FastAPI, UploadFile, File, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
from PIL import Image

# Import app modules
from app.database import init_database, save_detection, get_recent_detections
from app.detector import get_detector, CopperDetector

# Configuration
APP_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_DIR = os.path.join(APP_DIR, 'uploads')
STATIC_DIR = os.path.join(APP_DIR, 'static')
TEMPLATES_DIR = os.path.join(APP_DIR, 'templates')
os.makedirs(UPLOAD_DIR, exist_ok=True)

def read_html_file(filename: str) -> str:
    """Read HTML template file"""
    filepath = os.path.join(TEMPLATES_DIR, filename)
    with open(filepath, 'r', encoding='utf-8') as f:
        return f.read()

# Lifespan for startup/shutdown
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan events"""
    # Startup
    print("🚀 Starting Copper Detection API...")
    init_database()
    print("✅ Database initialized")

    # Initialize detector
    detector = get_detector()
    if detector.session:
        print("✅ ONNX Model loaded")
    else:
        print("⚠️ ONNX Model not found - detection will be simulated")

    yield

    # Shutdown
    print("👋 Shutting down...")

# Create FastAPI app
app = FastAPI(
    title="Copper Detection API",
    description="WebRTC Camera Detection for Copper Elements",
    version="1.0.0",
    lifespan=lifespan
)

# Static files and templates
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
app.mount("/uploads", StaticFiles(directory=UPLOAD_DIR), name="uploads")
templates = Jinja2Templates(directory=TEMPLATES_DIR)

# Pydantic models
class DetectionResult(BaseModel):
    id: int
    timestamp: str
    detection_count: int
    confidence_avg: float
    element_type: str
    status: str
    camera_source: Optional[str] = None

class DetectionResponse(BaseModel):
    detections: int
    confidence_avg: float
    element: str
    status: str
    saved_id: Optional[int] = None

class FrameDetectionRequest(BaseModel):
    image: str  # Base64 encoded image
    camera_source: str = "webcam"

# ==================== ROUTES ====================

@app.get("/", response_class=HTMLResponse)
async def root():
    """Serve main page"""
    return HTMLResponse(content=read_html_file("index.html"))

@app.get("/camera", response_class=HTMLResponse)
async def camera_page():
    """Serve camera detection page"""
    return HTMLResponse(content=read_html_file("camera.html"))

@app.get("/upload", response_class=HTMLResponse)
async def upload_page():
    """Serve upload page"""
    return HTMLResponse(content=read_html_file("upload.html"))

@app.get("/history", response_class=HTMLResponse)
async def history_page():
    """Serve detection history page"""
    return HTMLResponse(content=read_html_file("history.html"))

# ==================== API ENDPOINTS ====================

@app.post("/api/detect/frame")
async def detect_frame(data: FrameDetectionRequest):
    """Detect copper in a single frame from WebRTC

    Args:
        data: JSON with base64 encoded image

    Returns:
        Detection results
    """
    try:
        # Decode base64 image
        image_data = base64.b64decode(data.image.split(',')[1] if ',' in data.image else data.image)
        nparr = np.frombuffer(image_data, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if image is None:
            raise HTTPException(status_code=400, detail="Invalid image data")

        # Run detection
        detector = get_detector()
        detections = detector.detect(image)

        # Calculate stats
        detection_count = len(detections)
        confidence_avg = np.mean([d.confidence for d in detections]) if detections else 0.0

        # Save to database
        saved_id = save_detection({
            'detection_count': detection_count,
            'confidence_avg': float(confidence_avg),
            'element_type': 'copper',
            'status': 'detected' if detection_count > 0 else 'not_found',
            'camera_source': data.camera_source,
        })

        return DetectionResponse(
            detections=detection_count,
            confidence_avg=float(confidence_avg),
            element='copper',
            status='detected' if detection_count > 0 else 'not_found',
            saved_id=saved_id
        )

    except Exception as e:
        print(f"Detection error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/detect/image")
async def detect_image(file: UploadFile = File(...)):
    """Upload and detect copper in an image

    Args:
        file: Image file upload

    Returns:
        Detection results with saved image path
    """
    try:
        # Validate file type
        if not file.content_type.startswith('image/'):
            raise HTTPException(status_code=400, detail="File must be an image")

        # Read image
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if image is None:
            raise HTTPException(status_code=400, detail="Invalid image file")

        # Save uploaded image
        filename = f"{uuid.uuid4()}.jpg"
        filepath = os.path.join(UPLOAD_DIR, filename)
        cv2.imwrite(filepath, image)

        # Run detection
        detector = get_detector()
        detections = detector.detect(image)

        # Draw detections on image
        result_image = detector.draw_detections(image, detections)

        # Save result image
        result_filename = f"result_{filename}"
        result_filepath = os.path.join(UPLOAD_DIR, result_filename)
        cv2.imwrite(result_filepath, result_image)

        # Calculate stats
        detection_count = len(detections)
        confidence_avg = np.mean([d.confidence for d in detections]) if detections else 0.0

        # Save to database
        saved_id = save_detection({
            'detection_count': detection_count,
            'confidence_avg': float(confidence_avg),
            'image_path': result_filepath,
            'element_type': 'copper',
            'status': 'detected' if detection_count > 0 else 'not_found',
        })

        # Convert result image to base64
        _, buffer = cv2.imencode('.jpg', result_image)
        result_base64 = base64.b64encode(buffer).decode()

        return {
            "detections": detection_count,
            "confidence_avg": float(confidence_avg),
            "element": "copper",
            "status": "detected" if detection_count > 0 else "not_found",
            "saved_id": saved_id,
            "image_path": f"/uploads/{result_filename}",
            "result_image": f"data:image/jpeg;base64,{result_base64}"
        }

    except HTTPException:
        raise
    except Exception as e:
        print(f"Upload detection error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/detections", response_model=List[DetectionResult])
async def get_detections(limit: int = 50):
    """Get recent detection records

    Args:
        limit: Maximum number of records to return

    Returns:
        List of detection records
    """
    try:
        detections = get_recent_detections(limit)

        # Convert datetime objects to strings
        for det in detections:
            if det.get('timestamp'):
                det['timestamp'] = str(det['timestamp'])
            if det.get('created_at'):
                det['created_at'] = str(det['created_at'])

        return detections

    except Exception as e:
        print(f"Get detections error: {e}")
        return []

@app.get("/api/stats")
async def get_stats():
    """Get detection statistics"""
    try:
        detections = get_recent_detections(1000)

        if not detections:
            return {
                "total_detections": 0,
                "avg_confidence": 0,
                "detection_rate": 0,
                "recent_count": 0
            }

        total = len(detections)
        detected = sum(1 for d in detections if d.get('status') == 'detected')
        avg_conf = np.mean([d.get('confidence_avg', 0) for d in detections])

        # Last 24 hours
        from datetime import timedelta
        recent = sum(1 for d in detections
                    if d.get('timestamp') and
                    datetime.now() - d['timestamp'] < timedelta(hours=24))

        return {
            "total_detections": total,
            "avg_confidence": round(float(avg_conf), 3),
            "detection_rate": round(detected / total * 100, 1) if total > 0 else 0,
            "recent_count": recent
        }

    except Exception as e:
        print(f"Stats error: {e}")
        return {
            "total_detections": 0,
            "avg_confidence": 0,
            "detection_rate": 0,
            "recent_count": 0
        }

@app.get("/uploads/{filename}")
async def get_upload(filename: str):
    """Serve uploaded files"""
    filepath = os.path.join(UPLOAD_DIR, filename)
    if not os.path.exists(filepath):
        raise HTTPException(status_code=404, detail="File not found")

    return StreamingResponse(
        open(filepath, "rb"),
        media_type="image/jpeg"
    )

# ==================== MAIN ====================

if __name__ == "__main__":
    import uvicorn

    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", 8000))

    print(f"""
    ╔══════════════════════════════════════════════════════╗
    ║     Copper Detection API - WebRTC Camera             ║
    ╠══════════════════════════════════════════════════════╣
    ║  Local:    http://localhost:{port}                     ║
    ║  Network:  http://{host}:{port}                    ║
    ╠══════════════════════════════════════════════════════╣
    ║  Endpoints:                                          ║
    ║  • /          - Main page                           ║
    ║  • /camera    - WebRTC Camera detection             ║
    ║  • /upload    - Image upload detection              ║
    ║  • /history   - Detection history                   ║
    ╚══════════════════════════════════════════════════════╝
    """)

    uvicorn.run("main:app", host=host, port=port, reload=True)
