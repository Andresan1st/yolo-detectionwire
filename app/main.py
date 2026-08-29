"""
FastAPI server for Wire Branch Counter with ONNX Runtime Web
"""
import os
import sqlite3
from datetime import datetime
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
from fastapi import Request
from typing import Optional

# Database config
DB_PATH = os.getenv("DB_PATH", "detections.db")
DB_TABLE = os.getenv("DB_TABLE", "machine_detection_copper")

BASE_DIR = Path(__file__).parent
STATIC_DIR = BASE_DIR / "static"
TEMPLATES_DIR = BASE_DIR / "templates"

# Global model (for non-ONNX fallback API)
_model = None


def init_db():
    """Initialize SQLite database"""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute(f"""
        CREATE TABLE IF NOT EXISTS {DB_TABLE} (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            countc INTEGER,
            dt_ins TEXT,
            seq_time TEXT,
            area TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    conn.commit()
    conn.close()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup and shutdown"""
    # Startup
    init_db()
    print("✅ Database initialized")
    yield
    # Shutdown
    print("👋 Server shutdown")


# Create FastAPI app
app = FastAPI(
    title="Wire Branch Counter",
    description="YOLO Detection with ONNX Runtime Web",
    version="2.0",
    lifespan=lifespan
)

# CORS for development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
async def root():
    """Serve main page"""
    return FileResponse(str(TEMPLATES_DIR / "index.html"))


@app.get("/health")
async def health():
    """Health check endpoint"""
    return {
        "status": "ok",
        "model": "onnx_runtime_web",
        "model_loaded": True,
        "device": "browser"
    }


@app.post("/api/save-detection")
async def save_detection(request: Request):
    """Save detection result to database"""
    data = await request.json()

    countc = data.get("countc", 0)
    dt_ins = data.get("dt_ins")
    seq_time = data.get("seq_time")
    area = data.get("area", "default")

    if not dt_ins:
        local_time = datetime.now()
        dt_ins = local_time.strftime("%Y-%m-%d")
    if not seq_time:
        seq_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute(
            f"INSERT INTO {DB_TABLE} (countc, dt_ins, seq_time, area) VALUES (?, ?, ?, ?)",
            (countc, dt_ins, seq_time, area)
        )
        conn.commit()
        inserted_id = cursor.lastrowid
        conn.close()

        return JSONResponse({
            "success": True,
            "id": inserted_id,
            "count": countc,
            "area": area,
            "dt_ins": dt_ins,
            "seq_time": seq_time
        })
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/detections")
async def get_detections(limit: int = 100, area: Optional[str] = None):
    """Get recent detections"""
    try:
        conn = sqlite3.connect(DB_PATH)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()

        if area:
            cursor.execute(
                f"SELECT * FROM {DB_TABLE} WHERE area = ? ORDER BY id DESC LIMIT ?",
                (area, limit)
            )
        else:
            cursor.execute(
                f"SELECT * FROM {DB_TABLE} ORDER BY id DESC LIMIT ?",
                (limit,)
            )

        rows = cursor.fetchall()
        conn.close()

        return JSONResponse({
            "success": True,
            "data": [dict(row) for row in rows]
        })
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# Mount static files
if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)
