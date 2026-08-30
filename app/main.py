"""
FastAPI server
Wire Branch Counter
YOLO ONNX Runtime Web

Inference dilakukan langsung di browser.
Python hanya menangani:

- serve HTML
- serve ONNX
- health check
- save detection
- get detection
"""

import os
import sqlite3
from pathlib import Path
from datetime import datetime
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse


# =========================================================
# CONFIG
# =========================================================

BASE_DIR = Path(__file__).resolve().parent
STATIC_DIR = BASE_DIR / "static"
TEMPLATES_DIR = BASE_DIR / "templates"
MODEL_PATH = BASE_DIR / "best.onnx"

DB_PATH = os.getenv(
    "DB_PATH",
    str(BASE_DIR / "detections.db")
)

DB_TABLE = os.getenv(
    "DB_TABLE",
    "machine_detection_copper"
)


# =========================================================
# DATABASE
# =========================================================

def init_db():
    """
    Initialize SQLite database.
    """

    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    cursor.execute(
        f"""
        CREATE TABLE IF NOT EXISTS {DB_TABLE} (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            countc INTEGER NOT NULL,
            dt_ins TEXT NOT NULL,
            seq_time TEXT NOT NULL,
            area TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        """
    )

    conn.commit()
    conn.close()


# =========================================================
# LIFESPAN
# =========================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    print("")
    print("=" * 60)
    print("🔌 Wire Branch Counter")
    print("=" * 60)

    print(f"📁 Base directory : {BASE_DIR}")
    print(f"📦 ONNX model     : {MODEL_PATH}")
    print(f"🗄️ Database       : {DB_PATH}")
    print(f"📋 Table          : {DB_TABLE}")

    print("=" * 60)

    init_db()

    print("✅ Database initialized")

    if MODEL_PATH.exists():
        model_size = MODEL_PATH.stat().st_size
        model_size_mb = model_size / (1024 * 1024)

        print(f"✅ ONNX found: {model_size_mb:.2f} MB")
    else:
        print("⚠️ WARNING: best.onnx tidak ditemukan!")

    print("=" * 60)
    print("")

    yield

    print("👋 Server shutdown")


# =========================================================
# FASTAPI
# =========================================================

app = FastAPI(
    title="Wire Branch Counter",
    description="YOLO Wire Counter with ONNX Runtime Web",
    version="2.0.0",
    lifespan=lifespan
)


# =========================================================
# CORS
# =========================================================

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# =========================================================
# ROOT
# =========================================================

@app.get("/")
async def root():
    index_path = TEMPLATES_DIR / "index.html"

    if not index_path.exists():
        raise HTTPException(
            status_code=404,
            detail="templates/index.html tidak ditemukan"
        )

    return FileResponse(str(index_path))


# =========================================================
# HEALTH
# =========================================================

@app.get("/health")
async def health():
    model_exists = MODEL_PATH.exists()

    return {
        "status": "ok",
        "model": "onnx_runtime_web",
        "model_loaded": model_exists,
        "device": "browser",
        "model_file": str(MODEL_PATH),
        "database": DB_PATH,
        "table": DB_TABLE
    }


# =========================================================
# SAVE DETECTION
# =========================================================

@app.post("/api/save-detection")
async def save_detection(request: Request):

    try:
        data = await request.json()

    except Exception:
        raise HTTPException(
            status_code=400,
            detail="Invalid JSON"
        )

    # -----------------------------------------------------
    # GET DATA
    # -----------------------------------------------------

    countc = data.get("countc", 0)
    dt_ins = data.get("dt_ins")
    seq_time = data.get("seq_time")
    area = data.get("area", "default")

    # -----------------------------------------------------
    # VALIDATE COUNT
    # -----------------------------------------------------

    try:
        countc = int(countc)

    except (ValueError, TypeError):
        raise HTTPException(
            status_code=400,
            detail="countc harus berupa angka"
        )

    if countc < 0:
        raise HTTPException(
            status_code=400,
            detail="countc tidak boleh negatif"
        )

    # -----------------------------------------------------
    # AREA
    # -----------------------------------------------------

    if area is None:
        area = "default"

    area = str(area).strip()

    if not area:
        area = "default"

    # -----------------------------------------------------
    # DATE
    # -----------------------------------------------------

    if not dt_ins:
        dt_ins = datetime.now().strftime("%Y-%m-%d")

    # -----------------------------------------------------
    # DATETIME
    # -----------------------------------------------------

    if not seq_time:
        seq_time = datetime.now().strftime(
            "%Y-%m-%d %H:%M:%S"
        )

    # -----------------------------------------------------
    # INSERT
    # -----------------------------------------------------

    conn = None

    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()

        query = f"""
            INSERT INTO {DB_TABLE}
            (
                countc,
                dt_ins,
                seq_time,
                area
            )
            VALUES (?, ?, ?, ?)
        """

        cursor.execute(
            query,
            (
                countc,
                dt_ins,
                seq_time,
                area
            )
        )

        conn.commit()

        inserted_id = cursor.lastrowid

        return JSONResponse(
            status_code=200,
            content={
                "success": True,
                "id": inserted_id,
                "count": countc,
                "area": area,
                "dt_ins": dt_ins,
                "seq_time": seq_time
            }
        )

    except sqlite3.Error as e:

        if conn:
            conn.rollback()

        raise HTTPException(
            status_code=500,
            detail=f"Database error: {e}"
        )

    finally:

        if conn:
            conn.close()


# =========================================================
# GET DETECTIONS
# =========================================================

@app.get("/api/detections")
async def get_detections(
    limit: int = 100,
    area: Optional[str] = None
):

    # -----------------------------------------------------
    # LIMIT
    # -----------------------------------------------------

    if limit < 1:
        limit = 1

    if limit > 1000:
        limit = 1000

    conn = None

    try:
        conn = sqlite3.connect(DB_PATH)
        conn.row_factory = sqlite3.Row

        cursor = conn.cursor()

        # -------------------------------------------------
        # FILTER AREA
        # -------------------------------------------------

        if area:
            query = f"""
                SELECT *
                FROM {DB_TABLE}
                WHERE area = ?
                ORDER BY id DESC
                LIMIT ?
            """

            cursor.execute(
                query,
                (
                    area,
                    limit
                )
            )

        else:
            query = f"""
                SELECT *
                FROM {DB_TABLE}
                ORDER BY id DESC
                LIMIT ?
            """

            cursor.execute(
                query,
                (limit,)
            )

        rows = cursor.fetchall()

        data = [
            dict(row)
            for row in rows
        ]

        return JSONResponse({
            "success": True,
            "count": len(data),
            "data": data
        })

    except sqlite3.Error as e:

        raise HTTPException(
            status_code=500,
            detail=f"Database error: {e}"
        )

    finally:

        if conn:
            conn.close()


# =========================================================
# GET SINGLE DETECTION
# =========================================================

@app.get("/api/detections/{detection_id}")
async def get_detection(
    detection_id: int
):

    conn = None

    try:
        conn = sqlite3.connect(DB_PATH)
        conn.row_factory = sqlite3.Row

        cursor = conn.cursor()

        query = f"""
            SELECT *
            FROM {DB_TABLE}
            WHERE id = ?
            LIMIT 1
        """

        cursor.execute(
            query,
            (detection_id,)
        )

        row = cursor.fetchone()

        if not row:
            raise HTTPException(
                status_code=404,
                detail="Detection tidak ditemukan"
            )

        return JSONResponse({
            "success": True,
            "data": dict(row)
        })

    except sqlite3.Error as e:

        raise HTTPException(
            status_code=500,
            detail=f"Database error: {e}"
        )

    finally:

        if conn:
            conn.close()


# =========================================================
# SERVE ONNX
# =========================================================

@app.get("/best.onnx")
async def serve_onnx_model():

    if not MODEL_PATH.exists():
        raise HTTPException(
            status_code=404,
            detail=(
                "best.onnx tidak ditemukan "
                f"di {MODEL_PATH}"
            )
        )

    return FileResponse(
        path=str(MODEL_PATH),
        media_type="application/octet-stream",
        filename="best.onnx",
        headers={
            "Cache-Control": "public, max-age=31536000, immutable",
            "Content-Length": str(MODEL_PATH.stat().st_size)
        }
    )


# =========================================================
# STATIC
# =========================================================

if STATIC_DIR.exists():
    app.mount(
        "/static",
        StaticFiles(
            directory=str(STATIC_DIR)
        ),
        name="static"
    )


# =========================================================
# MAIN
# =========================================================

if __name__ == "__main__":

    import uvicorn

    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True
    )