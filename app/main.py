"""
Backend Flask untuk Copper Wire Branch Detection
Streaming video dengan deteksi YOLO real-time
"""

import base64
import os
from datetime import datetime, timezone, timedelta
from pathlib import Path
from flask import Flask, request, jsonify, Response, render_template, send_file
from flask_cors import CORS
from dotenv import load_dotenv
import cv2
import numpy as np

# Asia/Jakarta timezone (UTC+7)
JAKARTA_TZ = timezone(timedelta(hours=7))

load_dotenv()

# Config
ROOT = Path(__file__).resolve().parent.parent
TEMPLATE_DIR = Path(__file__).resolve().parent / "templates"
MODEL_PATH = Path(os.getenv("MODEL_PATH", "models/best.pt"))
if not MODEL_PATH.is_absolute():
    MODEL_PATH = ROOT / MODEL_PATH

CONFIDENCE = float(os.getenv("CONFIDENCE", "0.35"))
IOU = float(os.getenv("IOU", "0.45"))
DEVICE = os.getenv("DEVICE", "cpu")
MAX_FRAME_WIDTH = int(os.getenv("MAX_FRAME_WIDTH", "640"))

# Database
DB_HOST = os.getenv("DB_HOST", "localhost")
DB_USER = os.getenv("DB_USER", "sa")
DB_PASSWORD = os.getenv("DB_PASSWORD", "")
DB_NAME = os.getenv("DB_NAME", "master")
DB_TABLE = os.getenv("DB_TABLE", "machine_detection_copper")

# Flask app
app = Flask(__name__)
app.template_folder = str(TEMPLATE_DIR)
CORS(app)

# YOLO model
_model = None


def get_model():
    global _model
    if _model is not None:
        return _model
    if not MODEL_PATH.exists():
        raise RuntimeError(f"Model tidak ditemukan: {MODEL_PATH}")
    _model = cv2.dnn.readNet(str(MODEL_PATH))
    return _model


def detect_in_image(frame):
    """Deteksi objek dalam gambar dengan YOLO."""
    from ultralytics import YOLO

    global _model
    if _model is None:
        _model = YOLO(str(MODEL_PATH))

    height, width = frame.shape[:2]

    # Resize jika terlalu besar
    if width > MAX_FRAME_WIDTH:
        scale = MAX_FRAME_WIDTH / width
        frame_resized = cv2.resize(frame, (MAX_FRAME_WIDTH, int(height * scale)))
    else:
        frame_resized = frame

    # Deteksi dengan YOLO
    result = _model.predict(frame_resized, conf=CONFIDENCE, iou=IOU, device=DEVICE, verbose=False)[0]

    detections = []
    names = result.names or {}
    boxes = result.boxes

    if boxes is not None:
        for box in boxes:
            coordinates = box.xyxy[0].cpu().numpy().astype(int).tolist()
            class_id = int(box.cls[0].item())
            confidence = float(box.conf[0].item())
            label = str(names.get(class_id, "unknown"))

            x1, y1, x2, y2 = coordinates

            detections.append({
                "class_id": class_id,
                "label": label,
                "confidence": round(confidence, 3),
                "bbox": {"x1": x1, "y1": y1, "x2": x2, "y2": y2}
            })

            # Gambar bounding box
            cv2.rectangle(frame_resized, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(
                frame_resized,
                f"{label} {confidence:.0%}",
                (x1, max(24, y1 - 8)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2,
                cv2.LINE_AA,
            )

    # Encode annotated image
    ret, buffer = cv2.imencode('.jpg', frame_resized, [cv2.IMWRITE_JPEG_QUALITY, 85])
    if not ret:
        return detections, None

    annotated_base64 = base64.b64encode(buffer).decode('ascii')
    return detections, annotated_base64


# ==================== ROUTES ====================

@app.route('/')
def index():
    """Serve frontend HTML"""
    return render_template('index.html')


@app.route('/<area>')
def index_area(area):
    """Serve frontend with area identifier"""
    if area in ['api', 'health', 'info', 'ws', 'docs', 'video_feed']:
        return "Not found", 404
    return render_template('index.html')


@app.route('/health')
def health():
    """Health check"""
    return jsonify({
        "status": "ok",
        "model": str(MODEL_PATH),
        "model_loaded": _model is not None,
        "device": DEVICE
    })


@app.route('/video_feed')
def video_feed():
    """Video streaming dengan deteksi YOLO"""
    return Response(
        generate_frames(),
        mimetype='multipart/x-mixed-replace; boundary=frame'
    )


def generate_frames():
    """Generator untuk streaming video dengan bounding box"""
    global _model
    from ultralytics import YOLO

    # Buka kamera
    cap = cv2.VideoCapture(0)

    # Set resolusi
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    # Load model
    if _model is None:
        _model = YOLO(str(MODEL_PATH))

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Resize jika perlu
            if frame.shape[1] > MAX_FRAME_WIDTH:
                scale = MAX_FRAME_WIDTH / frame.shape[1]
                frame = cv2.resize(frame, (MAX_FRAME_WIDTH, int(frame.shape[0] * scale)))

            # Deteksi dengan YOLO
            _, annotated_frame = detect_in_image(frame)

            if annotated_frame is None:
                annotated_frame = frame

            # Encode ke JPEG
            ret, buffer = cv2.imencode('.jpg', annotated_frame, [cv2.IMWRITE_JPEG_QUALITY, 70])
            if not ret:
                continue

            # Yield frame
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')

    finally:
        cap.release()


@app.route('/api/detect', methods=['POST'])
def detect_upload():
    """Deteksi dengan upload file"""
    if 'file' not in request.files:
        return jsonify({"error": "No file provided"}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No file selected"}), 400

    # Read image
    file_bytes = np.frombuffer(file.read(), dtype=np.uint8)
    frame = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    if frame is None:
        return jsonify({"error": "Cannot decode image"}), 400

    # Detect
    detections, annotated_base64 = detect_in_image(frame)

    return jsonify({
        "success": True,
        "count": len(detections),
        "detections": detections,
        "annotated_image": annotated_base64
    })


@app.route('/api/save-detection', methods=['POST'])
def save_detection():
    """Simpan hasil deteksi ke database"""
    data = request.json or {}

    # Database connection
    try:
        import pymssql
        conn = pymssql.connect(
            server=DB_HOST,
            user=DB_USER,
            password=DB_PASSWORD,
            database=DB_NAME
        )
        cursor = conn.cursor()

        # Get timestamp
        local_time = datetime.now()
        dt_ins = data.get('dt_ins') or local_time.strftime("%Y-%m-%d")
        seq_time = data.get('seq_time') or local_time.strftime("%Y-%m-%d %H:%M:%S")
        area = data.get('area') or "default"

        cursor.execute(
            f"INSERT INTO {DB_TABLE} (countc, dt_ins, seq_time, area) OUTPUT INSERTED.id VALUES (%s, %s, %s, %s)",
            (data.get('countc', 0), dt_ins, seq_time, area)
        )
        inserted_id = cursor.fetchone()[0]

        conn.commit()
        cursor.close()
        conn.close()

        return jsonify({
            "success": True,
            "id": inserted_id,
            "count": data.get('countc', 0),
            "message": f"Saved successfully at {seq_time}"
        })

    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


# ==================== MAIN ====================

if __name__ == '__main__':
    print(f"""
╔═══════════════════════════════════════════════════════════╗
║         Wire Branch Detection - Flask Server             ║
╠═══════════════════════════════════════════════════════════╣
║  Model: {MODEL_PATH}          ║
║  Device: {DEVICE}                                          ║
║  Confidence: {CONFIDENCE}                                      ║
╠═══════════════════════════════════════════════════════════╣
║  Main:     http://localhost:5000                       ║
║  Stream:   http://localhost:5000/video_feed             ║
╚═══════════════════════════════════════════════════════════╝
    """)

    # Cek environment untuk HTTPS
    USE_HTTPS = os.getenv('USE_HTTPS', 'false').lower() == 'true'

    if USE_HTTPS:
        # Generate self-signed cert jika belum ada
        if not os.path.exists('cert.pem'):
            import subprocess
            subprocess.run(['openssl', 'req', '-new', '-x509', '-keyout', 'key.pem',
                         '-out', 'cert.pem', '-days', '365', '-nodes',
                         '-subj', '/CN=localhost'], check=True)
        app.run(host='0.0.0.0', port=5000, debug=True, threaded=True,
                ssl_context=('cert.pem', 'key.pem'))
    else:
        app.run(host='0.0.0.0', port=5000, debug=True, threaded=True)
