# Wire Branch Detection API - Laravel Integration

Backend API untuk deteksi kawat tembaga bercabang. Dapat diakses dari Laravel via HTTP request.

## Cara Menjalankan

```bash
cd /home/nbit01/Desktop/project/yolov8-wire
source .venv/bin/activate
pip install -r requirements.txt
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

## API Endpoints

### 1. Health Check
```
GET /health
```
Response:
```json
{
  "status": "ok",
  "model": "models/best.pt",
  "model_loaded": true,
  "device": "cpu",
  "confidence": 0.35,
  "iou": 0.45
}
```

### 2. Deteksi dengan Base64 (Recommended untuk Laravel)
```
POST /api/detect/base64
Content-Type: application/json
```

Body:
```json
{
  "image": "base64_encoded_image_without_prefix",
  "include_annotated": true
}
```

Response:
```json
{
  "success": true,
  "count": 3,
  "detections": [
    {
      "class_id": 0,
      "label": "split_wire",
      "confidence": 0.85,
      "bbox": {
        "x1": 100,
        "y1": 200,
        "x2": 150,
        "y2": 250,
        "width": 50,
        "height": 50
      }
    }
  ],
  "annotated_image": "base64_encoded_jpeg_image"
}
```

### 3. Deteksi dengan File Upload
```
POST /api/detect
Content-Type: multipart/form-data
```

Field: `file` (image file)

### 4. Deteksi Batch (Banyak Gambar Sekaligus)
```
POST /api/detect/batch
Content-Type: application/json
```

Body:
```json
{
  "images": ["base64_1", "base64_2", "base64_3"],
  "include_annotated": true
}
```

## Contoh Kode Laravel

### Install Guzzle HTTP Client
```bash
composer require guzzlehttp/guzzle
```

### Service Class

```php
<?php

namespace App\Services;

use Illuminate\Support\Facades\Http;
use Illuminate\Support\Facades\Log;

class WireDetectionService
{
    protected string $apiUrl;
    protected float $confidenceThreshold;

    public function __construct()
    {
        $this->apiUrl = env('WIRE_DETECTION_API_URL', 'http://localhost:8000');
        $this->confidenceThreshold = (float) env('WIRE_DETECTION_CONFIDENCE', 0.35);
    }

    /**
     * Deteksi kawat dari gambar base64
     */
    public function detectFromBase64(string $base64Image, bool $includeAnnotated = true): array
    {
        try {
            // Hapus prefix data:image jika ada
            $imageData = $base64Image;
            if (str_contains($imageData, ',')) {
                $imageData = explode(',', $imageData)[1];
            }

            $response = Http::timeout(30)->post("{$this->apiUrl}/api/detect/base64", [
                'image' => $imageData,
                'include_annotated' => $includeAnnotated,
            ]);

            if ($response->successful()) {
                return [
                    'success' => true,
                    'count' => $response->json('count'),
                    'detections' => $response->json('detections'),
                    'annotated_image' => $response->json('annotated_image'),
                ];
            }

            return [
                'success' => false,
                'error' => $response->json('detail', 'Unknown error'),
            ];
        } catch (\Exception $e) {
            Log::error('Wire detection error: ' . $e->getMessage());
            return [
                'success' => false,
                'error' => $e->getMessage(),
            ];
        }
    }

    /**
     * Deteksi kawat dari file upload
     */
    public function detectFromFile($file): array
    {
        try {
            $response = Http::timeout(30)
                ->attach('file', file_get_contents($file->getRealPath()), $file->getClientOriginalName())
                ->post("{$this->apiUrl}/api/detect");

            if ($response->successful()) {
                return [
                    'success' => true,
                    'count' => $response->json('count'),
                    'detections' => $response->json('detections'),
                    'annotated_image' => $response->json('annotated_image'),
                ];
            }

            return [
                'success' => false,
                'error' => $response->json('detail', 'Unknown error'),
            ];
        } catch (\Exception $e) {
            Log::error('Wire detection error: ' . $e->getMessage());
            return [
                'success' => false,
                'error' => $e->getMessage(),
            ];
        }
    }

    /**
     * Deteksi batch dari array base64
     */
    public function detectBatch(array $base64Images, bool $includeAnnotated = true): array
    {
        try {
            // Bersihkan prefix dari semua gambar
            $images = array_map(function ($img) {
                if (str_contains($img, ',')) {
                    return explode(',', $img)[1];
                }
                return $img;
            }, $base64Images);

            $response = Http::timeout(120)->post("{$this->apiUrl}/api/detect/batch", [
                'images' => $images,
                'include_annotated' => $includeAnnotated,
            ]);

            if ($response->successful()) {
                return [
                    'success' => true,
                    'total' => $response->json('total'),
                    'processed' => $response->json('processed'),
                    'results' => $response->json('results'),
                ];
            }

            return [
                'success' => false,
                'error' => $response->json('detail', 'Unknown error'),
            ];
        } catch (\Exception $e) {
            Log::error('Wire batch detection error: ' . $e->getMessage());
            return [
                'success' => false,
                'error' => $e->getMessage(),
            ];
        }
    }
}
```

### Controller Example

```php
<?php

namespace App\Http\Controllers;

use App\Services\WireDetectionService;
use Illuminate\Http\Request;
use Illuminate\Http\JsonResponse;

class WireDetectionController extends Controller
{
    protected WireDetectionService $detectionService;

    public function __construct(WireDetectionService $detectionService)
    {
        $this->detectionService = $detectionService;
    }

    /**
     * Deteksi dari base64 image
     */
    public function detect(Request $request): JsonResponse
    {
        $request->validate([
            'image' => 'required|string',
        ]);

        $result = $this->detectionService->detectFromBase64(
            $request->input('image'),
            $request->boolean('include_annotated', true)
        );

        return response()->json($result);
    }

    /**
     * Deteksi dari file upload
     */
    public function detectUpload(Request $request): JsonResponse
    {
        $request->validate([
            'image' => 'required|image|mimes:jpeg,png,jpg|max:5120',
        ]);

        $result = $this->detectionService->detectFromFile($request->file('image'));

        return response()->json($result);
    }

    /**
     * Deteksi batch
     */
    public function detectBatch(Request $request): JsonResponse
    {
        $request->validate([
            'images' => 'required|array|min:1',
            'images.*' => 'required|string',
        ]);

        $result = $this->detectionService->detectBatch(
            $request->input('images'),
            $request->boolean('include_annotated', true)
        );

        return response()->json($result);
    }
}
```

### Routes (routes/api.php)

```php
use App\Http\Controllers\WireDetectionController;

Route::prefix('wire')->group(function () {
    Route::post('/detect', [WireDetectionController::class, 'detect']);
    Route::post('/detect/upload', [WireDetectionController::class, 'detectUpload']);
    Route::post('/detect/batch', [WireDetectionController::class, 'detectBatch']);
});
```

### Environment (.env)

```env
WIRE_DETECTION_API_URL=http://localhost:8000
WIRE_DETECTION_CONFIDENCE=0.35
```

## Konfigurasi

| Variable | Default | Deskripsi |
|----------|---------|-----------|
| `CONFIDENCE` | 0.35 | Threshold deteksi (0-1) |
| `IOU` | 0.45 | IoU threshold untuk NMS |
| `DEVICE` | cpu | cpu atau 0 (GPU) |
| `MAX_FRAME_WIDTH` | 1280 | Ukuran max gambar |

## Response Format

### Success Response
```json
{
  "success": true,
  "count": 3,
  "detections": [
    {
      "class_id": 0,
      "label": "split_wire",
      "confidence": 0.85,
      "bbox": {
        "x1": 100,
        "y1": 200,
        "x2": 150,
        "y2": 250,
        "width": 50,
        "height": 50
      }
    }
  ],
  "annotated_image": "base64_jpeg..."
}
```

### Error Response
```json
{
  "success": false,
  "error": "Pesan error"
}
```
