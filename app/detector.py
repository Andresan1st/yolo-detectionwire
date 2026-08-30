"""
YOLO/ONNX Detector for copper detection
"""
import cv2
import numpy as np
import onnxruntime as ort
import os
from dotenv import load_dotenv
from dataclasses import dataclass
from typing import List, Tuple, Optional

load_dotenv()

@dataclass
class Detection:
    """Detection result"""
    class_id: int
    class_name: str
    confidence: float
    bbox: Tuple[int, int, int, int]  # x1, y1, x2, y2

    @property
    def area(self) -> int:
        """Calculate bounding box area"""
        x1, y1, x2, y2 = self.bbox
        return (x2 - x1) * (y2 - y1)

class CopperDetector:
    """ONNX-based copper detection model"""

    def __init__(self, model_path: str = None, confidence: float = 0.55, iou: float = 0.45):
        """Initialize detector

        Args:
            model_path: Path to ONNX model file
            confidence: Confidence threshold for detections
            iou: IoU threshold for NMS
        """
        self.model_path = model_path or os.getenv('MODEL_PATH', 'app/best.onnx')
        self.confidence = confidence
        self.iou = iou

        # Try multiple possible model paths
        possible_paths = [
            self.model_path,
            os.path.join(os.path.dirname(__file__), 'best.onnx'),
            os.path.join(os.path.dirname(os.path.dirname(__file__)), 'app', 'best.onnx'),
            os.path.join(os.path.dirname(os.path.dirname(__file__)), 'best.onnx'),
        ]

        self.session = None
        for path in possible_paths:
            if os.path.exists(path):
                try:
                    self.session = ort.InferenceSession(path, providers=['CPUExecutionProvider'])
                    print(f"Model loaded from: {path}")
                    break
                except Exception as e:
                    print(f"Failed to load model from {path}: {e}")

        if self.session is None:
            print("Warning: No model file found. Detection will return empty results.")

        # Get model info
        if self.session:
            self.input_name = self.session.get_inputs()[0].name
            self.output_names = [output.name for output in self.session.get_outputs()]

    def preprocess(self, image: np.ndarray) -> np.ndarray:
        """Preprocess image for inference"""
        # Model expects 1280x1280 input
        input_width = 1280
        input_height = 1280
        self.original_shape = image.shape[:2]

        # Resize to model input size
        resized = cv2.resize(image, (input_width, input_height))

        # Normalize
        normalized = resized.astype(np.float32) / 255.0

        # Transpose to NCHW format
        transposed = np.transpose(normalized, (2, 0, 1))

        # Add batch dimension
        batched = np.expand_dims(transposed, axis=0)

        return batched

    def postprocess(self, outputs, original_shape: Tuple[int, int]) -> List[Detection]:
        """Postprocess model outputs to Detection objects"""
        detections = []

        if self.session is None:
            return detections

        # Parse YOLO output (format varies by model)
        # Assuming output shape is (1, num_predictions, 84) where 84 = 4(box) + 80(classes)
        try:
            predictions = outputs[0] if isinstance(outputs, list) else outputs

            if len(predictions.shape) == 3:
                predictions = predictions[0]  # Remove batch dimension

            # Transpose to (num_predictions, 84)
            if predictions.shape[0] < predictions.shape[1]:
                predictions = predictions.T

            # Filter by confidence
            conf_threshold = self.confidence

            for pred in predictions:
                # Get box coordinates (first 4 values)
                box = pred[:4]

                # Get class scores (remaining values)
                class_scores = pred[4:]

                # Get max confidence and class
                max_score = np.max(class_scores)
                if max_score < conf_threshold:
                    continue

                class_id = np.argmax(class_scores)

                # Convert box from center format to corner format
                cx, cy, w, h = box
                x1 = int((cx - w / 2) * original_shape[1] / 1280)
                y1 = int((cy - h / 2) * original_shape[0] / 1280)
                x2 = int((cx + w / 2) * original_shape[1] / 1280)
                y2 = int((cy + h / 2) * original_shape[0] / 1280)

                detections.append(Detection(
                    class_id=int(class_id),
                    class_name=f"copper_{class_id}",  # Placeholder name
                    confidence=float(max_score),
                    bbox=(x1, y1, x2, y2)
                ))

        except Exception as e:
            print(f"Postprocess error: {e}")

        return detections

    def detect(self, image: np.ndarray) -> List[Detection]:
        """Detect objects in image

        Args:
            image: Input image in BGR format (numpy array)

        Returns:
            List of Detection objects
        """
        if self.session is None:
            return []

        # Preprocess
        input_tensor = self.preprocess(image)
        original_shape = image.shape[:2]

        # Run inference
        outputs = self.session.run(self.output_names, {self.input_name: input_tensor})

        # Postprocess
        detections = self.postprocess(outputs, original_shape)

        return detections

    def draw_detections(self, image: np.ndarray, detections: List[Detection]) -> np.ndarray:
        """Draw bounding boxes on image"""
        output = image.copy()

        for det in detections:
            x1, y1, x2, y2 = det.bbox

            # Draw box
            cv2.rectangle(output, (x1, y1), (x2, y2), (0, 255, 0), 2)

            # Draw label
            label = f"{det.class_name}: {det.confidence:.2f}"
            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]

            # Label background
            cv2.rectangle(output, (x1, y1 - label_size[1] - 10),
                         (x1 + label_size[0], y1), (0, 255, 0), -1)

            # Label text
            cv2.putText(output, label, (x1, y1 - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)

        return output

# Global detector instance
_detector = None

def get_detector() -> CopperDetector:
    """Get or create global detector instance"""
    global _detector
    if _detector is None:
        _detector = CopperDetector()
    return _detector
