import os
from pathlib import Path

from ultralytics import YOLO


class LocalInferenceClient:
    def __init__(self, model_path, confidence=0.4, image_size=640):
        candidate = Path(model_path)
        if not candidate.is_absolute():
            candidate = Path(__file__).resolve().parents[1] / model_path
        abs_path = str(candidate)
        if not os.path.exists(abs_path):
            raise FileNotFoundError(f"Local model not found: {abs_path}")
        self.model = YOLO(abs_path)
        self.confidence = confidence
        self.image_size = image_size

    def infer(self, image_path, model_id=None):
        results = self.model.predict(source=image_path, conf=self.confidence, imgsz=self.image_size)
        predictions = []

        for result in results:
            for box in result.boxes:
                xywh = box.xywh[0].tolist()
                cls = int(box.cls[0].item()) if hasattr(box.cls[0], "item") else int(box.cls[0])
                conf = float(box.conf[0].item()) if hasattr(box.conf[0], "item") else float(box.conf[0])
                predictions.append({
                    'x': float(xywh[0]),
                    'y': float(xywh[1]),
                    'width': float(xywh[2]),
                    'height': float(xywh[3]),
                    'confidence': conf,
                    'class': result.names.get(cls, str(cls))
                })

        return {'predictions': predictions}
