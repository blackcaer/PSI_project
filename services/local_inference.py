import os
from pathlib import Path

from ultralytics import YOLO


class LocalInferenceClient:
    VIDEO_EXTENSIONS = {'.mp4', '.mov', '.avi', '.mkv', '.webm', '.flv', '.wmv'}

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
        is_video = Path(image_path).suffix.lower() in {'.mp4', '.mov', '.avi', '.mkv', '.webm', '.flv', '.wmv'}
        output_path = None

        if is_video:
            output_root = Path(__file__).resolve().parents[1] / 'uploads'
            output_root.mkdir(parents=True, exist_ok=True)
            output_name = Path(image_path).stem + '_annotated'
            results = self.model.predict(
                source=image_path,
                conf=self.confidence,
                imgsz=self.image_size,
                save=True,
                project=str(output_root),
                name=output_name,
                exist_ok=True
            )
            candidate_path = output_root / output_name / Path(image_path).name
            if candidate_path.exists():
                output_path = str(candidate_path)
            else:
                outputs = list((output_root / output_name).glob('*'))
                if outputs:
                    output_path = str(outputs[0])
        else:
            results = self.model.predict(source=image_path, conf=self.confidence, imgsz=self.image_size)

        predictions = []
        for result in results:
            for box in result.boxes:
                xywh = box.xywh[0].tolist()
                cls = int(box.cls[0].item()) if hasattr(box.cls[0], "item") else int(box.cls[0])
                conf = float(box.conf[0].item()) if hasattr(box.conf[0], "item") else float(box.conf[0])
                prediction = {
                    'x': float(xywh[0]),
                    'y': float(xywh[1]),
                    'width': float(xywh[2]),
                    'height': float(xywh[3]),
                    'confidence': conf,
                    'class': result.names.get(cls, str(cls))
                }
                if is_video:
                    prediction['frame'] = 0
                predictions.append(prediction)

        return {
            'predictions': predictions,
            'type': 'video' if is_video else 'image',
            'output_path': output_path
        }
