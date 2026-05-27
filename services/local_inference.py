import os
from pathlib import Path
import cv2

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

    def _convert_avi_to_mp4(self, avi_path):
        mp4_path = str(Path(avi_path).with_suffix('.mp4'))
        cap = cv2.VideoCapture(avi_path)
        
        fourcc = cv2.VideoWriter_fourcc(*'avc1')  # type: ignore
        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps == 0:
            fps = 30.0  # Default to 30 fps if not detected
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        width = width if width % 2 == 0 else width - 1
        height = height if height % 2 == 0 else height - 1
        
        out = cv2.VideoWriter(mp4_path, fourcc, fps, (width, height))
        
        if not out.isOpened():
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # type: ignore
            out = cv2.VideoWriter(mp4_path, fourcc, fps, (width, height))
        
        frame_count = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if frame.shape[1] != width or frame.shape[0] != height:
                frame = cv2.resize(frame, (width, height))
            out.write(frame)
            frame_count += 1
        
        cap.release()
        out.release()
        
        print(f"Converted {frame_count} frames from AVI to MP4: {mp4_path}")
        
        if os.path.exists(mp4_path) and os.path.getsize(mp4_path) > 0:
            if os.path.exists(avi_path):
                os.remove(avi_path)
            return mp4_path
        else:
            print(f"Warning: MP4 conversion failed, keeping AVI: {avi_path}")
            return avi_path

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
            output_dir = output_root / output_name
            video_files = list(output_dir.glob('*.avi')) + list(output_dir.glob('*.mp4'))
            if video_files:
                avi_path = str(video_files[0])
                if avi_path.endswith('.avi'):
                    output_path = self._convert_avi_to_mp4(avi_path)
                else:
                    output_path = avi_path
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
