from flask import Flask, render_template, request, jsonify
import config
import os
from pathlib import Path
from werkzeug.exceptions import RequestEntityTooLarge

from services.inference_client import InferenceClient
from services.local_inference import LocalInferenceClient
from utils.image_utils import draw_predictions

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 200 * 1024 * 1024

os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)


@app.errorhandler(RequestEntityTooLarge)
def handle_file_too_large(_error):
    return jsonify({'error': 'File too large. Maximum upload size is 200 MB.'}), 413

local_client = None
remote_client = InferenceClient(
    api_url=config.API_URL,
    api_key=config.API_KEY
)


def get_client(backend):
    global local_client
    if backend == 'remote':
        return remote_client

    if local_client is None:
        local_client = LocalInferenceClient(
            model_path=config.LOCAL_MODEL_PATH,
            confidence=config.YOLO_CONFIDENCE,
            image_size=config.YOLO_IMAGE_SIZE
        )
    return local_client


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/upload', methods=['POST'])
def upload():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    if file and file.filename:
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(filepath)
        backend = request.form.get('backend', config.INFERENCE_BACKEND)
        is_video = Path(filepath).suffix.lower() in LocalInferenceClient.VIDEO_EXTENSIONS
        
        try:
            result = get_client(backend).infer(filepath, config.MODEL_ID)
            
            predictions = result.get('predictions', [])
            annotated_image = None
            if not is_video:
                annotated_image = draw_predictions(filepath, predictions)
            
            return jsonify({
                'success': True,
                'predictions': predictions,
                'annotated_image': annotated_image,
                'count': len(predictions)
            })
        
        except Exception as e:
            return jsonify({'error': str(e)}), 500
        
        finally:
            if os.path.exists(filepath):
                os.remove(filepath)
    
    return jsonify({'error': 'No file provided'}), 400


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
