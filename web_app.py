from flask import Flask, render_template, request, jsonify, send_file
import config
import os
from pathlib import Path
from werkzeug.exceptions import RequestEntityTooLarge
import uuid

from services.inference_client import InferenceClient
from services.local_inference import LocalInferenceClient
from utils.image_utils import draw_predictions

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 200 * 1024 * 1024

os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

processed_videos = {}


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
    if 'files' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400
    
    files = request.files.getlist('files')
    if not files or all(f.filename == '' for f in files):
        return jsonify({'error': 'No file selected'}), 400
    
    backend = request.form.get('backend', config.INFERENCE_BACKEND)
    results = []
    
    for file in files:
        if file and file.filename:
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(filepath)
            is_video = Path(filepath).suffix.lower() in LocalInferenceClient.VIDEO_EXTENSIONS
            
            try:
                result = get_client(backend).infer(filepath, config.MODEL_ID)
                
                predictions = result.get('predictions', [])
                annotated_image = None
                video_id = None
                
                if is_video:
                    output_path = result.get('output_path')
                    if output_path and os.path.exists(output_path):
                        video_id = str(uuid.uuid4())
                        processed_videos[video_id] = output_path
                        print(f"Video saved: {file.filename} -> {output_path} (ID: {video_id})")
                    else:
                        print(f"Video output not found for {file.filename}, output_path: {output_path}")
                else:
                    annotated_image = draw_predictions(filepath, predictions)
                
                results.append({
                    'filename': file.filename,
                    'predictions': predictions,
                    'annotated_image': annotated_image,
                    'video_id': video_id,
                    'is_video': is_video,
                    'count': len(predictions)
                })
            
            except Exception as e:
                results.append({
                    'filename': file.filename,
                    'error': str(e)
                })
            
            finally:
                if os.path.exists(filepath):
                    os.remove(filepath)
    
    return jsonify({
        'success': True,
        'results': results
    })


@app.route('/video/<video_id>')
def serve_video(video_id):
    video_path = processed_videos.get(video_id)
    if not video_path or not os.path.exists(video_path):
        return jsonify({'error': 'Video not found'}), 404
    
    ext = Path(video_path).suffix.lower()
    mimetype_map = {
        '.mp4': 'video/mp4',
        '.avi': 'video/x-msvideo',
        '.mov': 'video/quicktime',
        '.mkv': 'video/x-matroska',
        '.webm': 'video/webm',
        '.flv': 'video/x-flv',
        '.wmv': 'video/x-ms-wmv'
    }
    mimetype = mimetype_map.get(ext, 'video/mp4')
    
    file_size = os.path.getsize(video_path)
    
    range_header = request.headers.get('Range')
    if not range_header:
        return send_file(
            video_path,
            mimetype=mimetype,
            as_attachment=False,
            conditional=True
        )
    
    byte_range = range_header.replace('bytes=', '').split('-')
    start = int(byte_range[0]) if byte_range[0] else 0
    end = int(byte_range[1]) if len(byte_range) > 1 and byte_range[1] else file_size - 1
    
    if start >= file_size or end >= file_size:
        return 'Requested Range Not Satisfiable', 416
    
    length = end - start + 1
    
    with open(video_path, 'rb') as f:
        f.seek(start)
        data = f.read(length)
    
    from flask import Response
    response = Response(
        data,
        206,
        mimetype=mimetype,
        direct_passthrough=True
    )
    response.headers.add('Content-Range', f'bytes {start}-{end}/{file_size}')
    response.headers.add('Accept-Ranges', 'bytes')
    response.headers.add('Content-Length', str(length))
    response.headers.add('Cache-Control', 'no-cache')
    
    return response


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
