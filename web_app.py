from flask import Flask, render_template, request, jsonify
import config
import os

from services.inference_client import InferenceClient
from utils.image_utils import draw_predictions

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024

os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)


client = InferenceClient(
    api_url=config.API_URL,
    api_key=config.API_KEY
)


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
        
        try:
            result = client.infer(filepath, config.MODEL_ID)
            
            predictions = result.get('predictions', [])
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
