from flask import Flask, render_template, request, jsonify
from PIL import Image, ImageDraw, ImageFont
import requests
import config
import os
import base64
from io import BytesIO

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024

os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)


class InferenceClient:
    def __init__(self, api_url, api_key):
        self.api_url = api_url
        self.api_key = api_key
    
    def infer(self, image_path, model_id):
        url = f"{self.api_url}/{model_id}"
        
        with open(image_path, "rb") as f:
            response = requests.post(
                url,
                files={"file": f},
                params={"api_key": self.api_key}
            )
        
        response.raise_for_status()
        return response.json()


def draw_predictions(image_path, predictions):
    image = Image.open(image_path)
    draw = ImageDraw.Draw(image)
    
    try:
        font = ImageFont.truetype("arial.ttf", 20)
    except:
        font = ImageFont.load_default()
    
    for pred in predictions:
        x = pred.get('x', 0)
        y = pred.get('y', 0)
        width = pred.get('width', 0)
        height = pred.get('height', 0)
        
        x1 = x - width / 2
        y1 = y - height / 2
        x2 = x + width / 2
        y2 = y + height / 2
        
        draw.rectangle([x1, y1, x2, y2], outline="red", width=3)
        
        label = f"{pred.get('class', 'N/A')} {pred.get('confidence', 0):.2%}"
        draw.text((x1, y1 - 25), label, fill="red", font=font)
    
    buffer = BytesIO()
    image.save(buffer, format='PNG')
    buffer.seek(0)
    return base64.b64encode(buffer.getvalue()).decode()


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
    
    if file:
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


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)

# Made with Bob
