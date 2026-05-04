# Traffic Signs Detection App

Web application for traffic sign detection using Roboflow model with visual bounding boxes.

## Setup

1. Create virtual environment:
```bash
python -m venv .venv
```

2. Activate virtual environment:
```bash
# Windows
.venv\Scripts\activate

# Linux/Mac
source .venv/bin/activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Configure API credentials:
```bash
# Copy example config
copy config.example.py config.py

# Edit config.py with your credentials
```

## Run

### Web Application (Recommended)
```bash
python web_app.py
```
Then open browser at: http://localhost:5000

### Desktop Application (Legacy)
```bash
python app.py
```

## Features

- Web-based interface
- Image upload and preview
- Real-time inference
- Visual bounding boxes with labels
- Confidence scores
- Detailed detection results

## Usage

1. Open web browser at http://localhost:5000
2. Click "Choose Image" to select an image
3. Click "Run Detection" to analyze
4. View annotated image with bounding boxes and detection details