# Traffic Signs Detection App

Simple Python application for traffic sign detection using Roboflow model.

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

```bash
python app.py
```

## Usage

1. Click "Select Image" to choose an image file
2. Click "Run Inference" to detect traffic signs
3. View results in the text area below