from PIL import Image, ImageDraw, ImageFont
import base64
from io import BytesIO


def draw_predictions(image_path, predictions):
    """Draw bounding boxes and labels on image based on predictions."""
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
