import cv2
import numpy as np
from PIL import Image
import io
import base64

def process_base64_image(base64_string: str) -> np.ndarray:
    """Convert and process a base64 image string to a numpy array."""
    img_data = base64.b64decode(base64_string)
    img = Image.open(io.BytesIO(img_data))
    
    if img.mode != 'RGB':
        img = img.convert('RGB')
    
    image = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
    
    # Resize if needed
    max_dimension = 1280
    min_dimension = 320
    height, width = image.shape[:2]
    
    if max(height, width) > max_dimension:
        scale = max_dimension / max(height, width)
        image = cv2.resize(image, None, fx=scale, fy=scale)
    elif min(height, width) < min_dimension:
        scale = min_dimension / min(height, width)
        image = cv2.resize(image, None, fx=scale, fy=scale)
    
    return image

def encode_image_to_base64(image: np.ndarray) -> str:
    """Convert an OpenCV image to base64 string."""
    _, buffer = cv2.imencode('.jpg', image)
    return base64.b64encode(buffer).decode('utf-8') 