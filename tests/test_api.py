import os
import base64
import requests
import json
from PIL import Image
import io
from typing import Dict, Any

def image_to_base64(image_path: str) -> str:
    """Convert an image file to base64 string."""
    with Image.open(image_path) as img:
        # Convert to RGB if necessary
        if img.mode != 'RGB':
            img = img.convert('RGB')
        
        # Save to bytes
        img_byte_arr = io.BytesIO()
        img.save(img_byte_arr, format='JPEG')
        img_byte_arr = img_byte_arr.getvalue()
        
        # Convert to base64
        return base64.b64encode(img_byte_arr).decode('utf-8')

def test_predict_endpoint(image_path: str, api_url: str = "http://localhost:5001") -> Dict[str, Any]:
    """Test the predict endpoint with a single image."""
    try:
        # Convert image to base64
        base64_image = image_to_base64(image_path)
        
        # Prepare the request
        headers = {'Content-Type': 'application/json'}
        data = {'image': base64_image}
        
        # Send POST request to the endpoint
        response = requests.post(f"{api_url}/predict", 
                               headers=headers,
                               data=json.dumps(data))
        
        # Check if request was successful
        response.raise_for_status()
        
        return {
            'status': 'success',
            'image': os.path.basename(image_path),
            'response': response.json()
        }
    
    except requests.exceptions.RequestException as e:
        return {
            'status': 'error',
            'image': os.path.basename(image_path),
            'error': str(e)
        }
    except Exception as e:
        return {
            'status': 'error',
            'image': os.path.basename(image_path),
            'error': f"Unexpected error: {str(e)}"
        }

def test_all_images(image_dir: str, api_url: str = "http://localhost:5001") -> None:
    """Test the predict endpoint with all images in a directory."""
    # Get all image files
    image_files = [f for f in os.listdir(image_dir) 
                  if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    
    if not image_files:
        print(f"No image files found in {image_dir}")
        return
    
    # Create output directory for annotated images
    output_dir = os.path.join(os.path.dirname(image_dir), 'test_output')
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Testing {len(image_files)} images...")
    print(f"Annotated images will be saved to: {output_dir}")
    print("-" * 50)
    
    # Test each image
    for image_file in image_files:
        image_path = os.path.join(image_dir, image_file)
        result = test_predict_endpoint(image_path, api_url)
        
        # Print results
        print(f"\nTesting image: {result['image']}")
        if result['status'] == 'success':
            response = result['response']
            print(f"Landmarks detected: {response.get('landmarks_detected', False)}")
            print(f"Number of hands detected: {response.get('num_hands_detected', 0)}")
            print(f"Detected signs: {response.get('detected_signs', [])}")
            
            # Save annotated image if available
            if 'annotated_image' in response:
                output_path = os.path.join(output_dir, f"annotated_{image_file}")
                img_data = base64.b64decode(response['annotated_image'])
                with open(output_path, 'wb') as f:
                    f.write(img_data)
                print(f"Saved annotated image to: {output_path}")
            
            # Print landmark information if available
            if response.get('hand_landmarks'):
                for hand_idx, landmarks in enumerate(response['hand_landmarks']):
                    print(f"\nHand {hand_idx + 1} Landmarks:")
                    # Print a few key landmarks for verification
                    key_points = {
                        'Wrist': 0,
                        'Thumb tip': 4,
                        'Index tip': 8,
                        'Middle tip': 12,
                        'Ring tip': 16,
                        'Pinky tip': 20
                    }
                    for name, idx in key_points.items():
                        if idx < len(landmarks):
                            point = landmarks[idx]
                            print(f"  {name}: x={point['x']:.3f}, y={point['y']:.3f}, z={point['z']:.3f}")
            else:
                print("No hand landmarks detected in the image")
        else:
            print(f"Error: {result['error']}")
    
    print("\nTesting complete!")

if __name__ == "__main__":
    # Directory containing test images
    test_dir = os.path.join(os.path.dirname(__file__), 'test_images')
    
    if not os.path.exists(test_dir):
        print(f"Test directory not found: {test_dir}")
        exit(1)
    
    # Add requests to requirements if not already present
    try:
        import requests
    except ImportError:
        print("Installing required package: requests")
        os.system("pip install requests")
    
    # Run tests
    test_all_images(test_dir) 