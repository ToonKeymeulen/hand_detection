import os
import cv2
import time
from typing import Dict, List
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from hand_detector import HandSignDetector

def run_benchmark(image_dir: str) -> Dict[str, List[float]]:
    """Run benchmark on test images and return metrics."""
    detector = HandSignDetector()
    metrics = {
        'processing_times': [],
        'detected_signs': []
    }
    
    # Process each image in the test directory
    for image_file in os.listdir(image_dir):
        if not image_file.endswith(('.jpg', '.jpeg', '.png')):
            continue
            
        image_path = os.path.join(image_dir, image_file)
        image = cv2.imread(image_path)
        
        if image is None:
            print(f"Failed to load image: {image_file}")
            continue
        
        # Measure processing time
        start_time = time.time()
        signs, _ = detector.process_image(image)
        processing_time = time.time() - start_time
        
        metrics['processing_times'].append(processing_time)
        metrics['detected_signs'].append(signs)
        
        print(f"Processed {image_file}:")
        print(f"  Detected signs: {signs}")
        print(f"  Processing time: {processing_time:.3f}s")
    
    # Calculate average processing time
    if metrics['processing_times']:
        avg_time = sum(metrics['processing_times']) / len(metrics['processing_times'])
        print(f"\nAverage processing time: {avg_time:.3f}s")
    
    return metrics

if __name__ == '__main__':
    test_dir = os.path.join(os.path.dirname(__file__), 'test_images')
    
    if not os.path.exists(test_dir):
        print(f"Test directory not found: {test_dir}")
        sys.exit(1)
    
    if not os.listdir(test_dir):
        print("No test images found. Please add images to the test_images directory.")
        sys.exit(1)
    
    run_benchmark(test_dir) 