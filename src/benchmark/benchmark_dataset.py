import os
import cv2
import numpy as np
from typing import  List, Tuple

class HaGRIDDataset:
    def __init__(self):
        self.target_gestures = {
            "peace": "Peace Sign",
            "palm": "Open Hand",
            "fist": "Fist"
        }

    def load_dataset(self) -> List[Tuple[np.ndarray, List[str]]]:
        """Load the dataset images from local directory."""
        dataset = []
        
        # Define directories for each gesture
        gesture_dirs = {
            "Open Hand": os.path.join("src", "benchmark", "data", "open"),
            "Peace Sign": os.path.join("src", "benchmark", "data", "peace"),
            "Fist": os.path.join("src", "benchmark", "data", "fist")
        }
        
        # Load images for each gesture
        for gesture, directory in gesture_dirs.items():
            if os.path.exists(directory):
                print(f"\nLoading {gesture} images...")
                for img_name in os.listdir(directory):
                    if img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                        img_path = os.path.join(directory, img_name)
                        image = cv2.imread(img_path)
                        if image is not None:
                            dataset.append((image, [gesture]))
                            print(f"Loaded {img_name}")
            else:
                print(f"Directory not found: {directory}")
        
        print(f"\nLoaded {len(dataset)} images total")
        return dataset 