import os
import json
import cv2
import numpy as np
from typing import Dict, List, Tuple
import requests
from tqdm import tqdm
import shutil

class HaGRIDDataset:
    def __init__(self, base_dir: str = "benchmark/data"):
        self.base_dir = base_dir
        self.image_dir = os.path.join(base_dir, "images")
        self.annotation_file = os.path.join(base_dir, "annotations.json")
        self.target_gestures = {
            "peace": "Peace Sign",
            "palm": "Open Hand"
        }
        
        os.makedirs(self.base_dir, exist_ok=True)
        os.makedirs(self.image_dir, exist_ok=True)
        
    def download_dataset(self):
        """Not needed for local images."""
        pass
    
    def load_dataset(self) -> List[Tuple[np.ndarray, List[str]]]:
        """Load the dataset images from local directory."""
        dataset = []
        
        # Define directories for each gesture
        gesture_dirs = {
            "Open Hand": os.path.join("src", "benchmark", "data", "open"),
            "Peace Sign": os.path.join("src", "benchmark", "data", "peace")
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