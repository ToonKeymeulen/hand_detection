import os
import time
import json
from typing import Dict, List, Any
import numpy as np
from tqdm import tqdm
import cv2

from ..detector.hand_detector import HandSignDetector
from .benchmark_dataset import HaGRIDDataset

class HandSignBenchmark:
    def __init__(self, output_dir: str = "benchmark/results"):
        self.detector = HandSignDetector()
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
    def run_benchmark(self, dataset: List[tuple]) -> Dict[str, Any]:
        """Run benchmark on the dataset and return metrics."""
        results = {
            'total_images': len(dataset),
            'processing_times': [],
            'accuracy': {},
            'confusion_matrix': {},
            'errors': []
        }
        
        # Initialize confusion matrix
        for true_label in ["Peace Sign", "Open Hand"]:
            results['confusion_matrix'][true_label] = {
                "Peace Sign": 0,
                "Open Hand": 0,
                "Other/None": 0
            }
        
        # Create directory for annotated images
        annotated_dir = os.path.join(self.output_dir, "annotated_images")
        os.makedirs(annotated_dir, exist_ok=True)
        
        print("Running benchmark...")
        for idx, (image, true_labels) in enumerate(tqdm(dataset)):
            try:
                # Time the detection
                start_time = time.time()
                annotated_frame, detected_signs = self.detector.process_frame(image)
                processing_time = time.time() - start_time
                results['processing_times'].append(processing_time)
                
                # Update confusion matrix
                for true_label in true_labels:
                    if detected_signs:
                        for detected_sign in detected_signs:
                            if detected_sign in ["Peace Sign", "Open Hand"]:
                                results['confusion_matrix'][true_label][detected_sign] += 1
                    else:
                        results['confusion_matrix'][true_label]["Other/None"] += 1
                
                # Save annotated image if it's a peace sign
                if "Peace Sign" in true_labels:
                    # Add text showing what was detected
                    detection_text = f"Detected: {', '.join(detected_signs) if detected_signs else 'None'}"
                    cv2.putText(annotated_frame, detection_text, (10, 30),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                    
                    # Add finger states if landmarks were detected
                    if hasattr(self.detector, 'last_finger_states') and self.detector.last_finger_states is not None:
                        states = ['Thumb', 'Index', 'Middle', 'Ring', 'Pinky']
                        for i, (finger, state) in enumerate(zip(states, self.detector.last_finger_states)):
                            state_text = f"{finger}: {'Up' if state else 'Down'}"
                            cv2.putText(annotated_frame, state_text, (10, 60 + i * 25),
                                      cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
                    
                    # Save the annotated image
                    output_path = os.path.join(annotated_dir, f"peace_sign_{idx}.jpg")
                    cv2.imwrite(output_path, annotated_frame)
                        
            except Exception as e:
                results['errors'].append(str(e))
        
        # Calculate metrics
        results['average_time'] = np.mean(results['processing_times'])
        results['std_time'] = np.std(results['processing_times'])
        
        # Calculate accuracy for each gesture
        total_correct = 0
        total_samples = 0
        for true_label, predictions in results['confusion_matrix'].items():
            correct = predictions[true_label]
            total = sum(predictions.values())
            if total > 0:
                accuracy = correct / total
                results['accuracy'][true_label] = accuracy
                total_correct += correct
                total_samples += total
        
        # Calculate overall accuracy
        results['overall_accuracy'] = total_correct / total_samples if total_samples > 0 else 0
        
        # Save results
        output_file = os.path.join(self.output_dir, 'benchmark_results.json')
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=4)
        
        return results

def main():
    try:
        # Initialize dataset
        dataset = HaGRIDDataset()
        
        # Load dataset
        print("Loading dataset...")
        images_and_labels = dataset.load_dataset()
        
        if not images_and_labels:
            print("Error: No valid images loaded from dataset")
            return
        
        print(f"Loaded {len(images_and_labels)} images for benchmarking")
        
        # Run benchmark
        benchmark = HandSignBenchmark()
        results = benchmark.run_benchmark(images_and_labels)
        
        # Print summary
        print("\nBenchmark Results:")
        print(f"Total images: {results['total_images']}")
        print(f"Average processing time: {results['average_time']:.3f}s ± {results['std_time']:.3f}s")
        print("\nAccuracy by gesture:")
        for gesture, acc in results['accuracy'].items():
            print(f"{gesture}: {acc:.2%}")
        print(f"\nOverall accuracy: {results['overall_accuracy']:.2%}")
        
        # Print confusion matrix
        print("\nConfusion Matrix:")
        for true_label, predictions in results['confusion_matrix'].items():
            print(f"\nTrue label '{true_label}':")
            for pred_label, count in predictions.items():
                print(f"  Predicted as '{pred_label}': {count}")
        
        if results['errors']:
            print(f"\nEncountered {len(results['errors'])} errors during benchmarking")
            
    except Exception as e:
        print(f"Error running benchmark: {e}")

if __name__ == "__main__":
    main() 