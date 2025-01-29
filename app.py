# Import required libraries
import cv2                  # OpenCV library for image processing and computer vision
import numpy as np         # NumPy for numerical operations on arrays
from flask import Flask, request, jsonify  # Flask for creating web server endpoints
import base64             # For encoding/decoding base64 image data
from hand_detector import HandSignDetector  # Custom class for detecting hand signs
import threading          # For running multiple processes concurrently
from PIL import Image, ImageTk  # PIL for image processing, ImageTk for displaying images in tkinter

import time              # For adding delays and timing operations
import tkinter as tk     # GUI framework for creating the window
from queue import Queue  # Thread-safe queue for sharing frames between threads
import socket           # For network socket operations
from src.gui.video_window import VideoWindow
from src.api.routes import api

# Initialize Flask app and global objects
app = Flask(__name__)    # Create Flask application instance
detector = HandSignDetector()  # Initialize hand sign detector
frame_queue = Queue(maxsize=1)  # Create queue to store latest frame (only keeps one frame)
should_exit = threading.Event()  # Event flag to signal when program should exit

def find_free_port(start_port=5001, max_port=5010):
    """Find available port for Flask server"""
    for port in range(start_port, max_port + 1):
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)  # Create test socket
        try:
            sock.bind(('0.0.0.0', port))  # Try to bind to port
            sock.close()  # Close test socket
            return port  # Return available port
        except OSError:
            continue  # Try next port if current is in use
    return None

def run_webcam():
    """Main webcam capture and processing loop"""
    try:
        # Set up webcam capture
        cap = cv2.VideoCapture(0)  # Initialize webcam (device 0)
        if not cap.isOpened():
            print("Error: Could not open webcam")
            return

        print("\nWebcam initialized successfully!")
        print("Processing frames... The window should appear shortly.")
        
        # Main processing loop
        while not should_exit.is_set():
            ret, frame = cap.read()  # Read frame from webcam
            if not ret:
                print("Error: Could not read frame")
                break

            try:
                # Process current frame
                annotated_frame, signs = detector.process_frame(frame)  # Detect signs
                
                # Update frame queue, dropping old frame if necessary
                if frame_queue.full():
                    frame_queue.get_nowait()  # Remove old frame
                frame_queue.put_nowait((annotated_frame, signs))  # Add new frame
                
            except Exception as e:
                print(f"Error processing frame: {e}")
                continue
            
            time.sleep(0.03)  # Add small delay to limit frame rate
        
    except Exception as e:
        print(f"Error in webcam thread: {e}")
    
    finally:
        # Clean up resources
        print("Cleaning up resources...")
        if 'cap' in locals():
            cap.release()  # Release webcam
        print("Cleanup complete!")

def run_flask():
    """Initialize and run Flask server"""
    app.register_blueprint(api)
    port = find_free_port()  # Find available port
    if port is None:
        print("Error: Could not find a free port for the Flask server")
        return
    
    print(f"\nStarting Flask server on port {port}")
    app.run(host='0.0.0.0', port=port, use_reloader=False)  # Start Flask server

# Main program entry point
if __name__ == '__main__':
    print("Starting Hand Sign Detection...")
    print("Initializing webcam and Flask server...")
    
    # Initialize and start webcam thread
    webcam_thread = threading.Thread(target=run_webcam)  # Create webcam thread
    webcam_thread.daemon = True  # Set as daemon so it exits when main program exits
    webcam_thread.start()  # Start webcam thread
    
    # Initialize and start Flask server thread
    flask_thread = threading.Thread(target=run_flask)  # Create Flask thread
    flask_thread.daemon = True  # Set as daemon so it exits when main program exits
    flask_thread.start()  # Start Flask thread
    
    # Start GUI in main thread
    window = VideoWindow(frame_queue, should_exit)
    window.run() 