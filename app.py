import cv2
import numpy as np
from flask import Flask, request, jsonify
import base64
from hand_detector import HandSignDetector
import threading
from PIL import Image, ImageTk
import io
import sys
import time
import tkinter as tk
from queue import Queue
import threading
import socket

app = Flask(__name__)
detector = HandSignDetector()
frame_queue = Queue(maxsize=1)
should_exit = threading.Event()

class VideoWindow:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("Hand Sign Detection")
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)
        
        # Create canvas for video display
        self.canvas = tk.Canvas(self.root, width=640, height=480)
        self.canvas.pack()
        
        # Create label for detected signs
        self.sign_label = tk.Label(self.root, text="No signs detected", font=("Arial", 14))
        self.sign_label.pack()
        
        self.update_frame()
    
    def on_closing(self):
        should_exit.set()
        self.root.quit()
        self.root.destroy()
    
    def update_frame(self):
        try:
            if not frame_queue.empty():
                frame, signs = frame_queue.get_nowait()
                
                # Convert OpenCV BGR to RGB
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                # Convert to PIL Image and then to PhotoImage
                image = Image.fromarray(frame_rgb)
                image = image.resize((640, 480), Image.Resampling.LANCZOS)
                self.photo = ImageTk.PhotoImage(image=image)
                
                # Update canvas
                self.canvas.create_image(0, 0, image=self.photo, anchor=tk.NW)
                
                # Update sign label
                if signs:
                    self.sign_label.config(text=f"Detected signs: {', '.join(signs)}")
                else:
                    self.sign_label.config(text="No signs detected")
        
        except Exception as e:
            print(f"Error updating frame: {e}")
        
        if not should_exit.is_set():
            self.root.after(30, self.update_frame)

def process_base64_image(base64_string):
    """Convert base64 image to numpy array."""
    img_data = base64.b64decode(base64_string)
    img = Image.open(io.BytesIO(img_data))
    return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)

@app.route('/predict', methods=['POST'])
def predict():
    """Endpoint for processing images sent as base64 strings."""
    if not request.json or 'image' not in request.json:
        return jsonify({'error': 'No image provided'}), 400

    try:
        # Convert base64 image to numpy array
        image = process_base64_image(request.json['image'])
        
        # Process the image
        signs, _ = detector.process_image(image)
        
        return jsonify({
            'detected_signs': signs
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

def run_webcam():
    """Run the webcam capture and detection loop."""
    try:
        # Initialize webcam
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("Error: Could not open webcam")
            return

        print("\nWebcam initialized successfully!")
        print("Processing frames... The window should appear shortly.")
        
        while not should_exit.is_set():
            ret, frame = cap.read()
            if not ret:
                print("Error: Could not read frame")
                break

            try:
                # Process frame
                annotated_frame, signs = detector.process_frame(frame)
                
                # Update the frame queue (drop frames if queue is full)
                if frame_queue.full():
                    frame_queue.get_nowait()  # Remove old frame
                frame_queue.put_nowait((annotated_frame, signs))
                
            except Exception as e:
                print(f"Error processing frame: {e}")
                continue
            
            time.sleep(0.03)  # Limit frame rate
        
    except Exception as e:
        print(f"Error in webcam thread: {e}")
    
    finally:
        # Clean up
        print("Cleaning up resources...")
        if 'cap' in locals():
            cap.release()
        print("Cleanup complete!")

def find_free_port(start_port=5001, max_port=5010):
    """Find a free port to use for the Flask server."""
    for port in range(start_port, max_port + 1):
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        try:
            sock.bind(('0.0.0.0', port))
            sock.close()
            return port
        except OSError:
            continue
    return None

def run_flask():
    """Run the Flask server in a separate thread."""
    port = find_free_port()
    if port is None:
        print("Error: Could not find a free port for the Flask server")
        return
    
    print(f"\nStarting Flask server on port {port}")
    app.run(host='0.0.0.0', port=port, use_reloader=False)

if __name__ == '__main__':
    print("Starting Hand Sign Detection...")
    print("Initializing webcam and Flask server...")
    
    # Start webcam in a separate thread
    webcam_thread = threading.Thread(target=run_webcam)
    webcam_thread.daemon = True
    webcam_thread.start()
    
    # Start Flask in a separate thread
    flask_thread = threading.Thread(target=run_flask)
    flask_thread.daemon = True
    flask_thread.start()
    
    # Run GUI in the main thread
    window = VideoWindow()
    window.root.mainloop() 