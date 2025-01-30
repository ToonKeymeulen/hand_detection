import cv2
from flask import Flask
from src.detector.hand_detector import HandSignDetector
import threading
import time
from queue import Queue
import socket
from src.gui.video_window import VideoWindow
from src.api.routes import api

# Initialize Flask app and global objects
app = Flask(__name__)    
frame_queue = Queue(maxsize=1)  # Create queue to store latest frame (only keeps one frame)
should_exit = threading.Event()  # Event flag to signal when program should exit
detector = HandSignDetector()  # Create an instance of HandSignDetector

def find_free_port(start_port=5001, max_port=5010):
    """Find available port for Flask server"""
    for port in range(start_port, max_port + 1):
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        try:
            sock.bind(('0.0.0.0', port))
            sock.close()
            return port
        except OSError:
            continue
    return None

def run_webcam():
    """Main webcam capture and processing loop"""
    try:
        # Set up webcam capture
        cap = cv2.VideoCapture(0)  # Initialize webcam (device 0), the default
        if not cap.isOpened():
            print("Error: Could not open webcam")
            return

        print("\nWebcam initialized successfully!")
        print("Processing frames... The window should appear shortly.")
        
        # FPS monitoring variables
        frame_count = 0
        fps_start_time = time.time()
        fps = 0
        
        # Main processing loop
        while not should_exit.is_set():
            ret, frame = cap.read()  # Read frame from webcam
            if not ret:
                print("Error: Could not read frame")
                break

            try:
                # Process current frame
                annotated_frame, signs = detector.process_frame(frame)  # Detect signs
                
                # Calculate and display FPS
                frame_count += 1
                if frame_count % 30 == 0:  # Update FPS every 30 frames
                    current_time = time.time()
                    fps = 30 / (current_time - fps_start_time)
                    fps_start_time = current_time
                    print(f"Current FPS: {fps:.2f}")
                
                # Add FPS text to frame
                cv2.putText(annotated_frame, f"FPS: {fps:.1f}", (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                
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