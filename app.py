# Import required libraries
import cv2                  # OpenCV library for image processing and computer vision
import numpy as np         # NumPy for numerical operations on arrays
from flask import Flask, request, jsonify  # Flask for creating web server endpoints
import base64             # For encoding/decoding base64 image data
from hand_detector import HandSignDetector  # Custom class for detecting hand signs
import threading          # For running multiple processes concurrently
from PIL import Image, ImageTk  # PIL for image processing, ImageTk for displaying images in tkinter
import io                 # For handling byte streams
import sys               # For system-level operations
import time              # For adding delays and timing operations
import tkinter as tk     # GUI framework for creating the window
from queue import Queue  # Thread-safe queue for sharing frames between threads
import socket           # For network socket operations

# Initialize Flask app and global objects
app = Flask(__name__)    # Create Flask application instance
detector = HandSignDetector()  # Initialize hand sign detector
frame_queue = Queue(maxsize=1)  # Create queue to store latest frame (only keeps one frame)
should_exit = threading.Event()  # Event flag to signal when program should exit

class VideoWindow:
    def __init__(self):
        # Initialize the main window
        self.root = tk.Tk()  # Create main window
        self.root.title("Hand Sign Detection")  # Set window title
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)  # Handle window close event
        
        # Create and configure the video display canvas
        self.canvas = tk.Canvas(self.root, width=640, height=480)  # Create canvas for video
        self.canvas.pack()  # Add canvas to window
        
        # Create and configure the label for displaying detected signs
        self.sign_label = tk.Label(self.root, text="No signs detected", font=("Arial", 14))
        self.sign_label.pack()  # Add label to window
        
        self.update_frame()  # Start the frame update loop
    
    def on_closing(self):
        # Handle window closing event
        should_exit.set()  # Signal all threads to stop
        self.root.quit()   # Stop the tkinter event loop
        self.root.destroy()  # Destroy the window
    
    def update_frame(self):
        try:
            # Check if there's a new frame available
            # thread safe comms between webcam thread and GUI thread.
            if not frame_queue.empty():
                frame, signs = frame_queue.get_nowait()  # Get latest frame and detected signs
                # not blocking

                
                # Convert frame color format for display
                ## BGR was used for historical reasons
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB
                
                # Convert frame to format suitable for tkinter
                image = Image.fromarray(frame_rgb)  # Convert numpy array to PIL Image, Tkinter uses PIL
                image = image.resize((640, 480), Image.Resampling.LANCZOS)  # Resize image, consistent size, control memory usage, prevent scaling artifacts
                ## other options are nearest, bilinear and bicubic. Each have there own use cases. Don't think this is relevant for us. 
                self.photo = ImageTk.PhotoImage(image=image)  # Convert to PhotoImage, avoid garbage collection if set as instance variable
                ## is required because only format tkinter can display
                
                # Update the canvas with new frame
                self.canvas.create_image(0, 0, image=self.photo, anchor=tk.NW)
                
                # Update the text showing detected signs
                if signs:
                    self.sign_label.config(text=f"Detected signs: {', '.join(signs)}")
                else:
                    self.sign_label.config(text="No signs detected")
        
        except Exception as e:
            print(f"Error updating frame: {e}")  # Log any errors during frame update
        
        # Schedule next frame update if program is still running
        if not should_exit.is_set():
            self.root.after(30, self.update_frame)  # Update every 30ms (approx. 33 fps)

def process_base64_image(base64_string):
    """Convert base64 image to numpy array and preprocess for better detection."""
    try:
        # Decode base64 image
        img_data = base64.b64decode(base64_string)
        img = Image.open(io.BytesIO(img_data))
        
        # Convert to RGB if necessary
        if img.mode != 'RGB':
            img = img.convert('RGB')
        
        # Convert to numpy array
        image = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
        
        # Resize if image is too large or too small
        max_dimension = 1280
        min_dimension = 320
        height, width = image.shape[:2]
        
        if max(height, width) > max_dimension:
            scale = max_dimension / max(height, width)
            image = cv2.resize(image, None, fx=scale, fy=scale)
        elif min(height, width) < min_dimension:
            scale = min_dimension / min(height, width)
            image = cv2.resize(image, None, fx=scale, fy=scale)
        
        # Enhance contrast
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
        cl = clahe.apply(l)
        enhanced = cv2.merge((cl,a,b))
        image = cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)
        
        return image
    except Exception as e:
        print(f"Error processing image: {e}")
        return None

@app.route('/predict', methods=['POST'])
def predict():
    """Flask endpoint for processing images from HTTP requests"""
    # Validate request contains image data
    if not request.json or 'image' not in request.json:
        return jsonify({'error': 'No image provided'}), 400

    try:
        # Process the received image
        image = process_base64_image(request.json['image'])  # Convert base64 to image
        
        # Get landmarks and signs
        frame_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = detector.hands.process(frame_rgb)
        
        # Create a copy of the image for annotation
        annotated_image = image.copy()
        
        # Initialize response data
        response_data = {
            'detected_signs': [],
            'landmarks_detected': False,
            'num_hands_detected': 0,
            'hand_landmarks': []
        }
        
        if results.multi_hand_landmarks:
            response_data['landmarks_detected'] = True
            response_data['num_hands_detected'] = len(results.multi_hand_landmarks)
            
            for hand_landmarks in results.multi_hand_landmarks:
                # Draw landmarks on the image
                detector.mp_draw.draw_landmarks(
                    annotated_image,
                    hand_landmarks,
                    detector.mp_hands.HAND_CONNECTIONS,
                    detector.mp_draw.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=4),
                    detector.mp_draw.DrawingSpec(color=(0, 0, 255), thickness=2)
                )
                
                # Convert landmarks to list for JSON serialization
                landmarks_list = [
                    {
                        'x': landmark.x,
                        'y': landmark.y,
                        'z': landmark.z
                    }
                    for landmark in hand_landmarks.landmark
                ]
                response_data['hand_landmarks'].append(landmarks_list)
                
                # Detect signs
                sign = detector._detect_sign(hand_landmarks)
                if sign:
                    response_data['detected_signs'].append(sign)
                    # Add text for detected sign
                    h, w, _ = annotated_image.shape
                    cv2.putText(annotated_image, sign,
                              (int(hand_landmarks.landmark[0].x * w) - 20,
                               int(hand_landmarks.landmark[0].y * h) - 20),
                              cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        # Convert annotated image to base64
        _, buffer = cv2.imencode('.jpg', annotated_image)
        annotated_image_base64 = base64.b64encode(buffer).decode('utf-8')
        response_data['annotated_image'] = annotated_image_base64
        
        return jsonify(response_data)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

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

def run_flask():
    """Initialize and run Flask server"""
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
    window = VideoWindow()  # Create main window
    window.root.mainloop()  # Start GUI event loop 