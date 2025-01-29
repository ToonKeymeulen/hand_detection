import cv2
import mediapipe as mp
import numpy as np
from typing import Tuple, List, Optional
import time

class HandSignDetector:
    def __init__(self):
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=True,  # True for images
            max_num_hands=1,  # Limit to 1 hand for more stable detection
            min_detection_confidence=0.6,  # Balanced confidence threshold
            min_tracking_confidence=0.6
        )
        self.mp_draw = mp.solutions.drawing_utils
        self.wave_buffer = []
        self.wave_threshold = 8  # Buffer size for wave detection
        self.last_wave_time = 0  # Track last wave detection time
        self.wave_cooldown = 2.0  # Cooldown in seconds
        self.min_wave_amplitude = 0.2  # Minimum horizontal movement required
        self.last_finger_states = None  # Store last detected finger states
        
        # Add angle smoothing buffers
        self.angle_buffers = {
            'Thumb': [],
            'Index': [],
            'Middle': [],
            'Ring': [],
            'Pinky': []
        }
        self.smoothing_window = 3  # Number of frames to average over

    def process_frame(self, frame: np.ndarray) -> Tuple[np.ndarray, List[str]]:
        """Process a frame and return the annotated frame with detected signs."""
        if frame is None:
            print("Error: Received empty frame")
            return frame, []

        # Ensure consistent image size and type
        target_size = (640, 480)
        frame = cv2.resize(frame, target_size)
        frame = frame.astype(np.uint8)  # Ensure 8-bit image

        # Convert to RGB for MediaPipe
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Single detection with high confidence
        results = self.hands.process(frame_rgb)
        detected_signs = []

        if results and results.multi_hand_landmarks:
            # If multiple hands detected, use the one with highest confidence
            best_hand_idx = 0
            if len(results.multi_hand_landmarks) > 1 and results.multi_handedness:
                confidences = [hand.classification[0].score for hand in results.multi_handedness]
                best_hand_idx = confidences.index(max(confidences))
            
            # Only process the best hand
            hand_landmarks = results.multi_hand_landmarks[best_hand_idx]
            try:
                # Draw hand landmarks
                self.mp_draw.draw_landmarks(
                    frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS)
                
                # Print confidence score if available
                if results.multi_handedness:
                    confidence = results.multi_handedness[best_hand_idx].classification[0].score
                    print(f"\nHand detection confidence: {confidence:.3f}")
                
                # Detect signs
                sign = self._detect_sign(hand_landmarks)
                if sign:
                    detected_signs.append(sign)
                    # Add text above the hand
                    h, w, _ = frame.shape
                    landmark_x = int(hand_landmarks.landmark[0].x * w)
                    landmark_y = int(hand_landmarks.landmark[0].y * h)
                    cv2.putText(frame, sign, (landmark_x - 20, landmark_y - 20),
                              cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

                # Check for wave motion
                if self._detect_wave(hand_landmarks):
                    detected_signs.append("Waving")
                    cv2.putText(frame, "Waving!", (50, 50),
                              cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            except Exception as e:
                print(f"Error processing hand landmarks: {e}")

        return frame, detected_signs

    def _detect_sign(self, landmarks) -> Optional[str]:
        """Detect the hand sign based on landmark positions."""
        try:
            # Get finger states (extended/closed)
            finger_states = self._get_finger_states(landmarks)
            if finger_states is None:
                return None
            
            # Print overall finger state pattern
            print("\nFinger Pattern:", "".join("1" if state else "0" for state in finger_states))
            print("-" * 30)
            
            # Create pattern string for easier matching
            pattern = "".join("1" if state else "0" for state in finger_states)
            
            # Define sign patterns
            sign_patterns = {
                "11111": "Open Hand",
                "01100": "Peace Sign",
                "10001": "Surfing Sign",
                "01001": "Rock Sign",
                "11000": "Gun Sign",
                "11110": "Four",
                "11100": "Three",
                "11000": "Two",
                "01000": "One",
                "00000": "Fist"
            }
            
            # Return matching sign or None
            return sign_patterns.get(pattern)
            
        except Exception as e:
            print(f"Error in _detect_sign: {e}")
            return None

    def _get_finger_states(self, landmarks) -> Optional[List[bool]]:
        """Get the state of each finger (extended/closed) using y-coordinate comparison."""
        try:
            if not landmarks or not hasattr(landmarks, 'landmark'):
                print("Invalid landmarks object")
                return None

            # Landmark indices for each finger [tip, pip (middle joint)]
            finger_landmarks = {
                'Thumb': [4, 2],    # Special case: compare with mcp due to thumb orientation
                'Index': [8, 6],    # Compare tip with pip
                'Middle': [12, 10], 
                'Ring': [16, 14],   
                'Pinky': [20, 18]   
            }
            
            states = []
            print("\nFinger States:")
            print("-" * 50)
            
            # Process each finger
            for finger_name, (tip_idx, pip_idx) in finger_landmarks.items():
                try:
                    # Get coordinates of tip and joint
                    tip = landmarks.landmark[tip_idx]
                    joint = landmarks.landmark[pip_idx]
                    wrist = landmarks.landmark[0]  # Wrist landmark for reference
                    
                    if finger_name == 'Thumb':
                        # For thumb, calculate angle between tip (4), IP joint (3), and MCP joint (2)
                        tip = np.array([tip.x, tip.y])
                        ip_joint = np.array([landmarks.landmark[3].x, landmarks.landmark[3].y])
                        mcp_joint = np.array([landmarks.landmark[2].x, landmarks.landmark[2].y])
                        
                        # Calculate angle at the IP joint
                        angle = self._calculate_angle(tip, ip_joint, mcp_joint)
                        
                        # Thumb is extended if angle is greater than 150 degrees
                        extended = angle > 150
                    else:
                        # Other fingers are extended if tip is higher than joint
                        extended = tip.y < joint.y
                    
                    states.append(extended)
                    print(f"{finger_name:6}: {'Extended' if extended else 'Closed'}")
                
                except IndexError as e:
                    print(f"Error processing {finger_name}: Invalid landmark index - {e}")
                    return None
                except Exception as e:
                    print(f"Error processing {finger_name}: {e}")
                    return None
            
            # Print binary pattern
            pattern = "".join("1" if state else "0" for state in states)
            print(f"\nBinary pattern: {pattern}")
            print("-" * 50)
            
            # Store the finger states
            self.last_finger_states = states
            
            return states
        except Exception as e:
            print(f"Error in _get_finger_states: {e}")
            return None

    def _calculate_angle(self, point1: np.ndarray, point2: np.ndarray, point3: np.ndarray) -> float:
        """
        Calculate the angle between three points in degrees.
        The angle is calculated at point2 (the joint).
        """
        try:
            # Create vectors from the points
            vector1 = point1 - point2  # Vector from joint to tip
            vector2 = point3 - point2  # Vector from joint to base
            
            # Calculate dot product
            dot_product = np.dot(vector1, vector2)
            
            # Calculate magnitudes
            magnitude1 = np.linalg.norm(vector1)
            magnitude2 = np.linalg.norm(vector2)
            
            # Calculate angle in degrees
            cos_angle = dot_product / (magnitude1 * magnitude2)
            cos_angle = np.clip(cos_angle, -1.0, 1.0)  # Handle numerical errors
            angle = np.degrees(np.arccos(cos_angle))
            
            return angle
        except Exception as e:
            print(f"Error calculating angle: {e}")
            return 0.0

    def _detect_wave(self, landmarks) -> bool:
        """
        Detect waving motion based on horizontal hand movement.
        Requirements:
        1. Hand must be in open hand position (all fingers extended)
        2. Consistent left-right oscillation
        3. Time-based cooldown between detections
        """
        try:
            if not landmarks or not hasattr(landmarks, 'landmark'):
                return False

            # Check if enough time has passed since last wave
            current_time = time.time()
            if current_time - self.last_wave_time < self.wave_cooldown:
                return False

            # Verify open hand position (all fingers extended)
            if not self.last_finger_states or not all(self.last_finger_states):
                return False

            # Track palm center x-coordinate (using wrist and middle finger MCP)
            palm_x = (landmarks.landmark[0].x + landmarks.landmark[9].x) / 2
            self.wave_buffer.append(palm_x)
            
            if len(self.wave_buffer) > self.wave_threshold:
                self.wave_buffer.pop(0)
                
                # Check for consistent left-right motion
                if len(self.wave_buffer) >= 6:
                    # Calculate horizontal movement amplitude
                    amplitude = max(self.wave_buffer) - min(self.wave_buffer)
                    if amplitude < self.min_wave_amplitude:
                        return False

                    # Count direction changes (left-to-right and right-to-left)
                    direction_changes = 0
                    for i in range(len(self.wave_buffer) - 2):
                        # Calculate differences between consecutive positions
                        diff1 = self.wave_buffer[i+1] - self.wave_buffer[i]
                        diff2 = self.wave_buffer[i+2] - self.wave_buffer[i+1]
                        
                        # Check for direction change with minimum movement
                        min_movement = 0.03  # Minimum movement threshold
                        if diff1 * diff2 < 0 and abs(diff1) > min_movement and abs(diff2) > min_movement:
                            direction_changes += 1
                    
                    # Require at least 2 direction changes (complete left-right-left motion)
                    if direction_changes >= 2:
                        self.last_wave_time = current_time
                        return True
            
            return False
        except Exception as e:
            print(f"Error in _detect_wave: {e}")
            return False

    def process_image(self, image: np.ndarray) -> Tuple[List[str], np.ndarray]:
        """Process a single image and return detected signs and annotated image."""
        annotated_image, signs = self.process_frame(image)
        return signs, annotated_image 