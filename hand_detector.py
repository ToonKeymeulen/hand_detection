import cv2
import mediapipe as mp
import numpy as np
from typing import Tuple, List, Optional

class HandSignDetector:
    def __init__(self):
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False, # for video, do not run detection on every model seperately. 
            max_num_hands=2, # speaks for itself
            min_detection_confidence=0.7,
            min_tracking_confidence=0.5
        )
        self.mp_draw = mp.solutions.drawing_utils
        self.wave_buffer = []
        self.wave_threshold = 5  # frames to detect wave, TODO: Reason about this, why is this a good value? Look for whether another value is more suitable?

    def process_frame(self, frame: np.ndarray) -> Tuple[np.ndarray, List[str]]:
        """Process a frame and return the annotated frame with detected signs."""
        if frame is None:
            print("Error: Received empty frame")
            return frame, []

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) # required for mediapipe, opencv uses BGR
        results = self.hands.process(frame_rgb)
        detected_signs = []

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                try:
                    # Draw hand landmarks
                    self.mp_draw.draw_landmarks(
                        frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS)
                    
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
                    continue

        return frame, detected_signs

    def _detect_sign(self, landmarks) -> Optional[str]:
        """Detect the hand sign based on landmark positions."""
        try:
            # Extract finger states (extended/closed)
            finger_states = self._get_finger_states(landmarks)
            
            if finger_states is None:
                return None
            
            # Open hand: all fingers extended
            if all(finger_states):
                return "Open Hand"
            

            # finger 0 => thumb
            # finger 1 => index
            # finger 2 => middle
            # finger 3 => ring
            # finger 4 => pinky
            
            # Peace sign: index and middle fingers extended, others closed
            if (finger_states[1] and finger_states[2] and 
                not finger_states[0] and not finger_states[3] and not finger_states[4]):
                return "Peace Sign"
            
            # Surfing sign: thumb and pinky extended, others closed
            if (finger_states[0] and finger_states[4] and 
                not finger_states[1] and not finger_states[2] and not finger_states[3]):
                return "Surfing Sign"
            
            # Rock sign (bonus): index and pinky extended, others closed
            if (finger_states[1] and finger_states[4] and 
                not finger_states[0] and not finger_states[2] and not finger_states[3]):
                return "Rock Sign"
            
            return None
        except Exception as e:
            print(f"Error in _detect_sign: {e}")
            return None

    def _get_finger_states(self, landmarks) -> Optional[List[bool]]:
        """Get the state of each finger (extended/closed)."""
        try:
            if not landmarks or not hasattr(landmarks, 'landmark'):
                print("Invalid landmarks object")
                return None

            ## landmark detection gemini api
            ## https://ai.google.dev/edge/mediapipe/solutions/vision/hand_landmarker

            # Landmark indices for fingertips
            finger_tips = [4, 8, 12, 16, 20] 
            # Landmark indices for finger bases
            finger_bases = [2, 5, 9, 13, 17] 
            
            # Verify all required landmarks are present
            ## TODO: Reason about this. Is this a good way to do it? Should all landmarks be present?
            for idx in finger_tips + finger_bases:
                if idx >= len(landmarks.landmark):
                    print(f"Missing landmark index: {idx}")
                    return None
            
            states = []
            for tip, base in zip(finger_tips, finger_bases):
                # Check if finger is extended by comparing y coordinates
                # For thumb, use x coordinate instead
                if tip == 4:  # Thumb
                    extended = (landmarks.landmark[tip].x < landmarks.landmark[base].x)
                ## all other fingers
                else:
                    extended = (landmarks.landmark[tip].y < landmarks.landmark[base].y)
                states.append(extended)
            
            return states
        except Exception as e:
            print(f"Error in _get_finger_states: {e}")
            return None

    def _detect_wave(self, landmarks) -> bool:
        """Detect waving motion based on hand movement."""
        ## TODO: Should this not be smarter? As in what are the fingers doing? Imo waving is not just wrist movement. The stretching of the fingers is also important.
        try:
            if not landmarks or not hasattr(landmarks, 'landmark') or len(landmarks.landmark) == 0:
                return False

            wrist_y = landmarks.landmark[0].y # wrist landmark
            self.wave_buffer.append(wrist_y)
            
            if len(self.wave_buffer) > self.wave_threshold:
                self.wave_buffer.pop(0)
                
                # Check for oscillating motion
                if len(self.wave_buffer) >= 4: ## TODO: this should be a variable, depends on the threshold
                    oscillations = 0
                    for i in range(len(self.wave_buffer) - 2):
                        if (self.wave_buffer[i] - self.wave_buffer[i+1]) * \
                           (self.wave_buffer[i+1] - self.wave_buffer[i+2]) < 0:
                            oscillations += 1
                    
                    return oscillations >= 2
            
            return False
        except Exception as e:
            print(f"Error in _detect_wave: {e}")
            return False

    def process_image(self, image: np.ndarray) -> Tuple[List[str], np.ndarray]:
        """Process a single image and return detected signs and annotated image."""
        annotated_image, signs = self.process_frame(image)
        return signs, annotated_image 