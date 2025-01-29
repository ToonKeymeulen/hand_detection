import cv2
import mediapipe as mp
import numpy as np
from typing import Tuple, List, Optional
import time

class HandSignDetector:
    def __init__(self):
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=True,
            max_num_hands=1,
            min_detection_confidence=0.6,
            min_tracking_confidence=0.6
        )
        self.mp_draw = mp.solutions.drawing_utils
        self.wave_buffer = []
        self.wave_threshold = 8
        self.last_wave_time = 0
        self.wave_cooldown = 2.0
        self.min_wave_amplitude = 0.2
        self.last_finger_states = None

    def process_frame(self, frame: np.ndarray) -> Tuple[np.ndarray, List[str]]:
        if frame is None:
            return frame, []

        frame = cv2.resize(frame, (640, 480))
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.hands.process(frame_rgb)
        detected_signs = []

        if results and results.multi_hand_landmarks:
            hand_landmarks = results.multi_hand_landmarks[0]
            self.mp_draw.draw_landmarks(frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS)
            
            sign = self._detect_sign(hand_landmarks)
            if sign:
                detected_signs.append(sign)
                h, w, _ = frame.shape
                landmark_x = int(hand_landmarks.landmark[0].x * w)
                landmark_y = int(hand_landmarks.landmark[0].y * h)
                cv2.putText(frame, sign, (landmark_x - 20, landmark_y - 20),
                          cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            if self._detect_wave(hand_landmarks):
                detected_signs.append("Waving")
                cv2.putText(frame, "Waving!", (50, 50),
                          cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        return frame, detected_signs

    def _detect_sign(self, landmarks) -> Optional[str]:
        finger_states = self._get_finger_states(landmarks)
        if finger_states is None:
            return None
        
        pattern = "".join("1" if state else "0" for state in finger_states)
        sign_patterns = {
            "11111": "Open Hand",
            "01100": "Peace Sign",
            "10001": "Surfing Sign",
            "00000": "Fist"
        }
        
        return sign_patterns.get(pattern)

    def _get_finger_states(self, landmarks) -> Optional[List[bool]]:
        if not landmarks or not hasattr(landmarks, 'landmark'):
            return None

        finger_landmarks = {
            'Thumb': [4, 2],
            'Index': [8, 6],
            'Middle': [12, 10], 
            'Ring': [16, 14],   
            'Pinky': [20, 18]   
        }
        
        states = []
        for finger_name, (tip_idx, pip_idx) in finger_landmarks.items():
            tip = landmarks.landmark[tip_idx]
            joint = landmarks.landmark[pip_idx]
            
            if finger_name == 'Thumb':
                tip_point = np.array([tip.x, tip.y])
                ip_joint = np.array([landmarks.landmark[3].x, landmarks.landmark[3].y])
                mcp_joint = np.array([landmarks.landmark[2].x, landmarks.landmark[2].y])
                angle = self._calculate_angle(tip_point, ip_joint, mcp_joint)
                extended = angle > 150
            else:
                extended = tip.y < joint.y
            
            states.append(extended)
        
        self.last_finger_states = states
        return states

    def _calculate_angle(self, point1: np.ndarray, point2: np.ndarray, point3: np.ndarray) -> float:
        vector1 = point1 - point2
        vector2 = point3 - point2
        dot_product = np.dot(vector1, vector2)
        magnitude1 = np.linalg.norm(vector1)
        magnitude2 = np.linalg.norm(vector2)
        cos_angle = dot_product / (magnitude1 * magnitude2)
        cos_angle = np.clip(cos_angle, -1.0, 1.0)
        return np.degrees(np.arccos(cos_angle))

    def _detect_wave(self, landmarks) -> bool:
        if not landmarks or not hasattr(landmarks, 'landmark'):
            return False

        current_time = time.time()
        if current_time - self.last_wave_time < self.wave_cooldown:
            return False

        if not self.last_finger_states or not all(self.last_finger_states):
            return False

        palm_x = (landmarks.landmark[0].x + landmarks.landmark[9].x) / 2
        self.wave_buffer.append(palm_x)
        
        if len(self.wave_buffer) > self.wave_threshold:
            self.wave_buffer.pop(0)
            
            if len(self.wave_buffer) >= 6:
                amplitude = max(self.wave_buffer) - min(self.wave_buffer)
                if amplitude < self.min_wave_amplitude:
                    return False

                direction_changes = 0
                for i in range(len(self.wave_buffer) - 2):
                    diff1 = self.wave_buffer[i+1] - self.wave_buffer[i]
                    diff2 = self.wave_buffer[i+2] - self.wave_buffer[i+1]
                    
                    min_movement = 0.03
                    if diff1 * diff2 < 0 and abs(diff1) > min_movement and abs(diff2) > min_movement:
                        direction_changes += 1
                
                if direction_changes >= 2:
                    self.last_wave_time = current_time
                    return True
        
        return False 