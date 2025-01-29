from flask import Blueprint, request, jsonify
import cv2
from ..detector.hand_detector import HandSignDetector
from ..utils.image_processing import process_base64_image, encode_image_to_base64

api = Blueprint('api', __name__)
detector = HandSignDetector()

@api.route('/predict', methods=['POST'])
def predict():
    if not request.json or 'image' not in request.json:
        return jsonify({'error': 'No image provided'}), 400

    try:
        image = process_base64_image(request.json['image'])
        frame_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = detector.hands.process(frame_rgb)
        annotated_image = image.copy()
        
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
                detector.mp_draw.draw_landmarks(
                    annotated_image,
                    hand_landmarks,
                    detector.mp_hands.HAND_CONNECTIONS
                )
                
                landmarks_list = [
                    {'x': landmark.x, 'y': landmark.y, 'z': landmark.z}
                    for landmark in hand_landmarks.landmark
                ]
                response_data['hand_landmarks'].append(landmarks_list)
                
                sign = detector._detect_sign(hand_landmarks)
                if sign:
                    response_data['detected_signs'].append(sign)
                    h, w, _ = annotated_image.shape
                    cv2.putText(annotated_image, sign,
                              (int(hand_landmarks.landmark[0].x * w) - 20,
                               int(hand_landmarks.landmark[0].y * h) - 20),
                              cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        response_data['annotated_image'] = encode_image_to_base64(annotated_image)
        return jsonify(response_data)
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500 