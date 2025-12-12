import cv2
import numpy as np
import mediapipe as mp
import tensorflow as tf
from collections import deque

# ---- Normalization (must match training exactly) ----
def normalize_landmarks(landmarks):
    pts = np.array(landmarks).reshape(-1, 2)   # shape (21,2)

    wrist = pts[0]  # landmark 0 as origin
    pts = pts - wrist

    max_val = np.max(np.abs(pts))
    if max_val > 0:
        pts = pts / max_val

    return pts.flatten().tolist()

# ---- Load model and label encoder ----
model = tf.keras.models.load_model("hand_gesture_model.keras")
label_encoder = np.load("label_encoder.npy", allow_pickle=True)

pred_buffer = deque(maxlen=7)

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.5)
mp_draw = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)

print("🎥 Real-time Gesture Recognition Running... Press 'q' to exit.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(img_rgb)

    if results.multi_hand_landmarks:
        hand_landmarks = results.multi_hand_landmarks[0]

        # ---- Extract all 21 landmarks ----
        landmarks = []
        for lm in hand_landmarks.landmark:
            landmarks.append(lm.x)
            landmarks.append(lm.y)

        # ENSURE we have EXACTLY 42 values
        if len(landmarks) == 42:
            # ---- Normalize ----
            norm = normalize_landmarks(landmarks)

            # ---- Predict ----
            prediction = model.predict(np.array([norm]), verbose=0)
            confidence = np.max(prediction)
            pred_class = np.argmax(prediction)

            if confidence > 0.60:
                pred_buffer.append(pred_class)

            if len(pred_buffer) > 0:
                stable_pred = max(set(pred_buffer), key=pred_buffer.count)
                gesture_name = label_encoder[stable_pred]
            else:
                gesture_name = "..."

            # ---- Draw output ----
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            cv2.putText(frame, gesture_name, (10, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)

    cv2.imshow("Hand Gesture Recognition", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyA
