import os
import cv2
import numpy as np
import mediapipe as mp

DATASET_PATH = "dataset"

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=True)
mp_draw = mp.solutions.drawing_utils

all_data = []
all_labels = []

def normalize_landmarks(landmarks):
    # Convert to numpy array (21 landmarks x 2)
    pts = np.array(landmarks).reshape(-1, 2)

    # Use wrist as origin
    wrist = pts[0]
    pts = pts - wrist

    # Scale by maximum absolute value
    max_val = np.max(np.abs(pts))
    if max_val > 0:
        pts = pts / max_val

    return pts.flatten().tolist()

for label in os.listdir(DATASET_PATH):
    folder = os.path.join(DATASET_PATH, label)

    for file in os.listdir(folder):
        img_path = os.path.join(folder, file)
        img = cv2.imread(img_path)

        if img is None:
            continue

        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = hands.process(img_rgb)

        if results.multi_hand_landmarks:
            lm = results.multi_hand_landmarks[0]

            landmarks = []
            for p in lm.landmark:
                landmarks.append(p.x)
                landmarks.append(p.y)

            normalized = normalize_landmarks(landmarks)

            all_data.append(normalized)
            all_labels.append(label)

np.save("landmarks_data.npy", {"data": all_data, "labels": all_labels})
print("Saved normalized landmark dataset!")
