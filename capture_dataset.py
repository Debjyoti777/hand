import cv2
import os

# ✋ Change this to the gesture you want to record
gesture_name = "hello"
save_path = f"dataset/{gesture_name}"

# Create folder if it doesn't exist
os.makedirs(save_path, exist_ok=True)

cam = cv2.VideoCapture(0)
count = 0
print("📸 Press 's' to save image, 'q' to quit")

while True:
    ret, frame = cam.read()
    if not ret:
        print("Error: Camera not found.")
        break

    frame = cv2.flip(frame, 1)  # mirror effect
    cv2.putText(frame, f"{gesture_name} - {count}", (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow("Capture Gesture", frame)

    key = cv2.waitKey(1)
    if key == ord('s'):
        file_name = os.path.join(save_path, f"{count}.jpg")
        cv2.imwrite(file_name, frame)
        print(f"✅ Saved {file_name}")
        count += 1
    elif key == ord('q'):
        break

cam.release()
cv2.destroyAllWindows()
