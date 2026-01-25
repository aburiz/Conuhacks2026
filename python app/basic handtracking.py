import cv2
import urllib.request
from pathlib import Path

import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# Download the new MediaPipe hand landmarker model if missing.
MODEL_URL = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
MODEL_PATH = Path(__file__).with_name("hand_landmarker.task")

# Minimal hand connections so we can draw without the legacy solutions API.
HAND_CONNECTIONS = (
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (0, 17), (17, 18), (18, 19), (19, 20),
    (16, 20), (8, 12), (12, 16), (4, 8)
)


def ensure_model():
    if MODEL_PATH.exists():
        return
    print("Downloading hand_landmarker.task ...")
    urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)


def create_landmarker():
    options = vision.HandLandmarkerOptions(
        base_options=python.BaseOptions(model_asset_path=str(MODEL_PATH)),
        running_mode=vision.RunningMode.VIDEO,
        num_hands=2,
        min_hand_detection_confidence=0.5,
        min_hand_presence_confidence=0.5,
        min_tracking_confidence=0.5,
    )
    return vision.HandLandmarker.create_from_options(options)


def draw_fingertips_bgr(frame, landmarks):
    """Draw only fingertip points (thumb + four fingers)."""

    h, w, _ = frame.shape
    pts = [(int(l.x * w), int(l.y * h)) for l in landmarks]
    fingertip_ids = [4, 8, 12, 16, 20]
    fingertip_names = {4: "Thumb", 8: "Index", 12: "Middle", 16: "Ring", 20: "Pinky"}

    for tip_id in fingertip_ids:
        x, y = pts[tip_id]
        cv2.circle(frame, (x, y), 10, (0, 255, 255), -1)
        cv2.putText(frame, fingertip_names[tip_id], (x - 30, y - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
        cv2.putText(frame, fingertip_names[tip_id], (x - 30, y - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)


def count_fingers_up(landmarks, image_shape):
    """Return how many fingers are up (thumb + 4 fingers).

    Uses a simple heuristic: for fingers 1-4, compare tip.y vs pip.y.
    For thumb, compare tip.x vs pip.x assuming a front-facing camera.
    """

    h, w, _ = image_shape
    pts = [(l.x * w, l.y * h) for l in landmarks]

    fingers_up = 0

    # Thumb: tip 4 vs joint 3 (horizontal check). This is approximate without handedness.
    thumb_tip_x, thumb_tip_y = pts[4]
    thumb_joint_x, thumb_joint_y = pts[3]
    if thumb_tip_x < thumb_joint_x:  # thumb pointing left-ish
        fingers_up += 1

    # Index, middle, ring, pinky: tip vs pip (higher -> finger up)
    finger_tips = [8, 12, 16, 20]
    finger_pips = [6, 10, 14, 18]
    for tip, pip in zip(finger_tips, finger_pips):
        tip_y = pts[tip][1]
        pip_y = pts[pip][1]
        if tip_y < pip_y:
            fingers_up += 1

    return fingers_up


def classify_gesture(landmarks, image_shape):
    fingers = count_fingers_up(landmarks, image_shape)
    if fingers >= 4:
        return "Open hand"
    if fingers <= 1:
        return "Closed fist"
    return "Partial / Unknown"


def main():
    ensure_model()

    try:
        landmarker = create_landmarker()
    except Exception as exc:  # noqa: BLE001
        print(f"Failed to create hand landmarker: {exc}")
        return

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Could not open camera")
        return

    timestamp_ms = 0
    print("Starting hand tracking... Press 'q' to quit")

    while cap.isOpened():
        success, bgr = cap.read()
        if not success:
            break

        timestamp_ms += 33  # rough 30 FPS tick
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        result = landmarker.detect_for_video(mp_image, timestamp_ms)

        if result.hand_landmarks:
            for hand in result.hand_landmarks:
                draw_fingertips_bgr(bgr, hand)

        cv2.imshow("Gesture Demo", bgr)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
