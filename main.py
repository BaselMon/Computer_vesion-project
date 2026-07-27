import cv2
import mediapipe as mp
from ultralytics import YOLO

# ── Load Models ──────────────────────────────────────────────
yolo_model = YOLO("yolov8n.pt")

mp_hands    = mp.solutions.hands
mp_face     = mp.solutions.face_detection
mp_drawing  = mp.solutions.drawing_utils

hands       = mp_hands.Hands(max_num_hands=2, min_detection_confidence=0.7)
face_detect = mp_face.FaceDetection(min_detection_confidence=0.7)

# COCO class index for sports ball
BALL_CLASS_ID = 32

# ── Open Webcam ───────────────────────────────────────────────
cap = cv2.VideoCapture(0)

print("Running... Press 'q' to quit.")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    rgb   = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    h, w  = frame.shape[:2]

    # ── 1. Face Detection (MediaPipe) ─────────────────────────
    face_results = face_detect.process(rgb)
    if face_results.detections:
        for detection in face_results.detections:
            bbox  = detection.location_data.relative_bounding_box
            x1    = int(bbox.xmin * w)
            y1    = int(bbox.ymin * h)
            bw    = int(bbox.width  * w)
            bh    = int(bbox.height * h)
            x2, y2 = x1 + bw, y1 + bh

            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, "Face", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    # ── 2. Hand Detection (MediaPipe) ─────────────────────────
    hand_results = hands.process(rgb)
    if hand_results.multi_hand_landmarks:
        for hand_landmarks in hand_results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(frame, hand_landmarks,
                                      mp_hands.HAND_CONNECTIONS)
            xs = [lm.x * w for lm in hand_landmarks.landmark]
            ys = [lm.y * h for lm in hand_landmarks.landmark]
            x1, y1 = int(min(xs)) - 10, int(min(ys)) - 10
            x2, y2 = int(max(xs)) + 10, int(max(ys)) + 10
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 165, 0), 2)
            cv2.putText(frame, "Hand", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 165, 0), 2)

    # ── 3. Ball Detection (YOLO) ──────────────────────────────
    yolo_results = yolo_model(frame, verbose=False)[0]
    for box in yolo_results.boxes:
        cls = int(box.cls[0])
        if cls == BALL_CLASS_ID:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = float(box.conf[0])
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
            cv2.putText(frame, f"Ball {conf:.0%}", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

    # ── Display ───────────────────────────────────────────────
    cv2.imshow("Triple Object Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# ── Cleanup ───────────────────────────────────────────────────
cap.release()
cv2.destroyAllWindows()