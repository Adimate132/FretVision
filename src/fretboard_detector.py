import cv2
from ultralytics import YOLO
from collections import deque

# --------------------
# Config
# --------------------
MODEL_PATH = r"data\model\pose\colab trained\weights\best.pt"

CONF_THRESH = 0.6        # <<<<<< adjust this
APPEAR_FRAMES = 3         # frames required before showing
DISAPPEAR_FRAMES = 5      # frames allowed missing before hiding

# --------------------
# Load model
# --------------------
model = YOLO(MODEL_PATH)

# --------------------
# Webcam
# --------------------
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    raise RuntimeError("Could not open webcam")

# --------------------
# Temporal smoothing state
# --------------------
visible_counter = 0
missing_counter = 0
show_detection = False

# --------------------
# Loop
# --------------------
while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame, conf=CONF_THRESH, imgsz=640, verbose=False)
    r = results[0]

    has_detection = r.boxes is not None and len(r.boxes) > 0

    # --------------------
    # Smoothing logic
    # --------------------
    if has_detection:
        visible_counter += 1
        missing_counter = 0
    else:
        missing_counter += 1
        visible_counter = 0

    if visible_counter >= APPEAR_FRAMES:
        show_detection = True

    if missing_counter >= DISAPPEAR_FRAMES:
        show_detection = False

    # --------------------
    # Render
    # --------------------
    if show_detection and has_detection:
        frame = r.plot()

    cv2.imshow("YOLOv8 Pose - Live Feed", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# --------------------
# Cleanup
# --------------------
cap.release()
cv2.destroyAllWindows()
