#! python3.11

import cv2
from ultralytics import YOLO
import numpy as np

# --------------------
# YOLO model setup
# --------------------
MODEL_PATH = r"runs\pose\last-hopefully2\weights\best.pt"
model = YOLO(MODEL_PATH)
model.model.eval()

CONF_THRESH = 0.4
KEYPOINT_COLOR = (0, 255, 0)

# --------------------
# Temporal smoothing and jump rejection
# --------------------
SMOOTH_ALPHA = 0.8    # EMA smoothing factor
MAX_JUMP = 75         # max pixels a keypoint can jump
prev_kpts = None

# --------------------
# Keypoint indices for your guitar fretboard
# --------------------
TOP_L = 0
BOTTOM_L = 3
TOP_R = 1
BOTTOM_R = 2

# --------------------
# Manual reference point setup
# --------------------
anchor_points = None
click_counter = 0
click_order = [TOP_L, TOP_R, BOTTOM_R, BOTTOM_L]

def mouse_callback(event, x, y, flags, param):
    global anchor_points, click_counter
    if event == cv2.EVENT_LBUTTONDOWN:
        if anchor_points is None:
            anchor_points = np.zeros((4, 2), dtype=np.float32)
        if click_counter < 4:
            anchor_points[click_counter] = [x, y]
            print(f"Anchor point {click_counter} set at ({x}, {y})")
            click_counter += 1

# --------------------
# Camera setup
# --------------------
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
cv2.namedWindow("FretVision")
cv2.setMouseCallback("FretVision", mouse_callback)

print("Click 4 reference points: top-left, top-right, bottom-right, bottom-left")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)  # mirror

    # Show instructions until all points clicked
    if anchor_points is None or click_counter < 4:
        for i in range(click_counter):
            cv2.circle(frame, tuple(anchor_points[i].astype(int)), 5, (0, 0, 255), -1)
        cv2.putText(frame, "Click 4 reference points (TL, TR, BR, BL)", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        cv2.imshow("FretVision", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
        continue

    # YOLO inference
    results = model(frame, conf=CONF_THRESH, imgsz=416, verbose=False)

    for r in results:
        if r.keypoints is None:
            continue

        kpts_xy = r.keypoints.xy.cpu().numpy()
        kpts_conf = r.keypoints.conf.cpu().numpy()

        if kpts_xy.shape[0] == 0:
            continue

        # --------------------
        # Initialize prev_kpts using reference clicks (first detection)
        # --------------------
        if prev_kpts is None or prev_kpts.shape != kpts_xy.shape:
            prev_kpts = kpts_xy.copy()
            # seed left/right corners with reference clicks
            prev_kpts[0, TOP_L] = anchor_points[0]
            prev_kpts[0, TOP_R] = anchor_points[1]
            prev_kpts[0, BOTTOM_R] = anchor_points[2]
            prev_kpts[0, BOTTOM_L] = anchor_points[3]

        # --------------------
        # EMA smoothing
        # --------------------
        smoothed_kpts = SMOOTH_ALPHA * prev_kpts + (1.0 - SMOOTH_ALPHA) * kpts_xy

        # --------------------
        # Jump rejection (hard clamp)
        # --------------------
        delta = np.linalg.norm(smoothed_kpts - prev_kpts, axis=2)
        jump_mask = delta > MAX_JUMP
        smoothed_kpts[jump_mask] = prev_kpts[jump_mask]

        # Update previous
        prev_kpts = smoothed_kpts.copy()

        # --------------------
        # Draw keypoints
        # --------------------
        for i in range(smoothed_kpts.shape[0]):
            for j, (x, y) in enumerate(smoothed_kpts[i]):
                if kpts_conf[i][j] > CONF_THRESH:
                    cv2.circle(frame, (int(x), int(y)), 5, KEYPOINT_COLOR, -1)

    # Draw reference anchors for visualization
    for pt in anchor_points:
        cv2.circle(frame, tuple(pt.astype(int)), 5, (0, 0, 255), 2)

    cv2.imshow("FretVision", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
