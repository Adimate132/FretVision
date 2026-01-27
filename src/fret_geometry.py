# overlay from model + string estimation
import cv2
import numpy as np
from ultralytics import YOLO

# 1. Setup Model and Camera
MODEL_PATH = r"data\model\pose\colab trained\weights\best.pt"
model = YOLO(MODEL_PATH)

cap = cv2.VideoCapture(0)

# ADJUST THIS: 0.1 means strings start 10% away from the edge.
# Increase this value to move strings further inward.
STRING_INSET = 0.1 

while cap.isOpened():
    success, frame = cap.read()
    if not success: break

    results = model(frame, conf=0.5, verbose=False, stream=True)

    for r in results:
        # Defensive Check: Ensure detection exists
        if r.keypoints is None or len(r.keypoints.xy) == 0:
            continue

        kpts = r.keypoints.xy[0].cpu().numpy()
        confs = r.keypoints.conf[0].cpu().numpy()

        # Defensive Check: Ensure all 4 corners are confident
        if len(kpts) >= 4 and all(c > 0.5 for c in confs[:4]):
            
            # Mapping: 0:L-Nut, 1:R-Nut, 2:LastFret-R, 3:LastFret-L
            lnut, rnut, lf_r, lf_l = kpts[0], kpts[1], kpts[2], kpts[3]

            num_strings = 6
            
            # 2. Draw the 6 Inset Strings (so the 1st and 6th strings aren't on the edge of the board)
            for i in range(num_strings):
                # Calculate 't' so it stays between STRING_INSET and (1 - STRING_INSET)
                # Example: if inset is 0.1, t moves from 0.1 to 0.9
                if num_strings > 1:
                    t = STRING_INSET + (i * (1 - 2 * STRING_INSET) / (num_strings - 1))
                else:
                    t = 0.5

                # Points on the Nut
                x_start = int(lnut[0] + t * (rnut[0] - lnut[0]))
                y_start = int(lnut[1] + t * (rnut[1] - lnut[1]))
                
                # Points on the Last Fret (matching the side of the nut)
                x_end = int(lf_l[0] + t * (lf_r[0] - lf_l[0]))
                y_end = int(lf_l[1] + t * (lf_r[1] - lf_l[1]))

                # Draw strings (Cyan for better visibility)
                cv2.line(frame, (x_start, y_start), (x_end, y_end), (255, 255, 0), 1)
                
                # Optional: Small dots to see string positions
                cv2.circle(frame, (x_start, y_start), 2, (0, 0, 255), -1)

            # 3. Draw Fretboard Boundary (Blue)
            fret_points = np.array([lnut, rnut, lf_r, lf_l], dtype=np.int32)
            cv2.polylines(frame, [fret_points], isClosed=True, color=(255, 0, 0), thickness=2)

        else:
            cv2.putText(frame, "Searching for fretboard...", (10, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 1)

    cv2.imshow("FretVision - Inset Strings", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()