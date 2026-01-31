import cv2
import numpy as np
from ultralytics import YOLO

# 1. Setup Model and Camera
MODEL_PATH = r"data\model\pose\colab trained\weights\best.pt"
model = YOLO(MODEL_PATH)

cap = cv2.VideoCapture(0)

# CONFIGURATION
STRING_INSET = 0.1 
NUM_FRETS = 21  # Your guitar's fret count

def get_fret_positions(num_frets):
    """
    Calculates the relative distance of each fret from the nut (0.0) 
    to the bridge (1.0) using the standard rule of 17.817.
    """
    fret_ratios = []
    constant = 17.817
    remaining_length = 1.0  # Scale of 0 to 1
    
    current_pos_from_nut = 0
    for i in range(num_frets + 1):
        if i > 0:
            fret_ratios.append(current_pos_from_nut)
        
        # Calculate distance to next fret
        dist_to_next = remaining_length / constant
        current_pos_from_nut += dist_to_next
        remaining_length -= dist_to_next
        
    # We normalize these so the 'last fret' (21) is exactly 1.0
    # This ensures the drawing fits perfectly between your YOLO keypoints
    max_val = fret_ratios[-1]
    return [r / max_val for r in fret_ratios]

FRET_T_STEPS = get_fret_positions(NUM_FRETS)

while cap.isOpened():
    success, frame = cap.read()
    if not success: break

    results = model(frame, conf=0.5, verbose=False, stream=True)

    for r in results:
        if r.keypoints is None or len(r.keypoints.xy) == 0:
            continue

        kpts = r.keypoints.xy[0].cpu().numpy()
        confs = r.keypoints.conf[0].cpu().numpy()

        if len(kpts) >= 4 and all(c > 0.5 for c in confs[:4]):
            # Mapping: 0:L-Nut, 1:R-Nut, 2:LastFret-R, 3:LastFret-L
            lnut, rnut, lf_r, lf_l = kpts[0], kpts[1], kpts[2], kpts[3]

            # --- 2. Draw Frets (Calculated Logarithmically) ---
            for fret_t in FRET_T_STEPS:
                # Interpolate down the left side (Nut L to LastFret L)
                fx_left = lnut[0] + fret_t * (lf_l[0] - lnut[0])
                fy_left = lnut[1] + fret_t * (lf_l[1] - lnut[1])
                
                # Interpolate down the right side (Nut R to LastFret R)
                fx_right = rnut[0] + fret_t * (lf_r[0] - rnut[0])
                fy_right = rnut[1] + fret_t * (lf_r[1] - rnut[1])
                
                cv2.line(frame, (int(fx_left), int(fy_left)), 
                         (int(fx_right), int(fy_right)), (200, 200, 200), 1)

            # --- 3. Draw the 6 Inset Strings ---
            num_strings = 6
            for i in range(num_strings):
                t = STRING_INSET + (i * (1 - 2 * STRING_INSET) / (num_strings - 1)) if num_strings > 1 else 0.5

                x_start = int(lnut[0] + t * (rnut[0] - lnut[0]))
                y_start = int(lnut[1] + t * (rnut[1] - lnut[1]))
                x_end = int(lf_l[0] + t * (lf_r[0] - lf_l[0]))
                y_end = int(lf_l[1] + t * (lf_r[1] - lf_l[1]))

                cv2.line(frame, (x_start, y_start), (x_end, y_end), (255, 255, 0), 1)

            # --- 4. Draw Fretboard Boundary ---
            fret_points = np.array([lnut, rnut, lf_r, lf_l], dtype=np.int32)
            cv2.polylines(frame, [fret_points], isClosed=True, color=(255, 0, 0), thickness=2)

        else:
            cv2.putText(frame, "Searching for fretboard...", (10, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 1)

    cv2.imshow("FretVision - Strings & Frets", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()