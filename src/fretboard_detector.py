import cv2
from ultralytics import YOLO

# --------------------
# Load model
# --------------------
MODEL_PATH = r"data\model\pose\colab trained\weights\best.pt"
model = YOLO(MODEL_PATH)

# --------------------
# Open webcam
# --------------------
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    raise RuntimeError("Could not open webcam")

# --------------------
# Live inference loop
# --------------------
while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Run inference
    results = model(frame, conf=0.25, imgsz=640, verbose=False)

    # Draw predictions
    annotated_frame = results[0].plot()

    # Display
    cv2.imshow("YOLOv8 Pose - Live Feed", annotated_frame)

    # Exit on 'q'
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# --------------------
# Cleanup
# --------------------
cap.release()
cv2.destroyAllWindows()
