import cv2
import mediapipe as mp
import time
import numpy as np

# New Tasks API Imports
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

class SmoothHandTracker:
    def __init__(self, model_path='hand_landmarker.task', num_hands=2, min_con=0.5, smoothing=0.25):
        """
        Initializes the Hand Landmarker using the new Tasks API.
        :param model_path: Path to your .task file
        :param smoothing: EMA alpha (lower is smoother)
        """
        self.smoothing_factor = smoothing
        self.previous_landmarks = None
        
        # Configure Task Options
        base_options = python.BaseOptions(model_asset_path=model_path)
        options = vision.HandLandmarkerOptions(
            base_options=base_options,
            running_mode=vision.RunningMode.VIDEO, # Efficient for webcam
            num_hands=num_hands,
            min_hand_detection_confidence=min_con,
            min_hand_presence_confidence=min_con,
            min_tracking_confidence=min_con
        )
        self.detector = vision.HandLandmarker.create_from_options(options)

    def find_hands(self, frame):
        # Convert OpenCV BGR to MediaPipe Image object
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)
        
        # New API requires a timestamp in milliseconds
        timestamp_ms = int(time.time() * 1000)
        result = self.detector.detect_for_video(mp_image, timestamp_ms)
        
        if result.hand_landmarks:
            for hand_idx, hand_lms in enumerate(result.hand_landmarks):
                # Apply smoothing
                if self.previous_landmarks is not None and len(self.previous_landmarks) > hand_idx:
                    for i, lm in enumerate(hand_lms):
                        prev = self.previous_landmarks[hand_idx][i]
                        lm.x = prev.x + self.smoothing_factor * (lm.x - prev.x)
                        lm.y = prev.y + self.smoothing_factor * (lm.y - prev.y)
                        lm.z = prev.z + self.smoothing_factor * (lm.z - prev.z)
                
                # Draw Landmarks (Modern way)
                self._draw_on_frame(frame, hand_lms)
            
            self.previous_landmarks = result.hand_landmarks
        return frame

    def _draw_on_frame(self, frame, landmarks):
        h, w, _ = frame.shape
        
        connections = [
            (0, 1), (1, 2), (2, 3), (3, 4), (0, 5), (5, 6), (6, 7), (7, 8),
            (0, 9), (9, 10), (10, 11), (11, 12), (0, 13), (13, 14), (14, 15), (15, 16),
            (0, 17), (17, 18), (18, 19), (19, 20), (5, 9), (9, 13), (13, 17)
        ]

        # 1. Draw connections (lines remain constant thickness for clarity)
        for start_idx, end_idx in connections:
            p1 = (int(landmarks[start_idx].x * w), int(landmarks[start_idx].y * h))
            p2 = (int(landmarks[end_idx].x * w), int(landmarks[end_idx].y * h))
            cv2.line(frame, p1, p2, (0, 255, 0), 2)

        # 2. Draw landmarks with Dynamic Scaling
        for lm in landmarks:
            cx, cy = int(lm.x * w), int(lm.y * h)
            
            # DEPTH MATH: 
            # We take the Z value (usually between -0.1 and 0.1)
            # We scale it so closer points (negative Z) are larger.
            # Base size 5, minus the Z value scaled by 40.
            # Use max() to ensure radius never drops below 1.
            dynamic_radius = max(1, int(6 - (lm.z * 40)))

            # Draw the point
            cv2.circle(frame, (cx, cy), dynamic_radius, (255, 255, 255), cv2.FILLED)
            cv2.circle(frame, (cx, cy), dynamic_radius, (0, 255, 0), 1)

def main():
    cap = cv2.VideoCapture(0)
    cap.set(3, 640) # Efficiency: lower res
    cap.set(4, 480)
    
    # Ensure the model_path matches where you saved the .task file
    tracker = SmoothHandTracker(model_path='hand_landmarker.task', smoothing=0.2)
    
    while cap.isOpened():
        success, frame = cap.read()
        if not success: break
        
        frame = cv2.flip(frame, 1)
        frame = tracker.find_hands(frame)
        
        cv2.imshow("Tasks API - Smooth Hand Tracking", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()