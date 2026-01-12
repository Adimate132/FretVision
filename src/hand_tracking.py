import cv2
import os
import mediapipe as mp
from mediapipe.tasks.python.vision import (
    HandLandmarker,
    HandLandmarkerOptions,
    HandLandmarkerResult,
    RunningMode
)
from mediapipe.tasks.python.core.base_options import BaseOptions

# path to the model file, relative to this script
MODEL_PATH = os.path.join(os.path.dirname(__file__), "hand_landmarker.task")

# global variable to store latest results
latest_result = None

# callback function to store hand landmarks
def hand_landmarks_callback(result: HandLandmarkerResult, input_image: mp.Image, timestamp_ms: int):
    # store the latest detection results in a global variable
    global latest_result
    latest_result = result

# create hand landmarker options
options = HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=MODEL_PATH),  # load local .task model
    running_mode=RunningMode.LIVE_STREAM,                  # process frames continuously
    num_hands=2,                                           # detect up to 2 hands
    result_callback=hand_landmarks_callback               # attach the callback
)

# initialize the landmarker
landmarker = HandLandmarker.create_from_options(options)

# open webcam
cap = cv2.VideoCapture(0)                   # 0 = default camera
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)     # set width of camera feed
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)    # set height of camera feed

# make the opencv window resizable
cv2.namedWindow("hand tracking", cv2.WINDOW_NORMAL)

# function to draw hand landmarks and labels
def draw_landmarks(frame, result):
    # iterate through detected hands
    for hand, handedness in zip(result.hand_landmarks, result.handedness):
        # fix left/right labels for mirrored webcam
        label = handedness[0].category_name
        if label == "Left":
            label = "Right"
        else:
            label = "Left"

        # draw individual landmarks as green dots
        for lm in hand:
            x, y = int(lm.x * frame.shape[1]), int(lm.y * frame.shape[0])
            cv2.circle(frame, (x, y), 5, (0, 255, 0), -1)

        # draw connections between landmarks as blue lines
        connections = [
            (0,1),(1,2),(2,3),(3,4),
            (0,5),(5,6),(6,7),(7,8),
            (0,9),(9,10),(10,11),(11,12),
            (0,13),(13,14),(14,15),(15,16),
            (0,17),(17,18),(18,19),(19,20)
        ]
        for start, end in connections:
            x1, y1 = int(hand[start].x * frame.shape[1]), int(hand[start].y * frame.shape[0])
            x2, y2 = int(hand[end].x * frame.shape[1]), int(hand[end].y * frame.shape[0])
            cv2.line(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)

        # label the hand near the wrist (landmark 0)
        cx, cy = int(hand[0].x * frame.shape[1]), int(hand[0].y * frame.shape[0])
        cv2.putText(frame, label, (cx - 20, cy - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

# main loop to read frames from webcam
frame_index = 0
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break  # stop if camera feed fails

    # mirror the frame for natural interaction
    frame = cv2.flip(frame, 1)

    # convert opencv frame to mediapipe image
    rgb_frame = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)

    # send frame to the landmarker
    landmarker.detect_async(rgb_frame, frame_index)
    frame_index += 1

    # draw landmarks on the frame if results are available
    if latest_result:
        draw_landmarks(frame, latest_result)

    # show the frame in the resizable window
    cv2.imshow("hand tracking", frame)

    # exit if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# release webcam and close window
cap.release()
cv2.destroyAllWindows()
