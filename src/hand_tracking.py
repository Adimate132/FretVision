#! python3.11

import cv2
import os
import mediapipe as mp
from mediapipe.tasks.python.vision import (HandLandmarker, HandLandmarkerOptions,
                                           HandLandmarkerResult, RunningMode)
from mediapipe.tasks.python.core.base_options import BaseOptions
from overlay import draw_hands

# path to the MediaPipe model file
MODEL_PATH = os.path.join(os.path.dirname(__file__), "hand_landmarker.task")

# global variable to store latest results
latest_result = None

# callback function to store hand landmarks
def hand_landmarks_callback(result: HandLandmarkerResult, input_image: mp.Image, timestamp_ms: int): 
    global latest_result
    latest_result = result

# create hand landmarker options
options = HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=MODEL_PATH),
    running_mode=RunningMode.LIVE_STREAM,
    num_hands=2,
    result_callback=hand_landmarks_callback
)

# initialize the landmarker
landmarker = HandLandmarker.create_from_options(options)

# open webcam
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

# make the OpenCV window resizable
cv2.namedWindow("hand tracking", cv2.WINDOW_NORMAL)

# main loop to read frames from webcam
frame_index = 0
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # mirror the frame for natural interaction
    frame = cv2.flip(frame, 1)

    # convert OpenCV frame to MediaPipe image
    rgb_frame = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)

    # send frame to the landmarker
    landmarker.detect_async(rgb_frame, frame_index)
    frame_index += 1

    # draw hands if results are available
    if latest_result:
        draw_hands(frame, latest_result)

    # show the frame
    cv2.imshow("hand tracking", frame)

    # exit if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# release webcam and close window
cap.release()
cv2.destroyAllWindows()
