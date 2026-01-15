import cv2

def draw_hands(frame, hands_result):
    """
    Draws hand landmarks and connections on the frame.
    hands_result: HandLandmarkerResult from MediaPipe
    """
    for hand, handedness in zip(hands_result.hand_landmarks, hands_result.handedness):
        # fix left/right labels for mirrored webcam
        label = handedness[0].category_name
        if label == "Left":
            label = "Right"
        else:
            label = "Left"

        # draw landmarks as green dots
        for lm in hand:
            x, y = int(lm.x * frame.shape[1]), int(lm.y * frame.shape[0])
            cv2.circle(frame, (x, y), 5, (0, 255, 0), -1)

        # draw connections as blue lines
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
