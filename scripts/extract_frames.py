import cv2
import os

# -----------------------------
# CONFIGURATION
# -----------------------------
VIDEO_DIR = "data/raw_videos"  # folder containing your guitar videos
OUT_DIR = "data/frames"        # folder to save extracted frames
FPS = 30                        # number of frames per second to extract

# -----------------------------
# PREPARE FOLDERS
# -----------------------------
# create folders if they don't exist
os.makedirs(VIDEO_DIR, exist_ok=True)
os.makedirs(OUT_DIR, exist_ok=True)

# -----------------------------
# CHECK FOR VIDEOS
# -----------------------------
video_files = [f for f in os.listdir(VIDEO_DIR) if f.lower().endswith((".mp4", ".mov", ".avi", ".mkv"))]
if not video_files:
    print(f"No video files found in {VIDEO_DIR}. Please add MP4/MOV/AVI/MKV files and rerun.")
    exit()

# -----------------------------
# PROCESS VIDEOS
# -----------------------------
for video_file in video_files:
    video_path = os.path.join(VIDEO_DIR, video_file)
    print(f"Processing {video_file}...")
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print(f"Failed to open {video_file}, skipping.")
        continue

    video_fps = cap.get(cv2.CAP_PROP_FPS)
    step = max(int(video_fps / FPS), 1)

    frame_index = 0
    saved_frame_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_index % step == 0:
            # create a unique frame filename
            frame_filename = f"{os.path.splitext(video_file)[0]}_{saved_frame_count:04d}.jpg"
            frame_path = os.path.join(OUT_DIR, frame_filename)
            cv2.imwrite(frame_path, frame)
            saved_frame_count += 1

        frame_index += 1

    cap.release()
    print(f"Saved {saved_frame_count} frames from {video_file}.")

print("All videos processed!")
