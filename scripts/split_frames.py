#! python3.11
import os
import shutil
import random

# -----------------------------
# CONFIG
# -----------------------------
FRAMES_DIR = "data/frames"   # folder with all frames
BATCH_SIZE = 200             # frames per batch
OUT_DIR = FRAMES_DIR         # output parent folder (will create batch subfolders)

# -----------------------------
# Collect frames
# -----------------------------
all_frames = [f for f in os.listdir(FRAMES_DIR)
              if f.lower().endswith(('.jpg', '.png')) and os.path.isfile(os.path.join(FRAMES_DIR, f))]

print(f"Found {len(all_frames)} frames.")

# Shuffle frames
random.shuffle(all_frames)

# -----------------------------
# Create batches
# -----------------------------
batch_num = 1
for i in range(0, len(all_frames), BATCH_SIZE):
    batch_frames = all_frames[i:i+BATCH_SIZE]
    batch_folder = os.path.join(OUT_DIR, f"batch{batch_num}")
    os.makedirs(batch_folder, exist_ok=True)

    for frame in batch_frames:
        src_path = os.path.join(FRAMES_DIR, frame)
        dst_path = os.path.join(batch_folder, frame)
        shutil.move(src_path, dst_path)  # move frames into batch folder

    print(f"Batch {batch_num} created with {len(batch_frames)} frames.")
    batch_num += 1

print("All frames split into batches!")
