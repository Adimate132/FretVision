# Introduction
![Image of fretboard detection](./repo-images/coverimg.jpg)
FretVision is a work-in-progress computer vision project that uses a standard webcam to track a guitarist’s fretting hand and infer chord shapes in real time.

The system is vision-only and does not rely on audio input. It uses Python, OpenCV, and MediaPipe to detect hand landmarks, track finger positions, and map those positions onto a guitar fretboard.

The long-term goal of this project is to reliably identify guitar chords on acoustic, classical, or electric guitars using only visual information. Furthermore, adding interactive overlays onto the fretboard could potentially be explored. This repository currently focuses on achieving stable, low-latency hand landmark detection as a foundation for future stages such as finger state classification, fretboard mapping, and chord inference.

In the future, audio and visual information could be combined, enabling polyphonic audio discernment to fine-tune chord detection and improve accuracy.

# Installation
1. Install Python 3.11 (64-bit) from python.org
Make sure to check `“Add Python to PATH”` during installation.
Verify installation:
`py -3.11 --version`

2. Clone the repository 
cd into FretVision directory

3. Install dependencies:
	`py -3.11 -m pip install --upgrade pip`
	`py -3.11 -m pip install -r requirements.txt`

4. Verify installation:
`py -3.11 -c "import cv2, mediapipe, numpy; print(cv2.__version__, mediapipe.__version__, numpy.__version__)"`

5. Run the hand tracking demo:
`py src/hand_tracking.py`

`Note: If any version issues occur, run w/ python3.11 explicitly:`
`py -3.11 src/hand_tracking.py`

A window should pop up utilizing your camera.
Press q to quit the program.

# Requirements
* Python 3.11 (64-bit)
* Webcam
* A 6 string acoustic/classical/electric guitar

## Dataset & Labeling
- Images were labeled using **Supervisely**
- Export format: **YOLOv8 segmentation**
- Only labeled images were used for training
- Unlabeled images were excluded to avoid background bias

Dataset paths are defined in `data.yaml`.

## Training
Segmentation training was run with:

`yolo task=segment mode=train model=yolov8n-seg.pt data=data.yaml epochs=50 imgsz=960 batch=8`

Training outputs are generated under `runs/segment/train/` and are not committed.

## Inference (Single Image)
Run segmentation on a single image:

`yolo task=segment mode=predict model=runs\segment\train\weights\best.pt source=data\frames\pathToImg conf=0.3 save=True show=True`

For batch inference, `source` may be a folder instead of a single image.

## Model Weights
Trained weights are stored in:
`models/`
  
# Notes
- Python 3.11 is required for MediaPipe stability on Windows.
- If multiple Python versions are installed, always use py -3.11 to avoid version mismatches.

- The model may output multiple detections for a single fretboard; this is handled via confidence filtering and post-processing.
- String positions are inferred mathematically from the fretboard geometry rather than being explicitly labeled.

## Git Ignore
Training data, raw videos, YOLO runs, and caches are excluded by design.  
See `.gitignore` for details.

# Next steps
This repo is a WIP and so this readme will be updated accordingly.

**TO DO (rough list):**
* Calculate fret positions
* Detect fingers on the fretboard (w/ mediapipe or custom detection)
* Map fingers to their corresponding fret and string
* Infer chords from finger positions 