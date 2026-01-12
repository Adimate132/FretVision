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
* A 6 string acoustic/classical guitar
  
 # Notes
Python 3.11 is required for MediaPipe stability on Windows.

If multiple Python versions are installed, always use py -3.11 to avoid version mismatches.

# Next steps
This repo is a WIP and so this readme will be updated accordingly.

**TO DO:**
* Landmark normalization
* Finger state detection
* Fretboard mapping