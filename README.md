# FretVision

Vision-only guitar chord detection using Python, OpenCV, and MediaPipe.

## Quick Start (Windows, Python 3.11)

> These steps create an isolated virtual environment so dependencies work reliably on Windows.

### 1. Install Python

* Install **Python 3.11 (64-bit)** from python.org
* Check **“Add Python to PATH”** during install

### 2. Clone the repo

```bash
git clone <your-repo-url>
cd FretVision
```

### 3. Create a virtual environment

```bash
py -3.11 -m venv venv
```

### 4. Activate the virtual environment

```bash
venv\Scripts\activate
```

You should see `(venv)` at the start of your terminal prompt.

### 5. Install dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### 6. Verify installation

```bash
python -c "import cv2, mediapipe, numpy; print(cv2.__version__, mediapipe.__version__, numpy.__version__)"
```

Expected output:

```
4.9.0 0.10.31 1.26.4
```

### 7. Run hand tracking demo

```bash
python src/hand_tracking.py
```

Press **ESC** to quit.

---

## Requirements

* Python 3.11 (64-bit)
* Webcam

## Notes

* Do **not** commit the `venv/` folder to git
* If you have a newer system Python installed, the virtual environment ensures FretVision still works

---

Next steps: landmark normalization, finger state detection, fretboard mapping.
