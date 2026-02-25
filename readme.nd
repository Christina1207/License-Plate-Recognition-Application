# 🚗 Automatic License Plate Recognition (ALPR) – Streamlit App

## Goal of this project
This project builds an **Automatic License Plate Recognition (ALPR)** application that:
1. **Detects license plates in a video** using a trained **YOLO** model.
2. **Tracks detected plates across frames** (so the same vehicle/plate keeps a consistent ID).
3. **Reads the license plate characters (OCR)** using **PaddleOCR**.
4. Produces a **more stable final plate text** by combining OCR results across multiple frames (temporal voting).
5. Displays the processed video in a **Streamlit web app** with bounding boxes and the recognized plate text.

---

## What the application does (high-level flow)
1. User uploads an **.mp4 video** in the Streamlit UI.
2. Each video frame is processed:
   - Run **YOLO plate detection + tracking** (`model.track(...)`).
   - For each tracked plate:
     - Crop the plate region (with padding).
     - Enhance the crop for OCR (resize + contrast + deskew/rectify attempts).
     - Run **PaddleOCR** on the crop every N frames (to reduce computation).
     - Clean/validate OCR text and store it in a per-track history buffer.
   - Pick the most common OCR result for each track (temporal vote).
   - Draw bounding box + stable recognized text onto the frame.
3. Streamlit displays the annotated frames live until processing finishes.

---

## Tech stack
- **Python**
- **Streamlit** (web UI)
- **Ultralytics YOLO** (license plate detection + tracking)
- **OpenCV (cv2)** (video IO, cropping, image processing, drawing)
- **PaddleOCR** (text recognition)
- **NumPy** (image and geometry operations)

---

## Project structure (key files)
- `app.py` — Streamlit application containing the full ALPR pipeline (detection, tracking, OCR, visualization).
- `saved_models/license_plate_best.pt` — YOLO weights file (must exist locally for detection to work).
- `botsort.yaml` — tracker configuration file used by Ultralytics tracking.

---

## Steps of the work done (pipeline details)

### 1) Streamlit interface
- The UI provides:
  - A title and description.
  - A video uploader (`st.file_uploader`) restricted to `.mp4`.
  - A button to start processing.

### 2) Tunable parameters (configuration)
The app defines a few important parameters you can adjust:
- `YOLO_CONF_THRESH` — minimum detection confidence.
- `OCR_EVERY_N_FRAMES` — how often OCR runs per track (performance vs accuracy tradeoff).
- `HISTORY_LEN` — number of OCR results stored per tracked plate for voting.
- `OCR_CONF_THRESH` — minimum OCR confidence to accept recognized text.

### 3) Model loading with caching
- Models are loaded once using `@st.cache_resource`:
  - YOLO detector/tracker model from `saved_models/license_plate_best.pt`
  - PaddleOCR model (`PaddleOCR(use_angle_cls=True, lang="en")`)
- The app also reports whether PaddleOCR is running on **GPU** or **CPU**.

### 4) Plate text cleaning + validation
To reduce OCR noise, text is:
- Uppercased and restricted to **A–Z and 0–9**
- Trimmed to a short maximum length (intended to keep only the meaningful tail of long strings)
- Validated to ensure it looks like a plausible “international plate” format (alphanumeric only)

### 5) Parsing PaddleOCR output
The OCR parsing function is designed for PaddleOCR dictionary-style output:
- Reads `rec_texts` and `rec_scores`
- Filters by `OCR_CONF_THRESH`
- Cleans and validates the text
- Returns:
  - the cleaned plate text
  - the best confidence seen for that crop

### 6) Plate image pre-processing (to improve OCR)
Before OCR, the cropped plate image is improved using:
- **Padding-based crop** to avoid cutting off characters
- **Resizing** to a consistent height for OCR
- **Perspective rectification** (attempt to warp plate region into a rectangle)
- **Deskewing** (rotate plate region so text is horizontal)
- **CLAHE** contrast enhancement to improve readability

### 7) Video processing + tracking + temporal voting
For each frame:
- YOLO detection + tracking is run using `model.track(... tracker="botsort.yaml")`
- Each plate gets a `track_id`
- OCR results are stored per track in a fixed-size history buffer (`deque(maxlen=HISTORY_LEN)`)
- The displayed plate text is chosen by **majority vote** (`Counter(history).most_common(1)`), making output more stable than single-frame OCR

### 8) Output visualization
- The app draws:
  - green bounding boxes around plates
  - a filled label background and the stabilized OCR text
- Streamlit shows the annotated frames as the video is processed.

---

## How to run the app

### 1) Install dependencies
Create a virtual environment (recommended), then install dependencies:

```bash
pip install streamlit opencv-python ultralytics paddleocr paddlepaddle numpy