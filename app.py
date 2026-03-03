import os
import re
import tempfile
import numpy as np
from collections import defaultdict, deque , Counter

import cv2
import streamlit as st
from ultralytics import YOLO
os.environ["PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK"] = "True"
from paddleocr import PaddleOCR
import paddle


# ---------------------------------------------------------------------
# App UI
# ---------------------------------------------------------------------
st.title("ALPR: Auto License Plate Recognition App")
st.write("Upload a video to detect and recognize vehicle license plates.")

# ---------------------------------------------------------------------
# Tunable parameters
# ---------------------------------------------------------------------
YOLO_WEIGHTS = "saved_models/license_plate_best.pt"
TRACKER_CFG = "botsort.yaml"

YOLO_CONF_THRESH = 0.30         # conf threshold for YOLO detections 
OCR_EVERY_N_FRAMES = 2          # OCR frequency per track
HISTORY_LEN = 15                # temporal voting window
OCR_CONF_THRESH = 0.20          # conf threshold for accepting OCR text candidates
# ---------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------
os.environ["PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK"] = "True"

@st.cache_resource
def load_models():
    model = YOLO(YOLO_WEIGHTS)

    ocr_model = PaddleOCR(use_angle_cls=True,lang="en")
    return model, ocr_model

model, ocr_reader = load_models()

if paddle.is_compiled_with_cuda():
    st.success(f"PaddleOCR device: {paddle.get_device()}")
else:
    st.warning("PaddleOCR is running on CPU. Install a CUDA-enabled PaddlePaddle build to enable GPU.")

# ---------------------------------------------------------------------
# Text validation/cleanup (adapted + fixed slicing)
# ---------------------------------------------------------------------
def detect_international_plate(text: str) -> bool:
    if len(text) < 3:
        return False
    for ch in text:
        if not ("0" <= ch <= "9" or "A" <= ch <= "Z"):
            return False
    return True

def process_text(text: str) -> str:
    if not text:
        return ""

    text = "".join(ch for ch in text.upper() if ch.isalnum())

    # Keep last 7..10 characters 
    if len(text) > 10:
        text = text[-10:]

    return text if detect_international_plate(text) else ""

# ---------------------------------------------------------------------
# PaddleOCR parsing for your dict-style output
# ---------------------------------------------------------------------
def parse_paddle_dict_output(ocr_results, conf_thresh: float) -> tuple[str, float]:
    """
    Your output format:
    [
      {
        ...,
        "rec_texts": ["NNNEMAN"],
        "rec_scores": [0.3593],
        ...
      }
    ]
    Returns: (cleaned_text, best_conf)
    """
    if not ocr_results or not isinstance(ocr_results, list):
        return "", 0.0

    best_conf = 0.0
    collected = []

    for item in ocr_results:
        if not isinstance(item, dict):
            continue

        rec_texts = item.get("rec_texts") or []
        rec_scores = item.get("rec_scores") or []

        if not rec_scores:
            rec_scores = [0.0] * len(rec_texts)

        for t, s in zip(rec_texts, rec_scores):
            try:
                s = float(s)
            except Exception:
                s = 0.0

            best_conf = max(best_conf, s)

            if t and s >= conf_thresh:
                
                cleaned = "".join(re.findall(r"[A-Z0-9]", str(t).upper()))
                cleaned = process_text(cleaned)
                if cleaned:
                    collected.append(cleaned)

    return "".join(collected), best_conf


# ---------------------------------------------------------------------
# Enhancements to the cropped plate image before OCR
# ---------------------------------------------------------------------
def crop_with_padding(frame, x1, y1, x2, y2, pad=0.20):
    h, w = frame.shape[:2]
    bw, bh = (x2 - x1), (y2 - y1)
    px, py = int(bw * pad), int(bh * pad)
    x1p, y1p = max(0, x1 - px), max(0, y1 - py)
    x2p, y2p = min(w, x2 + px), min(h, y2 + py)
    return frame[y1p:y2p, x1p:x2p]

def resize_plate_for_ocr(img_bgr, target_h=64):
    h, w = img_bgr.shape[:2]
    if h == 0: 
        return img_bgr
    scale = target_h / h
    new_w = max(1, int(w * scale))
    return cv2.resize(img_bgr, (new_w, target_h), interpolation=cv2.INTER_CUBIC)

def clahe_bgr(img_bgr):
    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l2 = clahe.apply(l)
    merged = cv2.merge((l2, a, b))
    return cv2.cvtColor(merged, cv2.COLOR_LAB2BGR)


def deskew_plate(bgr):
    """
    Rotates the crop so the dominant rectangular plate region becomes horizontal.
    Works best after resize (so edges are clearer).
    """
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)

    edges = cv2.Canny(gray, 50, 150)

    # Find contours on edges
    cnts, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return bgr

    # Use the largest contour as a proxy for plate boundary
    c = max(cnts, key=cv2.contourArea)
    if cv2.contourArea(c) < 50:  # too small / noisy
        return bgr

    rect = cv2.minAreaRect(c)  # ((cx,cy),(w,h),angle)
    angle = rect[-1]

    # minAreaRect angle conventions are weird:
    # angle is in [-90, 0). We want a small rotation to horizontal.
    if angle < -45:
        angle = angle + 90

    # Only rotate if it’s a meaningful tilt
    if abs(angle) < 2.0:
        return bgr

    (h, w) = bgr.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(bgr, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    return rotated




def order_points(pts):
    # pts: (4,2)
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]  # top-left
    rect[2] = pts[np.argmax(s)]  # bottom-right
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]  # top-right
    rect[3] = pts[np.argmax(diff)]  # bottom-left
    return rect

def rectify_plate_perspective(bgr):
    """
    Attempts to find a 4-corner contour and warp it to a rectangle.
    If it can’t find a good quad, returns original.
    """
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.bilateralFilter(gray, 7, 50, 50)
    edges = cv2.Canny(gray, 50, 150)

    cnts, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return bgr

    cnts = sorted(cnts, key=cv2.contourArea, reverse=True)[:10]

    quad = None
    for c in cnts:
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)
        if len(approx) == 4 and cv2.contourArea(approx) > 200:
            quad = approx
            break

    if quad is None:
        return bgr

    pts = quad.reshape(4, 2).astype("float32")
    rect = order_points(pts)
    (tl, tr, br, bl) = rect

    widthA = np.linalg.norm(br - bl)
    widthB = np.linalg.norm(tr - tl)
    maxW = int(max(widthA, widthB))

    heightA = np.linalg.norm(tr - br)
    heightB = np.linalg.norm(tl - bl)
    maxH = int(max(heightA, heightB))

    if maxW < 20 or maxH < 20:
        return bgr

    dst = np.array([
        [0, 0],
        [maxW - 1, 0],
        [maxW - 1, maxH - 1],
        [0, maxH - 1]
    ], dtype="float32")

    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(bgr, M, (maxW, maxH), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    return warped

# ---------------------------------------------------------------------
# Video processing
# ---------------------------------------------------------------------
def process_video(video_path: str):
    """
    Processes a video file to detect, track, and recognize license plates frame by frame.
    Args:
        video_path (str): Path to the input video file.
    Workflow:
        1. Initializes video capture and a Streamlit frame placeholder for displaying results.
        2. Sets up a history buffer for each detected plate track to stabilize OCR results over time.
        3. Iterates through each frame of the video:
            a. Reads the next frame; stops if the video ends.
            b. Runs object detection and tracking on the frame to find license plates.
            c. For each detected plate:
                - Skips if detection confidence is too low or if no track ID is assigned.
                - Crops the plate region from the frame, applying padding and boundary checks.
                - Periodically (or for new tracks), preprocesses the plate image (resize, rectify, deskew, enhance) for OCR.
                - Runs OCR to extract text from the plate, parsing results and filtering by confidence.
                - Updates the plate's history buffer with recognized text.
                - Selects the most frequent (stable) text from the history buffer for display.
                - Draws a bounding box and overlays the stable plate text on the frame.
            d. Displays the processed frame in the Streamlit app.
        4. Releases video resources and notifies the user upon completion.
    Rationale for Order:
        - Detection and tracking are performed first to localize and associate plates across frames.
        - Cropping and preprocessing are done before OCR to maximize recognition accuracy.
        - OCR is run selectively to save computation, using history to stabilize results and reduce flicker.
        - Drawing and display are done last to provide real-time visual feedback to the user.
    Returns:
        None. The function displays processed frames in a Streamlit interface and shows a success message when done.
    """
    cap = cv2.VideoCapture(video_path)
    st_frame = st.empty()

    plate_history = defaultdict(lambda: deque(maxlen=HISTORY_LEN))
    frame_count = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        h, w, _ = frame.shape

        results = model.track(frame, persist=True, tracker=TRACKER_CFG, verbose=False)

        for r in results:
            for box in r.boxes:
                if box.id is None:
                    continue

                conf = float(box.conf.item())
                if conf < YOLO_CONF_THRESH:
                    continue

                track_id = int(box.id.item())

                x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(w, x2), min(h, y2)

                plate_crop = crop_with_padding(frame, x1, y1, x2, y2)
                if plate_crop.size == 0:
                    continue

                # Run OCR only occasionally per track to save time
                is_new_track = track_id not in plate_history
                if is_new_track or (frame_count % OCR_EVERY_N_FRAMES == 0):
                    # Paddle expects RGB when given numpy arrays
                    
                    plate_crop = resize_plate_for_ocr(plate_crop)
                    plate = rectify_plate_perspective(plate_crop)
                    plate = resize_plate_for_ocr(plate, target_h=64)
                    plate_crop = deskew_plate(plate_crop)
                    plate_crop = clahe_bgr(plate_crop)
                    plate_rgb = cv2.cvtColor(plate_crop, cv2.COLOR_BGR2RGB)

                    # Use cls=True for angle classification
                    ocr_results = ocr_reader.ocr(plate_rgb)

                    full_text, best_conf = parse_paddle_dict_output(
                        ocr_results, conf_thresh=OCR_CONF_THRESH
                    )

                    if full_text:
                        plate_history[track_id].append(full_text)

                # Choose stable text (temporal vote)
                stable_text = ""
                history = plate_history[track_id]
                if history:
                    stable_text = Counter(history).most_common(1)[0][0]

                # Draw
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                if stable_text:
                    (text_w, text_h), _ = cv2.getTextSize(
                        stable_text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2
                    )
                    cv2.rectangle(
                        frame,
                        (x1, y1 - text_h - 10),
                        (x1 + text_w + 10, y1),
                        (0, 255, 0),
                        -1,
                    )
                    cv2.putText(frame,stable_text,
                        (x1 + 5, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (255, 255, 255),
                        2,
                    )

        st_frame.image(frame, channels="BGR")

    cap.release()
    st.success("Processing Complete!")

    
# ---------------------------------------------------------------------
# Streamlit file uploader
# ---------------------------------------------------------------------
uploaded_file = st.file_uploader("Upload a video (.mp4)", type="mp4")
if uploaded_file is not None:
    tfile = tempfile.NamedTemporaryFile(delete=False)
    tfile.write(uploaded_file.read())

    if st.button("Start Processing"):
        process_video(tfile.name)