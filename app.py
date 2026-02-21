import os
import re
import tempfile
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
st.title("🚗 ALPR: Auto License Plate Recognition App")
st.write("Upload a video to detect and recognize vehicle license plates.")

# ---------------------------------------------------------------------
# Tunable parameters
# ---------------------------------------------------------------------
YOLO_WEIGHTS = "saved_models/license_plate_best.pt"
TRACKER_CFG = "botsort.yaml"

YOLO_CONF_THRESH = 0.30
OCR_EVERY_N_FRAMES = 5          # OCR frequency per track
HISTORY_LEN = 15                # temporal voting window
OCR_CONF_THRESH = 0.20          # your sample showed ~0.36; start lower than 0.4
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
# Repo-style text validation/cleanup (adapted + fixed slicing)
# ---------------------------------------------------------------------
def detect_international_plate(text: str) -> bool:
    if len(text) < 3:
        return False
    for ch in text:
        if not ("0" <= ch <= "9" or "A" <= ch <= "Z"):
            return False
    return True

def process_text_like_repo(text: str) -> str:
    if not text:
        return ""

    text = "".join(ch for ch in text.upper() if ch.isalnum())

    # Keep last 7..10 characters (repo intent)
    if len(text) > 10:
        text = text[-10:]
    if len(text) > 9:
        text = text[-9:]
    if len(text) > 8:
        text = text[-8:]
    if len(text) > 7:
        text = text[-7:]

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
                # Clean to A-Z0-9 first, then apply repo-ish final validation
                cleaned = "".join(re.findall(r"[A-Z0-9]", str(t).upper()))
                cleaned = process_text_like_repo(cleaned)
                if cleaned:
                    collected.append(cleaned)

    # Repo behavior: concatenate text parts; for plates usually only one anyway
    return "".join(collected), best_conf

# ---------------------------------------------------------------------
# Video processing
# ---------------------------------------------------------------------
def process_video(video_path: str):
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

                plate_crop = frame[y1:y2, x1:x2]
                if plate_crop.size == 0:
                    continue

                # Run OCR only occasionally per track to save time
                is_new_track = track_id not in plate_history
                if is_new_track or (frame_count % OCR_EVERY_N_FRAMES == 0):
                    # Paddle expects RGB when given numpy arrays
                    plate_rgb = cv2.cvtColor(plate_crop, cv2.COLOR_BGR2RGB)

                    # Use cls=True to match repo behavior (angle classification)
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
                    cv2.putText(
                        frame,
                        stable_text,
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