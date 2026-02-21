import streamlit as st
import cv2
import re
import tempfile
import numpy as np
from ultralytics import YOLO
import os
os.environ["PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK"] = "True"
from paddleocr import PaddleOCR
from collections import defaultdict, deque
import paddle



st.title("🚗 ALPR: Auto License Plate Recognition App")
st.write("Upload a video to detect and recognize vehicle license plates.")

@st.cache_resource
def load_models():
    # Load your YOLO model
    model = YOLO("saved_models/license_plate_best.pt")

    # Use absolute paths to be safe
    base_path = os.path.abspath(r'D:\ALPR project\paddle_models')
    
    det_dir = os.path.join(base_path, 'en_PP-OCRv3_det_infer')
    rec_dir = os.path.join(base_path, 'en_PP-OCRv3_rec_infer')
    cls_dir = os.path.join(base_path, 'ch_ppocr_mobile_v2.0_cls_infer')

    # Verify paths exist before loading to catch errors early
    for p in [det_dir, rec_dir, cls_dir]:
        if not os.path.exists(p):
            raise FileNotFoundError(f"Model directory not found: {p}")

    gpu_available = paddle.device.is_compiled_with_cuda()
    print(f"Det path exists: {os.path.exists(os.path.join(det_dir, 'inference.pdmodel'))}")
    print(f"Rec path exists: {os.path.exists(os.path.join(rec_dir, 'inference.pdmodel'))}")
    print(f"Cls path exists: {os.path.exists(os.path.join(cls_dir, 'inference.pdmodel'))}")

    ocr_model = PaddleOCR(
      use_angle_cls=True,
        lang='en'
    )
    
    return model, ocr_model

model, ocr_reader = load_models()

if paddle.is_compiled_with_cuda():
    st.success(f"PaddleOCR device: {paddle.get_device()}")
else:
    st.warning("PaddleOCR is running on CPU. Install a CUDA-enabled PaddlePaddle build to enable GPU.")

from collections import Counter

def parse_paddle_dict_output(ocr_results, conf_thresh=0.4):
    """
    Parses Paddle doc-style output like you printed:
    [
      {
        ...,
        "rec_texts": [...],
        "rec_scores": [...],
        ...
      }
    ]
    Returns: (text, best_conf)
    """
    if not ocr_results or not isinstance(ocr_results, list):
        return "", 0.0

    text_parts = []
    best_conf = 0.0

    for item in ocr_results:
        if not isinstance(item, dict):
            continue

        rec_texts = item.get("rec_texts") or []
        rec_scores = item.get("rec_scores") or []

        # If scores missing, treat as 0.0
        if not rec_scores:
            rec_scores = [0.0] * len(rec_texts)

        for t, s in zip(rec_texts, rec_scores):
            try:
                s = float(s)
            except Exception:
                s = 0.0

            best_conf = max(best_conf, s)

            if s >= conf_thresh and t:
                clean = "".join(re.findall(r"[A-Z0-9]", str(t).upper()))
                if clean:
                    text_parts.append(clean)

    return "".join(text_parts), best_conf

def parse_paddleocr_results(ocr_results, conf_thresh=0.4):
    """
    Returns (full_text, best_conf) from PaddleOCR output.
    Handles common nesting differences across PaddleOCR versions.
    """
    if not ocr_results:
        return "", 0.0

    # Common case: ocr_results = [lines] where lines = [line1, line2, ...]
    # Sometimes it is already lines (not wrapped).
    if len(ocr_results) == 1 and isinstance(ocr_results[0], list) and ocr_results and (
        len(ocr_results[0]) == 0 or isinstance(ocr_results[0][0], (list, tuple))
    ):
        lines = ocr_results[0]
    else:
        lines = ocr_results

    text_parts = []
    best_conf = 0.0

    for line in lines:
        if not line or len(line) < 2:
            continue

        # Typical: line[1] == (text, conf)
        text_conf = line[1]

        detected_text = None
        ocr_conf = None

        if isinstance(text_conf, (tuple, list)) and len(text_conf) >= 2:
            detected_text = text_conf[0]
            ocr_conf = float(text_conf[1])
        elif isinstance(text_conf, dict):
            # Rare variant
            detected_text = text_conf.get("text")
            ocr_conf = float(text_conf.get("score", 0.0))

        if detected_text is None or ocr_conf is None:
            continue

        best_conf = max(best_conf, ocr_conf)

        if ocr_conf >= conf_thresh:
            clean = "".join(re.findall(r"[A-Z0-9]", str(detected_text).upper()))
            if clean:
                text_parts.append(clean)

    return "".join(text_parts), best_conf

def detect_international_plate(text: str) -> bool:
    if len(text) < 3:
        return False
    for ch in text:
        if not ("0" <= ch <= "9" or "A" <= ch <= "Z"):
            return False
    return True

def process_text_like_repo(text: str) -> str:
    """
    Based on ProcessText + Detect_International_LicensePlate from Alfonso Blanco's script,
    but fixed to actually slice correctly.
    """
    if not text:
        return ""

    # Keep only alnum + uppercase
    text = "".join(ch for ch in text.upper() if ch.isalnum())

    # Keep last 7-10 chars (repo code intends this, but the slicing there is buggy)
    if len(text) > 10:
        text = text[-10:]
    if len(text) > 9:
        text = text[-9:]
    if len(text) > 8:
        text = text[-8:]
    if len(text) > 7:
        text = text[-7:]

    return text if detect_international_plate(text) else ""

def extract_text_from_paddle(ocr_results) -> str:
    """
    Mimics the repo behavior: concatenate all line texts into one string.
    But returns a cleaned/validated plate string via process_text_like_repo.
    """
    if not ocr_results:
        return ""

    parts = []
    # PaddleOCR returns either [ [line, line, ...] ] or list per image; handle both.
    for block in ocr_results:
        if block is None:
            continue
        for line in block:
            if not line or len(line) < 2:
                continue
            text_score = line[1]
            if not isinstance(text_score, (list, tuple)) or len(text_score) < 1:
                continue
            raw_text = text_score[0]
            if raw_text:
                parts.append(raw_text)

    joined = "".join(parts)
    return process_text_like_repo(joined)
def enhance_plate(plate_img):
    # 1. Convert to RGB 
    plate_rgb = cv2.cvtColor(plate_img, cv2.COLOR_BGR2RGB)
    
    # 2. Resize (Paddle prefers height around 48-64px internally
    h, w = plate_rgb.shape[:2]
    if h < 100:
        scale = 150 / h
        plate_rgb = cv2.resize(plate_rgb, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    
    # 3. Contrast Enhancement (CLAHE)
    lab = cv2.cvtColor(plate_rgb, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
    cl = clahe.apply(l)
    limg = cv2.merge((cl,a,b))
    final_plate = cv2.cvtColor(limg, cv2.COLOR_LAB2RGB)
    
    # 4. Light Denoising (Optional)
    final_plate = cv2.GaussianBlur(final_plate, (3, 3), 0)
    
    return final_plate

def process_video(video_path):
    cap = cv2.VideoCapture(video_path)
    st_frame = st.empty()
    
    # Store history for EACH specific tracked vehicle
    plate_history = defaultdict(lambda: deque(maxlen=15))
    frame_count = 0
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
            
        frame_count += 1
        h, w, _ = frame.shape
            
        # YOLO tracking
        results = model.track(frame, persist=True, tracker="botsort.yaml", verbose=False)
        
        for r in results:
            boxes = r.boxes
            for box in boxes:
                if box.id is None: continue 
                
                conf = box.conf.item()
                if conf < 0.3: continue
                
                track_id = int(box.id.item())
                
                # Unpack YOLO bounding box
                x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(w, x2), min(h, y2)
                
                plate_crop = frame[y1:y2, x1:x2]
                
                if plate_crop.size > 0:
                    is_new_track = track_id not in plate_history
                    
                    if is_new_track or (frame_count % 5 == 0):
                        # Repo-style OCR: run OCR on the crop (no enhancement functions for now)
                        ocr_results = ocr_reader.ocr(plate_crop)
                        
                        full_text, best_conf = parse_paddle_dict_output(ocr_results, conf_thresh=0.4)

                        if full_text:
                            plate_history[track_id].append(full_text)
                
                # STABILIZATION & DISPLAY
                current_history = plate_history[track_id]
                if len(current_history) > 0:
                    counts = Counter(current_history)
                    stable_text, stable_count = counts.most_common(1)[0]
                    
                    # Draw Box
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    
                    # Draw Label
                    (text_w, text_h), _ = cv2.getTextSize(stable_text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
                    cv2.rectangle(frame, (x1, y1 - text_h - 10), (x1 + text_w + 10, y1), (0, 255, 0), -1)
                    cv2.putText(frame, stable_text, (x1 + 5, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                else:
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        
        st_frame.image(frame, channels="BGR")
        
    cap.release()
    st.success("Processing Complete!")
    
# File uploader
uploaded_file = st.file_uploader("Upload a video (.mp4)", type= "mp4")
if uploaded_file is not None:
    tfile = tempfile.NamedTemporaryFile(delete=False) 
    tfile.write(uploaded_file.read())
    
    if st.button("Start Processing"):
        process_video(tfile.name)