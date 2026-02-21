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
                        enhanced_plate = enhance_plate(plate_crop)
                        
                        ocr_results = ocr_reader.ocr(enhanced_plate)
                        
                        text_parts = []
                        # Check if results exist and are not None
                        if ocr_results and ocr_results[0] is not None:
                            for res in ocr_results:
                                if res is None: continue # Skip if specific result is None
                                for line in res:
                                    # line structure is normally: [ [box_coords], (text, confidence) ]
                                    
                                    # 1. Safety Check: Ensure line has 2 elements and the second is a tuple/list
                                    if len(line) >= 2 and isinstance(line[1], (list, tuple)):
                                        text_data = line[1]
                                        
                                        # 2. Unpack Safety: Ensure we have exactly (Text, Score)
                                        if len(text_data) == 2:
                                            detected_text, ocr_conf = text_data
                                            
                                            # 3. Logic Fix: This check must be INSIDE the loop
                                            if ocr_conf > 0.4:
                                                # Clean non-alphanumeric
                                                clean_part = "".join(re.findall(r'[A-Z0-9]', detected_text.upper()))
                                                text_parts.append(clean_part)
                                    else:
                                        # Handle malformed Paddle output gracefully
                                        continue
                        else:
                            # Optional: Handle cases where OCR returned nothing
                            pass
                        
                        full_text = "".join(text_parts)
                        if full_text:
                            plate_history[track_id].append(full_text)
                
                # STABILIZATION & DISPLAY
                current_history = plate_history[track_id]
                if len(current_history) > 0:
                    if len(current_history) < 3:
                        stable_text = current_history[-1]
                    else:
                        # Find the most common text (mode)
                        stable_text = max(set(current_history), key=current_history.count)
                    
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