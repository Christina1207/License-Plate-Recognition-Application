import streamlit as st
import cv2
import tempfile
import numpy as np
from ultralytics import YOLO
import easyocr
from collections import defaultdict, deque

st.title("🚗 ALPR: Auto License Plate Recognition App")
st.write("Upload a video to detect and recognize vehicle license plates.")

@st.cache_resource
def load_models():
    # Load your fine-tuned model
    model = YOLO("saved_models/license_plate_best.pt")
    reader = easyocr.Reader(['en'], gpu=True,
        model_storage_directory='D:/ALPR project/easyocr_models', # Point to your local folder
        download_enabled=False # Tell it NOT to go to the internet
    )
    return model, reader

model, reader = load_models()

def enhance_plate(plate_img):
    # 1. Convert to Gray
    gray = cv2.cvtColor(plate_img, cv2.COLOR_BGR2GRAY)
    
    # 2. Upscale (EasyOCR loves big characters)
    # Target height of ~100-150 pixels for the plate
    scale_factor = 2 if gray.shape[0] < 100 else 1
    upscaled = cv2.resize(gray, None, fx=scale_factor, fy=scale_factor, interpolation=cv2.INTER_CUBIC)
    
    # 3. Bilateral Filter (Removes noise but keeps edges sharp)
    denoised = cv2.bilateralFilter(upscaled, 11, 17, 17)
    
    # 4. Adaptive Thresholding (Handles shadows/uneven lighting)
    thresh = cv2.adaptiveThreshold(denoised, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                   cv2.THRESH_BINARY, 11, 2)
    return thresh

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
                
                # FIX 1: Correctly unpack YOLO bounding box (it's a 2D array, we need the first element)
                x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
                
                # Prevent out-of-bounds crashes
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(w, x2), min(h, y2)
                
                # FIX 2: Actually CROP the image to only the license plate
                plate_crop = frame[y1:y2, x1:x2]
                
                if plate_crop.size > 0:
                    # SPEED BOOST: Only run OCR every 5th frame
                    if frame_count % 5 == 0:
                        enhanced_plate = enhance_plate(plate_crop)
                        plate_resized = cv2.resize(enhanced_plate, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
                        
                        ocr_results = reader.readtext(plate_resized, detail=1, allowlist='ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-')
                        
                        # FIX 3: Correctly unpack EasyOCR results (bbox, text, confidence)
                        text = ""
                        for bbox, detected_text, ocr_confidence in ocr_results:
                            if ocr_confidence > 0.3: 
                                text += detected_text
                                
                        text = text.strip()
                        if text:
                            # FIX 4: Save text to the specific track_id, not the dictionary itself
                            plate_history[track_id].append(text)
                
                # Get the stabilized text for THIS specific track ID
                stable_text = ""
                current_history = plate_history[track_id]
                
                if len(current_history) > 0: 
                    # Get the most frequently read text (mode) for this tracked car
                    stable_text = max(set(current_history), key=current_history.count)
                
                # Draw boxes
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                
                # Only draw the background and text if we actually have text
                if stable_text:
                    (text_w, text_h), _ = cv2.getTextSize(stable_text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
                    cv2.rectangle(frame, (x1, y1 - text_h - 10), (x1 + text_w + 10, y1), (0, 255, 0), -1)
                    cv2.putText(frame, stable_text, (x1 + 5, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Display the frame in the Streamlit web app
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