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

def process_video(video_path):
    cap = cv2.VideoCapture(video_path)
    
    st_frame = st.empty()
    
    plate_history = defaultdict(lambda: deque(maxlen=15))
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
            
        results = model.track(frame, persist=True, tracker="botsort.yaml", verbose=False)
        
        for r in results:
            boxes = r.boxes
            for box in boxes:
                if box.id is None: continue 
                
                conf =box.conf.item()
                if conf < 0.3: continue
                
                track_id = int(box.id.item())
                x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
                
                # Crop & OCR
                plate_crop = frame
                if plate_crop.size > 0:
                    gray = cv2.cvtColor(plate_crop, cv2.COLOR_BGR2GRAY)
                    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
                    gray = clahe.apply(gray)
                    plate_resized = cv2.resize(gray, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
                    
                    ocr_results = reader.readtext(plate_resized, detail=1, allowlist='ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789')
                    text = "".join([res[1] for res in ocr_results]).strip()
                    
                    if text:
                        plate_history[track_id].append(text)
                
                stable_text = ""
                if len(plate_history) > 0:
                    current_history = plate_history[track_id]
                    stable_text = max(set(current_history), key=current_history.count)
                
                # Draw boxes
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.rectangle(frame, (x1, y1 - 30), (x2, y1), (0, 255, 0), -1)
                cv2.putText(frame, stable_text, (x1 + 5, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
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