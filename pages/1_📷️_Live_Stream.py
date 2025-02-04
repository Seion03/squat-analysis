import av
import os
import sys
import cv2
import streamlit as st
import numpy as np
import threading
import time
from utils import get_mediapipe_pose
from process_frame import ProcessFrame
from thresholds import get_thresholds_beginner, get_thresholds_pro

st.title('AI Fitness Trainer: Squats Analysis')

mode = st.radio('Select Mode', ['Beginner', 'Pro'], horizontal=True)

thresholds = get_thresholds_beginner() if mode == 'Beginner' else get_thresholds_pro()

live_process_frame = ProcessFrame(thresholds=thresholds, flip_frame=True)
pose = get_mediapipe_pose()

# RTMP stream URL from NGINX server (replace with your actual RTMP URL)
RTMP_URL = "rtmp://localhost/live/stream"

# OpenCV VideoCapture to read from RTMP stream
cap = cv2.VideoCapture(RTMP_URL)
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Reduce latency by setting buffer size to 1

# Video writer setup for saving processed frames
output_video_file = "output_live.mp4"
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_video_file, fourcc, 20.0, (640, 480))

stframe = st.empty()
if 'download' not in st.session_state:
    st.session_state['download'] = False

frame_lock = threading.Lock()
latest_frame = None

def read_frames():
    global latest_frame
    while cap.isOpened():
        cap.grab()  # Skip old frames for lower latency
        ret, frame = cap.retrieve()
        if ret:
            with frame_lock:
                latest_frame = frame
        time.sleep(0.01)  # Prevent excessive CPU usage

# Start frame capture in a separate thread
threading.Thread(target=read_frames, daemon=True).start()

frame_counter = 0
while True:
    with frame_lock:
        if latest_frame is None:
            continue
        frame = latest_frame.copy()
    
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame, _ = live_process_frame.process(frame, pose)
    
    if frame_counter % 5 == 0:  # Reduce video writing frequency
        out.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
    frame_counter += 1
    
    # Display in Streamlit UI
    stframe.image(frame, channels="RGB", use_container_width=True)
    
    # Display in OpenCV window
    cv2.imshow("AI Fitness Trainer", cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
out.release()
cv2.destroyAllWindows()

# Download button for recorded video
download_button = st.empty()
if os.path.exists(output_video_file):
    with open(output_video_file, 'rb') as op_vid:
        download = download_button.download_button('Download Video', data=op_vid, file_name='output_live.mp4')
        
        if download:
            st.session_state['download'] = True

if os.path.exists(output_video_file) and st.session_state['download']:
    os.remove(output_video_file)
    st.session_state['download'] = False
    download_button.empty()
