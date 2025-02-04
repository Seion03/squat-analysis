
from flask import Flask, request, jsonify, send_file
import cv2
import os
import numpy as np
from utils import get_mediapipe_pose
from process_frame import ProcessFrame
from thresholds import get_thresholds_beginner

app = Flask(__name__)

# Initialize MediaPipe pose detection
pose = get_mediapipe_pose()
thresholds = get_thresholds_beginner()  # Or set based on desired mode
process_frame = ProcessFrame(thresholds=thresholds, flip_frame=True)

@app.route('/process_video', methods=['POST'])
def process_video_endpoint():
    try:
        # Get the video file from the request
        video_file = request.files['video']
        video_path = os.path.join('uploads', video_file.filename)
        video_file.save(video_path)

        # Open the video file
        cap = cv2.VideoCapture(video_path)
        output_path = os.path.join('processed', f'processed_{video_file.filename}')
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = None

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # Process the frame
            processed_frame, _ = process_frame.process(frame, pose)

            # Initialize the VideoWriter with the frame size
            if out is None:
                height, width, _ = processed_frame.shape
                out = cv2.VideoWriter(output_path, fourcc, cap.get(cv2.CAP_PROP_FPS), (width, height))

            out.write(processed_frame)

        cap.release()
        out.release()

        return send_file(output_path, as_attachment=True, mimetype='video/mp4')

    except Exception as e:
        return jsonify({"error": str(e)}), 500

    finally:
        # Cleanup uploaded and processed files
        if os.path.exists(video_path):
            os.remove(video_path)
        if os.path.exists(output_path):
            os.remove(output_path)

if __name__ == '__main__':
    os.makedirs('uploads', exist_ok=True)
    os.makedirs('processed', exist_ok=True)
    app.run(host='0.0.0.0', port=5000)
