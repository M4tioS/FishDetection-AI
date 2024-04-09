import streamlit as st
import cv2
import torch
import tempfile
from PIL import Image
import os
import pathlib

# Adjusting pathlib for compatibility
temp = pathlib.PosixPath
pathlib.PosixPath = pathlib.WindowsPath

# Define model paths
model_path_exp8 = 'yolov5/best_xl.pt'  # Update with the correct path
model_path_exp10 = 'yolov5/best_nano.pt'  # Update with the correct path

# Load models using torch.hub for YOLOv5 custom model loading
xl_model = torch.hub.load('ultralytics/yolov5', 'custom', path=model_path_exp8, force_reload=True)
nano_model = torch.hub.load('ultralytics/yolov5', 'custom', path=model_path_exp10, force_reload=True)

def process_video(video_path, nano_model, xl_model, save_dir):
    cap = cv2.VideoCapture(video_path)
    frame_number = 0
    fish_detected = False
    save_path = ""
    fps = cap.get(cv2.CAP_PROP_FPS)  # Get the frame rate of the video

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        results = nano_model([frame_rgb], size=640)

        for result in results.xyxy[0]:
            if int(result[5]) == 0:  # Class ID for fish
                xl_results = xl_model([frame_rgb], size=640)

                for xl_result in xl_results.xyxy[0]:
                    if int(xl_result[5]) == 0:
                        fish_detected = True
                        frame_filename = f"confirmed_fish_frame_{frame_number}.jpg"
                        save_path = os.path.join(save_dir, frame_filename)
                        cv2.imwrite(save_path, cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR))
                        detection_time_seconds = frame_number / fps  # Correct calculation for detection time
                        return True, frame_number, save_path, detection_time_seconds

        frame_number += 1

    cap.release()
    return fish_detected, frame_number, save_path, None

def main():
    st.title("Fish Detection Web App")

    uploaded_files = st.file_uploader("Upload video files", type=["mp4", "avi"], accept_multiple_files=True)
    
    if uploaded_files:
        save_dir = "confirmed_frames"
        os.makedirs(save_dir, exist_ok=True)

        if st.button("Start Processing"):
            for uploaded_file in uploaded_files:
                with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmpfile:
                    tmpfile.write(uploaded_file.read())
                    video_path = tmpfile.name

                fish_detected, frame_number, image_path, detection_time_seconds = process_video(video_path, nano_model, xl_model, save_dir)

                if fish_detected:
                    st.subheader(f"Fish Detected in {uploaded_file.name}!")
                    st.write(f"Fish was detected at frame number: {frame_number}, which is approximately {detection_time_seconds:.2f} seconds into the video.")
                    image = Image.open(image_path)
                    st.image(image, caption=f"Detected Fish Frame: {frame_number}")
                else:
                    st.subheader(f"Fish Not Detected in {uploaded_file.name} :(")
                os.remove(video_path)  # Cleanup temporary file

if __name__ == "__main__":
    main()
