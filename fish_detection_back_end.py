import logging
from fastapi import FastAPI, UploadFile, File
from typing import List
import uvicorn
import os
import cv2
import torch
from PIL import Image
import numpy as np
import pathlib
import time

# Setup logging to file
logging.basicConfig(level=logging.INFO, filename='back_end_logs.log', filemode='a',
                    format='%(asctime)s - %(levelname)s - %(message)s')

app = FastAPI(title="Fish Detection API")

# Fix for PosixPath error (occurs in Windows)
temp = pathlib.PosixPath
pathlib.PosixPath = pathlib.WindowsPath

# Weights paths
local_model_path_xl = 'yolov5/xl_weights_1.0.pt' 
local_model_path_nano = 'yolov5/nano_weights_1.0.pt'

# Load models from local paths
xl_model = torch.hub.load('ultralytics/yolov5', 'custom', path=local_model_path_xl, force_reload=True)
nano_model = torch.hub.load('ultralytics/yolov5', 'custom', path=local_model_path_nano, force_reload=True)


# Set the models to evaluation mode
xl_model.eval()
nano_model.eval()

# Logic for processing videos
@app.post("/process-videos/")
async def process_uploaded_videos(files: List[UploadFile] = File(...)):
    start_time = time.time()  # Start processing time measurement
    save_dir = "confirmed_frames"
    os.makedirs(save_dir, exist_ok=True)
    video_details = []

    for uploaded_file in files:
        temp_file_path = os.path.join(save_dir, uploaded_file.filename)
        logging.info(f"Starting processing file: {uploaded_file.filename}")
        with open(temp_file_path, "wb") as temp_file:
            content = await uploaded_file.read()
            temp_file.write(content)
        
        fish_detected, frame_number, image_path, detection_time_seconds, direction, fish_count = process_video(temp_file_path, nano_model, xl_model, save_dir)        
        if fish_detected:
            video_info = {
                "Name": uploaded_file.filename,
                "Frame Number": frame_number,
                "Approx. Time (s)": f"{detection_time_seconds:.2f}",
                "Image Path": image_path,
                "Label": "Fish",
                "Direction": direction,
                "Fish Count": 1
            }
            logging.info(f"File processed: {uploaded_file.filename}, Fish Detected at frame {frame_number}, Direction: {direction}")
        else:
            video_info = {
                "Name": uploaded_file.filename,
                "Frame Number": "N/A",
                "Approx. Time (s)": "N/A",
                "Label": "No Fish",
                "Direction": "N/A",
                "Fish Count": "N/A"
            }
            logging.info(f"File processed: {uploaded_file.filename}, No Fish Detected")
        video_details.append(video_info)

        os.remove(temp_file_path)  # Cleanup temporary file

    total_processing_time = time.time() - start_time
    logging.info(f"Processed {len(files)} files in {total_processing_time:.2f} seconds.")
    return {"video_details": video_details}

# Logic to process each video
def process_video(video_path, nano_model, xl_model, save_dir):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return False, -1, None, None, "N/A", 0

    frame_number = 0
    detected_frames = []
    gif_frames = []
    additional_frames_to_check = 10
    fish_detected_in_consecutive_frames = 0
    last_position = None

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break  # End of video

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = nano_model([frame_rgb], size=640)
        fish_detected_in_frame = False

        for result in results.xyxy[0]:
            if int(result[5]) == 0:  # Fish detected
                fish_detected_in_frame = True
                fish_detected_in_consecutive_frames += 1
                centroid = calculate_centroid(*result[:4])
                
                # Only add to gif_frames if less than 10 have been collected
                if len(gif_frames) < 10:
                    frame_resized = Image.fromarray(frame_rgb).resize((frame_rgb.shape[1] // 2, frame_rgb.shape[0] // 2))
                    gif_frames.append(frame_resized)

                # Update last_position based on the current frame's fish position
                frame_width = frame.shape[1]
                last_position = "Left" if centroid[0] < frame_width / 2 else "Right"
                break

        if not fish_detected_in_frame:
            # If fish was detected in previous frames but not the current one, start the additional frames count
            if fish_detected_in_consecutive_frames > 0:
                additional_frames_to_check -= 1

        if additional_frames_to_check <= 0:
            # If additional frames have been checked and fish hasn't reappeared, stop processing
            break

        frame_number += 1

    cap.release()

    direction = last_position if last_position else "N/A"
    gif_path = None
    if gif_frames:
        gif_path = os.path.join(save_dir, f"detected_fish_{int(time.time())}.gif")
        gif_frames[0].save(gif_path, save_all=True, append_images=gif_frames[1:], optimize=False, duration=100, loop=0)

    detection_time_seconds = frame_number / (cap.get(cv2.CAP_PROP_FPS) or 30)
    return bool(fish_detected_in_consecutive_frames), frame_number, gif_path, detection_time_seconds, direction, len(gif_frames)


def calculate_centroid(x_min, y_min, x_max, y_max):
    return ((x_min + x_max) / 2, (y_min + y_max) / 2)

def determine_direction(centroids):
    if len(centroids) < 2:
        return "Left"

    dx = centroids[-1][0] - centroids[0][0]
    return "Right" if dx > 0 else "Left"




### Add logic with checkbox to know if the video was correctly classified
### All boxes start with being checked (correctly classified), unchecking box means the video was incorrectly classified
### Incorrect classifications would be logged into a file classification_error.txt.
### Logs for classification_error.txt would be name of the video, prediction, and true prediction, data and time



if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)
