import pandas as pd
import cv2
import torch
import time
import os
import random
from pathlib import Path

# Load models
local_model_path_xl = 'yolov5/xl_weights_1.0.pt'
local_model_path_nano = 'yolov5/nano_weights_1.0.pt'
xl_model = torch.hub.load('ultralytics/yolov5', 'custom', path=local_model_path_xl, force_reload=True)
nano_model = torch.hub.load('ultralytics/yolov5', 'custom', path=local_model_path_nano, force_reload=True)
xl_model.eval()
nano_model.eval()

# Define paths
videos_dir = "videos"
labels_path = "laxaleir2022.PAL"

# Load labels
df_labels = pd.read_csv(labels_path, sep=";", encoding="latin-1", skiprows=1,
                        names=["Datetime", "Fish height", "Speed", "Position in scanner from bottom",
                               "Direction up=1 or down=0", "index", "Empty column", "Classification", "Silhouette"])
df_labels['Category'] = df_labels['Classification'].apply(lambda x: 1 if x != 'Not fish' else 0)

# Function to select 200 random videos
def select_random_videos(videos_dir, num_videos=200):
    all_videos = [video.name for video in Path(videos_dir).glob("*.mp4")]
    selected_videos = random.sample(all_videos, min(num_videos, len(all_videos)))
    return selected_videos

# Function to process a single video
def process_video(video_path, nano_model, xl_model, use_xl_model=False):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Failed to open video: {video_path}")
        return False

    detection_confirmed = False
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Convert frame to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Select the model based on use_xl_model flag
        model = xl_model if use_xl_model else nano_model
        results = model([frame_rgb], size=640)

        # Simplified logic to check for fish detection
        for result in results.xyxy[0]:
            if int(result[5]) == 0:  # Assuming class 0 is for fish
                detection_confirmed = True
                break  # Exit the loop after the first detection

        if detection_confirmed:
            break

    cap.release()
    return detection_confirmed

# Function to evaluate model performance
def evaluate(selected_videos, labels, nano_model, xl_model, use_nano_first):
    correct_predictions = 0
    total_time = 0

    for video_filename in selected_videos:
        video_id = video_filename.split('.')[0]
        if not video_id.isdigit() or int(video_id) not in labels['index'].values:
            continue
        video_path = os.path.join(videos_dir, video_filename)
        actual_category = labels.loc[labels['index'] == int(video_id), 'Category'].iloc[0]

        start_time = time.time()
        detection = process_video(video_path, nano_model, xl_model, use_xl_model=not use_nano_first)
        end_time = time.time()

        total_time += end_time - start_time
        detected_category = 1 if detection else 0
        correct_predictions += 1 if detected_category == actual_category else 0

    accuracy = correct_predictions / len(selected_videos)
    avg_time = total_time / len(selected_videos)
    return accuracy, avg_time

selected_videos = select_random_videos(videos_dir)
accuracy_nano_first, avg_time_nano_first = evaluate(selected_videos, df_labels, nano_model, xl_model, use_nano_first=True)
accuracy_xl_only, avg_time_xl_only = evaluate(selected_videos, df_labels, nano_model, xl_model, use_nano_first=False)

print(f"Nano then XL - Accuracy: {accuracy_nano_first:.2f}, Avg Time: {avg_time_nano_first:.2f}s")
print(f"XL only - Accuracy: {accuracy_xl_only:.2f}, Avg Time: {avg_time_xl_only:.2f}s")
