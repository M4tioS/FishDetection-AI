# From Pixels to Fins: Advancing Fish Detection with Machine Learning

## Table of Contents
1. [Project Overview](#project-overview)
2. [Features](#features)
3. [Installation](#installation)
4. [Usage](#usage)
5. [Dataset](#dataset)
6. [Model](#model)
7. [Performance & Metrics](#performance--metrics)
8. [System Architecture](#system-architecture)
9. [Video Demonstration](#video-demonstration)
10. [License](#license)
11. [Acknowledgements](#acknowledgements)

## Project Overview
This project uses machine learning to automate fish detection and classification in underwater videos. Developed to support ecological monitoring, the system significantly improves efficiency by reducing manual review efforts, enabling real-time fish detection in Icelandic rivers.

## Features
- **Real-time Fish Detection**: Quickly identifies fish in videos, classifying species when possible.
- **Optimized Processing**: Uses a "quick stopping" method to reduce time spent on videos with no detected fish.
- **High Scalability**: Suitable for large-scale data processing with average processing time reduced by up to 66%.
- **Deployment and Monitoring**: Hosted on a serverless Heroku environment with logging and performance monitoring.

## Installation
To set up the project locally, follow these steps:


### Clone the repository
```bash
git clone https://github.com/Pebble32/fish-detection-ai.git
```

### Navigate to the project directory
```bash
cd fish-detection-ai
```

### (Optional) Create a New Conda Environment
It is recommended to create a separate environment to manage dependencies.
```bash
conda create -n fish-detection-env python=3.9
```

### Install dependencies
```bash
pip install -r requirements.txt
```

## Usage
### Backend
```bash
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

```bash
streamlit run fish_detection_front_end.py
```


## Dataset
The dataset is sourced from the Marine and Freshwater Research Institute in Iceland, consisting of 20-40 second video clips captured in Icelandic rivers. Approximately 71% of these videos contain fish. Due to confidentiality, the actual dataset cannot be shared, but similar datasets could be used for similar purposes.

Gif of fish, gif of no fish

# Model
The system initially used a standard CNN, later transitioning to a YOLOv5 model for more efficient real-time video processing. Further model optimizations include:

- Quick Stopping: Reduces average processing time for videos with detected fish from 17-20 seconds to about 3.5 seconds, a 55.6% improvement. (Later dropped in favor of anomaly detection)
- Nano Model Integration: Reduces processing time for videos without fish to approximately 11 seconds, improving efficiency by 66.2%. (Later dropped in favor of downsampling with anomaly detection)
- Future Enhancements: Potential for YOLOv8 or YOLOv9 integration to improve recall and precision further. From initial testing YOLOv8 has higher recall and YOLOv9 has higher precission. In this project higher recall is more optimal.
- Frame skipping: After calculations and testing we found that we can only focus on every third frame and not lose any fish in Icelandic rivers that improves our classification speed by around 33%. 
- Anomaly Detection with Downsampling: Reduce image size and compare each frame with decaying average of previous frames to see if there is a significant difference. After testing multiple ways of detection this one proved to be most effective allowing us to classify each video in under 1 second acompanied with frame skipping 
Note: Model weights are not provided in this repository.

## Performance & Metrics
The project’s key performance indicators are accuracy, precision, recall, F1 score, and AUC, with the following baseline results:

- Accuracy: ~90% for YOLOv5
- Efficiency: Average processing time reduced by up to 98%
- Precision: Achieved over 95%, highlighting the high accuracy of positive fish detections.
- Recall: Stabilized around 96%, demonstrating the model's ability to capture true positives consistently.
- F1 Score: Approximetly 95.5%.
- AUC (Area Under the ROC Curve): Achived 0.9778
- mAP (Mean Average Precision):
  - mAP@0.5: Exceeded 97%, indicating excellent object detection performance at a 0.5 IoU threshold.
  - mAP@0.5:0.95: Reached ~83%, showing robust performance across a range of IoU thresholds for different object sizes and complexities.

## System Architecture
The system operates with a modular backend. Key components include:

- Logging & Monitoring: Tracks processing times, model performance, and user interactions.
- Anomaly Detection: Incorporates down-sampling and frame-skipping for improved efficiency.
- Periodic Model Updates: Includes scheduled retraining and data augmentation for ongoing performance improvement.
- Error Model Updates: Includes analyzing and retraining the model on data that was marked incorrectly labeled by the user

## Video Demonstration
Watch the video below to see how the model works in action:

Click the image above to watch the video.

## License
This project is licensed under the MIT License. See the LICENSE file for more details.

## Acknowledgements
Special thanks to the Marine and Freshwater Research Institute of Iceland for providing the dataset and support in this project. The team acknowledges PhD. Hafsteinn Einarsson for additional assistance and feedback throughout development.
