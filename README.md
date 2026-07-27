# Computer Vision Object Detection using YOLOv8 and MediaPipe

This project is a real-time computer vision application that uses a webcam to detect and track multiple objects. 
It combines YOLOv8, MediaPipe, and OpenCV to perform live detection and visualization.
The system uses MediaPipe for face detection and hand tracking, while YOLOv8 is used to detect sports balls from the video stream.
OpenCV is responsible for capturing webcam frames, processing images, and displaying the detection results in real time.

The main goal of this project is to demonstrate the integration of deep learning models with computer vision techniques for real-time object detection applications.

## Features

- Real-time webcam-based object detection.
- Face detection using MediaPipe.
- Hand landmark detection and tracking using MediaPipe.
- Sports ball detection using YOLOv8.
- Bounding boxes and confidence scores for detected objects.
- Live video processing and visualization using OpenCV.

## Technologies Used

- Python
- OpenCV
- YOLOv8 (Ultralytics)
- MediaPipe
- Deep Learning
- Computer Vision

## How It Works

The webcam captures live video frames, which are processed using OpenCV. 
Each frame is analyzed by MediaPipe to detect faces and hands, while YOLOv8 identifies sports balls within the scene. 
The detected objects are highlighted with different colored bounding boxes and labels, allowing users to observe the detection process in real time.

## Installation

Install the required libraries using:

pip install opencv-python mediapipe ultralytics

## Running the Project

Run the application using:

python main.py

After running the program, the webcam will open and start detecting faces, hands, and sports balls in real time. Press `q` to close the application.

## Detection Output

The system displays:

- Green bounding boxes for detected faces.
- Hand landmarks and bounding boxes for detected hands.
- Red bounding boxes for detected sports balls with confidence scores.

## Project Structure

Computer-Vision-Project

├── main.py  
├── yolov8n.pt  
├── README.md  
└── requirements.txt  

## Future Improvements

Possible improvements include:

- Training YOLOv8 on custom object classes.
- Adding object tracking capabilities.
- Improving detection accuracy in different environments.
- Integrating the system with IoT devices or mobile applications.

## Author

Computer Vision Project developed using Python, deep learning, and computer vision technologies.
