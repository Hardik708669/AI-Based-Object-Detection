# Object Detection & Shape Identification with Voice Feedback

## Overview
This project is a real-time object detection, tracking, and shape identification system that provides voice feedback using Text-to-Speech (TTS). It utilizes YOLOv8 for object detection and OpenCV for shape recognition.

## Features
- **Real-Time Object Detection:** Uses YOLOv8 to detect objects in a live webcam feed.
- **Shape Identification:** Identifies the shape of detected objects using contour analysis.
- **Object Tracking:** Implements ByteTrack to assign unique IDs to moving objects.
- **Voice Feedback:** Announces the detected object and its shape using `pyttsx3`.

## Technologies Used
- `YOLOv8` for object detection
- `OpenCV` for image processing
- `ByteTrack` for object tracking
- `pyttsx3` for text-to-speech conversion
- `supervision` for visualization and tracking

## Installation
1. Clone the repository:
   ```sh
   git clone https://github.com/your-username/object-detection-shape-voice.git
   cd object-detection-shape-voice
   ```
2. Install dependencies:
   ```sh
   pip install opencv-python numpy pyttsx3 ultralytics supervision
   ```
3. Run the application:
   ```sh
   python main.py
   ```

## Usage
- Start the script to launch the webcam-based object detection.
- The system will detect objects, recognize their shapes, and announce them via voice.
- Press `q` to exit the application.

## License
This project is open-source and available under the MIT License.

