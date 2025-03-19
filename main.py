import cv2
import numpy as np
import pyttsx3  # Text-to-Speech
from ultralytics import YOLO
import supervision as sv

# Initialize TTS Engine
engine = pyttsx3.init()
engine.setProperty("rate", 150)  # Adjust speed of speech

# Load YOLOv8 Model
model = YOLO("yolov8n.pt")

# Open Webcam or Video File
cap = cv2.VideoCapture(0)  # Use "video.mp4" for a video file

# ByteTrack Object Tracker
tracker = sv.ByteTrack()

# Function to identify shape of detected objects
def identify_shape(contour):
    shape = "Unknown"
    peri = cv2.arcLength(contour, True)
    approx = cv2.approxPolyDP(contour, 0.02 * peri, True)
    
    sides = len(approx)
    
    if sides == 3:
        shape = "Triangle"
    elif sides == 4:
        x, y, w, h = cv2.boundingRect(approx)
        aspect_ratio = w / float(h)
        shape = "Square" if 0.95 <= aspect_ratio <= 1.05 else "Rectangle"
    elif sides == 5:
        shape = "Pentagon"
    elif sides > 5:
        shape = "Circle"
    
    return shape

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Perform Object Detection & Tracking
    results = model(frame)

    # Convert Results to Displayable Format
    detections = sv.Detections.from_ultralytics(results[0])
    
    # Assign Unique IDs to Track Objects
    tracked_detections = tracker.update_with_detections(detections)

    for det, track_id in zip(tracked_detections.xyxy, tracked_detections.tracker_id):
        x1, y1, x2, y2 = map(int, det)  # Bounding box coordinates
        obj_class = tracked_detections.class_id[tracked_detections.tracker_id == track_id][0]

        # Extract Object from Frame
        object_roi = frame[y1:y2, x1:x2]

        # Convert to Grayscale & Detect Edges
        gray = cv2.cvtColor(object_roi, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blurred, 50, 150)

        # Find Contours
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Identify Shape
        detected_shape = "Unknown"
        for contour in contours:
            detected_shape = identify_shape(contour)
            cv2.putText(frame, f"Shape: {detected_shape}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
        
        # Announce Object and Shape
        object_name = model.names[obj_class]
        announcement = f"Detected {object_name}. Shape is {detected_shape}."
        print(announcement)
        engine.say(announcement)
        engine.runAndWait()

    # Visualize the Tracked Objects
    frame = sv.BoxAnnotator().annotate(scene=frame, detections=tracked_detections)

    # Show Output
    cv2.imshow("Object Detection + Shape Identification + Voice", frame)

    # Press 'q' to Exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
