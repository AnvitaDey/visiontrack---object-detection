# VisionTrack: Real-Time Object Detection & Counting

## 🚀 Overview

VisionTrack is a real-time computer vision system that performs object detection and counting using the YOLOv8 model and OpenCV. The system processes live webcam feed or video input, detects multiple object classes, and overlays bounding boxes, labels, and real-time statistics such as object counts and FPS.

---

## 🔥 Features

*  Real-time object detection using YOLOv8
*  Multi-class detection (80 classes from COCO dataset)
*  Object counting (total + per class)
*  Optional object tracking (prevents duplicate counting)
*  FPS monitoring for performance evaluation
*  Works with webcam or video files
*  Option to save annotated output video

---

## Model Details

* Model: YOLOv8 (Ultralytics)
* Default: `yolov8n.pt` (fast, lightweight)
* Trained on: COCO Dataset (80 object classes)
* No additional training required

---

## Tech Stack

* Python
* OpenCV
* Ultralytics YOLOv8
* NumPy

---

## Project Structure

```
visiontrack-object-detection/
│── detector.py        # Main detection and counting pipeline
│── requirements.txt   # Dependencies
│── README.md          # Project documentation
│── demo.mp4           # Sample output (optional)
```

---

## Performance

* Optimized for real-time performance on CPU
* Lightweight model ensures smooth inference
* FPS varies based on hardware and resolution

---

## Future Improvements

* Line-crossing object counting
* Region-based counting (ROI)
* Web-based dashboard (Streamlit)
* Integration with robotics/automation systems

---

## Use Cases

* Smart surveillance systems
* Traffic monitoring and analytics
* Retail footfall analysis
* Robotics perception systems

---

## 🙌 Acknowledgements

* Ultralytics YOLOv8
* OpenCV community
* COCO Dataset contributors
