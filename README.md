# 🚦 ITS Vehicle Speed Tracking & Color Recognition System

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![OpenCV](https://img.shields.io/badge/OpenCV-Computer%20Vision-green)
![YOLOv8](https://img.shields.io/badge/YOLOv8-Object%20Detection-orange)

An advanced Intelligent Transportation System (ITS) designed for real-time vehicle detection, tracking, speed measurement using OpenCV Homography Perspective Transformation, and adaptive vehicle body color extraction.

---
## 🎬 Demo Video


https://github.com/user-attachments/assets/e7dca7d4-195c-407c-bad2-9c1b11ace0c8







---
## 🌟 Key Features

- 🚘 **Real-Time Vehicle Detection & Tracking**:
  Powered by YOLOv8n object detection and centroid tracking for robust multi-vehicle association across frames.

- 📐 **ITS-Grade Speed Measurement via Homography**:
  Maps 2D camera pixel coordinates `(x, y)` to real-world 3D ground plane coordinates `(X_meters, Y_meters)` using `cv2.getPerspectiveTransform` to eliminate camera perspective distortion.

- 🛣️ **Tire-Road Contact Point Tracking**:
  Tracks the vehicle base tire contact line `(cx, y2)` on the road plane rather than bounding box center points, preventing bounding box expansion artifacts.

- 📈 **Sliding Window & EMA Speed Filtering**:
  Utilizes a multi-frame sliding window history (`dt >= 0.15s`) combined with Exponential Moving Average (EMA) smoothing to eliminate frame jitter and deliver rock-solid speed estimations.

- 🎨 **Dynamic & Adaptive Car Body Color Sampling**:
  Extracts color specifically from the car bonnet/hood region (`40%–80%` height, `20%–80%` width) to isolate painted vehicle metal from windshield glass reflections (sky/tree glare) and road shadows. Re-evaluates dynamically as vehicles move closer for maximum resolution.

- 🟦 **Color Swatch & Hex Code Overlay**:
  Displays a filled 14×14px color swatch square `■` using the true sampled BGR color alongside its exact Hex code (e.g., `#6A6D72`).

---

## 📐 System Architecture & Methodology

```text
 ┌────────────────┐      ┌────────────────┐      ┌─────────────────────────┐
 │   Input Video  │ ───► │ YOLOv8 Detection│ ───► │ Centroid Multi-Tracker  │
 └────────────────┘      └────────────────┘      └────────────┬────────────┘
                                                              │
 ┌────────────────────────┐      ┌────────────────────────┐   │
 │ UI Overlay (Speed,     │ ◄─── │ Adaptive Hood Color    │ ◄─┤
 │ Swatch & Hex Code)     │      │ & HSV Classification   │   │
 └───────────▲────────────┘      └────────────────────────┘   │
             │                                                │
             └────────────────── ┌────────────────────────┐   │
                                 │ OpenCV Homography      │ ◄─┘
                                 │ Real-World Speed (m/s) │
                                 └────────────────────────┘
```

### 1. Homography Perspective Transformation Matrix
In roadside cameras, vertical pixel movement near the horizon represents significantly larger real-world distances than pixel movement near the camera. We establish a perspective mapping matrix `M`:

`SRC_PTS -> cv2.getPerspectiveTransform -> DST_PTS (Meters)`

- **Source Camera Quadrilateral (`SRC_ROAD_PTS`)**: Selected road plane coordinates in `1280x720` frame.
- **Destination Metric Grid (`DST_ROAD_PTS`)**: Real-world dimensions (`8.0m` width × `35.0m` length).

### 2. Vehicle Body Hood Sampling & Color Classification
To extract pure paint color, the crop region is restricted to:

`Crop = Frame[y1 + 0.40 * h : y1 + 0.80 * h,  x1 + 0.20 * w : x1 + 0.80 * w]`

HSV Color Thresholds:
- **Black**: `V < 55`
- **White**: `S < 40` & `V > 160`
- **Gray**: `S < 40` & `V <= 160`
- **Chromatic**: Categorized via Hue (`H`) for Red, Orange, Yellow, Green, Blue, and Violet.

---

## 🛠️ Project Structure

```text
VehicleSpeedTracking/
├── main.py              # Core execution script (Detection, Tracking, Speed & Color)
├── tracker.py           # Centroid tracking module
├── requirements.txt     # Python dependency list
├── video.mp4            # Input traffic video
├── yolov8n.pt           # YOLOv8 nano pre-trained weights
├── docs/
│   └── changelog.md     # Architectural changelog & version history
└── README.md            # Project documentation
```

---

## ⚙️ Installation & Usage

### 1. Prerequisites
- Python 3.8+
- OpenCV (`opencv-python`)
- Ultralytics (`ultralytics`)
- NumPy (`numpy`)

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Run Application
Ensure `video.mp4` is located in the project directory, then run:
```bash
python main.py
```

Press `q` to terminate execution.

---

## 📊 Visual Output Format

Each vehicle bounding box displays:
- **Vehicle ID & Speed**: `ID:5  98 km/h`
- **Color Swatch Square**: `■` 14x14px filled square with true car hood color.
- **Hex Code**: `#6A6D72` below the bounding box.
