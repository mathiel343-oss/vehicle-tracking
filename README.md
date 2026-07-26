# 🚦 ITS Vehicle Speed Tracking & Color Recognition System

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![OpenCV](https://img.shields.io/badge/OpenCV-Computer%20Vision-green)
![YOLOv8](https://img.shields.io/badge/YOLOv8-Object%20Detection-orange)

An advanced Intelligent Transportation System (ITS) designed for real-time vehicle detection, tracking, speed measurement using OpenCV Homography Perspective Transformation, and adaptive vehicle body color extraction.

---

## 🎥 Demo Video




https://github.com/user-attachments/assets/348a9cfa-1048-4d6a-8abe-704f678ec466





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



