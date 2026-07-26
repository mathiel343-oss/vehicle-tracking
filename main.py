import cv2
import math
import numpy as np
import random
import time
import os
import glob
from ultralytics import YOLO

class SimpleTracker:
    def __init__(self, max_age=30, max_distance=100):
        self.max_age      = max_age
        self.max_distance = max_distance
        self.tracks       = []
        self.track_id_count = 1

    def update(self, detections):
        det_centers = []
        for det in detections:
            bbox = det[0]
            cx = bbox[0] + bbox[2] / 2.0
            cy = bbox[1] + bbox[3] / 2.0
            det_centers.append((cx, cy, bbox))
        updated_tracks = []
        assigned_det   = set()
        for track in self.tracks:
            tcx, tcy  = track["center"]
            best_dist = float("inf")
            best_idx  = -1
            for idx, (dcx, dcy, bbox) in enumerate(det_centers):
                if idx in assigned_det:
                    continue
                dist = math.hypot(dcx - tcx, dcy - tcy)
                if dist < best_dist and dist < self.max_distance:
                    best_dist = dist
                    best_idx  = idx
            if best_idx != -1:
                assigned_det.add(best_idx)
                dcx, dcy, bbox = det_centers[best_idx]
                track["bbox"]   = bbox
                track["center"] = (dcx, dcy)
                track["hits"]  += 1
                track["lost"]   = 0
                updated_tracks.append(track)
        for idx, (dcx, dcy, bbox) in enumerate(det_centers):
            if idx not in assigned_det:
                updated_tracks.append({"id": self.track_id_count, "bbox": bbox, "center": (dcx, dcy), "hits": 1, "lost": 0})
                self.track_id_count += 1
        for track in self.tracks:
            if track not in updated_tracks:
                track["lost"] += 1
                if track["lost"] < self.max_age:
                    updated_tracks.append(track)
        self.tracks = updated_tracks
        return [t for t in self.tracks if t["hits"] >= 2]


def get_body_crop(frame, x1, y1, x2, y2):
    """Crop car hood/bonnet (40-80% height, 20-80% width) to avoid windshield glass and tires."""
    h_img, w_img = frame.shape[:2]
    x1, y1, x2, y2 = max(0,int(x1)), max(0,int(y1)), min(w_img,int(x2)), min(h_img,int(y2))
    w, h = x2-x1, y2-y1
    if w < 10 or h < 10:
        return None
    y1b = y1 + int(h * 0.40)
    y2b = y1 + int(h * 0.80)
    x1b = x1 + int(w * 0.20)
    x2b = x1 + int(w * 0.80)
    crop = frame[y1b:y2b, x1b:x2b]
    return crop if crop.size > 0 else frame[y1:y2, x1:x2]


def classify_color(body_crop):
    """Classify color based on median HSV of valid car body pixels."""
    if body_crop is None or body_crop.size == 0:
        return "Unknown"
    hsv = cv2.cvtColor(body_crop, cv2.COLOR_BGR2HSV)
    H = hsv[:,:,0].flatten().astype(float)
    S = hsv[:,:,1].flatten().astype(float)
    V = hsv[:,:,2].flatten().astype(float)

    valid = (V > 30) & (V < 250)
    if valid.sum() < 5:
        valid = np.ones(len(V), dtype=bool)

    med_h = float(np.median(H[valid]))
    med_s = float(np.median(S[valid]))
    med_v = float(np.median(V[valid]))

    # Achromatic classification (Black, White, Gray)
    if med_v < 55:
        return "Black"
    if med_s < 40:
        if med_v > 160:
            return "White"
        else:
            return "Gray"

    # Chromatic classification (OpenCV H is 0-180)
    if med_h < 12 or med_h >= 168:
        return "Red"
    elif 12 <= med_h < 25:
        return "Orange"
    elif 25 <= med_h < 38:
        return "Yellow"
    elif 38 <= med_h < 85:
        return "Green"
    elif 85 <= med_h < 135:
        return "Blue"
    elif 135 <= med_h < 168:
        return "Violet"
    return "Gray"


def get_swatch_bgr(body_crop):
    """Return actual sampled median BGR of valid car body pixels."""
    if body_crop is None or body_crop.size == 0:
        return (128, 128, 128)
    hsv   = cv2.cvtColor(body_crop, cv2.COLOR_BGR2HSV)
    V     = hsv[:,:,2]
    valid = (V > 30) & (V < 250)
    if not np.any(valid):
        valid = np.ones_like(V, dtype=bool)
    b = int(np.median(body_crop[:,:,0][valid]))
    g = int(np.median(body_crop[:,:,1][valid]))
    r = int(np.median(body_crop[:,:,2][valid]))
    return (b, g, r)


script_dir    = os.path.dirname(os.path.abspath(__file__))
default_video = os.path.join(script_dir, "video.mp4")
if os.path.exists(default_video):
    video_path = default_video
else:
    candidates = glob.glob(os.path.join(script_dir, "*.mp4")) + glob.glob(os.path.join(script_dir, "*.avi")) + glob.glob(os.path.join(script_dir, "*.mov"))
    video_path = candidates[0] if candidates else None
    if video_path is None:
        print("Error: No video file found!")
        exit()

# ─── ITS Homography Perspective Transformation ──────────────────────────────
# 4 points forming road polygon in 1280x720 video frame
SRC_ROAD_PTS = np.float32([
    [480, 220],   # Top-Left
    [800, 220],   # Top-Right
    [1100, 710],  # Bottom-Right
    [220, 710]    # Bottom-Left
])

# Real-world metric dimensions corresponding to the road section (8 meters wide, 35 meters long)
DST_ROAD_PTS = np.float32([
    [0.0, 0.0],
    [8.0, 0.0],
    [8.0, 35.0],
    [0.0, 35.0]
])

HOMOGRAPHY_MATRIX = cv2.getPerspectiveTransform(SRC_ROAD_PTS, DST_ROAD_PTS)

def get_real_world_meters(px, py):
    """Map camera pixel coordinates to real-world ground plane (X_meters, Y_meters)."""
    pts = np.array([[[float(px), float(py)]]], dtype=np.float32)
    transformed = cv2.perspectiveTransform(pts, HOMOGRAPHY_MATRIX)
    return float(transformed[0][0][0]), float(transformed[0][0][1])


print(f"Video loaded: {video_path}")
model   = YOLO("yolov8n.pt")
tracker = SimpleTracker(max_age=25)
cap     = cv2.VideoCapture(video_path)
if not cap.isOpened():
    print("Error: Could not open video.")
    exit()

fps = cap.get(cv2.CAP_PROP_FPS)
if fps <= 0:
    fps = 30.0
frame_period_ms = 1000.0 / fps
frame_idx       = 0
vehicle_records = {}
colors_dict     = {}

while cap.isOpened():
    t_start = time.time()
    ret, frame = cap.read()
    if not ret:
        break
    frame_idx   += 1
    current_time = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
    if current_time <= 0:
        current_time = frame_idx / fps

    frame   = cv2.resize(frame, (1280, 720))
    results = model(frame, conf=0.5, stream=True, verbose=False)
    detections = []
    for r in results:
        for box in r.boxes:
            cls = int(box.cls[0])
            if cls in [2, 3, 5, 7]:
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                detections.append([[x1, y1, x2-x1, y2-y1], float(box.conf[0]), cls])

    confirmed_tracks = tracker.update(detections)

    for track in confirmed_tracks:
        obj_id       = track["id"]
        x1, y1, w, h = track["bbox"]
        x2, y2       = x1+w, y1+h
        cx           = x1 + w / 2.0
        cy_base      = y2  # Base contact point of vehicle tires with the road surface

        if obj_id not in vehicle_records:
            vehicle_records[obj_id] = {
                "history": [],
                "smooth_spd": None,
                "color": None,
                "swatch_bgr": None,
                "best_area": 0
            }
            random.seed(obj_id)
            colors_dict[obj_id] = (random.randint(50,255), random.randint(50,255), random.randint(50,255))

        rec = vehicle_records[obj_id]
        current_area = w * h

        # Dynamic & Adaptive Color Detection
        if rec["color"] is None or current_area > rec["best_area"] * 1.15:
            crop = get_body_crop(frame, x1, y1, x2, y2)
            col  = classify_color(crop)
            if col != "Unknown":
                rec["color"]      = col
                rec["swatch_bgr"] = get_swatch_bgr(crop)
                rec["best_area"]  = current_area

        # Real-World Metric Speed Calculation via Homography & Sliding Window
        mx, my = get_real_world_meters(cx, cy_base)
        rec["history"].append((mx, my, current_time))
        if len(rec["history"]) > 15:
            rec["history"].pop(0)

        if len(rec["history"]) >= 4:
            old_mx, old_my, old_t = rec["history"][0]
            cur_mx, cur_my, cur_t = rec["history"][-1]
            dt_win = cur_t - old_t

            if dt_win >= 0.15:
                dist_m   = math.hypot(cur_mx - old_mx, cur_my - old_my)
                raw_kmh  = (dist_m / dt_win) * 3.6

                if 15.0 <= raw_kmh <= 160.0:
                    prev = rec["smooth_spd"]
                    rec["smooth_spd"] = raw_kmh if prev is None else 0.20 * raw_kmh + 0.80 * prev

        box_color  = colors_dict[obj_id]
        spd        = rec["smooth_spd"]
        swatch_bgr = rec["swatch_bgr"]
        speed_text = f"{int(spd)} km/h" if spd is not None else "..."
        label      = f"ID:{obj_id}  {speed_text}"
        label_w    = len(label)*11 + 30
        lx, ly     = int(x1), int(y1)

        cv2.rectangle(frame, (lx,ly), (int(x2),int(y2)), box_color, 2)
        cv2.rectangle(frame, (lx, ly-26), (lx+label_w, ly), box_color, -1)
        cv2.putText(frame, label, (lx+4, ly-8), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (255,255,255), 1, cv2.LINE_AA)

        if swatch_bgr is not None:
            sw_x, sw_y = lx+label_w-22, ly-22
            # Draw square filled with actual sampled car body color
            cv2.rectangle(frame, (sw_x,sw_y), (sw_x+14,sw_y+14), swatch_bgr, -1)
            cv2.rectangle(frame, (sw_x,sw_y), (sw_x+14,sw_y+14), (210,210,210), 1)
            # Hex code text below box
            br, bg, bb = swatch_bgr[2], swatch_bgr[1], swatch_bgr[0]
            cv2.putText(frame, f"#{br:02X}{bg:02X}{bb:02X}", (lx+4,ly+12),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.32, (230,230,230), 1, cv2.LINE_AA)

    cv2.imshow("Vehicle Speed and Color", frame)

    if cv2.waitKey(1) == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
