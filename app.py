"""
VisioPark — Flask Web Application
================================================
Real-time parking lot monitoring with deep learning inference.
Streams video with annotated parking spots and serves live stats.
Supports runtime model switching between MobileNet, ResNet50, and VGG16.
"""

import os
<<<<<<< Updated upstream
import math
import pickle
import threading
import time
=======
os.environ["OPENCV_LOG_LEVEL"] = "ERROR"

import pickle
import threading
import time
import math
import tempfile
import shutil
>>>>>>> Stashed changes
from datetime import datetime, timedelta
from collections import deque

import cv2
import numpy as np
from flask import Flask, render_template, Response, jsonify, request

try:
    import geoai
except ImportError:
    geoai = None
    print("[WARN] geoai not installed — accident detection tab will be unavailable")

# ─── Lazy-load TensorFlow to speed up startup ───────────────────────
_tf = None
_tf_lock = threading.Lock()


def _get_tf():
    """Import TensorFlow once (thread-safe)."""
    global _tf
    if _tf is None:
        with _tf_lock:
            if _tf is None:
                os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
                import tensorflow as tf
                tf.get_logger().setLevel("ERROR")
                _tf = tf
    return _tf


# ─── Flask app ───────────────────────────────────────────────────────
app = Flask(__name__)

# ─── Import config ───────────────────────────────────────────────────
from config import (
    VIDEO_PATH, PICKLE_PATH,
    SPOT_WIDTH, SPOT_HEIGHT,
    MODEL_IMG_WIDTH, MODEL_IMG_HEIGHT,
    VIDEO_FRAME_WIDTH, VIDEO_FRAME_HEIGHT,
    FRAME_SKIP, CLASSIFICATION_THRESHOLD,
    MAX_EVENTS, BASE_DIR,
)

# ─── Available models registry ──────────────────────────────────────
AVAILABLE_MODELS = {
    "mobilenet": {
        "name": "MobileNetV2",
        "file": "mobilenet_model.h5",
        "path": os.path.join(BASE_DIR, "models", "mobilenet_model.h5"),
    },
    "resnet50": {
        "name": "ResNet50",
        "file": "resnet50_model.h5",
        "path": os.path.join(BASE_DIR, "models", "resnet50_model.h5"),
    },
    "vgg16": {
        "name": "VGG16",
        "file": "vgg16_model.h5",
        "path": os.path.join(BASE_DIR, "models", "vgg16_model.h5"),
    },
}

# ─── Active model state ─────────────────────────────────────────────
_model = None
_model_lock = threading.Lock()
_active_model_key = "mobilenet"  # default


def _load_model(model_key):
    """Load a model by key (thread-safe). Returns the loaded Keras model."""
    global _model, _active_model_key
    tf = _get_tf()
    info = AVAILABLE_MODELS[model_key]
    with _model_lock:
        _model = tf.keras.models.load_model(info["path"])
        _active_model_key = model_key
        print(f"[INFO] Model loaded: {info['name']} ({info['path']})")
    return _model


def _get_model():
    """Get the currently active model, loading default if needed."""
    global _model
    if _model is None:
        _load_model(_active_model_key)
    return _model


# ─── Accident Detection Config ──────────────────────────────────────
from config import (
    ACCIDENT_VIDEOS_DIR, ACCIDENT_PICKLE_PATH,
    ACCIDENT_SPOT_WIDTH, ACCIDENT_SPOT_HEIGHT,
    NUM_ACCIDENT_VIDEOS,
    ACCIDENT_EDGE_DIST_THRESH, ACCIDENT_DEBOUNCE_SEC,
)

ACCIDENT_VIDEO_PATHS = [
    os.path.join(ACCIDENT_VIDEOS_DIR, f"accident-{i}.mp4")
    for i in range(1, NUM_ACCIDENT_VIDEOS + 1)
]

accident_spots = []
if os.path.exists(ACCIDENT_PICKLE_PATH):
    with open(ACCIDENT_PICKLE_PATH, "rb") as f:
        accident_spots = pickle.load(f)

# ─── Load parking positions ─────────────────────────────────────────
with open(PICKLE_PATH, "rb") as f:
    POSITIONS = pickle.load(f)
TOTAL_SPOTS = len(POSITIONS)

# ─── Shared state ───────────────────────────────────────────────────
# spot_status: list of booleans — True = empty/available, False = occupied
spot_status = [True] * TOTAL_SPOTS
spot_status_lock = threading.Lock()

# Event log
event_log = deque(maxlen=MAX_EVENTS)
event_log_lock = threading.Lock()

# Latest annotated frame (JPEG bytes) for MJPEG streaming
latest_frame = None
frame_lock = threading.Lock()

<<<<<<< Updated upstream
# ─── Accident Detection State ──────────────────────────────────────
latest_accident_frame = None
accident_frame_lock = threading.Lock()
accident_debounce_map = {}
accident_video_index = 0  # which video is currently playing
=======
# ─── Accident detection state ───────────────────────────────────────
ACCIDENT_PICKLE_PATH = os.path.join(BASE_DIR, "parking_positions_portion.pkl")
accident_parking_spots = []
if os.path.exists(ACCIDENT_PICKLE_PATH):
    with open(ACCIDENT_PICKLE_PATH, "rb") as f:
        accident_parking_spots = pickle.load(f)
    print(f"[INFO] Loaded {len(accident_parking_spots)} accident parking spot coordinates.")
else:
    print(f"[WARN] Accident pickle not found at {ACCIDENT_PICKLE_PATH}")

accident_events = deque(maxlen=MAX_EVENTS)
accident_events_lock = threading.Lock()

ACCIDENT_SPOT_W = 75
ACCIDENT_SPOT_H = 150
ACCIDENT_EDGE_DIST_THRESH = 5.0
ACCIDENT_DEBOUNCE_SEC = 10.0

ACCIDENT_VIDEOS = [
    os.path.join(BASE_DIR, "data", "accidents", f"accident-{i}.mp4")
    for i in range(1, 4)
]
>>>>>>> Stashed changes


# ─── Helper: classify a batch of spot crops ──────────────────────────
def classify_spots_batch(crops, model):
    """Return a list of booleans: True if spot is empty (Unreserved), False if occupied."""
    if not crops:
        return []
    
    # Preprocess all crops
    processed_imgs = []
    for crop in crops:
        img = cv2.resize(crop, (MODEL_IMG_HEIGHT, MODEL_IMG_WIDTH))
        img = img.astype("float32") / 255.0
        processed_imgs.append(img)
        
    # Stack into a batch
    batch = np.array(processed_imgs)
    
    # Predict in one go
    preds = model.predict(batch, verbose=0)
    
    # Binary: sigmoid output → >threshold = Unreserved (1), else Reserved (0)
    return [float(p[0]) > CLASSIFICATION_THRESHOLD for p in preds]


# ─── Helper: log an event ────────────────────────────────────────────
def log_event(message, event_type="info"):
    with event_log_lock:
        event_log.appendleft({
            "time": datetime.now().strftime("%H:%M:%S"),
            "message": message,
            "type": event_type,
        })


# ─── Background video processing thread ─────────────────────────────
def video_processing_loop():
    """Continuously read video, run inference, update state."""
    global latest_frame, spot_status

    model = _get_model()

    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        print(f"[ERROR] Cannot open video: {VIDEO_PATH}")
        return

    log_event("System started — monitoring active", "system")

    frame_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            # Loop video
            cap.release()
            cap = cv2.VideoCapture(VIDEO_PATH)
            continue

        frame_count += 1

        # Resize to expected dimensions
        frame = cv2.resize(frame, (VIDEO_FRAME_WIDTH, VIDEO_FRAME_HEIGHT))

        # Only run inference every N frames for performance
        if frame_count % FRAME_SKIP == 0:
            # Grab the current model reference (may have been swapped)
            with _model_lock:
                model = _model

            crops_to_classify = []
            valid_indices = []
            new_status = [True] * TOTAL_SPOTS

            for i, pos in enumerate(POSITIONS):
                x, y = pos
                crop = frame[y:y + SPOT_HEIGHT, x:x + SPOT_WIDTH]
                if crop.size > 0:
                    crops_to_classify.append(crop)
                    valid_indices.append(i)

            if crops_to_classify:
                results = classify_spots_batch(crops_to_classify, model)
                for idx, is_empty in zip(valid_indices, results):
                    new_status[idx] = is_empty

            # Detect changes and log events
            with spot_status_lock:
                for i, (old, new) in enumerate(zip(spot_status, new_status)):
                    if old != new:
                        spot_id = f"#{i + 1}"
                        if new:
                            log_event(f"Spot {spot_id} is now available", "available")
                        else:
                            log_event(f"Car parked in spot {spot_id}", "occupied")

                # Check occupancy thresholds
                old_occupied = sum(1 for s in spot_status if not s)
                new_occupied = sum(1 for s in new_status if not s)
                old_pct = (old_occupied / TOTAL_SPOTS) * 100 if TOTAL_SPOTS else 0
                new_pct = (new_occupied / TOTAL_SPOTS) * 100 if TOTAL_SPOTS else 0

                # Alert at 50%, 75%, 90%, 100% thresholds
                for threshold in [50, 75, 90, 100]:
                    if old_pct < threshold <= new_pct:
                        log_event(
                            f"Lot is {int(new_pct)}% full ({new_occupied}/{TOTAL_SPOTS})",
                            "warning" if threshold < 100 else "critical"
                        )

                spot_status = new_status

        # Draw rectangles on frame
        with spot_status_lock:
            current_status = list(spot_status)

        for i, pos in enumerate(POSITIONS):
            x, y = pos
            is_empty = current_status[i] if i < len(current_status) else True
            color = (72, 199, 142) if is_empty else (60, 60, 220)  # BGR: green / red
            thickness = 2
            cv2.rectangle(frame, (x, y), (x + SPOT_WIDTH, y + SPOT_HEIGHT), color, thickness)
            # Small spot number
            cv2.putText(
                frame, str(i + 1),
                (x + 2, y + 12),
                cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1,
            )

        # Draw legend on frame
        cv2.rectangle(frame, (10, 10), (200, 70), (30, 30, 30), -1)
        cv2.rectangle(frame, (20, 20), (40, 35), (72, 199, 142), -1)
        cv2.putText(frame, "Available", (48, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.rectangle(frame, (20, 45), (40, 60), (60, 60, 220), -1)
        cv2.putText(frame, "Occupied", (48, 58), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        # Encode frame to JPEG
        _, buffer = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 75])
        with frame_lock:
            latest_frame = buffer.tobytes()

        # Small sleep to control frame rate (~15 FPS)
        time.sleep(0.0066)

    cap.release()


<<<<<<< Updated upstream
# ─── Accident video processing thread ──────────────────────────────
def accident_processing_loop():
    global latest_accident_frame, accident_video_index, accident_debounce_map

    video_idx = 0
    cap = None
    bg_subtractor = cv2.createBackgroundSubtractorMOG2()
    recent_alerts = {}

    while True:
        if cap is None or not cap.isOpened():
            video_path = ACCIDENT_VIDEO_PATHS[video_idx]
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                print(f"[ACCIDENT] Cannot open: {video_path}")
                time.sleep(2)
                continue
            bg_subtractor = cv2.createBackgroundSubtractorMOG2()
            accident_video_index = video_idx
            video_idx = (video_idx + 1) % len(ACCIDENT_VIDEO_PATHS)
            recent_alerts = {}

        ret, frame = cap.read()
        if not ret:
            cap.release()
            cap = None
            continue

        # --- Motion detection via background subtraction ---
        fg_mask = bg_subtractor.apply(frame)
        _, fg_mask = cv2.threshold(fg_mask, 200, 255, cv2.THRESH_BINARY)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, kernel)
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_CLOSE, kernel)

        contours, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        valid_contours = [cnt for cnt in contours if cv2.contourArea(cnt) > 200]
        num_cars = len(valid_contours)

        # --- VISUAL LAYER: DRAW SEMI-TRANSPARENT PARKING SPOTS ---
        if len(accident_spots) > 0:
            overlay = frame.copy()
            for pos in accident_spots:
                spot_x, spot_y = pos
                cv2.rectangle(overlay, (spot_x, spot_y),
                              (spot_x + ACCIDENT_SPOT_WIDTH, spot_y + ACCIDENT_SPOT_HEIGHT),
                              (200, 200, 200), 1)
            cv2.addWeighted(overlay, 0.4, frame, 0.6, 0, frame)

            for idx, pos in enumerate(accident_spots):
                spot_x, spot_y = pos
                display_num = str(idx + 1)
                cv2.putText(frame, display_num, (spot_x + 3, spot_y + 12),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1, cv2.LINE_AA)

        # --- CRASH ENGINE LOGIC ---
        colliding_indices = set()
        crash_centroids = []

        for i in range(num_cars):
            cnt_a = valid_contours[i]
            M = cv2.moments(cnt_a)
            if M["m00"] != 0:
                cX = int(M["m10"] / M["m00"])
                cY = int(M["m01"] / M["m00"])
            else:
                cX, cY = cnt_a[0][0][0], cnt_a[0][0][1]

            for j in range(i + 1, num_cars):
                cnt_b = valid_contours[j]
                min_dist = float('inf')
                for point in cnt_a:
                    pt = (float(point[0][0]), float(point[0][1]))
                    dist = cv2.pointPolygonTest(cnt_b, pt, True)
                    abs_dist = abs(dist)
                    if abs_dist < min_dist:
                        min_dist = abs_dist

                if min_dist <= ACCIDENT_EDGE_DIST_THRESH:
                    colliding_indices.add(i)
                    colliding_indices.add(j)
                    crash_centroids.append((cX, cY))

        # --- DEBOUNCE + ALERT LOGGING ---
        if len(colliding_indices) > 0 and len(accident_spots) > 0:
            avg_cx = int(sum(pt[0] for pt in crash_centroids) / len(crash_centroids))
            avg_cy = int(sum(pt[1] for pt in crash_centroids) / len(crash_centroids))

            closest_spot_idx = None
            min_spot_dist = float('inf')
            for idx, pos in enumerate(accident_spots):
                spot_x, spot_y = pos
                center_spot_x = spot_x + (ACCIDENT_SPOT_WIDTH / 2)
                center_spot_y = spot_y + (ACCIDENT_SPOT_HEIGHT / 2)
                dist_to_spot = math.sqrt((avg_cx - center_spot_x)**2 + (avg_cy - center_spot_y)**2)
                if dist_to_spot < min_spot_dist:
                    min_spot_dist = dist_to_spot
                    closest_spot_idx = idx

            current_time = datetime.now()
            should_trigger = True
            if closest_spot_idx is not None and closest_spot_idx in recent_alerts:
                last_time = recent_alerts[closest_spot_idx]
                if current_time - last_time < timedelta(seconds=ACCIDENT_DEBOUNCE_SEC):
                    should_trigger = False

            if should_trigger and closest_spot_idx is not None:
                recent_alerts[closest_spot_idx] = current_time
                log_event(
                    f"Accident detected near Spot #{closest_spot_idx + 1} at {current_time.strftime('%H:%M:%S')}",
                    "accident"
                )

        # --- VISUAL RENDERING FOR CAR DETECTIONS ---
        for idx, cnt in enumerate(valid_contours):
            M = cv2.moments(cnt)
            if M["m00"] != 0:
                cX = int(M["m10"] / M["m00"])
                cY = int(M["m01"] / M["m00"])
            else:
                cX, cY = cnt[0][0][0], cnt[0][0][1]

            if idx in colliding_indices:
                cv2.drawContours(frame, [cnt], -1, (0, 0, 255), 2)
                cv2.putText(frame, "CRASH", (cX - 20, cY),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2, cv2.LINE_AA)
            else:
                cv2.drawContours(frame, [cnt], -1, (0, 255, 0), 1)

        # Draw accident mode label
        current_video = os.path.basename(ACCIDENT_VIDEO_PATHS[accident_video_index])
        cv2.rectangle(frame, (10, 10), (240, 35), (30, 30, 30), -1)
        cv2.putText(frame, f"Accident Monitor: {current_video}", (15, 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        # Encode frame to JPEG
        _, buffer = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 75])
        with accident_frame_lock:
            latest_accident_frame = buffer.tobytes()

        time.sleep(0.033)  # ~30 FPS


# ─── MJPEG generator (main feed) ────────────────────────────────────
=======
# ─── Accident detection frame generator ────────────────────────────
def generate_accident_frames():
    """Process accident videos frame-by-frame, yield JPEG bytes for MJPEG streaming.
    Cycles through accident-1.mp4 → accident-2.mp4 → accident-3.mp4 → loop.
    Only runs while a client is connected (no background processing).
    """
    if geoai is None:
        yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n\r\n"
        return

    detector = geoai.CarDetector()
    spots = accident_parking_spots
    spot_w = ACCIDENT_SPOT_W
    spot_h = ACCIDENT_SPOT_H
    edge_dist_thresh = ACCIDENT_EDGE_DIST_THRESH
    debounce_sec = ACCIDENT_DEBOUNCE_SEC

    recent_alerts = {}
    tmp_dir = tempfile.mkdtemp()
    frame_temp = os.path.join(tmp_dir, "frame.tif")

    try:
        video_index = 0
        frame_count = 0

        while True:
            video_path = ACCIDENT_VIDEOS[video_index]
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                video_index = (video_index + 1) % len(ACCIDENT_VIDEOS)
                continue

            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                frame_count += 1

                cv2.imwrite(frame_temp, frame)
                mask_path = detector.generate_masks(frame_temp, min_object_area=800)

                # --- VISUAL LAYER: DRAW SEMI-TRANSPARENT PARKING SPOTS ---
                if len(spots) > 0:
                    overlay = frame.copy()
                    for idx, pos in enumerate(spots):
                        spot_x, spot_y = pos
                        cv2.rectangle(overlay, (spot_x, spot_y), (spot_x + spot_w, spot_y + spot_h), (200, 200, 200), 1)
                    cv2.addWeighted(overlay, 0.4, frame, 0.6, 0, frame)
                    for idx, pos in enumerate(spots):
                        spot_x, spot_y = pos
                        cv2.putText(frame, str(idx + 1), (spot_x + 3, spot_y + 12),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1, cv2.LINE_AA)

                if mask_path and os.path.exists(mask_path):
                    masks_img = cv2.imread(mask_path, cv2.IMREAD_UNCHANGED)

                    if masks_img is not None:
                        mask_8bit = (masks_img * 255).astype(np.uint8) if masks_img.max() <= 1 else masks_img.astype(np.uint8)
                        if len(mask_8bit.shape) == 3:
                            mask_8bit = mask_8bit[:, :, 0]

                        contours, _ = cv2.findContours(mask_8bit, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                        valid_contours = [cnt for cnt in contours if cv2.contourArea(cnt) > 200]
                        num_cars = len(valid_contours)

                        colliding_indices = set()
                        crash_centroids = []

                        # --- CRASH ENGINE LOGIC ---
                        for i in range(num_cars):
                            cnt_a = valid_contours[i]
                            M = cv2.moments(cnt_a)
                            cX = int(M["m10"] / M["m00"]) if M["m00"] != 0 else cnt_a[0][0][0]
                            cY = int(M["m01"] / M["m00"]) if M["m00"] != 0 else cnt_a[0][0][1]

                            for j in range(i + 1, num_cars):
                                cnt_b = valid_contours[j]
                                min_dist = float('inf')
                                for point in cnt_a:
                                    pt = (float(point[0][0]), float(point[0][1]))
                                    dist = cv2.pointPolygonTest(cnt_b, pt, True)
                                    abs_dist = abs(dist)
                                    if abs_dist < min_dist:
                                        min_dist = abs_dist
                                if min_dist <= edge_dist_thresh:
                                    colliding_indices.add(i)
                                    colliding_indices.add(j)
                                    crash_centroids.append((cX, cY))

                        # --- DUAL-CONDITION DEBOUNCE EVALUATION ---
                        if len(colliding_indices) > 0 and len(spots) > 0:
                            avg_cx = int(sum(pt[0] for pt in crash_centroids) / len(crash_centroids))
                            avg_cy = int(sum(pt[1] for pt in crash_centroids) / len(crash_centroids))

                            closest_spot_idx = None
                            min_spot_dist = float('inf')
                            for idx, pos in enumerate(spots):
                                spot_x, spot_y = pos
                                center_spot_x = spot_x + (spot_w / 2)
                                center_spot_y = spot_y + (spot_h / 2)
                                dist_to_spot = math.sqrt((avg_cx - center_spot_x)**2 + (avg_cy - center_spot_y)**2)
                                if dist_to_spot < min_spot_dist:
                                    min_spot_dist = dist_to_spot
                                    closest_spot_idx = idx

                            current_time = datetime.now()
                            should_trigger_alert = True

                            if closest_spot_idx in recent_alerts:
                                last_alert_time = recent_alerts[closest_spot_idx]
                                if (current_time - last_alert_time) < timedelta(seconds=debounce_sec):
                                    should_trigger_alert = False

                            if should_trigger_alert:
                                recent_alerts[closest_spot_idx] = current_time
                                log_entry = {
                                    "datetime": current_time.strftime("%Y-%m-%d %H:%M:%S"),
                                    "closest_spot_index": closest_spot_idx + 1,
                                    "frame": frame_count
                                }
                                with accident_events_lock:
                                    accident_events.appendleft({
                                        "time": current_time.strftime("%H:%M:%S"),
                                        "datetime": current_time.strftime("%Y-%m-%d %H:%M:%S"),
                                        "spot": closest_spot_idx + 1,
                                        "frame": frame_count,
                                    })

                        # --- VISUAL RENDERING FOR CAR DETECTIONS ---
                        for idx, cnt in enumerate(valid_contours):
                            M = cv2.moments(cnt)
                            if M["m00"] != 0:
                                cX = int(M["m10"] / M["m00"])
                                cY = int(M["m01"] / M["m00"])
                            else:
                                cX, cY = cnt[0][0][0], cnt[0][0][1]

                            if idx in colliding_indices:
                                cv2.drawContours(frame, [cnt], -1, (0, 0, 255), 2)
                                cv2.putText(frame, "CRASH", (cX - 20, cY),
                                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2, cv2.LINE_AA)
                            else:
                                cv2.drawContours(frame, [cnt], -1, (0, 255, 0), 1)

                    try:
                        os.remove(mask_path)
                    except OSError:
                        pass

                _, buffer = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
                yield (
                    b"--frame\r\n"
                    b"Content-Type: image/jpeg\r\n\r\n" + buffer.tobytes() + b"\r\n"
                )

            cap.release()
            video_index = (video_index + 1) % len(ACCIDENT_VIDEOS)

    finally:
        try:
            cap.release()
        except Exception:
            pass
        shutil.rmtree(tmp_dir, ignore_errors=True)


# ─── MJPEG generator ────────────────────────────────────────────────
>>>>>>> Stashed changes
def generate_mjpeg():
    """Yield MJPEG frames for streaming."""
    while True:
        with frame_lock:
            frame_data = latest_frame
        if frame_data is not None:
            yield (
                b"--frame\r\n"
                b"Content-Type: image/jpeg\r\n\r\n" + frame_data + b"\r\n"
            )
        time.sleep(0.0066)  # ~15 FPS


# ─── MJPEG generator (accident feed) ───────────────────────────────
def generate_accident_mjpeg():
    """Yield MJPEG frames for accident video streaming."""
    while True:
        with accident_frame_lock:
            frame_data = latest_accident_frame
        if frame_data is not None:
            yield (
                b"--frame\r\n"
                b"Content-Type: image/jpeg\r\n\r\n" + frame_data + b"\r\n"
            )
        time.sleep(0.033)  # ~30 FPS


# ─── Routes ──────────────────────────────────────────────────────────
@app.route("/")
def index():
    """Serve the main dashboard page."""
    spot_positions = [{"x": int(p[0]), "y": int(p[1])} for p in POSITIONS]
    accident_spot_positions = [{"x": int(p[0]), "y": int(p[1])} for p in accident_spots]
    return render_template(
        "index.html",
        total_spots=TOTAL_SPOTS,
        spot_positions=spot_positions,
        spot_w=SPOT_WIDTH,
        spot_h=SPOT_HEIGHT,
        frame_w=VIDEO_FRAME_WIDTH,
        frame_h=VIDEO_FRAME_HEIGHT,
        accident_spot_positions=accident_spot_positions,
        accident_spot_w=ACCIDENT_SPOT_WIDTH,
        accident_spot_h=ACCIDENT_SPOT_HEIGHT,
    )


@app.route("/video_feed")
def video_feed():
    """MJPEG video stream endpoint."""
    return Response(
        generate_mjpeg(),
        mimetype="multipart/x-mixed-replace; boundary=frame",
    )


<<<<<<< Updated upstream
@app.route("/accident_feed")
def accident_feed():
    """MJPEG accident video stream endpoint."""
    return Response(
        generate_accident_mjpeg(),
=======
@app.route("/accident_video_feed")
def accident_video_feed():
    """MJPEG stream for accident detection — only active while client is connected."""
    return Response(
        generate_accident_frames(),
>>>>>>> Stashed changes
        mimetype="multipart/x-mixed-replace; boundary=frame",
    )


@app.route("/api/status")
def api_status():
    """JSON endpoint for current parking status."""
    with spot_status_lock:
        current = list(spot_status)

    occupied = sum(1 for s in current if not s)
    available = sum(1 for s in current if s)
    pct = round((occupied / TOTAL_SPOTS) * 100, 1) if TOTAL_SPOTS else 0

    spots = []
    for i, is_empty in enumerate(current):
        spots.append({
            "id": i + 1,
            "status": "available" if is_empty else "occupied",
        })

    return jsonify({
        "total": TOTAL_SPOTS,
        "occupied": occupied,
        "available": available,
        "occupancy_pct": pct,
        "spots": spots,
    })


@app.route("/api/events")
def api_events():
    """JSON endpoint for recent events."""
    with event_log_lock:
        events = list(event_log)
    return jsonify({"events": events})


<<<<<<< Updated upstream
@app.route("/api/accident_info")
def api_accident_info():
    """JSON endpoint for current accident video info."""
    return jsonify({
        "video_index": accident_video_index,
        "current_video": os.path.basename(ACCIDENT_VIDEO_PATHS[accident_video_index]),
        "num_spots": len(accident_spots),
    })
=======
@app.route("/api/accident_events")
def api_accident_events():
    """JSON endpoint for recent accident detection events."""
    with accident_events_lock:
        events = list(accident_events)
    return jsonify({"events": events})
>>>>>>> Stashed changes


@app.route("/api/models")
def api_models():
    """JSON endpoint listing available models and the active one."""
    models = []
    for key, info in AVAILABLE_MODELS.items():
        models.append({
            "key": key,
            "name": info["name"],
            "active": key == _active_model_key,
        })
    return jsonify({"models": models, "active": _active_model_key})


@app.route("/api/models/switch", methods=["POST"])
def api_models_switch():
    """Switch the active inference model at runtime."""
    data = request.get_json(force=True)
    model_key = data.get("model")

    if model_key not in AVAILABLE_MODELS:
        return jsonify({"error": f"Unknown model: {model_key}"}), 400

    if model_key == _active_model_key:
        return jsonify({
            "message": f"{AVAILABLE_MODELS[model_key]['name']} is already active",
            "active": _active_model_key,
        })

    try:
        old_name = AVAILABLE_MODELS[_active_model_key]["name"]
        _load_model(model_key)
        new_name = AVAILABLE_MODELS[model_key]["name"]
        log_event(f"Model switched: {old_name} → {new_name}", "system")
        return jsonify({
            "message": f"Switched to {new_name}",
            "active": _active_model_key,
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ─── Main ────────────────────────────────────────────────────────────
if __name__ == "__main__":
    # Start the video processing in a background daemon thread
    processing_thread = threading.Thread(target=video_processing_loop, daemon=True)
    processing_thread.start()

    # Start the accident detection in a background daemon thread
    accident_thread = threading.Thread(target=accident_processing_loop, daemon=True)
    accident_thread.start()

    print("[INFO] Starting VisioPark Dashboard...")
    print("[INFO] Open http://127.0.0.1:5000 in your browser")
    app.run(host="0.0.0.0", port=5000, debug=False, threaded=True)
