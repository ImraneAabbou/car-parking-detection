"""
VisioPark — Flask Web Application
================================================
Real-time parking lot monitoring with deep learning inference.
Streams video with annotated parking spots and serves live stats.
Supports runtime model switching between MobileNet, ResNet50, and VGG16.
"""

import os
import pickle
import threading
import time
import math
from datetime import datetime, timedelta
from collections import deque

import cv2
import numpy as np
from flask import Flask, render_template, Response, jsonify, request

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

# ─── Accident Detection Module ───────────────────────────────────────
ACCIDENT_VIDEOS = [
    os.path.join(BASE_DIR, "data", "accidents", "accident-1.mp4"),
    os.path.join(BASE_DIR, "data", "accidents", "accident-2.mp4"),
    os.path.join(BASE_DIR, "data", "accidents", "accident-3.mp4"),
]
ACCIDENT_PICKLE = os.path.join(BASE_DIR, "parking_positions_portion.pkl")

# Accident shared state
accident_latest_frame = None
accident_frame_lock = threading.Lock()
accident_connections = 0
accident_connections_lock = threading.Lock()
accident_event_log = deque(maxlen=MAX_EVENTS)
accident_event_log_lock = threading.Lock()
accident_debounce_lock = threading.Lock()
accident_recent_alerts = {}
accident_current_video_index = 0
accident_video_lock = threading.Lock()

os.environ["OPENCV_LOG_LEVEL"] = "ERROR"


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


# ─── Accident video processing thread ────────────────────────────────
def accident_processing_loop():
    """Process accident videos in cycle when accident tab is active."""
    global accident_latest_frame, accident_current_video_index

    cap = None
    video_index = 0
    backSub = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=36, detectShadows=False)

    parking_spots = []
    if os.path.exists(ACCIDENT_PICKLE):
        try:
            with open(ACCIDENT_PICKLE, 'rb') as f:
                parking_spots = pickle.load(f)
            print(f"[ACCIDENT] Loaded {len(parking_spots)} parking spot coordinates.")
        except Exception as e:
            print(f"[ACCIDENT] Could not parse pickle: {e}")

    spot_w = 75
    spot_h = 150
    edge_dist_thresh = 5.0
    debounce_sec = 10.0

    temp_frame_path = os.path.join(BASE_DIR, "temp_accident_frame.tif")

    while True:
        with accident_connections_lock:
            active = accident_connections > 0

        if not active:
            if cap is not None:
                cap.release()
                cap = None
                backSub = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=36, detectShadows=False)
            with accident_frame_lock:
                accident_latest_frame = None
            time.sleep(0.5)
            continue

        if cap is None:
            with accident_video_lock:
                video_index = accident_current_video_index
            video_path = ACCIDENT_VIDEOS[video_index]
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                print(f"[ACCIDENT] Cannot open {video_path}")
                time.sleep(1)
                continue
            fps = cap.get(cv2.CAP_PROP_FPS)
            print(f"[ACCIDENT] Playing {os.path.basename(video_path)} @ {fps:.2f} FPS")

        ret, frame = cap.read()
        if not ret:
            cap.release()
            cap = None
            backSub = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=36, detectShadows=False)
            with accident_video_lock:
                accident_current_video_index = (video_index + 1) % len(ACCIDENT_VIDEOS)
            continue

        # Resize for consistency
        orig_h, orig_w = frame.shape[:2]
        if orig_w > 1280:
            scale = 1280 / orig_w
            new_w = 1280
            new_h = int(orig_h * scale)
            frame = cv2.resize(frame, (new_w, new_h))
            orig_h, orig_w = new_h, new_w

        fg_mask = backSub.apply(frame)
        _, fg_mask = cv2.threshold(fg_mask, 200, 255, cv2.THRESH_BINARY)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, kernel, iterations=2)
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_CLOSE, kernel, iterations=2)

        # --- VISUAL: draw parking spots semi-transparent ---
        if parking_spots:
            overlay = frame.copy()
            for pos in parking_spots:
                spot_x, spot_y = pos
                cv2.rectangle(overlay, (spot_x, spot_y), (spot_x + spot_w, spot_y + spot_h), (200, 200, 200), 1)
            cv2.addWeighted(overlay, 0.4, frame, 0.6, 0, frame)
            for idx, pos in enumerate(parking_spots):
                spot_x, spot_y = pos
                cv2.putText(frame, str(idx + 1), (spot_x + 3, spot_y + 12),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1, cv2.LINE_AA)

        # --- Car detection via contours on foreground mask ---
        contours, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        valid_contours = [cnt for cnt in contours if cv2.contourArea(cnt) > 400]

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
        if len(colliding_indices) > 0 and parking_spots:
            avg_cx = int(sum(pt[0] for pt in crash_centroids) / len(crash_centroids))
            avg_cy = int(sum(pt[1] for pt in crash_centroids) / len(crash_centroids))

            closest_spot_idx = None
            min_spot_dist = float('inf')
            for idx, pos in enumerate(parking_spots):
                spot_x, spot_y = pos
                center_spot_x = spot_x + (spot_w / 2)
                center_spot_y = spot_y + (spot_h / 2)
                dist_to_spot = math.sqrt((avg_cx - center_spot_x)**2 + (avg_cy - center_spot_y)**2)
                if dist_to_spot < min_spot_dist:
                    min_spot_dist = dist_to_spot
                    closest_spot_idx = idx

            current_time = datetime.now()
            should_trigger_alert = True

            with accident_debounce_lock:
                if closest_spot_idx in accident_recent_alerts:
                    last_alert_time = accident_recent_alerts[closest_spot_idx]
                    if current_time - last_alert_time < timedelta(seconds=debounce_sec):
                        should_trigger_alert = False

                if should_trigger_alert:
                    accident_recent_alerts[closest_spot_idx] = current_time

            if should_trigger_alert:
                spot_num = closest_spot_idx + 1
                msg = f"ACCIDENT at spot #{spot_num}"
                with accident_event_log_lock:
                    accident_event_log.appendleft({
                        "time": current_time.strftime("%H:%M:%S"),
                        "message": msg,
                        "type": "critical",
                        "spot": spot_num,
                        "datetime": current_time.strftime("%Y-%m-%d %H:%M:%S"),
                    })
                print(f"[ACCIDENT] >>> ALERT: {msg}")

        # --- VISUAL RENDERING ---
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

        # --- Video info overlay ---
        with accident_video_lock:
            vid_name = os.path.basename(ACCIDENT_VIDEOS[accident_current_video_index])
        cv2.putText(frame, f"Source: {vid_name}", (10, orig_h - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

        _, buffer = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 75])
        with accident_frame_lock:
            accident_latest_frame = buffer.tobytes()

        time.sleep(0.03)

    if cap:
        cap.release()
    if os.path.exists(temp_frame_path):
        os.remove(temp_frame_path)


# ─── MJPEG generator (parking) ───────────────────────────────────────
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


# ─── MJPEG generator (accident) ──────────────────────────────────────
def generate_accident_mjpeg():
    """Yield MJPEG frames for accident camera streaming."""
    global accident_connections
    with accident_connections_lock:
        accident_connections += 1
    try:
        while True:
            with accident_frame_lock:
                frame_data = accident_latest_frame
            if frame_data is not None:
                yield (
                    b"--frame\r\n"
                    b"Content-Type: image/jpeg\r\n\r\n" + frame_data + b"\r\n"
                )
            time.sleep(0.03)
    except GeneratorExit:
        pass
    finally:
        with accident_connections_lock:
            accident_connections -= 1


# ─── Routes ──────────────────────────────────────────────────────────
@app.route("/")
def index():
    """Serve the main dashboard page."""
    spot_positions = [{"x": int(p[0]), "y": int(p[1])} for p in POSITIONS]
    return render_template(
        "index.html",
        total_spots=TOTAL_SPOTS,
        spot_positions=spot_positions,
        spot_w=SPOT_WIDTH,
        spot_h=SPOT_HEIGHT,
        frame_w=VIDEO_FRAME_WIDTH,
        frame_h=VIDEO_FRAME_HEIGHT,
    )


@app.route("/video_feed")
def video_feed():
    """MJPEG video stream endpoint."""
    return Response(
        generate_mjpeg(),
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


@app.route("/accident_video_feed")
def accident_video_feed():
    """MJPEG video stream endpoint for accident camera."""
    return Response(
        generate_accident_mjpeg(),
        mimetype="multipart/x-mixed-replace; boundary=frame",
    )


@app.route("/api/accident_events")
def api_accident_events():
    """JSON endpoint for accident alert events."""
    with accident_event_log_lock:
        events = list(accident_event_log)
    return jsonify({"events": events})


@app.route("/api/accident_status")
def api_accident_status():
    """JSON endpoint for accident camera status."""
    with accident_video_lock:
        vid_index = accident_current_video_index
    with accident_connections_lock:
        active = accident_connections > 0
    return jsonify({
        "active": active,
        "current_video": os.path.basename(ACCIDENT_VIDEOS[vid_index]),
    })


# ─── Main ────────────────────────────────────────────────────────────
if __name__ == "__main__":
    # Start the video processing in a background daemon thread
    processing_thread = threading.Thread(target=video_processing_loop, daemon=True)
    processing_thread.start()

    # Start the accident processing in a background daemon thread
    accident_thread = threading.Thread(target=accident_processing_loop, daemon=True)
    accident_thread.start()

    print("[INFO] Starting VisioPark Dashboard...")
    print("[INFO] Open http://127.0.0.1:5000 in your browser")
    app.run(host="0.0.0.0", port=5000, debug=False, threaded=True)
