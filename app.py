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
from datetime import datetime
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

# ─── Accidents state ──────────────────────────────────────────────────
accident_event_log = deque(maxlen=MAX_EVENTS)
accident_event_log_lock = threading.Lock()

latest_accident_frame = None
accident_frame_lock = threading.Lock()

accidents_thread = None
accidents_thread_lock = threading.Lock()

# Active monitoring states (helps switch processes dynamically to save memory)
parking_monitoring_active = True
accidents_monitoring_active = False

# Shared variables for background accident detection worker
accident_worker_frame = None
accident_worker_frame_lock = threading.Lock()
accident_worker_busy = False

# Results from background accident detector worker
accident_detected_contours = []
accident_colliding_indices = set()
accident_results_lock = threading.Lock()


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
    cap = None
    log_event("System started — monitoring active", "system")

    frame_count = 0

    while True:
        if not parking_monitoring_active:
            if cap is not None:
                cap.release()
                cap = None
            time.sleep(0.5)
            continue

        if cap is None:
            cap = cv2.VideoCapture(VIDEO_PATH)
            if not cap.isOpened():
                print(f"[ERROR] Cannot open video: {VIDEO_PATH}")
                time.sleep(1.0)
                continue

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


# ─── Accidents background detector worker ───────────────────────────
def accidents_detector_worker(detector, temp_frame_path):
    """Worker thread that executes slow GeoAI inference on a background schedule."""
    global accidents_monitoring_active
    global accident_worker_frame, accident_worker_busy
    global accident_detected_contours, accident_colliding_indices

    import math
    import pickle
    from datetime import datetime, timedelta

    recent_alerts = {}
    debounce_sec = 10.0
    spot_w = 75
    spot_h = 150
    edge_dist_thresh = 5.0

    while accidents_monitoring_active:
        frame_to_process = None
        with accident_worker_frame_lock:
            if accident_worker_frame is not None:
                frame_to_process = accident_worker_frame.copy()
                accident_worker_frame = None

        if frame_to_process is None:
            time.sleep(0.1)
            continue

        accident_worker_busy = True

        try:
            cv2.imwrite(temp_frame_path, frame_to_process)

            mask_path = detector.generate_masks(temp_frame_path, min_object_area=800)

            valid_contours = []
            colliding_indices = set()

            if mask_path and os.path.exists(mask_path):
                masks = cv2.imread(mask_path, cv2.IMREAD_UNCHANGED)

                if masks is not None:
                    mask_8bit = (masks * 255).astype(np.uint8) if masks.max() <= 1 else masks.astype(np.uint8)
                    if len(mask_8bit.shape) == 3:
                        mask_8bit = mask_8bit[:, :, 0]

                    contours, _ = cv2.findContours(mask_8bit, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    valid_contours = [cnt for cnt in contours if cv2.contourArea(cnt) > 200]
                    num_cars = len(valid_contours)
                    
                    crash_centroids = []

                    # Crash logic
                    for k in range(num_cars):
                        cnt_a = valid_contours[k]
                        M = cv2.moments(cnt_a)
                        cX = int(M["m10"] / M["m00"]) if M["m00"] != 0 else cnt_a[0][0][0]
                        cY = int(M["m01"] / M["m00"]) if M["m00"] != 0 else cnt_a[0][0][1]

                        for j in range(k + 1, num_cars):
                            cnt_b = valid_contours[j]
                            min_dist = float("inf")
                            for point in cnt_a:
                                pt = (float(point[0][0]), float(point[0][1]))
                                dist = cv2.pointPolygonTest(cnt_b, pt, True)
                                abs_dist = abs(dist)
                                if abs_dist < min_dist:
                                    min_dist = abs_dist
                                    
                            if min_dist <= edge_dist_thresh:
                                colliding_indices.add(k)
                                colliding_indices.add(j)
                                crash_centroids.append((cX, cY))

                    # Load parking positions for the accidents view (parking_positions_portion.pkl)
                    accidents_positions_path = os.path.join(BASE_DIR, "parking_positions_portion.pkl")
                    accidents_positions = []
                    try:
                        with open(accidents_positions_path, "rb") as f:
                            accidents_positions = pickle.load(f)
                    except Exception as e:
                        print(f"[ERROR] Could not load accidents positions in worker: {e}")

                    if len(colliding_indices) > 0 and len(accidents_positions) > 0:
                        avg_cx = int(sum(pt[0] for pt in crash_centroids) / len(crash_centroids))
                        avg_cy = int(sum(pt[1] for pt in crash_centroids) / len(crash_centroids))

                        closest_spot_idx = None
                        min_spot_dist = float("inf")

                        for idx, pos in enumerate(accidents_positions):
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
                            time_difference = current_time - last_alert_time
                            if time_difference < timedelta(seconds=debounce_sec):
                                should_trigger_alert = False

                        if should_trigger_alert:
                            recent_alerts[closest_spot_idx] = current_time
                            with accident_event_log_lock:
                                accident_event_log.appendleft({
                                    "time": current_time.strftime("%H:%M:%S"),
                                    "message": f"Crash detected near spot #{closest_spot_idx + 1}",
                                    "type": "critical",
                                })

                try:
                    os.remove(mask_path)
                except OSError:
                    pass

            # Update final shared results
            with accident_results_lock:
                accident_detected_contours = valid_contours
                accident_colliding_indices = colliding_indices

        except Exception as e:
            print(f"[ERROR] Worker error: {e}")

        accident_worker_busy = False


# ─── Accidents processing thread ────────────────────────────────────
def accidents_processing_loop():
    """Process the 3 accident videos sequentially, streaming smoothly at 30 FPS."""
    global latest_accident_frame, accidents_thread
    global accidents_monitoring_active
    global accident_worker_frame, accident_worker_busy
    global accident_detected_contours, accident_colliding_indices

    import geoai
    import pickle
    import tempfile

    os.environ["OPENCV_LOG_LEVEL"] = "ERROR"
    detector = geoai.CarDetector()

    # Reset worker and detection state
    with accident_worker_frame_lock:
        accident_worker_frame = None
    accident_worker_busy = False

    with accident_results_lock:
        accident_detected_contours = []
        accident_colliding_indices = set()

    # Load parking positions for drawing the overlay
    accidents_positions_path = os.path.join(BASE_DIR, "parking_positions_portion.pkl")
    try:
        with open(accidents_positions_path, "rb") as f:
            accidents_positions = pickle.load(f)
    except Exception as e:
        print(f"[ERROR] Could not load accidents positions: {e}")
        accidents_positions = []

    spot_w = 75
    spot_h = 150

    with tempfile.NamedTemporaryFile(suffix=".tif", delete=False) as tmp:
        temp_frame_path = tmp.name

    # Start background detector thread
    worker_thread = threading.Thread(
        target=accidents_detector_worker,
        args=(detector, temp_frame_path),
        daemon=True
    )
    worker_thread.start()

    for i in range(1, 4):
        if not accidents_monitoring_active:
            break

        video_path = f"data/accidents/accident-{i}.mp4"
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"[ERROR] Cannot open video: {video_path}")
            continue

        with accident_event_log_lock:
            accident_event_log.appendleft({
                "time": datetime.now().strftime("%H:%M:%S"),
                "message": f"Started processing accident-{i}.mp4",
                "type": "system",
            })

        frame_count = 0

        while True:
            if not accidents_monitoring_active:
                break

            ret, frame = cap.read()
            if not ret:
                break

            frame_count += 1

            # Submit frame to worker if not busy
            if not accident_worker_busy:
                with accident_worker_frame_lock:
                    accident_worker_frame = frame.copy()

            # Retrieve results from worker thread
            with accident_results_lock:
                local_contours = list(accident_detected_contours)
                local_colliding = set(accident_colliding_indices)

            # Draw parking spots overlay (76 spots from parking_positions_portion.pkl)
            if len(accidents_positions) > 0:
                overlay = frame.copy()
                for idx, pos in enumerate(accidents_positions):
                    spot_x, spot_y = pos
                    cv2.rectangle(overlay, (spot_x, spot_y), (spot_x + spot_w, spot_y + spot_h), (200, 200, 200), 1)
                
                cv2.addWeighted(overlay, 0.4, frame, 0.6, 0, frame)

                for idx, pos in enumerate(accidents_positions):
                    spot_x, spot_y = pos
                    display_num = str(idx + 1)
                    cv2.putText(frame, display_num, (spot_x + 3, spot_y + 12), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1, cv2.LINE_AA)

            # Draw contours on the frame
            for idx, cnt in enumerate(local_contours):
                M = cv2.moments(cnt)
                if M["m00"] != 0:
                    cX = int(M["m10"] / M["m00"])
                    cY = int(M["m01"] / M["m00"])
                else:
                    cX, cY = cnt[0][0][0], cnt[0][0][1]

                if idx in local_colliding:
                    cv2.drawContours(frame, [cnt], -1, (0, 0, 255), 2)
                    cv2.putText(frame, "CRASH", (cX - 20, cY), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2, cv2.LINE_AA)
                else:
                    cv2.drawContours(frame, [cnt], -1, (0, 255, 0), 1)

            frame = cv2.resize(frame, (VIDEO_FRAME_WIDTH, VIDEO_FRAME_HEIGHT))
            _, buffer = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 75])
            with accident_frame_lock:
                latest_accident_frame = buffer.tobytes()

            # Smooth playback at 30 FPS
            time.sleep(0.033)

        cap.release()

    try:
        os.remove(temp_frame_path)
    except OSError:
        pass

    with accident_event_log_lock:
        accident_event_log.appendleft({
            "time": datetime.now().strftime("%H:%M:%S"),
            "message": "Finished processing all accident videos" if accidents_monitoring_active else "Accident processing stopped",
            "type": "info",
        })

    with accidents_thread_lock:
        accidents_thread = None


# ─── MJPEG generator ────────────────────────────────────────────────
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


def generate_accidents_mjpeg():
    """Yield MJPEG frames for accidents streaming."""
    while True:
        with accident_frame_lock:
            frame_data = latest_accident_frame
        if frame_data is not None:
            yield (
                b"--frame\r\n"
                b"Content-Type: image/jpeg\r\n\r\n" + frame_data + b"\r\n"
            )
        time.sleep(0.0066)  # ~15 FPS


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


@app.route("/accidents_feed")
def accidents_feed():
    """MJPEG video stream endpoint for accidents."""
    return Response(
        generate_accidents_mjpeg(),
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


@app.route("/api/accident_events")
def api_accident_events():
    """JSON endpoint for recent accident events."""
    with accident_event_log_lock:
        events = list(accident_event_log)
    return jsonify({"events": events})


@app.route("/api/start_accidents", methods=["POST"])
def api_start_accidents():
    """Start the background processing thread for accidents."""
    global accidents_thread
    with accidents_thread_lock:
        if accidents_thread is None or not accidents_thread.is_alive():
            accidents_thread = threading.Thread(target=accidents_processing_loop, daemon=True)
            accidents_thread.start()
            return jsonify({"status": "started", "message": "Accident detection processing started."})
        else:
            return jsonify({"status": "running", "message": "Accident detection is already running."})


@app.route("/api/set_active_tab", methods=["POST"])
def api_set_active_tab():
    """Dynamically set the active tab to optimize resources (memory and CPU)."""
    global parking_monitoring_active, accidents_monitoring_active
    data = request.get_json(force=True)
    tab = data.get("tab")
    
    if tab in ["video", "map"]:
        parking_monitoring_active = True
        accidents_monitoring_active = False
    elif tab == "accidents":
        parking_monitoring_active = False
        accidents_monitoring_active = True
        
    return jsonify({
        "status": "success",
        "parking_monitoring_active": parking_monitoring_active,
        "accidents_monitoring_active": accidents_monitoring_active
    })


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

    print("[INFO] Starting VisioPark Dashboard...")
    print("[INFO] Open http://127.0.0.1:5000 in your browser")
    app.run(host="0.0.0.0", port=5000, debug=False, threaded=True)
