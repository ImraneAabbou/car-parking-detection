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


# ─── Routes ──────────────────────────────────────────────────────────
@app.route("/")
def index():
    """Serve the main dashboard page."""
    return render_template("index.html", total_spots=TOTAL_SPOTS)


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


# ─── Main ────────────────────────────────────────────────────────────
if __name__ == "__main__":
    # Start the video processing in a background daemon thread
    processing_thread = threading.Thread(target=video_processing_loop, daemon=True)
    processing_thread.start()

    print("[INFO] Starting VisioPark Dashboard...")
    print("[INFO] Open http://127.0.0.1:5000 in your browser")
    app.run(host="0.0.0.0", port=5000, debug=False, threaded=True)
