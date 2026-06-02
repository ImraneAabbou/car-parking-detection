"""
Configuration for the Car Parking Detection Web Application.
"""
import os

# ─── Paths ───────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_PATH = os.path.join(BASE_DIR, "models", "mobilenet_model.h5")
VIDEO_PATH = os.path.join(BASE_DIR, "parking.mp4")
PICKLE_PATH = os.path.join(BASE_DIR, "car_position_parking.pkl")

# ─── Parking spot dimensions (from datacollection.py) ───────────────
SPOT_WIDTH = 30
SPOT_HEIGHT = 50

# ─── Model input size (MobileNetV2 configuration) ───────────────────
MODEL_IMG_WIDTH = 40
MODEL_IMG_HEIGHT = 65

# ─── Video processing ───────────────────────────────────────────────
VIDEO_FRAME_WIDTH = 1920
VIDEO_FRAME_HEIGHT = 1080
FRAME_SKIP = 3            # Process every N-th frame for performance
CLASSIFICATION_THRESHOLD = 0.5  # > threshold → Unreserved (empty)

# ─── Event log ───────────────────────────────────────────────────────
MAX_EVENTS = 50           # Maximum number of events to keep in memory
