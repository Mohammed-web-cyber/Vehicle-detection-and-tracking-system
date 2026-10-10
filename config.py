"""
Centralized configuration for the Vehicle Counting App.
"""
from pathlib import Path

# Project root (directory containing this file)
PROJECT_ROOT = Path(__file__).parent

# Display dimensions — mask and video frames are resized to these
DISPLAY_WIDTH = 1280
DISPLAY_HEIGHT = 720

# Default asset paths
DEFAULT_VIDEO_PATH = PROJECT_ROOT / "traffic1.mp4"
MASK_PATH = PROJECT_ROOT / "mask.png"
MODEL_PATH = PROJECT_ROOT / "yolov8l.pt"

# Vehicle classes to count (subset of COCO)
VEHICLE_CLASSES = {"car", "bicycle", "motorbike", "bus", "truck"}

# Detection defaults
DEFAULT_CONFIDENCE = 0.3
MIN_CONFIDENCE = 0.1
MAX_CONFIDENCE = 0.9

# Counting line defaults
DEFAULT_LINE_Y = 397
DEFAULT_LINE_X_LEFT = 350
DEFAULT_LINE_X_RIGHT = 1150

# SORT tracker defaults
SORT_MAX_AGE = 20
SORT_MIN_HITS = 3
SORT_IOU_THRESHOLD = 0.3

# Frame skip default (1 = process every frame)
DEFAULT_FRAME_SKIP = 1
