"""
Vehicle Counting and Tracking App
Uses YOLOv8 + SORT to detect, track, and count vehicles in video streams.
"""
import io
import time
import math
import csv

import streamlit as st
import cv2
import cvzone
import numpy as np
import pandas as pd

from sort import Sort
from config import (
    DISPLAY_WIDTH, DISPLAY_HEIGHT,
    DEFAULT_VIDEO_PATH, MASK_PATH,
    VEHICLE_CLASSES, DEFAULT_CONFIDENCE, MIN_CONFIDENCE, MAX_CONFIDENCE,
    DEFAULT_LINE_Y, DEFAULT_LINE_X_LEFT, DEFAULT_LINE_X_RIGHT,
    SORT_MAX_AGE, SORT_MIN_HITS, SORT_IOU_THRESHOLD,
    DEFAULT_FRAME_SKIP,
)
from detector import load_model

# ─── Page Config ────────────────────────────────────────────────────────────
st.set_page_config(
    layout="wide",
    page_title="Vehicle Counting App",
    page_icon="🚗",
)

# ─── Custom CSS ─────────────────────────────────────────────────────────────
st.markdown("""
<style>
    /* Metric cards */
    div[data-testid="stMetric"] {
        background: linear-gradient(135deg, #1e1e2f 0%, #2d2d44 100%);
        border: 1px solid rgba(255,255,255,0.08);
        border-radius: 12px;
        padding: 16px 20px;
        box-shadow: 0 4px 12px rgba(0,0,0,0.3);
    }
    div[data-testid="stMetricValue"] {
        font-size: 2rem;
        font-weight: 700;
    }
    /* Sidebar */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0e1117 0%, #1a1a2e 100%);
    }
</style>
""", unsafe_allow_html=True)

# ─── Sidebar Controls ──────────────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ Controls")

    st.subheader("📹 Video Source")
    uploaded_file = st.file_uploader("Upload a video", type=["mp4", "mov", "avi"])

    st.divider()
    st.subheader("🔍 Detection")
    confidence_thresh = st.slider(
        "Confidence threshold",
        min_value=MIN_CONFIDENCE, max_value=MAX_CONFIDENCE,
        value=DEFAULT_CONFIDENCE, step=0.05,
        help="Minimum confidence to accept a detection."
    )
    frame_skip = st.slider(
        "Process every N-th frame",
        min_value=1, max_value=10,
        value=DEFAULT_FRAME_SKIP,
        help="Skip frames to speed up processing (1 = every frame)."
    )

    st.divider()
    st.subheader("📏 Counting Line")
    line_y = st.slider("Vertical position (Y)", 0, DISPLAY_HEIGHT, DEFAULT_LINE_Y)
    col_left, col_right = st.columns(2)
    with col_left:
        line_x_left = st.number_input("Left X", 0, DISPLAY_WIDTH, DEFAULT_LINE_X_LEFT)
    with col_right:
        line_x_right = st.number_input("Right X", 0, DISPLAY_WIDTH, DEFAULT_LINE_X_RIGHT)

    st.divider()
    with st.expander("ℹ️ How it works"):
        st.markdown("""
        1. **Region of Interest** — A mask image limits detection to the road area only,
           reducing false positives and computation.
        2. **YOLOv8 Detection** — Each frame is processed by the YOLOv8-Large model to
           detect vehicles (car, bus, truck, motorbike, bicycle).
        3. **SORT Tracking** — The SORT algorithm (Kalman filter + Hungarian matching)
           assigns persistent IDs to vehicles across frames.
        4. **Counting Line** — When a tracked vehicle's center crosses the red line
           in the downward direction, it is counted once.
        """)

# ─── Title ──────────────────────────────────────────────────────────────────
st.title("🚗 Vehicle Counting & Tracking")
st.caption("Powered by YOLOv8 + SORT  •  Upload a video or use the default demo")

# ─── Session State Initialization ───────────────────────────────────────────
for key, default in {
    "total_count": [],
    "class_counts": {},        # {class_name: set_of_ids}
    "direction_up": 0,
    "direction_down": 0,
    "track_history": {},
    "csv_rows": [],
    "processing": False,
    "stop_requested": False,
}.items():
    if key not in st.session_state:
        st.session_state[key] = default


def reset_state():
    """Clear all tracking state for a fresh run."""
    st.session_state.total_count = []
    st.session_state.class_counts = {}
    st.session_state.direction_up = 0
    st.session_state.direction_down = 0
    st.session_state.track_history = {}
    st.session_state.csv_rows = []
    st.session_state.stop_requested = False


# ─── Load Resources ─────────────────────────────────────────────────────────
model = load_model()
mask = cv2.imread(str(MASK_PATH))
if mask is None:
    st.error(f"❌ Could not load mask image from `{MASK_PATH}`")
    st.stop()
mask = cv2.resize(mask, (DISPLAY_WIDTH, DISPLAY_HEIGHT))

# ─── Video Source ───────────────────────────────────────────────────────────
if uploaded_file is not None:
    temp_path = "temp_video.mp4"
    with open(temp_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    video_path = temp_path
else:
    video_path = str(DEFAULT_VIDEO_PATH)

cap = cv2.VideoCapture(video_path)
if not cap.isOpened():
    st.error(f"❌ Could not open video: `{video_path}`")
    st.stop()

total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
video_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0

# ─── Layout ─────────────────────────────────────────────────────────────────
# Top metrics row
m1, m2, m3, m4, m5 = st.columns(5)
metric_total = m1.empty()
metric_up = m2.empty()
metric_down = m3.empty()
metric_fps = m4.empty()
metric_vpm = m5.empty()

# Progress bar
progress_bar = st.progress(0, text="Ready — press **Start Processing** to begin.")

# Action buttons
btn_col1, btn_col2, btn_col3 = st.columns([1, 1, 4])
start_btn = btn_col1.button("▶️ Start Processing", type="primary", use_container_width=True)
stop_btn = btn_col2.button("⏹️ Stop", use_container_width=True)

if stop_btn:
    st.session_state.stop_requested = True

# Video frame placeholder
frame_placeholder = st.empty()

# Per-class breakdown (below video)
class_breakdown_placeholder = st.empty()

# ─── Processing Loop ────────────────────────────────────────────────────────
if start_btn:
    reset_state()
    tracker = Sort(max_age=SORT_MAX_AGE, min_hits=SORT_MIN_HITS, iou_threshold=SORT_IOU_THRESHOLD)

    limits = [int(line_x_left), int(line_y), int(line_x_right), int(line_y)]

    frame_idx = 0
    processed_count = 0
    start_time = time.time()

    while True:
        if st.session_state.stop_requested:
            st.toast("⏹️ Processing stopped by user.")
            break

        success, img = cap.read()
        if not success:
            break

        frame_idx += 1

        # Frame skipping
        if frame_idx % frame_skip != 0:
            continue

        processed_count += 1
        img = cv2.resize(img, (DISPLAY_WIDTH, DISPLAY_HEIGHT))

        # Region of interest
        img_region = cv2.bitwise_and(img, mask)
        results = model(img_region, stream=True, verbose=False)

        detections = np.empty((0, 6))  # [x1, y1, x2, y2, conf, cls_id]
        for r in results:
            boxes = r.boxes
            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0]
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

                conf = math.ceil((box.conf[0] * 100)) / 100
                cls_id = int(box.cls[0])
                name = model.names[cls_id]

                if name in VEHICLE_CLASSES and conf > confidence_thresh:
                    currentArray = np.array([x1, y1, x2, y2, conf, cls_id])
                    detections = np.vstack((detections, currentArray))

        # SORT expects [x1,y1,x2,y2,conf] — pass first 5 cols
        if detections.shape[0] > 0:
            det_for_sort = detections[:, :5]
            cls_map = {i: int(detections[i, 5]) for i in range(detections.shape[0])}
        else:
            det_for_sort = np.empty((0, 5))
            cls_map = {}

        resultsTracker = tracker.update(det_for_sort)

        # Draw counting line
        cv2.line(img, (limits[0], limits[1]), (limits[2], limits[3]), (0, 0, 255), 3)

        for result in resultsTracker:
            x1, y1, x2, y2, Id = result
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            w, h = x2 - x1, y2 - y1

            # Draw bounding box and ID
            cvzone.cornerRect(img, (x1, y1, w, h), l=9, rt=2, colorR=(255, 255, 0))
            cvzone.putTextRect(img, f'{int(Id)}', (max(0, x1), max(35, y1)),
                               scale=0.7, thickness=1, offset=10)

            cx, cy = x1 + w // 2, y1 + h // 2
            cv2.circle(img, (cx, cy), 5, (2, 0, 25), cv2.FILLED)

            # Track history for direction detection
            if Id not in st.session_state.track_history:
                st.session_state.track_history[Id] = []
            st.session_state.track_history[Id].append(cy)

            # Counting logic — check if center crosses the line
            if limits[0] < cx < limits[2] and limits[1] - 15 < cy < limits[1] + 15:
                if len(st.session_state.track_history[Id]) > 1:
                    prev_cy = st.session_state.track_history[Id][-2]
                    moving_down = prev_cy < cy
                    moving_up = prev_cy > cy

                    if st.session_state.total_count.count(Id) == 0:
                        st.session_state.total_count.append(Id)
                        cv2.line(img, (limits[0], limits[1]), (limits[2], limits[3]),
                                 (255, 0, 255), 3)

                        # Direction counting
                        if moving_down:
                            st.session_state.direction_down += 1
                        elif moving_up:
                            st.session_state.direction_up += 1

                        # Per-class counting — try to find the class from SORT match
                        # (approximate: use closest detection to this tracker center)
                        best_cls_name = "unknown"
                        if detections.shape[0] > 0:
                            centers = np.array([
                                ((detections[i, 0] + detections[i, 2]) / 2,
                                 (detections[i, 1] + detections[i, 3]) / 2)
                                for i in range(detections.shape[0])
                            ])
                            dists = np.sqrt((centers[:, 0] - cx)**2 + (centers[:, 1] - cy)**2)
                            closest_idx = np.argmin(dists)
                            best_cls_name = model.names[int(detections[closest_idx, 5])]

                        if best_cls_name not in st.session_state.class_counts:
                            st.session_state.class_counts[best_cls_name] = set()
                        st.session_state.class_counts[best_cls_name].add(Id)

                        # CSV row
                        direction_str = "down" if moving_down else "up"
                        elapsed = round(frame_idx / video_fps, 2)
                        st.session_state.csv_rows.append({
                            "frame": frame_idx,
                            "vehicle_id": int(Id),
                            "class": best_cls_name,
                            "direction": direction_str,
                            "timestamp_sec": elapsed,
                        })

        # Draw count on frame
        total_cnt = len(st.session_state.total_count)
        cvzone.putTextRect(img, f'Count: {total_cnt}', (50, 50),
                           scale=2, thickness=2,
                           colorT=(255, 255, 255),
                           colorR=(0, 0, 0),
                           border=3,
                           colorB=(255, 255, 255))

        # Convert and display
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        frame_placeholder.image(img_rgb, channels="RGB", use_container_width=True)

        # Update metrics
        elapsed_total = time.time() - start_time
        current_fps = processed_count / elapsed_total if elapsed_total > 0 else 0
        vpm = (total_cnt / elapsed_total * 60) if elapsed_total > 0 else 0

        metric_total.metric("🚗 Total Vehicles", total_cnt)
        metric_up.metric("⬆️ Going Up", st.session_state.direction_up)
        metric_down.metric("⬇️ Going Down", st.session_state.direction_down)
        metric_fps.metric("⚡ FPS", f"{current_fps:.1f}")
        metric_vpm.metric("📊 Vehicles/min", f"{vpm:.1f}")

        # Update progress
        pct = min(frame_idx / max(total_frames, 1), 1.0)
        progress_bar.progress(pct, text=f"Frame {frame_idx}/{total_frames}  •  {total_cnt} vehicles counted")

        # Per-class breakdown
        if st.session_state.class_counts:
            breakdown = {k: len(v) for k, v in st.session_state.class_counts.items()}
            class_breakdown_placeholder.dataframe(
                pd.DataFrame(
                    list(breakdown.items()),
                    columns=["Vehicle Class", "Count"]
                ).sort_values("Count", ascending=False),
                use_container_width=True,
                hide_index=True,
            )

    cap.release()
    progress_bar.progress(1.0, text="✅ Processing complete!")

    # ─── Export CSV ──────────────────────────────────────────────────────────
    if st.session_state.csv_rows:
        st.divider()
        st.subheader("📥 Export Results")
        output = io.StringIO()
        writer = csv.DictWriter(output, fieldnames=["frame", "vehicle_id", "class", "direction", "timestamp_sec"])
        writer.writeheader()
        writer.writerows(st.session_state.csv_rows)
        csv_bytes = output.getvalue().encode("utf-8")
        st.download_button(
            label="⬇️ Download CSV",
            data=csv_bytes,
            file_name="vehicle_counts.csv",
            mime="text/csv",
        )
