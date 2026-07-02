"""
YOLO detection wrapper with Streamlit caching.
"""
import streamlit as st
from ultralytics import YOLO
from config import MODEL_PATH


@st.cache_resource
def load_model(path: str = str(MODEL_PATH)) -> YOLO:
    """Load and cache the YOLO model so it persists across Streamlit re-runs."""
    return YOLO(path)
