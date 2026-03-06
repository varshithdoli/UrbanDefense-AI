import cv2
import numpy as np
import os

# Try importing tensorflow, but don't crash if unavailable
try:
    import tensorflow as tf
    MODEL_PATH = "Models/activity_model.h5"
    model = None
    if os.path.exists(MODEL_PATH):
        model = tf.keras.models.load_model(MODEL_PATH)
except ImportError:
    tf = None
    model = None

# --- Internal state ---
_prev_frame = None
_warmup_counter = 0
_WARMUP_FRAMES = 60       # Skip the first 60 frames (2 seconds at 30fps) for camera to stabilize
_consecutive_motion = 0
_CONSECUTIVE_REQUIRED = 10 # Need 10 consecutive high-motion frames before alerting

def reset_state():
    """Call this when starting a new video feed to avoid stale state."""
    global _prev_frame, _warmup_counter, _consecutive_motion
    _prev_frame = None
    _warmup_counter = 0
    _consecutive_motion = 0

def process_frame_for_activity(frame, motion_threshold=120000):
    """
    Detect suspicious high-motion activity in the frame.
    
    The motion_threshold is set very HIGH (120000) to avoid false positives.
    A normal webcam scene typically has motion_level of 5000-40000 from noise alone.
    Only genuinely large sudden movements (fights, stampedes) should exceed 120000.
    
    Returns: (annotated_frame, list_of_threat_strings)
    """
    global _prev_frame, _warmup_counter, _consecutive_motion
    
    threats = []
    
    # If we have a trained deep learning model, use it instead of heuristic
    if model is not None:
        try:
            resized = cv2.resize(frame, (224, 224))
            normalized = resized / 255.0
            prediction = model.predict(np.expand_dims(normalized, axis=0), verbose=0)
            if prediction[0][0] > 0.7:
                threats.append("Suspicious Activity Detected (AI Model)")
                cv2.putText(frame, "SUSPICIOUS ACTIVITY", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            return frame, threats
        except Exception:
            pass  # Fall through to heuristic

    # --- Motion-based heuristic fallback ---
    # Downscale frame for faster processing
    small = cv2.resize(frame, (320, 240))
    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (21, 21), 0)
    
    # Warmup: skip first N frames so camera auto-exposure / white balance settles
    _warmup_counter += 1
    if _warmup_counter <= _WARMUP_FRAMES:
        _prev_frame = gray
        return frame, threats
    
    if _prev_frame is None:
        _prev_frame = gray
        return frame, threats
        
    frame_delta = cv2.absdiff(_prev_frame, gray)
    # Use a higher binary threshold (40 instead of 25) to filter sensor noise
    thresh = cv2.threshold(frame_delta, 40, 255, cv2.THRESH_BINARY)[1]
    thresh = cv2.dilate(thresh, None, iterations=2)
    
    motion_level = np.sum(thresh) / 255.0
    
    # Require SUSTAINED high motion over many consecutive frames
    if motion_level > motion_threshold:
        _consecutive_motion += 1
    else:
        # Decay slowly so brief pauses don't reset the counter
        _consecutive_motion = max(0, _consecutive_motion - 2)
    
    # Only alert after N consecutive high-motion frames (not a single spike)
    if _consecutive_motion >= _CONSECUTIVE_REQUIRED:
        threats.append("Suspicious High Activity / Panic Detected")
        cv2.putText(frame, "HIGH ACTIVITY ALERT", (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                    
    _prev_frame = gray
    return frame, threats
