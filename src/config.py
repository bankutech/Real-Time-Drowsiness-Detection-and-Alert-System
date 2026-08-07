"""
Configuration module for Real-Time Drowsiness Detection and Alert System.
Centralizes all directories, model hyperparameters, facial landmark mappings,
thresholds, UI aesthetics, and feature definitions.
"""

import os
from pathlib import Path

# ==========================================
# 1. Project Directory Paths
# ==========================================
SRC_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SRC_DIR.parent

DATASET_DIR = PROJECT_ROOT / "dataset"
MODELS_DIR = PROJECT_ROOT / "models"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
LOGS_DIR = PROJECT_ROOT / "logs"
VIDEOS_DIR = PROJECT_ROOT / "videos"

# Sub-output directories
EDA_OUTPUT_DIR = OUTPUTS_DIR / "eda"
EVAL_OUTPUT_DIR = OUTPUTS_DIR / "evaluation"
CLUSTERING_OUTPUT_DIR = OUTPUTS_DIR / "clustering"
ONNX_MODELS_DIR = MODELS_DIR / "onnx"

# Ensure all directories exist
for directory in [
    DATASET_DIR,
    MODELS_DIR,
    ONNX_MODELS_DIR,
    OUTPUTS_DIR,
    LOGS_DIR,
    VIDEOS_DIR,
    EDA_OUTPUT_DIR,
    EVAL_OUTPUT_DIR,
    CLUSTERING_OUTPUT_DIR,
]:
    directory.mkdir(parents=True, exist_ok=True)

# Main File Paths
DATASET_CSV_PATH = DATASET_DIR / "driver_drowsiness_dataset.csv"
DATASET_CLEANED_CSV_PATH = DATASET_DIR / "cleaned_features.csv"
EVENTS_LOG_PATH = LOGS_DIR / "drowsiness_events.log"
SYSTEM_LOG_PATH = LOGS_DIR / "system.log"
EVALUATION_REPORT_PATH = EVAL_OUTPUT_DIR / "model_comparison_report.csv"
EVALUATION_SUMMARY_MD = EVAL_OUTPUT_DIR / "model_comparison_summary.md"

# ==========================================
# 2. Classification Schema & Taxonomy
# ==========================================
CLASS_LABELS = ["Alert", "Slightly Drowsy", "Drowsy", "Sleep"]
NUM_CLASSES = len(CLASS_LABELS)

LABEL_TO_ID = {label: idx for idx, label in enumerate(CLASS_LABELS)}
ID_TO_LABEL = {idx: label for idx, label in enumerate(CLASS_LABELS)}

# BGR Colors for OpenCV rendering
CLASS_COLORS_BGR = {
    "Alert": (50, 205, 50),            # Emerald Green
    "Slightly Drowsy": (0, 215, 255),  # Amber Yellow
    "Drowsy": (0, 140, 255),           # Vivid Orange
    "Sleep": (30, 30, 220),            # Crimson Red
}

# Hex Colors for Matplotlib / Visuals
CLASS_COLORS_HEX = {
    "Alert": "#32CD32",
    "Slightly Drowsy": "#FFD700",
    "Drowsy": "#FF8C00",
    "Sleep": "#DC143C",
}

# ==========================================
# 3. MediaPipe Face Mesh Landmark Indices
# ==========================================
# 6-point Eye Landmarks for Eye Aspect Ratio (EAR)
# P1: outer corner, P2: top-outer, P3: top-inner, P4: inner corner, P5: bottom-inner, P6: bottom-outer
LEFT_EYE_LANDMARKS = [362, 385, 387, 263, 373, 380]
RIGHT_EYE_LANDMARKS = [33, 160, 158, 133, 153, 144]

# Full eye contour landmarks for visual overlay
LEFT_EYE_FULL = [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398]
RIGHT_EYE_FULL = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246]

# Mouth Landmarks for Mouth Aspect Ratio (MAR)
# Outer & inner lip landmarks
MOUTH_OUTER_CORNER_L = 61
MOUTH_OUTER_CORNER_R = 291
MOUTH_TOP_LIP = 13
MOUTH_BOTTOM_LIP = 14
MOUTH_TOP_OUTER = 0
MOUTH_BOTTOM_OUTER = 17
MOUTH_LANDMARKS = [61, 291, 13, 14, 0, 17, 78, 308, 82, 312, 87, 317]

# 3D Canonical Model Points for Head Pose Estimation (Pitch, Yaw, Roll via solvePnP)
# Nose tip, Chin, Left eye left corner, Right eye right corner, Left mouth corner, Right mouth corner
HEAD_POSE_LANDMARK_IDS = [1, 199, 33, 263, 61, 291]
CANONICAL_3D_FACE_MODEL = [
    (0.0, 0.0, 0.0),          # Nose tip (landmark 1)
    (0.0, 330.0, -65.0),      # Chin (landmark 199, +Y is down in image space)
    (-225.0, -170.0, -135.0), # Right eye outer corner (landmark 33, -X, -Y)
    (225.0, -170.0, -135.0),  # Left eye outer corner (landmark 263, +X, -Y)
    (-150.0, 150.0, -125.0),  # Right mouth corner (landmark 61, -X, +Y)
    (150.0, 150.0, -125.0),   # Left mouth corner (landmark 291, +X, +Y)
]

# ==========================================
# 4. Feature Definitions
# ==========================================
FEATURE_COLUMNS = [
    "ear",
    "mar",
    "blink_duration",
    "blink_rate",
    "yawn_freq",
    "eye_closure_dur",
    "face_angle",
    "head_pitch",
    "head_yaw",
    "head_roll",
    "perclos",
    "ear_velocity",
    "ear_acceleration",
]

TARGET_COLUMN = "state"
REGRESSION_TARGET = "fatigue_score"

# ==========================================
# 5. Physical & Temporal Thresholds
# ==========================================
DEFAULT_FPS = 30.0

# Eye Aspect Ratio (EAR) Thresholds
EAR_ALERT_MIN = 0.28
EAR_DROWSY_THRESH = 0.23
EAR_SLEEP_THRESH = 0.18

# Mouth Aspect Ratio (MAR) Thresholds
MAR_YAWN_THRESH = 0.60
MAR_ALERT_MAX = 0.40

# Dynamic Lighting Augmentation (CLAHE)
CLAHE_CLIP_LIMIT = 2.5
CLAHE_GRID_SIZE = (8, 8)
LOW_LIGHT_LUMINANCE_THRESHOLD = 70.0  # Apply CLAHE when average LAB L-channel < 70
ENABLE_AUTO_CLAHE = True

# Temporal Windows & Counters (in frames @ 30 FPS)
BLINK_CONSEC_FRAMES_MIN = 2
BLINK_CONSEC_FRAMES_MAX = 10
DROWSY_EYE_CLOSURE_FRAMES = 12   # ~0.4s
SLEEP_EYE_CLOSURE_FRAMES = 30    # ~1.0s
YAWN_CONSEC_FRAMES = 15          # ~0.5s

PERCLOS_WINDOW_SIZE = 90         # 3-second sliding window at 30 FPS
PERCLOS_CLOSURE_THRESHOLD = 0.20 # Eye considered closed if EAR < 0.20

# Head Pose Angle Thresholds (in degrees)
HEAD_NOD_PITCH_THRESH = 22.0     # Head drooping down
HEAD_TURN_YAW_THRESH = 30.0      # Driver looking away
HEAD_TILT_ROLL_THRESH = 20.0     # Head tilting sideways

# ==========================================
# 6. Fatigue Score Weights (Linear Model Prior)
# ==========================================
# Fatigue Score ranges from 0 (Peak Alertness) to 100 (Deep Sleep / Extreme Fatigue)
FATIGUE_WEIGHTS = {
    "perclos": 35.0,         # PERCLOS has the highest correlation with fatigue
    "ear_deficit": 25.0,     # Normalized eye closure
    "blink_duration": 15.0,  # Prolonged blinks indicate drowsiness
    "yawn_factor": 15.0,     # Active yawning episodes
    "head_droop": 10.0,      # Head downward tilt
}

# ==========================================
# 7. Machine Learning Hyperparameters
# ==========================================
RANDOM_STATE = 42
TEST_SIZE = 0.20
CV_FOLDS = 5

# PCA
PCA_N_COMPONENTS = 5
PCA_VARIANCE_TARGET = 0.95

# SVM
SVM_C = 1.0
SVM_KERNEL = "rbf"
SVM_GAMMA = "scale"

# Tree & Ensemble
DECISION_TREE_MAX_DEPTH = 8
RF_N_ESTIMATORS = 100
RF_MAX_DEPTH = 10
ADABOOST_N_ESTIMATORS = 80
GRADIENT_BOOSTING_N_ESTIMATORS = 100
BAGGING_N_ESTIMATORS = 50

# ==========================================
# 8. Hidden Markov Model (Manual Pure NumPy)
# ==========================================
# 4 Hidden States: 0=Alert, 1=Slightly Drowsy, 2=Drowsy, 3=Sleep
HMM_N_STATES = 4
HMM_STATE_NAMES = CLASS_LABELS

# Realistic prior driver state transition probabilities
# (Drivers rarely jump directly from Alert to Sleep without passing through intermediate states)
HMM_DEFAULT_TRANSITION_MATRIX = [
    [0.85, 0.12, 0.02, 0.01],  # From Alert
    [0.10, 0.75, 0.12, 0.03],  # From Slightly Drowsy
    [0.02, 0.10, 0.73, 0.15],  # From Drowsy
    [0.01, 0.02, 0.12, 0.85],  # From Sleep
]

# Initial State Probabilities
HMM_DEFAULT_INITIAL_PROBS = [0.80, 0.15, 0.04, 0.01]

# Observation Space for discrete HMM (Discretized feature quintiles / tokens)
HMM_N_OBSERVATIONS = 5

# ==========================================
# 9. Alert System Configuration
# ==========================================
ALARM_COOLDOWN_SECONDS = 2.0
ALARM_BEEP_FREQUENCY = 1200      # Hz
ALARM_BEEP_DURATION_MS = 600     # ms
ENABLE_AUDIO_ALERT = True
ENABLE_HUD_ALERT = True
SAVE_ALERT_SNAPSHOTS = True

# ==========================================
# 10. Real-time Video Stream & Camera Config
# ==========================================
CAMERA_INDEX = 0
CAMERA_WIDTH = 1280
CAMERA_HEIGHT = 720
DISPLAY_WINDOW_NAME = "Real-Time Drowsiness Detection and Alert System"
SHOW_LANDMARKS = True
SHOW_HUD = True
SHOW_TELEMETRY = True
