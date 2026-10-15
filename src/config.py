
import os
from pathlib import Path

SRC_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SRC_DIR.parent

DATASET_DIR = PROJECT_ROOT / "dataset"
MODELS_DIR = PROJECT_ROOT / "models"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
LOGS_DIR = PROJECT_ROOT / "logs"
VIDEOS_DIR = PROJECT_ROOT / "videos"

EDA_OUTPUT_DIR = OUTPUTS_DIR / "eda"
EVAL_OUTPUT_DIR = OUTPUTS_DIR / "evaluation"
CLUSTERING_OUTPUT_DIR = OUTPUTS_DIR / "clustering"
ONNX_MODELS_DIR = MODELS_DIR / "onnx"

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

DATASET_CSV_PATH = DATASET_DIR / "driver_drowsiness_dataset.csv"
DATASET_CLEANED_CSV_PATH = DATASET_DIR / "cleaned_features.csv"
EVENTS_LOG_PATH = LOGS_DIR / "drowsiness_events.log"
SYSTEM_LOG_PATH = LOGS_DIR / "system.log"
EVALUATION_REPORT_PATH = EVAL_OUTPUT_DIR / "model_comparison_report.csv"
EVALUATION_SUMMARY_MD = EVAL_OUTPUT_DIR / "model_comparison_summary.md"

CLASS_LABELS = ["Alert", "Slightly Drowsy", "Drowsy", "Sleep"]
NUM_CLASSES = len(CLASS_LABELS)

LABEL_TO_ID = {label: idx for idx, label in enumerate(CLASS_LABELS)}
ID_TO_LABEL = {idx: label for idx, label in enumerate(CLASS_LABELS)}

CLASS_COLORS_BGR = {
    "Alert": (50, 205, 50),
    "Slightly Drowsy": (0, 215, 255),
    "Drowsy": (0, 140, 255),
    "Sleep": (30, 30, 220),
}

CLASS_COLORS_HEX = {
    "Alert": "#32CD32",
    "Slightly Drowsy": "#FFD700",
    "Drowsy": "#FF8C00",
    "Sleep": "#DC143C",
}

LEFT_EYE_LANDMARKS = [362, 385, 387, 263, 373, 380]
RIGHT_EYE_LANDMARKS = [33, 160, 158, 133, 153, 144]

LEFT_EYE_FULL = [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398]
RIGHT_EYE_FULL = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246]

MOUTH_OUTER_CORNER_L = 61
MOUTH_OUTER_CORNER_R = 291
MOUTH_TOP_LIP = 13
MOUTH_BOTTOM_LIP = 14
MOUTH_TOP_OUTER = 0
MOUTH_BOTTOM_OUTER = 17
MOUTH_LANDMARKS = [61, 291, 13, 14, 0, 17, 78, 308, 82, 312, 87, 317]

HEAD_POSE_LANDMARK_IDS = [1, 199, 33, 263, 61, 291]
CANONICAL_3D_FACE_MODEL = [
    (0.0, 0.0, 0.0),
    (0.0, 330.0, -65.0),
    (-225.0, -170.0, -135.0),
    (225.0, -170.0, -135.0),
    (-150.0, 150.0, -125.0),
    (150.0, 150.0, -125.0),
]

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

DEFAULT_FPS = 30.0

EAR_ALERT_MIN = 0.28
EAR_DROWSY_THRESH = 0.23
EAR_SLEEP_THRESH = 0.18

MAR_YAWN_THRESH = 0.60
MAR_ALERT_MAX = 0.40

CLAHE_CLIP_LIMIT = 2.5
CLAHE_GRID_SIZE = (8, 8)
LOW_LIGHT_LUMINANCE_THRESHOLD = 70.0
ENABLE_AUTO_CLAHE = True

BLINK_CONSEC_FRAMES_MIN = 2
BLINK_CONSEC_FRAMES_MAX = 10
DROWSY_EYE_CLOSURE_FRAMES = 12
SLEEP_EYE_CLOSURE_FRAMES = 30
YAWN_CONSEC_FRAMES = 15

PERCLOS_WINDOW_SIZE = 90
PERCLOS_CLOSURE_THRESHOLD = 0.20

HEAD_NOD_PITCH_THRESH = 22.0
HEAD_TURN_YAW_THRESH = 30.0
HEAD_TILT_ROLL_THRESH = 20.0

FATIGUE_WEIGHTS = {
    "perclos": 35.0,
    "ear_deficit": 25.0,
    "blink_duration": 15.0,
    "yawn_factor": 15.0,
    "head_droop": 10.0,
}

RANDOM_STATE = 42
TEST_SIZE = 0.20
CV_FOLDS = 5

PCA_N_COMPONENTS = 5
PCA_VARIANCE_TARGET = 0.95

SVM_C = 1.0
SVM_KERNEL = "rbf"
SVM_GAMMA = "scale"

DECISION_TREE_MAX_DEPTH = 8
RF_N_ESTIMATORS = 100
RF_MAX_DEPTH = 10
ADABOOST_N_ESTIMATORS = 80
GRADIENT_BOOSTING_N_ESTIMATORS = 100
BAGGING_N_ESTIMATORS = 50

HMM_N_STATES = 4
HMM_STATE_NAMES = CLASS_LABELS

HMM_DEFAULT_TRANSITION_MATRIX = [
    [0.85, 0.12, 0.02, 0.01],
    [0.10, 0.75, 0.12, 0.03],
    [0.02, 0.10, 0.73, 0.15],
    [0.01, 0.02, 0.12, 0.85],
]

HMM_DEFAULT_INITIAL_PROBS = [0.80, 0.15, 0.04, 0.01]

HMM_N_OBSERVATIONS = 5

ALARM_COOLDOWN_SECONDS = 2.0
ALARM_BEEP_FREQUENCY = 1200
ALARM_BEEP_DURATION_MS = 600
ENABLE_AUDIO_ALERT = True
ENABLE_HUD_ALERT = True
SAVE_ALERT_SNAPSHOTS = True

CAMERA_INDEX = 0
CAMERA_WIDTH = 1280
CAMERA_HEIGHT = 720
DISPLAY_WINDOW_NAME = "Real-Time Drowsiness Detection and Alert System"
SHOW_LANDMARKS = True
SHOW_HUD = True
SHOW_TELEMETRY = True
