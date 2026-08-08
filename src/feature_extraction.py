"""
Feature Extraction & Computer Vision Pipeline (Unit 1: Feature Extraction).
Extracts Eye Aspect Ratio (EAR), Mouth Aspect Ratio (MAR), Blink Rate/Duration,
PERCLOS, and Head Pose Euler Angles (Pitch, Yaw, Roll) via MediaPipe Face Mesh & OpenCV solvePnP.
"""

import math
import collections
from pathlib import Path
from typing import Tuple, Dict, Any, List, Optional

import cv2
import numpy as np

import mediapipe as mp
from mediapipe.tasks.python import vision
from mediapipe.tasks.python.core import base_options

from src import config
from src.utils import setup_logger, euclidean_distance_2d, draw_head_pose_axes

logger = setup_logger("FeatureExtraction")


def calculate_ear(eye_landmarks: List[Tuple[float, float]]) -> float:
    """
    Computes Eye Aspect Ratio (EAR) based on Soukupova & Cech (2016).
    Formula: EAR = (||p2 - p6|| + ||p3 - p5||) / (2 * ||p1 - p4||)
    """
    if len(eye_landmarks) < 6:
        return 0.0

    (x1, y1), (x2, y2), (x3, y3), (x4, y4), (x5, y5), (x6, y6) = eye_landmarks[:6]

    # Inlined fast Euclidean distances
    dist_v1 = math.hypot(x2 - x6, y2 - y6)
    dist_v2 = math.hypot(x3 - x5, y3 - y5)
    dist_h = math.hypot(x1 - x4, y1 - y4)

    if dist_h < 1e-6:
        return 0.0

    return (dist_v1 + dist_v2) / (2.0 * dist_h)


def calculate_mar(mouth_landmarks: Dict[str, Tuple[float, float]]) -> float:
    """
    Computes Mouth Aspect Ratio (MAR) using vertical lip opening vs horizontal width.
    Formula: MAR = (||top_outer - bot_outer|| + ||top_inner - bot_inner||) / (2 * ||left - right||)
    """
    try:
        (lx, ly) = mouth_landmarks["left"]
        (rx, ry) = mouth_landmarks["right"]
        (tlx, tly) = mouth_landmarks["top_lip"]
        (blx, bly) = mouth_landmarks["bot_lip"]
        (tox, toy) = mouth_landmarks["top_outer"]
        (box, boy) = mouth_landmarks["bot_outer"]

        dist_v1 = math.hypot(tlx - blx, tly - bly)
        dist_v2 = math.hypot(tox - box, toy - boy)
        dist_h = math.hypot(lx - rx, ly - ry)

        if dist_h < 1e-6:
            return 0.0

        return (dist_v1 + dist_v2) / (2.0 * dist_h)
    except (KeyError, TypeError):
        return 0.0


def enhance_low_light_clahe(
    frame_bgr: np.ndarray,
    clip_limit: float = config.CLAHE_CLIP_LIMIT,
    grid_size: Tuple[int, int] = config.CLAHE_GRID_SIZE,
    threshold: float = config.LOW_LIGHT_LUMINANCE_THRESHOLD,
    force_apply: bool = False,
) -> Tuple[np.ndarray, bool, float]:
    """
    Dynamic Lighting Augmentation via Contrast Limited Adaptive Histogram Equalization (CLAHE).
    Detects low-light or infrared underexposed conditions by measuring the mean luminance of the
    LAB L-channel. If mean luminance < threshold, CLAHE is applied to the L-channel to boost
    contrast and maximize MediaPipe facial landmark detection precision.

    Returns:
        Tuple of (processed_frame_bgr, is_enhanced_flag, mean_luminance)
    """
    if frame_bgr is None or frame_bgr.size == 0:
        return frame_bgr, False, 0.0

    lab = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2LAB)
    l_channel, a_channel, b_channel = cv2.split(lab)
    mean_luminance = float(np.mean(l_channel))

    if force_apply or mean_luminance < threshold:
        clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=grid_size)
        enhanced_l = clahe.apply(l_channel)
        enhanced_lab = cv2.merge([enhanced_l, a_channel, b_channel])
        enhanced_bgr = cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2BGR)
        return enhanced_bgr, True, mean_luminance

    return frame_bgr, False, mean_luminance


class BlinkAndYawnTracker:
    """
    Maintains temporal sliding windows and state machines to track:
    - Blink occurrences, blink duration (sec), blink rate (blinks/min)
    - Continuous eye closure duration (sec)
    - Yawn occurrences and frequency
    - PERCLOS (Percentage of Eye Closure over sliding window)
    - Temporal Differential Features: EAR Velocity (dEAR/dt) and EAR Acceleration (d^2EAR/dt^2)
    """

    def __init__(self, fps: float = config.DEFAULT_FPS, perclos_window_size: int = config.PERCLOS_WINDOW_SIZE):
        self.fps = fps
        self.perclos_window_size = perclos_window_size

        # PERCLOS rolling buffer (stores 1 if eye closed, 0 if open)
        self.perclos_buffer = collections.deque(maxlen=perclos_window_size)

        # Blink state tracking
        self.blink_counter = 0
        self.total_blinks = 0
        self.last_blink_duration = 0.18
        self.continuous_closure_frames = 0

        # Rolling 60-second window for blink rate calculation
        self.blink_timestamps = collections.deque()

        # Yawn state tracking
        self.yawn_counter = 0
        self.total_yawns = 0
        self.yawn_timestamps = collections.deque()
        self.is_yawning = False

        # Temporal Differential Tracking (Velocity & Acceleration)
        self.prev_ear: Optional[float] = None
        self.prev_velocity: float = 0.0
        self.last_time: Optional[float] = None

    def update(self, ear: float, mar: float, current_time: float) -> Dict[str, float]:
        """
        Updates temporal buffers with the latest frame measurements.
        """
        # 1. PERCLOS Update
        is_closed = 1 if ear < config.PERCLOS_CLOSURE_THRESHOLD else 0
        self.perclos_buffer.append(is_closed)
        perclos = float(sum(self.perclos_buffer) / max(1, len(self.perclos_buffer)))

        # 2. Eye Closure and Blink Duration
        if ear < config.EAR_DROWSY_THRESH:
            self.blink_counter += 1
            self.continuous_closure_frames += 1
        else:
            if config.BLINK_CONSEC_FRAMES_MIN <= self.blink_counter <= config.BLINK_CONSEC_FRAMES_MAX * 3:
                self.total_blinks += 1
                self.last_blink_duration = float(self.blink_counter / self.fps)
                self.blink_timestamps.append(current_time)
            self.blink_counter = 0
            self.continuous_closure_frames = 0

        continuous_closure_sec = float(self.continuous_closure_frames / self.fps)

        # Purge blink timestamps older than 60 seconds
        while self.blink_timestamps and (current_time - self.blink_timestamps[0] > 60.0):
            self.blink_timestamps.popleft()

        # Blink rate (per minute)
        blink_rate = float(len(self.blink_timestamps))

        # 3. Yawn Tracking
        if mar >= config.MAR_YAWN_THRESH:
            self.yawn_counter += 1
            if self.yawn_counter >= config.YAWN_CONSEC_FRAMES and not self.is_yawning:
                self.total_yawns += 1
                self.is_yawning = True
                self.yawn_timestamps.append(current_time)
        else:
            self.yawn_counter = 0
            self.is_yawning = False

        # Purge yawn timestamps older than 60 seconds
        while self.yawn_timestamps and (current_time - self.yawn_timestamps[0] > 60.0):
            self.yawn_timestamps.popleft()

        # Yawn frequency (normalized 0..1 per minute factor)
        yawn_freq = float(min(1.0, len(self.yawn_timestamps) / 5.0))

        # 4. Temporal Differential Dynamics (EAR Velocity & Acceleration)
        dt = 1.0 / self.fps
        if self.last_time is not None and current_time > self.last_time:
            dt = max(1e-4, current_time - self.last_time)

        if self.prev_ear is not None:
            ear_velocity = float((ear - self.prev_ear) / dt)
            ear_acceleration = float((ear_velocity - self.prev_velocity) / dt)
        else:
            ear_velocity = 0.0
            ear_acceleration = 0.0

        self.prev_ear = ear
        self.prev_velocity = ear_velocity
        self.last_time = current_time

        return {
            "perclos": perclos,
            "blink_count": self.total_blinks,
            "blink_duration": self.last_blink_duration,
            "blink_rate": blink_rate,
            "eye_closure_dur": continuous_closure_sec,
            "yawn_count": self.total_yawns,
            "yawn_freq": yawn_freq,
            "is_yawning": self.is_yawning,
            "ear_velocity": ear_velocity,
            "ear_acceleration": ear_acceleration,
        }

    def reset(self) -> None:
        self.perclos_buffer.clear()
        self.blink_counter = 0
        self.total_blinks = 0
        self.last_blink_duration = 0.18
        self.continuous_closure_frames = 0
        self.blink_timestamps.clear()
        self.yawn_counter = 0
        self.total_yawns = 0
        self.yawn_timestamps.clear()
        self.is_yawning = False
        self.prev_ear = None
        self.prev_velocity = 0.0
        self.last_time = None


_STATIC_MODEL_POINTS_3D = np.ascontiguousarray(config.CANONICAL_3D_FACE_MODEL, dtype=np.float64)
_STATIC_DIST_COEFFS = np.zeros((4, 1), dtype=np.float64)
_CAMERA_MATRIX_CACHE: Dict[Tuple[int, int], np.ndarray] = {}


def _get_cached_camera_matrix(w: int, h: int) -> np.ndarray:
    """Retrieves or caches 3x3 pinhole camera intrinsic matrix for given resolution."""
    key = (w, h)
    if key not in _CAMERA_MATRIX_CACHE:
        focal_length = float(w)
        center = (w / 2.0, h / 2.0)
        _CAMERA_MATRIX_CACHE[key] = np.array(
            [[focal_length, 0.0, center[0]], [0.0, focal_length, center[1]], [0.0, 0.0, 1.0]],
            dtype=np.float64,
        )
    return _CAMERA_MATRIX_CACHE[key]


def estimate_head_pose(
    landmarks_2d: Dict[int, Tuple[float, float]],
    frame_shape: Tuple[int, int, int],
) -> Tuple[float, float, float, float, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Calculates Head Pose Euler Angles (Pitch, Yaw, Roll) via cv2.solvePnP
    using 3D canonical facial model points.
    """
    h, w = frame_shape[:2]
    camera_matrix = _get_cached_camera_matrix(w, h)
    dist_coeffs = _STATIC_DIST_COEFFS

    # 2D Image Points
    image_points_2d = np.empty((len(config.HEAD_POSE_LANDMARK_IDS), 2), dtype=np.float64)
    for idx, lm_id in enumerate(config.HEAD_POSE_LANDMARK_IDS):
        if lm_id in landmarks_2d:
            image_points_2d[idx] = landmarks_2d[lm_id]
        else:
            image_points_2d[idx] = (w / 2.0, h / 2.0)

    # Solve PnP
    success, rotation_vec, translation_vec = cv2.solvePnP(
        _STATIC_MODEL_POINTS_3D,
        image_points_2d,
        camera_matrix,
        dist_coeffs,
        flags=cv2.SOLVEPNP_ITERATIVE,
    )

    if not success:
        return 0.0, 0.0, 0.0, 0.0, np.zeros(3), np.zeros(3), camera_matrix, dist_coeffs

    # Convert rotation vector to rotation matrix
    rot_mat, _ = cv2.Rodrigues(rotation_vec)

    # Extract Euler angles (in degrees)
    euler_angles, _, _, _, _, _ = cv2.RQDecomp3x3(rot_mat)
    
    # Normalize Euler angles to [-90, +90] range around neutral forward gaze
    def _norm_deg(a: float) -> float:
        a = (a + 180.0) % 360.0 - 180.0
        if a > 90.0:
            a -= 180.0
        elif a < -90.0:
            a += 180.0
        return float(a)

    pitch = _norm_deg(euler_angles[0])
    yaw = _norm_deg(euler_angles[1])
    roll = _norm_deg(euler_angles[2])

    # Face Angle magnitude
    face_angle = float(math.sqrt(pitch**2 + yaw**2 + roll**2))

    return pitch, yaw, roll, face_angle, rotation_vec, translation_vec, camera_matrix, dist_coeffs


def get_head_pose_direction(pitch: float, yaw: float, roll: float) -> str:
    """Provides human-readable semantic driver orientation and gaze direction."""
    if pitch > 18.0:
        return "Looking Down (Distracted)"
    elif pitch < -18.0:
        return "Looking Up"
    elif yaw < -20.0:
        return "Looking Left (Distracted)"
    elif yaw > 20.0:
        return "Looking Right (Distracted)"
    elif abs(roll) > 20.0:
        return "Head Tilted (Fatigued)"
    else:
        return "Facing Ahead (Attentive)"


def detect_eyewear(frame: np.ndarray, landmarks_2d_dict: Dict[int, Tuple[float, float]]) -> Tuple[bool, str]:
    """
    Computer Vision Eyewear & Glasses Detector.
    Analyzes horizontal nose bridge frame edge gradient, orbital rim contrast,
    and lens specular reflection / anti-reflective glare.
    """
    if len(landmarks_2d_dict) < 30:
        return False, "None"

    # Require inner eye corners (133, 362) and nose bridge landmark (6 or 168)
    if 133 not in landmarks_2d_dict or 362 not in landmarks_2d_dict:
        return False, "None"

    p_r = landmarks_2d_dict[133]  # Right inner eye corner
    p_l = landmarks_2d_dict[362]  # Left inner eye corner
    p_bridge = landmarks_2d_dict.get(6, landmarks_2d_dict.get(168, ((p_r[0] + p_l[0]) / 2.0, (p_r[1] + p_l[1]) / 2.0)))

    h_img, w_img = frame.shape[:2]
    x1 = max(0, int(min(p_r[0], p_l[0])))
    x2 = min(w_img, int(max(p_r[0], p_l[0])))
    y_mid = int(p_bridge[1])
    y1 = max(0, y_mid - 16)
    y2 = min(h_img, y_mid + 16)

    if x2 - x1 < 10 or y2 - y1 < 10:
        return False, "None"

    roi = frame[y1:y2, x1:x2]
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY) if len(roi.shape) == 3 else roi

    # Vertical gradient across horizontal bridge bar
    sobely = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    grad_y = float(np.mean(np.abs(sobely)))

    # Canny edge density across bridge
    edges = cv2.Canny(gray, 30, 85)
    edge_density = float(np.mean(edges > 0))

    # Specular reflection / glare on lenses
    glare_ratio = float(np.mean(gray > 210))

    # Overall glasses detection heuristic
    is_glasses = (grad_y > 8.5 and edge_density > 0.030) or (glare_ratio > 0.035) or (edge_density > 0.06)

    if is_glasses:
        label = "Glasses" if glare_ratio > 0.02 or edge_density < 0.15 else "Eyewear"
        return True, label

    return False, "None"


class FeatureExtractor:
    """
    Full CV Feature Extraction Engine.
    Integrates MediaPipe Face Landmarker with Blink/Yawn Tracker and solvePnP Head Pose.
    """

    def __init__(self, model_path: Optional[Path] = None, fps: float = config.DEFAULT_FPS):
        self.fps = fps
        self.tracker = BlinkAndYawnTracker(fps=fps)
        self.model_path = model_path or (config.MODELS_DIR / "face_landmarker.task")
        self.landmarker = None
        self._init_landmarker()

    def _init_landmarker(self) -> None:
        """Initializes MediaPipe FaceLandmarker with cached model."""
        if not self.model_path.exists():
            logger.warning(f"Face landmarker model not found at {self.model_path}. Auto-downloading...")
            try:
                import urllib.request
                url = "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task"
                urllib.request.urlretrieve(url, self.model_path)
                logger.info("Face landmarker task model downloaded successfully.")
            except Exception as e:
                logger.error(f"Failed to download FaceLandmarker model: {e}")
                return

        try:
            options = vision.FaceLandmarkerOptions(
                base_options=base_options.BaseOptions(model_asset_path=str(self.model_path)),
                running_mode=vision.RunningMode.IMAGE,
                num_faces=1,
                min_face_detection_confidence=0.5,
                min_face_presence_confidence=0.5,
                min_tracking_confidence=0.5,
                output_face_blendshapes=False,
                output_facial_transformation_matrixes=False,
            )
            self.landmarker = vision.FaceLandmarker.create_from_options(options)
            logger.info("MediaPipe FaceLandmarker initialized successfully.")
        except Exception as e:
            logger.error(f"MediaPipe FaceLandmarker initialization error: {e}")
            self.landmarker = None

    def process_frame(
        self,
        frame: np.ndarray,
        current_time: Optional[float] = None,
        draw_overlays: bool = True,
        apply_clahe: bool = config.ENABLE_AUTO_CLAHE,
    ) -> Tuple[Dict[str, float], np.ndarray, Dict[str, Any]]:
        """
        Processes a single BGR camera frame and returns:
        1. Feature dictionary matching config.FEATURE_COLUMNS (13 features)
        2. Annotated frame with overlays
        3. Raw telemetry & pose metadata
        """
        h, w = frame.shape[:2]
        t = current_time if current_time is not None else 0.0

        # 0. Dynamic Lighting Augmentation (CLAHE)
        enhanced_input_frame = frame
        clahe_applied = False
        luminance = 100.0
        if apply_clahe:
            enhanced_input_frame, clahe_applied, luminance = enhance_low_light_clahe(frame)

        annotated_frame = (enhanced_input_frame.copy() if clahe_applied else frame.copy()) if draw_overlays else frame

        landmarks_2d_dict = {}
        all_landmarks_px = []

        # 1. Detect Face Mesh Landmarks
        if self.landmarker is not None:
            try:
                rgb_frame = cv2.cvtColor(enhanced_input_frame, cv2.COLOR_BGR2RGB)
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
                detection_result = self.landmarker.detect(mp_image)

                if detection_result.face_landmarks:
                    face_lms = detection_result.face_landmarks[0]
                    for idx, lm in enumerate(face_lms):
                        px, py = float(lm.x * w), float(lm.y * h)
                        landmarks_2d_dict[idx] = (px, py)
                        all_landmarks_px.append((int(px), int(py)))
            except Exception as e:
                logger.debug(f"MediaPipe detection frame notice: {e}")

        # 2. Extract Geometric Metrics (EAR, MAR, Pose)
        if len(landmarks_2d_dict) >= 300:
            # Left & Right Eye Aspect Ratio
            left_eye_pts = [landmarks_2d_dict[i] for i in config.LEFT_EYE_LANDMARKS if i in landmarks_2d_dict]
            right_eye_pts = [landmarks_2d_dict[i] for i in config.RIGHT_EYE_LANDMARKS if i in landmarks_2d_dict]

            left_ear = calculate_ear(left_eye_pts)
            right_ear = calculate_ear(right_eye_pts)
            ear = float((left_ear + right_ear) / 2.0)

            # Mouth Aspect Ratio
            mouth_dict = {
                "left": landmarks_2d_dict.get(config.MOUTH_OUTER_CORNER_L, (0, 0)),
                "right": landmarks_2d_dict.get(config.MOUTH_OUTER_CORNER_R, (0, 0)),
                "top_lip": landmarks_2d_dict.get(config.MOUTH_TOP_LIP, (0, 0)),
                "bot_lip": landmarks_2d_dict.get(config.MOUTH_BOTTOM_LIP, (0, 0)),
                "top_outer": landmarks_2d_dict.get(config.MOUTH_TOP_OUTER, (0, 0)),
                "bot_outer": landmarks_2d_dict.get(config.MOUTH_BOTTOM_OUTER, (0, 0)),
            }
            mar = calculate_mar(mouth_dict)

            # Head Pose Estimation
            pitch, yaw, roll, face_angle, rvec, tvec, cam_mat, dist_coeff = estimate_head_pose(landmarks_2d_dict, frame.shape)

            # Visual Overlays
            if draw_overlays:
                # Draw eye contours
                for eye_indices in [config.LEFT_EYE_FULL, config.RIGHT_EYE_FULL]:
                    pts = np.array([landmarks_2d_dict[i] for i in eye_indices if i in landmarks_2d_dict], dtype=np.int32)
                    if len(pts) > 0:
                        cv2.polylines(annotated_frame, [pts], isClosed=True, color=(0, 255, 255), thickness=1, lineType=cv2.LINE_AA)

                # Draw mouth contour
                mouth_pts = np.array([landmarks_2d_dict[i] for i in config.MOUTH_LANDMARKS if i in landmarks_2d_dict], dtype=np.int32)
                if len(mouth_pts) > 0:
                    cv2.polylines(annotated_frame, [mouth_pts], isClosed=True, color=(255, 100, 0), thickness=1, lineType=cv2.LINE_AA)

                # Draw 3D Head Pose Axes on Nose
                draw_head_pose_axes(annotated_frame, rvec, tvec, cam_mat, dist_coeff, length=60.0)

        else:
            # Fallback values if face is not detected in frame
            ear = 0.30
            mar = 0.28
            pitch, yaw, roll, face_angle = 0.0, 0.0, 0.0, 0.0
            rvec, tvec, cam_mat, dist_coeff = np.zeros(3), np.zeros(3), np.eye(3), np.zeros((4, 1))

        # 3. Update Temporal Tracker (Blink, Yawn, PERCLOS, Velocity & Acceleration)
        tracker_res = self.tracker.update(ear=ear, mar=mar, current_time=t)

        # 4. Normalized Landmarks Summary for Client HUD Overlay
        landmarks_summary = {}
        if len(landmarks_2d_dict) >= 300:
            left_eye_norm = [[round(landmarks_2d_dict[i][0] / w, 4), round(landmarks_2d_dict[i][1] / h, 4)] for i in config.LEFT_EYE_FULL if i in landmarks_2d_dict]
            right_eye_norm = [[round(landmarks_2d_dict[i][0] / w, 4), round(landmarks_2d_dict[i][1] / h, 4)] for i in config.RIGHT_EYE_FULL if i in landmarks_2d_dict]
            mouth_norm = [[round(landmarks_2d_dict[i][0] / w, 4), round(landmarks_2d_dict[i][1] / h, 4)] for i in config.MOUTH_LANDMARKS if i in landmarks_2d_dict]
            nose_tip = [round(landmarks_2d_dict.get(1, (w / 2, h / 2))[0] / w, 4), round(landmarks_2d_dict.get(1, (w / 2, h / 2))[1] / h, 4)]
            
            all_x = [pt[0] for pt in landmarks_2d_dict.values()]
            all_y = [pt[1] for pt in landmarks_2d_dict.values()]
            face_box = [
                round(max(0.0, (min(all_x) - 10) / w), 4),
                round(max(0.0, (min(all_y) - 15) / h), 4),
                round(min(1.0, (max(all_x) + 10) / w), 4),
                round(min(1.0, (max(all_y) + 10) / h), 4)
            ]
            landmarks_summary = {
                "left_eye": left_eye_norm,
                "right_eye": right_eye_norm,
                "mouth": mouth_norm,
                "nose_tip": nose_tip,
                "face_box": face_box
            }

        # 5. Semantic Driving Orientation & Eyewear Analysis
        eyewear_detected, eyewear_label = detect_eyewear(frame, landmarks_2d_dict)
        head_pose_dir = get_head_pose_direction(pitch, yaw, roll)

        # 6. Assemble Standardized Feature Vector (13 Features)
        features = {
            "ear": float(ear),
            "mar": float(mar),
            "blink_duration": float(tracker_res["blink_duration"]),
            "blink_rate": float(tracker_res["blink_rate"]),
            "yawn_freq": float(tracker_res["yawn_freq"]),
            "eye_closure_dur": float(tracker_res["eye_closure_dur"]),
            "face_angle": float(face_angle),
            "head_pitch": float(pitch),
            "head_yaw": float(yaw),
            "head_roll": float(roll),
            "perclos": float(tracker_res["perclos"]),
            "ear_velocity": float(tracker_res["ear_velocity"]),
            "ear_acceleration": float(tracker_res["ear_acceleration"]),
        }

        telemetry = {
            "features": features,
            "blink_count": tracker_res["blink_count"],
            "yawn_count": tracker_res["yawn_count"],
            "is_yawning": tracker_res["is_yawning"],
            "landmarks_count": len(landmarks_2d_dict),
            "landmarks_summary": landmarks_summary,
            "head_pitch": pitch,
            "head_yaw": yaw,
            "head_roll": roll,
            "head_pose_direction": head_pose_dir,
            "eyewear_detected": eyewear_detected,
            "eyewear_label": eyewear_label,
            "ear_velocity": tracker_res["ear_velocity"],
            "ear_acceleration": tracker_res["ear_acceleration"],
            "clahe_applied": clahe_applied,
            "luminance": round(luminance, 1),
        }

        return features, annotated_frame, telemetry
