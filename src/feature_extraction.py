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

    p1, p2, p3, p4, p5, p6 = eye_landmarks[:6]

    # Vertical eye distances
    dist_v1 = euclidean_distance_2d(p2, p6)
    dist_v2 = euclidean_distance_2d(p3, p5)

    # Horizontal eye distance
    dist_h = euclidean_distance_2d(p1, p4)

    if dist_h < 1e-6:
        return 0.0

    ear = (dist_v1 + dist_v2) / (2.0 * dist_h)
    return float(ear)


def calculate_mar(mouth_landmarks: Dict[str, Tuple[float, float]]) -> float:
    """
    Computes Mouth Aspect Ratio (MAR) using vertical lip opening vs horizontal width.
    Formula: MAR = (||top_outer - bot_outer|| + ||top_inner - bot_inner||) / (2 * ||left - right||)
    """
    try:
        p_left = mouth_landmarks["left"]
        p_right = mouth_landmarks["right"]
        p_top_lip = mouth_landmarks["top_lip"]
        p_bot_lip = mouth_landmarks["bot_lip"]
        p_top_outer = mouth_landmarks["top_outer"]
        p_bot_outer = mouth_landmarks["bot_outer"]

        dist_v1 = euclidean_distance_2d(p_top_lip, p_bot_lip)
        dist_v2 = euclidean_distance_2d(p_top_outer, p_bot_outer)
        dist_h = euclidean_distance_2d(p_left, p_right)

        if dist_h < 1e-6:
            return 0.0

        mar = (dist_v1 + dist_v2) / (2.0 * dist_h)
        return float(mar)
    except KeyError:
        return 0.0


class BlinkAndYawnTracker:
    """
    Maintains temporal sliding windows and state machines to track:
    - Blink occurrences, blink duration (sec), blink rate (blinks/min)
    - Continuous eye closure duration (sec)
    - Yawn occurrences and frequency
    - PERCLOS (Percentage of Eye Closure over sliding window)
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

        return {
            "perclos": perclos,
            "blink_count": self.total_blinks,
            "blink_duration": self.last_blink_duration,
            "blink_rate": blink_rate,
            "eye_closure_dur": continuous_closure_sec,
            "yawn_count": self.total_yawns,
            "yawn_freq": yawn_freq,
            "is_yawning": self.is_yawning,
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


def estimate_head_pose(
    landmarks_2d: Dict[int, Tuple[float, float]],
    frame_shape: Tuple[int, int, int],
) -> Tuple[float, float, float, float, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Calculates Head Pose Euler Angles (Pitch, Yaw, Roll) via cv2.solvePnP
    using 3D canonical facial model points.
    """
    h, w = frame_shape[:2]

    # Focal length and optical center approximation
    focal_length = float(w)
    center = (w / 2.0, h / 2.0)
    camera_matrix = np.array(
        [[focal_length, 0, center[0]], [0, focal_length, center[1]], [0, 0, 1]],
        dtype=np.float64,
    )
    dist_coeffs = np.zeros((4, 1), dtype=np.float64)

    # 3D Model Points
    model_points_3d = np.array(config.CANONICAL_3D_FACE_MODEL, dtype=np.float64)

    # 2D Image Points
    image_points_2d = []
    for lm_id in config.HEAD_POSE_LANDMARK_IDS:
        if lm_id in landmarks_2d:
            image_points_2d.append(landmarks_2d[lm_id])
        else:
            # Fallback to center if landmark missing
            image_points_2d.append((w / 2.0, h / 2.0))

    image_points_2d = np.array(image_points_2d, dtype=np.float64)

    # Solve PnP
    success, rotation_vec, translation_vec = cv2.solvePnP(
        model_points_3d,
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
    # RQDecomp3x3 decomposes a 3x3 rotation matrix into 3 Euler angles
    euler_angles, _, _, _, _, _ = cv2.RQDecomp3x3(rot_mat)
    pitch = float(euler_angles[0])
    yaw = float(euler_angles[1])
    roll = float(euler_angles[2])

    # Face Angle magnitude
    face_angle = float(math.sqrt(pitch**2 + yaw**2 + roll**2))

    return pitch, yaw, roll, face_angle, rotation_vec, translation_vec, camera_matrix, dist_coeffs


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
    ) -> Tuple[Dict[str, float], np.ndarray, Dict[str, Any]]:
        """
        Processes a single BGR camera frame and returns:
        1. Feature dictionary matching config.FEATURE_COLUMNS
        2. Annotated frame with overlays
        3. Raw telemetry & pose metadata
        """
        h, w = frame.shape[:2]
        t = current_time if current_time is not None else 0.0
        annotated_frame = frame.copy() if draw_overlays else frame

        landmarks_2d_dict = {}
        all_landmarks_px = []

        # 1. Detect Face Mesh Landmarks
        if self.landmarker is not None:
            try:
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
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

        # 3. Update Temporal Tracker (Blink, Yawn, PERCLOS)
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

        # 5. Assemble Standardized Feature Vector
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
        }

        return features, annotated_frame, telemetry
