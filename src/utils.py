"""
Utility functions for Real-Time Drowsiness Detection and Alert System.
Includes logging configuration, sound/alarm management, HUD drawing utilities,
geometry helpers, and model serialization.
"""

import os
import sys
import time
import math
import wave
import struct
import logging
import threading
from pathlib import Path
from datetime import datetime
from typing import Tuple, List, Optional, Dict, Any

import cv2
import numpy as np
import joblib

os.environ["PYGAME_HIDE_SUPPORT_PROMPT"] = "1"

try:
    import pygame
    PYGAME_AVAILABLE = True
except ImportError:
    PYGAME_AVAILABLE = False

from src import config

# =====================================================================
# 1. Logging Setup
# =====================================================================
def setup_logger(name: str = "DrowsinessSystem", log_file: Optional[Path] = None, level: int = logging.INFO) -> logging.Logger:
    """
    Configures and returns a thread-safe logger with console and file handlers.
    """
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger

    logger.setLevel(level)
    formatter = logging.Formatter(
        "[%(asctime)s] [%(levelname)s] [%(name)s:%(funcName)s]: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Console Handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    console_handler.setLevel(level)
    logger.addHandler(console_handler)

    # File Handler
    target_file = log_file or config.SYSTEM_LOG_PATH
    target_file.parent.mkdir(parents=True, exist_ok=True)
    file_handler = logging.FileHandler(target_file, encoding="utf-8")
    file_handler.setFormatter(formatter)
    file_handler.setLevel(level)
    logger.addHandler(file_handler)

    return logger

logger = setup_logger("SystemUtils")


def log_drowsiness_event(
    frame_number: int,
    state: str,
    fatigue_score: float,
    ear: float,
    mar: float,
    perclos: float,
    action_taken: str = "ALARM_TRIGGERED",
    log_path: Optional[Path] = None,
) -> None:
    """
    Appends a structured event entry to the CSV/JSON drowsiness event log.
    """
    target_path = log_path or config.EVENTS_LOG_PATH
    file_exists = target_path.exists()
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]

    with open(target_path, "a", encoding="utf-8") as f:
        if not file_exists:
            f.write("timestamp,frame_number,state,fatigue_score,ear,mar,perclos,action_taken\n")
        f.write(
            f"{timestamp},{frame_number},{state},{fatigue_score:.2f},"
            f"{ear:.4f},{mar:.4f},{perclos:.4f},{action_taken}\n"
        )


# =====================================================================
# 2. Sound & Alarm Manager (Thread-Safe Audio Synthesizer)
# =====================================================================
class SoundManager:
    """
    Manages non-blocking audio alarm synthesis and playback using Pygame Mixer
    with robust fallback to Windows winsound / system alert.
    """

    def __init__(self, frequency: int = config.ALARM_BEEP_FREQUENCY, duration_ms: int = config.ALARM_BEEP_DURATION_MS):
        self.frequency = frequency
        self.duration_ms = duration_ms
        self.is_playing = False
        self.last_played_time = 0.0
        self.lock = threading.Lock()
        self.sound_object = None
        self._pygame_initialized = False
        self._init_attempted = False

    def _ensure_init(self) -> None:
        """Lazily initializes the audio mixer when first needed."""
        if self._init_attempted:
            return
        self._init_attempted = True
        if PYGAME_AVAILABLE:
            try:
                if not pygame.mixer.get_init():
                    pygame.mixer.init(frequency=44100, size=-16, channels=1, buffer=512)
                self._pygame_initialized = True
                self.sound_object = self._generate_sine_sound(self.frequency, self.duration_ms)
            except Exception as e:
                logger.warning(f"Pygame audio mixer init fallback: {e}")
                self._pygame_initialized = False

    def _generate_sine_sound(self, freq: int, duration_ms: int) -> Optional[Any]:
        """Generates a pure sine wave Pygame Sound object dynamically."""
        if not self._pygame_initialized:
            return None
        try:
            sample_rate = 44100
            n_samples = int(sample_rate * (duration_ms / 1000.0))
            buf = bytearray()
            for i in range(n_samples):
                t = float(i) / sample_rate
                # Sine wave with gentle attack/decay envelope
                env = min(1.0, min(i / 200.0, (n_samples - i) / 200.0))
                value = int(32767.0 * 0.7 * env * math.sin(2.0 * math.pi * freq * t))
                buf.extend(struct.pack("<h", value))
            return pygame.mixer.Sound(buffer=bytes(buf))
        except Exception as e:
            logger.warning(f"Failed to generate sine sound: {e}")
            return None

    def play_alarm(self, force: bool = False) -> None:
        """Plays the alarm sound asynchronously in a background thread."""
        self._ensure_init()
        current_time = time.time()
        if not force and (current_time - self.last_played_time < config.ALARM_COOLDOWN_SECONDS):
            return

        with self.lock:
            if self.is_playing:
                return
            self.is_playing = True
            self.last_played_time = current_time

        threading.Thread(target=self._play_worker, daemon=True).start()

    def _play_worker(self) -> None:
        try:
            if self._pygame_initialized and self.sound_object is not None:
                self.sound_object.play()
                time.sleep(self.duration_ms / 1000.0)
            elif sys.platform == "win32":
                import winsound
                winsound.Beep(self.frequency, self.duration_ms)
            else:
                sys.stdout.write("\a")
                sys.stdout.flush()
        except Exception as e:
            logger.debug(f"Audio playback notice: {e}")
        finally:
            with self.lock:
                self.is_playing = False

    def stop(self) -> None:
        if self._pygame_initialized:
            try:
                pygame.mixer.stop()
            except Exception:
                pass


# Global singleton sound manager instance
sound_manager = SoundManager()


# =====================================================================
# 3. Geometry & Math Helpers
# =====================================================================
def euclidean_distance_2d(p1: Tuple[float, float], p2: Tuple[float, float]) -> float:
    """Computes 2D Euclidean distance between two coordinate tuples."""
    return math.hypot(p1[0] - p2[0], p1[1] - p2[1])


def euclidean_distance_3d(p1: Tuple[float, float, float], p2: Tuple[float, float, float]) -> float:
    """Computes 3D Euclidean distance between two points."""
    return math.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2 + (p1[2] - p2[2]) ** 2)


def clamp(val: float, min_val: float, max_val: float) -> float:
    """Clamps a numeric value between min_val and max_val."""
    return max(min_val, min(max_val, val))


# =====================================================================
# 4. Computer Vision & UI Drawing Helpers
# =====================================================================
def draw_rounded_rect(
    img: np.ndarray,
    top_left: Tuple[int, int],
    bottom_right: Tuple[int, int],
    color: Tuple[int, int, int],
    radius: int = 10,
    thickness: int = -1,
) -> np.ndarray:
    """Draws a sleek rounded rectangle with anti-aliasing."""
    x1, y1 = top_left
    x2, y2 = bottom_right
    w = x2 - x1
    h = y2 - y1
    radius = min(radius, w // 2, h // 2)

    if thickness < 0:
        # Filled
        cv2.rectangle(img, (x1 + radius, y1), (x2 - radius, y2), color, -1)
        cv2.rectangle(img, (x1, y1 + radius), (x2, y2 - radius), color, -1)
        cv2.circle(img, (x1 + radius, y1 + radius), radius, color, -1)
        cv2.circle(img, (x2 - radius, y1 + radius), radius, color, -1)
        cv2.circle(img, (x1 + radius, y2 - radius), radius, color, -1)
        cv2.circle(img, (x2 - radius, y2 - radius), radius, color, -1)
    else:
        # Outline
        cv2.line(img, (x1 + radius, y1), (x2 - radius, y1), color, thickness)
        cv2.line(img, (x1 + radius, y2), (x2 - radius, y2), color, thickness)
        cv2.line(img, (x1, y1 + radius), (x1, y2 - radius), color, thickness)
        cv2.line(img, (x2, y1 + radius), (x2, y2 - radius), color, thickness)
        cv2.ellipse(img, (x1 + radius, y1 + radius), (radius, radius), 180, 0, 90, color, thickness)
        cv2.ellipse(img, (x2 - radius, y1 + radius), (radius, radius), 270, 0, 90, color, thickness)
        cv2.ellipse(img, (x1 + radius, y2 - radius), (radius, radius), 90, 0, 90, color, thickness)
        cv2.ellipse(img, (x2 - radius, y2 - radius), (radius, radius), 0, 0, 90, color, thickness)
    return img


def draw_hud_header(
    frame: np.ndarray,
    title: str = "Real-Time Drowsiness Detection System",
    fps: float = 30.0,
    driver_state: str = "Alert",
    fatigue_score: float = 0.0,
) -> np.ndarray:
    """Draws a modern frosted glass HUD header banner across the top of the frame."""
    h, w = frame.shape[:2]
    header_height = 70

    # Semi-transparent dark background banner
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, header_height), (20, 24, 30), -1)
    cv2.addWeighted(overlay, 0.75, frame, 0.25, 0, frame)

    # Accent bottom line
    state_color = config.CLASS_COLORS_BGR.get(driver_state, (50, 205, 50))
    cv2.line(frame, (0, header_height), (w, header_height), state_color, 2)

    # Title text
    cv2.putText(
        frame,
        title,
        (20, 30),
        cv2.FONT_HERSHEY_DUPLEX,
        0.7,
        (240, 240, 240),
        1,
        cv2.LINE_AA,
    )

    # FPS Counter
    fps_text = f"FPS: {fps:.1f}"
    cv2.putText(
        frame,
        fps_text,
        (20, 55),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        (160, 180, 200),
        1,
        cv2.LINE_AA,
    )

    # Driver State Badge (Right Side)
    badge_w, badge_h = 240, 44
    bx1 = w - badge_w - 20
    by1 = 13
    bx2 = bx1 + badge_w
    by2 = by1 + badge_h

    # Badge background
    draw_rounded_rect(frame, (bx1, by1), (bx2, by2), (35, 40, 50), radius=8, thickness=-1)
    draw_rounded_rect(frame, (bx1, by1), (bx2, by2), state_color, radius=8, thickness=2)

    # Status text
    status_str = f"STATUS: {driver_state.upper()}"
    cv2.putText(
        frame,
        status_str,
        (bx1 + 14, by1 + 28),
        cv2.FONT_HERSHEY_DUPLEX,
        0.55,
        state_color,
        1,
        cv2.LINE_AA,
    )

    return frame


def draw_telemetry_card(
    frame: np.ndarray,
    metrics: Dict[str, Any],
    driver_state: str = "Alert",
    fatigue_score: float = 0.0,
) -> np.ndarray:
    """Draws a compact, polished telemetry card on the left side of the frame."""
    h, w = frame.shape[:2]
    card_w, card_h = 290, 290
    cx1, cy1 = 20, 85
    cx2, cy2 = cx1 + card_w, cy1 + card_h

    # Frosted background
    overlay = frame.copy()
    draw_rounded_rect(overlay, (cx1, cy1), (cx2, cy2), (18, 22, 28), radius=10, thickness=-1)
    cv2.addWeighted(overlay, 0.80, frame, 0.20, 0, frame)
    draw_rounded_rect(frame, (cx1, cy1), (cx2, cy2), (60, 75, 95), radius=10, thickness=1)

    # Card Title
    cv2.putText(frame, "TELEMETRY", (cx1 + 15, cy1 + 25), cv2.FONT_HERSHEY_DUPLEX, 0.55, (220, 225, 235), 1, cv2.LINE_AA)

    # Fatigue Gauge Bar
    f_bar_y = cy1 + 52
    f_score_clamped = clamp(fatigue_score, 0.0, 100.0)
    cv2.putText(frame, f"Fatigue Score: {f_score_clamped:.1f}%", (cx1 + 15, f_bar_y - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (200, 200, 200), 1, cv2.LINE_AA)
    bar_w = card_w - 30
    cv2.rectangle(frame, (cx1 + 15, f_bar_y), (cx1 + 15 + bar_w, f_bar_y + 10), (45, 52, 65), -1)
    fill_w = int((f_score_clamped / 100.0) * bar_w)
    state_color = config.CLASS_COLORS_BGR.get(driver_state, (50, 205, 50))
    cv2.rectangle(frame, (cx1 + 15, f_bar_y), (cx1 + 15 + fill_w, f_bar_y + 10), state_color, -1)

    # Metrics listing
    ear_val = metrics.get("ear", 0.0)
    mar_val = metrics.get("mar", 0.0)
    blinks = metrics.get("blink_count", 0)
    blink_rate = metrics.get("blink_rate", 0.0)
    perclos = metrics.get("perclos", 0.0)
    pitch = metrics.get("head_pitch", 0.0)
    yaw = metrics.get("head_yaw", 0.0)

    items = [
        (f"EAR (Eye Ratio):", f"{ear_val:.3f}", (0, 255, 0) if ear_val >= config.EAR_DROWSY_THRESH else (0, 0, 255)),
        (f"MAR (Mouth Ratio):", f"{mar_val:.3f}", (0, 0, 255) if mar_val >= config.MAR_YAWN_THRESH else (200, 200, 200)),
        (f"PERCLOS (Eye Closed):", f"{perclos * 100:.1f}%", (0, 0, 255) if perclos > 0.30 else (200, 200, 200)),
        (f"Blinks / Rate:", f"{blinks} ({blink_rate:.1f}/min)", (200, 200, 200)),
        (f"Head Pitch / Yaw:", f"{pitch:.1f}deg / {yaw:.1f}deg", (200, 200, 200)),
        (f"HMM Smoothed:", f"{metrics.get('hmm_state', driver_state)}", state_color),
    ]

    row_y = f_bar_y + 32
    for label, val_str, val_color in items:
        cv2.putText(frame, label, (cx1 + 15, row_y), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (170, 180, 195), 1, cv2.LINE_AA)
        cv2.putText(frame, val_str, (cx1 + 165, row_y), cv2.FONT_HERSHEY_SIMPLEX, 0.42, val_color, 1, cv2.LINE_AA)
        row_y += 24

    return frame


def draw_alarm_banner(frame: np.ndarray, state: str, warning_text: str = "WARNING: DROWSINESS DETECTED!") -> np.ndarray:
    """Flashes an impossible-to-miss red/orange warning banner when driver is drowsy or asleep."""
    h, w = frame.shape[:2]
    banner_h = 75
    y1 = h - banner_h - 20
    y2 = h - 20
    x1 = 40
    x2 = w - 40

    overlay = frame.copy()
    bg_color = (0, 0, 200) if state == "Sleep" else (0, 110, 220)
    draw_rounded_rect(overlay, (x1, y1), (x2, y2), bg_color, radius=12, thickness=-1)
    cv2.addWeighted(overlay, 0.88, frame, 0.12, 0, frame)
    draw_rounded_rect(frame, (x1, y1), (x2, y2), (255, 255, 255), radius=12, thickness=3)

    # Flashing text
    msg = f"*** {warning_text.upper()} - PULL OVER SAFELY ***"
    text_size = cv2.getTextSize(msg, cv2.FONT_HERSHEY_DUPLEX, 0.75, 2)[0]
    text_x = x1 + (x2 - x1 - text_size[0]) // 2
    text_y = y1 + (banner_h + text_size[1]) // 2
    cv2.putText(frame, msg, (text_x, text_y), cv2.FONT_HERSHEY_DUPLEX, 0.75, (255, 255, 255), 2, cv2.LINE_AA)

    return frame


def draw_head_pose_axes(
    frame: np.ndarray,
    rotation_vec: np.ndarray,
    translation_vec: np.ndarray,
    camera_matrix: np.ndarray,
    dist_coeffs: np.ndarray,
    length: float = 50.0,
) -> np.ndarray:
    """Projects and draws 3D Cartesian orientation axes (X=Red, Y=Green, Z=Blue) on the nose."""
    try:
        axis_3d = np.float32([[length, 0, 0], [0, length, 0], [0, 0, length], [0, 0, 0]]).reshape(-1, 3)
        imgpts, _ = cv2.projectPoints(axis_3d, rotation_vec, translation_vec, camera_matrix, dist_coeffs)
        origin = tuple(imgpts[3].ravel().astype(int))
        x_axis = tuple(imgpts[0].ravel().astype(int))
        y_axis = tuple(imgpts[1].ravel().astype(int))
        z_axis = tuple(imgpts[2].ravel().astype(int))

        cv2.line(frame, origin, x_axis, (0, 0, 255), 2, cv2.LINE_AA)  # Pitch / X
        cv2.line(frame, origin, y_axis, (0, 255, 0), 2, cv2.LINE_AA)  # Yaw / Y
        cv2.line(frame, origin, z_axis, (255, 0, 0), 2, cv2.LINE_AA)  # Roll / Z
    except Exception:
        pass
    return frame


# =====================================================================
# 5. Model Serialization Helpers
# =====================================================================
def save_model(model_obj: Any, filename: str, directory: Optional[Path] = None) -> Path:
    """Serializes and saves a model artifact using joblib."""
    target_dir = directory or config.MODELS_DIR
    target_dir.mkdir(parents=True, exist_ok=True)
    file_path = target_dir / filename
    joblib.dump(model_obj, file_path)
    logger.info(f"Saved model artifact: {file_path}")
    return file_path


def load_model(filename: str, directory: Optional[Path] = None) -> Any:
    """Loads a serialized model artifact."""
    target_dir = directory or config.MODELS_DIR
    file_path = target_dir / filename
    if not file_path.exists():
        raise FileNotFoundError(f"Model file not found: {file_path}")
    model = joblib.load(file_path)
    logger.debug(f"Loaded model artifact: {file_path}")
    return model
