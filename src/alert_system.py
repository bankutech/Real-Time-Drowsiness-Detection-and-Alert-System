"""
Alert System Module.
Provides multi-tier audio/visual alarm dispatch, alert escalation, debouncing,
and structured telemetry event logging.
- Level 0: Normal / Safe (Silent, Green HUD)
- Level 1: Low / Moderate Fatigue (Warning Chime, Yellow HUD)
- Level 2: Critical Danger / Sleeping (Urgent Buzzer Alarm, Flashing Red HUD)
"""

import time
import os
import json
import threading
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple
from datetime import datetime

import numpy as np
import pandas as pd

from src import config
from src.utils import setup_logger

logger = setup_logger("AlertSystem")


class SoundGenerator:
    """
    Synthesizes and plays audio alert waveforms dynamically in memory.
    Uses pygame.mixer or winsound with graceful headless fallback.
    """

    def __init__(self, sample_rate: int = 22050):
        self.sample_rate = sample_rate
        self.audio_available = False
        self.pygame_initialized = False
        self.sound_cache: Dict[str, Any] = {}

        try:
            import pygame
            # Initialize pygame mixer safely
            pygame.mixer.init(frequency=sample_rate, size=-16, channels=2, buffer=512)
            self.pygame_initialized = True
            self.audio_available = True
            self._precompute_sounds()
            logger.info("Pygame audio mixer initialized successfully.")
        except Exception as e:
            logger.warning(f"Pygame audio unavailable ({e}). Checking OS fallback...")
            try:
                import winsound
                self.audio_available = True
                logger.info("Windows winsound audio backend available.")
            except Exception:
                logger.warning("Audio device unavailable. Running in visual-only mode.")

    def _precompute_sounds(self):
        """Generates raw PCM sine wave alert sounds."""
        import pygame

        # Warning Beep (880 Hz, 0.25s)
        t_warn = np.linspace(0, 0.25, int(self.sample_rate * 0.25), False)
        wave_warn = (np.sin(2 * np.pi * 880 * t_warn) * 32767 * 0.4).astype(np.int16)
        stereo_warn = np.column_stack([wave_warn, wave_warn])
        self.sound_cache["warning"] = pygame.sndarray.make_sound(stereo_warn)

        # Critical Alarm Tone (1200 Hz pulsed, 0.4s)
        t_crit = np.linspace(0, 0.4, int(self.sample_rate * 0.4), False)
        wave_crit = (np.sin(2 * np.pi * 1200 * t_crit) * 32767 * 0.7).astype(np.int16)
        stereo_crit = np.column_stack([wave_crit, wave_crit])
        self.sound_cache["critical"] = pygame.sndarray.make_sound(stereo_crit)

    def play_tone(self, alert_level: int):
        """Dispatches non-blocking audio alerts according to severity."""
        if not self.audio_available:
            return

        def _play():
            try:
                if self.pygame_initialized:
                    if alert_level == 1 and "warning" in self.sound_cache:
                        self.sound_cache["warning"].play()
                    elif alert_level == 2 and "critical" in self.sound_cache:
                        self.sound_cache["critical"].play()
                else:
                    import winsound
                    if alert_level == 1:
                        winsound.Beep(880, 200)
                    elif alert_level == 2:
                        winsound.Beep(1200, 400)
            except Exception as ex:
                logger.debug(f"Audio playback error: {ex}")

        threading.Thread(target=_play, daemon=True).start()


class AlertManager:
    """
    Manages alert state escalation, debounce thresholds, and event logging.
    """

    def __init__(
        self,
        warning_consecutive_frames: int = getattr(config, "DROWSY_EYE_CLOSURE_FRAMES", 12),
        critical_consecutive_frames: int = getattr(config, "SLEEP_EYE_CLOSURE_FRAMES", 30),
        log_dir: Optional[Path] = None,
        enable_audio: bool = getattr(config, "ENABLE_AUDIO_ALERT", True),
    ):
        self.warning_threshold = warning_consecutive_frames
        self.critical_threshold = critical_consecutive_frames
        self.enable_audio = enable_audio
        self.log_dir = log_dir or getattr(config, "OUTPUTS_DIR", Path("outputs"))
        self.log_dir.mkdir(parents=True, exist_ok=True)

        self.consecutive_drowsy = 0
        self.consecutive_sleeping = 0
        self.current_alert_level = 0
        self.current_status_text = "NORMAL"

        self.sound_gen = SoundGenerator() if enable_audio else None
        self.last_audio_trigger_time = 0.0
        self.audio_cooldown = 1.0  # seconds between repeated beeps

        self.event_log: List[Dict[str, Any]] = []
        self.log_file_csv = self.log_dir / "alert_log.csv"
        self.log_file_json = self.log_dir / "alert_events.json"

    def update(
        self,
        predicted_state: int,
        fatigue_score: float,
        ear: float,
        mar: float,
        perclos: float,
        head_pose: Tuple[float, float, float] = (0.0, 0.0, 0.0),
        frame_idx: int = 0,
    ) -> Dict[str, Any]:
        """
        Updates consecutive frame counters, evaluates debounce rules,
        triggers alarms, and logs events.
        """
        # State mapping: 0=Alert, 1=Drowsy, 2=Sleeping
        if predicted_state == 2:  # Sleeping
            self.consecutive_sleeping += 1
            self.consecutive_drowsy += 1
        elif predicted_state == 1:  # Drowsy
            self.consecutive_drowsy += 1
            self.consecutive_sleeping = max(0, self.consecutive_sleeping - 1)
        else:  # Alert
            self.consecutive_drowsy = max(0, self.consecutive_drowsy - 2)
            self.consecutive_sleeping = max(0, self.consecutive_sleeping - 3)

        # Determine Alert Level
        prev_level = self.current_alert_level
        if self.consecutive_sleeping >= self.critical_threshold or fatigue_score >= getattr(config, "CRITICAL_FATIGUE_THRESHOLD", 0.70):
            self.current_alert_level = 2
            self.current_status_text = "CRITICAL: SLEEPING DETECTED!"
        elif self.consecutive_drowsy >= self.warning_threshold or fatigue_score >= getattr(config, "DROWSY_FATIGUE_THRESHOLD", 0.45):
            self.current_alert_level = 1
            self.current_status_text = "WARNING: DROWSINESS DETECTED"
        else:
            self.current_alert_level = 0
            self.current_status_text = "DRIVER STATUS: ALERT"

        # Audio Dispatch with cooldown
        current_time = time.time()
        if self.current_alert_level > 0 and self.sound_gen:
            if (current_time - self.last_audio_trigger_time) >= self.audio_cooldown:
                self.sound_gen.play_tone(self.current_alert_level)
                self.last_audio_trigger_time = current_time

        # Structured Event Log entry
        event_entry = {
            "timestamp": datetime.now().isoformat(),
            "frame_idx": frame_idx,
            "predicted_state": int(predicted_state),
            "state_label": config.CLASS_LABELS[predicted_state],
            "alert_level": self.current_alert_level,
            "status_text": self.current_status_text,
            "fatigue_score": float(np.round(fatigue_score, 4)),
            "ear": float(np.round(ear, 4)),
            "mar": float(np.round(mar, 4)),
            "perclos": float(np.round(perclos, 4)),
            "pitch": float(np.round(head_pose[0], 2)),
            "yaw": float(np.round(head_pose[1], 2)),
            "roll": float(np.round(head_pose[2], 2)),
        }

        # Log transition or active warning/critical frames
        if self.current_alert_level > 0 or prev_level > 0 or (frame_idx % 30 == 0):
            self.event_log.append(event_entry)

        return event_entry

    def save_logs(self):
        """Flushes in-memory alert event records to disk."""
        if not self.event_log:
            logger.info("No alert events to save.")
            return

        df_events = pd.DataFrame(self.event_log)
        df_events.to_csv(self.log_file_csv, index=False)
        logger.info(f"Saved alert events CSV to {self.log_file_csv} ({len(df_events)} events)")

        with open(self.log_file_json, "w", encoding="utf-8") as f:
            json.dump(self.event_log, f, indent=2)
        logger.info(f"Saved alert events JSON to {self.log_file_json}")

    def reset(self):
        """Resets debounce counters and active alert levels."""
        self.consecutive_drowsy = 0
        self.consecutive_sleeping = 0
        self.current_alert_level = 0
        self.current_status_text = "DRIVER STATUS: ALERT"
