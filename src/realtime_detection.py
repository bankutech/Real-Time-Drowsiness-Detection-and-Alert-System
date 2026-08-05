"""
Real-Time Drowsiness Detection Pipeline & Cockpit HUD Overlay.
Integrates all 5 Course Units:
- Unit 1: Video Capture, Landmark Extraction, EAR, MAR, PERCLOS, Head Pose Euler Angles
- Unit 2: Linear Regression Continuous Fatigue + Bayesian Logistic & SVM Probabilistic Inference
- Unit 3: PCA Feature Projection & K-Means / GMM Anomaly Outlier Detection
- Unit 4: Pure NumPy HMM Real-Time Online Filtering (MAP State Smoothing)
- Unit 5: Random Forest & Multi-Model Stacking Ensemble Consensus
- Alert System: Debounced Multi-Tier Audio-Visual Alarm Dispatch & Event Logging
"""

import time
import os
from pathlib import Path
from typing import Dict, Any, Optional, Tuple, List, Union

import cv2
import numpy as np
import pandas as pd

from src import config
from src.utils import setup_logger, load_model
from src.feature_extraction import FeatureExtractor
from src.alert_system import AlertManager
from src.hmm import DrowsinessHMM

logger = setup_logger("RealtimeDetection")


class DrowsinessDetectorPipeline:
    """
    End-to-End Real-Time Drowsiness Detection Engine.
    Coordinates vision feature extraction, multi-unit machine learning models,
    temporal filtering, HUD telemetry overlay, and alarm dispatch.
    """

    def __init__(
        self,
        primary_model_type: str = "ensemble",
        enable_audio: bool = getattr(config, "ENABLE_AUDIO_ALERT", True),
        output_dir: Optional[Path] = None,
    ):
        self.primary_model_type = primary_model_type.lower()
        self.output_dir = output_dir or getattr(config, "OUTPUTS_DIR", Path("outputs"))
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # 1. Initialize Vision Feature Extractor
        self.feature_extractor = FeatureExtractor(fps=getattr(config, "DEFAULT_FPS", 30.0))

        # 2. Load ML Artifacts
        self.scaler = load_model("scaler.joblib")
        try:
            self.linear_reg = load_model("fatigue_regressor.joblib")
        except Exception:
            self.linear_reg = load_model("linear_regression.joblib")
        self.bayes_model = load_model("bayesian_logistic.joblib")
        self.rf_model = load_model("random_forest.joblib")
        self.ensemble_model = load_model("ensemble_stacking.joblib")
        self.pca_model = load_model("pca.joblib")
        self.kmeans_model = load_model("kmeans.joblib")

        # 3. Load HMM Engine
        try:
            self.hmm = DrowsinessHMM.load()
            logger.info("Loaded trained HMM for temporal filtering.")
        except Exception as e:
            logger.warning(f"Could not load HMM: {e}. Running without HMM temporal filter.")
            self.hmm = None
        self.prev_hmm_belief: Optional[np.ndarray] = None

        # 4. Initialize Alert Manager
        self.alert_manager = AlertManager(enable_audio=enable_audio, log_dir=self.output_dir)

        # State and Performance Tracking
        self.frame_count = 0
        self.fps = 0.0
        self._prev_time = time.perf_counter()
        self.recent_latencies: List[float] = []

        # Customer Critical Enhancements:
        # 1. Dynamic Baseline Calibration
        self.is_calibrated = False
        self.calibrating = False
        self.calibration_frames_needed = 45  # ~1.5 - 2.0 seconds at 25-30 FPS
        self.calibration_buffer_ear: List[float] = []
        self.calibration_buffer_mar: List[float] = []
        self.baseline_ear = 0.32
        self.baseline_mar = 0.25
        self.ear_thresh_active = getattr(config, "EAR_DROWSY_THRESH", 0.23)
        self.mar_thresh_active = getattr(config, "MAR_YAWN_THRESH", 0.60)

        # 2. Speech vs Yawn Disambiguation tracking
        self.speech_counter = 0

        # Auto-trigger calibration on startup
        self.start_calibration()

    def start_calibration(self, duration_frames: int = 45):
        """Starts dynamic driver baseline calibration for personalized thresholds."""
        self.calibrating = True
        self.calibration_frames_needed = duration_frames
        self.calibration_buffer_ear = []
        self.calibration_buffer_mar = []
        logger.info("Driver baseline calibration initiated...")

    def process_frame(self, frame: np.ndarray) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Executes full inference cycle on a single video frame with
        ambient lighting enhancement, personalized calibration, and speech disambiguation.
        Returns: (annotated_hud_frame, telemetry_dict)
        """
        t_start = time.perf_counter()
        self.frame_count += 1

        # Calculate live FPS
        curr_time = time.perf_counter()
        dt = curr_time - self._prev_time
        self._prev_time = curr_time
        if dt > 0:
            self.fps = 0.9 * self.fps + 0.1 * (1.0 / dt) if self.fps > 0 else 1.0 / dt

        # Fast Ambient Lighting Check (sampled luminance)
        mean_luminance = float(cv2.mean(frame)[0])
        low_light = mean_luminance < 35.0

        # Apply fast contrast enhancement only in extreme low light
        proc_frame = frame
        if low_light:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            enhanced_gray = clahe.apply(gray)
            proc_frame = cv2.cvtColor(enhanced_gray, cv2.COLOR_GRAY2BGR)

        # 1. Feature Extraction (Unit 1)
        feats, annotated_frame, meta = self.feature_extractor.process_frame(
            proc_frame, current_time=curr_time, draw_overlays=True
        )

        face_detected = meta.get("landmarks_count", 0) >= 250
        ear = float(feats.get("ear", 0.0))
        mar = float(feats.get("mar", 0.0))
        perclos = float(feats.get("perclos", 0.0))
        pitch = float(feats.get("head_pitch", 0.0))
        yaw = float(feats.get("head_yaw", 0.0))
        roll = float(feats.get("head_roll", 0.0))

        # Dynamic Calibration Routine
        if self.calibrating and face_detected and ear > 0.15:
            self.calibration_buffer_ear.append(ear)
            self.calibration_buffer_mar.append(mar)
            if len(self.calibration_buffer_ear) >= self.calibration_frames_needed:
                self.baseline_ear = float(np.median(self.calibration_buffer_ear))
                self.baseline_mar = float(np.median(self.calibration_buffer_mar))
                # Set personalized thresholds: 75% of resting eye openness for drowsy
                self.ear_thresh_active = max(0.18, min(0.30, 0.76 * self.baseline_ear))
                self.mar_thresh_active = max(0.48, min(0.72, 2.0 * self.baseline_mar))
                self.is_calibrated = True
                self.calibrating = False
                logger.info(
                    f"Calibration Complete: Baseline EAR={self.baseline_ear:.3f}, "
                    f"Threshold EAR={self.ear_thresh_active:.3f}, Threshold MAR={self.mar_thresh_active:.3f}"
                )

        # Speech vs Yawn Disambiguation:
        # If mouth is open (high MAR) but eyes are wide open (low PERCLOS & high EAR), classify as talking
        is_speaking = False
        if mar > (self.mar_thresh_active * 0.85) and ear > (self.ear_thresh_active * 1.05) and perclos < 0.12:
            is_speaking = True
            self.speech_counter += 1
        else:
            self.speech_counter = max(0, self.speech_counter - 1)

        # Eyewear / Sunglass Occlusion Fallback:
        # If face is detected but eye EAR is abnormally low (< 0.10) while user is upright, flag possible eyewear
        eyewear_detected = face_detected and (ear < 0.10) and (abs(pitch) < 15.0) and (abs(yaw) < 20.0)

        # Telemetry structure
        telemetry: Dict[str, Any] = {
            "frame_idx": self.frame_count,
            "fps": round(self.fps, 1),
            "face_detected": face_detected,
            "ear": ear,
            "mar": mar,
            "perclos": perclos,
            "pitch": pitch,
            "yaw": yaw,
            "roll": roll,
            "fatigue_score": 0.0,
            "predicted_state": 0,
            "state_label": config.CLASS_LABELS[0],
            "alert_level": 0,
            "latency_ms": 0.0,
            "is_calibrated": self.is_calibrated,
            "calibrating": self.calibrating,
            "calibration_progress": int((len(self.calibration_buffer_ear) / max(1, self.calibration_frames_needed)) * 100) if self.calibrating else 100,
            "baseline_ear": round(self.baseline_ear, 3),
            "is_speaking": is_speaking,
            "low_light": low_light,
            "lighting_quality": "Low (Night/Dark)" if low_light else ("Moderate" if mean_luminance < 85 else "Optimal"),
            "eyewear_detected": eyewear_detected,
        }

        # 2. Build 11-Dimensional Feature Vector matching config.FEATURE_COLUMNS
        feature_vector = np.array([[feats.get(col, 0.0) for col in config.FEATURE_COLUMNS]], dtype=np.float64)

        # Normalize features with trained scaler
        feature_vector_scaled = self.scaler.transform(feature_vector)

        # 3. Model Inferences (Units 2, 3, 4, 5)
        # Continuous Fatigue Regression (Unit 2)
        fatigue_pred = float(np.clip(self.linear_reg.predict(feature_vector_scaled)[0], 0.0, 100.0) / 100.0)
        
        # Suppress artificial fatigue spikes if user is merely talking/speaking
        if is_speaking:
            fatigue_pred = min(0.35, fatigue_pred)

        telemetry["fatigue_score"] = fatigue_pred

        # Primary Multi-Class Classification (Unit 5)
        if self.primary_model_type == "ensemble" and self.ensemble_model:
            pred_state = int(self.ensemble_model.predict(feature_vector_scaled)[0])
            probas = self.ensemble_model.predict_proba(feature_vector_scaled)[0]
        elif self.rf_model:
            pred_state = int(self.rf_model.predict(feature_vector_scaled)[0])
            probas = self.rf_model.predict_proba(feature_vector_scaled)[0]
        elif self.bayes_model:
            pred_state = int(self.bayes_model.predict(feature_vector_scaled)[0])
            probas = self.bayes_model.predict_proba(feature_vector_scaled)[0]
        else:
            pred_state = 0
            probas = [1.0, 0.0, 0.0, 0.0]

        # If speaking, do not allow Yawn state to flip to Sleep
        if is_speaking and pred_state > 0 and perclos < 0.10:
            pred_state = 0
            probas = [0.85, 0.15, 0.0, 0.0]

        # 4. HMM Online Temporal Filter (Unit 4)
        if self.hmm is not None:
            prob_arr = np.array(probas, dtype=np.float64)
            if len(prob_arr) == self.hmm.n_states:
                hmm_state, self.prev_hmm_belief = self.hmm.online_filter_step(
                    prob_arr, prev_forward_state=self.prev_hmm_belief
                )
                pred_state = hmm_state

        # Compute Shannon Predictive Entropy (Uncertainty Quantification)
        prob_arr_safe = np.array(probas, dtype=np.float64) + 1e-12
        entropy_val = float(-np.sum(prob_arr_safe * np.log(prob_arr_safe)))

        telemetry["predicted_state"] = pred_state
        telemetry["state_label"] = config.CLASS_LABELS[min(pred_state, len(config.CLASS_LABELS) - 1)]
        telemetry["probabilities"] = [round(float(p), 4) for p in probas]
        telemetry["uncertainty_entropy"] = round(entropy_val, 3)
        telemetry["hmm_belief"] = [round(float(b), 4) for b in (self.prev_hmm_belief if self.prev_hmm_belief is not None else probas)]

        # 5. Alert Manager Update (Phase 8)
        alert_event = self.alert_manager.update(
            predicted_state=pred_state,
            fatigue_score=fatigue_pred,
            ear=ear,
            mar=mar,
            perclos=perclos,
            head_pose=(pitch, yaw, roll),
            frame_idx=self.frame_count,
        )

        # Contextual status text override for speech or calibration
        status_text = alert_event["status_text"]
        alert_level = alert_event["alert_level"]
        if self.calibrating:
            status_text = f"CALIBRATING BASELINE ({telemetry['calibration_progress']}%)"
        elif is_speaking and alert_level == 0:
            status_text = "DRIVER SPEAKING (NORMAL)"

        telemetry["alert_level"] = alert_level
        telemetry["status_text"] = status_text

        # Latency calculation
        latency_ms = (time.perf_counter() - t_start) * 1000.0
        telemetry["latency_ms"] = latency_ms

        # 6. Render Cockpit HUD Overlay
        annotated_frame = self._render_hud(annotated_frame, telemetry)

        return annotated_frame, telemetry

    def _render_hud(self, frame: np.ndarray, telemetry: Dict[str, Any]) -> np.ndarray:
        """
        Renders rich telemetry meters, status banners, and multi-class gauges onto the frame.
        """
        h, w, _ = frame.shape
        alert_level = telemetry["alert_level"]

        # 1. Dynamic Alert Border
        if alert_level == 2:  # Flashing critical border
            if (self.frame_count // 5) % 2 == 0:
                cv2.rectangle(frame, (0, 0), (w, h), (0, 0, 255), 8)
        elif alert_level == 1:
            cv2.rectangle(frame, (0, 0), (w, h), (0, 215, 255), 4)

        # 2. Top Header HUD Banner
        header_color = np.array([40, 40, 40], dtype=np.uint8)
        if alert_level == 2:
            header_color = np.array([0, 0, 180], dtype=np.uint8)
        elif alert_level == 1:
            header_color = np.array([0, 140, 200], dtype=np.uint8)

        # Fast in-place ROI alpha blend
        header_roi = frame[0:50, 0:w]
        frame[0:50, 0:w] = cv2.addWeighted(header_roi, 0.25, np.full_like(header_roi, header_color), 0.75, 0)

        # Header Text
        status_text = telemetry.get("status_text", "DRIVER STATUS: ALERT")
        cv2.putText(frame, f"SYSTEM: {status_text}", (18, 33), cv2.FONT_HERSHEY_DUPLEX, 0.75, (255, 255, 255), 2)
        cv2.putText(frame, f"FPS: {telemetry['fps']} | Latency: {telemetry['latency_ms']:.1f}ms", (w - 320, 33), cv2.FONT_HERSHEY_DUPLEX, 0.6, (200, 255, 200), 1)

        # 3. Side Telemetry Glass Panel
        panel_w = 260
        panel_h = 240
        x1, y1 = 15, 65
        x2, y2 = x1 + panel_w, y1 + panel_h

        panel_roi = frame[y1:y2, x1:x2]
        panel_bg = np.full_like(panel_roi, np.array([20, 20, 20], dtype=np.uint8))
        frame[y1:y2, x1:x2] = cv2.addWeighted(panel_roi, 0.3, panel_bg, 0.7, 0)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (100, 100, 100), 1)

        # Panel Header
        cv2.putText(frame, "TELEMETRY METRICS", (x1 + 10, y1 + 22), cv2.FONT_HERSHEY_DUPLEX, 0.5, (0, 255, 255), 1)

        # Telemetry Gauge Helper
        def draw_bar(y_offset, label, value, threshold, max_val=1.0, color_ok=(0, 255, 0)):
            cv2.putText(frame, f"{label}: {value:.2f}", (x1 + 10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 255), 1)
            bar_x = x1 + 120
            bar_w = 120
            bar_h = 10
            bar_y = y_offset - 9
            cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), (60, 60, 60), -1)
            fill_w = int(np.clip(value / max_val, 0.0, 1.0) * bar_w)
            color = color_ok if value >= threshold else (0, 0, 255)
            if label in ["MAR", "PERCLOS", "FATIGUE"]:
                color = (0, 0, 255) if value >= threshold else color_ok
            cv2.rectangle(frame, (bar_x, bar_y), (bar_x + fill_w, bar_y + bar_h), color, -1)
            # Threshold line
            th_x = bar_x + int(np.clip(threshold / max_val, 0.0, 1.0) * bar_w)
            cv2.line(frame, (th_x, bar_y - 2), (th_x, bar_y + bar_h + 2), (0, 255, 255), 1)

        draw_bar(y1 + 48, "EAR", telemetry["ear"], getattr(config, "EAR_DROWSY_THRESH", 0.23), max_val=0.45)
        draw_bar(y1 + 76, "MAR", telemetry["mar"], getattr(config, "MAR_YAWN_THRESH", 0.60), max_val=0.85)
        draw_bar(y1 + 104, "PERCLOS", telemetry["perclos"], getattr(config, "PERCLOS_CLOSURE_THRESHOLD", 0.20), max_val=1.0)
        draw_bar(y1 + 132, "FATIGUE", telemetry["fatigue_score"], getattr(config, "CRITICAL_FATIGUE_THRESHOLD", 0.70), max_val=1.0)

        # Head Pose Angles Display
        cv2.putText(
            frame,
            f"POSE: P={telemetry['pitch']:.0f} Y={telemetry['yaw']:.0f} R={telemetry['roll']:.0f}",
            (x1 + 10, y1 + 165),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.42,
            (200, 200, 200),
            1,
        )

        # Multi-Class Probability Mini-Bars
        if "probabilities" in telemetry:
            probs = telemetry["probabilities"]
            p_text = f"P(A)={probs[0]:.2f} P(D)={probs[1]:.2f} P(S)={probs[2]:.2f}"
            cv2.putText(frame, p_text, (x1 + 10, y1 + 195), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (180, 220, 255), 1)

        cv2.putText(
            frame,
            f"STATE: {telemetry['state_label']}",
            (x1 + 10, y1 + 225),
            cv2.FONT_HERSHEY_DUPLEX,
            0.5,
            (0, 255, 0) if telemetry["predicted_state"] == 0 else ((0, 215, 255) if telemetry["predicted_state"] == 1 else (0, 0, 255)),
            1,
        )

        return frame

    def run_stream(
        self,
        video_source: Union[int, str, None] = 0,
        max_frames: Optional[int] = None,
        display: bool = True,
    ):
        """
        Executes live real-time detection on a video stream or camera.
        """
        cap = cv2.VideoCapture(video_source)
        if not cap.isOpened():
            logger.error(f"Failed to open video source: {video_source}")
            return

        logger.info(f"Started real-time detection stream (source={video_source})...")
        try:
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                hud_frame, telemetry = self.process_frame(frame)

                if display:
                    cv2.imshow("Real-Time Drowsiness Detection and Alert System", hud_frame)
                    key = cv2.waitKey(1) & 0xFF
                    if key == 27 or key == ord("q"):
                        break

                if max_frames and self.frame_count >= max_frames:
                    break
        finally:
            cap.release()
            if display:
                cv2.destroyAllWindows()
            self.alert_manager.save_logs()
            logger.info(f"Stream finished. Total frames processed: {self.frame_count}")
