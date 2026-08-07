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

        # 3. Load Hardware-Accelerated ONNX INT8 Engines for Sub-millisecond Execution
        self.onnx_ensemble = None
        self.onnx_rf = None
        self.onnx_bayes = None
        try:
            from src.onnx_exporter import ONNXInferenceEngine
            onnx_dir = config.MODELS_DIR / "onnx"
            p_ens = onnx_dir / "ensemble_stacking_int8.onnx"
            if not p_ens.exists():
                p_ens = onnx_dir / "ensemble_stacking.onnx"
            if p_ens.exists():
                self.onnx_ensemble = ONNXInferenceEngine(p_ens)

            p_rf = onnx_dir / "random_forest_int8.onnx"
            if not p_rf.exists():
                p_rf = onnx_dir / "random_forest.onnx"
            if p_rf.exists():
                self.onnx_rf = ONNXInferenceEngine(p_rf)

            p_bayes = onnx_dir / "bayesian_logistic_int8.onnx"
            if not p_bayes.exists():
                p_bayes = onnx_dir / "bayesian_logistic.onnx"
            if p_bayes.exists():
                self.onnx_bayes = ONNXInferenceEngine(p_bayes)
            logger.info("ONNX hardware acceleration engines initialized successfully.")
        except Exception as e:
            logger.warning(f"Could not initialize ONNX engines ({e}). Using Scikit-Learn fallback.")

        # 4. Load HMM Engine
        try:
            self.hmm = DrowsinessHMM.load()
            logger.info("Loaded trained HMM for temporal filtering.")
        except Exception as e:
            logger.warning(f"Could not load HMM: {e}. Running without HMM temporal filter.")
            self.hmm = None
        self.prev_hmm_belief: Optional[np.ndarray] = None

        # 5. Initialize Alert Manager
        self.alert_manager = AlertManager(enable_audio=enable_audio, log_dir=self.output_dir)

        # State and Performance Tracking
        self.frame_count = 0
        self.fps = 0.0
        self._prev_time = time.perf_counter()

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

    def process_frame(self, frame: np.ndarray, render_hud: bool = True) -> Tuple[np.ndarray, Dict[str, Any]]:
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

        # 1. Feature Extraction (Unit 1 with CLAHE and 13 Biometrics)
        feats, annotated_frame, meta = self.feature_extractor.process_frame(
            frame, current_time=curr_time, draw_overlays=render_hud, apply_clahe=True
        )

        face_detected = meta.get("landmarks_count", 0) >= 250
        ear = float(feats.get("ear", 0.0))
        mar = float(feats.get("mar", 0.0))
        perclos = float(feats.get("perclos", 0.0))
        pitch = float(feats.get("head_pitch", 0.0))
        yaw = float(feats.get("head_yaw", 0.0))
        roll = float(feats.get("head_roll", 0.0))
        ear_vel = float(feats.get("ear_velocity", 0.0))
        ear_acc = float(feats.get("ear_acceleration", 0.0))
        clahe_applied = meta.get("clahe_applied", False)
        luminance = meta.get("luminance", 100.0)

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

        # Eyewear / Glasses Analysis:
        eyewear_detected = meta.get("eyewear_detected", False)
        eyewear_label = meta.get("eyewear_label", "None")
        head_pose_dir = meta.get("head_pose_direction", "Facing Ahead (Attentive)")

        # Telemetry structure
        telemetry: Dict[str, Any] = {
            "frame_idx": self.frame_count,
            "fps": round(self.fps, 1),
            "face_detected": face_detected,
            "ear": ear,
            "mar": mar,
            "perclos": perclos,
            "ear_velocity": ear_vel,
            "ear_acceleration": ear_acc,
            "pitch": pitch,
            "yaw": yaw,
            "roll": roll,
            "head_pose_direction": head_pose_dir,
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
            "low_light": clahe_applied or (luminance < config.LOW_LIGHT_LUMINANCE_THRESHOLD),
            "clahe_applied": clahe_applied,
            "luminance": luminance,
            "lighting_quality": "Enhanced (CLAHE Active)" if clahe_applied else ("Optimal" if luminance >= 85 else "Moderate"),
            "eyewear_detected": eyewear_detected,
            "eyewear_label": eyewear_label,
            "landmarks": meta.get("landmarks_summary", {}),
        }

        # 2. Build 13-Dimensional Feature Vector matching config.FEATURE_COLUMNS
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

        # Primary Multi-Class Classification (Unit 5) with ONNX Hardware Acceleration
        if self.primary_model_type == "ensemble":
            if self.onnx_ensemble:
                pred_state = int(self.onnx_ensemble.predict(feature_vector_scaled)[0])
                probas = self.onnx_ensemble.predict_proba(feature_vector_scaled)[0]
            elif self.ensemble_model:
                pred_state = int(self.ensemble_model.predict(feature_vector_scaled)[0])
                probas = self.ensemble_model.predict_proba(feature_vector_scaled)[0]
            else:
                pred_state, probas = 0, [1.0, 0.0, 0.0, 0.0]
        elif self.primary_model_type == "rf":
            if self.onnx_rf:
                pred_state = int(self.onnx_rf.predict(feature_vector_scaled)[0])
                probas = self.onnx_rf.predict_proba(feature_vector_scaled)[0]
            elif self.rf_model:
                pred_state = int(self.rf_model.predict(feature_vector_scaled)[0])
                probas = self.rf_model.predict_proba(feature_vector_scaled)[0]
            else:
                pred_state, probas = 0, [1.0, 0.0, 0.0, 0.0]
        elif self.primary_model_type == "bayes":
            if self.onnx_bayes:
                pred_state = int(self.onnx_bayes.predict(feature_vector_scaled)[0])
                probas = self.onnx_bayes.predict_proba(feature_vector_scaled)[0]
            elif self.bayes_model:
                pred_state = int(self.bayes_model.predict(feature_vector_scaled)[0])
                probas = self.bayes_model.predict_proba(feature_vector_scaled)[0]
            else:
                pred_state, probas = 0, [1.0, 0.0, 0.0, 0.0]
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

        # 6. Render Minimalist Cockpit HUD Overlay (Clean & Unobtrusive)
        if render_hud:
            annotated_frame = self._render_hud(annotated_frame, telemetry)

        return annotated_frame, telemetry

    def _render_hud(self, frame: np.ndarray, telemetry: Dict[str, Any]) -> np.ndarray:
        """
        Renders a sleek, uncluttered cockpit HUD overlay that keeps the driver's face unobstructed.
        """
        h, w, _ = frame.shape
        alert_level = telemetry.get("alert_level", 0)

        # 1. Tiered Alert Perimeter Glow (Non-obtrusive)
        if alert_level == 2:  # Confirmed Sustained Critical Emergency
            if (self.frame_count // 4) % 2 == 0:
                cv2.rectangle(frame, (0, 0), (w, h), (0, 0, 240), 6)
        elif alert_level == 1:  # Soft Amber Warning
            cv2.rectangle(frame, (0, 0), (w, h), (0, 180, 255), 3)

        # 2. Sleek Top Status Pill Banner (Semi-transparent)
        header_h = 38
        overlay = frame.copy()
        bg_color = (25, 25, 25)
        if alert_level == 2:
            bg_color = (0, 0, 150)
        elif alert_level == 1:
            bg_color = (0, 110, 180)

        cv2.rectangle(overlay, (0, 0), (w, header_h), bg_color, -1)
        cv2.addWeighted(overlay, 0.70, frame, 0.30, 0, frame)

        # Status Dot & Text
        status_color = (0, 255, 120) if alert_level == 0 else ((0, 215, 255) if alert_level == 1 else (50, 50, 255))
        cv2.circle(frame, (18, 19), 6, status_color, -1, lineType=cv2.LINE_AA)
        
        status_text = telemetry.get("status_text", "DRIVER STATUS: ALERT")
        cv2.putText(frame, status_text, (32, 25), cv2.FONT_HERSHEY_DUPLEX, 0.55, (255, 255, 255), 1, lineType=cv2.LINE_AA)

        # Right-side Context Badges (Orientation & Eyewear)
        pose_dir = telemetry.get("head_pose_direction", "Facing Ahead")
        eyewear_lbl = " [Glasses]" if telemetry.get("eyewear_detected", False) else ""
        badge_text = f"{pose_dir}{eyewear_lbl}"
        
        (tw, _), _ = cv2.getTextSize(badge_text, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
        cv2.putText(frame, badge_text, (w - tw - 15, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 240, 255), 1, lineType=cv2.LINE_AA)

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
