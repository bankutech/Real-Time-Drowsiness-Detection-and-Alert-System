
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

    def __init__(
        self,
        primary_model_type: str = "ensemble",
        enable_audio: bool = getattr(config, "ENABLE_AUDIO_ALERT", True),
        output_dir: Optional[Path] = None,
    ):
        self.primary_model_type = primary_model_type.lower()
        self.output_dir = output_dir or getattr(config, "OUTPUTS_DIR", Path("outputs"))
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.feature_extractor = FeatureExtractor(fps=getattr(config, "DEFAULT_FPS", 30.0))

        self.scaler = load_model("scaler.joblib")
        try:
            self.linear_reg = load_model("fatigue_regressor.joblib")
        except Exception:
            self.linear_reg = load_model("linear_regression.joblib")
        self.bayes_model = load_model("bayesian_logistic.joblib")
        self.rf_model = load_model("random_forest.joblib")
        self.ensemble_model = load_model("ensemble_stacking.joblib")

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

        try:
            self.hmm = DrowsinessHMM.load()
            logger.info("Loaded trained HMM for temporal filtering.")
        except Exception as e:
            logger.warning(f"Could not load HMM: {e}. Running without HMM temporal filter.")
            self.hmm = None
        self.prev_hmm_belief: Optional[np.ndarray] = None

        self.alert_manager = AlertManager(enable_audio=enable_audio, log_dir=self.output_dir)

        self.frame_count = 0
        self.fps = 0.0
        self._prev_time = time.perf_counter()

        self.is_calibrated = False
        self.calibrating = False
        self.calibration_frames_needed = 45
        self.calibration_buffer_ear: List[float] = []
        self.calibration_buffer_mar: List[float] = []
        self.baseline_ear = 0.32
        self.baseline_mar = 0.25
        self.ear_thresh_active = getattr(config, "EAR_DROWSY_THRESH", 0.23)
        self.mar_thresh_active = getattr(config, "MAR_YAWN_THRESH", 0.60)

        self.speech_counter = 0

        self.start_calibration()

    def start_calibration(self, duration_frames: int = 45):
        self.calibrating = True
        self.calibration_frames_needed = duration_frames
        self.calibration_buffer_ear = []
        self.calibration_buffer_mar = []
        logger.info("Driver baseline calibration initiated...")

    def process_frame(self, frame: np.ndarray, render_hud: bool = True) -> Tuple[np.ndarray, Dict[str, Any]]:
        t_start = time.perf_counter()
        self.frame_count += 1

        curr_time = time.perf_counter()
        dt = curr_time - self._prev_time
        self._prev_time = curr_time
        if dt > 0:
            self.fps = 0.9 * self.fps + 0.1 * (1.0 / dt) if self.fps > 0 else 1.0 / dt

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

        if self.calibrating and face_detected and ear > 0.15:
            self.calibration_buffer_ear.append(ear)
            self.calibration_buffer_mar.append(mar)
            if len(self.calibration_buffer_ear) >= self.calibration_frames_needed:
                self.baseline_ear = float(np.median(self.calibration_buffer_ear))
                self.baseline_mar = float(np.median(self.calibration_buffer_mar))
                self.ear_thresh_active = max(0.18, min(0.30, 0.76 * self.baseline_ear))
                self.mar_thresh_active = max(0.48, min(0.72, 2.0 * self.baseline_mar))
                self.is_calibrated = True
                self.calibrating = False
                logger.info(
                    f"Calibration Complete: Baseline EAR={self.baseline_ear:.3f}, "
                    f"Threshold EAR={self.ear_thresh_active:.3f}, Threshold MAR={self.mar_thresh_active:.3f}"
                )

        is_speaking = False
        if mar > (self.mar_thresh_active * 0.85) and ear > (self.ear_thresh_active * 1.05) and perclos < 0.12:
            is_speaking = True
            self.speech_counter += 1
        else:
            self.speech_counter = max(0, self.speech_counter - 1)

        eyewear_detected = meta.get("eyewear_detected", False)
        eyewear_label = meta.get("eyewear_label", "None")
        head_pose_dir = meta.get("head_pose_direction", "Facing Ahead (Attentive)")

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

        feature_vector = np.array([[feats.get(col, 0.0) for col in config.FEATURE_COLUMNS]], dtype=np.float64)

        feature_vector_scaled = self.scaler.transform(feature_vector)

        fatigue_pred = float(np.clip(self.linear_reg.predict(feature_vector_scaled)[0], 0.0, 100.0) / 100.0)
        
        if is_speaking:
            fatigue_pred = min(0.35, fatigue_pred)

        telemetry["fatigue_score"] = fatigue_pred

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

        if is_speaking and pred_state > 0 and perclos < 0.10:
            pred_state = 0
            probas = [0.85, 0.15, 0.0, 0.0]

        if self.hmm is not None:
            prob_arr = np.array(probas, dtype=np.float64)
            if len(prob_arr) == self.hmm.n_states:
                hmm_state, self.prev_hmm_belief = self.hmm.online_filter_step(
                    prob_arr, prev_forward_state=self.prev_hmm_belief
                )
                pred_state = hmm_state

        prob_arr_safe = np.array(probas, dtype=np.float64) + 1e-12
        entropy_val = float(-np.sum(prob_arr_safe * np.log(prob_arr_safe)))

        telemetry["predicted_state"] = pred_state
        telemetry["state_label"] = config.CLASS_LABELS[min(pred_state, len(config.CLASS_LABELS) - 1)]
        telemetry["probabilities"] = [round(float(p), 4) for p in probas]
        telemetry["uncertainty_entropy"] = round(entropy_val, 3)
        telemetry["hmm_belief"] = [round(float(b), 4) for b in (self.prev_hmm_belief if self.prev_hmm_belief is not None else probas)]

        alert_event = self.alert_manager.update(
            predicted_state=pred_state,
            fatigue_score=fatigue_pred,
            ear=ear,
            mar=mar,
            perclos=perclos,
            head_pose=(pitch, yaw, roll),
            frame_idx=self.frame_count,
        )

        status_text = alert_event["status_text"]
        alert_level = alert_event["alert_level"]
        if self.calibrating:
            status_text = f"CALIBRATING BASELINE ({telemetry['calibration_progress']}%)"
        elif is_speaking and alert_level == 0:
            status_text = "DRIVER SPEAKING (NORMAL)"

        telemetry["alert_level"] = alert_level
        telemetry["status_text"] = status_text

        latency_ms = (time.perf_counter() - t_start) * 1000.0
        telemetry["latency_ms"] = latency_ms

        if render_hud:
            annotated_frame = self._render_hud(annotated_frame, telemetry)

        return annotated_frame, telemetry

    def _render_hud(self, frame: np.ndarray, telemetry: Dict[str, Any]) -> np.ndarray:
        h, w, _ = frame.shape
        alert_level = telemetry.get("alert_level", 0)

        if alert_level == 2:
            if (self.frame_count // 4) % 2 == 0:
                cv2.rectangle(frame, (0, 0), (w, h), (0, 0, 240), 6)
        elif alert_level == 1:
            cv2.rectangle(frame, (0, 0), (w, h), (0, 180, 255), 3)

        header_h = 38
        top_roi = frame[0:header_h, 0:w]
        bg_color = (25, 25, 25)
        if alert_level == 2:
            bg_color = (0, 0, 150)
        elif alert_level == 1:
            bg_color = (0, 110, 180)

        overlay_roi = top_roi.copy()
        cv2.rectangle(overlay_roi, (0, 0), (w, header_h), bg_color, -1)
        cv2.addWeighted(overlay_roi, 0.70, top_roi, 0.30, 0, top_roi)

        status_color = (0, 255, 120) if alert_level == 0 else ((0, 215, 255) if alert_level == 1 else (50, 50, 255))
        cv2.circle(frame, (18, 19), 6, status_color, -1, lineType=cv2.LINE_AA)
        
        status_text = telemetry.get("status_text", "DRIVER STATUS: ALERT")
        cv2.putText(frame, status_text, (32, 25), cv2.FONT_HERSHEY_DUPLEX, 0.55, (255, 255, 255), 1, lineType=cv2.LINE_AA)

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
