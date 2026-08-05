"""
Phase 8 Verification Suite: Alert System & Real-Time Detection Pipeline.
Validates multi-tier audio tones, alert escalation thresholds, event logging,
and end-to-end CV processing with cockpit HUD overlay.
"""

import sys
from pathlib import Path
import numpy as np
import cv2

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src import config
from src.alert_system import AlertManager, SoundGenerator
from src.realtime_detection import DrowsinessDetectorPipeline


def generate_synthetic_driver_frame(state_type: str = "alert") -> np.ndarray:
    """
    Generates a 640x480 synthetic video frame containing a simulated driver face.
    """
    frame = np.ones((480, 640, 3), dtype=np.uint8) * 45  # Dark interior cabin

    # Draw face oval
    center = (320, 240)
    cv2.ellipse(frame, center, (110, 150), 0, 0, 360, (180, 195, 215), -1)
    cv2.ellipse(frame, center, (110, 150), 0, 0, 360, (100, 120, 140), 2)

    # Eyes & Mouth based on state
    if state_type == "alert":
        # Open eyes
        cv2.circle(frame, (275, 210), 14, (255, 255, 255), -1)
        cv2.circle(frame, (275, 210), 6, (50, 40, 30), -1)
        cv2.circle(frame, (365, 210), 14, (255, 255, 255), -1)
        cv2.circle(frame, (365, 210), 6, (50, 40, 30), -1)
        # Closed/neutral mouth
        cv2.ellipse(frame, (320, 310), (25, 6), 0, 0, 360, (70, 70, 150), -1)

    elif state_type == "drowsy":
        # Narrow / half-closed eyes
        cv2.ellipse(frame, (275, 210), (14, 5), 0, 0, 360, (255, 255, 255), -1)
        cv2.circle(frame, (275, 210), 4, (50, 40, 30), -1)
        cv2.ellipse(frame, (365, 210), (14, 5), 0, 0, 360, (255, 255, 255), -1)
        cv2.circle(frame, (365, 210), 4, (50, 40, 30), -1)
        # Yawning / wide open mouth
        cv2.ellipse(frame, (320, 315), (28, 22), 0, 0, 360, (30, 30, 80), -1)

    else:  # sleeping
        # Fully closed slit eyes
        cv2.line(frame, (260, 212), (290, 212), (30, 30, 30), 3)
        cv2.line(frame, (350, 212), (380, 212), (30, 30, 30), 3)
        # Slack mouth
        cv2.ellipse(frame, (320, 320), (20, 10), 0, 0, 360, (50, 50, 100), -1)

    return frame


def test_phase8():
    print("=" * 60)
    print("STARTING PHASE 8 VERIFICATION TEST (ALERTING & REAL-TIME CV)")
    print("=" * 60)

    # 1. Test Sound Generator
    print("\n[1] Testing Sound Generator...")
    sound_gen = SoundGenerator()
    sound_gen.play_tone(1)
    sound_gen.play_tone(2)
    print("Sound generator initialized and tone dispatch tested.")

    # 2. Test Alert Manager Escalation & Debouncing
    print("\n[2] Testing Alert Manager Escalation & Debouncing...")
    alert_mgr = AlertManager(
        warning_consecutive_frames=5,
        critical_consecutive_frames=10,
        enable_audio=False,
    )

    # Normal alert frames
    for i in range(10):
        evt = alert_mgr.update(
            predicted_state=0,
            fatigue_score=0.1,
            ear=0.35,
            mar=0.15,
            perclos=0.05,
            frame_idx=i,
        )
    assert evt["alert_level"] == 0, f"Expected alert level 0, got {evt['alert_level']}"
    print("Normal state: Alert Level 0 (Safe)")

    # Drowsy frames -> escalate to Level 1
    for i in range(10, 18):
        evt = alert_mgr.update(
            predicted_state=1,
            fatigue_score=0.55,
            ear=0.22,
            mar=0.55,
            perclos=0.35,
            frame_idx=i,
        )
    assert evt["alert_level"] == 1, f"Expected alert level 1, got {evt['alert_level']}"
    print("Drowsy state: Alert Level 1 (Warning Triggered)")

    # Sleeping frames -> escalate to Level 2
    for i in range(18, 32):
        evt = alert_mgr.update(
            predicted_state=2,
            fatigue_score=0.85,
            ear=0.12,
            mar=0.20,
            perclos=0.75,
            frame_idx=i,
        )
    assert evt["alert_level"] == 2, f"Expected alert level 2, got {evt['alert_level']}"
    print("Sleeping state: Alert Level 2 (Critical Alarm Triggered)")

    # Save and verify alert logs
    alert_mgr.save_logs()
    assert (config.OUTPUTS_DIR / "alert_log.csv").exists(), "alert_log.csv must exist"
    assert (config.OUTPUTS_DIR / "alert_events.json").exists(), "alert_events.json must exist"
    print("Alert logs successfully written to disk.")

    # 3. Test End-to-End Real-Time Detection Pipeline
    print("\n[3] Testing End-to-End Real-Time Detection Pipeline...")
    pipeline = DrowsinessDetectorPipeline(primary_model_type="ensemble", enable_audio=False)

    # Process test stream of synthetic frames
    states = ["alert"] * 10 + ["drowsy"] * 10 + ["sleeping"] * 15
    for idx, state_name in enumerate(states):
        frame = generate_synthetic_driver_frame(state_name)
        hud_frame, telemetry = pipeline.process_frame(frame)

        assert hud_frame is not None, "HUD frame must not be None"
        assert hud_frame.shape == (480, 640, 3), f"Invalid HUD frame shape: {hud_frame.shape}"
        assert "latency_ms" in telemetry, "Telemetry must include latency_ms"
        assert "predicted_state" in telemetry, "Telemetry must include predicted_state"

    print(f"Successfully processed {len(states)} real-time video frames.")
    print(f"Sample Telemetry: FPS={telemetry['fps']} | Latency={telemetry['latency_ms']:.2f}ms | Fatigue={telemetry['fatigue_score']:.2f} | Alert Level={telemetry['alert_level']}")

    # Save annotated HUD preview artifact
    preview_path = config.OUTPUTS_DIR / "realtime_hud_preview.png"
    cv2.imwrite(str(preview_path), hud_frame)
    print(f"Saved real-time HUD preview to {preview_path}")

    print("\n" + "=" * 60)
    print("PHASE 8 VERIFICATION PASSED SUCCESSFULLY!")
    print("=" * 60)


if __name__ == "__main__":
    test_phase8()
