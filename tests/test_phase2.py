"""
Unit test and verification script for Phase 2 (Feature Extraction Pipeline).
"""

import sys
from pathlib import Path
import numpy as np
import cv2

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import src.config as config
from src.feature_extraction import (
    calculate_ear,
    calculate_mar,
    BlinkAndYawnTracker,
    estimate_head_pose,
    FeatureExtractor,
)


def test_phase2():
    print("=" * 60)
    print("STARTING PHASE 2 VERIFICATION TEST")
    print("=" * 60)

    # 1. Test EAR calculation with open vs closed eye coordinates
    # Open eye landmarks: p1=(0,0), p2=(2, 2.0), p3=(4, 2.0), p4=(6, 0), p5=(4, -2.0), p6=(2, -2.0)
    open_eye_pts = [(0.0, 0.0), (2.0, 2.0), (4.0, 2.0), (6.0, 0.0), (4.0, -2.0), (2.0, -2.0)]
    ear_open = calculate_ear(open_eye_pts)
    print(f"[OK] EAR Open Eye: {ear_open:.4f}")
    assert ear_open > 0.30, "Open eye EAR should be high"

    # Closed eye landmarks: vertical distances near zero (y=0.2 and -0.2)
    closed_eye_pts = [(0.0, 0.0), (2.0, 0.2), (4.0, 0.2), (6.0, 0.0), (4.0, -0.2), (2.0, -0.2)]
    ear_closed = calculate_ear(closed_eye_pts)
    print(f"[OK] EAR Closed Eye: {ear_closed:.4f}")
    assert ear_closed < 0.15, "Closed eye EAR should be low"

    # 2. Test MAR calculation (yawning vs normal)
    normal_mouth = {
        "left": (0.0, 0.0), "right": (10.0, 0.0),
        "top_lip": (5.0, 1.0), "bot_lip": (5.0, -1.0),
        "top_outer": (5.0, 1.5), "bot_outer": (5.0, -1.5),
    }
    mar_normal = calculate_mar(normal_mouth)
    print(f"[OK] MAR Normal Mouth: {mar_normal:.4f}")
    assert mar_normal < 0.40, "Normal mouth MAR should be below threshold"

    yawn_mouth = {
        "left": (0.0, 0.0), "right": (10.0, 0.0),
        "top_lip": (5.0, 7.0), "bot_lip": (5.0, -7.0),
        "top_outer": (5.0, 8.0), "bot_outer": (5.0, -8.0),
    }
    mar_yawn = calculate_mar(yawn_mouth)
    print(f"[OK] MAR Yawn Mouth: {mar_yawn:.4f}")
    assert mar_yawn > 0.60, "Yawn mouth MAR should be above threshold"

    # 3. Test Blink & Yawn Tracker & PERCLOS
    tracker = BlinkAndYawnTracker(fps=30.0, perclos_window_size=30)
    for f in range(30):
        # First 10 frames open, next 20 frames closed
        ear_val = 0.32 if f < 10 else 0.12
        res = tracker.update(ear=ear_val, mar=0.25, current_time=f / 30.0)

    print(f"[OK] Tracker PERCLOS: {res['perclos']:.2f}")
    assert res["perclos"] > 0.50, "PERCLOS should reflect closed eyes"

    # 4. Test solvePnP Head Pose Estimation
    # Simulate face centered in 640x480 frame
    mock_lms = {
        1: (320.0, 240.0),    # Nose tip
        199: (320.0, 350.0),  # Chin
        33: (240.0, 190.0),   # Left eye corner
        263: (400.0, 190.0),  # Right eye corner
        61: (270.0, 290.0),   # Left mouth
        291: (370.0, 290.0),  # Right mouth
    }
    pitch, yaw, roll, face_angle, rvec, tvec, cam_mat, dist = estimate_head_pose(mock_lms, (480, 640, 3))
    print(f"[OK] Head Pose computed: Pitch={pitch:.1f}deg, Yaw={yaw:.1f}deg, Roll={roll:.1f}deg, Angle={face_angle:.1f}deg")
    assert isinstance(face_angle, float) and not np.isnan(face_angle)

    # 5. Test FeatureExtractor end-to-end on synthetic image frame
    extractor = FeatureExtractor(fps=30.0)
    test_frame = np.full((480, 640, 3), 40, dtype=np.uint8)
    # Draw simple facial feature markers for contrast
    cv2.circle(test_frame, (320, 240), 100, (180, 180, 180), -1)

    features, annotated_frame, telemetry = extractor.process_frame(test_frame, current_time=0.0)
    print(f"[OK] FeatureExtractor processed frame. Feature count = {len(features)}")
    print(f"     Extracted keys: {list(features.keys())}")
    assert set(features.keys()) == set(config.FEATURE_COLUMNS), "Feature keys must match config.FEATURE_COLUMNS"
    assert annotated_frame.shape == test_frame.shape, "Annotated frame shape must match input"

    print("=" * 60)
    print("PHASE 2 VERIFICATION PASSED SUCCESSFULLY!")
    print("=" * 60)


if __name__ == "__main__":
    test_phase2()
