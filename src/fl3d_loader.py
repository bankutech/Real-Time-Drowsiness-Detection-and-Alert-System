
import re
import sys
import logging
import math
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import cv2

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src import config
from src.utils import load_model

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("FL3DLoader")

FL3D_STATE_MAP = {
    "alert": "Alert",
    "microsleep": "Sleep",
    "yawning": "Drowsy",
}

REQUIRED_FL3D_COLUMNS = [
    "file", "state",
    "left_eye_x", "left_eye_y", "right_eye_x", "right_eye_y",
    "nose_x", "nose_y",
    "left_mouth_x", "left_mouth_y", "right_mouth_x", "right_mouth_y",
    "landmark_roll",
]

UNRECOVERABLE_FEATURE_COLUMNS = [
    "ear", "mar", "blink_duration", "blink_rate", "yawn_freq",
    "eye_closure_dur", "perclos", "ear_velocity", "ear_acceleration",
]

_ASSUMED_CROP_W = 240.0
_ASSUMED_CROP_H = 320.0

_MODEL_POINTS_5PT_3D = np.array([
    [0.0, 0.0, 0.0],
    [-225.0, 170.0, -135.0],
    [225.0, 170.0, -135.0],
    [-150.0, -150.0, -125.0],
    [150.0, -150.0, -125.0],
], dtype=np.float64)

_DIST_COEFFS = np.zeros((4, 1), dtype=np.float64)
_CAMERA_MATRIX = np.array([
    [_ASSUMED_CROP_W, 0.0, _ASSUMED_CROP_W / 2.0],
    [0.0, _ASSUMED_CROP_W, _ASSUMED_CROP_H / 2.0],
    [0.0, 0.0, 1.0],
], dtype=np.float64)


def _normalize_deg(angle: float) -> float:
    a = (angle + 180.0) % 360.0 - 180.0
    if a > 90.0:
        a -= 180.0
    elif a < -90.0:
        a += 180.0
    return float(a)


def _estimate_pitch_yaw(row: pd.Series) -> tuple:
    image_points = np.array([
        [row.nose_x, row.nose_y],
        [row.left_eye_x, row.left_eye_y],
        [row.right_eye_x, row.right_eye_y],
        [row.left_mouth_x, row.left_mouth_y],
        [row.right_mouth_x, row.right_mouth_y],
    ], dtype=np.float64)

    success, rotation_vec, _ = cv2.solvePnP(
        _MODEL_POINTS_5PT_3D, image_points, _CAMERA_MATRIX, _DIST_COEFFS,
        flags=cv2.SOLVEPNP_EPNP,
    )
    if not success:
        return 0.0, 0.0

    rot_mat, _ = cv2.Rodrigues(rotation_vec)
    euler_angles, _, _, _, _, _ = cv2.RQDecomp3x3(rot_mat)
    pitch = _normalize_deg(euler_angles[0])
    yaw = _normalize_deg(euler_angles[1])
    return pitch, yaw


def load_fl3d_csv(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    missing = [c for c in REQUIRED_FL3D_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(
            f"'{csv_path}' does not look like the FL3D statistical landmark "
            f"export -- missing columns: {missing}"
        )
    return df


def build_real_dataset_from_fl3d(
    csv_path: Path,
    max_samples_per_class: Optional[int] = 1500,
    random_state: int = config.RANDOM_STATE,
) -> pd.DataFrame:
    logger.info(f"Loading FL3D landmark table from {csv_path} ...")
    df = load_fl3d_csv(csv_path)
    logger.info(f"Loaded {len(df)} raw FL3D rows. Class counts:\n{df['state'].value_counts()}")

    df["state_lower"] = df["state"].astype(str).str.strip().str.lower()
    unknown = sorted(set(df["state_lower"]) - set(FL3D_STATE_MAP))
    if unknown:
        logger.warning(f"Dropping {len(df[df['state_lower'].isin(unknown)])} rows with unmapped state labels: {unknown}")
    df = df[df["state_lower"].isin(FL3D_STATE_MAP)].copy()
    df["mapped_state"] = df["state_lower"].map(FL3D_STATE_MAP)

    if max_samples_per_class is not None:
        sampled_parts = []
        for _, group in df.groupby("mapped_state"):
            n = min(len(group), max_samples_per_class)
            sampled_parts.append(group.sample(n=n, random_state=random_state))
        df = pd.concat(sampled_parts, ignore_index=True)
        logger.info(f"Stratified sample -> {len(df)} rows (cap={max_samples_per_class}/class). New counts:\n{df['mapped_state'].value_counts()}")

    df["video_id"] = df["file"].apply(lambda p: Path(str(p)).parent.name)
    frame_idx = df["file"].astype(str).str.extract(r"frame(\d+)", expand=False)
    df["frame_idx_in_video"] = pd.to_numeric(frame_idx, errors="coerce").fillna(0).astype(int)
    df = df.sort_values(["video_id", "frame_idx_in_video"]).reset_index(drop=True)
    df["frame_number"] = np.arange(1, len(df) + 1)
    df["timestamp"] = (df["frame_idx_in_video"] / config.DEFAULT_FPS).round(3)

    logger.info("Estimating head pose (pitch/yaw via solvePnP, roll = FL3D landmark_roll) ...")
    pitches, yaws = [], []
    for _, row in df.iterrows():
        p, y = _estimate_pitch_yaw(row)
        pitches.append(p)
        yaws.append(y)
    df["head_pitch"] = pitches
    df["head_yaw"] = yaws
    df["head_roll"] = df["landmark_roll"].astype(float)
    df["face_angle"] = np.sqrt(df["head_pitch"] ** 2 + df["head_yaw"] ** 2 + df["head_roll"] ** 2)

    for col in UNRECOVERABLE_FEATURE_COLUMNS:
        df[col] = np.nan
    df["fatigue_score"] = np.nan

    try:
        imputation_values = load_model("imputer.joblib")
        for col in UNRECOVERABLE_FEATURE_COLUMNS:
            if col in imputation_values:
                df[col] = df[col].fillna(imputation_values[col])
        logger.info("Filled unrecoverable feature columns using models/imputer.joblib training-set medians.")
    except FileNotFoundError:
        logger.warning(
            "models/imputer.joblib not found -- unrecoverable feature columns will stay NaN. "
            "Run 'python main.py --mode train' first, then re-run this loader."
        )

    df["state"] = df["mapped_state"]

    out_cols = ["frame_number", "timestamp"] + config.FEATURE_COLUMNS + ["fatigue_score", "state"]
    result = df[out_cols].copy()
    logger.info(f"Final real-world dataset: {len(result)} rows. Class distribution:\n{result['state'].value_counts()}")
    return result


def download_fl3d_raw_frames(target_dir: Path = None) -> Path:
    import kagglehub

    if target_dir is None:
        target_dir = config.DATASET_DIR / "real_kaggle"
    target_dir.mkdir(parents=True, exist_ok=True)

    dataset_handle = "matjazmuc/frame-level-driver-drowsiness-detection-fl3d"
    logger.info(f"Downloading Kaggle dataset '{dataset_handle}' via kagglehub ...")
    download_path = Path(kagglehub.dataset_download(dataset_handle))
    logger.info(f"Downloaded to cache: {download_path}")
    return download_path


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Build dataset/real_driver_drowsiness_dataset.csv from FL3D.")
    parser.add_argument(
        "--csv", type=str, default=None,
        help="Path to the FL3D statistical landmark CSV. Defaults to dataset/real_kaggle/FL3D_statistical_analysis_dataset.csv",
    )
    parser.add_argument("--max-per-class", type=int, default=1500, help="Stratified cap per class (use 0 for no cap / full dataset).")
    args = parser.parse_args()

    csv_path = Path(args.csv) if args.csv else (config.DATASET_DIR / "real_kaggle" / "FL3D_statistical_analysis_dataset.csv")
    if not csv_path.exists():
        logger.error(
            f"FL3D CSV not found at {csv_path}. Download it from "
            f"https://www.kaggle.com/datasets/matjazmuc/frame-level-driver-drowsiness-detection-fl3d "
            f"and place it there, or pass --csv <path>."
        )
        return

    cap = None if args.max_per_class == 0 else args.max_per_class
    result = build_real_dataset_from_fl3d(csv_path, max_samples_per_class=cap)

    out_csv = config.DATASET_DIR / "real_driver_drowsiness_dataset.csv"
    result.to_csv(out_csv, index=False)

    print("\n" + "=" * 70)
    print(f"SUCCESS: FL3D real-world dataset built ({len(result)} samples).")
    print(f"File: {out_csv}")
    print("NOTE: ear/mar/blink/perclos-family columns are median-imputed, not")
    print("measured (FL3D's 5-point landmarks don't include eye/mouth aperture).")
    print("Only head_pitch, head_yaw, head_roll, face_angle carry real signal.")
    print("Run 'python main.py --mode validate-real' to benchmark the")
    print("already-trained models against this real-world dataset.")
    print("=" * 70)


if __name__ == "__main__":
    main()
