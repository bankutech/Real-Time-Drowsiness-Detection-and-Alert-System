"""
Preprocessing module (Unit 1: Data Understanding & Cleaning).
Implements dataset loading, realistic Driver Drowsiness Dataset (DDD) synthesis,
missing value detection and imputation, duplicate removal, outlier handling,
feature scaling, and stratified train-test splitting.
"""

from pathlib import Path
from typing import Tuple, Dict, Any, Optional

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, RobustScaler

from src import config
from src.utils import setup_logger, save_model, load_model

logger = setup_logger("Preprocessing")


def generate_synthetic_ddd_dataset(
    n_samples: int = 4000,
    random_state: int = config.RANDOM_STATE,
    output_path: Optional[Path] = None,
    inject_artifacts: bool = True,
) -> pd.DataFrame:
    """
    Generates a realistic feature dataset modeled after the Driver Drowsiness Dataset (DDD)
    and physiological facial landmark measurements across 4 states:
    Alert, Slightly Drowsy, Drowsy, Sleep.

    Also calculates ground-truth Fatigue Score (0-100) and injects realistic missing values,
    duplicates, and sensor outliers for Unit 1 preprocessing demonstration.
    """
    np.random.seed(random_state)
    logger.info(f"Generating synthetic Driver Drowsiness Dataset ({n_samples} samples)...")

    # Sample allocation across 4 classes
    class_proportions = [0.35, 0.25, 0.25, 0.15]  # Alert, Slightly Drowsy, Drowsy, Sleep
    samples_per_class = [int(n_samples * p) for p in class_proportions]
    samples_per_class[0] += n_samples - sum(samples_per_class)  # Balance total

    records = []

    # Distribution parameters: (mean, std) for each state
    # Features: ear, mar, blink_duration, blink_rate, yawn_freq, eye_closure_dur, face_angle, pitch, yaw, roll, perclos
    state_profiles = {
        "Alert": {
            "ear": (0.32, 0.025),
            "mar": (0.28, 0.035),
            "blink_duration": (0.18, 0.03),
            "blink_rate": (14.0, 2.5),
            "yawn_freq": (0.05, 0.04),
            "eye_closure_dur": (0.08, 0.02),
            "face_angle": (0.0, 3.0),
            "head_pitch": (2.0, 2.5),
            "head_yaw": (0.0, 3.0),
            "head_roll": (0.0, 2.0),
            "perclos": (0.04, 0.02),
            "base_fatigue": (12.0, 4.0),
        },
        "Slightly Drowsy": {
            "ear": (0.25, 0.020),
            "mar": (0.36, 0.050),
            "blink_duration": (0.28, 0.04),
            "blink_rate": (22.0, 3.5),
            "yawn_freq": (0.20, 0.08),
            "eye_closure_dur": (0.20, 0.04),
            "face_angle": (4.0, 3.5),
            "head_pitch": (7.0, 3.5),
            "head_yaw": (4.0, 4.0),
            "head_roll": (3.0, 2.5),
            "perclos": (0.18, 0.04),
            "base_fatigue": (38.0, 6.0),
        },
        "Drowsy": {
            "ear": (0.20, 0.018),
            "mar": (0.54, 0.100),
            "blink_duration": (0.45, 0.07),
            "blink_rate": (28.0, 4.5),
            "yawn_freq": (0.55, 0.12),
            "eye_closure_dur": (0.55, 0.10),
            "face_angle": (12.0, 5.0),
            "head_pitch": (18.0, 5.0),
            "head_yaw": (10.0, 6.0),
            "head_roll": (8.0, 4.0),
            "perclos": (0.44, 0.07),
            "base_fatigue": (68.0, 7.0),
        },
        "Sleep": {
            "ear": (0.13, 0.015),
            "mar": (0.38, 0.060),
            "blink_duration": (0.95, 0.15),
            "blink_rate": (6.0, 2.0),
            "yawn_freq": (0.15, 0.08),
            "eye_closure_dur": (1.80, 0.35),
            "face_angle": (24.0, 6.0),
            "head_pitch": (30.0, 6.5),
            "head_yaw": (14.0, 7.0),
            "head_roll": (15.0, 5.0),
            "perclos": (0.85, 0.08),
            "base_fatigue": (92.0, 4.0),
        },
    }

    frame_counter = 1
    for state_idx, state_label in enumerate(config.CLASS_LABELS):
        count = samples_per_class[state_idx]
        prof = state_profiles[state_label]

        for _ in range(count):
            ear = float(np.clip(np.random.normal(prof["ear"][0], prof["ear"][1]), 0.08, 0.45))
            mar = float(np.clip(np.random.normal(prof["mar"][0], prof["mar"][1]), 0.15, 0.95))
            blink_dur = float(np.clip(np.random.normal(prof["blink_duration"][0], prof["blink_duration"][1]), 0.05, 2.50))
            blink_rate = float(np.clip(np.random.normal(prof["blink_rate"][0], prof["blink_rate"][1]), 1.0, 45.0))
            yawn_freq = float(np.clip(np.random.normal(prof["yawn_freq"][0], prof["yawn_freq"][1]), 0.0, 1.0))
            eye_close_dur = float(np.clip(np.random.normal(prof["eye_closure_dur"][0], prof["eye_closure_dur"][1]), 0.0, 4.0))
            pitch = float(np.clip(np.random.normal(prof["head_pitch"][0], prof["head_pitch"][1]), -10.0, 60.0))
            yaw = float(np.clip(np.random.normal(prof["head_yaw"][0], prof["head_yaw"][1]), -45.0, 45.0))
            roll = float(np.clip(np.random.normal(prof["head_roll"][0], prof["head_roll"][1]), -35.0, 35.0))
            face_angle = float(np.hypot(pitch, np.hypot(yaw, roll)))
            perclos = float(np.clip(np.random.normal(prof["perclos"][0], prof["perclos"][1]), 0.0, 1.0))

            # Fatigue Score (0-100)
            fatigue = float(
                np.clip(
                    np.random.normal(prof["base_fatigue"][0], prof["base_fatigue"][1])
                    + 10.0 * perclos
                    - 8.0 * (ear - 0.25)
                    + 5.0 * (mar - 0.35)
                    + 0.1 * pitch,
                    0.0,
                    100.0,
                )
            )

            records.append({
                "frame_number": frame_counter,
                "timestamp": frame_counter / 30.0,
                "ear": ear,
                "mar": mar,
                "blink_duration": blink_dur,
                "blink_rate": blink_rate,
                "yawn_freq": yawn_freq,
                "eye_closure_dur": eye_close_dur,
                "face_angle": face_angle,
                "head_pitch": pitch,
                "head_yaw": yaw,
                "head_roll": roll,
                "perclos": perclos,
                "fatigue_score": fatigue,
                "state": state_label,
            })
            frame_counter += 1

    df = pd.DataFrame(records)

    # Inject missing values, duplicates, and outliers for Unit 1 cleaning demonstration
    if inject_artifacts:
        # 1. Inject ~1.5% missing values randomly in feature columns
        mask_missing = np.random.rand(*df[config.FEATURE_COLUMNS].shape) < 0.015
        for col_idx, col in enumerate(config.FEATURE_COLUMNS):
            df.loc[mask_missing[:, col_idx], col] = np.nan

        # 2. Inject 20 duplicate rows
        duplicates = df.sample(n=25, random_state=random_state)
        df = pd.concat([df, duplicates], ignore_index=True)

        # 3. Inject a few realistic sensor glitch outliers (e.g. noise burst)
        outlier_indices = df.sample(n=10, random_state=random_state).index
        df.loc[outlier_indices[:5], "ear"] = 1.25  # Impossible EAR > 1.0
        df.loc[outlier_indices[5:], "blink_duration"] = 15.0  # Glitch blink duration

        logger.info(f"Injected artifacts for Unit 1 EDA: NaNs, duplicates ({len(duplicates)}), outliers ({len(outlier_indices)})")

    # Shuffle dataset
    df = df.sample(frac=1.0, random_state=random_state).reset_index(drop=True)

    target_file = output_path or config.DATASET_CSV_PATH
    df.to_csv(target_file, index=False)
    logger.info(f"Dataset generated and saved to: {target_file} (Shape: {df.shape})")
    return df


class DataPreprocessor:
    """
    Unit 1 Data Preprocessing Pipeline.
    Handles data loading, missing value imputation, duplicate removal,
    outlier capping/filtering, train-test splitting, and feature scaling.
    """

    def __init__(self, scaler_type: str = "standard"):
        self.scaler_type = scaler_type
        self.scaler = StandardScaler() if scaler_type == "standard" else RobustScaler()
        self.feature_columns = config.FEATURE_COLUMNS
        self.target_column = config.TARGET_COLUMN
        self.regression_target = config.REGRESSION_TARGET
        self.imputation_values: Dict[str, float] = {}

    def load_or_generate_dataset(self, file_path: Optional[Path] = None) -> pd.DataFrame:
        """Loads dataset from disk or automatically generates it if not present."""
        target_path = file_path or config.DATASET_CSV_PATH
        if not target_path.exists():
            logger.warning(f"Dataset file {target_path} not found. Generating realistic DDD dataset...")
            return generate_synthetic_ddd_dataset(output_path=target_path)
        logger.info(f"Loading raw dataset from {target_path}...")
        return pd.read_csv(target_path)

    def clean_dataset(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        Executes full Unit 1 data cleaning pipeline:
        1. Duplicate detection & removal
        2. Missing value detection & median imputation
        3. Sensor outlier capping / filtering via IQR bounds
        """
        initial_rows = len(df)
        cleaning_report = {}

        # 1. Duplicates
        duplicates_count = int(df.duplicated().sum())
        df_cleaned = df.drop_duplicates().copy().reset_index(drop=True)
        cleaning_report["duplicates_removed"] = duplicates_count
        logger.info(f"Removed {duplicates_count} duplicate rows ({initial_rows} -> {len(df_cleaned)} rows)")

        # 2. Missing Values
        missing_summary = df_cleaned[self.feature_columns].isnull().sum().to_dict()
        total_missing = sum(missing_summary.values())
        cleaning_report["missing_values_imputed"] = total_missing

        for col in self.feature_columns:
            median_val = float(df_cleaned[col].median())
            self.imputation_values[col] = median_val
            df_cleaned[col] = df_cleaned[col].fillna(median_val)

        logger.info(f"Imputed {total_missing} missing values using column medians.")

        # 3. Outlier Handling via IQR Filtering on Physical Sensor Bounds
        outlier_count = 0
        for col in self.feature_columns:
            q1 = df_cleaned[col].quantile(0.01)
            q3 = df_cleaned[col].quantile(0.99)
            iqr = q3 - q1
            lower_bound = q1 - 2.5 * iqr
            upper_bound = q3 + 2.5 * iqr

            # Specific physical constraints
            if col == "ear":
                lower_bound = max(0.05, lower_bound)
                upper_bound = min(0.60, upper_bound)
            elif col == "mar":
                lower_bound = max(0.10, lower_bound)
                upper_bound = min(1.20, upper_bound)
            elif col == "perclos":
                lower_bound = max(0.0, lower_bound)
                upper_bound = min(1.0, upper_bound)

            outliers_mask = (df_cleaned[col] < lower_bound) | (df_cleaned[col] > upper_bound)
            outlier_count += int(outliers_mask.sum())
            # Cap outliers to bounds
            df_cleaned[col] = df_cleaned[col].clip(lower=lower_bound, upper=upper_bound)

        cleaning_report["outliers_capped"] = outlier_count
        logger.info(f"Capped {outlier_count} extreme sensor outliers within domain bounds.")

        # Save cleaned dataset
        df_cleaned.to_csv(config.DATASET_CLEANED_CSV_PATH, index=False)
        logger.info(f"Saved cleaned dataset to {config.DATASET_CLEANED_CSV_PATH} (Shape: {df_cleaned.shape})")

        return df_cleaned, cleaning_report

    def prepare_xy(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Extracts feature matrix X, multi-class labels y (encoded 0..3), and continuous fatigue scores.
        """
        X = df[self.feature_columns].values
        # Encode state strings to integers
        y_labels = df[self.target_column].values
        y = np.array([config.LABEL_TO_ID.get(lbl, 0) for lbl in y_labels], dtype=np.int32)
        y_fatigue = df[self.regression_target].values if self.regression_target in df else np.zeros(len(df))

        return X, y, y_fatigue

    def split_and_scale(
        self,
        X: np.ndarray,
        y: np.ndarray,
        y_fatigue: Optional[np.ndarray] = None,
        test_size: float = config.TEST_SIZE,
        random_state: int = config.RANDOM_STATE,
    ) -> Dict[str, Any]:
        """
        Stratified train-test split followed by feature scaling.
        Saves fitted scaler to models directory.
        """
        if y_fatigue is None:
            y_fatigue = np.zeros(len(y))

        indices = np.arange(len(X))
        train_idx, test_idx = train_test_split(
            indices,
            test_size=test_size,
            random_state=random_state,
            stratify=y,
        )

        X_train_raw, X_test_raw = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        y_fatigue_train, y_fatigue_test = y_fatigue[train_idx], y_fatigue[test_idx]

        # Fit Scaler on training set only to prevent data leakage
        X_train = self.scaler.fit_transform(X_train_raw)
        X_test = self.scaler.transform(X_test_raw)

        # Save fitted scaler
        save_model(self.scaler, "scaler.joblib")
        save_model(self.imputation_values, "imputer.joblib")

        logger.info(f"Data split & scaled: Train={X_train.shape[0]} samples, Test={X_test.shape[0]} samples.")

        return {
            "X_train": X_train,
            "X_test": X_test,
            "y_train": y_train,
            "y_test": y_test,
            "y_fatigue_train": y_fatigue_train,
            "y_fatigue_test": y_fatigue_test,
            "scaler": self.scaler,
            "feature_names": self.feature_columns,
        }

    def transform_sample(self, feature_dict: Dict[str, float]) -> np.ndarray:
        """Transforms a single feature dictionary for real-time live inference."""
        vec = []
        for col in self.feature_columns:
            val = feature_dict.get(col, self.imputation_values.get(col, 0.0))
            vec.append(val)
        vec_arr = np.array(vec, dtype=np.float32).reshape(1, -1)
        return self.scaler.transform(vec_arr)
