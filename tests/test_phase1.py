import sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.preprocessing import DataPreprocessor, generate_synthetic_ddd_dataset
from src.statistics_analysis import StatisticalAnalyzer
import src.config as config


def test_phase1():
    print("=" * 60)
    print("STARTING PHASE 1 VERIFICATION TEST")
    print("=" * 60)

    # 1. Initialize preprocessor
    preprocessor = DataPreprocessor(scaler_type="standard")

    # 2. Load or generate raw dataset
    df_raw = preprocessor.load_or_generate_dataset()
    print(f"[OK] Raw dataset loaded: shape = {df_raw.shape}")
    assert len(df_raw) >= 1000, "Dataset must have at least 1000 records"

    # 3. Clean dataset (missing value imputation, duplicate removal, outlier handling)
    df_clean, cleaning_report = preprocessor.clean_dataset(df_raw)
    print(f"[OK] Dataset cleaned: shape = {df_clean.shape}")
    print(f"     Cleaning report: {cleaning_report}")
    assert df_clean[config.FEATURE_COLUMNS].isnull().sum().sum() == 0, "No NaNs should remain after cleaning"

    # 4. Prepare X, y, y_fatigue
    X, y, y_fatigue = preprocessor.prepare_xy(df_clean)
    print(f"[OK] Prepared X={X.shape}, y={y.shape}, y_fatigue={y_fatigue.shape}")
    assert len(np.unique(y)) == 4, "Must contain all 4 state classes (0..3)"

    # 5. Train-test split and scaling
    split_data = preprocessor.split_and_scale(X, y, y_fatigue, test_size=0.20)
    print(f"[OK] Split & Scaled: X_train={split_data['X_train'].shape}, X_test={split_data['X_test'].shape}")
    assert split_data["X_train"].shape[1] == len(config.FEATURE_COLUMNS)

    # 6. Single sample transformation test for real-time inference
    sample_feat = {
        "ear": 0.31,
        "mar": 0.29,
        "blink_duration": 0.18,
        "blink_rate": 15.0,
        "yawn_freq": 0.05,
        "eye_closure_dur": 0.08,
        "face_angle": 2.0,
        "head_pitch": 2.5,
        "head_yaw": 1.0,
        "head_roll": 0.5,
        "perclos": 0.05,
    }
    vec_scaled = preprocessor.transform_sample(sample_feat)
    print(f"[OK] Real-time single sample transformed: shape = {vec_scaled.shape}")
    assert vec_scaled.shape == (1, len(config.FEATURE_COLUMNS))

    # 7. Exploratory Data Analysis & Visualizations
    analyzer = StatisticalAnalyzer(df_clean)
    analysis_results = analyzer.run_full_analysis()
    print(f"[OK] Full statistical analysis executed.")
    for plot_name, path in analysis_results["generated_plots"].items():
        print(f"     Generated: {plot_name} -> {path}")
        assert Path(path).exists(), f"Plot {path} was not created"

    print("=" * 60)
    print("PHASE 1 VERIFICATION PASSED SUCCESSFULLY!")
    print("=" * 60)


if __name__ == "__main__":
    import numpy as np
    test_phase1()
