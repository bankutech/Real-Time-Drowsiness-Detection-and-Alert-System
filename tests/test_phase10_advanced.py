"""
Phase 10 Advanced Architecture Tests.
Validates:
1. Temporal Differential Features (EAR Velocity dEAR/dt & EAR Acceleration d^2EAR/dt^2).
2. Dynamic Lighting Augmentation (CLAHE for Nighttime / Low-Light & Infrared Video).
3. ONNX / TensorRT Export Pipeline & Dynamic INT8 Quantization Parity Benchmark.
"""

import sys
import time
import unittest
import numpy as np
import cv2
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import config
from src.feature_extraction import BlinkAndYawnTracker, enhance_low_light_clahe
from src.preprocessing import DataPreprocessor, generate_synthetic_ddd_dataset
from src.decision_tree import DrowsinessDecisionTree
from src.random_forest import DrowsinessRandomForest
from src.bayesian_logistic import BayesianLogisticClassifier
from src.onnx_exporter import export_model_to_onnx, ONNXInferenceEngine


class TestPhase10AdvancedArchitecture(unittest.TestCase):

    def setUp(self):
        self.fps = 30.0
        self.tracker = BlinkAndYawnTracker(fps=self.fps)

    def test_01_temporal_differential_features(self):
        """Validates EAR velocity and acceleration during rapid blink vs slow microsleep."""
        # 1. Simulate rapid natural blink: Open (0.32) -> Rapid Close (0.12) in 0.06s
        t0 = 0.0
        res0 = self.tracker.update(ear=0.32, mar=0.28, current_time=t0)
        self.assertEqual(res0["ear_velocity"], 0.0)
        self.assertEqual(res0["ear_acceleration"], 0.0)

        t1 = t0 + (1.0 / self.fps)
        res1 = self.tracker.update(ear=0.15, mar=0.28, current_time=t1)
        # Velocity should be negative (closing eye)
        self.assertLess(res1["ear_velocity"], -1.0)

        # Rebound open: 0.15 -> 0.32
        t2 = t1 + (1.0 / self.fps)
        res2 = self.tracker.update(ear=0.32, mar=0.28, current_time=t2)
        # Velocity should be positive (opening eye) and acceleration positive
        self.assertGreater(res2["ear_velocity"], 1.0)
        self.assertGreater(res2["ear_acceleration"], 0.0)

        # 2. Reset and simulate slow drowsy eyelid droop (small negative velocity)
        self.tracker.reset()
        t = 0.0
        res = self.tracker.update(ear=0.30, mar=0.30, current_time=t)
        for i in range(1, 10):
            t += (1.0 / self.fps)
            ear_val = 0.30 - (0.005 * i)
            res = self.tracker.update(ear=ear_val, mar=0.30, current_time=t)
            # Velocity should be small negative
            self.assertAlmostEqual(res["ear_velocity"], -0.15, delta=0.05)

    def test_02_dynamic_lighting_clahe(self):
        """Validates automatic low-light detection and CLAHE contrast enhancement."""
        # Create a synthetic dark / nighttime underexposed frame (luminance ~ 30)
        dark_frame = np.full((480, 640, 3), 30, dtype=np.uint8)
        # Add a subtle face-like contrast pattern in the center
        cv2.circle(dark_frame, (320, 240), 100, (45, 45, 45), -1)

        enhanced_frame, is_applied, luminance = enhance_low_light_clahe(
            dark_frame,
            threshold=70.0,
            clip_limit=3.0,
            grid_size=(8, 8),
        )

        self.assertTrue(is_applied, "CLAHE should be triggered for dark frame (L < 70)")
        self.assertLess(luminance, 70.0)
        self.assertEqual(enhanced_frame.shape, dark_frame.shape)

        # Test normal optimal daylight frame (luminance ~ 140)
        bright_frame = np.full((480, 640, 3), 140, dtype=np.uint8)
        _, is_bright_applied, bright_lum = enhance_low_light_clahe(bright_frame, threshold=70.0)
        self.assertFalse(is_bright_applied, "CLAHE should NOT be applied when luminance is optimal")
        self.assertGreaterEqual(bright_lum, 70.0)

    def test_03_13_feature_preprocessor_pipeline(self):
        """Validates synthetic dataset generation and scaling with 13 features."""
        test_csv_path = config.PROJECT_ROOT / "outputs" / "test_ddd_dataset.csv"
        df = generate_synthetic_ddd_dataset(n_samples=500, inject_artifacts=True, output_path=test_csv_path)
        self.assertIn("ear_velocity", df.columns)
        self.assertIn("ear_acceleration", df.columns)
        self.assertEqual(len(config.FEATURE_COLUMNS), 13)

        preprocessor = DataPreprocessor()
        df_clean, report = preprocessor.clean_dataset(df)
        X, y, y_fatigue = preprocessor.prepare_xy(df_clean)
        self.assertEqual(X.shape[1], 13)

        splits = preprocessor.split_and_scale(X, y, y_fatigue)
        self.assertEqual(splits["X_train"].shape[1], 13)
        self.assertEqual(splits["X_test"].shape[1], 13)

    def test_04_onnx_export_and_int8_quantization(self):
        """Validates ONNX conversion, INT8 quantization, and numerical parity."""
        # Train a fast Decision Tree and Bayesian Logistic Classifier on 13 features
        X_train = np.random.randn(200, 13).astype(np.float32)
        y_train = np.random.choice([0, 1, 2, 3], size=200)

        dt = DrowsinessDecisionTree(max_depth=5)
        dt.fit(X_train, y_train)

        onnx_path, quant_path, metrics = export_model_to_onnx(
            dt.model, "test_decision_tree", n_features=13
        )

        self.assertTrue(onnx_path.exists(), "ONNX model file must exist")
        self.assertGreater(metrics["speedup_factor"], 0.0)
        self.assertLess(metrics["mse_parity"], 1e-4)

        # Test ONNXInferenceEngine on sample batches
        engine = ONNXInferenceEngine(onnx_path)
        sample_input = np.random.randn(10, 13).astype(np.float32)
        preds = engine.predict(sample_input)
        probas = engine.predict_proba(sample_input)

        self.assertEqual(len(preds), 10)
        self.assertEqual(probas.shape, (10, 4))
        # Probabilities must sum to ~1.0
        np.testing.assert_allclose(np.sum(probas, axis=1), np.ones(10), atol=1e-3)


if __name__ == "__main__":
    unittest.main(verbosity=2)
