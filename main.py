"""
Real-Time Drowsiness Detection and Alert System
Main Orchestration & Pipeline Entry Point.

This module unifies all 5 syllabus units into a complete, runnable CLI:
- Unit 1: Preprocessing & Exploratory Data Analysis
- Unit 2: Linear Regression, Bayesian Logistic Regression, SVM
- Unit 3: PCA & Unsupervised Clustering (K-Means, GMM, Hierarchical)
- Unit 4: Hidden Markov Model (Sequential Modeling & Viterbi Decoding)
- Unit 5: Decision Tree, Random Forest, AdaBoost, Ensemble Learning
- Evaluation: Unified Multi-Model Benchmarking & Report Generation
- Real-Time: Live OpenCV Cockpit HUD & Multi-Tier Audio/Visual Alerting
"""

import sys
import argparse
import time
from pathlib import Path
from typing import Optional

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import cv2
import numpy as np
import pandas as pd

from src import config
from src.utils import setup_logger
from src.preprocessing import generate_synthetic_ddd_dataset, DataPreprocessor
from src.statistics_analysis import StatisticalAnalyzer
from src.linear_regression import FatigueScoreRegressor
from src.bayesian_logistic import BayesianLogisticClassifier
from src.svm_classifier import DrowsinessSVMClassifier
from src.pca import PCAReducer
from src.kmeans import DrowsinessKMeans
from src.gmm import DrowsinessGMM
from src.hierarchical import DrowsinessHierarchicalClustering
from src.hmm import DrowsinessHMM
from src.decision_tree import DrowsinessDecisionTree
from src.random_forest import DrowsinessRandomForest
from src.adaboost import DrowsinessAdaBoost
from src.ensemble import DrowsinessEnsemble
from src.evaluation import ModelEvaluator
from src.realtime_detection import DrowsinessDetectorPipeline

logger = setup_logger("MainPipeline")


def run_training_pipeline() -> None:
    """Executes full training lifecycle across all 5 syllabus units."""
    logger.info("=" * 70)
    logger.info("STARTING FULL END-TO-END MODEL TRAINING PIPELINE (UNITS 1 - 5)")
    logger.info("=" * 70)

    # 1. Unit 1: Data Generation, Preprocessing & EDA
    logger.info("\n>>> [UNIT 1] Preprocessing & Exploratory Data Analysis")
    preprocessor = DataPreprocessor()
    raw_df = preprocessor.load_or_generate_dataset(config.DATASET_CSV_PATH)
    df_cleaned, report = preprocessor.clean_dataset(raw_df)
    df_cleaned.to_csv(config.DATASET_CLEANED_CSV_PATH, index=False)
    logger.info(f"Saved cleaned dataset to {config.DATASET_CLEANED_CSV_PATH}")

    analyzer = StatisticalAnalyzer(df_cleaned, output_dir=config.EDA_OUTPUT_DIR)
    analyzer.run_full_analysis()

    X, y, y_fatigue = preprocessor.prepare_xy(df_cleaned)
    splits = preprocessor.split_and_scale(X, y, y_fatigue)

    X_train = splits["X_train"]
    X_test = splits["X_test"]
    y_train = splits["y_train"]
    y_test = splits["y_test"]
    y_train_fatigue = splits["y_fatigue_train"]
    y_test_fatigue = splits["y_fatigue_test"]

    # 2. Unit 2: Regression, Bayesian Logistic & SVM
    logger.info("\n>>> [UNIT 2] Linear Models & Support Vector Machines")
    regressor = FatigueScoreRegressor(model_type="ridge", alpha=1.0)
    regressor.fit(X_train, y_train_fatigue)
    reg_metrics = regressor.evaluate(X_test, y_test_fatigue)
    regressor.save()

    bayes_clf = BayesianLogisticClassifier()
    bayes_clf.fit(X_train, y_train)
    bayes_metrics = bayes_clf.evaluate(X_test, y_test)
    bayes_clf.save()

    svm_linear = DrowsinessSVMClassifier(kernel="linear", C=1.0)
    svm_linear.fit(X_train, y_train)
    svm_linear.evaluate(X_test, y_test)
    svm_linear.save("svm_linear.joblib")

    svm_rbf = DrowsinessSVMClassifier(kernel="rbf", C=1.0)
    svm_rbf.fit(X_train, y_train)
    svm_rbf.evaluate(X_test, y_test)
    svm_rbf.save("svm_rbf.joblib")

    # 3. Unit 3: PCA & Unsupervised Clustering
    logger.info("\n>>> [UNIT 3] Dimensionality Reduction & Unsupervised Clustering")
    pca = PCAReducer(n_components=5)
    pca.fit(X_train)
    pca.plot_scree_plot(output_dir=config.CLUSTERING_OUTPUT_DIR)
    pca.plot_2d_projection(X_train, y_train, output_dir=config.CLUSTERING_OUTPUT_DIR)
    pca.save()

    kmeans = DrowsinessKMeans(n_clusters=config.NUM_CLASSES)
    kmeans.fit(X_train)
    kmeans.evaluate_clustering(X_train, y_train)
    kmeans.plot_elbow_and_silhouette(X_train, output_dir=config.CLUSTERING_OUTPUT_DIR)
    kmeans.save()

    gmm = DrowsinessGMM(n_components=config.NUM_CLASSES)
    gmm.fit(X_train)
    gmm.evaluate(X_train, y_train)
    gmm.plot_aic_bic_curves(X_train, output_dir=config.CLUSTERING_OUTPUT_DIR)
    gmm.save()

    hier = DrowsinessHierarchicalClustering(n_clusters=config.NUM_CLASSES)
    hier.fit_predict(X_train[:400])
    hier.plot_dendrogram(output_dir=config.CLUSTERING_OUTPUT_DIR)
    hier.save()

    # 4. Unit 4: Hidden Markov Model
    logger.info("\n>>> [UNIT 4] Hidden Markov Model & Sequential Smoothing")
    hmm = DrowsinessHMM(n_states=config.NUM_CLASSES)
    hmm.fit_emissions(X_train, y_train)
    # Unit 4 Baum-Welch Expectation-Maximization refinement on sequential time-series
    hmm.fit_baum_welch(X_train[:600], max_iter=25, update_emissions=True)
    hmm.plot_transition_matrix(output_dir=config.EVAL_OUTPUT_DIR)

    raw_preds = bayes_clf.predict(X_test)
    emission_probas = bayes_clf.predict_proba(X_test)
    viterbi_preds, _ = hmm.viterbi_decode(emission_probas)
    hmm.plot_sequence_decoding(y_test, raw_preds, viterbi_preds, output_dir=config.EVAL_OUTPUT_DIR)
    hmm.save()

    # 5. Unit 5: Decision Tree, Random Forest, AdaBoost & Ensembles
    logger.info("\n>>> [UNIT 5] Tree-Based & Ensemble Learning Architectures")
    dt = DrowsinessDecisionTree(max_depth=8)
    dt.fit(X_train, y_train)
    dt.evaluate(X_test, y_test)
    dt.plot_tree_structure(output_dir=config.EVAL_OUTPUT_DIR)
    dt.save()

    rf = DrowsinessRandomForest(n_estimators=100, max_depth=10)
    rf.fit(X_train, y_train)
    rf.evaluate(X_test, y_test)
    rf.plot_feature_importances(output_dir=config.EVAL_OUTPUT_DIR)
    rf.plot_oob_error_curve(X_train, y_train, output_dir=config.EVAL_OUTPUT_DIR)
    rf.save()

    ada = DrowsinessAdaBoost(n_estimators=80, learning_rate=0.8)
    ada.fit(X_train, y_train)
    ada.evaluate(X_test, y_test)
    ada.plot_estimator_weights(output_dir=config.EVAL_OUTPUT_DIR)
    ada.save()

    ensemble = DrowsinessEnsemble(voting_mode="soft")
    ensemble.fit(X_train, y_train)
    ensemble.evaluate(X_test, y_test)
    ensemble.save()

    logger.info("=" * 70)
    logger.info("ALL UNITS 1-5 SUCCESSFULLY TRAINED AND PERSISTED TO DISK!")
    logger.info("=" * 70)


def run_evaluation_pipeline() -> None:
    """Executes unified benchmarking suite across all models and generates comparison tables."""
    logger.info("=" * 70)
    logger.info("STARTING UNIFIED MODEL EVALUATION & BENCHMARKING SUITE")
    logger.info("=" * 70)

    # 1. Load Preprocessed Data
    preprocessor = DataPreprocessor()
    df_raw = preprocessor.load_or_generate_dataset()
    df_clean, _ = preprocessor.clean_dataset(df_raw)
    X, y, y_fatigue = preprocessor.prepare_xy(df_clean)
    split_res = preprocessor.split_and_scale(X, y, y_fatigue)
    X_train, X_test = split_res["X_train"], split_res["X_test"]
    y_train, y_test = split_res["y_train"], split_res["y_test"]

    evaluator = ModelEvaluator(output_dir=config.EVAL_OUTPUT_DIR)

    # 2. Benchmark Bayesian Logistic Regression
    bayes_lr = BayesianLogisticClassifier.load()
    evaluator.benchmark_model(
        name="Bayesian Logistic",
        predict_fn=bayes_lr.predict,
        predict_proba_fn=bayes_lr.predict_proba,
        X_test=X_test,
        y_test=y_test,
        model_file="bayesian_logistic.joblib",
    )

    # 3. Benchmark SVM Linear
    svm_lin = DrowsinessSVMClassifier.load("svm_linear.joblib")
    evaluator.benchmark_model(
        name="SVM (Linear)",
        predict_fn=svm_lin.predict,
        predict_proba_fn=svm_lin.predict_proba,
        X_test=X_test,
        y_test=y_test,
        model_file="svm_linear.joblib",
    )

    # 4. Benchmark SVM RBF
    svm_rbf = DrowsinessSVMClassifier.load("svm_rbf.joblib")
    evaluator.benchmark_model(
        name="SVM (RBF)",
        predict_fn=svm_rbf.predict,
        predict_proba_fn=svm_rbf.predict_proba,
        X_test=X_test,
        y_test=y_test,
        model_file="svm_rbf.joblib",
    )

    # 5. Benchmark Decision Tree
    dt = DrowsinessDecisionTree.load()
    evaluator.benchmark_model(
        name="Decision Tree",
        predict_fn=dt.predict,
        predict_proba_fn=dt.predict_proba,
        X_test=X_test,
        y_test=y_test,
        model_file="decision_tree.joblib",
    )

    # 6. Benchmark Random Forest
    rf = DrowsinessRandomForest.load()
    evaluator.benchmark_model(
        name="Random Forest",
        predict_fn=rf.predict,
        predict_proba_fn=rf.predict_proba,
        X_test=X_test,
        y_test=y_test,
        model_file="random_forest.joblib",
    )

    # 7. Benchmark AdaBoost
    ada = DrowsinessAdaBoost.load()
    evaluator.benchmark_model(
        name="AdaBoost",
        predict_fn=ada.predict,
        predict_proba_fn=ada.predict_proba,
        X_test=X_test,
        y_test=y_test,
        model_file="adaboost.joblib",
    )

    # 8. Benchmark Stacking Ensemble
    ensemble = DrowsinessEnsemble.load()
    evaluator.benchmark_model(
        name="Stacking Ensemble",
        predict_fn=ensemble.predict_stacking,
        predict_proba_fn=ensemble.predict_proba,
        X_test=X_test,
        y_test=y_test,
        model_file="ensemble_stacking.joblib",
    )

    # 9. Benchmark HMM Viterbi Decoder
    hmm = DrowsinessHMM.load()
    def hmm_viterbi_pred(X_batch):
        emission_p = bayes_lr.predict_proba(X_batch)
        path, _ = hmm.viterbi_decode(emission_p)
        return path

    evaluator.benchmark_model(
        name="HMM (Viterbi)",
        predict_fn=hmm_viterbi_pred,
        predict_proba_fn=bayes_lr.predict_proba,
        X_test=X_test,
        y_test=y_test,
        model_file="hmm.joblib",
    )

    # 10. Generate Visuals & Summary Reports
    evaluator.plot_confusion_matrices()
    evaluator.plot_roc_curves(y_test)
    evaluator.plot_calibration_curves(y_test)
    evaluator.plot_benchmark_comparison()

    summary_df = evaluator.generate_summary_table()

    print("\n" + "=" * 70)
    print("UNIFIED MODEL COMPARISON LEADERBOARD")
    print("=" * 70)
    print(summary_df.to_string(index=False))
    print("=" * 70)


def run_live_detection(camera_idx: int = 0, video_path: Optional[str] = None, enable_audio: bool = True) -> None:
    """Launches real-time OpenCV detection stream on webcam or video file."""
    logger.info("=" * 70)
    logger.info("LAUNCHING REAL-TIME DROWSINESS DETECTION & ALERT SYSTEM")
    logger.info("=" * 70)

    source = video_path if video_path else camera_idx
    if video_path:
        cap = cv2.VideoCapture(video_path)
    else:
        cap = None
        if sys.platform.startswith("win"):
            try:
                cap = cv2.VideoCapture(camera_idx, cv2.CAP_DSHOW)
                if cap.isOpened():
                    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
                    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            except Exception:
                pass
        if cap is None or not cap.isOpened():
            cap = cv2.VideoCapture(camera_idx)
            if cap.isOpened():
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
                cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    if not cap or not cap.isOpened():
        logger.error(f"Cannot open video source: {source}")
        return

    pipeline = DrowsinessDetectorPipeline(primary_model_type="ensemble", enable_audio=enable_audio)

    window_title = config.DISPLAY_WINDOW_NAME
    cv2.namedWindow(window_title, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_title, 1280, 720)

    logger.info("Real-Time Stream Active. Controls: [Q] Quit | [M] Switch Model | [S] Save Snapshot")

    models_list = ["ensemble", "random_forest", "bayesian_logistic", "decision_tree"]
    model_idx = 0

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                logger.info("End of video stream reached.")
                break

            # Process frame through full CV + ML + HMM + HUD pipeline
            hud_frame, telemetry = pipeline.process_frame(frame)

            # Display frame
            cv2.imshow(window_title, hud_frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q") or key == 27:  # Q or ESC
                break
            elif key == ord("m"):
                model_idx = (model_idx + 1) % len(models_list)
                pipeline.primary_model_type = models_list[model_idx]
                logger.info(f"Switched Primary ML Classifier to: {pipeline.primary_model_type.upper()}")
            elif key == ord("s"):
                snap_path = config.OUTPUTS_DIR / f"snapshot_frame_{telemetry['frame_idx']}.png"
                cv2.imwrite(str(snap_path), hud_frame)
                logger.info(f"Saved telemetry snapshot to {snap_path}")

    finally:
        cap.release()
        cv2.destroyAllWindows()
        pipeline.alert_manager.save_logs()
        logger.info("Stream closed and alert logs successfully saved.")


def run_simulation_demo(duration_frames: int = 250, enable_audio: bool = True, show_window: bool = True) -> None:
    """Executes an interactive driver simulation demo transitioning through states."""
    logger.info("=" * 70)
    logger.info("RUNNING DRIVER DROWSINESS SIMULATION DEMO")
    logger.info("=" * 70)

    from tests.test_phase8 import generate_synthetic_driver_frame

    pipeline = DrowsinessDetectorPipeline(primary_model_type="ensemble", enable_audio=enable_audio)

    window_title = "Driver Drowsiness Simulation Demo (OpenCV HUD)"
    if show_window:
        try:
            cv2.namedWindow(window_title, cv2.WINDOW_NORMAL)
            cv2.resizeWindow(window_title, 960, 540)
        except Exception:
            show_window = False

    # 4-stage progressive sequence: Alert -> Slight Fatigue -> Yawning/Drowsy -> Severe Microsleep
    sequence = (
        ["alert"] * 60
        + ["drowsy"] * 80
        + ["sleeping"] * 70
        + ["alert"] * 40
    )

    logger.info(f"Playing simulated driver trajectory ({len(sequence)} frames)...")

    for idx, state_type in enumerate(sequence[:duration_frames]):
        frame = generate_synthetic_driver_frame(state_type)

        # Inject stage description on frame
        stage_desc = f"SIMULATION SCENARIO: {state_type.upper()} PHASE (Frame {idx+1}/{len(sequence)})"
        cv2.putText(frame, stage_desc, (15, 460), cv2.FONT_HERSHEY_DUPLEX, 0.45, (0, 255, 255), 1)

        hud_frame, telemetry = pipeline.process_frame(frame)

        if show_window:
            try:
                cv2.imshow(window_title, hud_frame)
                key = cv2.waitKey(30) & 0xFF
                if key == ord("q") or key == 27:
                    break
            except Exception:
                show_window = False

    if show_window:
        try:
            cv2.destroyAllWindows()
        except Exception:
            pass

    pipeline.alert_manager.save_logs()
    logger.info("Simulation demo finished successfully.")


def main():
    parser = argparse.ArgumentParser(
        description="Real-Time Drowsiness Detection and Alert System",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["train", "evaluate", "live", "demo", "all"],
        default="all",
        help=(
            "Execution mode:\n"
            "  train    : Train all ML models across Units 1-5\n"
            "  evaluate : Run unified benchmarking and output leaderboard\n"
            "  live     : Start live detection on webcam or video\n"
            "  demo     : Run simulated driver drowsiness test with HUD\n"
            "  all      : Train models, benchmark evaluation, and run demo\n"
        ),
    )
    parser.add_argument("--camera", type=int, default=0, help="Camera index for live detection (default: 0)")
    parser.add_argument("--video", type=str, default=None, help="Path to video file for offline analysis")
    parser.add_argument("--no-audio", action="store_true", help="Disable audio alarm tones")

    args = parser.parse_args()
    enable_audio = not args.no_audio

    if args.mode == "train":
        run_training_pipeline()
    elif args.mode == "evaluate":
        run_evaluation_pipeline()
    elif args.mode == "live":
        run_live_detection(camera_idx=args.camera, video_path=args.video, enable_audio=enable_audio)
    elif args.mode == "demo":
        run_simulation_demo(enable_audio=enable_audio)
    elif args.mode == "all":
        run_training_pipeline()
        run_evaluation_pipeline()
        logger.info("\nPipeline execution complete. Launching demo...")
        run_simulation_demo(duration_frames=120, enable_audio=False)


if __name__ == "__main__":
    main()
