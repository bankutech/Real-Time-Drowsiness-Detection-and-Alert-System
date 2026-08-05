"""
Verification script for Phase 6 (Unit 5: Tree-Based Models & Ensemble Methods).
"""

import sys
from pathlib import Path
import numpy as np

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import src.config as config
from src.preprocessing import DataPreprocessor
from src.decision_tree import DrowsinessDecisionTree
from src.random_forest import DrowsinessRandomForest
from src.adaboost import DrowsinessAdaBoost
from src.ensemble import DrowsinessEnsemble


def test_phase6():
    print("=" * 60)
    print("STARTING PHASE 6 VERIFICATION TEST (TREES & ENSEMBLES)")
    print("=" * 60)

    # 1. Load Preprocessed Data
    preprocessor = DataPreprocessor()
    df_raw = preprocessor.load_or_generate_dataset()
    df_clean, _ = preprocessor.clean_dataset(df_raw)
    X, y, y_fatigue = preprocessor.prepare_xy(df_clean)
    split_res = preprocessor.split_and_scale(X, y, y_fatigue)
    X_train, X_test = split_res["X_train"], split_res["X_test"]
    y_train, y_test = split_res["y_train"], split_res["y_test"]

    # 2. Decision Tree
    print("\n--- Testing CART Decision Tree ---")
    dt = DrowsinessDecisionTree(criterion="gini", max_depth=5)
    dt.fit(X_train, y_train)
    dt_metrics = dt.evaluate(X_test, y_test)
    print(f"[OK] Decision Tree Test Accuracy: {dt_metrics['accuracy'] * 100:.2f}%, Depth: {dt_metrics['tree_depth']}")
    assert dt_metrics["accuracy"] > 0.90, "Decision tree test accuracy should exceed 90%"

    tree_plot = dt.plot_tree_structure(max_depth_plot=3)
    assert tree_plot.exists(), "Tree structure plot should be generated"

    dt_imp_plot = dt.plot_feature_importances()
    assert dt_imp_plot.exists(), "DT feature importances plot should be generated"
    dt.save()

    # 3. Random Forest
    print("\n--- Testing Random Forest Ensemble ---")
    rf = DrowsinessRandomForest(n_estimators=75, max_depth=10, oob_score=True)
    rf.fit(X_train, y_train)
    rf_metrics = rf.evaluate(X_test, y_test)
    print(f"[OK] Random Forest Test Accuracy: {rf_metrics['accuracy'] * 100:.2f}%, OOB Score: {rf_metrics['oob_score'] * 100:.2f}%")
    assert rf_metrics["accuracy"] > 0.95, "Random forest test accuracy should exceed 95%"

    rf_imp_plot = rf.plot_feature_importances()
    assert rf_imp_plot.exists(), "RF feature importances plot should be generated"

    oob_plot = rf.plot_oob_convergence(X_train, y_train, X_test, y_test, tree_range=[10, 25, 50, 75])
    assert oob_plot.exists(), "RF OOB convergence plot should be generated"
    rf.save()

    # 4. AdaBoost
    print("\n--- Testing AdaBoost Ensemble ---")
    ada = DrowsinessAdaBoost(n_estimators=40, learning_rate=1.0, base_depth=1)
    ada.fit(X_train, y_train)
    ada_metrics = ada.evaluate(X_test, y_test)
    print(f"[OK] AdaBoost Test Accuracy: {ada_metrics['accuracy'] * 100:.2f}%")
    assert ada_metrics["accuracy"] > 0.85, "AdaBoost test accuracy should exceed 85%"

    staged_plot = ada.plot_stagewise_error(X_train, y_train, X_test, y_test)
    assert staged_plot.exists(), "AdaBoost stagewise error plot should be generated"
    ada.save()

    # 5. Multi-Model Voting & Stacking Ensembles
    print("\n--- Testing Multi-Model Voting & Stacking Ensembles ---")
    ensemble = DrowsinessEnsemble(voting_mode="soft")
    ensemble.fit(X_train, y_train)
    ens_metrics = ensemble.evaluate(X_test, y_test)
    print(f"[OK] Ensemble Voting Acc: {ens_metrics['voting_accuracy'] * 100:.2f}%, Stacking Acc: {ens_metrics['stacking_accuracy'] * 100:.2f}%")
    assert ens_metrics["voting_accuracy"] > 0.95, "Voting ensemble accuracy should exceed 95%"
    assert ens_metrics["stacking_accuracy"] > 0.95, "Stacking ensemble accuracy should exceed 95%"
    ensemble.save()

    print("\n" + "=" * 60)
    print("PHASE 6 VERIFICATION PASSED SUCCESSFULLY!")
    print("=" * 60)


if __name__ == "__main__":
    test_phase6()
