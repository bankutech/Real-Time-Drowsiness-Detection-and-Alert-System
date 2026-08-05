"""
Verification script for Phase 7 (Unified Evaluation & Benchmarking Suite).
"""

import sys
from pathlib import Path
import numpy as np

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import src.config as config
from src.preprocessing import DataPreprocessor
from src.evaluation import ModelEvaluator
from src.bayesian_logistic import BayesianLogisticClassifier
from src.svm_classifier import DrowsinessSVMClassifier
from src.decision_tree import DrowsinessDecisionTree
from src.random_forest import DrowsinessRandomForest
from src.adaboost import DrowsinessAdaBoost
from src.ensemble import DrowsinessEnsemble
from src.hmm import DrowsinessHMM


def test_phase7():
    print("=" * 60)
    print("STARTING PHASE 7 VERIFICATION TEST (UNIFIED BENCHMARKING)")
    print("=" * 60)

    # 1. Load Preprocessed Data
    preprocessor = DataPreprocessor()
    df_raw = preprocessor.load_or_generate_dataset()
    df_clean, _ = preprocessor.clean_dataset(df_raw)
    X, y, y_fatigue = preprocessor.prepare_xy(df_clean)
    split_res = preprocessor.split_and_scale(X, y, y_fatigue)
    X_train, X_test = split_res["X_train"], split_res["X_test"]
    y_train, y_test = split_res["y_train"], split_res["y_test"]

    evaluator = ModelEvaluator()

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
        B = hmm.compute_emission_probs(X_batch)
        path, _ = hmm.viterbi(B)
        return path

    def hmm_posterior_proba(X_batch):
        B = hmm.compute_emission_probs(X_batch)
        _, gamma = hmm.posterior_decode(B)
        return gamma

    evaluator.benchmark_model(
        name="HMM (Viterbi)",
        predict_fn=hmm_viterbi_pred,
        predict_proba_fn=hmm_posterior_proba,
        X_test=X_test,
        y_test=y_test,
        model_file="hmm.joblib",
    )

    # 10. Generate Summary Table & Visual Artifacts
    print("\n--- Summary Performance Table ---")
    df_summary = evaluator.generate_summary_table()
    print(df_summary.to_string())

    cm_path = evaluator.plot_confusion_matrices()
    assert cm_path.exists(), "Confusion matrices plot must exist"

    roc_path = evaluator.plot_roc_curves(y_test)
    assert roc_path.exists(), "ROC curves plot must exist"

    bench_path = evaluator.plot_benchmark_comparison()
    assert bench_path.exists(), "Benchmark plot must exist"

    print("\n" + "=" * 60)
    print("PHASE 7 VERIFICATION PASSED SUCCESSFULLY!")
    print("=" * 60)


if __name__ == "__main__":
    test_phase7()
