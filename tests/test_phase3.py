"""
Verification script for Phase 3 (Unit 2: Linear Regression, Bayesian Logistic Regression, and SVM).
"""

import sys
from pathlib import Path
import numpy as np

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import src.config as config
from src.preprocessing import DataPreprocessor
from src.linear_regression import FatigueScoreRegressor
from src.bayesian_logistic import BayesianLogisticClassifier
from src.svm_classifier import DrowsinessSVMClassifier


def test_phase3():
    print("=" * 60)
    print("STARTING PHASE 3 VERIFICATION TEST")
    print("=" * 60)

    # 1. Load Preprocessed Data
    preprocessor = DataPreprocessor()
    df_raw = preprocessor.load_or_generate_dataset()
    df_clean, report = preprocessor.clean_dataset(df_raw)
    X, y, y_fatigue = preprocessor.prepare_xy(df_clean)
    split_res = preprocessor.split_and_scale(X, y, y_fatigue)
    X_train, X_test = split_res["X_train"], split_res["X_test"]
    y_train, y_test = split_res["y_train"], split_res["y_test"]
    y_f_train, y_f_test = split_res["y_fatigue_train"], split_res["y_fatigue_test"]

    print(f"[OK] Data loaded: Train={X_train.shape[0]}, Test={X_test.shape[0]}, Features={X_train.shape[1]}")

    # 2. Linear Regression (Continuous Fatigue Score)
    print("\n--- Testing Linear & Ridge Regression ---")
    regressor = FatigueScoreRegressor(model_type="ridge", alpha=1.0)
    regressor.fit(X_train, y_f_train)
    metrics_reg = regressor.evaluate(X_test, y_f_test)
    print(f"[OK] Ridge Regression: R^2={metrics_reg['r2_score']:.4f}, RMSE={metrics_reg['rmse']:.4f}, MAE={metrics_reg['mae']:.4f}")
    assert metrics_reg["r2_score"] > 0.80, "Regression R^2 score must be > 0.80"

    # Feature importances
    df_feat_imp = regressor.get_feature_importances()
    print(f"[OK] Top 3 most influential regression features:\n{df_feat_imp.head(3)}")
    res_plot_path = regressor.plot_residual_analysis(X_test, y_f_test)
    assert res_plot_path.exists(), "Residual plot should be saved"
    reg_model_path = regressor.save()
    assert reg_model_path.exists(), "Regressor model should be saved"

    # 3. Bayesian Logistic Regression
    print("\n--- Testing Bayesian Logistic Regression ---")
    bayes_clf = BayesianLogisticClassifier(prior_variance=1.0)
    bayes_clf.fit(X_train, y_train)
    metrics_bayes = bayes_clf.evaluate(X_test, y_test)
    print(f"[OK] Bayesian Logistic: Accuracy={metrics_bayes['accuracy']:.4f}, Macro-F1={metrics_bayes['f1_macro']:.4f}, Loss={metrics_bayes['log_loss']:.4f}")
    assert metrics_bayes["accuracy"] > 0.85, "Bayesian logistic accuracy should be > 0.85"

    # Test single sample posterior & entropy
    sample_feat = X_test[0:1]
    preds, probs, entropy = bayes_clf.predict_with_uncertainty(sample_feat)
    print(f"[OK] Sample MAP Prediction: {preds[0]} ({config.CLASS_LABELS[preds[0]]})")
    print(f"     Posterior Probs: {np.round(probs[0], 4)}, Entropy: {entropy[0]:.4f}")
    assert np.isclose(np.sum(probs[0]), 1.0), "Posterior probabilities must sum to 1.0"
    posterior_plot_path = bayes_clf.plot_posterior_distribution(probs[0], sample_title="Validation Sample 0")
    assert posterior_plot_path.exists(), "Posterior plot should be saved"
    bayes_model_path = bayes_clf.save()
    assert bayes_model_path.exists(), "Bayesian model should be saved"

    # 4. Support Vector Machines (Linear & RBF)
    print("\n--- Testing Support Vector Machines ---")
    # Linear Kernel
    svm_lin = DrowsinessSVMClassifier(kernel="linear", C=1.0)
    svm_lin.fit(X_train, y_train)
    metrics_lin = svm_lin.evaluate(X_test, y_test)
    print(f"[OK] SVM (Linear Kernel): Accuracy={metrics_lin['accuracy']:.4f}, Macro-F1={metrics_lin['f1_macro']:.4f}, Support Vectors={metrics_lin['n_support_vectors']}")
    assert metrics_lin["accuracy"] > 0.85, "Linear SVM accuracy must be > 0.85"
    svm_lin.save()

    # RBF Kernel
    svm_rbf = DrowsinessSVMClassifier(kernel="rbf", C=2.0, gamma="scale")
    svm_rbf.fit(X_train, y_train)
    metrics_rbf = svm_rbf.evaluate(X_test, y_test)
    print(f"[OK] SVM (RBF Kernel): Accuracy={metrics_rbf['accuracy']:.4f}, Macro-F1={metrics_rbf['f1_macro']:.4f}, Support Vectors={metrics_rbf['n_support_vectors']}")
    assert metrics_rbf["accuracy"] > 0.85, "RBF SVM accuracy must be > 0.85"
    boundary_plot_path = svm_rbf.plot_decision_boundary_2d(X_test, y_test)
    assert boundary_plot_path.exists(), "SVM 2D decision boundary plot should be saved"
    rbf_model_path = svm_rbf.save()
    assert rbf_model_path.exists(), "SVM RBF model should be saved"

    print("\n" + "=" * 60)
    print("PHASE 3 VERIFICATION PASSED SUCCESSFULLY!")
    print("=" * 60)


if __name__ == "__main__":
    test_phase3()
