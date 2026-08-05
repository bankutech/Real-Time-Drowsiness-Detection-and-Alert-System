"""
Ensemble Learning Module (Unit 5: Advanced Ensemble Methods).
Implements Soft/Hard Voting Ensembles and Stacking Classifiers with Meta-Learners,
combining heterogeneous models (Bayesian Logistic, SVM, Random Forest, AdaBoost).
"""

from pathlib import Path
from typing import Dict, Any, Optional, Tuple, List

import numpy as np
import pandas as pd
from sklearn.ensemble import VotingClassifier, StackingClassifier, RandomForestClassifier, AdaBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, f1_score, classification_report

from src import config
from src.utils import setup_logger, save_model, load_model

logger = setup_logger("Ensemble")


class DrowsinessEnsemble:
    """
    Unit 5 Unified Multi-Model Ensemble Engine.
    Leverages consensus from diverse algorithm families for maximum generalization.
    """

    def __init__(self, voting_mode: str = "soft"):
        self.voting_mode = voting_mode

        # Base estimators
        rf = RandomForestClassifier(n_estimators=50, max_depth=10, random_state=config.RANDOM_STATE, n_jobs=-1)
        ada = AdaBoostClassifier(n_estimators=40, random_state=config.RANDOM_STATE)
        log_reg = LogisticRegression(max_iter=500, random_state=config.RANDOM_STATE)
        svm_calibrated = CalibratedClassifierCV(
            estimator=SVC(kernel="rbf", C=2.0, random_state=config.RANDOM_STATE),
            ensemble=False,
        )

        self.estimators = [
            ("random_forest", rf),
            ("adaboost", ada),
            ("bayesian_logistic", log_reg),
            ("svm_rbf", svm_calibrated),
        ]

        # 1. Voting Classifier
        self.voting_model = VotingClassifier(
            estimators=self.estimators,
            voting=voting_mode,
            n_jobs=-1,
        )

        # 2. Stacking Classifier with Meta-Learner Logistic Regression
        self.stacking_model = StackingClassifier(
            estimators=self.estimators,
            final_estimator=LogisticRegression(max_iter=500, random_state=config.RANDOM_STATE),
            cv=5,
            n_jobs=-1,
        )

        self.class_names = config.CLASS_LABELS
        self.metrics: Dict[str, Any] = {}

    def fit(self, X_train: np.ndarray, y_train: np.ndarray) -> "DrowsinessEnsemble":
        """Fits both Voting and Stacking ensemble architectures."""
        logger.info(f"Fitting Voting Ensemble ({self.voting_mode}) on {X_train.shape[0]} samples...")
        self.voting_model.fit(X_train, y_train)

        logger.info(f"Fitting Stacking Ensemble (5-fold CV meta-learner) on {X_train.shape[0]} samples...")
        self.stacking_model.fit(X_train, y_train)

        logger.info("Ensemble models successfully fitted.")
        return self

    def predict_voting(self, X: np.ndarray) -> np.ndarray:
        """Predicts class labels via Voting."""
        return self.voting_model.predict(X)

    def predict_stacking(self, X: np.ndarray) -> np.ndarray:
        """Predicts class labels via Stacking meta-learner."""
        return self.stacking_model.predict(X)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predicts ensemble soft probabilities (defaults to Stacking)."""
        return self.stacking_model.predict_proba(X)

    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, Any]:
        """Evaluates both Voting and Stacking ensembles."""
        y_pred_v = self.predict_voting(X_test)
        y_pred_s = self.predict_stacking(X_test)

        acc_v = float(accuracy_score(y_test, y_pred_v))
        f1_v = float(f1_score(y_test, y_pred_v, average="macro"))

        acc_s = float(accuracy_score(y_test, y_pred_s))
        f1_s = float(f1_score(y_test, y_pred_s, average="macro"))

        self.metrics = {
            "voting_accuracy": acc_v,
            "voting_macro_f1": f1_v,
            "stacking_accuracy": acc_s,
            "stacking_macro_f1": f1_s,
            "stacking_report": classification_report(y_test, y_pred_s, target_names=self.class_names, output_dict=True),
        }
        logger.info(f"Ensemble Evaluation: Voting Acc={acc_v:.4f} (F1={f1_v:.4f}) | Stacking Acc={acc_s:.4f} (F1={f1_s:.4f})")
        return self.metrics

    def save(self, voting_filename: str = "ensemble_voting.joblib", stacking_filename: str = "ensemble_stacking.joblib") -> Tuple[Path, Path]:
        """Saves ensemble artifacts."""
        p1 = save_model(self.voting_model, voting_filename)
        p2 = save_model(self.stacking_model, stacking_filename)
        return p1, p2

    @classmethod
    def load(cls, voting_filename: str = "ensemble_voting.joblib", stacking_filename: str = "ensemble_stacking.joblib") -> "DrowsinessEnsemble":
        """Loads pre-trained ensemble artifacts."""
        instance = cls()
        instance.voting_model = load_model(voting_filename)
        instance.stacking_model = load_model(stacking_filename)
        return instance
