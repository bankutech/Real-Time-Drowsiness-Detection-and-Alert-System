"""
AdaBoost Classifier Module (Unit 5: Adaptive Boosting & Sequential Ensembles).
Implements AdaBoost with Decision Stump weak learners, sequential sample re-weighting,
and stagewise training vs test error convergence tracking.
"""

from pathlib import Path
from typing import Dict, Any, Optional, Tuple, List

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, f1_score, zero_one_loss, classification_report

from src import config
from src.utils import setup_logger, save_model, load_model

logger = setup_logger("AdaBoost")


class DrowsinessAdaBoost:
    """
    Unit 5 AdaBoost Classifier Engine.
    Sequentially builds an ensemble of weak learners (decision stumps)
    by adaptively focusing on previously misclassified driver frames.
    """

    def __init__(
        self,
        n_estimators: int = 50,
        learning_rate: float = 1.0,
        base_depth: int = 1,
    ):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.base_depth = base_depth

        base_estimator = DecisionTreeClassifier(max_depth=base_depth, random_state=config.RANDOM_STATE)
        self.model = AdaBoostClassifier(
            estimator=base_estimator,
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            random_state=config.RANDOM_STATE,
        )
        self.feature_names = config.FEATURE_COLUMNS
        self.class_names = config.CLASS_LABELS
        self.metrics: Dict[str, Any] = {}

    def fit(self, X_train: np.ndarray, y_train: np.ndarray) -> "DrowsinessAdaBoost":
        """Fits AdaBoost ensemble on training data."""
        logger.info(f"Fitting AdaBoost (n_estimators={self.n_estimators}, lr={self.learning_rate}, stump_depth={self.base_depth}) on {X_train.shape[0]} samples...")
        self.model.fit(X_train, y_train)
        logger.info(f"AdaBoost fitted with {len(self.model.estimators_)} weak estimators.")
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predicts class labels."""
        return self.model.predict(X)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predicts class probability distributions."""
        return self.model.predict_proba(X)

    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, Any]:
        """Evaluates classification accuracy and macro F1."""
        y_pred = self.predict(X_test)
        acc = float(accuracy_score(y_test, y_pred))
        macro_f1 = float(f1_score(y_test, y_pred, average="macro"))

        self.metrics = {
            "accuracy": acc,
            "macro_f1": macro_f1,
            "n_estimators": self.n_estimators,
            "learning_rate": self.learning_rate,
            "classification_report": classification_report(y_test, y_pred, target_names=self.class_names, output_dict=True),
        }
        logger.info(f"AdaBoost Evaluation: Accuracy={acc:.4f}, Macro-F1={macro_f1:.4f}")
        return self.metrics

    def plot_stagewise_error(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_test: np.ndarray,
        y_test: np.ndarray,
        output_dir: Optional[Path] = None,
    ) -> Path:
        """Visualizes sequential error reduction as boosting iterations progress."""
        out_dir = output_dir or config.EVAL_OUTPUT_DIR
        out_dir.mkdir(parents=True, exist_ok=True)

        train_errors = []
        test_errors = []

        for y_train_pred in self.model.staged_predict(X_train):
            train_errors.append(zero_one_loss(y_train, y_train_pred))

        for y_test_pred in self.model.staged_predict(X_test):
            test_errors.append(zero_one_loss(y_test, y_test_pred))

        fig, ax = plt.subplots(figsize=(9, 5))
        iters = np.arange(1, len(train_errors) + 1)
        ax.plot(iters, train_errors, "b-", linewidth=2.0, label="Training Error")
        ax.plot(iters, test_errors, "r--", linewidth=2.0, label="Test Error")

        ax.set_title(f"AdaBoost Stagewise Learning Curve (Estimators: {self.n_estimators}, LR: {self.learning_rate})", fontsize=13, fontweight="bold")
        ax.set_xlabel("Boosting Iteration (m)", fontsize=11)
        ax.set_ylabel("Zero-One Error Rate", fontsize=11)
        ax.legend(framealpha=0.9)
        ax.grid(True, linestyle="--", alpha=0.5)

        plt.tight_layout()
        save_path = out_dir / "adaboost_stagewise_error.png"
        plt.savefig(save_path, dpi=300)
        plt.close()
        logger.info(f"Saved AdaBoost stagewise error plot to {save_path}")
        return save_path

    def save(self, filename: str = "adaboost.joblib") -> Path:
        """Saves model artifact."""
        return save_model(self, filename)

    @classmethod
    def load(cls, filename: str = "adaboost.joblib") -> "DrowsinessAdaBoost":
        """Loads pre-trained model artifact."""
        return load_model(filename)
