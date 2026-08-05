"""
Bayesian Logistic Regression Module (Unit 2: Bayesian Logistic Regression).
Performs probabilistic multi-class driver state classification with Gaussian weight priors,
MAP estimation, posterior probabilities, and Shannon entropy for uncertainty quantification.
"""

from pathlib import Path
from typing import Dict, Any, Optional, Tuple, List

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, log_loss, f1_score

from src import config
from src.utils import setup_logger, save_model, load_model

logger = setup_logger("BayesianLogistic")


class BayesianLogisticClassifier:
    """
    Unit 2 Bayesian Logistic Classifier.
    Implements multi-class classification with Gaussian prior over weights (MAP inference),
    posterior class probability distributions, and predictive entropy.
    """

    def __init__(self, prior_variance: float = 1.0, max_iter: int = 1000):
        self.prior_variance = prior_variance
        # In Bayesian logistic regression, a zero-mean Gaussian prior N(0, sigma^2 I)
        # corresponds to L2 regularization with C = sigma^2
        self.C = float(prior_variance)
        self.max_iter = max_iter

        self.model = LogisticRegression(
            C=self.C,
            solver="lbfgs",
            max_iter=max_iter,
            random_state=config.RANDOM_STATE,
        )
        self.classes_: np.ndarray = np.array([])
        self.metrics: Dict[str, Any] = {}

    def fit(self, X_train: np.ndarray, y_train: np.ndarray) -> "BayesianLogisticClassifier":
        """Fits the MAP logistic regression model on training data."""
        logger.info(f"Fitting Bayesian Logistic Regression (prior_variance={self.prior_variance}) on {X_train.shape[0]} samples...")
        self.model.fit(X_train, y_train)
        self.classes_ = self.model.classes_
        logger.info(f"Bayesian Logistic Regression fitted. Classes: {list(self.classes_)}")
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predicts the Maximum A Posteriori (MAP) class label."""
        return self.model.predict(X)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Computes posterior class probabilities P(y = k | x)."""
        return self.model.predict_proba(X)

    def predict_with_uncertainty(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Computes MAP class prediction, posterior probabilities, and Shannon Entropy (uncertainty).
        Entropy H(x) = -sum(p_k * log(p_k + 1e-12))
        """
        probs = self.predict_proba(X)
        preds = self.predict(X)

        # Shannon Entropy across 4 classes (max = log(4) = 1.386 nats)
        eps = 1e-12
        entropy = -np.sum(probs * np.log(probs + eps), axis=1)

        return preds, probs, entropy

    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, Any]:
        """Evaluates classification accuracy, macro F1, and multi-class log loss."""
        preds, probs, entropy = self.predict_with_uncertainty(X_test)

        acc = float(accuracy_score(y_test, preds))
        f1_macro = float(f1_score(y_test, preds, average="macro"))
        loss = float(log_loss(y_test, probs))
        avg_entropy = float(np.mean(entropy))

        self.metrics = {
            "accuracy": acc,
            "f1_macro": f1_macro,
            "log_loss": loss,
            "mean_uncertainty_entropy": avg_entropy,
            "classification_report": classification_report(y_test, preds, target_names=config.CLASS_LABELS, output_dict=True),
        }
        logger.info(f"Bayesian Logistic Results: Accuracy={acc:.4f}, Macro-F1={f1_macro:.4f}, LogLoss={loss:.4f}, MeanEntropy={avg_entropy:.4f}")
        return self.metrics

    def plot_posterior_distribution(self, sample_probabilities: np.ndarray, sample_title: str = "Driver State Posterior", output_dir: Optional[Path] = None) -> Path:
        """Plots posterior probability bar chart for a single observation."""
        out_dir = output_dir or config.EVAL_OUTPUT_DIR
        out_dir.mkdir(parents=True, exist_ok=True)

        fig, ax = plt.subplots(figsize=(8, 4.5))
        colors = ["#2ecc71", "#f39c12", "#e67e22", "#e74c3c"]

        bars = ax.bar(config.CLASS_LABELS, sample_probabilities, color=colors, edgecolor="black", alpha=0.85, width=0.55)
        ax.set_ylim(0.0, 1.05)
        ax.set_ylabel("Posterior Probability P(State | Features)", fontsize=11)
        ax.set_title(f"Bayesian Posterior Distribution: {sample_title}", fontsize=13, fontweight="bold")
        ax.grid(axis="y", linestyle="--", alpha=0.6)

        for bar in bars:
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2.0, h + 0.02, f"{h * 100:.1f}%", ha="center", va="bottom", fontsize=10, fontweight="bold")

        plt.tight_layout()
        save_path = out_dir / "bayesian_posterior_sample.png"
        plt.savefig(save_path, dpi=300)
        plt.close()
        logger.info(f"Saved Bayesian posterior distribution plot to {save_path}")
        return save_path

    def save(self, filename: str = "bayesian_logistic.joblib") -> Path:
        """Saves the trained model artifact."""
        return save_model(self, filename)

    @classmethod
    def load(cls, filename: str = "bayesian_logistic.joblib") -> "BayesianLogisticClassifier":
        """Loads a pre-trained model artifact."""
        return load_model(filename)
