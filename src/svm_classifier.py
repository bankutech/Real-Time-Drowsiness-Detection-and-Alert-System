"""
Support Vector Machine Module (Unit 2: Support Vector Machines).
Implements Linear and RBF Kernel SVM classifiers for multi-class driver drowsiness detection.
Includes hyperplane margin extraction, support vector counting, and 2D decision boundary visualization.
"""

from pathlib import Path
from typing import Dict, Any, Optional, Tuple, List

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, classification_report, f1_score

from src import config
from src.utils import setup_logger, save_model, load_model

logger = setup_logger("SVMClassifier")


class DrowsinessSVMClassifier:
    """
    Unit 2 Support Vector Machine Classifier.
    Supports Linear and RBF kernels with probability calibration and decision boundary analysis.
    """

    def __init__(self, kernel: str = "rbf", C: float = 1.0, gamma: str = "scale"):
        self.kernel = kernel.lower()
        self.C = C
        self.gamma = gamma

        _base_svc = SVC(
            C=self.C,
            kernel=self.kernel,
            gamma=self.gamma,
            random_state=config.RANDOM_STATE,
        )
        self.model = CalibratedClassifierCV(_base_svc, ensemble=False)
        self.classes_: np.ndarray = np.array([])
        self.metrics: Dict[str, Any] = {}

    def fit(self, X_train: np.ndarray, y_train: np.ndarray) -> "DrowsinessSVMClassifier":
        """Fits the SVM model on training features."""
        logger.info(f"Fitting SVM ({self.kernel.upper()} kernel, C={self.C}) on {X_train.shape[0]} samples...")
        self.model.fit(X_train, y_train)
        self.classes_ = self.model.classes_
        # Access support vector stats from the underlying fitted SVC estimator
        base_estimator = self.model.calibrated_classifiers_[0].estimator
        n_sv = int(sum(base_estimator.n_support_))
        sv_per_class = [int(x) for x in base_estimator.n_support_]
        self._n_support_vectors = n_sv
        self._support_per_class = sv_per_class
        logger.info(f"SVM fitted. Total Support Vectors: {n_sv} ({sv_per_class} per class)")
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predicts class labels for input samples."""
        return self.model.predict(X)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Calculates calibrated class probabilities."""
        return self.model.predict_proba(X)

    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, Any]:
        """Evaluates classification accuracy and macro F1."""
        preds = self.predict(X_test)
        probs = self.predict_proba(X_test)

        acc = float(accuracy_score(y_test, preds))
        f1_macro = float(f1_score(y_test, preds, average="macro"))

        self.metrics = {
            "kernel": self.kernel,
            "C": self.C,
            "accuracy": acc,
            "f1_macro": f1_macro,
            "n_support_vectors": getattr(self, "_n_support_vectors", 0),
            "support_vectors_per_class": getattr(self, "_support_per_class", []),
            "classification_report": classification_report(y_test, preds, target_names=config.CLASS_LABELS, output_dict=True),
        }
        logger.info(f"SVM ({self.kernel.upper()}) Test Accuracy: {acc:.4f}, Macro-F1: {f1_macro:.4f}")
        return self.metrics

    def plot_decision_boundary_2d(self, X: np.ndarray, y: np.ndarray, output_dir: Optional[Path] = None) -> Path:
        """
        Projects features to 2D using PCA and trains a 2D visualization SVM
        to plot the separating decision boundary, margin lines, and support vectors.
        """
        out_dir = output_dir or config.EVAL_OUTPUT_DIR
        out_dir.mkdir(parents=True, exist_ok=True)

        pca = PCA(n_components=2, random_state=config.RANDOM_STATE)
        X_2d = pca.fit_transform(X)

        # Train a 2D SVM specifically for decision boundary plotting
        svm_2d = SVC(kernel=self.kernel, C=self.C, gamma=self.gamma, random_state=config.RANDOM_STATE)
        svm_2d.fit(X_2d, y)

        x_min, x_max = X_2d[:, 0].min() - 1.0, X_2d[:, 0].max() + 1.0
        y_min, y_max = X_2d[:, 1].min() - 1.0, X_2d[:, 1].max() + 1.0
        xx, yy = np.meshgrid(np.linspace(x_min, x_max, 300), np.linspace(y_min, y_max, 300))

        Z = svm_2d.predict(np.c_[xx.ravel(), yy.ravel()])
        Z = Z.reshape(xx.shape)

        fig, ax = plt.subplots(figsize=(10, 7))
        # Contour filled decision regions
        cmap_bg = plt.get_cmap("coolwarm")
        ax.contourf(xx, yy, Z, alpha=0.3, cmap=cmap_bg)

        # Plot data points
        colors = ["#2ecc71", "#f39c12", "#e67e22", "#e74c3c"]
        for idx, label_name in enumerate(config.CLASS_LABELS):
            mask = y == idx
            ax.scatter(
                X_2d[mask, 0],
                X_2d[mask, 1],
                c=colors[idx],
                label=f"Class {idx}: {label_name}",
                edgecolor="k",
                s=40,
                alpha=0.8,
            )

        # Highlight Support Vectors
        sv = svm_2d.support_vectors_
        ax.scatter(
            sv[:, 0],
            sv[:, 1],
            s=90,
            facecolors="none",
            edgecolors="black",
            linewidth=1.2,
            label=f"Support Vectors (N={len(sv)})",
        )

        ax.set_title(
            f"SVM Decision Boundary & Hyperplane Margin ({self.kernel.upper()} Kernel)\nPCA Explained Variance: {pca.explained_variance_ratio_.sum()*100:.1f}%",
            fontsize=13,
            fontweight="bold",
        )
        ax.set_xlabel("Principal Component 1 (EAR & Eye Dynamics)", fontsize=11)
        ax.set_ylabel("Principal Component 2 (MAR & Head Pose)", fontsize=11)
        ax.legend(loc="upper right", framealpha=0.9)
        ax.grid(True, linestyle=":", alpha=0.6)

        plt.tight_layout()
        save_path = out_dir / f"svm_decision_boundary_{self.kernel}.png"
        plt.savefig(save_path, dpi=300)
        plt.close()
        logger.info(f"Saved SVM decision boundary plot to {save_path}")
        return save_path

    def save(self, filename: Optional[str] = None) -> Path:
        """Saves the trained model artifact."""
        name = filename or f"svm_{self.kernel}.joblib"
        return save_model(self, name)

    @classmethod
    def load(cls, filename: str) -> "DrowsinessSVMClassifier":
        """Loads a pre-trained model artifact."""
        return load_model(filename)
