"""
Gaussian Mixture Model Module (Unit 3: GMM & Expectation-Maximization).
Performs soft probabilistic clustering with EM algorithm, covariance structure analysis,
AIC/BIC model selection, and 2D Gaussian confidence ellipses.
"""

from pathlib import Path
from typing import Dict, Any, Optional, Tuple, List

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score, adjusted_rand_score, normalized_mutual_info_score

from src import config
from src.utils import setup_logger, save_model, load_model

logger = setup_logger("GMM")


class DrowsinessGMM:
    """
    Unit 3 Gaussian Mixture Model Clustering Engine.
    Uses Expectation-Maximization (EM) to fit Gaussian components with soft cluster responsibilities.
    """

    def __init__(self, n_components: int = config.NUM_CLASSES, covariance_type: str = "full"):
        self.n_components = n_components
        self.covariance_type = covariance_type
        self.gmm = GaussianMixture(
            n_components=n_components,
            covariance_type=covariance_type,
            max_iter=200,
            random_state=config.RANDOM_STATE,
        )
        self.metrics: Dict[str, Any] = {}

    def fit(self, X: np.ndarray) -> "DrowsinessGMM":
        """Fits GMM model using the EM algorithm."""
        logger.info(f"Fitting GMM (k={self.n_components}, covariance={self.covariance_type}) on {X.shape[0]} samples...")
        self.gmm.fit(X)
        logger.info(f"GMM converged: {self.gmm.converged_} in {self.gmm.n_iter_} iterations. Lower bound log-likelihood: {self.gmm.lower_bound_:.4f}")
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Assigns samples to the component with highest posterior responsibility."""
        return self.gmm.predict(X)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Computes posterior probabilities (responsibilities gamma_ik) for all components."""
        return self.gmm.predict_proba(X)

    def evaluate(self, X: np.ndarray, y_ground_truth: Optional[np.ndarray] = None) -> Dict[str, Any]:
        """Calculates AIC, BIC, Log-Likelihood, Silhouette, and ARI/NMI."""
        labels = self.predict(X)
        aic_val = float(self.gmm.aic(X))
        bic_val = float(self.gmm.bic(X))
        log_lik = float(self.gmm.score(X))
        sil = float(silhouette_score(X, labels))

        self.metrics = {
            "n_components": self.n_components,
            "covariance_type": self.covariance_type,
            "aic": aic_val,
            "bic": bic_val,
            "log_likelihood_per_sample": log_lik,
            "silhouette_score": sil,
        }

        if y_ground_truth is not None:
            ari = float(adjusted_rand_score(y_ground_truth, labels))
            nmi = float(normalized_mutual_info_score(y_ground_truth, labels))
            self.metrics["adjusted_rand_index"] = ari
            self.metrics["normalized_mutual_info"] = nmi
            logger.info(f"GMM Evaluation: BIC={bic_val:.1f}, AIC={aic_val:.1f}, Silhouette={sil:.4f}, ARI={ari:.4f}")
        else:
            logger.info(f"GMM Evaluation: BIC={bic_val:.1f}, AIC={aic_val:.1f}, Silhouette={sil:.4f}")

        return self.metrics

    def run_model_selection_aic_bic(
        self,
        X: np.ndarray,
        k_range: range = range(2, 8),
        output_dir: Optional[Path] = None,
    ) -> Tuple[Path, Dict[int, float], Dict[int, float]]:
        """Computes and plots AIC & BIC curves across different numbers of Gaussian components."""
        out_dir = output_dir or config.CLUSTERING_OUTPUT_DIR
        out_dir.mkdir(parents=True, exist_ok=True)

        aics = {}
        bics = {}

        for k in k_range:
            gm = GaussianMixture(n_components=k, covariance_type=self.covariance_type, random_state=config.RANDOM_STATE)
            gm.fit(X)
            aics[k] = float(gm.aic(X))
            bics[k] = float(gm.bic(X))

        fig, ax = plt.subplots(figsize=(9, 5))
        k_vals = list(k_range)
        ax.plot(k_vals, [aics[k] for k in k_vals], "bo-", linewidth=2.0, label="AIC (Akaike Information Criterion)")
        ax.plot(k_vals, [bics[k] for k in k_vals], "rs-", linewidth=2.0, label="BIC (Bayesian Information Criterion)")
        ax.axvline(self.n_components, color="green", linestyle=":", label=f"Selected K={self.n_components}")

        ax.set_title(f"GMM Model Selection: AIC & BIC Curves ({self.covariance_type.upper()} Covariance)", fontsize=13, fontweight="bold")
        ax.set_xlabel("Number of Gaussian Components (K)", fontsize=11)
        ax.set_ylabel("Information Criterion Score (Lower is Better)", fontsize=11)
        ax.legend(framealpha=0.9)
        ax.grid(True, linestyle="--", alpha=0.5)

        plt.tight_layout()
        save_path = out_dir / "gmm_aic_bic.png"
        plt.savefig(save_path, dpi=300)
        plt.close()
        logger.info(f"Saved GMM AIC/BIC plot to {save_path}")
        return save_path, aics, bics

    def plot_clusters_with_ellipses_2d(self, X_2d: np.ndarray, output_dir: Optional[Path] = None) -> Path:
        """
        Fits a 2D GMM on projected PCA features and plots 2-standard-deviation confidence ellipses.
        """
        out_dir = output_dir or config.CLUSTERING_OUTPUT_DIR
        out_dir.mkdir(parents=True, exist_ok=True)

        gmm_2d = GaussianMixture(n_components=self.n_components, covariance_type="full", random_state=config.RANDOM_STATE)
        labels_2d = gmm_2d.fit_predict(X_2d)

        fig, ax = plt.subplots(figsize=(9, 6.5))
        colors = ["#2ecc71", "#f39c12", "#e67e22", "#e74c3c"]

        for k in range(self.n_components):
            pts = X_2d[labels_2d == k]
            color = colors[k % len(colors)]
            ax.scatter(pts[:, 0], pts[:, 1], c=color, s=35, alpha=0.7, edgecolor="k", linewidth=0.3, label=f"Component {k}")

            # Covariance ellipse
            mean = gmm_2d.means_[k]
            cov = gmm_2d.covariances_[k]
            v, w = np.linalg.eigh(cov)
            u = w[0] / np.linalg.norm(w[0])
            angle = np.arctan2(u[1], u[0]) * 180.0 / np.pi

            # 2 standard deviations ellipse (95% confidence region)
            v = 2.0 * np.sqrt(2.0) * np.sqrt(np.maximum(v, 1e-6))
            ell = Ellipse(xy=mean, width=v[0], height=v[1], angle=angle, edgecolor=color, facecolor=color, alpha=0.2, linewidth=2.0)
            ax.add_patch(ell)
            ax.scatter(mean[0], mean[1], marker="x", s=150, c="black", linewidth=2.0)

        ax.set_title(f"GMM 2D Clustering with Gaussian Confidence Ellipses (K={self.n_components})", fontsize=13, fontweight="bold")
        ax.set_xlabel("Principal Component 1", fontsize=11)
        ax.set_ylabel("Principal Component 2", fontsize=11)
        ax.legend(loc="upper right", framealpha=0.9)
        ax.grid(True, linestyle="--", alpha=0.5)

        plt.tight_layout()
        save_path = out_dir / "gmm_clusters_ellipses.png"
        plt.savefig(save_path, dpi=300)
        plt.close()
        logger.info(f"Saved GMM confidence ellipses plot to {save_path}")
        return save_path

    def save(self, filename: str = "gmm.joblib") -> Path:
        """Saves the trained model artifact."""
        return save_model(self, filename)

    @classmethod
    def load(cls, filename: str = "gmm.joblib") -> "DrowsinessGMM":
        """Loads a pre-trained model artifact."""
        return load_model(filename)
