"""
Principal Component Analysis Module (Unit 3: PCA & Dimensionality Reduction).
Performs PCA feature transformation, Scree plot analysis, cumulative variance quantification,
and 2D/3D projection visualizations with feature loading vectors.
"""

from pathlib import Path
from typing import Dict, Any, Optional, Tuple, List

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

from src import config
from src.utils import setup_logger, save_model, load_model

logger = setup_logger("PCA")


class PCAReducer:
    """
    Unit 3 Principal Component Analysis Engine.
    Reduces feature dimensionality while preserving maximum physiological variance.
    """

    def __init__(self, n_components: Optional[int] = None, variance_threshold: float = 0.95):
        self.n_components = n_components
        self.variance_threshold = variance_threshold

        self.pca = PCA(n_components=n_components, random_state=config.RANDOM_STATE)
        self.feature_names = config.FEATURE_COLUMNS
        self.explained_variance_ratio: np.ndarray = np.array([])
        self.cumulative_variance_ratio: np.ndarray = np.array([])

    def fit(self, X: np.ndarray) -> "PCAReducer":
        """Fits PCA model to feature matrix."""
        logger.info(f"Fitting PCA on {X.shape[0]} samples with {X.shape[1]} features...")
        self.pca.fit(X)
        self.explained_variance_ratio = self.pca.explained_variance_ratio_
        self.cumulative_variance_ratio = np.cumsum(self.explained_variance_ratio)

        # Determine number of components needed for variance threshold
        n_needed = int(np.argmax(self.cumulative_variance_ratio >= self.variance_threshold) + 1)
        logger.info(f"PCA fitted. Total components: {len(self.explained_variance_ratio)}. Components for {self.variance_threshold*100:.0f}% variance: {n_needed}")
        return self

    def transform(self, X: np.ndarray, n_dims: Optional[int] = None) -> np.ndarray:
        """Projects feature matrix onto top principal components."""
        X_trans = self.pca.transform(X)
        if n_dims is not None:
            return X_trans[:, :n_dims]
        return X_trans

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        """Fits and transforms feature matrix."""
        self.fit(X)
        return self.transform(X)

    def get_loadings(self) -> pd.DataFrame:
        """Extracts component loading matrix (eigenvectors * sqrt(eigenvalues))."""
        loadings = self.pca.components_.T * np.sqrt(self.pca.explained_variance_)
        cols = [f"PC{i+1}" for i in range(loadings.shape[1])]
        df_loadings = pd.DataFrame(loadings, index=self.feature_names, columns=cols)
        return df_loadings

    def plot_scree(self, output_dir: Optional[Path] = None) -> Path:
        """Generates Scree Plot with individual and cumulative explained variance ratios."""
        out_dir = output_dir or config.CLUSTERING_OUTPUT_DIR
        out_dir.mkdir(parents=True, exist_ok=True)

        n_comp = len(self.explained_variance_ratio)
        x_axis = np.arange(1, n_comp + 1)

        fig, ax1 = plt.subplots(figsize=(9, 5))

        # Individual variance bars
        ax1.bar(x_axis, self.explained_variance_ratio * 100.0, color="#3498db", alpha=0.75, width=0.55, label="Individual Variance (%)")
        ax1.set_xlabel("Principal Component Index", fontsize=11, fontweight="bold")
        ax1.set_ylabel("Explained Variance (%)", color="#2980b9", fontsize=11, fontweight="bold")
        ax1.set_xticks(x_axis)
        ax1.set_ylim(0, max(self.explained_variance_ratio * 100.0) + 10)
        ax1.grid(axis="y", linestyle="--", alpha=0.5)

        # Cumulative variance line
        ax2 = ax1.twinx()
        ax2.plot(x_axis, self.cumulative_variance_ratio * 100.0, color="#e74c3c", marker="o", linewidth=2.2, label="Cumulative Variance (%)")
        ax2.axhline(self.variance_threshold * 100.0, color="#27ae60", linestyle=":", linewidth=1.8, label=f"{self.variance_threshold*100:.0f}% Threshold")
        ax2.set_ylabel("Cumulative Explained Variance (%)", color="#c0392b", fontsize=11, fontweight="bold")
        ax2.set_ylim(0, 105)

        plt.title("PCA Scree Plot & Cumulative Explained Variance (Unit 3)", fontsize=13, fontweight="bold")
        fig.tight_layout()

        save_path = out_dir / "pca_scree_plot.png"
        plt.savefig(save_path, dpi=300)
        plt.close()
        logger.info(f"Saved PCA Scree plot to {save_path}")
        return save_path

    def plot_scree_plot(self, output_dir: Optional[Path] = None) -> Path:
        """Alias for plot_scree."""
        return self.plot_scree(output_dir=output_dir)

    def plot_2d_projection(self, X: np.ndarray, y: np.ndarray, title: str = "PCA 2D Projection", output_dir: Optional[Path] = None) -> Path:
        """Plots 2D projection colored by driver states."""
        out_dir = output_dir or config.CLUSTERING_OUTPUT_DIR
        out_dir.mkdir(parents=True, exist_ok=True)

        X_2d = self.transform(X, n_dims=2)

        fig, ax = plt.subplots(figsize=(9, 6.5))
        colors = ["#2ecc71", "#f39c12", "#e67e22", "#e74c3c"]

        for idx, label_name in enumerate(config.CLASS_LABELS):
            mask = y == idx
            ax.scatter(
                X_2d[mask, 0],
                X_2d[mask, 1],
                c=colors[idx],
                label=f"Class {idx}: {label_name}",
                alpha=0.75,
                edgecolor="k",
                linewidth=0.5,
                s=35,
            )

        var_pc1 = self.explained_variance_ratio[0] * 100.0 if len(self.explained_variance_ratio) > 0 else 0
        var_pc2 = self.explained_variance_ratio[1] * 100.0 if len(self.explained_variance_ratio) > 1 else 0

        ax.set_xlabel(f"PC1 ({var_pc1:.1f}% Variance)", fontsize=11, fontweight="bold")
        ax.set_ylabel(f"PC2 ({var_pc2:.1f}% Variance)", fontsize=11, fontweight="bold")
        ax.set_title(f"{title} (Total Variance: {var_pc1 + var_pc2:.1f}%)", fontsize=13, fontweight="bold")
        ax.legend(loc="upper right", framealpha=0.9)
        ax.grid(True, linestyle="--", alpha=0.5)

        plt.tight_layout()
        save_path = out_dir / "pca_2d_projection.png"
        plt.savefig(save_path, dpi=300)
        plt.close()
        logger.info(f"Saved PCA 2D projection plot to {save_path}")
        return save_path

    def save(self, filename: str = "pca.joblib") -> Path:
        """Saves the trained PCA artifact."""
        return save_model(self, filename)

    @classmethod
    def load(cls, filename: str = "pca.joblib") -> "PCAReducer":
        """Loads a pre-trained PCA artifact."""
        return load_model(filename)
