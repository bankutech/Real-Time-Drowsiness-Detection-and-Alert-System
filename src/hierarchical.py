"""
Hierarchical Clustering Module (Unit 3: Agglomerative Hierarchical Clustering & Dendrograms).
Computes linkage matrices (Ward, Complete, Average), Cophenetic correlation coefficients,
generates hierarchical dendrogram visualizations, and evaluates cluster partitions.
"""

from pathlib import Path
from typing import Dict, Any, Optional, Tuple, List

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage, cophenet
from scipy.spatial.distance import pdist
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score, adjusted_rand_score, normalized_mutual_info_score

from src import config
from src.utils import setup_logger, save_model, load_model

logger = setup_logger("HierarchicalClustering")


class DrowsinessHierarchicalClustering:
    """
    Unit 3 Hierarchical Agglomerative Clustering Engine.
    Builds bottom-up tree hierarchies and analyzes dendrogram cutoffs.
    """

    def __init__(self, n_clusters: int = config.NUM_CLASSES, linkage_method: str = "ward"):
        self.n_clusters = n_clusters
        self.linkage_method = linkage_method.lower()

        self.model = AgglomerativeClustering(
            n_clusters=n_clusters,
            metric="euclidean" if self.linkage_method == "ward" else "euclidean",
            linkage=self.linkage_method,
        )
        self.linkage_matrix: Optional[np.ndarray] = None
        self.cophenetic_corr: float = 0.0
        self.metrics: Dict[str, Any] = {}

    def fit_predict(self, X: np.ndarray) -> np.ndarray:
        """Fits hierarchical clustering and returns cluster labels."""
        logger.info(f"Fitting Agglomerative Clustering (k={self.n_clusters}, linkage={self.linkage_method}) on {X.shape[0]} samples...")
        labels = self.model.fit_predict(X)

        # Compute scipy linkage matrix for dendrogram and cophenetic correlation
        # Use a representative subsample if dataset is very large to ensure fast linkage computation
        sample_size = min(len(X), 1500)
        indices = np.random.RandomState(config.RANDOM_STATE).choice(len(X), size=sample_size, replace=False)
        X_sub = X[indices]

        self.linkage_matrix = linkage(X_sub, method=self.linkage_method)
        c, _ = cophenet(self.linkage_matrix, pdist(X_sub))
        self.cophenetic_corr = float(c)

        logger.info(f"Hierarchical Clustering fitted. Cophenetic Correlation: {self.cophenetic_corr:.4f}")
        return labels

    def evaluate(self, X: np.ndarray, y_ground_truth: Optional[np.ndarray] = None) -> Dict[str, Any]:
        """Calculates Silhouette score, Cophenetic correlation, and ARI/NMI."""
        labels = self.model.labels_
        sil = float(silhouette_score(X, labels))

        self.metrics = {
            "n_clusters": self.n_clusters,
            "linkage_method": self.linkage_method,
            "cophenetic_correlation": self.cophenetic_corr,
            "silhouette_score": sil,
        }

        if y_ground_truth is not None:
            ari = float(adjusted_rand_score(y_ground_truth, labels))
            nmi = float(normalized_mutual_info_score(y_ground_truth, labels))
            self.metrics["adjusted_rand_index"] = ari
            self.metrics["normalized_mutual_info"] = nmi
            logger.info(f"Hierarchical Evaluation: Silhouette={sil:.4f}, Cophenetic={self.cophenetic_corr:.4f}, ARI={ari:.4f}")
        else:
            logger.info(f"Hierarchical Evaluation: Silhouette={sil:.4f}, Cophenetic={self.cophenetic_corr:.4f}")

        return self.metrics

    def plot_dendrogram(self, *args, output_dir: Optional[Path] = None, max_d: Optional[float] = None, **kwargs) -> Path:
        """Generates Hierarchical Tree Dendrogram visualization."""
        if self.linkage_matrix is None:
            raise ValueError("Model must be fitted before plotting dendrogram.")

        out_dir = output_dir or (args[0] if len(args) > 0 and isinstance(args[0], (str, Path)) else config.CLUSTERING_OUTPUT_DIR)
        out_dir.mkdir(parents=True, exist_ok=True)

        fig, ax = plt.subplots(figsize=(12, 6))

        dendrogram(
            self.linkage_matrix,
            truncate_mode="lastp",
            p=30,
            leaf_rotation=45.0,
            leaf_font_size=10.0,
            show_contracted=True,
            ax=ax,
            color_threshold=max_d,
        )

        if max_d is not None:
            ax.axhline(y=max_d, c="red", linestyle="--", linewidth=1.5, label=f"Threshold cut = {max_d:.1f}")
            ax.legend(loc="upper right")

        ax.set_title(
            f"Hierarchical Agglomerative Dendrogram ({self.linkage_method.capitalize()} Linkage)\nCophenetic Correlation: {self.cophenetic_corr:.4f}",
            fontsize=13,
            fontweight="bold",
        )
        ax.set_xlabel("Cluster / Subtree (Sample Size)", fontsize=11)
        ax.set_ylabel("Distance (Dissimilarity)", fontsize=11)
        ax.grid(axis="y", linestyle="--", alpha=0.5)

        plt.tight_layout()
        save_path = out_dir / "hierarchical_dendrogram.png"
        plt.savefig(save_path, dpi=300)
        plt.close()
        logger.info(f"Saved Hierarchical Dendrogram to {save_path}")
        return save_path

    def plot_clusters_2d(self, X: np.ndarray, X_2d: np.ndarray, output_dir: Optional[Path] = None) -> Path:
        """Plots 2D cluster assignments in PCA projection space."""
        out_dir = output_dir or config.CLUSTERING_OUTPUT_DIR
        out_dir.mkdir(parents=True, exist_ok=True)

        labels = self.model.labels_

        fig, ax = plt.subplots(figsize=(9, 6.5))
        cmap = plt.get_cmap("viridis")

        scatter = ax.scatter(
            X_2d[:, 0],
            X_2d[:, 1],
            c=labels,
            cmap=cmap,
            s=35,
            alpha=0.75,
            edgecolor="k",
            linewidth=0.4,
        )

        ax.set_title(f"Hierarchical Agglomerative Clusters (K={self.n_clusters}) in 2D PCA Space", fontsize=13, fontweight="bold")
        ax.set_xlabel("Principal Component 1", fontsize=11)
        ax.set_ylabel("Principal Component 2", fontsize=11)
        plt.colorbar(scatter, ax=ax, label="Cluster ID")
        ax.grid(True, linestyle="--", alpha=0.5)

        plt.tight_layout()
        save_path = out_dir / "hierarchical_clusters_2d.png"
        plt.savefig(save_path, dpi=300)
        plt.close()
        logger.info(f"Saved Hierarchical 2D cluster plot to {save_path}")
        return save_path

    def save(self, filename: str = "hierarchical.joblib") -> Path:
        """Saves the model artifact."""
        return save_model(self, filename)

    @classmethod
    def load(cls, filename: str = "hierarchical.joblib") -> "DrowsinessHierarchicalClustering":
        """Loads a pre-trained model artifact."""
        return load_model(filename)
