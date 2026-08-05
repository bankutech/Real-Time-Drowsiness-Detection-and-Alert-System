"""
K-Means Clustering Module (Unit 3: Unsupervised Learning & Clustering).
Performs K-Means clustering with Elbow Method inertia analysis, Silhouette coefficient evaluation,
cluster centroid characterization, and Adjusted Rand Index benchmarking.
"""

from pathlib import Path
from typing import Dict, Any, Optional, Tuple, List

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, silhouette_samples, adjusted_rand_score, normalized_mutual_info_score

from src import config
from src.utils import setup_logger, save_model, load_model

logger = setup_logger("KMeansClustering")


class DrowsinessKMeans:
    """
    Unit 3 K-Means Clustering Engine.
    Discovers natural driver fatigue clusters without ground-truth supervision.
    """

    def __init__(self, n_clusters: int = config.NUM_CLASSES):
        self.n_clusters = n_clusters
        self.kmeans = KMeans(
            n_clusters=n_clusters,
            init="k-means++",
            n_init=10,
            max_iter=300,
            random_state=config.RANDOM_STATE,
        )
        self.feature_names = config.FEATURE_COLUMNS
        self.cluster_centers: np.ndarray = np.array([])
        self.metrics: Dict[str, Any] = {}

    def fit(self, X: np.ndarray) -> "DrowsinessKMeans":
        """Fits K-Means clustering on feature matrix."""
        logger.info(f"Fitting K-Means (k={self.n_clusters}) on {X.shape[0]} samples...")
        self.kmeans.fit(X)
        self.cluster_centers = self.kmeans.cluster_centers_
        logger.info("K-Means fitting complete.")
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Assigns samples to the nearest cluster centroid."""
        return self.kmeans.predict(X)

    def evaluate_clustering(self, X: np.ndarray, y_ground_truth: Optional[np.ndarray] = None) -> Dict[str, Any]:
        """Calculates Silhouette Score, Inertia, and supervised alignment metrics if labels provided."""
        labels = self.predict(X)
        sil_score = float(silhouette_score(X, labels))
        inertia = float(self.kmeans.inertia_)

        self.metrics = {
            "n_clusters": self.n_clusters,
            "inertia": inertia,
            "silhouette_score": sil_score,
        }

        if y_ground_truth is not None:
            ari = float(adjusted_rand_score(y_ground_truth, labels))
            nmi = float(normalized_mutual_info_score(y_ground_truth, labels))
            self.metrics["adjusted_rand_index"] = ari
            self.metrics["normalized_mutual_info"] = nmi
            logger.info(f"K-Means Evaluation: Silhouette={sil_score:.4f}, ARI={ari:.4f}, NMI={nmi:.4f}")
        else:
            logger.info(f"K-Means Evaluation: Silhouette={sil_score:.4f}, Inertia={inertia:.2f}")

        return self.metrics

    def run_elbow_and_silhouette_analysis(
        self,
        X: np.ndarray,
        k_range: range = range(2, 8),
        output_dir: Optional[Path] = None,
    ) -> Tuple[Path, Dict[int, float], Dict[int, float]]:
        """
        Runs K-Means over k_range to compute Inertia (Elbow) and Silhouette Scores.
        Generates comprehensive multi-panel diagnostic plot.
        """
        out_dir = output_dir or config.CLUSTERING_OUTPUT_DIR
        out_dir.mkdir(parents=True, exist_ok=True)

        inertias = {}
        silhouettes = {}

        for k in k_range:
            km = KMeans(n_clusters=k, init="k-means++", n_init=10, random_state=config.RANDOM_STATE)
            km_labels = km.fit_predict(X)
            inertias[k] = float(km.inertia_)
            silhouettes[k] = float(silhouette_score(X, km_labels))

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

        # 1. Elbow Method
        k_vals = list(k_range)
        ax1.plot(k_vals, [inertias[k] for k in k_vals], "bo-", linewidth=2.0, markersize=7)
        ax1.set_title("K-Means Elbow Method (Inertia vs K)", fontsize=12, fontweight="bold")
        ax1.set_xlabel("Number of Clusters (K)", fontsize=10)
        ax1.set_ylabel("Inertia (Sum of Squared Distances)", fontsize=10)
        ax1.grid(True, linestyle="--", alpha=0.5)

        # 2. Silhouette Score vs K
        ax2.plot(k_vals, [silhouettes[k] for k in k_vals], "ro-", linewidth=2.0, markersize=7)
        ax2.axvline(self.n_clusters, color="green", linestyle=":", label=f"Selected K={self.n_clusters}")
        ax2.set_title("Silhouette Score vs Number of Clusters", fontsize=12, fontweight="bold")
        ax2.set_xlabel("Number of Clusters (K)", fontsize=10)
        ax2.set_ylabel("Mean Silhouette Coefficient", fontsize=10)
        ax2.legend()
        ax2.grid(True, linestyle="--", alpha=0.5)

        plt.tight_layout()
        save_path = out_dir / "kmeans_elbow_silhouette.png"
        plt.savefig(save_path, dpi=300)
        plt.close()
        logger.info(f"Saved Elbow & Silhouette plot to {save_path}")
        return save_path, inertias, silhouettes

    def plot_clusters_2d(self, X: np.ndarray, X_2d: np.ndarray, output_dir: Optional[Path] = None) -> Path:
        """Plots 2D cluster assignments with projected centroids."""
        out_dir = output_dir or config.CLUSTERING_OUTPUT_DIR
        out_dir.mkdir(parents=True, exist_ok=True)

        cluster_labels = self.predict(X)

        fig, ax = plt.subplots(figsize=(9, 6.5))
        cmap = plt.get_cmap("viridis")

        scatter = ax.scatter(
            X_2d[:, 0],
            X_2d[:, 1],
            c=cluster_labels,
            cmap=cmap,
            s=35,
            alpha=0.75,
            edgecolor="k",
            linewidth=0.4,
        )

        # Compute cluster centroids in 2D
        for k in range(self.n_clusters):
            pts = X_2d[cluster_labels == k]
            if len(pts) > 0:
                c_2d = pts.mean(axis=0)
                ax.scatter(c_2d[0], c_2d[1], marker="X", s=200, c="red", edgecolor="black", linewidth=1.5)
                ax.text(c_2d[0] + 0.1, c_2d[1] + 0.1, f"C{k}", fontsize=11, fontweight="bold", color="darkred")

        ax.set_title(f"K-Means Discovered Clusters (K={self.n_clusters}) in 2D PCA Space", fontsize=13, fontweight="bold")
        ax.set_xlabel("Principal Component 1", fontsize=11)
        ax.set_ylabel("Principal Component 2", fontsize=11)
        plt.colorbar(scatter, ax=ax, label="Cluster ID")
        ax.grid(True, linestyle="--", alpha=0.5)

        plt.tight_layout()
        save_path = out_dir / "kmeans_clusters_2d.png"
        plt.savefig(save_path, dpi=300)
        plt.close()
        logger.info(f"Saved K-Means 2D cluster plot to {save_path}")
        return save_path

    def save(self, filename: str = "kmeans.joblib") -> Path:
        """Saves the trained model artifact."""
        return save_model(self, filename)

    @classmethod
    def load(cls, filename: str = "kmeans.joblib") -> "DrowsinessKMeans":
        """Loads a pre-trained model artifact."""
        return load_model(filename)
