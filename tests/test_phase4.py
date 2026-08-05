"""
Verification script for Phase 4 (Unit 3: PCA, K-Means, GMM, Hierarchical Clustering).
"""

import sys
from pathlib import Path
import numpy as np

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import src.config as config
from src.preprocessing import DataPreprocessor
from src.pca import PCAReducer
from src.kmeans import DrowsinessKMeans
from src.gmm import DrowsinessGMM
from src.hierarchical import DrowsinessHierarchicalClustering


def test_phase4():
    print("=" * 60)
    print("STARTING PHASE 4 VERIFICATION TEST")
    print("=" * 60)

    # 1. Load Preprocessed Data
    preprocessor = DataPreprocessor()
    df_raw = preprocessor.load_or_generate_dataset()
    df_clean, _ = preprocessor.clean_dataset(df_raw)
    X, y, y_fatigue = preprocessor.prepare_xy(df_clean)
    split_res = preprocessor.split_and_scale(X, y, y_fatigue)
    X_train, X_test = split_res["X_train"], split_res["X_test"]
    y_train, y_test = split_res["y_train"], split_res["y_test"]

    # 2. PCA Dimensionality Reduction
    print("\n--- Testing Principal Component Analysis (PCA) ---")
    pca_engine = PCAReducer(n_components=None, variance_threshold=0.95)
    pca_engine.fit(X_train)
    scree_path = pca_engine.plot_scree()
    assert scree_path.exists(), "PCA Scree plot should be generated"

    X_train_2d = pca_engine.transform(X_train, n_dims=2)
    proj_path = pca_engine.plot_2d_projection(X_train, y_train, title="PCA 2D State Separation")
    assert proj_path.exists(), "PCA 2D projection plot should be generated"

    loadings = pca_engine.get_loadings()
    print(f"[OK] PCA Top loadings for PC1 and PC2:\n{loadings[['PC1', 'PC2']].head(4)}")
    pca_engine.save()

    # 3. K-Means Clustering
    print("\n--- Testing K-Means Clustering ---")
    kmeans = DrowsinessKMeans(n_clusters=config.NUM_CLASSES)
    kmeans.fit(X_train)
    km_metrics = kmeans.evaluate_clustering(X_train, y_ground_truth=y_train)
    print(f"[OK] K-Means: Silhouette={km_metrics['silhouette_score']:.4f}, ARI={km_metrics['adjusted_rand_index']:.4f}")
    assert km_metrics["silhouette_score"] > 0.15, "Silhouette score should be positive"

    elbow_path, _, _ = kmeans.run_elbow_and_silhouette_analysis(X_train, k_range=range(2, 7))
    assert elbow_path.exists(), "K-Means Elbow/Silhouette plot should be generated"

    km_2d_path = kmeans.plot_clusters_2d(X_train, X_train_2d)
    assert km_2d_path.exists(), "K-Means 2D cluster plot should be generated"
    kmeans.save()

    # 4. Gaussian Mixture Models (GMM)
    print("\n--- Testing Gaussian Mixture Models (GMM) ---")
    gmm = DrowsinessGMM(n_components=config.NUM_CLASSES, covariance_type="full")
    gmm.fit(X_train)
    gmm_metrics = gmm.evaluate(X_train, y_ground_truth=y_train)
    print(f"[OK] GMM: BIC={gmm_metrics['bic']:.1f}, AIC={gmm_metrics['aic']:.1f}, Silhouette={gmm_metrics['silhouette_score']:.4f}, ARI={gmm_metrics['adjusted_rand_index']:.4f}")
    assert gmm_metrics["silhouette_score"] > 0.15, "GMM Silhouette score should be positive"

    aic_bic_path, _, _ = gmm.run_model_selection_aic_bic(X_train, k_range=range(2, 7))
    assert aic_bic_path.exists(), "GMM AIC/BIC plot should be generated"

    gmm_2d_path = gmm.plot_clusters_with_ellipses_2d(X_train_2d)
    assert gmm_2d_path.exists(), "GMM Confidence ellipses plot should be generated"
    gmm.save()

    # 5. Hierarchical Agglomerative Clustering
    print("\n--- Testing Hierarchical Clustering ---")
    hier = DrowsinessHierarchicalClustering(n_clusters=config.NUM_CLASSES, linkage_method="ward")
    hier.fit_predict(X_train)
    hier_metrics = hier.evaluate(X_train, y_ground_truth=y_train)
    print(f"[OK] Hierarchical: Silhouette={hier_metrics['silhouette_score']:.4f}, Cophenetic={hier_metrics['cophenetic_correlation']:.4f}, ARI={hier_metrics['adjusted_rand_index']:.4f}")
    assert hier_metrics["cophenetic_correlation"] > 0.50, "Cophenetic correlation should be > 0.50"

    dendro_path = hier.plot_dendrogram(max_d=50.0)
    assert dendro_path.exists(), "Hierarchical Dendrogram plot should be generated"

    hier_2d_path = hier.plot_clusters_2d(X_train, X_train_2d)
    assert hier_2d_path.exists(), "Hierarchical 2D cluster plot should be generated"
    hier.save()

    print("\n" + "=" * 60)
    print("PHASE 4 VERIFICATION PASSED SUCCESSFULLY!")
    print("=" * 60)


if __name__ == "__main__":
    test_phase4()
