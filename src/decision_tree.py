"""
Decision Tree Classifier Module (Unit 5: Tree-Based Models).
Implements CART Decision Tree with Gini/Entropy criteria, depth regularization,
cost-complexity pruning, tree structure visualization, and feature importance rankings.
"""

from pathlib import Path
from typing import Dict, Any, Optional, Tuple, List

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score, f1_score, classification_report

from src import config
from src.utils import setup_logger, save_model, load_model

logger = setup_logger("DecisionTree")


class DrowsinessDecisionTree:
    """
    Unit 5 Decision Tree Classifier Engine.
    Constructs interpretable orthogonal decision boundaries with rule tracing.
    """

    def __init__(
        self,
        criterion: str = "gini",
        max_depth: Optional[int] = 5,
        min_samples_split: int = 10,
        min_samples_leaf: int = 5,
        ccp_alpha: float = 0.0,
    ):
        self.criterion = criterion
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.ccp_alpha = ccp_alpha

        self.model = DecisionTreeClassifier(
            criterion=criterion,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            ccp_alpha=ccp_alpha,
            random_state=config.RANDOM_STATE,
        )
        self.feature_names = config.FEATURE_COLUMNS
        self.class_names = config.CLASS_LABELS
        self.metrics: Dict[str, Any] = {}

    def fit(self, X_train: np.ndarray, y_train: np.ndarray) -> "DrowsinessDecisionTree":
        """Fits Decision Tree on training data."""
        logger.info(f"Fitting Decision Tree (criterion={self.criterion}, max_depth={self.max_depth}) on {X_train.shape[0]} samples...")
        self.model.fit(X_train, y_train)
        logger.info(f"Decision Tree fitted. Total tree nodes: {self.model.tree_.node_count}, Max actual depth: {self.model.get_depth()}")
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
            "tree_depth": int(self.model.get_depth()),
            "node_count": int(self.model.tree_.node_count),
            "classification_report": classification_report(y_test, y_pred, target_names=self.class_names, output_dict=True),
        }
        logger.info(f"Decision Tree Evaluation: Accuracy={acc:.4f}, Macro-F1={macro_f1:.4f}, Depth={self.metrics['tree_depth']}")
        return self.metrics

    def get_feature_importances(self) -> pd.DataFrame:
        """Returns Gini importance for all physiological features."""
        df_imp = pd.DataFrame({
            "feature": self.feature_names,
            "importance": self.model.feature_importances_,
        }).sort_values(by="importance", ascending=False).reset_index(drop=True)
        return df_imp

    def plot_tree_structure(self, output_dir: Optional[Path] = None, max_depth_plot: int = 3) -> Path:
        """Renders high-resolution graphic tree visualization."""
        out_dir = output_dir or config.EVAL_OUTPUT_DIR
        out_dir.mkdir(parents=True, exist_ok=True)

        fig, ax = plt.subplots(figsize=(16, 9))
        plot_tree(
            self.model,
            max_depth=max_depth_plot,
            feature_names=self.feature_names,
            class_names=self.class_names,
            filled=True,
            rounded=True,
            fontsize=9,
            ax=ax,
            precision=2,
        )
        ax.set_title(f"CART Decision Tree Structure (Criterion: {self.criterion.upper()}, Depth: {self.model.get_depth()})", fontsize=13, fontweight="bold")

        plt.tight_layout()
        save_path = out_dir / "decision_tree_structure.png"
        plt.savefig(save_path, dpi=300)
        plt.close()
        logger.info(f"Saved Decision Tree structure plot to {save_path}")
        return save_path

    def plot_feature_importances(self, output_dir: Optional[Path] = None) -> Path:
        """Bar plot of Gini / Information Gain feature importances."""
        out_dir = output_dir or config.EVAL_OUTPUT_DIR
        out_dir.mkdir(parents=True, exist_ok=True)

        df_imp = self.get_feature_importances()

        fig, ax = plt.subplots(figsize=(9, 5.5))
        y_pos = np.arange(len(df_imp))
        ax.barh(y_pos, df_imp["importance"], color="#2980b9", alpha=0.85, edgecolor="k")
        ax.set_yticks(y_pos)
        ax.set_yticklabels(df_imp["feature"], fontsize=10, fontweight="bold")
        ax.invert_yaxis()  # top feature on top
        ax.set_xlabel("Mean Decrease Impurity (Gini Importance)", fontsize=11, fontweight="bold")
        ax.set_title("Decision Tree Feature Importance Rankings (Unit 5)", fontsize=13, fontweight="bold")
        ax.grid(axis="x", linestyle="--", alpha=0.5)

        for i, v in enumerate(df_imp["importance"]):
            if v > 0.005:
                ax.text(v + 0.005, i, f"{v:.3f}", va="center", fontsize=9, fontweight="bold")

        plt.tight_layout()
        save_path = out_dir / "decision_tree_feature_importances.png"
        plt.savefig(save_path, dpi=300)
        plt.close()
        logger.info(f"Saved Decision Tree feature importance plot to {save_path}")
        return save_path

    def save(self, filename: str = "decision_tree.joblib") -> Path:
        """Saves model artifact."""
        return save_model(self, filename)

    @classmethod
    def load(cls, filename: str = "decision_tree.joblib") -> "DrowsinessDecisionTree":
        """Loads pre-trained model artifact."""
        return load_model(filename)
