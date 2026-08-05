"""
Random Forest Classifier Module (Unit 5: Bagging & Ensemble Methods).
Implements Random Forest with bootstrap aggregation, Out-of-Bag (OOB) score tracking,
ensemble variance reduction, and MDI feature importances with standard deviation error bars.
"""

from pathlib import Path
from typing import Dict, Any, Optional, Tuple, List

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, classification_report

from src import config
from src.utils import setup_logger, save_model, load_model

logger = setup_logger("RandomForest")


class DrowsinessRandomForest:
    """
    Unit 5 Random Forest Ensemble Engine.
    Aggregates diverse decorrelated decision trees to achieve low variance and high robustness.
    """

    def __init__(
        self,
        n_estimators: int = 100,
        max_depth: Optional[int] = 12,
        min_samples_split: int = 4,
        oob_score: bool = True,
    ):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.oob_score = oob_score

        self.model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            oob_score=oob_score,
            random_state=config.RANDOM_STATE,
            n_jobs=-1,
        )
        self.feature_names = config.FEATURE_COLUMNS
        self.class_names = config.CLASS_LABELS
        self.metrics: Dict[str, Any] = {}

    def fit(self, X_train: np.ndarray, y_train: np.ndarray) -> "DrowsinessRandomForest":
        """Fits Random Forest on training data."""
        logger.info(f"Fitting Random Forest (n_estimators={self.n_estimators}, max_depth={self.max_depth}) on {X_train.shape[0]} samples...")
        self.model.fit(X_train, y_train)
        oob_res = f", OOB Score: {self.model.oob_score_:.4f}" if self.oob_score else ""
        logger.info(f"Random Forest fitted ({len(self.model.estimators_)} trees{oob_res}).")
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predicts class labels via majority soft voting."""
        return self.model.predict(X)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predicts averaged class probabilities across all decision trees."""
        return self.model.predict_proba(X)

    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, Any]:
        """Evaluates model performance and returns metrics."""
        y_pred = self.predict(X_test)
        acc = float(accuracy_score(y_test, y_pred))
        macro_f1 = float(f1_score(y_test, y_pred, average="macro"))

        self.metrics = {
            "accuracy": acc,
            "macro_f1": macro_f1,
            "oob_score": float(self.model.oob_score_) if self.oob_score else None,
            "n_estimators": self.n_estimators,
            "classification_report": classification_report(y_test, y_pred, target_names=self.class_names, output_dict=True),
        }
        logger.info(f"Random Forest Evaluation: Accuracy={acc:.4f}, Macro-F1={macro_f1:.4f}, OOB={self.metrics['oob_score']}")
        return self.metrics

    def get_feature_importances(self) -> pd.DataFrame:
        """Computes feature importances and standard deviation across all trees."""
        importances = self.model.feature_importances_
        std = np.std([tree.feature_importances_ for tree in self.model.estimators_], axis=0)

        df_imp = pd.DataFrame({
            "feature": self.feature_names,
            "importance": importances,
            "std": std,
        }).sort_values(by="importance", ascending=False).reset_index(drop=True)
        return df_imp

    def plot_feature_importances(self, output_dir: Optional[Path] = None) -> Path:
        """Generates feature importance bar chart with error bars representing inter-tree variance."""
        out_dir = output_dir or config.EVAL_OUTPUT_DIR
        out_dir.mkdir(parents=True, exist_ok=True)

        df_imp = self.get_feature_importances()

        fig, ax = plt.subplots(figsize=(9, 5.5))
        y_pos = np.arange(len(df_imp))
        ax.barh(y_pos, df_imp["importance"], xerr=df_imp["std"], color="#27ae60", alpha=0.85, edgecolor="k", capsize=4)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(df_imp["feature"], fontsize=10, fontweight="bold")
        ax.invert_yaxis()
        ax.set_xlabel("Mean Decrease in Impurity (MDI)", fontsize=11, fontweight="bold")
        ax.set_title("Random Forest Feature Importances & Inter-Tree Variance (Unit 5)", fontsize=13, fontweight="bold")
        ax.grid(axis="x", linestyle="--", alpha=0.5)

        plt.tight_layout()
        save_path = out_dir / "random_forest_feature_importance.png"
        plt.savefig(save_path, dpi=300)
        plt.close()
        logger.info(f"Saved Random Forest feature importance plot to {save_path}")
        return save_path

    def plot_oob_convergence(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_test: np.ndarray,
        y_test: np.ndarray,
        tree_range: List[int] = [10, 25, 50, 75, 100, 150],
        output_dir: Optional[Path] = None,
    ) -> Path:
        """Plots OOB error and Test error as a function of the number of trees."""
        out_dir = output_dir or config.EVAL_OUTPUT_DIR
        out_dir.mkdir(parents=True, exist_ok=True)

        oob_errors = []
        test_errors = []

        for n in tree_range:
            rf = RandomForestClassifier(
                n_estimators=n,
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
                oob_score=True,
                random_state=config.RANDOM_STATE,
                n_jobs=-1,
            )
            rf.fit(X_train, y_train)
            oob_err = 1.0 - float(rf.oob_score_)
            test_err = 1.0 - float(accuracy_score(y_test, rf.predict(X_test)))
            oob_errors.append(oob_err)
            test_errors.append(test_err)

        fig, ax = plt.subplots(figsize=(9, 5))
        ax.plot(tree_range, oob_errors, "bo-", linewidth=2.0, label="Out-of-Bag (OOB) Error")
        ax.plot(tree_range, test_errors, "rs-", linewidth=2.0, label="Holdout Test Error")
        ax.set_title("Random Forest Convergence: OOB & Test Error vs Number of Trees", fontsize=13, fontweight="bold")
        ax.set_xlabel("Number of Estimator Trees", fontsize=11)
        ax.set_ylabel("Classification Error Rate", fontsize=11)
        ax.legend(framealpha=0.9)
        ax.grid(True, linestyle="--", alpha=0.5)

        plt.tight_layout()
        save_path = out_dir / "random_forest_oob_convergence.png"
        plt.savefig(save_path, dpi=300)
        plt.close()
        logger.info(f"Saved Random Forest OOB convergence plot to {save_path}")
        return save_path

    def plot_oob_error_curve(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_test: Optional[np.ndarray] = None,
        y_test: Optional[np.ndarray] = None,
        output_dir: Optional[Path] = None,
    ) -> Path:
        """Alias and wrapper for OOB error curve plotting."""
        X_t = X_train if X_test is None else X_test
        y_t = y_train if y_test is None else y_test
        return self.plot_oob_convergence(X_train, y_train, X_t, y_t, output_dir=output_dir)

    def save(self, filename: str = "random_forest.joblib") -> Path:
        """Saves model artifact."""
        return save_model(self, filename)

    @classmethod
    def load(cls, filename: str = "random_forest.joblib") -> "DrowsinessRandomForest":
        """Loads pre-trained model artifact."""
        return load_model(filename)
