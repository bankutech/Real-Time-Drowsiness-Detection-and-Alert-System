"""
Linear Regression Module (Unit 2: Linear Regression & Regularization).
Estimates continuous Fatigue Score (0-100) using Linear, Ridge (L2), and Lasso (L1) Regression.
Evaluates R^2, RMSE, MAE, and extracts feature weights and residual diagnostics.
"""

from pathlib import Path
from typing import Dict, Any, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

from src import config
from src.utils import setup_logger, save_model, load_model, clamp

logger = setup_logger("LinearRegression")


class FatigueScoreRegressor:
    """
    Unit 2 Continuous Fatigue Score Regressor.
    Predicts driver fatigue score in [0, 100] from standardized facial features.
    """

    def __init__(self, model_type: str = "ridge", alpha: float = 1.0):
        self.model_type = model_type.lower()
        self.alpha = alpha

        if self.model_type == "linear":
            self.model = LinearRegression()
        elif self.model_type == "lasso":
            self.model = Lasso(alpha=alpha, random_state=config.RANDOM_STATE)
        else:
            self.model = Ridge(alpha=alpha, random_state=config.RANDOM_STATE)

        self.feature_names = config.FEATURE_COLUMNS
        self.metrics: Dict[str, float] = {}

    def fit(self, X_train: np.ndarray, y_train: np.ndarray) -> "FatigueScoreRegressor":
        """Fits the regression model on training feature matrix and continuous fatigue scores."""
        logger.info(f"Fitting {self.model_type.upper()} Regression model on {X_train.shape[0]} samples...")
        self.model.fit(X_train, y_train)
        logger.info("Regression model fitting complete.")
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predicts continuous fatigue scores clamped to [0, 100]."""
        raw_preds = self.model.predict(X)
        return np.clip(raw_preds, 0.0, 100.0)

    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, float]:
        """Calculates R^2, RMSE, and MAE on test partition."""
        y_pred = self.predict(X_test)

        r2 = float(r2_score(y_test, y_pred))
        rmse = float(np.sqrt(mean_squared_error(y_test, y_pred)))
        mae = float(mean_absolute_error(y_test, y_pred))

        self.metrics = {
            "r2_score": r2,
            "rmse": rmse,
            "mae": mae,
            "model_type": self.model_type,
        }
        logger.info(f"Regression Performance ({self.model_type.upper()}): R^2={r2:.4f}, RMSE={rmse:.4f}, MAE={mae:.4f}")
        return self.metrics

    def get_feature_importances(self) -> pd.DataFrame:
        """Extracts linear coefficients indicating the direction and magnitude of each feature."""
        coefs = self.model.coef_
        df_coefs = pd.DataFrame({
            "feature": self.feature_names,
            "coefficient": coefs,
            "abs_impact": np.abs(coefs),
        }).sort_values(by="abs_impact", ascending=False).reset_index(drop=True)
        return df_coefs

    def plot_residual_analysis(self, X_test: np.ndarray, y_test: np.ndarray, output_dir: Optional[Path] = None) -> Path:
        """Generates actual vs predicted and residual diagnostic plots."""
        out_dir = output_dir or config.EVAL_OUTPUT_DIR
        out_dir.mkdir(parents=True, exist_ok=True)

        y_pred = self.predict(X_test)
        residuals = y_test - y_pred

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # 1. Actual vs Predicted
        axes[0].scatter(y_test, y_pred, color="#2b5c8f", alpha=0.55, edgecolor="none", s=35)
        min_val = min(y_test.min(), y_pred.min())
        max_val = max(y_test.max(), y_pred.max())
        axes[0].plot([min_val, max_val], [min_val, max_val], "r--", linewidth=1.8, label="Ideal (y=x)")
        axes[0].set_title(f"Actual vs Predicted Fatigue Score ({self.model_type.upper()})", fontsize=12, fontweight="bold")
        axes[0].set_xlabel("Ground Truth Fatigue Score", fontsize=10)
        axes[0].set_ylabel("Predicted Fatigue Score", fontsize=10)
        axes[0].legend()

        # 2. Residual Distribution
        axes[1].scatter(y_pred, residuals, color="#e06666", alpha=0.55, edgecolor="none", s=35)
        axes[1].axhline(0, color="black", linestyle="--", linewidth=1.2)
        axes[1].set_title("Residual Plot (Errors vs Predictions)", fontsize=12, fontweight="bold")
        axes[1].set_xlabel("Predicted Fatigue Score", fontsize=10)
        axes[1].set_ylabel("Residuals (Actual - Pred)", fontsize=10)

        plt.tight_layout()
        save_path = out_dir / f"regression_residuals_{self.model_type}.png"
        plt.savefig(save_path, dpi=300)
        plt.close()
        logger.info(f"Saved regression residual plot to {save_path}")
        return save_path

    def save(self, filename: str = "fatigue_regressor.joblib") -> Path:
        """Saves the trained regression model artifact."""
        return save_model(self, filename)

    @classmethod
    def load(cls, filename: str = "fatigue_regressor.joblib") -> "FatigueScoreRegressor":
        """Loads a pre-trained regression model artifact."""
        return load_model(filename)
