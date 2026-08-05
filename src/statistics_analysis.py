"""
Statistical Analysis & Exploratory Data Analysis (Unit 1: Data Understanding).
Computes summary statistics, correlation matrices, class distributions, and generates
visualizations (histograms, heatmaps, boxplots, scatter plots).
"""

import json
from pathlib import Path
from typing import Dict, Any, Optional

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from src import config
from src.utils import setup_logger

logger = setup_logger("StatisticsAnalysis")

# Configure plotting style
plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["DejaVu Sans", "Arial", "Helvetica"]
plt.rcParams["axes.edgecolor"] = "#cccccc"
plt.rcParams["axes.linewidth"] = 0.8


class StatisticalAnalyzer:
    """
    Performs comprehensive statistical analysis and visualization for Unit 1.
    """

    def __init__(self, df: pd.DataFrame, output_dir: Optional[Path] = None):
        self.df = df.copy()
        self.output_dir = output_dir or config.EDA_OUTPUT_DIR
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.feature_cols = [c for c in config.FEATURE_COLUMNS if c in self.df.columns]
        self.target_col = config.TARGET_COLUMN
        self.regression_col = config.REGRESSION_TARGET

    def compute_summary_statistics(self) -> pd.DataFrame:
        """Computes mean, median, std, min, max, skewness, and kurtosis for all features."""
        numeric_cols = self.feature_cols + ([self.regression_col] if self.regression_col in self.df.columns else [])
        stats_df = self.df[numeric_cols].describe().T
        stats_df["median"] = self.df[numeric_cols].median()
        stats_df["skew"] = self.df[numeric_cols].skew()
        stats_df["kurtosis"] = self.df[numeric_cols].kurtosis()

        summary_csv = self.output_dir / "summary_statistics.csv"
        stats_df.to_csv(summary_csv)
        logger.info(f"Saved feature summary statistics to {summary_csv}")
        return stats_df

    def compute_class_distribution(self) -> Dict[str, Any]:
        """Calculates frequency counts and percentages for each driver state."""
        counts = self.df[self.target_col].value_counts().to_dict()
        percentages = (self.df[self.target_col].value_counts(normalize=True) * 100).round(2).to_dict()

        dist_data = {
            "counts": counts,
            "percentages": percentages,
            "total_samples": len(self.df),
        }
        with open(self.output_dir / "class_distribution.json", "w", encoding="utf-8") as f:
            json.dump(dist_data, f, indent=4)

        logger.info(f"Class Distribution: {counts}")
        return dist_data

    def plot_class_distribution(self) -> Path:
        """Plots driver state distribution bar chart."""
        plt.figure(figsize=(8, 5))
        counts = self.df[self.target_col].value_counts()
        colors = [config.CLASS_COLORS_HEX.get(lbl, "#3388ff") for lbl in counts.index]

        bars = plt.bar(counts.index, counts.values, color=colors, edgecolor="#222222", linewidth=1.2, width=0.55)
        plt.title("Driver State Class Distribution (Unit 1)", fontsize=14, fontweight="bold", pad=12)
        plt.xlabel("Driver State", fontsize=11, fontweight="semibold")
        plt.ylabel("Sample Count", fontsize=11, fontweight="semibold")

        for bar in bars:
            yval = bar.get_height()
            pct = (yval / len(self.df)) * 100
            plt.text(bar.get_x() + bar.get_width() / 2.0, yval + len(self.df) * 0.01, f"{int(yval)}\n({pct:.1f}%)", ha="center", va="bottom", fontsize=10, fontweight="bold")

        plt.tight_layout()
        save_path = self.output_dir / "class_distribution.png"
        plt.savefig(save_path, dpi=300)
        plt.close()
        logger.info(f"Saved class distribution plot to {save_path}")
        return save_path

    def plot_correlation_heatmap(self) -> Path:
        """Computes and plots Pearson correlation matrix between all features and fatigue score."""
        numeric_cols = self.feature_cols + ([self.regression_col] if self.regression_col in self.df.columns else [])
        corr_matrix = self.df[numeric_cols].corr()

        plt.figure(figsize=(10, 8))
        mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
        cmap = sns.diverging_palette(230, 20, as_cmap=True)

        sns.heatmap(
            corr_matrix,
            mask=mask,
            cmap=cmap,
            vmax=1.0,
            vmin=-1.0,
            center=0,
            annot=True,
            fmt=".2f",
            square=True,
            linewidths=0.5,
            cbar_kws={"shrink": 0.8, "label": "Pearson Correlation"},
        )
        plt.title("Feature Correlation Matrix (Unit 1)", fontsize=14, fontweight="bold", pad=15)
        plt.tight_layout()

        save_path = self.output_dir / "correlation_heatmap.png"
        plt.savefig(save_path, dpi=300)
        plt.close()
        logger.info(f"Saved correlation heatmap to {save_path}")
        return save_path

    def plot_feature_histograms(self) -> Path:
        """Plots distribution histograms with KDE for key physiological features across states."""
        key_features = ["ear", "mar", "blink_duration", "blink_rate", "perclos", "head_pitch"]
        fig, axes = plt.subplots(2, 3, figsize=(15, 9))
        axes = axes.flatten()

        for idx, feat in enumerate(key_features):
            if feat in self.df.columns:
                ax = axes[idx]
                for state_label in config.CLASS_LABELS:
                    subset = self.df[self.df[self.target_col] == state_label]
                    sns.kdeplot(
                        subset[feat],
                        ax=ax,
                        label=state_label,
                        color=config.CLASS_COLORS_HEX.get(state_label, "#3388ff"),
                        fill=True,
                        alpha=0.25,
                        linewidth=1.8,
                    )
                ax.set_title(f"Distribution of {feat.upper()}", fontsize=12, fontweight="bold")
                ax.set_xlabel(feat, fontsize=10)
                ax.set_ylabel("Density", fontsize=10)
                if idx == 0:
                    ax.legend(title="Driver State", fontsize=9)

        plt.suptitle("Feature Distributions Across Driver States (Unit 1)", fontsize=15, fontweight="bold", y=1.00)
        plt.tight_layout()

        save_path = self.output_dir / "feature_histograms.png"
        plt.savefig(save_path, dpi=300)
        plt.close()
        logger.info(f"Saved feature histograms to {save_path}")
        return save_path

    def plot_feature_boxplots(self) -> Path:
        """Plots boxplots showing median, IQR, and variance of features across states."""
        key_features = ["ear", "mar", "blink_duration", "perclos"]
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        axes = axes.flatten()
        palette = [config.CLASS_COLORS_HEX.get(lbl, "#3388ff") for lbl in config.CLASS_LABELS]

        for idx, feat in enumerate(key_features):
            if feat in self.df.columns:
                ax = axes[idx]
                sns.boxplot(
                    data=self.df,
                    x=self.target_col,
                    y=feat,
                    order=config.CLASS_LABELS,
                    hue=self.target_col,
                    palette=palette,
                    legend=False,
                    ax=ax,
                    width=0.5,
                    fliersize=3,
                    linewidth=1.2,
                )
                ax.set_title(f"{feat.upper()} by Driver State", fontsize=12, fontweight="bold")
                ax.set_xlabel("Driver State", fontsize=10)
                ax.set_ylabel(feat, fontsize=10)

        plt.suptitle("Feature Boxplots & State Separability (Unit 1)", fontsize=15, fontweight="bold", y=0.99)
        plt.tight_layout()

        save_path = self.output_dir / "feature_boxplots.png"
        plt.savefig(save_path, dpi=300)
        plt.close()
        logger.info(f"Saved feature boxplots to {save_path}")
        return save_path

    def plot_scatter_plots(self) -> Path:
        """Plots pairwise 2D scatter plots of critical fatigue interaction indicators."""
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))

        palette = {lbl: config.CLASS_COLORS_HEX.get(lbl, "#3388ff") for lbl in config.CLASS_LABELS}

        # Plot 1: EAR vs MAR
        sns.scatterplot(
            data=self.df,
            x="ear",
            y="mar",
            hue=self.target_col,
            palette=palette,
            alpha=0.65,
            edgecolor="none",
            s=40,
            ax=axes[0],
        )
        axes[0].set_title("Eye Aspect Ratio (EAR) vs Mouth Aspect Ratio (MAR)", fontsize=12, fontweight="bold")
        axes[0].set_xlabel("Eye Aspect Ratio (EAR)", fontsize=10)
        axes[0].set_ylabel("Mouth Aspect Ratio (MAR)", fontsize=10)

        # Plot 2: PERCLOS vs Fatigue Score
        sns.scatterplot(
            data=self.df,
            x="perclos",
            y=self.regression_col,
            hue=self.target_col,
            palette=palette,
            alpha=0.65,
            edgecolor="none",
            s=40,
            ax=axes[1],
        )
        axes[1].set_title("PERCLOS vs Continuous Fatigue Score", fontsize=12, fontweight="bold")
        axes[1].set_xlabel("PERCLOS (% Eye Closure)", fontsize=10)
        axes[1].set_ylabel("Fatigue Score (0-100)", fontsize=10)

        plt.suptitle("Pairwise Interaction & Discriminability Scatter Plots (Unit 1)", fontsize=14, fontweight="bold", y=1.00)
        plt.tight_layout()

        save_path = self.output_dir / "scatter_plots.png"
        plt.savefig(save_path, dpi=300)
        plt.close()
        logger.info(f"Saved scatter plots to {save_path}")
        return save_path

    def run_full_analysis(self) -> Dict[str, Any]:
        """Runs full Unit 1 statistical and EDA pipeline and outputs all artifacts."""
        logger.info("Executing comprehensive Unit 1 statistical analysis...")
        summary_stats = self.compute_summary_statistics()
        class_dist = self.compute_class_distribution()

        plot_paths = {
            "class_dist_plot": str(self.plot_class_distribution()),
            "heatmap_plot": str(self.plot_correlation_heatmap()),
            "histogram_plot": str(self.plot_feature_histograms()),
            "boxplot_plot": str(self.plot_feature_boxplots()),
            "scatter_plot": str(self.plot_scatter_plots()),
        }
        logger.info("Unit 1 Statistical Analysis completed successfully.")
        return {
            "summary_stats_shape": summary_stats.shape,
            "class_distribution": class_dist,
            "generated_plots": plot_paths,
        }
