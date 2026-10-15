
import time
import os
from pathlib import Path
from typing import Dict, Any, Optional, Tuple, List, Union

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    roc_curve,
    auc,
)
from sklearn.preprocessing import label_binarize

from src import config
from src.utils import setup_logger, load_model

logger = setup_logger("Evaluation")


class ModelEvaluator:

    def __init__(self, output_dir: Optional[Path] = None):
        self.output_dir = output_dir or config.EVAL_OUTPUT_DIR
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.class_names = config.CLASS_LABELS
        self.n_classes = config.NUM_CLASSES
        self.results: Dict[str, Dict[str, Any]] = {}

    def benchmark_model(
        self,
        name: str,
        predict_fn,
        predict_proba_fn,
        X_test: np.ndarray,
        y_test: np.ndarray,
        model_file: Optional[str] = None,
        warmup_runs: int = 5,
        bench_runs: int = 25,
    ) -> Dict[str, Any]:
        logger.info(f"Benchmarking model: {name}...")

        for _ in range(warmup_runs):
            _ = predict_fn(X_test[:5])

        start_time = time.perf_counter()
        for _ in range(bench_runs):
            y_pred = predict_fn(X_test)
        total_time = time.perf_counter() - start_time

        avg_latency_ms = (total_time / (bench_runs * len(X_test))) * 1000.0
        throughput_fps = (bench_runs * len(X_test)) / total_time

        y_pred = predict_fn(X_test)
        acc = float(accuracy_score(y_test, y_pred))
        prec_macro = float(precision_score(y_test, y_pred, average="macro", zero_division=0))
        rec_macro = float(recall_score(y_test, y_pred, average="macro", zero_division=0))
        f1_macro = float(f1_score(y_test, y_pred, average="macro", zero_division=0))
        cm = confusion_matrix(y_test, y_pred)

        roc_auc = None
        y_proba = None
        if predict_proba_fn is not None:
            try:
                y_proba = np.asarray(predict_proba_fn(X_test))
                present = np.unique(y_test)
                if present.size < 2:
                    raise ValueError(f"only {present.size} class(es) present in y_test")
                y_test_bin = label_binarize(y_test, classes=present)
                y_score = y_proba[:, present]
                if y_test_bin.shape[1] == 1:
                    roc_auc = float(roc_auc_score(y_test_bin.ravel(), y_score[:, 1]))
                else:
                    roc_auc = float(roc_auc_score(y_test_bin, y_score, multi_class="ovr", average="macro"))
                present_set = set(present.tolist())
                absent = [c for c in range(self.n_classes) if c not in present_set]
                if absent:
                    labels = [config.ID_TO_LABEL.get(c, str(c)) for c in absent]
                    logger.info(
                        f"[{name}] ROC AUC macro-averaged over the {present.size} classes with "
                        f"y_test support; no samples for {labels} -> excluded"
                    )
            except Exception as e:
                logger.warning(f"ROC AUC computation skipped for {name}: {e}")

        size_kb = 0.0
        if model_file:
            path = config.MODELS_DIR / model_file
            if path.exists():
                size_kb = os.path.getsize(path) / 1024.0

        res = {
            "name": name,
            "accuracy": acc,
            "precision_macro": prec_macro,
            "recall_macro": rec_macro,
            "f1_macro": f1_macro,
            "roc_auc": roc_auc,
            "latency_ms": avg_latency_ms,
            "throughput_fps": throughput_fps,
            "size_kb": size_kb,
            "y_pred": y_pred,
            "y_proba": y_proba,
            "confusion_matrix": cm,
        }

        self.results[name] = res
        logger.info(f"[{name}] Acc={acc*100:.2f}%, F1={f1_macro*100:.2f}%, Latency={avg_latency_ms:.3f}ms, Throughput={throughput_fps:.1f} fps")
        return res

    def generate_summary_table(self, report_stem: Optional[str] = None) -> pd.DataFrame:
        rows = []
        for name, data in self.results.items():
            rows.append({
                "Model": name,
                "Accuracy (%)": round(data["accuracy"] * 100, 2),
                "Precision (%)": round(data["precision_macro"] * 100, 2),
                "Recall (%)": round(data["recall_macro"] * 100, 2),
                "Macro F1 (%)": round(data["f1_macro"] * 100, 2),
                "ROC AUC": round(data["roc_auc"], 4) if data["roc_auc"] is not None else "N/A",
                "Latency (ms)": round(data["latency_ms"], 4),
                "Throughput (FPS)": round(data["throughput_fps"], 1),
                "Size (KB)": round(data["size_kb"], 1),
            })

        df = pd.DataFrame(rows).sort_values(by="Macro F1 (%)", ascending=False).reset_index(drop=True)
        stem = report_stem or "model_comparison_metrics"
        is_canonical = stem == "model_comparison_metrics"

        csv_path = self.output_dir / f"{stem}.csv"
        df.to_csv(csv_path, index=False)
        if is_canonical:
            df.to_csv(config.EVALUATION_REPORT_PATH, index=False)
            logger.info(f"Saved metrics CSV to {csv_path} and {config.EVALUATION_REPORT_PATH}")
        else:
            logger.info(f"Saved metrics CSV to {csv_path} (canonical report paths left untouched)")

        md_path = self.output_dir / f"{stem}.md"
        headers = list(df.columns)
        md_lines = [
            "# Model Performance & Real-Time Benchmark Comparison\n",
            "| " + " | ".join(headers) + " |",
            "| " + " | ".join(["---"] * len(headers)) + " |",
        ]
        for _, row in df.iterrows():
            md_lines.append("| " + " | ".join(str(row[h]) for h in headers) + " |")
        md_lines.append("")

        with open(md_path, "w", encoding="utf-8") as f:
            f.write("\n".join(md_lines))
        if is_canonical:
            with open(config.EVALUATION_SUMMARY_MD, "w", encoding="utf-8") as f:
                f.write("\n".join(md_lines))
            logger.info(f"Saved metrics Markdown to {md_path} and {config.EVALUATION_SUMMARY_MD}")
        else:
            logger.info(f"Saved metrics Markdown to {md_path} (canonical report paths left untouched)")

        return df

    def plot_confusion_matrices(self) -> Path:
        models = [name for name in self.results if "confusion_matrix" in self.results[name]]
        n_models = len(models)
        if n_models == 0:
            raise ValueError("No model results available to plot.")

        cols = 3
        rows = int(np.ceil(n_models / cols))
        fig, axes = plt.subplots(rows, cols, figsize=(15, 4.5 * rows))
        axes = np.array(axes).flatten()

        for idx, name in enumerate(models):
            cm = self.results[name]["confusion_matrix"]
            ax = axes[idx]
            sns.heatmap(
                cm,
                annot=True,
                fmt="d",
                cmap="Blues",
                xticklabels=self.class_names,
                yticklabels=self.class_names,
                cbar=False,
                ax=ax,
                annot_kws={"size": 10, "weight": "bold"},
            )
            acc = self.results[name]["accuracy"]
            f1 = self.results[name]["f1_macro"]
            ax.set_title(f"{name}\nAcc: {acc*100:.1f}% | F1: {f1*100:.1f}%", fontsize=11, fontweight="bold")
            ax.set_xlabel("Predicted Label", fontsize=9)
            ax.set_ylabel("True Label", fontsize=9)

        for idx in range(n_models, len(axes)):
            fig.delaxes(axes[idx])

        plt.suptitle("Multi-Model Confusion Matrices Comparison (Unit 5)", fontsize=15, fontweight="bold", y=1.002)
        plt.tight_layout()

        save_path = self.output_dir / "all_models_confusion_matrices.png"
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close()
        logger.info(f"Saved all confusion matrices plot to {save_path}")
        return save_path

    def plot_roc_curves(self, y_test: np.ndarray) -> Path:
        fig, ax = plt.subplots(figsize=(10, 7))
        y_test_bin = label_binarize(y_test, classes=range(self.n_classes))

        colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b", "#e377c2", "#7f7f7f"]
        c_idx = 0

        for name, data in self.results.items():
            if data["y_proba"] is not None:
                fpr = dict()
                tpr = dict()
                roc_auc_dict = dict()
                for i in range(self.n_classes):
                    fpr[i], tpr[i], _ = roc_curve(y_test_bin[:, i], data["y_proba"][:, i])
                    roc_auc_dict[i] = auc(fpr[i], tpr[i])

                all_fpr = np.unique(np.concatenate([fpr[i] for i in range(self.n_classes)]))
                mean_tpr = np.zeros_like(all_fpr)
                for i in range(self.n_classes):
                    mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])
                mean_tpr /= self.n_classes

                macro_auc = auc(all_fpr, mean_tpr)
                color = colors[c_idx % len(colors)]
                ax.plot(all_fpr, mean_tpr, label=f"{name} (Macro AUC = {macro_auc:.3f})", color=color, linewidth=2.0)
                c_idx += 1

        ax.plot([0, 1], [0, 1], "k--", lw=1.5, label="Random Guess (AUC = 0.500)")
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel("False Positive Rate (FPR)", fontsize=11, fontweight="bold")
        ax.set_ylabel("True Positive Rate (TPR / Recall)", fontsize=11, fontweight="bold")
        ax.set_title("Receiver Operating Characteristic (ROC) Multi-Model Curves", fontsize=13, fontweight="bold")
        ax.legend(loc="lower right", fontsize=9, framealpha=0.9)
        ax.grid(True, linestyle="--", alpha=0.5)

        plt.tight_layout()
        save_path = self.output_dir / "multi_model_roc_curves.png"
        plt.savefig(save_path, dpi=300)
        plt.close()
        logger.info(f"Saved multi-model ROC curves plot to {save_path}")
        return save_path

    def plot_benchmark_comparison(self) -> Path:
        df = self.generate_summary_table()

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))

        scatter = ax1.scatter(
            df["Latency (ms)"],
            df["Macro F1 (%)"],
            s=df["Size (KB)"] * 0.5 + 80,
            c=np.arange(len(df)),
            cmap="viridis",
            alpha=0.85,
            edgecolors="black",
            linewidth=1.5,
        )
        for _, row in df.iterrows():
            ax1.annotate(
                row["Model"],
                (row["Latency (ms)"], row["Macro F1 (%)"]),
                xytext=(6, 2),
                textcoords="offset points",
                fontsize=9,
                fontweight="bold",
            )
        ax1.set_xlabel("Inference Latency per Sample (ms) [Lower is Better]", fontsize=10, fontweight="bold")
        ax1.set_ylabel("Macro F1-Score (%) [Higher is Better]", fontsize=10, fontweight="bold")
        ax1.set_title("Pareto Efficiency: Accuracy vs Latency (Bubble Size = Model File Size)", fontsize=11, fontweight="bold")
        ax1.grid(True, linestyle="--", alpha=0.5)

        y_pos = np.arange(len(df))
        ax2.barh(y_pos, df["Throughput (FPS)"], color="#3498db", alpha=0.85, edgecolor="k")
        ax2.set_yticks(y_pos)
        ax2.set_yticklabels(df["Model"], fontsize=10, fontweight="bold")
        ax2.invert_yaxis()
        ax2.set_xlabel("Throughput (Frames Per Second)", fontsize=10, fontweight="bold")
        ax2.set_title("Model Throughput / Processing Speed (FPS)", fontsize=11, fontweight="bold")
        ax2.grid(axis="x", linestyle="--", alpha=0.5)

        for i, v in enumerate(df["Throughput (FPS)"]):
            ax2.text(v + 50, i, f"{int(v)} fps", va="center", fontsize=8.5, fontweight="bold")

        plt.suptitle("System Performance & Computational Efficiency Benchmark (Unit 5)", fontsize=13, fontweight="bold")
        plt.tight_layout()

        save_path = self.output_dir / "benchmark_comparison.png"
        plt.savefig(save_path, dpi=300)
        plt.close()
        logger.info(f"Saved benchmark comparison plot to {save_path}")
        return save_path

    def plot_calibration_curves(self, y_test: np.ndarray) -> Path:
        from sklearn.calibration import calibration_curve
        from sklearn.metrics import brier_score_loss

        fig, ax = plt.subplots(figsize=(9, 6.5))
        y_test_bin = label_binarize(y_test, classes=range(self.n_classes))

        drowsy_idx = 2
        ax.plot([0, 1], [0, 1], "k--", label="Perfect Calibration (Ideal)", lw=1.5)

        colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b"]
        c_idx = 0

        for name, data in self.results.items():
            if data["y_proba"] is not None:
                prob_pos = data["y_proba"][:, drowsy_idx]
                y_true_binary = (y_test == drowsy_idx).astype(int)

                prob_true, prob_pred = calibration_curve(y_true_binary, prob_pos, n_bins=10, strategy="uniform")
                brier = brier_score_loss(y_true_binary, prob_pos)

                color = colors[c_idx % len(colors)]
                ax.plot(prob_pred, prob_true, marker="o", lw=1.8, color=color, label=f"{name} (Brier={brier:.4f})")
                c_idx += 1

        ax.set_xlabel("Mean Predicted Probability P(State = Drowsy)", fontsize=11, fontweight="bold")
        ax.set_ylabel("Empirical True Fraction of Positives", fontsize=11, fontweight="bold")
        ax.set_title("Probability Calibration Reliability Diagram (Unit 2/5)", fontsize=13, fontweight="bold")
        ax.legend(loc="lower right", fontsize=9, framealpha=0.9)
        ax.grid(True, linestyle="--", alpha=0.5)

        plt.tight_layout()
        save_path = self.output_dir / "model_calibration_curves.png"
        plt.savefig(save_path, dpi=300)
        plt.close()
        logger.info(f"Saved probability calibration curves to {save_path}")
        return save_path
