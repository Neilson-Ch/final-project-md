"""
Credit Scoring - Evaluation Module
=====================================
OOP-based evaluation: metrics, reports, confusion matrix,
ROC curves, and MLflow artifact logging.
"""

import os
import json
import logging
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns

import mlflow

from typing import Any, Dict, List, Optional

from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    roc_auc_score, classification_report, confusion_matrix,
    ConfusionMatrixDisplay, log_loss,
)

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

CLASS_NAMES = ["Poor", "Standard", "Good"]   # matches label 0/1/2


# ─────────────────────────────────────────────
#  Metrics Calculator
# ─────────────────────────────────────────────

class MetricsCalculator:
    """
    Computes a comprehensive set of classification metrics.
    """
    def __init__(self, class_names: List[str] = None):
        self.class_names = class_names or CLASS_NAMES

    def compute(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_proba: Optional[np.ndarray] = None,
    ) -> Dict[str, Any]:
        metrics: Dict[str, Any] = {}

        # Standard metrics
        metrics["accuracy"]         = accuracy_score(y_true, y_pred)
        metrics["f1_macro"]         = f1_score(y_true, y_pred, average="macro",   zero_division=0)
        metrics["f1_weighted"]      = f1_score(y_true, y_pred, average="weighted",zero_division=0)
        metrics["precision_macro"]  = precision_score(y_true, y_pred, average="macro",   zero_division=0)
        metrics["recall_macro"]     = recall_score(y_true, y_pred,    average="macro",   zero_division=0)

        # Per-class F1
        per_class_f1 = f1_score(y_true, y_pred, average=None, zero_division=0)
        for i, cls in enumerate(self.class_names):
            if i < len(per_class_f1):
                metrics[f"f1_{cls.lower()}"] = per_class_f1[i]

        # Probabilistic metrics (if proba available)
        if y_proba is not None:
            try:
                metrics["roc_auc_ovr"] = roc_auc_score(
                    y_true, y_proba, multi_class="ovr", average="macro"
                )
                metrics["roc_auc_ovo"] = roc_auc_score(
                    y_true, y_proba, multi_class="ovo", average="macro"
                )
                metrics["log_loss"] = log_loss(y_true, y_proba)
            except Exception as e:
                logger.warning(f"Could not compute proba-metrics: {e}")

        return metrics

    def classification_report_str(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
    ) -> str:
        return classification_report(
            y_true, y_pred,
            target_names=self.class_names,
            zero_division=0
        )


# ─────────────────────────────────────────────
#  Plot Helpers
# ─────────────────────────────────────────────

class PlotFactory:
    """Generates evaluation plots and saves them as PNG files."""

    PALETTE = {"Poor": "#E74C3C", "Standard": "#F39C12", "Good": "#2ECC71"}

    @staticmethod
    def confusion_matrix_plot(
        y_true: np.ndarray,
        y_pred: np.ndarray,
        class_names: List[str],
        save_path: str,
        title: str = "Confusion Matrix",
    ) -> str:
        cm = confusion_matrix(y_true, y_pred)
        fig, ax = plt.subplots(figsize=(7, 5))
        sns.heatmap(
            cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=class_names,
            yticklabels=class_names,
            linewidths=0.5, ax=ax
        )
        ax.set_title(title, fontsize=14, fontweight="bold", pad=12)
        ax.set_ylabel("True Label", fontsize=11)
        ax.set_xlabel("Predicted Label", fontsize=11)
        plt.tight_layout()
        plt.savefig(save_path, dpi=150)
        plt.close(fig)
        return save_path

    @staticmethod
    def normalized_confusion_matrix_plot(
        y_true: np.ndarray,
        y_pred: np.ndarray,
        class_names: List[str],
        save_path: str,
        title: str = "Confusion Matrix (Normalized)",
    ) -> str:
        cm = confusion_matrix(y_true, y_pred, normalize="true")
        fig, ax = plt.subplots(figsize=(7, 5))
        sns.heatmap(
            cm, annot=True, fmt=".2f", cmap="Blues",
            xticklabels=class_names,
            yticklabels=class_names,
            vmin=0, vmax=1,
            linewidths=0.5, ax=ax
        )
        ax.set_title(title, fontsize=14, fontweight="bold", pad=12)
        ax.set_ylabel("True Label", fontsize=11)
        ax.set_xlabel("Predicted Label", fontsize=11)
        plt.tight_layout()
        plt.savefig(save_path, dpi=150)
        plt.close(fig)
        return save_path

    @staticmethod
    def roc_curve_plot(
        y_true: np.ndarray,
        y_proba: np.ndarray,
        class_names: List[str],
        save_path: str,
        title: str = "ROC Curves (OvR)",
    ) -> str:
        from sklearn.preprocessing import label_binarize
        from sklearn.metrics import roc_curve, auc

        classes = list(range(len(class_names)))
        y_bin   = label_binarize(y_true, classes=classes)

        fig, ax = plt.subplots(figsize=(7, 5))
        colors  = ["#E74C3C", "#F39C12", "#2ECC71"]

        for i, (cls_name, color) in enumerate(zip(class_names, colors)):
            fpr, tpr, _ = roc_curve(y_bin[:, i], y_proba[:, i])
            roc_auc     = auc(fpr, tpr)
            ax.plot(fpr, tpr, lw=2, color=color,
                    label=f"{cls_name} (AUC = {roc_auc:.3f})")

        ax.plot([0, 1], [0, 1], "k--", lw=1)
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.02])
        ax.set_xlabel("False Positive Rate", fontsize=11)
        ax.set_ylabel("True Positive Rate", fontsize=11)
        ax.set_title(title, fontsize=14, fontweight="bold", pad=12)
        ax.legend(loc="lower right", fontsize=10)
        ax.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(save_path, dpi=150)
        plt.close(fig)
        return save_path

    @staticmethod
    def feature_importance_plot(
        feature_names: List[str],
        importances: np.ndarray,
        save_path: str,
        title: str = "Top Feature Importances",
        top_n: int = 20,
    ) -> str:
        pairs     = sorted(zip(feature_names, importances),
                           key=lambda x: x[1], reverse=True)[:top_n]
        names_top = [p[0] for p in pairs]
        imps_top  = [p[1] for p in pairs]

        fig, ax = plt.subplots(figsize=(9, max(4, top_n * 0.35)))
        bars    = ax.barh(names_top[::-1], imps_top[::-1],
                          color="#3498DB", edgecolor="white")
        ax.set_title(title, fontsize=13, fontweight="bold", pad=10)
        ax.set_xlabel("Importance", fontsize=11)
        ax.xaxis.set_major_formatter(mticker.FormatStrFormatter("%.4f"))
        ax.grid(axis="x", alpha=0.3)
        plt.tight_layout()
        plt.savefig(save_path, dpi=150)
        plt.close(fig)
        return save_path

    @staticmethod
    def metrics_bar_chart(
        model_metrics: Dict[str, Dict],
        metric_keys: List[str],
        save_path: str,
        title: str = "Model Comparison",
    ) -> str:
        """Grouped bar chart comparing multiple models on several metrics."""
        model_names = list(model_metrics.keys())
        n_models    = len(model_names)
        n_metrics   = len(metric_keys)
        x           = np.arange(n_metrics)
        width       = 0.8 / n_models
        colors      = plt.cm.Set2(np.linspace(0, 0.8, n_models))

        fig, ax = plt.subplots(figsize=(10, 5))
        for i, (name, color) in enumerate(zip(model_names, colors)):
            vals = [model_metrics[name].get(m, 0) for m in metric_keys]
            offset = (i - n_models / 2 + 0.5) * width
            bars = ax.bar(x + offset, vals, width,
                          label=name, color=color, edgecolor="white")
            for bar in bars:
                h = bar.get_height()
                ax.annotate(f"{h:.3f}",
                            xy=(bar.get_x() + bar.get_width() / 2, h),
                            xytext=(0, 3), textcoords="offset points",
                            ha="center", va="bottom", fontsize=7)

        ax.set_title(title, fontsize=14, fontweight="bold", pad=12)
        ax.set_xticks(x)
        ax.set_xticklabels([m.replace("_", "\n") for m in metric_keys], fontsize=9)
        ax.set_ylim(0, 1.1)
        ax.set_ylabel("Score", fontsize=11)
        ax.legend(fontsize=9)
        ax.grid(axis="y", alpha=0.3)
        plt.tight_layout()
        plt.savefig(save_path, dpi=150)
        plt.close(fig)
        return save_path


# ─────────────────────────────────────────────
#  Model Evaluator  (main class)
# ─────────────────────────────────────────────

class ModelEvaluator:
    """
    Evaluates a fitted BaseModel instance on a test set.
    Produces metrics, textual reports, and plots.

    Parameters
    ----------
    model         : A fitted BaseModel subclass instance
    class_names   : Label names for display (default CLASS_NAMES)
    feature_names : Column names for feature-importance plot
    artifact_dir  : Local directory to write plot/report files
    """

    def __init__(
        self,
        model,
        class_names: Optional[List[str]] = None,
        feature_names: Optional[List[str]] = None,
        artifact_dir: str = "/tmp/credit_scoring_artifacts",
    ):
        self.model         = model
        self.class_names   = class_names or CLASS_NAMES
        self.feature_names = feature_names
        self.artifact_dir  = artifact_dir
        os.makedirs(artifact_dir, exist_ok=True)

        self.metrics_calc_ = MetricsCalculator(self.class_names)
        self._last_metrics: Dict = {}

    # ── main evaluate ──────────────────────────
    def evaluate(
        self,
        X_test: np.ndarray,
        y_test: np.ndarray,
    ) -> Dict[str, Any]:
        """
        Run predictions, compute all metrics, return metric dict.
        """
        y_pred  = self.model.predict(X_test)
        y_proba = None
        try:
            y_proba = self.model.predict_proba(X_test)
        except Exception:
            pass

        self._last_metrics = self.metrics_calc_.compute(y_test, y_pred, y_proba)
        self._last_y_test   = y_test
        self._last_y_pred   = y_pred
        self._last_y_proba  = y_proba

        logger.info(f"[{self.model.model_name}] Evaluation complete. "
                    f"Accuracy={self._last_metrics['accuracy']:.4f}  "
                    f"F1-macro={self._last_metrics['f1_macro']:.4f}")
        return self._last_metrics

    # ── artifact logging ───────────────────────
    def log_artifacts_to_mlflow(
        self,
        X_test: np.ndarray,
        y_test: np.ndarray,
        run_id: Optional[str] = None,
    ):
        """
        Generate plots & text reports, then log them as MLflow artifacts.
        Must be called inside (or passing the run_id of) an active MLflow run.
        """
        mn = self.model.model_name
        prefix = os.path.join(self.artifact_dir, mn)

        # ── classification report ─────────────
        report_str  = self.metrics_calc_.classification_report_str(
            self._last_y_test, self._last_y_pred
        )
        report_path = f"{prefix}_classification_report.txt"
        with open(report_path, "w") as f:
            f.write(f"Model: {mn}\n\n")
            f.write(report_str)

        # ── confusion matrix ──────────────────
        cm_path = PlotFactory.confusion_matrix_plot(
            self._last_y_test, self._last_y_pred,
            self.class_names,
            save_path=f"{prefix}_confusion_matrix.png",
            title=f"{mn} – Confusion Matrix",
        )
        cm_norm_path = PlotFactory.normalized_confusion_matrix_plot(
            self._last_y_test, self._last_y_pred,
            self.class_names,
            save_path=f"{prefix}_confusion_matrix_norm.png",
            title=f"{mn} – Confusion Matrix (Normalized)",
        )

        # ── ROC curves ────────────────────────
        roc_path = None
        if self._last_y_proba is not None:
            roc_path = PlotFactory.roc_curve_plot(
                self._last_y_test, self._last_y_proba,
                self.class_names,
                save_path=f"{prefix}_roc_curves.png",
                title=f"{mn} – ROC Curves",
            )

        # ── feature importance plot ────────────
        fi_path = None
        if (hasattr(self.model, "get_feature_importances")
                and self.feature_names is not None):
            fi_path = PlotFactory.feature_importance_plot(
                self.feature_names,
                self.model.get_feature_importances(),
                save_path=f"{prefix}_feature_importance.png",
                title=f"{mn} – Feature Importance",
            )

        # ── metrics JSON ──────────────────────
        metrics_path = f"{prefix}_metrics.json"
        with open(metrics_path, "w") as f:
            json.dump({k: v for k, v in self._last_metrics.items()
                       if isinstance(v, (int, float))}, f, indent=2)

        # ── log to MLflow ─────────────────────
        def _log(path, subfolder):
            if path and os.path.exists(path):
                mlflow.log_artifact(path, artifact_path=subfolder)

        _log(report_path, "reports")
        _log(cm_path,      "plots")
        _log(cm_norm_path, "plots")
        _log(roc_path,     "plots")
        _log(fi_path,      "plots")
        _log(metrics_path, "reports")

        logger.info(f"[{mn}] Artifacts logged to MLflow.")

    # ── print report ──────────────────────────
    def print_report(self):
        if not self._last_metrics:
            print("No evaluation run yet.")
            return
        print(f"\n{'─'*55}")
        print(f"  Evaluation Report — {self.model.model_name}")
        print(f"{'─'*55}")
        for k, v in self._last_metrics.items():
            if isinstance(v, float):
                print(f"  {k:<30} {v:.4f}")
        print(f"{'─'*55}")
        print(self.metrics_calc_.classification_report_str(
            self._last_y_test, self._last_y_pred
        ))


# ─────────────────────────────────────────────
#  Comparative Evaluator (multiple models)
# ─────────────────────────────────────────────

class ComparativeEvaluator:
    """
    Takes the results dict from TrainingPipeline.run_all() and
    produces a side-by-side comparison chart + summary table.
    """

    def __init__(
        self,
        results: Dict[str, Dict],
        artifact_dir: str = "/tmp/credit_scoring_artifacts",
    ):
        self.results      = results          # model_name → {metrics, run_id, model}
        self.artifact_dir = artifact_dir
        os.makedirs(artifact_dir, exist_ok=True)

    def plot_comparison(
        self,
        metric_keys: Optional[List[str]] = None,
        save_path: Optional[str] = None,
    ) -> str:
        metric_keys = metric_keys or [
            "accuracy", "f1_macro", "f1_weighted",
            "roc_auc_ovr", "precision_macro", "recall_macro"
        ]
        model_metrics = {name: res["metrics"]
                         for name, res in self.results.items()}
        save_path = save_path or os.path.join(
            self.artifact_dir, "model_comparison.png"
        )
        PlotFactory.metrics_bar_chart(
            model_metrics, metric_keys, save_path,
            title="Credit Scoring — Model Comparison"
        )
        logger.info(f"Comparison chart saved → {save_path}")
        return save_path

    def summary_dataframe(self) -> pd.DataFrame:
        rows = []
        for name, res in self.results.items():
            row = {"model": name, **{
                k: v for k, v in res["metrics"].items()
                if isinstance(v, (int, float))
            }}
            rows.append(row)
        df = pd.DataFrame(rows).set_index("model")
        return df.sort_values("f1_macro", ascending=False)

    def log_comparison_to_mlflow(self, experiment_name: str = "CreditScoring"):
        """Log the comparison plot under a dedicated 'comparison' run."""
        chart_path = self.plot_comparison()
        df = self.summary_dataframe()
        csv_path   = os.path.join(self.artifact_dir, "model_comparison.csv")
        df.to_csv(csv_path)

        tracking_uri = mlflow.get_tracking_uri()
        mlflow.set_tracking_uri(tracking_uri)
        mlflow.set_experiment(experiment_name)
        with mlflow.start_run(run_name="ModelComparison"):
            mlflow.set_tag("run_type", "comparison")
            mlflow.log_artifact(chart_path, "comparison")
            mlflow.log_artifact(csv_path,   "comparison")
            # also log best scores as metrics
            best_row = df.iloc[0]
            for col in ["accuracy", "f1_macro", "roc_auc_ovr"]:
                if col in best_row:
                    mlflow.log_metric(f"best_{col}", float(best_row[col]))
        logger.info("Comparison artifacts logged to MLflow.")
