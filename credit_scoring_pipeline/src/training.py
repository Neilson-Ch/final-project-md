"""
Credit Scoring - Training Module
==================================
OOP-based training pipeline with MLflow experiment tracking.
Supports: Logistic Regression, Random Forest, XGBoost, LightGBM.
"""

import os
import time
import pickle
import logging
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import mlflow
import mlflow.sklearn
import mlflow.xgboost
import mlflow.lightgbm

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Tuple

from sklearn.linear_model    import LogisticRegression
from sklearn.ensemble        import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score
from xgboost                 import XGBClassifier
from lightgbm                import LGBMClassifier

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────
#  Hyperparameter Configs (dataclasses)
# ─────────────────────────────────────────────

@dataclass
class LogisticRegressionConfig:
    C: float             = 1.0
    max_iter: int        = 1000
    solver: str          = "lbfgs"
    class_weight: str    = "balanced"
    random_state: int    = 42

@dataclass
class RandomForestConfig:
    n_estimators: int    = 200
    max_depth: int       = 10
    min_samples_split: int = 5
    min_samples_leaf: int  = 2
    class_weight: str    = "balanced"
    n_jobs: int          = -1
    random_state: int    = 42

@dataclass
class XGBoostConfig:
    n_estimators: int    = 300
    max_depth: int       = 6
    learning_rate: float = 0.05
    subsample: float     = 0.8
    colsample_bytree: float = 0.8
    use_label_encoder: bool = False
    eval_metric: str     = "mlogloss"
    n_jobs: int          = -1
    random_state: int    = 42

@dataclass
class LightGBMConfig:
    n_estimators: int    = 300
    max_depth: int       = 6
    learning_rate: float = 0.05
    num_leaves: int      = 31
    subsample: float     = 0.8
    colsample_bytree: float = 0.8
    class_weight: str    = "balanced"
    n_jobs: int          = -1
    random_state: int    = 42
    verbose: int         = -1


# ─────────────────────────────────────────────
#  Abstract Base Model
# ─────────────────────────────────────────────

class BaseModel(ABC):
    """
    Abstract base class for all credit-scoring ML models.
    Subclasses must implement: build(), get_mlflow_logger().
    """

    def __init__(self, config):
        self.config      = config
        self.model_      = None
        self.model_name: str = "BaseModel"
        self.cv_scores_: Optional[np.ndarray] = None
        self.train_time_: float = 0.0

    # ── Abstract interface ─────────────────────
    @abstractmethod
    def build(self) -> Any:
        """Instantiate and return the sklearn-compatible estimator."""
        ...

    @abstractmethod
    def log_model_to_mlflow(self, artifact_path: str):
        """Log the fitted model to the active MLflow run."""
        ...

    # ── Concrete methods ───────────────────────
    def fit(self, X_train: np.ndarray, y_train: np.ndarray,
            cv: int = 5) -> "BaseModel":
        """
        Fit the model and run stratified k-fold CV for a preliminary
        performance estimate.
        """
        self.model_ = self.build()
        logger.info(f"[{self.model_name}] Starting training on "
                    f"{X_train.shape[0]:,} samples …")

        # Cross-validation (accuracy as quick sanity check)
        skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)
        self.cv_scores_ = cross_val_score(
            self.model_, X_train, y_train,
            cv=skf, scoring="accuracy", n_jobs=-1
        )
        logger.info(f"[{self.model_name}] CV accuracy: "
                    f"{self.cv_scores_.mean():.4f} ± {self.cv_scores_.std():.4f}")

        # Full fit on entire training set
        t0 = time.time()
        self.model_.fit(X_train, y_train)
        self.train_time_ = time.time() - t0
        logger.info(f"[{self.model_name}] Training complete in "
                    f"{self.train_time_:.2f}s")
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        self._check_fitted()
        return self.model_.predict(X)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        self._check_fitted()
        return self.model_.predict_proba(X)

    def get_params(self) -> Dict[str, Any]:
        """Return config as a flat dict for MLflow param logging."""
        return vars(self.config)

    def save(self, path: str):
        """Pickle the fitted model to disk."""
        self._check_fitted()
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self.model_, f)
        logger.info(f"[{self.model_name}] Model saved → {path}")

    def _check_fitted(self):
        if self.model_ is None:
            raise RuntimeError(f"{self.model_name} is not fitted yet.")


# ─────────────────────────────────────────────
#  Concrete Model Classes
# ─────────────────────────────────────────────

class LogisticRegressionModel(BaseModel):
    def __init__(self, config: Optional[LogisticRegressionConfig] = None):
        super().__init__(config or LogisticRegressionConfig())
        self.model_name = "LogisticRegression"

    def build(self) -> LogisticRegression:
        return LogisticRegression(**vars(self.config))

    def log_model_to_mlflow(self, artifact_path: str = "model"):
        mlflow.sklearn.log_model(self.model_, artifact_path)


class RandomForestModel(BaseModel):
    def __init__(self, config: Optional[RandomForestConfig] = None):
        super().__init__(config or RandomForestConfig())
        self.model_name = "RandomForest"

    def build(self) -> RandomForestClassifier:
        return RandomForestClassifier(**vars(self.config))

    def log_model_to_mlflow(self, artifact_path: str = "model"):
        mlflow.sklearn.log_model(self.model_, artifact_path)

    def get_feature_importances(self) -> np.ndarray:
        self._check_fitted()
        return self.model_.feature_importances_


class XGBoostModel(BaseModel):
    def __init__(self, config: Optional[XGBoostConfig] = None):
        super().__init__(config or XGBoostConfig())
        self.model_name = "XGBoost"

    def build(self) -> XGBClassifier:
        cfg = vars(self.config).copy()
        return XGBClassifier(**cfg)

    def log_model_to_mlflow(self, artifact_path: str = "model"):
        mlflow.xgboost.log_model(self.model_, artifact_path)

    def get_feature_importances(self) -> np.ndarray:
        self._check_fitted()
        return self.model_.feature_importances_


class LightGBMModel(BaseModel):
    def __init__(self, config: Optional[LightGBMConfig] = None):
        super().__init__(config or LightGBMConfig())
        self.model_name = "LightGBM"

    def build(self) -> LGBMClassifier:
        return LGBMClassifier(**vars(self.config))

    def log_model_to_mlflow(self, artifact_path: str = "model"):
        mlflow.lightgbm.log_model(self.model_, artifact_path)

    def get_feature_importances(self) -> np.ndarray:
        self._check_fitted()
        return self.model_.feature_importances_


# ─────────────────────────────────────────────
#  Model Registry  (factory helper)
# ─────────────────────────────────────────────

MODEL_REGISTRY: Dict[str, type] = {
    "logistic_regression": LogisticRegressionModel,
    "random_forest":       RandomForestModel,
    "xgboost":             XGBoostModel,
    "lightgbm":            LightGBMModel,
}

def get_model(name: str, config=None) -> BaseModel:
    """Factory: return an instantiated model by name."""
    name = name.lower()
    if name not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model '{name}'. "
                         f"Choose from: {list(MODEL_REGISTRY)}")
    return MODEL_REGISTRY[name](config)


# ─────────────────────────────────────────────
#  MLflow Training Orchestrator
# ─────────────────────────────────────────────

class TrainingPipeline:
    """
    Orchestrates training of one or more models, logs everything to
    MLflow, and tracks the best model by a chosen metric.

    Parameters
    ----------
    experiment_name : MLflow experiment label
    tracking_uri    : MLflow tracking server URI (local folder by default)
    cv_folds        : Number of stratified CV folds during fit()
    best_metric     : Metric to use for selecting the best model
                      ('accuracy', 'f1_macro', 'roc_auc_ovr')
    """

    def __init__(
        self,
        experiment_name: str = "CreditScoring",
        tracking_uri: str    = "./mlruns",
        cv_folds: int        = 5,
        best_metric: str     = "f1_macro",
    ):
        self.experiment_name = experiment_name
        self.tracking_uri    = tracking_uri
        self.cv_folds        = cv_folds
        self.best_metric     = best_metric

        self.results_: Dict[str, Dict] = {}   # model_name → metrics
        self.best_model_name_: Optional[str]  = None
        self.best_model_: Optional[BaseModel] = None

        # Configure MLflow — use SQLite backend
        # (MLflow ≥ 3.x deprecated the plain file-store)
        if self.tracking_uri.startswith("./") or not self.tracking_uri.startswith(("http", "sqlite")):
            db_path = os.path.join(os.path.dirname(self.tracking_uri) or ".", "mlflow.db")
            self.tracking_uri = f"sqlite:///{os.path.abspath(db_path)}"
        mlflow.set_tracking_uri(self.tracking_uri)
        mlflow.set_experiment(self.experiment_name)
        logger.info(f"MLflow tracking URI : {self.tracking_uri}")
        logger.info(f"MLflow experiment   : {self.experiment_name}")

    # ── run one model ──────────────────────────
    def run(
        self,
        model: BaseModel,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_test:  np.ndarray,
        y_test:  np.ndarray,
        feature_names: Optional[list] = None,
        extra_tags: Optional[Dict] = None,
    ) -> Dict[str, Any]:
        """
        Train *model*, evaluate on test set, log everything to MLflow.
        Returns a dict of evaluation metrics.
        """
        from src.evaluation import ModelEvaluator   # local import to avoid circular

        run_name = f"{model.model_name}_{int(time.time())}"

        with mlflow.start_run(run_name=run_name) as run:
            run_id = run.info.run_id
            logger.info(f"MLflow run started: {run_name}  (run_id={run_id})")

            # ── tags ──────────────────────────
            mlflow.set_tag("model_type", model.model_name)
            mlflow.set_tag("experiment", self.experiment_name)
            if extra_tags:
                for k, v in extra_tags.items():
                    mlflow.set_tag(k, v)

            # ── hyperparameters ───────────────
            mlflow.log_params(model.get_params())
            mlflow.log_param("cv_folds", self.cv_folds)

            # ── training ──────────────────────
            model.fit(X_train, y_train, cv=self.cv_folds)

            # log CV metrics
            mlflow.log_metric("cv_accuracy_mean", float(model.cv_scores_.mean()))
            mlflow.log_metric("cv_accuracy_std",  float(model.cv_scores_.std()))
            mlflow.log_metric("train_time_sec",   model.train_time_)

            # ── evaluation ────────────────────
            evaluator = ModelEvaluator(model, feature_names=feature_names)
            metrics   = evaluator.evaluate(X_test, y_test)

            # log every metric
            for metric_name, value in metrics.items():
                if isinstance(value, (int, float)):
                    mlflow.log_metric(metric_name, float(value))

            # log confusion matrix & classification report as artifacts
            evaluator.log_artifacts_to_mlflow(X_test, y_test,
                                               run_id=run_id)

            # ── model artifact ────────────────
            model.log_model_to_mlflow(artifact_path="model")

            # ── feature importance (if available) ─
            if hasattr(model, "get_feature_importances") and feature_names:
                importances = model.get_feature_importances()
                fi_path = f"/tmp/{model.model_name}_feature_importance.txt"
                with open(fi_path, "w") as f:
                    pairs = sorted(zip(feature_names, importances),
                                   key=lambda x: x[1], reverse=True)
                    for fname, imp in pairs:
                        f.write(f"{fname}: {imp:.6f}\n")
                mlflow.log_artifact(fi_path, artifact_path="feature_importance")

            logger.info(f"[{model.model_name}] Test metrics: "
                        f"accuracy={metrics.get('accuracy', 0):.4f}  "
                        f"f1_macro={metrics.get('f1_macro', 0):.4f}")

        # store result
        self.results_[model.model_name] = {
            "run_id":   run_id,
            "metrics":  metrics,
            "model":    model,
        }
        return metrics

    # ── run all models ─────────────────────────
    def run_all(
        self,
        model_names: list,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_test:  np.ndarray,
        y_test:  np.ndarray,
        feature_names: Optional[list] = None,
        configs: Optional[Dict] = None,
    ) -> Dict[str, Dict]:
        """
        Train a list of models sequentially, log each to MLflow,
        then elect the best one.
        """
        configs = configs or {}
        for name in model_names:
            logger.info(f"\n{'='*55}")
            logger.info(f"  Training model: {name.upper()}")
            logger.info(f"{'='*55}")
            model = get_model(name, configs.get(name))
            self.run(model, X_train, y_train, X_test, y_test,
                     feature_names=feature_names)

        self._elect_best_model()
        return self.results_

    # ── best model election ────────────────────
    def _elect_best_model(self):
        if not self.results_:
            return
        best_name  = max(
            self.results_,
            key=lambda n: self.results_[n]["metrics"].get(self.best_metric, 0)
        )
        best_score = self.results_[best_name]["metrics"].get(self.best_metric, 0)
        self.best_model_name_ = best_name
        self.best_model_      = self.results_[best_name]["model"]

        logger.info(f"\n★  Best model: {best_name}  "
                    f"({self.best_metric}={best_score:.4f})")

        # tag the winning run in MLflow
        best_run_id = self.results_[best_name]["run_id"]
        with mlflow.start_run(run_id=best_run_id):
            mlflow.set_tag("best_model", "true")
            mlflow.set_tag("selection_metric", self.best_metric)

    # ── summary ───────────────────────────────
    def summary(self) -> None:
        print(f"\n{'='*65}")
        print(f"  TRAINING SUMMARY  |  experiment: {self.experiment_name}")
        print(f"{'='*65}")
        header = f"{'Model':<22} {'Accuracy':>10} {'F1-Macro':>10} {'AUC-OVR':>10}"
        print(header)
        print("-" * 55)
        for name, res in self.results_.items():
            m = res["metrics"]
            star = " ★" if name == self.best_model_name_ else ""
            print(f"{name:<22} "
                  f"{m.get('accuracy', 0):>10.4f} "
                  f"{m.get('f1_macro', 0):>10.4f} "
                  f"{m.get('roc_auc_ovr', 0):>10.4f}"
                  f"{star}")
        print("="*65)
