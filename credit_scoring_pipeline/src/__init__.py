"""
Credit Scoring ML Pipeline
============================
src package — exposes the main public classes.
"""

from src.preprocessing import (
    CreditDataPreprocessor,
    DataLoader,
    LABEL_MAP,
    TARGET_COLUMN,
)
from src.training import (
    TrainingPipeline,
    get_model,
    LogisticRegressionConfig,
    RandomForestConfig,
    XGBoostConfig,
    LightGBMConfig,
)
from src.evaluation import (
    ModelEvaluator,
    ComparativeEvaluator,
    MetricsCalculator,
)

__all__ = [
    "CreditDataPreprocessor",
    "DataLoader",
    "LABEL_MAP",
    "TARGET_COLUMN",
    "TrainingPipeline",
    "get_model",
    "LogisticRegressionConfig",
    "RandomForestConfig",
    "XGBoostConfig",
    "LightGBMConfig",
    "ModelEvaluator",
    "ComparativeEvaluator",
    "MetricsCalculator",
]
