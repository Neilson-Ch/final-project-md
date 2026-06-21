"""
Credit Scoring — Main Training Entrypoint
==========================================
Run this script to execute the full ML pipeline:
  1. Load & split data
  2. Preprocess (fit on train, transform test)
  3. Train 4 models with MLflow tracking
  4. Evaluate & compare all models
  5. Persist the best model to disk

Usage
-----
    cd /home/claude/credit_scoring
    python train.py                            # default settings
    python train.py --data data/data_A.csv
    python train.py --models xgboost lightgbm
    python train.py --experiment MyCreditExp --best_metric f1_macro
"""

import os
import sys
import argparse
import logging
import pickle

# Ensure src/ is importable
sys.path.insert(0, os.path.dirname(__file__))

import mlflow

from src.preprocessing   import CreditDataPreprocessor, DataLoader
from src.training        import (
    TrainingPipeline,
    LogisticRegressionConfig, RandomForestConfig,
    XGBoostConfig, LightGBMConfig,
)
from src.evaluation      import ComparativeEvaluator

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s — %(levelname)s — %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("training.log", mode="a"),
    ]
)
logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────
#  CLI
# ─────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(
        description="Credit Scoring ML Training Pipeline"
    )
    parser.add_argument("--data",        default="data/data_A.csv",
                        help="Path to raw CSV data file")
    parser.add_argument("--experiment",  default="CreditScoring",
                        help="MLflow experiment name")
    parser.add_argument("--tracking_uri",default="./mlruns",
                        help="MLflow tracking URI (local folder or server URL)")
    parser.add_argument("--models",      nargs="+",
                        default=["logistic_regression", "random_forest",
                                 "xgboost", "lightgbm"],
                        choices=["logistic_regression", "random_forest",
                                 "xgboost", "lightgbm"],
                        help="Which models to train")
    parser.add_argument("--best_metric", default="f1_macro",
                        choices=["accuracy", "f1_macro", "f1_weighted",
                                 "roc_auc_ovr", "roc_auc_ovo"],
                        help="Metric used to elect the best model")
    parser.add_argument("--test_size",   type=float, default=0.2)
    parser.add_argument("--cv_folds",    type=int,   default=5)
    parser.add_argument("--scaler",      default="standard",
                        choices=["standard", "minmax"])
    parser.add_argument("--random_seed", type=int,   default=42)
    return parser.parse_args()


# ─────────────────────────────────────────────
#  Custom hyperparameter configs (tune here)
# ─────────────────────────────────────────────

def build_configs(random_seed: int) -> dict:
    return {
        "logistic_regression": LogisticRegressionConfig(
            C=0.5, max_iter=2000, random_state=random_seed
        ),
        "random_forest": RandomForestConfig(
            n_estimators=300, max_depth=12, random_state=random_seed
        ),
        "xgboost": XGBoostConfig(
            n_estimators=400, max_depth=6,
            learning_rate=0.05, random_state=random_seed
        ),
        "lightgbm": LightGBMConfig(
            n_estimators=400, max_depth=6,
            learning_rate=0.05, random_state=random_seed
        ),
    }


# ─────────────────────────────────────────────
#  Main Pipeline
# ─────────────────────────────────────────────

def main():
    args = parse_args()
    logger.info("="*65)
    logger.info("  CREDIT SCORING ML PIPELINE — START")
    logger.info("="*65)
    logger.info(f"  Data file   : {args.data}")
    logger.info(f"  Experiment  : {args.experiment}")
    logger.info(f"  Models      : {args.models}")
    logger.info(f"  Best metric : {args.best_metric}")
    logger.info("="*65)

    # ── 1. Load data ──────────────────────────
    loader   = DataLoader(
        filepath=args.data,
        test_size=args.test_size,
        random_state=args.random_seed,
    )
    raw_df   = loader.load()
    train_df, test_df = loader.train_test_split(raw_df)

    # ── 2. Preprocess ─────────────────────────
    prep = CreditDataPreprocessor(scaler_method=args.scaler)
    X_train, y_train = prep.fit_transform(train_df)
    X_test,  y_test  = prep.transform(test_df)

    feature_names = prep.get_feature_names()
    logger.info(f"Features after preprocessing: {len(feature_names)}")
    logger.info(f"Train  : {X_train.shape}  |  Test : {X_test.shape}")

    # Persist preprocessor for inference reuse
    os.makedirs("models", exist_ok=True)
    with open("models/preprocessor.pkl", "wb") as f:
        pickle.dump(prep, f)
    logger.info("Preprocessor saved → models/preprocessor.pkl")

    # ── 3. Train all models ───────────────────
    pipeline = TrainingPipeline(
        experiment_name=args.experiment,
        tracking_uri=args.tracking_uri,
        cv_folds=args.cv_folds,
        best_metric=args.best_metric,
    )

    configs  = build_configs(args.random_seed)

    results  = pipeline.run_all(
        model_names=args.models,
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
        feature_names=feature_names,
        configs=configs,
    )

    # ── 4. Print summary ──────────────────────
    pipeline.summary()

    # ── 5. Comparative evaluation & comparison chart ──
    comp_eval = ComparativeEvaluator(results)
    summary_df = comp_eval.summary_dataframe()
    print("\n── Model Comparison Table ──")
    print(summary_df.to_string())
    comp_eval.log_comparison_to_mlflow(experiment_name=args.experiment)

    # ── 6. Persist best model ─────────────────
    best_name  = pipeline.best_model_name_
    best_model = pipeline.best_model_
    if best_model is not None:
        best_path = f"models/best_model_{best_name}.pkl"
        best_model.save(best_path)
        logger.info(f"\n★  Best model saved → {best_path}")

    logger.info("\n  PIPELINE COMPLETE.")
    return results


if __name__ == "__main__":
    main()
