from pathlib import Path
from typing import Tuple
import joblib
import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
from sklearn.pipeline import Pipeline
from preprocess import CreditScorePreprocessor
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, classification_report
 
class CreditScoreModelTrainer:
    """Handles pipeline assembly, training, evaluation, and artifact tracking."""
 
    def __init__(
        self,
        experiment_name: str = "Credit Score Prediction",
        artifact_path: str = "artifacts",
        random_state: int = 42,
    ):
        self.experiment_name = experiment_name
        self.artifact_dir = Path(artifact_path)
        self.preprocessor = CreditScorePreprocessor(random_state=random_state)
        self.random_state = random_state
 
        self.artifact_dir.mkdir(parents=True, exist_ok=True)
        mlflow.set_experiment(self.experiment_name)
 
    def _get_candidate_models(self) -> dict:
        """Phase 1: 4 model dengan default params."""
        return {
            "RandomForest": RandomForestClassifier(random_state=self.random_state, n_jobs=-1),
            "XGBoost": XGBClassifier(eval_metric="mlogloss", random_state=self.random_state),
            "CatBoost": CatBoostClassifier(verbose=0, random_state=self.random_state),
            "LightGBM": LGBMClassifier(random_state=self.random_state),
        }

    def _get_tuning_configs(self) -> dict:
        """Phase 2: param grid hanya untuk RF, XGBoost, CatBoost.
        LightGBM tidak ada config-nya -> kalau LGBM menang Phase 1, skip tuning."""
        return {
            "RandomForest": {
                "model_class": RandomForestClassifier,
                 "param_grid": [
                    {
                        "max_depth": None,
                        "min_samples_leaf": 1,
                        "min_samples_split": 2,
                        "n_estimators": 100,
                    }
                ],
                "fixed_params": {"random_state": self.random_state, "n_jobs": -1},
            },
            "XGBoost": {
                "model_class": XGBClassifier,
                "param_grid": [
                    {"n_estimators": 100, "max_depth": 4},
                    {"n_estimators": 200, "max_depth": 6},
                    {"n_estimators": 300, "max_depth": 8},
                ],
                "fixed_params": {"eval_metric": "mlogloss", "random_state": self.random_state},
            },
        }

    def _tune_model(
        self, name: str, transformer, x_train, y_train, x_val, y_val
    ) -> Tuple[Pipeline, float, str] | None:
        """Tune model `name` dengan param grid-nya.
        Return None kalau model tidak punya tuning config (e.g. LightGBM)."""
        configs = self._get_tuning_configs()

        if name not in configs:
            print(f"⚠️  {name} tidak punya tuning config -> skip tuning, pakai hasil Phase 1.")
            return None

        config = configs[name]
        best_pipeline, best_score, best_run_id = None, -1.0, None

        print(f"\n=== Phase 2: Tuning {name} ===")
        for params in config["param_grid"]:
            all_params = {**params, **config["fixed_params"]}
            pipeline = Pipeline([
                ("preprocessing", transformer),
                ("classifier", config["model_class"](**all_params)),
            ])

            run_label = "_".join(f"{k}{v}" for k, v in params.items())
            with mlflow.start_run(run_name=f"tune_{name}_{run_label}") as run:
                mlflow.log_param("phase", "hyperparameter_tuning")
                mlflow.log_param("model", name)
                mlflow.log_params(params)

                pipeline.fit(x_train, y_train)

                val_pred = pipeline.predict(x_val)
                val_f1  = f1_score(y_val, val_pred, average="weighted")
                val_acc = accuracy_score(y_val, val_pred)
                mlflow.log_metric("val_f1_weighted", val_f1)
                mlflow.log_metric("val_accuracy", val_acc)
                print(f"  {params} -> Val F1: {val_f1:.4f}")

                if val_f1 > best_score:
                    best_score    = val_f1
                    best_pipeline = pipeline
                    best_run_id   = run.info.run_id

        # Tandai run terbaik hasil tuning di MLflow
        with mlflow.start_run(run_id=best_run_id):
            mlflow.set_tag("best_model", "true")
            mlflow.set_tag("selected_by", "val_f1_weighted")

        return best_pipeline, best_score, best_run_id


    def run(self, data_path: str | Path) -> Tuple[str, pd.DataFrame, pd.Series]:
        x_train, x_val, x_test, y_train, y_val, y_test = self.preprocessor.clean_and_split(data_path)
        x_train, x_val, x_test, mlb = self.preprocessor.encode_loan_types(x_train, x_val, x_test)
        transformer = self.preprocessor.get_transformer(x_train)

        # ── Phase 1: bandingkan 4 model default ───────────────────────────────
        print("=== Phase 1: Model Selection (default params) ===")
        best_p1_score, best_p1_name = -1.0, None
        best_p1_pipeline, best_p1_run_id = None, None

        for name, classifier in self._get_candidate_models().items():
            pipeline = Pipeline([
                ("preprocessing", transformer),
                ("classifier", classifier),
            ])
            with mlflow.start_run(run_name=f"phase1_{name}") as run:
                mlflow.log_param("phase", "model_selection")
                mlflow.log_param("model", name)
                pipeline.fit(x_train, y_train)

                val_pred = pipeline.predict(x_val)
                val_f1  = f1_score(y_val, val_pred, average="weighted")
                val_acc = accuracy_score(y_val, val_pred)
                mlflow.log_metric("val_f1_weighted", val_f1)
                mlflow.log_metric("val_accuracy", val_acc)
                print(f"{name:<12} -> Val F1: {val_f1:.4f} | Val Acc: {val_acc:.4f}")

                if val_f1 > best_p1_score:
                    best_p1_score    = val_f1
                    best_p1_name     = name
                    best_p1_pipeline = pipeline
                    best_p1_run_id   = run.info.run_id

        print(f"\n🏆 Terpilih: {best_p1_name} (Val F1={best_p1_score:.4f}) -> lanjut tuning")

        # ── Phase 2: tune model terpilih ──────────────────────────────────────
        if best_p1_name == "RandomForest":
            tune_result = self._tune_model(
                name="RandomForest",
                transformer=transformer,
                x_train=x_train,
                y_train=y_train,
                x_val=x_val,
                y_val=y_val,
            )

            tuned_pipeline, tuned_score, tuned_run_id = tune_result

            # Jangan gunakan hasil tuning jika performanya lebih buruk
            if tuned_score >= best_p1_score:
                final_pipeline = tuned_pipeline
                final_score = tuned_score
                final_run_id = tuned_run_id

                print(
                    f"✅ Final Random Forest hasil tuning "
                    f"(Val F1={final_score:.4f})"
                )
            else:
                final_pipeline = best_p1_pipeline
                final_score = best_p1_score
                final_run_id = best_p1_run_id

                print(
                    f"⚠️ Tuning tidak meningkatkan performa. "
                    f"Menggunakan Random Forest baseline "
                    f"(Val F1={final_score:.4f})"
                )

        else:
            # Jika model terbaik ternyata bukan Random Forest,
            # jangan tuning model lain.
            final_pipeline = best_p1_pipeline
            final_score = best_p1_score
            final_run_id = best_p1_run_id

            print(
                f"ℹ️ Best model adalah {best_p1_name}. "
                "Tuning Random Forest dilewati."
            )
            
        if tune_result is not None:
            final_pipeline, final_score, final_run_id = tune_result
            print(f"✅ Setelah tuning: {best_p1_name} Val F1={final_score:.4f}")
        else:
            # LightGBM menang Phase 1 -> tidak ada tuning config, pakai hasil Phase 1
            final_pipeline = best_p1_pipeline
            final_score    = best_p1_score
            final_run_id   = best_p1_run_id
            print(f"✅ Tanpa tuning: {best_p1_name} Val F1={final_score:.4f}")

        # ── Simpan artifact final ─────────────────────────────────────────────
        joblib.dump(final_pipeline, self.artifact_dir / "credit_score_pipeline.pkl")
        joblib.dump(mlb,            self.artifact_dir / "loan_type_mlb.pkl")

        # Log model ke run yang benar supaya eval.py bisa load pakai run_id
        with mlflow.start_run(run_id=final_run_id):
            mlflow.sklearn.log_model(
                final_pipeline, name="model",
                serialization_format=mlflow.sklearn.SERIALIZATION_FORMAT_PICKLE,
            )

        print(f"\n💾 Model final disimpan ke {self.artifact_dir}")
        return final_run_id, x_test, y_test
