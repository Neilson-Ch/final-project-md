"""
inference.py
=============
Modul inferencing untuk model Credit Score Prediction.

Membungkus pipeline (.pkl) hasil training + MultiLabelBinarizer (.pkl) untuk
Type_of_Loan ke dalam satu class `CreditScorePredictor` yang menerima input
mentah (dict, sama seperti satu baris di CSV asli) dan mengembalikan hasil
prediksi siap pakai (label, probabilitas per kelas).

Dipakai oleh:
- app.py (Streamlit UI)
- api.py (FastAPI endpoint /predict)
- test_inference.py (automated test case per kelas)
"""

from pathlib import Path
import joblib
import pandas as pd

from train import CreditScorePreprocessor


class CreditScorePredictor:
    """Wrapper inferencing: raw dict -> cleaning -> encoding -> prediksi."""

    LABEL_MAP = {0: "Poor", 1: "Standard", 2: "Good"}

    # Kolom yang WAJIB ada di input mentah (sebelum cleaning), sesuai skema data_A.csv
    REQUIRED_FIELDS = [
        "Month", "Age", "Occupation", "Annual_Income", "Monthly_Inhand_Salary",
        "Num_Bank_Accounts", "Num_Credit_Card", "Interest_Rate", "Num_of_Loan",
        "Type_of_Loan", "Delay_from_due_date", "Num_of_Delayed_Payment",
        "Changed_Credit_Limit", "Num_Credit_Inquiries", "Credit_Mix",
        "Outstanding_Debt", "Credit_Utilization_Ratio", "Credit_History_Age",
        "Payment_of_Min_Amount", "Total_EMI_per_month", "Amount_invested_monthly",
        "Payment_Behaviour", "Monthly_Balance",
    ]

    def __init__(self, artifact_dir: str | Path = "artifacts"):
        artifact_dir = Path(artifact_dir)
        pipeline_path = artifact_dir / "credit_score_pipeline.pkl"
        mlb_path = artifact_dir / "loan_type_mlb.pkl"

        if not pipeline_path.exists():
            raise FileNotFoundError(
                f"Pipeline tidak ditemukan di {pipeline_path}. Jalankan training "
                f"(credit_score_pipeline.py) terlebih dahulu."
            )
        if not mlb_path.exists():
            raise FileNotFoundError(
                f"MultiLabelBinarizer tidak ditemukan di {mlb_path}. Jalankan training "
                f"terlebih dahulu agar artifact loan_type_mlb.pkl dibuat."
            )
       
        self.pipeline = joblib.load(pipeline_path)
        self.mlb = joblib.load(mlb_path)
        self.preprocessor = CreditScorePreprocessor()

    def _validate(self, raw: dict) -> None:
        missing = [f for f in self.REQUIRED_FIELDS if f not in raw]
        if missing:
            raise ValueError(f"Field input berikut wajib diisi: {missing}")

        # Validasi range di sini, supaya tidak diam-diam ke-filter habis oleh _clean_raw
        if not (0 < raw["Age"] <= 100):
            raise ValueError("Age harus di antara 1-100")
        if not (0 < raw["Num_Bank_Accounts"] <= 20):
            raise ValueError("Num_Bank_Accounts harus di antara 1-20")
        if raw["Num_Credit_Card"] > 30:
            raise ValueError("Num_Credit_Card maksimal 30")
        
    def _encode_loan_types(self, df: pd.DataFrame) -> pd.DataFrame:
        """Pengganti preprocessor.transform_loan_types (tidak ada di train.py).
        Logic-nya sama persis dengan _encode() di dalam encode_loan_types(),
        tapi pakai MLB yang SUDAH fit (self.mlb) -> transform saja, bukan fit ulang."""
        encoded = pd.DataFrame(
            self.mlb.transform(df["Loan_List"]),
            columns=self.mlb.classes_,
            index=df.index,
        )
        out = pd.concat([df, encoded], axis=1)
        return out.drop(columns=["Type_of_Loan", "Loan_List"], errors="ignore")

    def _prepare(self, raw: dict) -> pd.DataFrame:
        """Replikasi urutan transformasi training, sepenuhnya di sisi inference.py
        (train.py tidak diubah / tidak perlu retrain)."""
        self._validate(raw)

        record = dict(raw)
        # _clean_raw() di train.py mengharuskan kolom Credit_Score ada (karena
        # ditulis untuk dataset training). Kita kasih dummy valid value supaya
        # tidak KeyError, lalu buang lagi -> tidak pernah ikut jadi fitur prediksi.
        record["Credit_Score"] = "Standard"

        df = pd.DataFrame([record])
        df = self.preprocessor._clean_raw(df)
        df = df.drop(columns=["Credit_Score"], errors="ignore")
        df = self._encode_loan_types(df)
        return df

    def predict(self, raw: dict) -> dict:
        """Prediksi satu record. Mengembalikan dict siap dijadikan JSON."""
        
        df = self._prepare(raw)

        pred_code = int(self.pipeline.predict(df)[0])
        proba = self.pipeline.predict_proba(df)[0]

        # Urutan kolom proba mengikuti self.pipeline.classes_, bukan diasumsikan 0,1,2
        proba_dict = {
            self.LABEL_MAP[int(cls)]: float(p)
            for cls, p in zip(self.pipeline.classes_, proba)
        }

        return {
            "prediction": self.LABEL_MAP[pred_code],
            "prediction_code": pred_code,
            "probability": proba_dict,
        }

    def predict_batch(self, raw_list: list[dict]) -> list[dict]:
        """Prediksi banyak record sekaligus. Tiap dict divalidasi & diproses satu-satu
        agar pesan error tetap jelas menunjuk record mana yang bermasalah."""
        return [self.predict(raw) for raw in raw_list]


if __name__ == "__main__":
    # Quick smoke test manual
    predictor = CreditScorePredictor(artifact_dir="artifacts")
    sample = {
        "Month": "January", "Age": 30, "Occupation": "Engineer",
        "Annual_Income": "50000.0", "Monthly_Inhand_Salary": 4000.0,
        "Num_Bank_Accounts": 3, "Num_Credit_Card": 4, "Interest_Rate": 12,
        "Num_of_Loan": "2", "Type_of_Loan": "Auto Loan, and Personal Loan",
        "Delay_from_due_date": 5, "Num_of_Delayed_Payment": "3",
        "Changed_Credit_Limit": "5.5", "Num_Credit_Inquiries": 2.0,
        "Credit_Mix": "Good", "Outstanding_Debt": "500.0",
        "Credit_Utilization_Ratio": 30.0, "Credit_History_Age": "10 Years and 3 Months",
        "Payment_of_Min_Amount": "Yes", "Total_EMI_per_month": 100.0,
        "Amount_invested_monthly": "50.0", "Payment_Behaviour": "Low_spent_Small_value_payments",
        "Monthly_Balance": "300.0",
    }
    print(predictor.predict(sample))
