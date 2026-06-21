from pathlib import Path
from typing import Tuple
import joblib
import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, MultiLabelBinarizer
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, classification_report


class CreditScorePreprocessor:
    """Handles raw dataset cleaning, feature engineering, and train/val/test split.

    Semua pembersihan data mentah (cell 2-66 di notebook) dilakukan di sini,
    SEBELUM split, karena langkah-langkah ini tidak bergantung pada distribusi
    train set (mis. strip karakter '_', filter umur tidak valid, mapping label).
    Imputasi nilai hilang (median per kolom) TIDAK dilakukan di sini — itu
    didelegasikan ke ColumnTransformer di CreditScoreModelTrainer agar fit
    hanya pada x_train dan tidak terjadi data leakage ke val/test.
    """

    SCORE_MAP = {"Poor": 0, "Standard": 1, "Good": 2}

    # Kolom yang di notebook awalnya bertipe object karena ada karakter '_'
    # (mis. "100433.58_") sehingga perlu dibersihkan jadi numerik.
    ERROR_COLS = [
        "Age", "Annual_Income", "Num_of_Loan", "Num_of_Delayed_Payment",
        "Outstanding_Debt", "Monthly_Balance", "Amount_invested_monthly",
        "Changed_Credit_Limit",
    ]

    # Urutan bulan untuk OrdinalEncoder (Month dipakai sebagai fitur numerik ordinal 1-8)
    MONTH_ORDER = ["January", "February", "March", "April", "May", "June", "July", "August"]

    def __init__(self, test_size: float = 0.2, val_size: float = 0.1, random_state: int = 42):
        self.test_size = test_size
        self.val_size = val_size
        self.random_state = random_state

    def _clean_raw(self, df: pd.DataFrame) -> pd.DataFrame:
        """Replikasi cell 2-66: drop identitas, fix tipe data, filter outlier, feature engineering."""
        df = df.copy()

        # Drop kolom identitas yang tidak berguna untuk model (cell 2)
        # Catatan: 'Month' TIDAK di-drop karena dipakai sebagai fitur ordinal (lihat ColumnTransformer)
        df = df.drop(
            columns=["Unnamed: 0", "ID", "Customer_ID", "Name", "SSN"],
            errors="ignore",
        )

        # Bersihkan kolom yang mengandung '_' lalu konversi ke numerik (cell 7)
        for col in self.ERROR_COLS:
            if col in df.columns:
                df[col] = df[col].astype(str).str.replace("_", "", regex=False).replace("", np.nan)
                df[col] = pd.to_numeric(df[col], errors="coerce")

        # Filter umur tidak valid (cell 13-15)
        df = df[(df["Age"] > 0) & (df["Age"] <= 100)]

        # Occupation kosong -> Unknown (cell 17)
        df["Occupation"] = df["Occupation"].replace("_______", "Unknown")

        # Filter jumlah rekening bank tidak wajar (cell 21-24)
        df = df[(df["Num_Bank_Accounts"] > 0) & (df["Num_Bank_Accounts"] <= 20)]

        # Filter jumlah kartu kredit tidak wajar (cell 26-29)
        df = df[df["Num_Credit_Card"] <= 30]

        # Interest rate > 100% dianggap invalid (cell 31)
        df.loc[df["Interest_Rate"] > 100, "Interest_Rate"] = np.nan

        # Num_of_Loan negatif dianggap invalid (cell 34-36)
        df.loc[df["Num_of_Loan"] < 0, "Num_of_Loan"] = np.nan

        # Feature engineering dari Type_of_Loan (cell 39-41)
        df["Loan_Not_Specified"] = (
            df["Type_of_Loan"].str.contains("Not Specified", na=False).astype(int)
        )
        df["Loan_List"] = df["Type_of_Loan"].apply(self._clean_loans)
        df["Num_Loan_Types"] = df["Loan_List"].apply(len)
        df["Loan_Info_Missing"] = df["Type_of_Loan"].isna().astype(int)

        # Delay_from_due_date negatif -> Paid_Early flag, lalu clip ke 0 (cell 45-47)
        df["Paid_Early"] = (df["Delay_from_due_date"] < 0).astype(int)
        df["Delay_from_due_date"] = df["Delay_from_due_date"].clip(lower=0)

        # Num_of_Delayed_Payment negatif dianggap invalid (cell 48)
        df.loc[df["Num_of_Delayed_Payment"] < 0, "Num_of_Delayed_Payment"] = np.nan

        # Credit_Mix '-' -> Unknown (cell 53)
        df["Credit_Mix"] = df["Credit_Mix"].replace("-", "Unknown")

        # Parse "X Years and Y Months" jadi total bulan (cell 56)
        temp = df["Credit_History_Age"].str.extract(r"(\d+)\s+Years?\s+and\s+(\d+)\s+Months?")
        df["Credit_History_Months"] = temp[0].astype(float) * 12 + temp[1].astype(float)
        df = df.drop(columns=["Credit_History_Age"])

        # Payment_Behaviour placeholder '!@9#%8' -> Unknown (cell 63)
        df["Payment_Behaviour"] = df["Payment_Behaviour"].replace("!@9#%8", "Unknown")

        # Monthly_Balance negatif dianggap invalid (cell 65)
        df.loc[df["Monthly_Balance"] < 0, "Monthly_Balance"] = np.nan

        # Mapping target ke ordinal 0/1/2 (cell 67)
        df["Credit_Score"] = df["Credit_Score"].map(self.SCORE_MAP)
        df = df.dropna(subset=["Credit_Score"])
        df["Credit_Score"] = df["Credit_Score"].astype(int)

        return df

    @staticmethod
    def _clean_loans(text):
        """Replikasi clean_loans() cell 40: parse string Type_of_Loan jadi list unik."""
        if pd.isna(text):
            return []
        text = text.replace(", and ", ", ").replace(" and ", ", ")
        loans = [loan.strip() for loan in text.split(",")]
        loans = [loan for loan in loans if loan not in ["", "Not Specified"]]
        return list(dict.fromkeys(loans))

    def clean_single_record(self, raw: dict) -> pd.DataFrame:
        """Versi clean_raw untuk SATU record (inference), bukan batch training.

        Bedanya dengan _clean_raw: tidak ada baris yang di-drop (df = df[mask]),
        karena untuk satu input itu akan menghasilkan DataFrame kosong. Nilai
        di luar rentang wajar di-clip/dianggap NaN agar tetap bisa diproses oleh
        SimpleImputer di pipeline, alih-alih ditolak.
        """
        df = pd.DataFrame([raw])

        df = df.drop(columns=["Unnamed: 0", "ID", "Customer_ID", "Name", "SSN"], errors="ignore")

        for col in self.ERROR_COLS:
            if col in df.columns:
                df[col] = df[col].astype(str).str.replace("_", "", regex=False).replace("", np.nan)
                df[col] = pd.to_numeric(df[col], errors="coerce")

        # Age tidak wajar -> NaN (bukan drop baris)
        df.loc[(df["Age"] <= 0) | (df["Age"] > 100), "Age"] = np.nan

        df["Occupation"] = df["Occupation"].replace("_______", "Unknown")

        df.loc[(df["Num_Bank_Accounts"] <= 0) | (df["Num_Bank_Accounts"] > 20), "Num_Bank_Accounts"] = np.nan
        df.loc[df["Num_Credit_Card"] > 30, "Num_Credit_Card"] = np.nan
        df.loc[df["Interest_Rate"] > 100, "Interest_Rate"] = np.nan
        df.loc[df["Num_of_Loan"] < 0, "Num_of_Loan"] = np.nan

        df["Loan_Not_Specified"] = df["Type_of_Loan"].str.contains("Not Specified", na=False).astype(int)
        df["Loan_List"] = df["Type_of_Loan"].apply(self._clean_loans)
        df["Num_Loan_Types"] = df["Loan_List"].apply(len)
        df["Loan_Info_Missing"] = df["Type_of_Loan"].isna().astype(int)

        df["Paid_Early"] = (df["Delay_from_due_date"] < 0).astype(int)
        df["Delay_from_due_date"] = df["Delay_from_due_date"].clip(lower=0)

        df.loc[df["Num_of_Delayed_Payment"] < 0, "Num_of_Delayed_Payment"] = np.nan
        df["Credit_Mix"] = df["Credit_Mix"].replace("-", "Unknown")

        temp = df["Credit_History_Age"].str.extract(r"(\d+)\s+Years?\s+and\s+(\d+)\s+Months?")
        df["Credit_History_Months"] = temp[0].astype(float) * 12 + temp[1].astype(float)
        df = df.drop(columns=["Credit_History_Age"])

        df["Payment_Behaviour"] = df["Payment_Behaviour"].replace("!@9#%8", "Unknown")
        df.loc[df["Monthly_Balance"] < 0, "Monthly_Balance"] = np.nan

        return df

    def clean_and_split(
        self, data_path: str | Path
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.Series]:
        """Load CSV mentah, bersihkan, lalu split jadi train/val/test (80/10/10, stratified)."""
        df = pd.read_csv(Path(data_path))
        df = self._clean_raw(df)

        X = df.drop(columns=["Credit_Score"])
        y = df["Credit_Score"]

        x_train, x_temp, y_train, y_temp = train_test_split(
            X, y, test_size=(self.test_size + self.val_size),
            random_state=self.random_state, stratify=y,
        )
        val_ratio = self.val_size / (self.test_size + self.val_size)
        x_val, x_test, y_val, y_test = train_test_split(
            x_temp, y_temp, test_size=(1 - val_ratio),
            random_state=self.random_state, stratify=y_temp,
        )

        return x_train, x_val, x_test, y_train, y_val, y_test

    def get_transformer(self, x_train: pd.DataFrame) -> ColumnTransformer:
        """ColumnTransformer untuk median imputation (numerik) + OHE (kategorikal)
        + OrdinalEncoder khusus Month (urutan kalender, bukan one-hot).

        Loan_List/Type_of_Loan TIDAK dimasukkan di sini karena di-encode terpisah
        dengan MultiLabelBinarizer (lihat encode_loan_types), sesuai notebook cell 83-85.
        Month juga dikeluarkan dari num_feat karena masih bertipe string (nama bulan)
        pada titik ini dan ditangani lewat encoder ordinal-nya sendiri.
        """
        exclude_cols = {"Type_of_Loan", "Loan_List", "Month"}
        num_feat = [
            c for c in x_train.select_dtypes(include=["int64", "float64"]).columns
            if c not in exclude_cols
        ]
        cat_feat = ["Occupation", "Payment_Behaviour", "Credit_Mix", "Payment_of_Min_Amount"]

        numeric_pipeline = Pipeline([
            ("num_imputer", SimpleImputer(strategy="median")),
        ])

        month_pipeline = Pipeline([
            ("month_imputer", SimpleImputer(strategy="most_frequent")),
            ("month_encoder", OrdinalEncoder(
                categories=[self.MONTH_ORDER],
                handle_unknown="use_encoded_value",
                unknown_value=-1,
            )),
        ])

        categorical_pipeline = Pipeline([
            ("cat_imputer", SimpleImputer(strategy="most_frequent")),
            ("cat_encoder", OneHotEncoder(drop="first", handle_unknown="ignore")),
        ])

        return ColumnTransformer(transformers=[
            ("numPreprocess", numeric_pipeline, num_feat),
            ("catPreprocess", categorical_pipeline, cat_feat),
            ("monthPreprocess", month_pipeline, ["Month"]),
        ], remainder="drop")

    @staticmethod
    def encode_loan_types(
        x_train: pd.DataFrame, x_val: pd.DataFrame, x_test: pd.DataFrame
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, MultiLabelBinarizer]:
        """MultiLabelBinarizer untuk Loan_List, fit HANYA pada x_train (cell 83-85).

        Dilakukan di luar ColumnTransformer karena MultiLabelBinarizer tidak
        kompatibel dengan antarmuka transformer kolom tunggal sklearn.
        mlb yang sudah di-fit dikembalikan agar bisa disimpan sebagai artifact
        terpisah dan dipakai ulang secara konsisten saat inference.
        """
        mlb = MultiLabelBinarizer()
        mlb.fit(x_train["Loan_List"])

        def _encode(data: pd.DataFrame) -> pd.DataFrame:
            encoded = pd.DataFrame(
                mlb.transform(data["Loan_List"]), columns=mlb.classes_, index=data.index,
            )
            out = pd.concat([data, encoded], axis=1)
            return out.drop(columns=["Type_of_Loan", "Loan_List"])

        return _encode(x_train), _encode(x_val), _encode(x_test), mlb

    @staticmethod
    def transform_loan_types(df: pd.DataFrame, mlb: MultiLabelBinarizer) -> pd.DataFrame:
        """Terapkan mlb yang SUDAH di-fit (dari training) ke data baru saat inference.

        Tidak melakukan fit ulang -> konsisten dengan kolom loan yang dipelajari model.
        """
        encoded = pd.DataFrame(
            mlb.transform(df["Loan_List"]), columns=mlb.classes_, index=df.index,
        )
        out = pd.concat([df, encoded], axis=1)
        return out.drop(columns=["Type_of_Loan", "Loan_List"])


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

    def run(self, data_path: str | Path) -> Tuple[str, pd.DataFrame, pd.Series]:
        x_train, x_val, x_test, y_train, y_val, y_test = self.preprocessor.clean_and_split(data_path)

        # MultiLabelBinarizer untuk Type_of_Loan, fit hanya di train (cell 83-85)
        x_train, x_val, x_test, mlb = self.preprocessor.encode_loan_types(x_train, x_val, x_test)

        transformer = self.preprocessor.get_transformer(x_train)

        # Random Forest dengan hyperparameter tuning sesuai cell 101 notebook
        credit_pred_pipeline = Pipeline([
            ("preprocessing", transformer),
            ("classifier", RandomForestClassifier(
                n_estimators=300,
                max_depth=None,
                random_state=self.random_state,
                n_jobs=-1,
            )),
        ])

        with mlflow.start_run() as run:
            mlflow.log_param("n_estimators", 300)
            mlflow.log_param("max_depth", "None")
            mlflow.log_param("model", "RandomForestClassifier")

            credit_pred_pipeline.fit(x_train, y_train)

            # Evaluasi di validation set, dicatat ke MLflow
            val_pred = credit_pred_pipeline.predict(x_val)
            val_acc = accuracy_score(y_val, val_pred)
            val_f1 = f1_score(y_val, val_pred, average="weighted")
            mlflow.log_metric("val_accuracy", val_acc)
            mlflow.log_metric("val_f1_weighted", val_f1)
            print(f"Validation  -> Accuracy: {val_acc:.4f} | F1 (weighted): {val_f1:.4f}")
            print(classification_report(y_val, val_pred, target_names=["Poor", "Standard", "Good"]))

            # Local save: pipeline utama + mlb (dibutuhkan saat inference karena
            # encode_loan_types dijalankan manual di luar sklearn Pipeline)
            model_file_path = self.artifact_dir / "credit_score_pipeline.pkl"
            mlb_file_path = self.artifact_dir / "loan_type_mlb.pkl"
            joblib.dump(credit_pred_pipeline, model_file_path)
            joblib.dump(mlb, mlb_file_path)
            mlflow.sklearn.log_model(
                credit_pred_pipeline,
                name="model",
                serialization_format=mlflow.sklearn.SERIALIZATION_FORMAT_PICKLE,
            )
            mlflow.log_artifact(str(mlb_file_path))

            print(f"✅ Model trained & saved locally to {model_file_path}")
            print(f"✅ Loan-type MultiLabelBinarizer saved to {mlb_file_path}")
            return run.info.run_id, x_test, y_test


if __name__ == "__main__":
    trainer = CreditScoreModelTrainer()
    run_id, x_test, y_test = trainer.run("data_A.csv")

    # Evaluasi final di test set (load ulang dari artifact, simulasi pipeline produksi)
    pipeline = joblib.load(trainer.artifact_dir / "credit_score_pipeline.pkl")
    y_pred_test = pipeline.predict(x_test)

    print("\n=== EVALUASI FINAL: Random Forest pada Test Set ===")
    print(f"Accuracy     : {accuracy_score(y_test, y_pred_test):.4f}")
    print(f"F1 (weighted): {f1_score(y_test, y_pred_test, average='weighted'):.4f}")
    print(classification_report(y_test, y_pred_test, target_names=["Poor", "Standard", "Good"]))
