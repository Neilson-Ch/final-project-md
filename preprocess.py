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
from sklearn.preprocessing import OneHotEncoder, MultiLabelBinarizer, OrdinalEncoder
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from sklearn.metrics import accuracy_score, f1_score, classification_report

class CreditScorePreprocessor:
    """Handles raw dataset cleaning, feature engineering, and train/val/test split."""
 
    SCORE_MAP = {"Poor": 0, "Standard": 1, "Good": 2}
 
    MONTH_ORDER = ["January", "February", "March", "April", "May", "June", "July", "August"]
    
    # type errors
    ERROR_COLS = [
        "Age", "Annual_Income", "Num_of_Loan", "Num_of_Delayed_Payment",
        "Outstanding_Debt", "Monthly_Balance", "Amount_invested_monthly",
        "Changed_Credit_Limit",
    ]
 
    def __init__(self, test_size: float = 0.2, val_size: float = 0.1, random_state: int = 42):
        self.test_size = test_size
        self.val_size = val_size
        self.random_state = random_state
 
    def _clean_raw(self, df: pd.DataFrame) -> pd.DataFrame:
        """Replikasi cell 2-66: drop identitas, fix tipe data, filter outlier, feature engineering."""
        df = df.copy()
 
        # Drop kolom 
        df = df.drop(
            columns=["Unnamed: 0", "ID", "Customer_ID", "Name", "SSN"],
            errors="ignore",
        )
 
        # Bersihkan kolom yang mengandung '_'
        for col in self.ERROR_COLS:
            if col in df.columns:
                df[col] = df[col].astype(str).str.replace("_", "", regex=False).replace("", np.nan)
                df[col] = pd.to_numeric(df[col], errors="coerce")
 
        # Filter umur tidak valid 
        df = df[(df["Age"] > 0) & (df["Age"] <= 100)]
 
        # Occupation kosong 
        df["Occupation"] = df["Occupation"].replace("_______", "Unknown")
 
        # Filter jumlah rekening bank tidak wajar 
        df = df[(df["Num_Bank_Accounts"] > 0) & (df["Num_Bank_Accounts"] <= 20)]
 
        # Filter jumlah kartu kredit tidak wajar
        df = df[df["Num_Credit_Card"] <= 30]
 
        # Interest rate > 100% 
        df.loc[df["Interest_Rate"] > 100, "Interest_Rate"] = np.nan
 
        # Num_of_Loan negatif dianggap invalid
        df.loc[df["Num_of_Loan"] < 0, "Num_of_Loan"] = np.nan
 
        # Feature engineering dari Type_of_Loan
        # df["Loan_Not_Specified"] = (
        #     df["Type_of_Loan"].str.contains("Not Specified", na=False).astype(int)
        # )
        df["Loan_List"] = df["Type_of_Loan"].apply(self._clean_loans)
        df["Num_Loan_Types"] = df["Loan_List"].apply(len)
        df["Loan_Info_Missing"] = df["Type_of_Loan"].isna().astype(int)
 
        # Delay_from_due_date negatif -> Paid_Early flag, lalu clip ke 0
        df["Paid_Early"] = (df["Delay_from_due_date"] < 0).astype(int)
        df["Delay_from_due_date"] = df["Delay_from_due_date"].clip(lower=0)
 
        # Num_of_Delayed_Payment negatif dianggap invalid
        df.loc[df["Num_of_Delayed_Payment"] < 0, "Num_of_Delayed_Payment"] = np.nan
 
        # Credit_Mix '-'
        df["Credit_Mix"] = df["Credit_Mix"].replace("-", "Unknown")
 
        # Parse "X Years and Y Months" jadi total months
        temp = df["Credit_History_Age"].str.extract(r"(\d+)\s+Years?\s+and\s+(\d+)\s+Months?")
        df["Credit_History_Months"] = temp[0].astype(float) * 12 + temp[1].astype(float)
        df = df.drop(columns=["Credit_History_Age"])
 
        # Payment_Behaviour'!@9#%8'
        df["Payment_Behaviour"] = df["Payment_Behaviour"].replace("!@9#%8", "Unknown")
 
        # Monthly_Balance negatif
        df.loc[df["Monthly_Balance"] < 0, "Monthly_Balance"] = np.nan
 
        # Mapping target ke ordinal
        df["Credit_Score"] = df["Credit_Score"].map(self.SCORE_MAP)
        df = df.dropna(subset=["Credit_Score"])
        df["Credit_Score"] = df["Credit_Score"].astype(int)
 
        return df
 
    @staticmethod
    def _clean_loans(text):
        """parse string Type_of_Loan jadi list unik."""
        if pd.isna(text):
            return []
        text = text.replace(", and ", ", ").replace(" and ", ", ")
        loans = [loan.strip() for loan in text.split(",")]
        loans = [loan for loan in loans if loan not in ["", "Not Specified"]]
        return list(dict.fromkeys(loans))
 
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
        """ColumnTransformer untuk median imputation (numerik) + OHE (kategorikal)."""
        
        exclude_cols = {"Type_of_Loan", "Loan_List"}
        num_feat = [
            c for c in x_train.select_dtypes(include=["int64", "float64"]).columns
            if c not in exclude_cols
        ]
        cat_feat = ["Occupation", "Payment_Behaviour", "Payment_of_Min_Amount"]
 
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
        
        credit_mix_pipeline = Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OrdinalEncoder(
                categories=[["Unknown", "Bad", "Standard", "Good"]],
                handle_unknown="use_encoded_value",
                unknown_value=-1,
            )),
        ])
        
        categorical_pipeline = Pipeline([
            ("cat_imputer", SimpleImputer(strategy="most_frequent")),
            ("cat_encoder", OneHotEncoder(drop="first", handle_unknown="ignore"))
        ])
        
        return ColumnTransformer(transformers=[
            ("numPreprocess", numeric_pipeline, num_feat),
            ("catPreprocess", categorical_pipeline, cat_feat),
            ("monthPreprocess", month_pipeline, ["Month"]),
            ("credPreprocess", credit_mix_pipeline, ["Credit_Mix"]),
        ], remainder="drop")
 
    @staticmethod
    def encode_loan_types(
        x_train: pd.DataFrame, x_val: pd.DataFrame, x_test: pd.DataFrame
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """MultiLabelBinarizer untuk Loan_List"""
        
        mlb = MultiLabelBinarizer()
        mlb.fit(x_train["Loan_List"])
 
        def _encode(data: pd.DataFrame) -> pd.DataFrame:
            encoded = pd.DataFrame(
                mlb.transform(data["Loan_List"]), columns=mlb.classes_, index=data.index,
            )
            out = pd.concat([data, encoded], axis=1)
            return out.drop(columns=["Type_of_Loan", "Loan_List"])
 
        return _encode(x_train), _encode(x_val), _encode(x_test), mlb
