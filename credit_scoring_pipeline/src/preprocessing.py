"""
Credit Scoring - Preprocessing Module
======================================
OOP-based preprocessing pipeline for credit scoring data.
"""

import re
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
import logging
import warnings
warnings.filterwarnings('ignore')

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────
#  Constants
# ─────────────────────────────────────────────
TARGET_COLUMN = "Credit_Score"
DROP_COLUMNS   = ["Unnamed: 0", "ID", "Customer_ID", "Name", "SSN"]

LABEL_MAP = {"Poor": 0, "Standard": 1, "Good": 2}

CATEGORICAL_FEATURES = [
    "Month", "Occupation", "Credit_Mix",
    "Payment_of_Min_Amount", "Payment_Behaviour",
]

NUMERIC_FEATURES = [
    "Age", "Annual_Income", "Monthly_Inhand_Salary",
    "Num_Bank_Accounts", "Num_Credit_Card", "Interest_Rate",
    "Num_of_Loan", "Delay_from_due_date", "Num_of_Delayed_Payment",
    "Changed_Credit_Limit", "Num_Credit_Inquiries",
    "Outstanding_Debt", "Credit_Utilization_Ratio",
    "Credit_History_Age_Months",           # engineered
    "Total_EMI_per_month", "Amount_invested_monthly",
    "Monthly_Balance",
]


# ─────────────────────────────────────────────
#  Individual Transformers
# ─────────────────────────────────────────────

class DirtyValueCleaner(BaseEstimator, TransformerMixin):
    """
    Cleans known dirty / noisy values in the raw CSV:
      - strips trailing/leading underscores and special chars from numeric strings
      - replaces placeholder strings ('_______', '!@9#%8', 'NM') with NaN
      - coerces columns that should be numeric
    """
    NUMERIC_STR_COLS = [
        "Annual_Income", "Monthly_Inhand_Salary",
        "Num_of_Loan", "Num_of_Delayed_Payment",
        "Changed_Credit_Limit", "Num_Credit_Inquiries",
        "Outstanding_Debt", "Amount_invested_monthly",
        "Monthly_Balance", "Age",
    ]
    PLACEHOLDER_RE = re.compile(r'^[_\s!@#$%^&*]+$')

    def fit(self, X, y=None):
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()

        # Replace placeholder strings with NaN in all object columns
        for col in X.select_dtypes(include="object").columns:
            X[col] = X[col].apply(
                lambda v: np.nan if (isinstance(v, str) and
                                     self.PLACEHOLDER_RE.match(v.strip())) else v
            )
            # Also handle 'NM' in Payment_of_Min_Amount
            if col == "Payment_of_Min_Amount":
                X[col] = X[col].replace("NM", np.nan)
            # Known garbage in Payment_Behaviour
            if col == "Payment_Behaviour":
                valid = {
                    "Low_spent_Small_value_payments",
                    "Low_spent_Medium_value_payments",
                    "Low_spent_Large_value_payments",
                    "High_spent_Small_value_payments",
                    "High_spent_Medium_value_payments",
                    "High_spent_Large_value_payments",
                }
                X[col] = X[col].apply(lambda v: v if v in valid else np.nan)

        # Coerce numeric columns – strip stray underscores / letters
        for col in self.NUMERIC_STR_COLS:
            if col in X.columns:
                X[col] = (
                    X[col].astype(str)
                          .str.replace(r'[^0-9.\-]', '', regex=True)
                          .replace('', np.nan)
                )
                X[col] = pd.to_numeric(X[col], errors='coerce')

        logger.info("DirtyValueCleaner: raw value cleaning complete.")
        return X


class CreditHistoryAgeParser(BaseEstimator, TransformerMixin):
    """
    Converts 'Credit_History_Age' (e.g. '14 Years and 8 Months')
    into a single integer representing total months.
    """
    YEAR_RE  = re.compile(r'(\d+)\s*Year',  re.IGNORECASE)
    MONTH_RE = re.compile(r'(\d+)\s*Month', re.IGNORECASE)

    def fit(self, X, y=None):
        return self

    def _parse(self, text):
        if pd.isna(text):
            return np.nan
        years  = int(m.group(1)) if (m := self.YEAR_RE.search(str(text)))  else 0
        months = int(m.group(1)) if (m := self.MONTH_RE.search(str(text))) else 0
        return years * 12 + months

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()
        if "Credit_History_Age" in X.columns:
            X["Credit_History_Age_Months"] = X["Credit_History_Age"].apply(self._parse)
            X.drop(columns=["Credit_History_Age"], inplace=True)
        return X


class FeatureDropper(BaseEstimator, TransformerMixin):
    """Drops irrelevant / PII columns."""
    def __init__(self, columns=None):
        self.columns = columns or DROP_COLUMNS

    def fit(self, X, y=None):
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        cols_to_drop = [c for c in self.columns if c in X.columns]
        return X.drop(columns=cols_to_drop)


class TypeOfLoanEncoder(BaseEstimator, TransformerMixin):
    """
    Converts the free-text 'Type_of_Loan' column into a numeric
    feature: count of distinct loan types mentioned.
    """
    def fit(self, X, y=None):
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()
        if "Type_of_Loan" in X.columns:
            X["Num_Loan_Types"] = (
                X["Type_of_Loan"]
                .fillna("")
                .apply(lambda v: len([t for t in re.split(r',| and ', v) if t.strip()]))
            )
            X.drop(columns=["Type_of_Loan"], inplace=True)
        return X


class CategoricalEncoder(BaseEstimator, TransformerMixin):
    """
    Label-encodes known categorical columns. Unseen categories during
    transform are mapped to -1 (unknown).
    """
    def __init__(self, cols=None):
        self.cols = cols or CATEGORICAL_FEATURES
        self.encoders_ = {}

    def fit(self, X, y=None):
        for col in self.cols:
            if col in X.columns:
                le = LabelEncoder()
                # fit on non-null values + a sentinel for NaN
                vals = X[col].fillna("__MISSING__").astype(str).values
                le.fit(vals)
                self.encoders_[col] = le
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()
        for col, le in self.encoders_.items():
            if col not in X.columns:
                continue
            vals = X[col].fillna("__MISSING__").astype(str)
            # map unseen labels to -1
            known = set(le.classes_)
            vals = vals.apply(lambda v: v if v in known else "__MISSING__")
            X[col] = le.transform(vals)
        return X


class NumericImputer(BaseEstimator, TransformerMixin):
    """Median-imputes all numeric columns."""
    def __init__(self):
        self.imputer_ = None
        self.num_cols_ = []

    def fit(self, X, y=None):
        self.num_cols_ = X.select_dtypes(include=[np.number]).columns.tolist()
        self.imputer_  = SimpleImputer(strategy="median")
        self.imputer_.fit(X[self.num_cols_])
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()
        X[self.num_cols_] = self.imputer_.transform(X[self.num_cols_])
        return X


class FeatureScaler(BaseEstimator, TransformerMixin):
    """StandardScaler on numeric features (post imputation)."""
    def __init__(self, method="standard"):
        self.method   = method
        self.scaler_  = None
        self.num_cols_ = []

    def fit(self, X, y=None):
        self.num_cols_ = X.select_dtypes(include=[np.number]).columns.tolist()
        self.scaler_   = StandardScaler() if self.method == "standard" else MinMaxScaler()
        self.scaler_.fit(X[self.num_cols_])
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()
        X[self.num_cols_] = self.scaler_.transform(X[self.num_cols_])
        return X


# ─────────────────────────────────────────────
#  Orchestrating Preprocessor Class
# ─────────────────────────────────────────────

class CreditDataPreprocessor:
    """
    Full preprocessing pipeline for credit-scoring data.

    Usage
    -----
    prep = CreditDataPreprocessor()
    X_train, y_train = prep.fit_transform(df_train)
    X_test,  y_test  = prep.transform(df_test)
    """

    def __init__(self, scaler_method: str = "standard"):
        self.scaler_method = scaler_method
        self.pipeline_     = None
        self.label_encoder_ = None
        self.feature_names_: list = []

    # ── internal pipeline builder ──────────────
    def _build_pipeline(self) -> Pipeline:
        return Pipeline([
            ("drop_cols",        FeatureDropper()),
            ("dirty_cleaner",    DirtyValueCleaner()),
            ("history_parser",   CreditHistoryAgeParser()),
            ("loan_type_enc",    TypeOfLoanEncoder()),
            ("cat_encoder",      CategoricalEncoder()),
            ("num_imputer",      NumericImputer()),
            ("scaler",           FeatureScaler(method=self.scaler_method)),
        ])

    # ── target extraction ──────────────────────
    def _extract_target(self, df: pd.DataFrame):
        if TARGET_COLUMN not in df.columns:
            raise ValueError(f"Target column '{TARGET_COLUMN}' not found.")
        y_raw = df[TARGET_COLUMN].map(LABEL_MAP)
        if y_raw.isna().any():
            bad = df[TARGET_COLUMN][y_raw.isna()].unique()
            raise ValueError(f"Unknown target labels: {bad}")
        return y_raw.astype(int).values

    # ── public API ─────────────────────────────
    def fit_transform(self, df: pd.DataFrame):
        """Fit the pipeline on training data and return (X_array, y_array)."""
        logger.info(f"fit_transform: input shape {df.shape}")
        y = self._extract_target(df)

        X_df = df.drop(columns=[TARGET_COLUMN], errors='ignore')

        self.pipeline_ = self._build_pipeline()
        X_transformed  = self.pipeline_.fit_transform(X_df)

        # Store feature names for interpretability
        self.feature_names_ = (
            self.pipeline_.named_steps["scaler"]
                          .num_cols_
        )

        logger.info(f"fit_transform: output shape {X_transformed.shape}, "
                    f"features={len(self.feature_names_)}")
        return X_transformed, y

    def transform(self, df: pd.DataFrame):
        """Apply fitted pipeline to new data. Returns (X_array, y_array or None)."""
        if self.pipeline_ is None:
            raise RuntimeError("Pipeline not fitted. Call fit_transform first.")

        has_target = TARGET_COLUMN in df.columns
        y = self._extract_target(df) if has_target else None
        X_df = df.drop(columns=[TARGET_COLUMN], errors='ignore')

        X_transformed = self.pipeline_.transform(X_df)
        logger.info(f"transform: output shape {X_transformed.shape}")
        return X_transformed, y

    # ── utility ────────────────────────────────
    def get_feature_names(self) -> list:
        return self.feature_names_

    def get_label_map(self) -> dict:
        return LABEL_MAP

    def get_inverse_label_map(self) -> dict:
        return {v: k for k, v in LABEL_MAP.items()}


# ─────────────────────────────────────────────
#  Data Loader Helper
# ─────────────────────────────────────────────

class DataLoader:
    """Loads raw CSV and returns train/test splits."""

    def __init__(self, filepath: str, test_size: float = 0.2, random_state: int = 42):
        self.filepath     = filepath
        self.test_size    = test_size
        self.random_state = random_state

    def load(self) -> pd.DataFrame:
        df = pd.read_csv(self.filepath)
        logger.info(f"DataLoader: loaded {df.shape[0]:,} rows × {df.shape[1]} cols")
        return df

    def train_test_split(self, df: pd.DataFrame):
        from sklearn.model_selection import train_test_split
        train_df, test_df = train_test_split(
            df, test_size=self.test_size,
            random_state=self.random_state,
            stratify=df[TARGET_COLUMN]
        )
        logger.info(f"DataLoader: train={len(train_df):,} | test={len(test_df):,}")
        return train_df, test_df
