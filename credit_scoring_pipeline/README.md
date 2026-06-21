# Credit Scoring ML Pipeline

Pipeline pelatihan machine learning berbasis OOP untuk penilaian performa kredit nasabah, dilengkapi dengan **MLflow** untuk experiment tracking, logging metrik, dan model registry.

---

## Struktur Proyek

```
credit_scoring/
├── data/
│   └── data_A.csv                  # Dataset mentah
├── models/
│   ├── preprocessor.pkl            # Fitted preprocessor (disimpan otomatis)
│   └── best_model_XGBoost.pkl      # Model terbaik (disimpan otomatis)
├── src/
│   ├── __init__.py
│   ├── preprocessing.py            # Class: CreditDataPreprocessor, DataLoader
│   ├── training.py                 # Class: TrainingPipeline, BaseModel + subclass
│   └── evaluation.py              # Class: ModelEvaluator, ComparativeEvaluator
├── mlruns/                         # MLflow artifacts
├── mlflow.db                       # MLflow SQLite tracking database
├── train.py                        # Entry point utama
├── requirements.txt
└── training.log                    # Log file training
```

---

## Arsitektur OOP

### `src/preprocessing.py`
| Class | Tanggung Jawab |
|---|---|
| `DirtyValueCleaner` | Membersihkan nilai kotor (placeholder, karakter aneh) |
| `CreditHistoryAgeParser` | Parsing teks "14 Years and 8 Months" → integer bulan |
| `FeatureDropper` | Menghapus kolom PII & tidak relevan |
| `TypeOfLoanEncoder` | Mengubah text jenis pinjaman → jumlah numerik |
| `CategoricalEncoder` | Label encoding dengan penanganan kategori tidak dikenal |
| `NumericImputer` | Median imputation untuk nilai hilang |
| `FeatureScaler` | StandardScaler / MinMaxScaler |
| **`CreditDataPreprocessor`** | **Orkestrasi seluruh pipeline preprocessing** |
| `DataLoader` | Load CSV dan stratified train/test split |

### `src/training.py`
| Class | Tanggung Jawab |
|---|---|
| `BaseModel` *(abstract)* | Interface seragam: `fit()`, `predict()`, `predict_proba()`, `save()` |
| `LogisticRegressionModel` | Implementasi Logistic Regression |
| `RandomForestModel` | Implementasi Random Forest |
| `XGBoostModel` | Implementasi XGBoost |
| `LightGBMModel` | Implementasi LightGBM |
| **`TrainingPipeline`** | **Orkestrasi training + MLflow logging + pemilihan model terbaik** |

### `src/evaluation.py`
| Class | Tanggung Jawab |
|---|---|
| `MetricsCalculator` | Menghitung accuracy, F1, AUC, log-loss, per-class F1 |
| `PlotFactory` | Membuat confusion matrix, ROC curves, feature importance, comparison chart |
| **`ModelEvaluator`** | **Evaluasi model tunggal + logging artifact ke MLflow** |
| `ComparativeEvaluator` | Perbandingan semua model, ringkasan DataFrame, chart gabungan |

---

## Cara Penggunaan

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Jalankan pipeline penuh (semua model)
```bash
python train.py
```

### 3. Training model tertentu saja
```bash
python train.py --models xgboost lightgbm
```

### 4. Ubah metrik seleksi model terbaik
```bash
python train.py --best_metric roc_auc_ovr
```

### 5. Semua opsi CLI
```
--data          Path ke CSV (default: data/data_A.csv)
--experiment    Nama experiment MLflow (default: CreditScoring)
--tracking_uri  URI MLflow tracking (default: ./mlruns → auto-convert ke SQLite)
--models        Model yang dilatih: logistic_regression random_forest xgboost lightgbm
--best_metric   Metrik pemilihan terbaik: accuracy f1_macro f1_weighted roc_auc_ovr
--test_size     Proporsi test set (default: 0.2)
--cv_folds      Jumlah fold cross-validation (default: 5)
--scaler        Metode scaling: standard | minmax (default: standard)
--random_seed   Random seed (default: 42)
```

### 6. Lihat hasil eksperimen di MLflow UI
```bash
mlflow ui --backend-store-uri sqlite:///mlflow.db --port 5000
# Buka browser: http://localhost:5000
```

---

## Hasil Eksperimen (data_A.csv)

| Model | Accuracy | F1-Macro | AUC-OVR |
|---|---|---|---|
| **XGBoost** ★ | **0.7358** | **0.7129** | **0.8802** |
| LightGBM | 0.7112 | 0.7018 | 0.8753 |
| Random Forest | 0.6940 | 0.6856 | 0.8673 |
| Logistic Regression | 0.5762 | 0.5758 | 0.7657 |

★ Model terpilih sebagai **best model** berdasarkan F1-Macro.

---

## Yang Dicatat MLflow per Run

**Parameters:** semua hyperparameter model, jumlah CV folds, scaler method

**Metrics:** accuracy, f1_macro, f1_weighted, precision_macro, recall_macro, f1_per_class (poor/standard/good), roc_auc_ovr, roc_auc_ovo, log_loss, cv_accuracy_mean, cv_accuracy_std, train_time_sec

**Artifacts:**
- Model artifact (sklearn / xgboost / lightgbm native format)
- `reports/classification_report.txt`
- `reports/metrics.json`
- `plots/confusion_matrix.png`
- `plots/confusion_matrix_norm.png`
- `plots/roc_curves.png`
- `plots/feature_importance.png`
- `feature_importance/feature_importance.txt`

**Tags:** `model_type`, `best_model` (true/false), `selection_metric`

---

## Retraining

Cukup jalankan ulang `train.py` — setiap run akan dicatat sebagai run baru di experiment yang sama sehingga dapat dibandingkan riwayat seluruh percobaan.

```bash
# Retraining dengan hyperparameter berbeda
python train.py --models xgboost --experiment CreditScoring_v2
```
