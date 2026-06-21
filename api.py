"""
api.py
======
REST API untuk inferencing model Credit Score Prediction.

Endpoint:
    POST /predict   -> menerima JSON fitur mentah satu nasabah, mengembalikan
                        prediksi kelas (Poor/Standard/Good) + probabilitas.

Jalankan dengan:
    uvicorn api:app --reload --port 8000

Dokumentasi interaktif otomatis tersedia di http://localhost:8000/docs
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import Optional

from inference import CreditScorePredictor

app = FastAPI(
    title="Credit Score Prediction API",
    description="API inferencing untuk model klasifikasi Credit Score (Poor/Standard/Good)",
    version="1.0.0",
)

predictor = CreditScorePredictor(artifact_dir="artifacts")


class CreditScoreRequest(BaseModel):
    """Skema input /predict. Field sama seperti kolom data_A.csv (tanpa Credit_Score)."""

    Month: str = Field(..., examples=["January"])
    Age: float = Field(..., examples=[30])
    Occupation: str = Field(..., examples=["Engineer"])
    Annual_Income: str = Field(..., examples=["50000.0"])
    Monthly_Inhand_Salary: float = Field(..., examples=[4000.0])
    Num_Bank_Accounts: float = Field(..., examples=[3])
    Num_Credit_Card: float = Field(..., examples=[4])
    Interest_Rate: float = Field(..., examples=[12])
    Num_of_Loan: str = Field(..., examples=["2"])
    Type_of_Loan: Optional[str] = Field(None, examples=["Auto Loan, and Personal Loan"])
    Delay_from_due_date: float = Field(..., examples=[5])
    Num_of_Delayed_Payment: str = Field(..., examples=["3"])
    Changed_Credit_Limit: str = Field(..., examples=["5.5"])
    Num_Credit_Inquiries: float = Field(..., examples=[2.0])
    Credit_Mix: str = Field(..., examples=["Good"])
    Outstanding_Debt: str = Field(..., examples=["500.0"])
    Credit_Utilization_Ratio: float = Field(..., examples=[30.0])
    Credit_History_Age: str = Field(..., examples=["10 Years and 3 Months"])
    Payment_of_Min_Amount: str = Field(..., examples=["Yes"])
    Total_EMI_per_month: float = Field(..., examples=[100.0])
    Amount_invested_monthly: str = Field(..., examples=["50.0"])
    Payment_Behaviour: str = Field(..., examples=["Low_spent_Small_value_payments"])
    Monthly_Balance: str = Field(..., examples=["300.0"])


class CreditScoreResponse(BaseModel):
    prediction: str
    prediction_code: int
    probability: dict[str, float]


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict", response_model=CreditScoreResponse)
def predict(payload: CreditScoreRequest):
    try:
        result = predictor.predict(payload.model_dump())
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inferencing gagal: {e}")
    return result
