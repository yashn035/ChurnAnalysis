"""
FastAPI Application for Customer Churn Risk Prediction API.
Includes GET /health readiness probe and POST /predict inference endpoint.
"""

import os
from typing import Optional

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# Initialize FastAPI App
app = FastAPI(
    title="Customer Churn Risk Prediction API",
    description="REST API for predicting subscriber churn probability and risk segmentation.",
    version="1.0.0",
)

# Enable CORS Middleware restricted to authorized frontend origins
allowed_origins = [
    "http://localhost:8501",
    "http://127.0.0.1:8501",
    "http://localhost:3000",
    "http://127.0.0.1:3000",
    "https://yashn035-churn-analysis.streamlit.app",
    "https://your-app.streamlit.app",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)

# Global variables for loaded ML artifacts
MODEL_PATH = "models/model.pkl" if os.path.exists("models/model.pkl") else "model.pkl"
SCALER_PATH = (
    "models/scaler.pkl" if os.path.exists("models/scaler.pkl") else "scaler.pkl"
)
SELECTOR_PATH = (
    "models/selector.pkl" if os.path.exists("models/selector.pkl") else "selector.pkl"
)

model = None
scaler = None
selector = None

# Load model artifacts on startup
if (
    os.path.exists(MODEL_PATH)
    and os.path.exists(SCALER_PATH)
    and os.path.exists(SELECTOR_PATH)
):
    try:
        model = joblib.load(MODEL_PATH)
        scaler = joblib.load(SCALER_PATH)
        selector = joblib.load(SELECTOR_PATH)
        print("ML model, scaler, and feature selector loaded successfully.")
    except Exception as e:
        print(f"Error loading model artifacts: {e}")


class ChurnPredictionInput(BaseModel):
    Contract: str = Field(
        default="Month-to-month", description="Month-to-month, One year, or Two year"
    )
    tenure: int = Field(default=12, description="Tenure in months", ge=0)
    MonthlyCharges: float = Field(
        default=75.0, description="Monthly charges amount ($)", ge=0.0
    )
    InternetService: str = Field(
        default="Fiber optic", description="Fiber optic, DSL, or No"
    )
    PaymentMethod: str = Field(
        default="Electronic check",
        description="Electronic check, Mailed check, Bank transfer (automatic), or Credit card (automatic)",
    )
    TechSupport: str = Field(
        default="No", description="Yes, No, or No internet service"
    )
    OnlineSecurity: str = Field(
        default="No", description="Yes, No, or No internet service"
    )
    PaperlessBilling: str = Field(default="Yes", description="Yes or No")
    SeniorCitizen: int = Field(default=0, description="0 or 1")
    gender: str = Field(default="Male", description="Male or Female")
    Partner: str = Field(default="No", description="Yes or No")
    Dependents: str = Field(default="No", description="Yes or No")
    PhoneService: str = Field(default="Yes", description="Yes or No")
    MultipleLines: str = Field(default="No", description="Yes, No, or No phone service")
    OnlineBackup: str = Field(
        default="No", description="Yes, No, or No internet service"
    )
    DeviceProtection: str = Field(
        default="No", description="Yes, No, or No internet service"
    )
    StreamingTV: str = Field(
        default="No", description="Yes, No, or No internet service"
    )
    StreamingMovies: str = Field(
        default="No", description="Yes, No, or No internet service"
    )
    TotalCharges: Optional[float] = Field(
        default=None, description="Total charges amount ($)"
    )


def preprocess_input(input_data: ChurnPredictionInput) -> pd.DataFrame:
    """Preprocess single API request payload to match model training feature matrix."""
    contract_map = {"Month-to-month": 0, "One year": 1, "Two year": 2}
    internet_map = {"No": 0, "DSL": 1, "Fiber optic": 2}
    payment_map = {
        "Electronic check": 0,
        "Mailed check": 1,
        "Bank transfer (automatic)": 2,
        "Credit card (automatic)": 3,
    }
    binary_map = {"No": 0, "Yes": 1}
    tri_map = {"No": 0, "No internet service": 1, "No phone service": 1, "Yes": 2}

    # Validate categorical field encodings
    validations = [
        ("Contract", input_data.Contract, contract_map),
        ("InternetService", input_data.InternetService, internet_map),
        ("PaymentMethod", input_data.PaymentMethod, payment_map),
        ("Partner", input_data.Partner, binary_map),
        ("Dependents", input_data.Dependents, binary_map),
        ("PhoneService", input_data.PhoneService, binary_map),
        ("PaperlessBilling", input_data.PaperlessBilling, binary_map),
        ("MultipleLines", input_data.MultipleLines, tri_map),
        ("OnlineSecurity", input_data.OnlineSecurity, tri_map),
        ("OnlineBackup", input_data.OnlineBackup, tri_map),
        ("DeviceProtection", input_data.DeviceProtection, tri_map),
        ("TechSupport", input_data.TechSupport, tri_map),
        ("StreamingTV", input_data.StreamingTV, tri_map),
        ("StreamingMovies", input_data.StreamingMovies, tri_map),
    ]

    try:
        for field_name, value, mapping in validations:
            if value not in mapping:
                allowed_vals = ", ".join(f"'{k}'" for k in mapping.keys())
                raise ValueError(
                    f"Invalid value '{value}' for field '{field_name}'. Allowed values: {allowed_vals}"
                )
    except ValueError as ve:
        raise HTTPException(status_code=422, detail=str(ve))

    total_charges = input_data.TotalCharges
    if total_charges is None:
        total_charges = float(input_data.tenure * input_data.MonthlyCharges)

    # Calculate tenure_group & avg_monthly_charge (check tenure == 0)
    tenure_val = input_data.tenure
    if tenure_val == 0:
        avg_monthly_charge = 0.0
    else:
        avg_monthly_charge = float(total_charges / tenure_val)

    if tenure_val <= 12:
        tenure_group = 0
    elif tenure_val <= 24:
        tenure_group = 1
    elif tenure_val <= 48:
        tenure_group = 2
    else:
        tenure_group = 3

    has_online_security = 1 if input_data.OnlineSecurity == "Yes" else 0
    has_tech_support = 1 if input_data.TechSupport == "Yes" else 0

    feature_dict = {
        "gender": 1 if input_data.gender == "Male" else 0,
        "SeniorCitizen": input_data.SeniorCitizen,
        "Partner": binary_map[input_data.Partner],
        "Dependents": binary_map[input_data.Dependents],
        "tenure": tenure_val,
        "PhoneService": binary_map[input_data.PhoneService],
        "MultipleLines": tri_map[input_data.MultipleLines],
        "InternetService": internet_map[input_data.InternetService],
        "OnlineSecurity": tri_map[input_data.OnlineSecurity],
        "OnlineBackup": tri_map[input_data.OnlineBackup],
        "DeviceProtection": tri_map[input_data.DeviceProtection],
        "TechSupport": tri_map[input_data.TechSupport],
        "StreamingTV": tri_map[input_data.StreamingTV],
        "StreamingMovies": tri_map[input_data.StreamingMovies],
        "Contract": contract_map[input_data.Contract],
        "PaperlessBilling": binary_map[input_data.PaperlessBilling],
        "PaymentMethod": payment_map[input_data.PaymentMethod],
        "MonthlyCharges": float(input_data.MonthlyCharges),
        "TotalCharges": float(total_charges),
        "tenure_group": tenure_group,
        "avg_monthly_charge": float(avg_monthly_charge),
        "has_online_security": has_online_security,
        "has_tech_support": has_tech_support,
    }

    return pd.DataFrame([feature_dict])


@app.get("/health", summary="Health Probe Endpoint")
def health_check():
    """
    Readiness probe endpoint confirming API status, model file existence on disk,
    and in-memory ML artifact initialization.
    """
    model_file_exists = os.path.exists(MODEL_PATH)
    is_loaded = model is not None and scaler is not None and selector is not None

    status_str = "healthy" if (is_loaded and model_file_exists) else "degraded"
    return {
        "status": status_str,
        "model_loaded": is_loaded,
        "model_file_exists": model_file_exists,
        "model_path": MODEL_PATH,
    }


@app.post(
    "/predict",
    summary="Churn Risk Prediction Endpoint",
    response_description="Predicted subscriber churn probability, risk classification level, and binary churn flag.",
)
def predict_churn(payload: ChurnPredictionInput):
    r"""
    Predict customer churn risk score and probability using pre-trained ML models.

    ### Request Body Parameters:
    - **Contract**: Subscriber contract type (`'Month-to-month'`, `'One year'`, `'Two year'`).
    - **tenure**: Total tenure duration in months ($\ge 0$).
    - **MonthlyCharges**: Monthly subscription charge in USD ($\ge 0.0$).
    - **InternetService**: Internet service tier (`'Fiber optic'`, `'DSL'`, `'No'`).
    - **PaymentMethod**: Billing payment method (`'Electronic check'`, `'Mailed check'`, `'Bank transfer (automatic)'`, `'Credit card (automatic)'`).
    - **TechSupport** / **OnlineSecurity**: Add-on security services (`'Yes'`, `'No'`, `'No internet service'`).
    - **SeniorCitizen**: Senior citizen status flag (`0` or `1`).

    ### Returns:
    - **churn_probability**: Float between 0.0 and 1.0 representing churn probability.
    - **risk_level**: Operational risk tier classification (`'High'`, `'Medium'`, `'Low'`).
    - **predicted_churn**: Binary churn flag (`1` for high risk $P > 0.50$, `0` for retained).

    ### Error Responses:
    - **422 Unprocessable Entity**: Invalid input schema or unrecognized categorical string value.
    - **503 Service Unavailable**: ML model artifacts not loaded into memory.
    - **500 Internal Server Error**: Downstream inference execution error.
    """
    if model is None or scaler is None or selector is None:
        raise HTTPException(
            status_code=503,
            detail="ML Model not initialized. Run 'python churn_analysis.py' first.",
        )

    try:
        raw_df = preprocess_input(payload)
        scaled_features = scaler.transform(raw_df)
        selected_features = selector.transform(scaled_features)

        prob = float(model.predict_proba(selected_features)[0, 1])
        prob_rounded = round(prob, 4)

        if prob_rounded > 0.65:
            risk_level = "High"
        elif prob_rounded > 0.35:
            risk_level = "Medium"
        else:
            risk_level = "Low"

        predicted_churn = int(prob_rounded > 0.50)

        return {
            "churn_probability": prob_rounded,
            "risk_level": risk_level,
            "predicted_churn": predicted_churn,
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference error: {str(e)}")
