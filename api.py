"""
FastAPI Application for Customer Churn Risk Prediction API.
Includes GET /health readiness probe and POST /predict inference endpoint.
"""

import os
import joblib
import numpy as np
import pandas as pd
from typing import Optional
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# Initialize FastAPI App
app = FastAPI(
    title="Customer Churn Risk Prediction API",
    description="REST API for predicting subscriber churn probability and risk segmentation.",
    version="1.0.0"
)

# Enable CORS Middleware for React & Web Frontends
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables for loaded ML artifacts
MODEL_PATH = "model.pkl"
SCALER_PATH = "scaler.pkl"
SELECTOR_PATH = "selector.pkl"

model = None
scaler = None
selector = None

# Load model artifacts on startup
if os.path.exists(MODEL_PATH) and os.path.exists(SCALER_PATH) and os.path.exists(SELECTOR_PATH):
    try:
        model = joblib.load(MODEL_PATH)
        scaler = joblib.load(SCALER_PATH)
        selector = joblib.load(SELECTOR_PATH)
        print("ML model, scaler, and feature selector loaded successfully.")
    except Exception as e:
        print(f"Error loading model artifacts: {e}")


class ChurnPredictionInput(BaseModel):
    Contract: str = Field(default="Month-to-month", description="Month-to-month, One year, or Two year")
    tenure: int = Field(default=12, description="Tenure in months", ge=0)
    MonthlyCharges: float = Field(default=75.0, description="Monthly charges amount ($)", ge=0.0)
    InternetService: str = Field(default="Fiber optic", description="Fiber optic, DSL, or No")
    PaymentMethod: str = Field(default="Electronic check", description="Electronic check, Mailed check, Bank transfer (automatic), or Credit card (automatic)")
    TechSupport: str = Field(default="No", description="Yes, No, or No internet service")
    OnlineSecurity: str = Field(default="No", description="Yes, No, or No internet service")
    PaperlessBilling: str = Field(default="Yes", description="Yes or No")
    SeniorCitizen: int = Field(default=0, description="0 or 1")
    gender: str = Field(default="Male", description="Male or Female")
    Partner: str = Field(default="No", description="Yes or No")
    Dependents: str = Field(default="No", description="Yes or No")
    PhoneService: str = Field(default="Yes", description="Yes or No")
    MultipleLines: str = Field(default="No", description="Yes, No, or No phone service")
    OnlineBackup: str = Field(default="No", description="Yes, No, or No internet service")
    DeviceProtection: str = Field(default="No", description="Yes, No, or No internet service")
    StreamingTV: str = Field(default="No", description="Yes, No, or No internet service")
    StreamingMovies: str = Field(default="No", description="Yes, No, or No internet service")
    TotalCharges: Optional[float] = Field(default=None, description="Total charges amount ($)")


def preprocess_input(input_data: ChurnPredictionInput) -> pd.DataFrame:
    """Preprocess single API request payload to match model training feature matrix."""
    contract_map = {'Month-to-month': 0, 'One year': 1, 'Two year': 2}
    internet_map = {'No': 0, 'DSL': 1, 'Fiber optic': 2}
    payment_map = {
        'Electronic check': 0,
        'Mailed check': 1,
        'Bank transfer (automatic)': 2,
        'Credit card (automatic)': 3
    }
    binary_map = {'No': 0, 'Yes': 1}
    tri_map = {'No': 0, 'No internet service': 1, 'No phone service': 1, 'Yes': 2}

    total_charges = input_data.TotalCharges
    if total_charges is None:
        total_charges = float(input_data.tenure * input_data.MonthlyCharges)

    # Calculate tenure_group
    tenure_val = input_data.tenure
    if tenure_val <= 12:
        tenure_group = 0
    elif tenure_val <= 24:
        tenure_group = 1
    elif tenure_val <= 48:
        tenure_group = 2
    else:
        tenure_group = 3

    avg_monthly_charge = (total_charges / tenure_val) if tenure_val > 0 else 0.0
    has_online_security = 1 if input_data.OnlineSecurity == 'Yes' else 0
    has_tech_support = 1 if input_data.TechSupport == 'Yes' else 0

    feature_dict = {
        'gender': 1 if input_data.gender == 'Male' else 0,
        'SeniorCitizen': input_data.SeniorCitizen,
        'Partner': binary_map.get(input_data.Partner, 0),
        'Dependents': binary_map.get(input_data.Dependents, 0),
        'tenure': tenure_val,
        'PhoneService': binary_map.get(input_data.PhoneService, 1),
        'MultipleLines': tri_map.get(input_data.MultipleLines, 0),
        'InternetService': internet_map.get(input_data.InternetService, 2),
        'OnlineSecurity': tri_map.get(input_data.OnlineSecurity, 0),
        'OnlineBackup': tri_map.get(input_data.OnlineBackup, 0),
        'DeviceProtection': tri_map.get(input_data.DeviceProtection, 0),
        'TechSupport': tri_map.get(input_data.TechSupport, 0),
        'StreamingTV': tri_map.get(input_data.StreamingTV, 0),
        'StreamingMovies': tri_map.get(input_data.StreamingMovies, 0),
        'Contract': contract_map.get(input_data.Contract, 0),
        'PaperlessBilling': binary_map.get(input_data.PaperlessBilling, 1),
        'PaymentMethod': payment_map.get(input_data.PaymentMethod, 0),
        'MonthlyCharges': float(input_data.MonthlyCharges),
        'TotalCharges': float(total_charges),
        'tenure_group': tenure_group,
        'avg_monthly_charge': float(avg_monthly_charge),
        'has_online_security': has_online_security,
        'has_tech_support': has_tech_support
    }

    return pd.DataFrame([feature_dict])


@app.get("/health", summary="Health Probe Endpoint")
def health_check():
    """Readiness probe endpoint confirming API status and model initialization."""
    is_loaded = (model is not None and scaler is not None and selector is not None)
    return {
        "status": "healthy",
        "model_loaded": is_loaded
    }


@app.post("/predict", summary="Churn Risk Prediction Endpoint")
def predict_churn(payload: ChurnPredictionInput):
    """Predict churn probability and return risk classification (Low, Medium, High)."""
    if model is None or scaler is None or selector is None:
        raise HTTPException(
            status_code=503,
            detail="ML Model not initialized. Run 'python churn_analysis.py' first."
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
            "predicted_churn": predicted_churn
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference error: {str(e)}")
