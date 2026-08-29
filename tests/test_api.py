"""
Unit tests for api.py FastAPI endpoints (/health and /predict).
"""

from api import ChurnPredictionInput, health_check, predict_churn


def test_health_check_endpoint():
    """Test GET /health probe returns healthy status."""
    response = health_check()
    assert response["status"] == "healthy"
    assert response["model_loaded"] is True


def test_predict_churn_high_risk():
    """Test POST /predict with high-risk subscriber profile."""
    payload = ChurnPredictionInput(
        Contract="Month-to-month",
        tenure=2,
        MonthlyCharges=95.0,
        InternetService="Fiber optic",
        PaymentMethod="Electronic check",
        TechSupport="No",
        OnlineSecurity="No",
    )
    result = predict_churn(payload)
    assert "churn_probability" in result
    assert "risk_level" in result
    assert "predicted_churn" in result
    assert result["risk_level"] in ["High", "Medium", "Low"]
    assert 0.0 <= result["churn_probability"] <= 1.0


def test_predict_churn_low_risk():
    """Test POST /predict with low-risk subscriber profile."""
    payload = ChurnPredictionInput(
        Contract="Two year",
        tenure=60,
        MonthlyCharges=25.0,
        InternetService="DSL",
        PaymentMethod="Bank transfer (automatic)",
        TechSupport="Yes",
        OnlineSecurity="Yes",
    )
    result = predict_churn(payload)
    assert result["risk_level"] == "Low"
    assert result["predicted_churn"] == 0


def test_predict_churn_invalid_category():
    """Test POST /predict with unknown category raises 422 HTTPException."""
    import pytest
    from fastapi import HTTPException

    payload = ChurnPredictionInput(
        Contract="InvalidContractType",
        tenure=12,
        MonthlyCharges=50.0,
    )
    with pytest.raises(HTTPException) as exc_info:
        predict_churn(payload)

    assert exc_info.value.status_code == 422
    assert (
        "Invalid value 'InvalidContractType' for field 'Contract'"
        in exc_info.value.detail
    )


def test_predict_churn_tenure_zero():
    """Test POST /predict with tenure=0 calculates avg_monthly_charge as 0.0 without division by zero."""
    payload = ChurnPredictionInput(
        Contract="Month-to-month",
        tenure=0,
        MonthlyCharges=50.0,
    )
    result = predict_churn(payload)
    assert "churn_probability" in result
    assert result["risk_level"] in ["High", "Medium", "Low"]
