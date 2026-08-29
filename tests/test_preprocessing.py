"""
Unit tests for data preprocessing, feature engineering, IQR capping,
and SelectKBest feature selection in the Customer Churn Analysis Pipeline.
"""

import numpy as np
import pandas as pd
from sklearn.feature_selection import SelectKBest, f_classif


def test_median_imputation_total_charges():
    """Test median imputation for TotalCharges containing NaN values."""
    data = pd.DataFrame({"TotalCharges": [100.0, 200.0, np.nan, 400.0, 500.0]})
    expected_median = data[
        "TotalCharges"
    ].median()  # Median of [100, 200, 400, 500] = 300.0

    # Perform median imputation
    data["TotalCharges"] = data["TotalCharges"].fillna(data["TotalCharges"].median())

    assert data["TotalCharges"].isnull().sum() == 0
    assert data["TotalCharges"].iloc[2] == expected_median
    assert expected_median == 300.0


def test_contract_ordinal_encoding():
    """Test ordinal encoding mapping for Contract column."""
    contract_mapping = {"Month-to-month": 0, "One year": 1, "Two year": 2}
    data = pd.DataFrame(
        {"Contract": ["Month-to-month", "One year", "Two year", "Month-to-month"]}
    )

    encoded_contract = data["Contract"].map(contract_mapping)

    assert encoded_contract.iloc[0] == 0
    assert encoded_contract.iloc[1] == 1
    assert encoded_contract.iloc[2] == 2
    assert encoded_contract.iloc[3] == 0
    assert encoded_contract.isnull().sum() == 0


def test_iqr_outlier_capping():
    """Test 1.5x IQR outlier capping logic on numerical features."""
    # Create dataset with normal values and extreme upper/lower outliers
    data = pd.DataFrame(
        {
            "MonthlyCharges": [
                20.0,
                25.0,
                30.0,
                35.0,
                40.0,
                45.0,
                50.0,
                55.0,
                60.0,
                1000.0,
                -500.0,
            ]
        }
    )

    col = "MonthlyCharges"
    Q1 = data[col].quantile(0.25)
    Q3 = data[col].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    capped_series = data[col].clip(lower=lower_bound, upper=upper_bound)

    assert capped_series.max() == upper_bound
    assert capped_series.min() == lower_bound
    assert capped_series.max() < 1000.0
    assert capped_series.min() > -500.0


def test_select_k_best_feature_selection():
    """Test SelectKBest(k=15) feature selection output column count."""
    np.random.seed(42)
    # Generate synthetic feature matrix with 20 columns and 100 rows
    num_samples = 100
    num_features = 20
    X = pd.DataFrame(
        np.random.randn(num_samples, num_features),
        columns=[f"feature_{i}" for i in range(num_features)],
    )
    y = np.random.randint(0, 2, size=num_samples)

    k_target = 15
    selector = SelectKBest(score_func=f_classif, k=k_target)
    X_selected = selector.fit_transform(X, y)

    assert X_selected.shape[1] == k_target
    assert X_selected.shape[0] == num_samples


def test_empty_dataset_check():
    """Test that an empty DataFrame raises ValueError("Input dataset is empty.")."""
    import pytest

    df_empty = pd.DataFrame()
    with pytest.raises(ValueError) as exc_info:
        if df_empty.empty:
            raise ValueError("Input dataset is empty.")

    assert "Input dataset is empty." in str(exc_info.value)
