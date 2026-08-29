"""
CLI Pipeline Runner for Customer Churn Analysis
Accepts --data argument, executes machine learning modeling pipeline,
exports predictions to predictions_output.csv, and logs metrics & errors.
"""

import argparse
import os
import sys
import time
import warnings

import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from xgboost import XGBClassifier

from json_logger import get_json_logger, log_pipeline_step

warnings.filterwarnings("ignore")

logger = get_json_logger("churn_pipeline_cli", "logs/pipeline.jsonl")

RANDOM_STATE = 42


def run_pipeline(data_path, output_path="data/processed/predictions_output.csv"):
    try:
        t_start = time.time()
        logger.info(f"Starting Churn Pipeline execution on input file: '{data_path}'")

        if not os.path.exists(data_path):
            raise FileNotFoundError(f"Input data file '{data_path}' does not exist.")

        df = pd.read_csv(data_path)
        log_pipeline_step(logger, "data_load", time.time() - t_start, n_samples=len(df))

        if "customerID" not in df.columns:
            logger.warning(
                "'customerID' column not found. Generating default index customer IDs."
            )
            df["customerID"] = [f"CUST-{i+1000}" for i in range(len(df))]

        customer_ids = df["customerID"].copy()
        df_model = df.drop(columns=["customerID"])

        # 1. Target check and conversion
        if "Churn" not in df_model.columns:
            raise KeyError("Dataset must contain a 'Churn' column.")

        df_model["Churn"] = (
            df_model["Churn"]
            .astype(str)
            .str.strip()
            .map({"Yes": 1, "No": 0, "1": 1, "0": 0})
        )
        if df_model["Churn"].isnull().any():
            logger.warning(
                "Unmapped values found in 'Churn' target. Filling missing target with 0."
            )
            df_model["Churn"] = df_model["Churn"].fillna(0).astype(int)

        churn_rate = df_model["Churn"].mean()
        logger.info(f"Target distribution - Actual Churn Rate: {churn_rate:.2%}")

        # 2. TotalCharges median imputation
        if "TotalCharges" in df_model.columns:
            if df_model["TotalCharges"].dtype == "object":
                df_model["TotalCharges"] = pd.to_numeric(
                    df_model["TotalCharges"].astype(str).str.strip(), errors="coerce"
                )
            median_total = df_model["TotalCharges"].median()
            df_model["TotalCharges"] = df_model["TotalCharges"].fillna(median_total)

        # 3. Categorical missing value mode imputation
        for col in df_model.select_dtypes(include=["object", "category"]).columns:
            if col != "Churn" and df_model[col].isnull().sum() > 0:
                mode_val = df_model[col].mode()[0]
                df_model[col] = df_model[col].fillna(mode_val)

        readable_df = df_model.copy()
        readable_df["customerID"] = customer_ids

        # 4. Ordinal encodings & Feature Engineering
        t_fe = time.time()
        if "Contract" in df_model.columns:
            df_model["Contract"] = (
                df_model["Contract"]
                .map({"Month-to-month": 0, "One year": 1, "Two year": 2})
                .fillna(0)
                .astype(int)
            )

        if "InternetService" in df_model.columns:
            df_model["InternetService"] = (
                df_model["InternetService"]
                .map({"No": 0, "DSL": 1, "Fiber optic": 2})
                .fillna(0)
                .astype(int)
            )

        if "PaymentMethod" in df_model.columns:
            df_model["PaymentMethod"] = (
                df_model["PaymentMethod"]
                .map(
                    {
                        "Electronic check": 0,
                        "Mailed check": 1,
                        "Bank transfer (automatic)": 2,
                        "Credit card (automatic)": 3,
                    }
                )
                .fillna(0)
                .astype(int)
            )

        # 5. Feature Engineering
        if "tenure" in df_model.columns:
            df_model["tenure_group"] = (
                pd.cut(
                    df_model["tenure"],
                    bins=[0, 12, 24, 48, 72],
                    labels=[0, 1, 2, 3],
                    include_lowest=True,
                )
                .fillna(0)
                .astype(int)
            )

            if "TotalCharges" in df_model.columns:
                df_model["avg_monthly_charge"] = np.where(
                    df_model["tenure"] > 0,
                    df_model["TotalCharges"] / df_model["tenure"],
                    0.0,
                )

        if "OnlineSecurity" in df_model.columns:
            df_model["has_online_security"] = (
                df_model["OnlineSecurity"] == "Yes"
            ).astype(int)

        if "TechSupport" in df_model.columns:
            df_model["has_tech_support"] = (df_model["TechSupport"] == "Yes").astype(
                int
            )

        # 6. Encode remaining string columns with LabelEncoder
        for col in df_model.select_dtypes(include=["object"]).columns:
            le = LabelEncoder()
            df_model[col] = le.fit_transform(df_model[col].astype(str))

        # 7. Outlier treatment (1.5x IQR rule)
        for col in ["MonthlyCharges", "TotalCharges"]:
            if col in df_model.columns:
                Q1 = df_model[col].quantile(0.25)
                Q3 = df_model[col].quantile(0.75)
                IQR = Q3 - Q1
                df_model[col] = df_model[col].clip(
                    lower=Q1 - 1.5 * IQR, upper=Q3 + 1.5 * IQR
                )
        log_pipeline_step(
            logger, "feature_engineering", time.time() - t_fe, n_samples=len(df_model)
        )

        # 8. Train-Test Split & Scaling
        X = df_model.drop(columns=["Churn"])
        y = df_model["Churn"]

        X_train, X_test, y_train, y_test, ids_train, ids_test = train_test_split(
            X, y, customer_ids, test_size=0.20, stratify=y, random_state=RANDOM_STATE
        )

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        raw_feature_names = X.columns.tolist()

        # 9. Feature Selection SelectKBest (k=15 or min(15, num_features))
        k_val = min(15, X_train_scaled.shape[1])
        selector = SelectKBest(score_func=f_classif, k=k_val)
        X_train_sel = selector.fit_transform(X_train_scaled, y_train)
        X_test_sel = selector.transform(X_test_scaled)

        selected_indices = selector.get_support(indices=True)
        selected_feature_names = [raw_feature_names[i] for i in selected_indices]

        # 10. SMOTE Class Balancing
        t_smote = time.time()
        smote = SMOTE(random_state=RANDOM_STATE)
        X_train_res, y_train_res = smote.fit_resample(X_train_sel, y_train)
        log_pipeline_step(
            logger, "smote", time.time() - t_smote, n_samples=len(X_train_res)
        )

        # 11. GridSearchCV Tuning
        t_train = time.time()
        param_grid_lr = {"C": [0.01, 0.1, 1, 10]}
        grid_lr = GridSearchCV(
            LogisticRegression(random_state=RANDOM_STATE, max_iter=1000),
            param_grid_lr,
            cv=5,
            scoring="roc_auc",
            n_jobs=-1,
        )
        grid_lr.fit(X_train_res, y_train_res)

        param_grid_rf = {"n_estimators": [50, 100, 200], "max_depth": [3, 5, 8]}
        grid_rf = GridSearchCV(
            RandomForestClassifier(random_state=RANDOM_STATE),
            param_grid_rf,
            cv=5,
            scoring="roc_auc",
            n_jobs=-1,
        )
        grid_rf.fit(X_train_res, y_train_res)

        param_grid_xgb = {
            "n_estimators": [50, 100],
            "max_depth": [3, 5],
            "learning_rate": [0.01, 0.05, 0.1],
        }
        grid_xgb = GridSearchCV(
            XGBClassifier(random_state=RANDOM_STATE, eval_metric="logloss"),
            param_grid_xgb,
            cv=5,
            scoring="roc_auc",
            n_jobs=-1,
        )
        grid_xgb.fit(X_train_res, y_train_res)
        log_pipeline_step(
            logger, "model_training", time.time() - t_train, n_samples=len(X_train_res)
        )

        models = {
            "Logistic Regression": grid_lr.best_estimator_,
            "Random Forest": grid_rf.best_estimator_,
            "XGBoost": grid_xgb.best_estimator_,
        }

        best_auc = -1.0
        best_model_name = None
        best_model = None

        for name, model in models.items():
            y_prob = model.predict_proba(X_test_sel)[:, 1]
            auc_score = roc_auc_score(y_test, y_prob)
            if auc_score > best_auc:
                best_auc = auc_score
                best_model_name = name
                best_model = model

        # 12. Extract Feature Importances / Top Drivers
        if hasattr(best_model, "feature_importances_"):
            importances = best_model.feature_importances_
        else:
            importances = np.abs(best_model.coef_[0])

        feature_imp_df = pd.DataFrame(
            {"Feature": selected_feature_names, "Importance": importances}
        ).sort_values(by="Importance", ascending=False)

        top_5_drivers = feature_imp_df.head(5)["Feature"].tolist()

        # 13. Export Predictions
        t_export = time.time()
        test_indices = y_test.index
        output_df = readable_df.loc[test_indices].copy()
        output_df["actual_churn"] = y_test.values
        output_df["predicted_churn"] = best_model.predict(X_test_sel)
        output_df["churn_probability"] = np.round(
            best_model.predict_proba(X_test_sel)[:, 1], 4
        )

        output_df.to_csv(output_path, index=False)
        log_pipeline_step(
            logger,
            "prediction_export",
            time.time() - t_export,
            n_samples=len(output_df),
        )

        # 14. Print Executive Console Summary
        print("\n" + "=" * 80)
        print("          CHURN RETENTION PIPELINE SUMMARY REPORT")
        print("=" * 80)
        print(f" Input Data File     : {data_path}")
        print(f" Predictions Saved To: {output_path}")
        print(f" Total Rows Evaluated: {len(output_df)}")
        print(f" Overall Churn Rate  : {churn_rate:.2%}")
        print(f" Best Model Selected : {best_model_name}")
        print(f" Test Set AUC-ROC    : {best_auc:.4f}")
        print(f" Top 5 Churn Drivers : {', '.join(top_5_drivers)}")
        print("=" * 80 + "\n")

        return True

    except Exception as e:
        logger.error(f"Pipeline execution failed with error: {str(e)}", exc_info=True)
        print(
            f"\n[ERROR] Pipeline Execution Failed: {str(e)}\nCheck 'churn_pipeline.log' for detailed stack trace."
        )
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(
        description="Run Customer Churn Machine Learning Pipeline"
    )
    default_path = (
        "data/customer_data.csv"
        if os.path.exists("data/customer_data.csv")
        else "customer_data.csv"
    )
    parser.add_argument(
        "--data",
        type=str,
        default=default_path,
        help=f"Path to input customer dataset CSV file (default: '{default_path}')",
    )
    args = parser.parse_args()

    run_pipeline(args.data)


if __name__ == "__main__":
    main()
