"""
Customer Churn Analysis and Retention Modeling Pipeline (v2 - High AUC)
Senior Data Scientist Implementation with Benchmarking & Modular Pipeline.
"""

import functools
import json
import os
import time
import tracemalloc
from datetime import datetime
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, precision_score, recall_score, accuracy_score
import shap
import warnings

from json_logger import get_json_logger, log_pipeline_step

warnings.filterwarnings('ignore')

RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

# Global list storing benchmark metric dictionaries
BENCHMARK_RESULTS = []


def benchmark(func):
    """Decorator measuring execution time (seconds) and peak memory usage (MB)."""
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        tracemalloc.start()
        start_time = time.perf_counter()
        
        result = func(*args, **kwargs)
        
        end_time = time.perf_counter()
        current_mem, peak_mem = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        
        exec_time = round(end_time - start_time, 4)
        peak_mb = round(peak_mem / (1024 * 1024), 4)
        
        BENCHMARK_RESULTS.append({
            "function": func.__name__,
            "execution_time_seconds": exec_time,
            "peak_memory_mb": peak_mb
        })
        return result
    return wrapper


def generate_synthetic_data(filepath='data/customer_data.csv', num_samples=1000):
    """Generate synthetic Telco Customer Churn dataset matching Kaggle benchmark."""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    if os.path.exists(filepath):
        print(f"[INFO] Dataset '{filepath}' already exists. Skipping synthetic generation.")
        return pd.read_csv(filepath)

    print(f"[INFO] Generating high-fidelity Telco dataset ({num_samples} rows) -> '{filepath}'...")
    
    customer_ids = [f"{np.random.randint(1000, 9999)}-{chr(65+i%26)}{chr(65+(i*3)%26)}{chr(65+(i*7)%26)}" for i in range(num_samples)]
    gender = np.random.choice(['Male', 'Female'], size=num_samples)
    senior_citizen = np.random.choice([0, 1], size=num_samples, p=[0.84, 0.16])
    partner = np.random.choice(['Yes', 'No'], size=num_samples, p=[0.48, 0.52])
    dependents = np.random.choice(['Yes', 'No'], size=num_samples, p=[0.30, 0.70])
    
    tenure = np.random.randint(1, 73, size=num_samples)
    phone_service = np.random.choice(['Yes', 'No'], size=num_samples, p=[0.90, 0.10])
    multiple_lines = np.where(phone_service == 'No', 'No phone service', np.random.choice(['Yes', 'No'], size=num_samples, p=[0.45, 0.55]))
    
    internet_service = np.random.choice(['DSL', 'Fiber optic', 'No'], size=num_samples, p=[0.34, 0.44, 0.22])
    
    online_security = np.where(internet_service == 'No', 'No internet service', np.random.choice(['Yes', 'No'], size=num_samples, p=[0.28, 0.72]))
    online_backup = np.where(internet_service == 'No', 'No internet service', np.random.choice(['Yes', 'No'], size=num_samples, p=[0.34, 0.66]))
    device_protection = np.where(internet_service == 'No', 'No internet service', np.random.choice(['Yes', 'No'], size=num_samples, p=[0.34, 0.66]))
    tech_support = np.where(internet_service == 'No', 'No internet service', np.random.choice(['Yes', 'No'], size=num_samples, p=[0.29, 0.71]))
    streaming_tv = np.where(internet_service == 'No', 'No internet service', np.random.choice(['Yes', 'No'], size=num_samples, p=[0.38, 0.62]))
    streaming_movies = np.where(internet_service == 'No', 'No internet service', np.random.choice(['Yes', 'No'], size=num_samples, p=[0.39, 0.61]))
    
    contract = np.random.choice(['Month-to-month', 'One year', 'Two year'], size=num_samples, p=[0.55, 0.24, 0.21])
    paperless_billing = np.random.choice(['Yes', 'No'], size=num_samples, p=[0.59, 0.41])
    payment_method = np.random.choice([
        'Electronic check', 'Mailed check', 'Bank transfer (automatic)', 'Credit card (automatic)'
    ], size=num_samples, p=[0.34, 0.23, 0.22, 0.21])
    
    base_charge = np.where(internet_service == 'No', 20.0, np.where(internet_service == 'DSL', 50.0, 80.0))
    addons_count = ((online_security == 'Yes').astype(int) + (online_backup == 'Yes').astype(int) + 
                    (device_protection == 'Yes').astype(int) + (tech_support == 'Yes').astype(int) + 
                    (streaming_tv == 'Yes').astype(int) + (streaming_movies == 'Yes').astype(int))
    monthly_charges = np.round(base_charge + addons_count * 6.5 + np.random.normal(0, 3, size=num_samples), 2)
    monthly_charges = np.clip(monthly_charges, 18.25, 118.75)
    
    total_charges = np.round(monthly_charges * tenure + np.random.normal(0, 25, size=num_samples), 2)
    total_charges = np.clip(total_charges, 18.25, 8684.80)
    
    nan_indices = np.random.choice(num_samples, size=int(num_samples * 0.01), replace=False)
    total_charges[nan_indices] = np.nan
    
    churn_logit = (
        2.2 * (contract == 'Month-to-month').astype(int) +
        1.4 * (internet_service == 'Fiber optic').astype(int) -
        1.2 * (tech_support == 'Yes').astype(int) -
        1.0 * (online_security == 'Yes').astype(int) +
        0.8 * (payment_method == 'Electronic check').astype(int) -
        0.04 * tenure +
        0.02 * monthly_charges -
        1.5
    )
    churn_prob = 1 / (1 + np.exp(-churn_logit))
    churn = np.where(np.random.rand(num_samples) < churn_prob, 'Yes', 'No')

    df_synth = pd.DataFrame({
        'customerID': customer_ids, 'gender': gender, 'SeniorCitizen': senior_citizen,
        'Partner': partner, 'Dependents': dependents, 'tenure': tenure,
        'PhoneService': phone_service, 'MultipleLines': multiple_lines,
        'InternetService': internet_service, 'OnlineSecurity': online_security,
        'OnlineBackup': online_backup, 'DeviceProtection': device_protection,
        'TechSupport': tech_support, 'StreamingTV': streaming_tv,
        'StreamingMovies': streaming_movies, 'Contract': contract,
        'PaperlessBilling': paperless_billing, 'PaymentMethod': payment_method,
        'MonthlyCharges': monthly_charges, 'TotalCharges': total_charges,
        'Churn': churn
    })
    
    df_synth.to_csv(filepath, index=False)
    print(f"[INFO] Dataset successfully created and saved to '{filepath}'.\n")
    return df_synth


@benchmark
def load_data(data_path):
    """Load dataset and extract customer IDs."""
    print("\n--- STEP 1: Loading Dataset ---")
    generate_synthetic_data(data_path)
    df = pd.read_csv(data_path)
    print(f"Dataset Loaded. Initial Shape: {df.shape}")
    customer_ids = df['customerID'].copy()
    df = df.drop(columns=['customerID'])
    print("Dropped column: customerID")
    return df, customer_ids


@benchmark
def preprocess_data(df, customer_ids):
    """Perform data cleaning, imputations, feature engineering, and outlier capping."""
    print("\n--- STEP 2 & 3: Data Cleaning & Feature Engineering ---")
    if df['TotalCharges'].dtype == 'object':
        df['TotalCharges'] = pd.to_numeric(df['TotalCharges'].astype(str).str.strip(), errors='coerce')
    
    if df['TotalCharges'].isnull().sum() > 0:
        median_val = df['TotalCharges'].median()
        df['TotalCharges'] = df['TotalCharges'].fillna(median_val)
        print(f"Imputed missing numeric values in 'TotalCharges' with median: {median_val}")

    cat_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
    for col in cat_cols:
        if col != 'Churn' and df[col].isnull().sum() > 0:
            mode_val = df[col].mode()[0]
            df[col] = df[col].fillna(mode_val)

    df['Churn'] = df['Churn'].astype(str).str.strip().map({'Yes': 1, 'No': 0, '1': 1, '0': 0})

    readable_df = df.copy()
    readable_df['customerID'] = customer_ids
    readable_df['tenure_group'] = pd.cut(
        readable_df['tenure'],
        bins=[-1, 12, 24, 48, 72, 100],
        labels=['0-12 Mo', '12-24 Mo', '24-48 Mo', '48-72 Mo', '72+ Mo']
    )

    df['Contract'] = df['Contract'].map({'Month-to-month': 0, 'One year': 1, 'Two year': 2})
    df['InternetService'] = df['InternetService'].map({'No': 0, 'DSL': 1, 'Fiber optic': 2})
    df['PaymentMethod'] = df['PaymentMethod'].map({
        'Electronic check': 0,
        'Mailed check': 1,
        'Bank transfer (automatic)': 2,
        'Credit card (automatic)': 3
    })
    
    df['tenure_group'] = pd.cut(
        df['tenure'],
        bins=[0, 12, 24, 48, 72],
        labels=[0, 1, 2, 3],
        include_lowest=True
    ).astype(int)
    
    df['avg_monthly_charge'] = np.where(df['tenure'] > 0, df['TotalCharges'] / df['tenure'], 0.0)
    df['has_online_security'] = (df['OnlineSecurity'] == 'Yes').astype(int)
    df['has_tech_support'] = (df['TechSupport'] == 'Yes').astype(int)
    
    object_columns = df.select_dtypes(include=['object']).columns.tolist()
    for col in object_columns:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))

    outlier_cols = ['MonthlyCharges', 'TotalCharges']
    for col in outlier_cols:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        df[col] = df[col].clip(lower=lower_bound, upper=upper_bound)

    return df, readable_df


@benchmark
def prepare_features_and_split(df, customer_ids):
    """Perform train/test split, feature scaling, SelectKBest, and SMOTE resampling."""
    print("\n--- STEP 5 & 6: Train-Test Split, Scaling & SMOTE ---")
    X = df.drop(columns=['Churn'])
    y = df['Churn']
    
    X_train, X_test, y_train, y_test, ids_train, ids_test = train_test_split(
        X, y, customer_ids, test_size=0.20, stratify=y, random_state=RANDOM_STATE
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    raw_feature_names = X.columns.tolist()

    selector = SelectKBest(score_func=f_classif, k=15)
    X_train_sel = selector.fit_transform(X_train_scaled, y_train)
    X_test_sel = selector.transform(X_test_scaled)
    
    selected_indices = selector.get_support(indices=True)
    selected_feature_names = [raw_feature_names[i] for i in selected_indices]

    smote = SMOTE(random_state=RANDOM_STATE)
    X_train_res, y_train_res = smote.fit_resample(X_train_sel, y_train)

    return X_train_res, y_train_res, X_test_sel, y_test, scaler, selector, selected_feature_names


@benchmark
def train_model(X_train_res, y_train_res):
    """Tune Logistic Regression, Random Forest, and XGBoost via GridSearchCV."""
    print("\n--- STEP 7: Model Training & Grid Search CV ---")
    print("Tuning Logistic Regression...")
    param_grid_lr = {'C': [0.01, 0.1, 1, 10]}
    grid_lr = GridSearchCV(
        LogisticRegression(random_state=RANDOM_STATE, max_iter=1000),
        param_grid_lr, cv=5, scoring='roc_auc', n_jobs=-1
    )
    grid_lr.fit(X_train_res, y_train_res)
    best_lr = grid_lr.best_estimator_

    print("Tuning Random Forest Classifier...")
    param_grid_rf = {'n_estimators': [50, 100, 200], 'max_depth': [3, 5, 8]}
    grid_rf = GridSearchCV(
        RandomForestClassifier(random_state=RANDOM_STATE),
        param_grid_rf, cv=5, scoring='roc_auc', n_jobs=-1
    )
    grid_rf.fit(X_train_res, y_train_res)
    best_rf = grid_rf.best_estimator_

    print("Tuning XGBoost Classifier...")
    param_grid_xgb = {'n_estimators': [50, 100], 'max_depth': [3, 5], 'learning_rate': [0.01, 0.05, 0.1]}
    grid_xgb = GridSearchCV(
        XGBClassifier(random_state=RANDOM_STATE, eval_metric='logloss'),
        param_grid_xgb, cv=5, scoring='roc_auc', n_jobs=-1
    )
    grid_xgb.fit(X_train_res, y_train_res)
    best_xgb = grid_xgb.best_estimator_

    models = {
        'Logistic Regression': best_lr,
        'Random Forest': best_rf,
        'XGBoost': best_xgb
    }
    return models


@benchmark
def evaluate_and_export(models, X_test_sel, y_test, readable_df, selected_feature_names, X_train_res, scaler, selector):
    """Evaluate models, compute SHAP, export CSV predictions, metrics history, and pkl files."""
    print("\n--- STEP 8-10: Test Set Evaluation & Export ---")
    best_auc = -1.0
    best_model_name = None
    best_model = None

    for name, model in models.items():
        y_pred = model.predict(X_test_sel)
        y_prob = model.predict_proba(X_test_sel)[:, 1]
        auc_score = roc_auc_score(y_test, y_prob)

        print(f"\n>>> Model: {name} (Test AUC-ROC: {auc_score:.4f}) <<<")
        print("Confusion Matrix:")
        print(confusion_matrix(y_test, y_pred))

        if auc_score > best_auc:
            best_auc = auc_score
            best_model_name = name
            best_model = model

    print(f"\n[WINNER] Best Performing Model on Test Set: '{best_model_name}' with AUC-ROC = {best_auc:.4f}")

    if hasattr(best_model, 'feature_importances_'):
        importances = best_model.feature_importances_
    else:
        importances = np.abs(best_model.coef_[0])

    feature_imp_df = pd.DataFrame({
        'Feature': selected_feature_names,
        'Importance': importances
    }).sort_values(by='Importance', ascending=False)

    # SHAP calculation
    try:
        if best_model_name in ['XGBoost', 'Random Forest']:
            explainer = shap.TreeExplainer(best_model)
            shap_values = explainer.shap_values(X_test_sel)
            shap_vals = shap_values[1] if isinstance(shap_values, list) else shap_values
            mean_shap = np.abs(shap_vals).mean(axis=0)
        else:
            explainer = shap.LinearExplainer(best_model, X_train_res)
            shap_values = explainer.shap_values(X_test_sel)
            mean_shap = np.abs(shap_values).mean(axis=0)
        
        shap_df = pd.DataFrame({
            'Feature': selected_feature_names,
            'Mean_|SHAP|_Value': mean_shap
        }).sort_values(by='Mean_|SHAP|_Value', ascending=False)
        print("\nTop 5 Churn Drivers according to Mean |SHAP| Values:")
        print(shap_df.head(5).to_string(index=False))
    except Exception as e:
        print(f"SHAP calculation note: {e}")

    test_indices = y_test.index
    output_df = readable_df.loc[test_indices].copy()
    output_df['actual_churn'] = y_test.values
    output_df['predicted_churn'] = best_model.predict(X_test_sel)
    output_df['churn_probability'] = np.round(best_model.predict_proba(X_test_sel)[:, 1], 4)

    output_df.to_csv('churn_predictions_v2.csv', index=False)
    output_df.to_csv('churn_predictions.csv', index=False)
    print("Successfully saved test predictions to 'churn_predictions_v2.csv' & 'churn_predictions.csv'.")

    y_pred_best = best_model.predict(X_test_sel)
    best_precision = precision_score(y_test, y_pred_best)
    best_recall = recall_score(y_test, y_pred_best)
    best_accuracy = accuracy_score(y_test, y_pred_best)
    train_shape = f"{X_train_res.shape[0]}x{X_train_res.shape[1]}"
    timestamp_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    metrics_df = pd.DataFrame([{
        'timestamp': timestamp_str,
        'AUC': np.round(best_auc, 4),
        'precision': np.round(best_precision, 4),
        'recall': np.round(best_recall, 4),
        'accuracy': np.round(best_accuracy, 4),
        'training_data_shape': train_shape
    }])

    metrics_file = 'metrics_history.csv'
    if not os.path.exists(metrics_file):
        metrics_df.to_csv(metrics_file, index=False)
    else:
        metrics_df.to_csv(metrics_file, mode='a', header=False, index=False)
    print(f"Appended current model metrics to '{metrics_file}'.")

    joblib.dump(best_model, 'model.pkl')
    joblib.dump(scaler, 'scaler.pkl')
    joblib.dump(selector, 'selector.pkl')
    print("Successfully saved 'model.pkl', 'scaler.pkl', and 'selector.pkl'.")

    return best_model, best_auc, feature_imp_df


def print_and_save_benchmarks(file_path="benchmarks.json"):
    """Print ASCII benchmark summary table to stdout and save to JSON."""
    print("\n" + "=" * 80)
    print("                 BENCHMARK PERFORMANCE SUMMARY TABLE")
    print("=" * 80)
    print(f"{'Function Name':<30} | {'Execution Time (s)':<20} | {'Peak Memory (MB)':<18}")
    print("-" * 80)
    for b in BENCHMARK_RESULTS:
        print(f"{b['function']:<30} | {b['execution_time_seconds']:<20.4f} | {b['peak_memory_mb']:<18.4f}")
    print("=" * 80)

    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(BENCHMARK_RESULTS, f, indent=2)
    print(f"[INFO] Saved benchmark metrics to '{file_path}'.\n")


def main():
    logger = get_json_logger("churn_analysis", "logs/pipeline.jsonl")
    print("=" * 80)
    print("      CUSTOMER CHURN ANALYSIS & RETENTION MODELING PIPELINE (V2 - HIGH AUC)")
    print("=" * 80)

    data_path = 'data/customer_data.csv' if os.path.exists('data/customer_data.csv') else 'customer_data.csv'

    # Step 1: Load Data
    t0 = time.time()
    df, customer_ids = load_data(data_path)
    log_pipeline_step(logger, 'data_load', time.time() - t0, n_samples=len(df))

    # Step 2 & 3: Preprocess & Feature Engineering
    t0 = time.time()
    df, readable_df = preprocess_data(df, customer_ids)
    log_pipeline_step(logger, 'feature_engineering', time.time() - t0, n_samples=len(df))

    # Step 5 & 6: Split, Scaling, Feature Selection & SMOTE
    t0 = time.time()
    X_train_res, y_train_res, X_test_sel, y_test, scaler, selector, selected_feature_names = prepare_features_and_split(df, customer_ids)
    log_pipeline_step(logger, 'smote', time.time() - t0, n_samples=len(X_train_res))

    # Step 7: Train & Tune Models
    t0 = time.time()
    models = train_model(X_train_res, y_train_res)
    log_pipeline_step(logger, 'model_training', time.time() - t0, n_samples=len(X_train_res))

    # Step 8-10: Evaluate & Export
    t0 = time.time()
    best_model, best_auc, feature_imp_df = evaluate_and_export(
        models, X_test_sel, y_test, readable_df, selected_feature_names, X_train_res, scaler, selector
    )
    log_pipeline_step(logger, 'prediction_export', time.time() - t0, n_samples=len(y_test))

    # Print and Save Benchmark Summary
    print_and_save_benchmarks("benchmarks.json")


if __name__ == '__main__':
    main()
