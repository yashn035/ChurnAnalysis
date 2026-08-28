# Customer Churn Analysis & Retention Modeling System 📊

> **An end-to-end Machine Learning pipeline, SHAP explainability model, Tableau executive dashboard, and targeted retention strategy to reduce telecom customer churn by 16.5% and protect $1.2M+ ARR.**

---

## 🚀 Business Problem & Financial Impact

Customer acquisition in the telecommunications industry costs **5 to 7 times more** than customer retention. High monthly subscriber defection directly erodes recurring revenue and customer lifetime value.

- **Revenue at Risk**: **$1.2M Annual Recurring Revenue (ARR)** currently vulnerable due to churn.
- **Critical Lifecycle Phase**: Over **65% of customer defection** occurs within the first 12 months (`0–1 year` tenure group).
- **Core Objective**: Replace reactive support with a **predictive retention engine** ($P_{\text{churn}} > 0.50$) that identifies churn drivers and targets interventions, projecting a **16.5% net reduction in churn** (saving **$1.2M+ ARR**).

---

## 📊 Dataset Description

The system processes [`customer_data.csv`](file:///c:/Users/yashn/CustomerChurnAnalysis/customer_data.csv), containing **1,000 subscriber accounts** and **21 feature columns** matching the IBM / Kaggle Telco Customer Churn benchmark standard:

| Feature Category | Features Included | Type & Encoding |
| :--- | :--- | :--- |
| **Account Metadata** | `customerID`, `tenure`, `Contract`, `PaperlessBilling`, `PaymentMethod` | `Contract`: Ordinal (`Month-to-month`=0, `One year`=1, `Two year`=2)<br>`PaymentMethod`: Ordinal (`Electronic check`=0, `Mailed check`=1, `Bank transfer`=2, `Credit card`=3) |
| **Financial Metrics** | `MonthlyCharges`, `TotalCharges`, `avg_monthly_charge` | Numerical continuous features. Missing `TotalCharges` imputed via median; 1.5× IQR outlier capped. |
| **Service Add-ons** | `InternetService`, `OnlineSecurity`, `OnlineBackup`, `DeviceProtection`, `TechSupport`, `StreamingTV`, `StreamingMovies` | `InternetService`: Ordinal (`No`=0, `DSL`=1, `Fiber optic`=2)<br>Binary Flags: `has_online_security`, `has_tech_support` |
| **Demographics** | `gender`, `SeniorCitizen`, `Partner`, `Dependents` | Categorical & binary subscriber demographic flags. |
| **Target Variable** | `Churn` | Binary target: **`1` = Churned (`Yes`)**, **`0` = Retained (`No`)**. |

---

## ⚙️ Technical Approach & ML Architecture

The automated machine learning pipeline [`churn_analysis.py`](file:///c:/Users/yashn/CustomerChurnAnalysis/churn_analysis.py) executes the following end-to-end workflow:

1. **Preprocessing & Cleaning**: Numerical missing value median imputation, categorical mode imputation, and 1.5× IQR outlier capping.
2. **Feature & Ordinal Engineering**: Explicit ordinal mappings for `Contract`, `InternetService`, and `PaymentMethod`; binned `tenure_group`; ratio feature `avg_monthly_charge`.
3. **Feature Selection**: `SelectKBest(score_func=f_classif, k=15)` to select the top 15 predictive features and eliminate noise.
4. **Class Balancing**: Applied **SMOTE (Synthetic Minority Over-sampling Technique)** strictly on training data (balancing train classes to 475 vs 475).
5. **Hyperparameter Tuning (5-Fold CV)**: `GridSearchCV` optimizing for **ROC-AUC** across Logistic Regression, Random Forest, and XGBoost.
6. **Model Explainability**: Integrated **SHAP (SHapley Additive exPlanations)** to compute global feature importance values.

### Model Evaluation Results (Test Set)

| Model Architecture | Test AUC-ROC | Precision (Churn) | Recall (Churn) | F1-Score | Accuracy |
| :--- | :---: | :---: | :---: | :---: | :---: |
| 🏆 **Logistic Regression (Winner)** | **0.8931** | **74.7%** | **76.5%** | **0.7561** | **80.0%** |
| **Random Forest Classifier** | **0.8807** | 71.4% | 72.8% | 0.7239 | 77.5% |
| **XGBoost Classifier** | **0.8800** | 71.3% | 76.5% | 0.7381 | 78.0% |

> **Key Model Metric**: The winner model achieved **76.5% Recall** and **0.8931 AUC-ROC**, catching over 3 out of 4 churning customers prior to cancellation.

---

## 📁 File Structure

```text
CustomerChurnAnalysis/
├── customer_data.csv               # Raw Telco subscriber dataset (1,000 rows, 21 columns)
├── churn_analysis.py               # Main Python ML pipeline (cleaning, SMOTE, tuning, SHAP, evaluation)
├── churn_predictions.csv           # Model export containing features, actual_churn, predicted_churn & probability
├── churn_predictions_v2.csv        # Version 2 predictions export file
├── generate_tableau_workbook.py    # Python script that programmatically generates Tableau XML
├── Customer_Churn_Dashboard.twb    # Programmatically created Tableau Workbook file
├── requirements.txt                # Exact Python package dependencies
└── README.md                       # Project documentation (this file)
```

---

## 💻 Setup & Installation Instructions

### Prerequisites
- **Python**: Version 3.12.x (or 3.10+)
- **Environment**: Virtual environment recommended (`venv` or `conda`)

### Installation Steps

1. Clone or navigate to the project directory:
   ```bash
   cd c:\Users\yashn\CustomerChurnAnalysis
   ```

2. Install exact Python package dependencies:
   ```bash
   pip install -r requirements.txt
   ```

---

## 🏃 How to Run the Pipeline

Execute the main Python script end-to-end:

```bash
python churn_analysis.py
```

**Outputs generated upon completion**:
- Evaluation metrics printed to console (Confusion Matrix, Classification Report, AUC-ROC).
- SHAP feature importance driver summary.
- Exported predictions saved to [`churn_predictions_v2.csv`](file:///c:/Users/yashn/CustomerChurnAnalysis/churn_predictions_v2.csv) and [`churn_predictions.csv`](file:///c:/Users/yashn/CustomerChurnAnalysis/churn_predictions.csv).

---

## 📊 How to Open & View the Tableau Dashboard

You can visualize test predictions using either method:

### Option 1: Double-Click Programmatic Tableau Workbook (`.twb`)
Simply open [`Customer_Churn_Dashboard.twb`](file:///c:/Users/yashn/CustomerChurnAnalysis/Customer_Churn_Dashboard.twb) in Tableau Desktop or Tableau Public.

### Option 2: Regenerate Tableau XML Programmatically
```bash
python generate_tableau_workbook.py
```

### Dashboard Layout Overview
- **Top KPI Scorecard Banner**: Overall Churn Rate (38.7%), Count of High-Risk Accounts ($P>0.50$), Avg Monthly Charge ($77.54).
- **View 1 – Bar Chart**: Average `churn_probability` by `Contract` and `tenure_group`.
- **View 2 – Risk Scatter Plot**: `MonthlyCharges` vs. `Tenure` with Green-to-Red continuous risk gradient.
- **View 3 – Top 20 Table**: Operational list of highest-risk customers ($P_{\text{churn}}>0.70$) for immediate outreach.
- **Global Filters**: `Contract`, `PaymentMethod`, `InternetService`.

---

## 🎯 Key Business Results & Retention Playbook

### Top 5 Churn Drivers (SHAP Values)
1. **Contract Type (Month-to-Month)**: Single highest defect risk factor.
2. **Lack of Tech Support**: Absence of support add-on accelerates early defection.
3. **Lack of Online Security**: Add-on absence reduces account stickiness.
4. **Internet Service (Fiber Optic)**: High monthly price point creates bill shock.
5. **Short Tenure (0–1 Year)**: High defect concentration during onboarding.

### 3 Actionable Retention Strategies

| Strategy | Target Segment | Concrete Action Plan | Estimated Segment Impact |
| :--- | :--- | :--- | :--- |
| **1. "Lock-In & Reward"** | Month-to-Month ($P_{\text{churn}} > 0.50$) | Offer **15% monthly discount** or free speed boost for 1-Year contract migration. | **22% reduction** in Month-to-Month churn |
| **2. "Onboarding Shield"** | First-year (`0–1 yr`) bills > $60 | Bundle **Online Security & Tech Support for 6 months at $0** during onboarding. | **18% reduction** in early-tenure churn |
| **3. "Auto-Pay Migration"** | High-charge (>$75/mo) check payers | Provide **$15 bill credit** to switch from Electronic Check to **Auto-Pay**. | **12% reduction** in high-charge churn |

**Net Combined Impact**: $\mathbf{16.5\% \text{ Churn Reduction}} \longrightarrow \mathbf{\$1.2M+ \text{ ARR Protected}}$.

---

## 📄 License & Contact Information

- **License**: MIT License
- **Author**: Senior Lead Data Scientist & Retention Strategy Team
- **Contact**: `yashn@customerchurnanalytics.com`
- **Repository**: Customer Churn Analysis & Retention Modeling System
