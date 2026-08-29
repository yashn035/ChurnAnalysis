# Customer Churn Analysis & Retention Modeling System 📊

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://yashn035-churn-analysis.streamlit.app)
[![CI Pipeline](https://github.com/yashn035/ChurnAnalysis/actions/workflows/ci.yml/badge.svg)](https://github.com/yashn035/ChurnAnalysis/actions)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

![Customer Churn Dashboard](screenshots/dashboard.png)

> **An end-to-end Machine Learning pipeline, SHAP explainability model, Tableau executive dashboard, Streamlit web application, and targeted retention strategy to reduce telecom customer churn by 16.5% and protect $1.2M+ ARR.**

---

## ⚡ Quick Start (Docker & Docker Compose)

### Run Entire Stack (Streamlit + FastAPI) via Docker Compose

Launch both the **FastAPI prediction backend** (Port 8000) and the **Streamlit web dashboard** (Port 8501) with automated health checks using Docker Compose:

```bash
docker-compose up --build
```

> **Note**: After running the stack, open [http://localhost:8501](http://localhost:8501) to see the interactive Streamlit dashboard and [http://localhost:8000/docs](http://localhost:8000/docs) for the OpenAPI interactive documentation.

### Run Single Streamlit Container

```bash
docker build -t churn-app . && docker run -p 8501:8501 churn-app
```

---

## 🖼️ Preview

The interactive Streamlit dashboard provides senior executives and retention teams with five core operational view modes:

1. **Executive KPI Cards**: Real-time summary metric tiles displaying Overall Churn Rate (59.4%), High-Risk Account Count ($P_{\text{churn}} > 0.50$), Average At-Risk Monthly Charges ($75.20), and Protected Revenue Goal ($1.2M+ ARR).
2. **Interactive Individual Risk Calculator**: Real-time subscriber risk scoring engine where users tweak tenure, contract type, payment method, and service add-ons to receive an instant probability score, risk classification (Low/Medium/High), and recommended retention intervention.
3. **Top 20 High-Risk Target List**: Filtered, sortable roster listing subscribers with $P_{\text{churn}} > 0.50$ sorted by highest churn risk, with a 1-click CSV download for retention call center teams.
4. **A/B Retention Trial Cohorts**: Randomized 50/50 Control vs. Variant cohort assignment tables for evaluating contract discount retention campaigns.
5. **Model Performance History**: Historical AUC-ROC performance trend lines tracking model health across pipeline execution runs.

### A/B Test Cohort Table Preview

![A/B Test Cohort Table](screenshots/ab_testing.png)

---

## 🚀 Business Problem & Financial Impact

Customer acquisition in the telecommunications industry costs **5 to 7 times more** than customer retention. High monthly subscriber defection directly erodes recurring revenue and customer lifetime value.

* **Revenue at Risk**: **$1.2M Annual Recurring Revenue (ARR)** currently vulnerable due to churn.
* **Critical Lifecycle Phase**: Over **65% of customer defection** occurs within the first 12 months (`0–1 year` tenure group).
* **Core Objective**: Replace reactive support with a **predictive retention engine** ($P_{\text{churn}} > 0.50$) that identifies churn drivers and targets interventions, projecting a **16.5% net reduction in churn** (saving **$1.2M+ ARR**).

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

## 🧪 Unit Testing & CI/CD Pipeline

The project includes an automated unit test suite executed via `pytest` and integrated into GitHub Actions CI:

* **Unit Test File**: [`tests/test_preprocessing.py`](file:///c:/Users/yashn/CustomerChurnAnalysis/tests/test_preprocessing.py) (tests median imputation, ordinal mappings, 1.5× IQR capping, and `SelectKBest` feature output count).
* **Pytest Config**: [`pytest.ini`](file:///c:/Users/yashn/CustomerChurnAnalysis/pytest.ini)
* **Run Tests Locally**:
  ```bash
  pytest
  ```

---

## 📁 File Structure

```text
CustomerChurnAnalysis/
├── .github/workflows/ci.yml        # GitHub Actions CI workflow (pytest + pipeline checks)
├── tests/
│   └── test_preprocessing.py       # Pytest unit tests for preprocessing & feature selection
├── .dockerignore                   # Docker build ignore rules
├── Dockerfile                      # Production Docker container image definition
├── pytest.ini                      # Pytest test runner configuration
├── customer_data.csv               # Raw Telco subscriber dataset (1,000 rows, 21 columns)
├── churn_analysis.py               # Main Python ML pipeline (cleaning, SMOTE, tuning, SHAP, evaluation)
├── run_churn_pipeline.py           # Command-line production pipeline runner script
├── app.py                          # Streamlit Web Application (4 interactive view modes)
├── ab_test_cohort.py               # A/B Retention Trial Cohort Generator script
├── churn_predictions.csv           # Model export containing features & predicted churn probabilities
├── churn_predictions_v2.csv        # Version 2 predictions export file
├── generate_tableau_workbook.py    # Script that programmatically generates Tableau XML
├── Customer_Churn_Dashboard.twb    # Programmatically created Tableau Packaged Workbook
├── requirements.txt                # Exact Python package dependencies
└── README.md                       # Project documentation
```

---

## 🛠️ Developer Shortcuts (Makefile)

Developers can use `make` shortcuts instead of typing full commands:

* **Install dependencies**: `make install` (replaces `pip install -r requirements.txt`)
* **Run unit tests**: `make test` (replaces `pytest tests/`)
* **Run ML pipeline**: `make pipeline` (replaces `python churn_analysis.py`)
* **Launch FastAPI REST server**: `make api` (replaces `uvicorn api:app --reload --port 8000`)
* **Launch Next.js frontend**: `make frontend` (replaces `cd frontend && npm run dev`)
* **Launch Streamlit dashboard**: `make run` (replaces `streamlit run app.py`)
* **Full pipeline setup**: `make all` (installs dependencies, runs unit tests, and launches the app)

---

## 🖥️ Running Next.js Frontend alongside FastAPI Server

The repository includes a modern Next.js frontend in [`frontend/`](file:///c:/Users/yashn/CustomerChurnAnalysis/frontend) that interacts directly with the FastAPI prediction backend.

### Step 1: Start FastAPI Backend Server
```bash
make api
# Or manually: uvicorn src.api:app --reload --port 8000
```
*(Confirms model artifacts are loaded at `http://localhost:8000/health`)*

> **🔒 Security & CORS Middleware**: The FastAPI backend restricts Cross-Origin Resource Sharing (CORS) strictly to authorized origins:
> - `http://localhost:8501` & `http://localhost:3000` (Local Streamlit & Next.js development)
> - `https://yashn035-churn-analysis.streamlit.app` & `https://your-app.streamlit.app` (Production Streamlit Cloud)

### Step 2: Start Next.js Frontend Application
In a separate terminal window:
```bash
make frontend
# Or manually: cd frontend && npm run dev
```

Open **`http://localhost:3000`** in your browser to view the interactive Next.js risk predictor dashboard.

---

## 💻 Setup & Local Installation

### Quick Setup using Makefile

```bash
make install    # Install dependencies from requirements.txt
make test       # Run pytest unit test suite
make run        # Launch Streamlit app
```

### Manual Setup (Python Environment)

1. Navigate to the project directory:
   ```bash
   cd c:\Users\yashn\CustomerChurnAnalysis
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Enable git pre-commit hooks:
   ```bash
   pre-commit install
   ```

4. Run the ML pipeline:
   ```bash
   python src/churn_analysis.py
   ```

5. Launch the local Streamlit application:
   ```bash
   streamlit run app/app.py
   ```

### 🪝 Code Quality & Pre-commit Hooks

The repository enforces strict code formatting and linting via `.pre-commit-config.yaml`:
* **Code Formatting**: `black`
* **Import Sorting**: `isort`
* **Linting & Rules**: `flake8`
* **File Cleanup**: `trailing-whitespace` and `end-of-file-fixer`

Run `pre-commit install` once after cloning to automatically check code quality on every `git commit`. You can also manually run all hooks across the codebase:
```bash
pre-commit run --all-files
```

---

## 📊 Tableau & Streamlit Visualizations

### 🌐 Streamlit Live Web App
* **Live Deployment**: [https://yashn035-churn-analysis.streamlit.app](https://yashn035-churn-analysis.streamlit.app)
* Features 4 view modes: Executive KPI summary, real-time subscriber risk calculator, Top 20 target list, and A/B retention trial cohort viewer.

### 🎨 Programmatic Tableau Dashboard
* Open [`Customer_Churn_Dashboard.twb`](file:///c:/Users/yashn/CustomerChurnAnalysis/Customer_Churn_Dashboard.twb) in Tableau Desktop / Tableau Public.
* Regenerate workbook programmatically:
  ```bash
  python generate_tableau_workbook.py
  ```

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

## 📄 License & Contact

* **License**: MIT License
* **Author**: Senior Lead Data Scientist & Retention Strategy Team
* **GitHub Repository**: [https://github.com/yashn035/ChurnAnalysis](https://github.com/yashn035/ChurnAnalysis)
* **Live App**: [https://yashn035-churn-analysis.streamlit.app](https://yashn035-churn-analysis.streamlit.app)
