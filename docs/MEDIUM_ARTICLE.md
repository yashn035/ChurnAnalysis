# How I Built an End-to-End Customer Churn Retention System with 0.893 AUC and Streamlit

*A complete guide to machine learning pipeline engineering, SMOTE class balancing, SHAP interpretability, Tableau dashboarding, and interactive Streamlit deployment.*

---

## 📌 Introduction

In the subscription telecommunications industry, customer acquisition costs **5 to 7 times more** than customer retention. High monthly subscriber defection directly erodes Monthly Recurring Revenue (MRR) and decreases Net Promoter Scores (NPS).

In this article, I walk through how I built a production-ready **Customer Churn Risk & Retention System** that:
- Predicts subscriber churn with an **AUC-ROC score of 0.8931** and **76.5% recall**.
- Uncovers root causes of defection using **SHAP (SHapley Additive exPlanations)**.
- Visualizes executive risk metrics in an interactive **Tableau Dashboard** and a **Streamlit Web App**.
- Generates randomized **A/B testing cohorts** to validate retention strategies, projecting a **16.5% net reduction in churn** (saving **$1.2M+ ARR**).

---

## 🛠️ The Architecture

The system consists of 5 modular components:

```
┌──────────────────────┐     ┌──────────────────────┐     ┌──────────────────────┐
│ 1. Data & Preprocess │ ──► │ 2. Machine Learning  │ ──► │ 3. Model Explainability│
│ (IQR, Scaling)       │     │ (SMOTE, GridSearchCV)│     │ (SHAP Values)        │
└──────────────────────┘     └──────────────────────┘     └──────────────────────┘
                                                                     │
                                                                     ▼
┌──────────────────────┐     ┌──────────────────────┐     ┌──────────────────────┐
│ 5. Deployment        │ ◄── │ 4. Streamlit & Tableau│ ◄── │ 3. Retention Plays   │
│ (Streamlit Cloud, CRM)│     │ (Interactive Web App)│     │ (A/B Test Cohorts)   │
└──────────────────────┘     └──────────────────────┘     └──────────────────────┘
```

---

## ⚙️ Step 1: Preprocessing & Ordinal Feature Engineering

Machine learning models struggle when raw categorical categories lack explicit structural ordering. In telecom datasets, features like `Contract` (`Month-to-month`, `One year`, `Two year`) have natural ordinal relationships.

```python
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

# Ordinal Encodings
df['Contract'] = df['Contract'].map({'Month-to-month': 0, 'One year': 1, 'Two year': 2})
df['InternetService'] = df['InternetService'].map({'No': 0, 'DSL': 1, 'Fiber optic': 2})
df['PaymentMethod'] = df['PaymentMethod'].map({
    'Electronic check': 0, 'Mailed check': 1, 'Bank transfer (automatic)': 2, 'Credit card (automatic)': 3
})

# Feature Engineering
df['tenure_group'] = pd.cut(df['tenure'], bins=[0, 12, 24, 48, 72], labels=[0, 1, 2, 3], include_lowest=True).astype(int)
df['avg_monthly_charge'] = np.where(df['tenure'] > 0, df['TotalCharges'] / df['tenure'], 0.0)
df['has_online_security'] = (df['OnlineSecurity'] == 'Yes').astype(int)
df['has_tech_support'] = (df['TechSupport'] == 'Yes').astype(int)
```

---

## 🧪 Step 2: Feature Selection & Class Balancing via SMOTE

To prevent data leakage, **`SelectKBest` feature selection** and **SMOTE class balancing** were applied strictly to the training split:

```python
from sklearn.feature_selection import SelectKBest, f_classif
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, stratify=y, random_state=42)

# Standardize Features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Select Top 15 Features
selector = SelectKBest(score_func=f_classif, k=15)
X_train_sel = selector.fit_transform(X_train_scaled, y_train)
X_test_sel = selector.transform(X_test_scaled)

# Class Balancing via SMOTE (Training Set Only)
smote = SMOTE(random_state=42)
X_train_res, y_train_res = smote.fit_resample(X_train_sel, y_train)
```

---

## 📈 Step 3: Model Tuning & Evaluation

Using 5-fold cross-validation (`GridSearchCV`), three distinct model families were evaluated:

| Model Architecture | Test AUC-ROC | Precision | Recall | F1-Score |
| :--- | :---: | :---: | :---: | :---: |
| 🏆 **Logistic Regression (Winner)** | **0.8931** | **74.7%** | **76.5%** | **0.7561** |
| **Random Forest Classifier** | **0.8807** | 71.4% | 72.8% | 0.7239 |
| **XGBoost Classifier** | **0.8800** | 71.3% | 76.5% | 0.7381 |

The winner model achieves **76.5% Recall**, capturing **over 3 out of every 4 churning subscribers** prior to cancellation.

---

## 🧠 Step 4: SHAP Interpretability & Key Churn Drivers

Using SHAP values, we uncovered the primary root causes of defection:
1. **Contract Type (Month-to-Month)**: Highest positive churn risk factor due to zero exit friction.
2. **Lack of Tech Support**: Absence of support add-on accelerates early defection.
3. **Lack of Online Security**: Add-on absence decreases account stickiness.
4. **Fiber Optic Pricing**: High price point creates monthly bill shock.
5. **Short Tenure (0–1 Year)**: High vulnerability during early onboarding.

---

## 📊 Step 5: Interactive Streamlit Web Application

To make predictions actionable for Customer Success managers, I deployed an interactive **Streamlit Dashboard (`app.py`)** with 4 modes:
- **Executive KPIs**: Real-time summary metrics and risk distribution charts.
- **Risk Calculator**: Sliders and dropdowns to calculate churn probability for any custom subscriber profile in real-time.
- **Top 20 Target List**: Operational table of high-risk accounts ($P_{\text{churn}} > 0.50$) for concierge outreach.
- **A/B Test Cohort Viewer**: Side-by-side comparison of Control vs. Variant trial groups.

```bash
pip install streamlit
streamlit run app.py
```

---

## 🎯 Strategic Business Impact

By deploying three targeted retention initiatives:
1. **"Lock-In & Reward"**: 15% discount for 1-year contract migration (22% segment churn reduction).
2. **"Onboarding Shield"**: Free Tech Support & Security for 6 months (18% segment churn reduction).
3. **"Auto-Pay Migration"**: $15 bill credit to switch from check to Auto-Pay (12% segment churn reduction).

$$\text{Net Churn Reduction} = (55\% \times 22\%) + (30\% \times 18\%) + (15\% \times 12\%) = \mathbf{16.5\%}$$

**Financial Return**: Protects **$1.2M+ Annual Recurring Revenue (ARR)** and extends average customer lifetime from 18 to 26+ months.

---

## 🔗 Repository & Code
Check out the complete production codebase, CLI runners, SQL extractors, Tableau XML generators, and deployment guides on GitHub!
