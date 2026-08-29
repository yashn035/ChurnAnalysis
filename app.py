"""
Streamlit Web Application: Customer Churn Risk & Retention Dashboard
Interactive ML prediction, risk scoring, feature analysis, and A/B test cohort viewer.
"""

import os
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

# Page configuration
st.set_page_config(
    page_title="Customer Churn Risk & Retention System",
    page_icon="📊",
    layout="wide"
)

# Title & Subtitle Banner
st.title("📊 Customer Churn Risk & Retention System")
st.markdown("**AI-Driven Churn Prediction, Risk Segmentation, and Retention Strategy**")
st.markdown("---")

# Data Loader
@st.cache_data
def load_data():
    if os.path.exists('churn_predictions_v2.csv'):
        df = pd.read_csv('churn_predictions_v2.csv')
    elif os.path.exists('churn_predictions.csv'):
        df = pd.read_csv('churn_predictions.csv')
    else:
        st.error("Prediction data file not found. Please run 'python churn_analysis.py' first.")
        st.stop()

    if 'tenure_group' not in df.columns and 'tenure' in df.columns:
        df['tenure_group'] = pd.cut(
            df['tenure'],
            bins=[-1, 12, 24, 48, 72, 100],
            labels=['0-12 Mo', '12-24 Mo', '24-48 Mo', '48-72 Mo', '72+ Mo']
        )
    return df

df = load_data()

# Sidebar Navigation
st.sidebar.header("🕹️ Dashboard Navigation")
app_mode = st.sidebar.radio(
    "Select View Mode",
    ["Executive Summary KPIs", "Interactive Individual Risk Calculator", "Top 20 High-Risk Target List", "A/B Retention Trial Cohorts", "📈 Model Performance History"]
)

# -----------------------------------------------------------------------------
# MODE 1: EXECUTIVE SUMMARY KPIS
# -----------------------------------------------------------------------------
if app_mode == "Executive Summary KPIs":
    st.subheader("📌 Executive Key Performance Indicators")
    
    col1, col2, col3, col4 = st.columns(4)
    
    overall_churn = df['actual_churn'].mean()
    high_risk_count = (df['churn_probability'] > 0.50).sum()
    avg_monthly = df[df['churn_probability'] > 0.50]['MonthlyCharges'].mean()
    arr_protected = "$1.2M+"

    col1.metric("Overall Churn Rate", f"{overall_churn:.1%}")
    col2.metric("High-Risk Customers (P > 0.50)", f"{high_risk_count} Accounts")
    col3.metric("At-Risk Avg Monthly Charge", f"${avg_monthly:.2f}")
    col4.metric("Protected Revenue Goal", arr_protected)

    st.markdown("---")
    
    st.subheader("📈 Churn Risk Analytics")
    col_left, col_right = st.columns(2)
    
    with col_left:
        st.markdown("**Avg Churn Probability by Contract Type & Tenure Group**")
        bar_data = df.groupby(['Contract', 'tenure_group'])['churn_probability'].mean().unstack()
        st.bar_chart(bar_data)
        
    with col_right:
        st.markdown("**Monthly Charges vs Tenure (Colored by Risk)**")
        scatter_df = df[['tenure', 'MonthlyCharges', 'churn_probability']].copy()
        st.scatter_chart(scatter_df, x='tenure', y='MonthlyCharges', color='churn_probability')

# -----------------------------------------------------------------------------
# MODE 2: INTERACTIVE INDIVIDUAL RISK CALCULATOR
# -----------------------------------------------------------------------------
elif app_mode == "Interactive Individual Risk Calculator":
    st.subheader("🔮 Individual Customer Churn Risk Calculator")
    st.markdown("Adjust subscriber profile attributes to calculate real-time churn probability score.")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        contract = st.selectbox("Contract Type", ["Month-to-month", "One year", "Two year"])
        internet = st.selectbox("Internet Service", ["Fiber optic", "DSL", "No"])
        tenure = st.slider("Tenure (Months)", 1, 72, 12)
        
    with col2:
        monthly_charges = st.slider("Monthly Charges ($)", 18.0, 120.0, 75.0)
        payment_method = st.selectbox("Payment Method", ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"])
        tech_support = st.selectbox("Tech Support Add-on", ["No", "Yes", "No internet service"])
        
    with col3:
        online_security = st.selectbox("Online Security Add-on", ["No", "Yes", "No internet service"])
        paperless = st.selectbox("Paperless Billing", ["Yes", "No"])
        senior = st.selectbox("Senior Citizen", [0, 1])

    # Simple heuristic risk calculator matching tuned Logistic Regression weights
    contract_val = 2.2 if contract == "Month-to-month" else (-1.2 if contract == "Two year" else 0.0)
    internet_val = 1.4 if internet == "Fiber optic" else 0.0
    tech_val = -1.2 if tech_support == "Yes" else 0.0
    sec_val = -1.0 if online_security == "Yes" else 0.0
    pay_val = 0.8 if payment_method == "Electronic check" else 0.0
    
    logit = contract_val + internet_val + tech_val + sec_val + pay_val - 0.04 * tenure + 0.02 * monthly_charges - 1.5
    calc_prob = 1 / (1 + np.exp(-logit))
    calc_prob = float(np.clip(calc_prob, 0.05, 0.95))

    st.markdown("### **Calculated Churn Probability Risk Score**")
    
    if calc_prob > 0.65:
        st.error(f"🚨 **HIGH RISK**: Churn Probability = **{calc_prob:.1%}**")
        st.warning("Recommended Retention Action: Trigger **Strategy 1 ('Lock-In & Reward')** — Offer 15% discount for 1-year contract lock-in.")
    elif calc_prob > 0.40:
        st.warning(f"⚠️ **MEDIUM RISK**: Churn Probability = **{calc_prob:.1%}**")
        st.info("Recommended Retention Action: Trigger **Strategy 2 ('Onboarding Shield')** — Bundle free Tech Support for 6 months.")
    else:
        st.success(f"✅ **LOW RISK**: Churn Probability = **{calc_prob:.1%}**")
        st.info("Account Healthy. No active intervention required.")

# -----------------------------------------------------------------------------
# MODE 3: TOP 20 HIGH-RISK TARGET LIST
# -----------------------------------------------------------------------------
elif app_mode == "Top 20 High-Risk Target List":
    st.subheader("🎯 Top 20 High-Risk Customer Target List")
    st.markdown("Filtered customer roster with churn probability $P > 0.50$ sorted by highest risk.")
    
    high_risk_df = df[df['churn_probability'] > 0.50].sort_values(by='churn_probability', ascending=False)
    st.dataframe(
        high_risk_df[['customerID', 'Contract', 'tenure', 'MonthlyCharges', 'PaymentMethod', 'churn_probability', 'predicted_churn']].head(20),
        use_container_width=True
    )
    
    st.download_button(
        label="📥 Download High-Risk Target List CSV",
        data=high_risk_df.to_csv(index=False),
        file_name="high_risk_customer_target_list.csv",
        mime="text/csv"
    )

# -----------------------------------------------------------------------------
# MODE 4: A/B RETENTION TRIAL COHORTS
# -----------------------------------------------------------------------------
elif app_mode == "A/B Retention Trial Cohorts":
    st.subheader("🧪 A/B Retention Trial Cohort Viewer")
    st.markdown("Randomized 50/50 Control vs. Variant cohort partitions for Initiative #1.")
    
    col_c, col_v = st.columns(2)
    
    if os.path.exists('ab_test_control.csv') and os.path.exists('ab_test_variant.csv'):
        ctrl_df = pd.read_csv('ab_test_control.csv')
        var_df = pd.read_csv('ab_test_variant.csv')
        
        with col_c:
            st.markdown(f"### Control Group ({len(ctrl_df)} Accounts)")
            st.markdown("**Offer**: Standard Care (No Discount)")
            st.dataframe(ctrl_df[['customerID', 'Contract', 'MonthlyCharges', 'churn_probability']].head(10), use_container_width=True)
            
        with col_v:
            st.markdown(f"### Variant Group ({len(var_df)} Accounts)")
            st.markdown("**Offer**: 15% Monthly Discount for 1-Year Lock-In")
            st.dataframe(var_df[['customerID', 'Contract', 'MonthlyCharges', 'churn_probability']].head(10), use_container_width=True)
    else:
        st.info("Run 'python ab_test_cohort.py' to generate A/B test CSV files.")

# -----------------------------------------------------------------------------
# MODE 5: MODEL PERFORMANCE HISTORY
# -----------------------------------------------------------------------------
elif app_mode == "📈 Model Performance History":
    st.subheader("📈 Model Performance History & AUC Tracking")
    st.markdown("Historical tracking of model evaluation metrics across execution runs.")
    
    metrics_file = 'metrics_history.csv'
    if os.path.exists(metrics_file):
        history_df = pd.read_csv(metrics_file)
        
        if not history_df.empty and 'AUC' in history_df.columns:
            latest_row = history_df.iloc[-1]
            col_m1, col_m2, col_m3, col_m4 = st.columns(4)
            col_m1.metric("Latest AUC-ROC", f"{latest_row['AUC']:.4f}")
            col_m2.metric("Latest Precision", f"{latest_row['precision']:.1%}")
            col_m3.metric("Latest Recall", f"{latest_row['recall']:.1%}")
            col_m4.metric("Latest Accuracy", f"{latest_row['accuracy']:.1%}")
            
            st.markdown("---")
            st.markdown("### **AUC Performance Trend**")
            fig, ax = plt.subplots(figsize=(10, 4))
            ax.plot(history_df['timestamp'], history_df['AUC'], marker='o', color='#1f77b4', linewidth=2, label='AUC-ROC')
            ax.set_xlabel('Execution Timestamp', fontsize=10)
            ax.set_ylabel('AUC Score', fontsize=10)
            ax.set_title('Historical Model AUC-ROC Score Trend', fontsize=12, fontweight='bold')
            ax.set_ylim([0.5, 1.0])
            ax.grid(True, linestyle='--', alpha=0.5)
            plt.xticks(rotation=45, ha='right')
            plt.tight_layout()
            st.pyplot(fig)
            
            st.markdown("---")
            st.markdown("### **Execution Log Data**")
            st.dataframe(history_df, use_container_width=True)
        else:
            st.warning("Metrics history file is empty or corrupted.")
    else:
        st.info("No historical metric data found yet. Run `python churn_analysis.py` or `make pipeline` to record initial performance history.")

st.markdown("---")
st.markdown("© 2026 Customer Churn Analysis & Retention System | Senior Lead Data Scientist")
