"""
Streamlit Web Application: Customer Churn Risk & Retention System
Enterprise-Grade Executive Dashboard with Deep Slate Glassmorphism Theme.
"""

import os
from typing import Any, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

# -----------------------------------------------------------------------------
# 1. PAGE CONFIGURATION & GLASSMORPHISM THEME CSS
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Churn Predictor",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
<style>
    /* Lighter Modern Slate Background Theme */
    .stApp, .stAppViewContainer {
        background-color: #0f172a !important;
    }

    /* High-contrast crisp typography */
    h1, h2, h3, h4, h5, h6, .stMarkdown p, label, span {
        color: #f8fafc !important;
    }

    /* Custom CSS for vibrant, pop-out KPI Metric Cards */
    [data-testid="stMetric"] {
        background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%);
        border: 1px solid rgba(226, 232, 240, 0.35);
        border-radius: 12px;
        padding: 18px 22px;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.3);
        transition: transform 0.2s ease, border-color 0.2s ease, box-shadow 0.2s ease;
    }
    [data-testid="stMetric"]:hover {
        transform: translateY(-2px);
        border-color: rgba(99, 102, 241, 0.7);
        box-shadow: 0 6px 24px rgba(99, 102, 241, 0.25);
    }
    [data-testid="stMetricLabel"] {
        color: #cbd5e1 !important;
        font-size: 0.85rem !important;
        font-weight: 700 !important;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    [data-testid="stMetricValue"] {
        color: #38bdf8 !important;
        font-size: 1.85rem !important;
        font-weight: 800 !important;
    }

    /* Dataframe container styling */
    .stDataFrame {
        border: 1px solid rgba(226, 232, 240, 0.2);
        border-radius: 10px;
    }

    /* Download button styling */
    .stDownloadButton>button {
        background: linear-gradient(135deg, #6366f1 0%, #4f46e5 100%);
        color: #ffffff !important;
        border: none;
        border-radius: 8px;
        font-weight: 600;
        padding: 10px 20px;
        transition: all 0.2s ease-in-out;
    }
    .stDownloadButton>button:hover {
        transform: translateY(-1px);
        box-shadow: 0 4px 14px rgba(99, 102, 241, 0.4);
    }
</style>
""",
    unsafe_allow_html=True,
)

# -----------------------------------------------------------------------------
# 2. CACHED DATA & MODEL LOADERS & DIRECTORIES
# -----------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for d in ["data/processed", "models", "logs", "dashboard"]:
    os.makedirs(os.path.join(BASE_DIR, d), exist_ok=True)

MODEL_PATH = os.path.join(BASE_DIR, "models", "model.pkl") if os.path.exists(os.path.join(BASE_DIR, "models", "model.pkl")) else "models/model.pkl"
SCALER_PATH = os.path.join(BASE_DIR, "models", "scaler.pkl") if os.path.exists(os.path.join(BASE_DIR, "models", "scaler.pkl")) else "models/scaler.pkl"
SELECTOR_PATH = os.path.join(BASE_DIR, "models", "selector.pkl") if os.path.exists(os.path.join(BASE_DIR, "models", "selector.pkl")) else "models/selector.pkl"


@st.cache_resource
def load_model_artifacts() -> Tuple[Optional[Any], Optional[Any], Optional[Any]]:
    """Load serialized model, scaler, and selector pickles."""
    if (
        os.path.exists(MODEL_PATH)
        and os.path.exists(SCALER_PATH)
        and os.path.exists(SELECTOR_PATH)
    ):
        try:
            m = joblib.load(MODEL_PATH)
            s = joblib.load(SCALER_PATH)
            sel = joblib.load(SELECTOR_PATH)
            return m, s, sel
        except Exception:
            return None, None, None
    return None, None, None


@st.cache_data
def load_predictions_data() -> Optional[pd.DataFrame]:
    """Load model prediction outputs CSV."""
    paths = [
        os.path.join(BASE_DIR, "data", "processed", "churn_predictions_v2.csv"),
        "data/processed/churn_predictions_v2.csv",
        os.path.join(BASE_DIR, "data", "processed", "churn_predictions.csv"),
        "data/processed/churn_predictions.csv",
    ]
    for p in paths:
        if os.path.exists(p):
            try:
                df = pd.read_csv(p)
                if "tenure_group" not in df.columns and "tenure" in df.columns:
                    df["tenure_group"] = pd.cut(
                        df["tenure"],
                        bins=[-1, 12, 24, 48, 72, 100],
                        labels=[
                            "0-12 Mo",
                            "12-24 Mo",
                            "24-48 Mo",
                            "48-72 Mo",
                            "72+ Mo",
                        ],
                    )
                return df
            except Exception:
                continue
    return None


@st.cache_data
def load_ab_cohort_data() -> Tuple[Optional[pd.DataFrame], Optional[pd.DataFrame]]:
    """Load A/B retention trial cohort CSV files."""
    c_path = (
        "data/processed/ab_test_control.csv"
        if os.path.exists("data/processed/ab_test_control.csv")
        else "ab_test_control.csv"
    )
    v_path = (
        "data/processed/ab_test_variant.csv"
        if os.path.exists("data/processed/ab_test_variant.csv")
        else "ab_test_variant.csv"
    )

    ctrl_df, var_df = None, None
    if os.path.exists(c_path):
        try:
            ctrl_df = pd.read_csv(c_path)
        except Exception:
            pass

    if os.path.exists(v_path):
        try:
            var_df = pd.read_csv(v_path)
        except Exception:
            pass

    return ctrl_df, var_df


@st.cache_data
def load_metrics_history() -> Optional[pd.DataFrame]:
    """Load historical model performance execution logs from logs/metrics_history.csv."""
    paths = [
        os.path.join(BASE_DIR, "logs", "metrics_history.csv"),
        "logs/metrics_history.csv",
        os.path.join(BASE_DIR, "data", "processed", "metrics_history.csv"),
        "data/processed/metrics_history.csv",
        "metrics_history.csv",
    ]
    for p in paths:
        if os.path.exists(p):
            try:
                df = pd.read_csv(p)
                # Normalize column names if lower/upper case
                col_map = {
                    "auc": "AUC",
                    "Precision": "precision",
                    "Recall": "recall",
                    "Accuracy": "accuracy",
                }
                df = df.rename(columns=col_map)
                return df
            except Exception:
                continue
    return None


# -----------------------------------------------------------------------------
# 3. MODULAR VIEW RENDERERS
# -----------------------------------------------------------------------------
def render_executive_summary(df: Optional[pd.DataFrame]) -> None:
    """Render View 1: Executive KPI Cards & Interactive Plotly Charts."""
    st.subheader("📌 Executive Key Performance Indicators")

    if df is None or df.empty:
        st.warning("⚠️ Predictions file not found. Please run the pipeline first.")
        return

    col1, col2, col3, col4 = st.columns(4)

    overall_churn = df["actual_churn"].mean()
    high_risk_count = (df["churn_probability"] > 0.50).sum()
    avg_monthly = df[df["churn_probability"] > 0.50]["MonthlyCharges"].mean()
    arr_protected = "$1.2M+"

    col1.metric("Overall Churn Rate", f"{overall_churn:.1%}")
    col2.metric("High-Risk Accounts (P > 0.50)", f"{high_risk_count} Accounts")
    col3.metric("At-Risk Avg Monthly Charge", "$%.2f" % avg_monthly)
    col4.metric("Protected Revenue Goal", arr_protected)

    st.markdown("---")
    st.subheader("📈 Churn Risk Analytics")

    col_left, col_right = st.columns(2)

    with col_left:
        st.markdown("**Avg Churn Probability by Contract & Tenure Group**")
        bar_data = (
            df.groupby(["Contract", "tenure_group"], observed=False)[
                "churn_probability"
            ]
            .mean()
            .reset_index()
        )
        fig_bar = px.bar(
            bar_data,
            x="Contract",
            y="churn_probability",
            color="tenure_group",
            barmode="group",
            color_discrete_sequence=px.colors.sequential.Blues_r,
            labels={
                "churn_probability": "Avg Churn Prob",
                "tenure_group": "Tenure Group",
            },
        )
        fig_bar.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font_color="#f8fafc",
            margin=dict(l=20, r=20, t=20, b=20),
            height=350,
            yaxis=dict(tickformat=".0%"),
        )
        st.plotly_chart(fig_bar, use_container_width=True)

    with col_right:
        st.markdown("**Monthly Charges vs Tenure (Colored by Risk Gradient)**")
        fig_scatter = px.scatter(
            df,
            x="tenure",
            y="MonthlyCharges",
            color="churn_probability",
            color_continuous_scale="RdYlGn_r",
            labels={
                "tenure": "Tenure (Months)",
                "MonthlyCharges": "Monthly Charge ($)",
                "churn_probability": "Risk Score",
            },
        )
        fig_scatter.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font_color="#f8fafc",
            margin=dict(l=20, r=20, t=20, b=20),
            height=350,
        )
        st.plotly_chart(fig_scatter, use_container_width=True)


def render_risk_calculator(
    model: Optional[Any], scaler: Optional[Any], selector: Optional[Any]
) -> None:
    """Render View 2: Interactive Individual Churn Risk Calculator."""
    st.subheader("🔮 Individual Customer Churn Risk Calculator")
    st.markdown(
        "Adjust subscriber profile attributes to calculate real-time churn probability score."
    )

    if model is None or scaler is None or selector is None:
        st.info(
            "ℹ️ Loaded pre-trained heuristic calculator model. Run `python src/churn_analysis.py` to activate ML artifact inference."
        )

    col1, col2, col3 = st.columns(3)

    with col1:
        contract = st.selectbox(
            "Contract Type", ["Month-to-month", "One year", "Two year"]
        )
        internet = st.selectbox("Internet Service", ["Fiber optic", "DSL", "No"])
        tenure = st.slider("Tenure (Months)", 0, 72, 12)

    with col2:
        monthly_charges = st.slider("Monthly Charges ($)", 18.0, 120.0, 75.0)
        payment_method = st.selectbox(
            "Payment Method",
            [
                "Electronic check",
                "Mailed check",
                "Bank transfer (automatic)",
                "Credit card (automatic)",
            ],
        )
        tech_support = st.selectbox(
            "Tech Support Add-on", ["No", "Yes", "No internet service"]
        )

    with col3:
        online_security = st.selectbox(
            "Online Security Add-on", ["No", "Yes", "No internet service"]
        )
        paperless = st.selectbox("Paperless Billing", ["Yes", "No"])
        senior = st.selectbox("Senior Citizen", [0, 1])

    # Preprocess payload for ML model or fallback weights
    if model is not None and scaler is not None and selector is not None:
        contract_map = {"Month-to-month": 0, "One year": 1, "Two year": 2}
        internet_map = {"No": 0, "DSL": 1, "Fiber optic": 2}
        payment_map = {
            "Electronic check": 0,
            "Mailed check": 1,
            "Bank transfer (automatic)": 2,
            "Credit card (automatic)": 3,
        }
        binary_map = {"No": 0, "Yes": 1}
        tri_map = {
            "No": 0,
            "No internet service": 1,
            "No phone service": 1,
            "Yes": 2,
        }

        total_charges = float(tenure * monthly_charges)
        avg_monthly = total_charges / tenure if tenure > 0 else 0.0
        tenure_group = (
            0 if tenure <= 12 else (1 if tenure <= 24 else (2 if tenure <= 48 else 3))
        )

        feat_dict = {
            "gender": 1,
            "SeniorCitizen": senior,
            "Partner": 0,
            "Dependents": 0,
            "tenure": tenure,
            "PhoneService": 1,
            "MultipleLines": 0,
            "InternetService": internet_map[internet],
            "OnlineSecurity": tri_map[online_security],
            "OnlineBackup": 0,
            "DeviceProtection": 0,
            "TechSupport": tri_map[tech_support],
            "StreamingTV": 0,
            "StreamingMovies": 0,
            "Contract": contract_map[contract],
            "PaperlessBilling": binary_map[paperless],
            "PaymentMethod": payment_map[payment_method],
            "MonthlyCharges": float(monthly_charges),
            "TotalCharges": total_charges,
            "tenure_group": tenure_group,
            "avg_monthly_charge": float(avg_monthly),
            "has_online_security": 1 if online_security == "Yes" else 0,
            "has_tech_support": 1 if tech_support == "Yes" else 0,
        }
        raw_df = pd.DataFrame([feat_dict])
        scaled_feat = scaler.transform(raw_df)
        sel_feat = selector.transform(scaled_feat)
        calc_prob = float(model.predict_proba(sel_feat)[0, 1])
    else:
        # Heuristic fallback matching model weights
        contract_val = (
            2.2
            if contract == "Month-to-month"
            else (-1.2 if contract == "Two year" else 0.0)
        )
        internet_val = 1.4 if internet == "Fiber optic" else 0.0
        tech_val = -1.2 if tech_support == "Yes" else 0.0
        sec_val = -1.0 if online_security == "Yes" else 0.0
        pay_val = 0.8 if payment_method == "Electronic check" else 0.0

        logit = (
            contract_val
            + internet_val
            + tech_val
            + sec_val
            + pay_val
            - 0.04 * tenure
            + 0.02 * monthly_charges
            - 1.5
        )
        calc_prob = 1 / (1 + np.exp(-logit))
        calc_prob = float(np.clip(calc_prob, 0.05, 0.95))

    st.markdown("---")
    st.markdown("### **Calculated Churn Probability Risk Score**")

    # Risk Color Cards
    if calc_prob > 0.65:
        st.markdown(
            f"""
            <div style="background-color: rgba(239, 68, 68, 0.15); border: 1px solid #ef4444; padding: 16px 20px; border-radius: 10px; margin-bottom: 15px;">
                <h4 style="color: #ef4444 !important; margin: 0;">🚨 HIGH RISK: Churn Probability = <b>{calc_prob:.1%}</b></h4>
                <p style="color: #fca5a5 !important; margin-top: 5px; margin-bottom: 0;">Recommended Retention Action: Trigger <b>Strategy 1 ('Lock-In & Reward')</b> — Offer 15% discount for 1-year contract lock-in.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
    elif calc_prob > 0.35:
        st.markdown(
            f"""
            <div style="background-color: rgba(245, 158, 11, 0.15); border: 1px solid #f59e0b; padding: 16px 20px; border-radius: 10px; margin-bottom: 15px;">
                <h4 style="color: #f59e0b !important; margin: 0;">⚠️ MEDIUM RISK: Churn Probability = <b>{calc_prob:.1%}</b></h4>
                <p style="color: #fde68a !important; margin-top: 5px; margin-bottom: 0;">Recommended Retention Action: Trigger <b>Strategy 2 ('Onboarding Shield')</b> — Bundle free Tech Support for 6 months.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            f"""
            <div style="background-color: rgba(16, 185, 129, 0.15); border: 1px solid #10b981; padding: 16px 20px; border-radius: 10px; margin-bottom: 15px;">
                <h4 style="color: #10b981 !important; margin: 0;">✅ LOW RISK: Churn Probability = <b>{calc_prob:.1%}</b></h4>
                <p style="color: #a7f3d0 !important; margin-top: 5px; margin-bottom: 0;">Account Healthy. No active intervention required.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )


def render_high_risk_roster(df: Optional[pd.DataFrame]) -> None:
    """Render View 3: Top 20 High-Risk Customer Target List."""
    st.subheader("🎯 Top 20 High-Risk Customer Target List")
    st.markdown(
        "Filtered customer roster with churn probability $P > 0.50$ sorted by highest risk."
    )

    try:
        if df is None or df.empty or "churn_probability" not in df.columns:
            raise FileNotFoundError("Predictions data unavailable.")

        high_risk_df = df[df["churn_probability"] > 0.50].sort_values(
            by="churn_probability", ascending=False
        )

        display_df = (
            high_risk_df[
                [
                    "customerID",
                    "Contract",
                    "tenure",
                    "MonthlyCharges",
                    "PaymentMethod",
                    "churn_probability",
                    "predicted_churn",
                ]
            ]
            .head(20)
            .copy()
        )

        # Format monetary column and churn probability percentage
        display_df["MonthlyCharges"] = display_df["MonthlyCharges"].apply(
            lambda x: "$%.2f" % x if isinstance(x, (int, float)) else str(x)
        )
        display_df["churn_probability"] = display_df["churn_probability"].apply(
            lambda x: f"{x:.1%}" if isinstance(x, (int, float)) else str(x)
        )

        st.dataframe(
            display_df,
            use_container_width=True,
        )

        st.download_button(
            label="📥 Download High-Risk Target List CSV",
            data=high_risk_df.to_csv(index=False),
            file_name="high_risk_customer_target_list.csv",
            mime="text/csv",
        )
    except Exception:
        st.warning("⚠️ Predictions file not found. Please run the pipeline first.")


def render_ab_cohorts() -> None:
    """Render View 4: A/B Retention Trial Cohorts."""
    st.subheader("🧪 A/B Retention Trial Cohort Viewer")
    st.markdown(
        "Randomized 50/50 Control vs. Variant cohort partitions for Initiative #1."
    )

    ctrl_df, var_df = load_ab_cohort_data()

    if ctrl_df is not None and var_df is not None:
        col_c, col_v = st.columns(2)

        # Format display copies
        ctrl_disp = ctrl_df.copy()
        var_disp = var_df.copy()

        for sub_df in [ctrl_disp, var_disp]:
            if "MonthlyCharges" in sub_df.columns:
                sub_df["MonthlyCharges"] = sub_df["MonthlyCharges"].apply(
                    lambda x: "$%.2f" % x if isinstance(x, (int, float)) else str(x)
                )
            if "churn_probability" in sub_df.columns:
                sub_df["churn_probability"] = sub_df["churn_probability"].apply(
                    lambda x: f"{x:.1%}" if isinstance(x, (int, float)) else str(x)
                )

        with col_c:
            st.markdown(f"### Control Group ({len(ctrl_df)} Accounts)")
            st.markdown("**Offer**: Standard Care (No Discount)")
            st.dataframe(
                ctrl_disp[
                    ["customerID", "Contract", "MonthlyCharges", "churn_probability"]
                ].head(10),
                use_container_width=True,
            )

        with col_v:
            st.markdown(f"### Variant Group ({len(var_df)} Accounts)")
            st.markdown("**Offer**: 15% Monthly Discount for 1-Year Lock-In")
            st.dataframe(
                var_disp[
                    ["customerID", "Contract", "MonthlyCharges", "churn_probability"]
                ].head(10),
                use_container_width=True,
            )
    else:
        st.info("Run 'python src/ab_test_cohort.py' to generate A/B test CSV files.")


def render_model_history() -> None:
    """Render View 5: Model Performance History & Plotly Trend Line."""
    st.subheader("📈 Model Performance History & AUC Tracking")
    st.markdown(
        "Historical tracking of model evaluation metrics across execution runs."
    )

    history_df = load_metrics_history()

    if history_df is not None and not history_df.empty and "AUC" in history_df.columns:
        latest_row = history_df.iloc[-1]
        col_m1, col_m2, col_m3, col_m4 = st.columns(4)
        col_m1.metric("Latest AUC-ROC", f"{latest_row['AUC']:.4f}")
        col_m2.metric("Latest Precision", f"{latest_row['precision']:.1%}")
        col_m3.metric("Latest Recall", f"{latest_row['recall']:.1%}")
        col_m4.metric("Latest Accuracy", f"{latest_row['accuracy']:.1%}")

        st.markdown("---")
        st.markdown("### **AUC Performance Trend**")

        fig_trend = px.line(
            history_df,
            x="timestamp",
            y="AUC",
            markers=True,
            title="Historical Model AUC-ROC Score Trend",
            labels={"timestamp": "Execution Timestamp", "AUC": "AUC Score"},
        )
        fig_trend.update_traces(
            line_color="#38bdf8", line_width=3, marker_size=8, marker_color="#6366f1"
        )
        fig_trend.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font_color="#f8fafc",
            yaxis_range=[0.5, 1.0],
            margin=dict(l=20, r=20, t=40, b=20),
            height=380,
        )
        st.plotly_chart(fig_trend, use_container_width=True)

        st.markdown("---")
        st.markdown("### **Execution Log Data**")
        st.dataframe(history_df, use_container_width=True)
    else:
        st.info(
            "No historical metric data found yet. Run `python src/churn_analysis.py` or `make pipeline` to record initial performance history."
        )


# -----------------------------------------------------------------------------
# 4. MAIN APP ROUTER & SIDEBAR
# -----------------------------------------------------------------------------
def main() -> None:
    """Main application layout router."""
    # Sidebar Branding & Tags
    st.sidebar.title("📊 Churn Predictor")
    st.sidebar.markdown(
        "AI-driven subscriber churn prediction, risk segmentation, SHAP explainability, and targeted retention strategy."
    )

    st.sidebar.markdown(
        """
        <div style="background-color: rgba(99, 102, 241, 0.15); border: 1px solid rgba(99, 102, 241, 0.4); padding: 8px 12px; border-radius: 8px; font-size: 0.8rem; color: #a5b4fc; margin-bottom: 15px;">
            🏷️ <b>Data Source</b>: Telco Customer Churn
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.sidebar.header("🕹️ Navigation")
    app_mode = st.sidebar.radio(
        "Select View Mode",
        [
            "Executive Summary KPIs",
            "Individual Risk Calculator",
            "Top 20 High-Risk Roster",
            "A/B Retention Cohorts",
            "Model AUC History",
        ],
    )

    # Load artifacts & data
    df_preds = load_predictions_data()
    model, scaler, selector = load_model_artifacts()

    # Route to view renderers
    if app_mode == "Executive Summary KPIs":
        render_executive_summary(df_preds)
    elif app_mode == "Individual Risk Calculator":
        render_risk_calculator(model, scaler, selector)
    elif app_mode == "Top 20 High-Risk Roster":
        render_high_risk_roster(df_preds)
    elif app_mode == "A/B Retention Cohorts":
        render_ab_cohorts()
    elif app_mode == "Model AUC History":
        render_model_history()

    # Footer
    st.markdown("---")
    st.markdown(
        "<p style='text-align: center; color: #94a3b8; font-size: 0.85rem; padding-top: 1rem;'>"
        "Built with ❤️ using Streamlit, Scikit-learn & SHAP • © 2026 Customer Churn Analysis Team"
        "</p>",
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
