# ☁️ Streamlit Community Cloud Deployment Guide

Deploy your interactive Customer Churn Dashboard web app ([`app.py`](file:///c:/Users/yashn/CustomerChurnAnalysis/app.py)) live on the internet for **FREE** using Streamlit Community Cloud!

---

## 🚀 1-Click Cloud Deployment Steps

### Step 1: Push Project Code to GitHub
Ensure your latest code is pushed to your public GitHub repository:
```bash
git add .
git commit -m "Add Streamlit web app dashboard"
git push origin main
```

### Step 2: Sign in to Streamlit Cloud
1. Go to [share.streamlit.io](https://share.streamlit.io) or [streamlit.io/cloud](https://streamlit.io/cloud).
2. Click **Sign in with GitHub**.

### Step 3: Deploy New App
1. Click the **Create App** (or **New App**) button in the top right.
2. Select **I already have an app**.
3. Fill in the repository details:
   - **Repository**: `yourusername/customer-churn-retention`
   - **Branch**: `main`
   - **Main file path**: `app.py`
4. Click **Deploy!**

Within 60 seconds, your interactive web dashboard will be live at `https://yourusername-customer-churn.streamlit.app`! 🎉

---

## 🏷️ Add Streamlit Badge to your `README.md`

Once deployed, add this markdown badge to the top of your [`README.md`](file:///c:/Users/yashn/CustomerChurnAnalysis/README.md) to showcase your live web app link:

```markdown
[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://yourusername-customer-churn.streamlit.app)
```

---

## 🔒 Handling Requirements & Data Files

Streamlit Cloud will automatically detect [`requirements.txt`](file:///c:/Users/yashn/CustomerChurnAnalysis/requirements.txt) and install all required Python packages (`pandas`, `scikit-learn`, `imbalanced-learn`, `xgboost`, `shap`, `matplotlib`, `seaborn`).

It also reads [`churn_predictions_v2.csv`](file:///c:/Users/yashn/CustomerChurnAnalysis/churn_predictions_v2.csv) directly from your repository to render the real-time analytics!
