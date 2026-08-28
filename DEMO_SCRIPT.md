# 🎙️ 2-Minute Video Demo Script: Customer Churn Retention System

**Target Duration**: 120 Seconds (2 Minutes)  
**Tone**: Professional, Authoritative, Energetic  
**Visual Focus**: Screen recording showing Python terminal, Tableau Dashboard, and A/B Test outputs  

---

## ⏱️ Timeline & Script Breakdown

### [0:00 - 0:25] Introduction & The $1.2M Business Problem
* **Visual**: Show project root directory in IDE / File Explorer, highlighting `README.md` and `customer_data.csv`.
* **Voiceover**:
  > "Hi everyone! In subscription telecommunications, customer acquisition costs 5 to 7 times more than retention. Today, I'm excited to walk you through an end-to-end Machine Learning and Business Intelligence solution that predicts customer churn, identifies root causes, and protects over **$1.2 Million in Annual Recurring Revenue**."

---

### [0:25 - 0:55] Technical Machine Learning Pipeline & AUC 0.893
* **Visual**: Open terminal and run `python churn_analysis.py`. Show the pipeline step-by-step logs: cleaning, SMOTE class balancing, 5-fold GridSearchCV, and model evaluation outputs.
* **Voiceover**:
  > "Our automated Python pipeline processes 1,000 subscriber accounts across 21 behavioral features. We handled missing data, engineered key features like average monthly charges, and applied SMOTE class balancing. Using 5-fold cross-validation across Logistic Regression, Random Forest, and XGBoost, our winning model achieved a **0.8931 AUC-ROC score** with a **76.5% recall rate**—catching 3 out of 4 churning subscribers before they cancel."

---

### [0:55 - 1:25] SHAP Explainability & Tableau Executive Dashboard
* **Visual**: Switch screen to the Tableau Dashboard (`Customer_Churn_Dashboard.twb`), hovering over the Risk Scatter Plot and Top 20 High-Risk Table.
* **Voiceover**:
  > "Model accuracy alone isn't enough; executives need transparency. Using SHAP values, we identified that Month-to-Month contracts and a lack of Tech Support are our primary churn drivers. We programmatically built this interactive Tableau dashboard, allowing Customer Success Managers to drill down into high-risk accounts with churn probabilities over 70% for immediate concierge outreach."

---

### [1:25 - 1:50] A/B Testing & 16.5% Revenue Retention Math
* **Visual**: Show `ab_test_control.csv` and `ab_test_variant.csv` files alongside the summary table from `ab_test_cohort.py`.
* **Voiceover**:
  > "To validate our retention plays safely, our system automatically generates 50/50 A/B testing cohorts. By offering a 15% annual contract discount to high-risk Month-to-Month users, bundling free onboarding security, and encouraging Auto-Pay migration, we project a cumulative **16.5% net reduction in churn**."

---

### [1:50 - 2:00] Conclusion & Call to Action
* **Visual**: Show GitHub repository page (`README.md` and `DEPLOYMENT.md`).
* **Voiceover**:
  > "This production-ready system includes CLI runners, SQL extractors, and automated model drift monitoring. Check out the full code and deployment guide on GitHub linked below. Thanks for watching!"
