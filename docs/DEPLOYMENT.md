# Production Deployment & Operations Guide 🛠️

**Project**: Customer Churn Analysis & Retention System
**Pipeline Entry Point**: [`run_churn_pipeline.py`](file:///c:/Users/yashn/CustomerChurnAnalysis/run_churn_pipeline.py)
**Predictions Output**: [`predictions_output.csv`](file:///c:/Users/yashn/CustomerChurnAnalysis/predictions_output.csv)

---

## 1. Automated Weekly Job Scheduling

To maintain up-to-date churn predictions and target high-risk accounts, schedule `run_churn_pipeline.py` to execute automatically once per week (e.g., every Monday at 02:00 AM).

### Option A: Windows Task Scheduler (Windows Server / Local)

1. Open **Task Scheduler** (`taskschd.msc`).
2. Click **Create Basic Task** in the Actions panel.
3. Set **Name**: `Weekly_Customer_Churn_Pipeline`.
4. Set **Trigger**: **Weekly** $\rightarrow$ Recur every 1 week on **Monday** at `02:00 AM`.
5. Set **Action**: **Start a Program**.
   - **Program/script**: `C:\Users\yashn\AppData\Local\Programs\Python\Python312\python.exe` (or your Python environment path).
   - **Add arguments**: `run_churn_pipeline.py --data customer_data.csv`
   - **Start in**: `c:\Users\yashn\CustomerChurnAnalysis`
6. Click **Finish**.

#### Automated PowerShell Command (Alternative):
```powershell
$action = New-ScheduledTaskAction -Execute "python.exe" -Argument "run_churn_pipeline.py --data customer_data.csv" -WorkingDirectory "c:\Users\yashn\CustomerChurnAnalysis"
$trigger = New-ScheduledTaskTrigger -Weekly -DaysOfWeek Monday -At 2am
Register-ScheduledTask -TaskName "Weekly_Churn_Pipeline" -Action $action -Trigger $trigger
```

---

### Option B: Linux Crontab (Linux / Cloud VM)

1. Open crontab editor:
   ```bash
   crontab -e
   ```
2. Add the following cron schedule (Runs every Monday at 02:00 AM):
   ```bash
   0 2 * * 1 cd /home/ubuntu/CustomerChurnAnalysis && /usr/bin/python3 run_churn_pipeline.py --data customer_data.csv >> churn_cron.log 2>&1
   ```

---

## 2. Passing New Data from a Database

In production environments, subscriber data lives in SQL databases (PostgreSQL, Snowflake, MySQL, BigQuery).

### Database Data Extractor Example (`fetch_live_data.py`)

Create a python database extract helper to query your data warehouse and output a clean `live_customer_data.csv`:

```python
"""
Database Data Extraction Helper for Churn Pipeline
"""
import pandas as pd
from sqlalchemy import create_engine

# Database Connection URI (PostgreSQL / Snowflake / MySQL)
DB_URI = "postgresql://user:password@localhost:5432/telecom_db"

def extract_weekly_data(output_csv='live_customer_data.csv'):
    engine = create_engine(DB_URI)

    query = """
    SELECT
        customer_id AS customerID,
        gender,
        senior_citizen AS SeniorCitizen,
        partner AS Partner,
        dependents AS Dependents,
        tenure_months AS tenure,
        phone_service AS PhoneService,
        multiple_lines AS MultipleLines,
        internet_service AS InternetService,
        online_security AS OnlineSecurity,
        online_backup AS OnlineBackup,
        device_protection AS DeviceProtection,
        tech_support AS TechSupport,
        streaming_tv AS StreamingTV,
        streaming_movies AS StreamingMovies,
        contract_type AS Contract,
        paperless_billing AS PaperlessBilling,
        payment_method AS PaymentMethod,
        monthly_charges AS MonthlyCharges,
        total_charges AS TotalCharges,
        churn_status AS Churn
    FROM active_subscribers_view;
    """

    df = pd.read_sql(query, engine)
    df.to_csv(output_csv, index=False)
    print(f"[SUCCESS] Extracted {len(df)} records to '{output_csv}'")
    return output_csv

if __name__ == '__main__':
    extract_weekly_data()
```

#### Run Database Extract + Pipeline via Shell:
```bash
python fetch_live_data.py && python run_churn_pipeline.py --data live_customer_data.csv
```

---

## 3. CRM Integration (Salesforce / HubSpot)

Once [`predictions_output.csv`](file:///c:/Users/yashn/CustomerChurnAnalysis/predictions_output.csv) is generated, synchronize the risk scores into your Customer Relationship Management (CRM) platform.

### Option A: Manual / Scheduled Bulk CSV Upload
- **Salesforce**: Use **Salesforce Data Loader** or **Workbench** to upsert `predictions_output.csv` matching on `customerID` $\rightarrow$ `Account.Customer_ID__c`. Map `churn_probability` $\rightarrow$ `Account.Churn_Probability__c` and `predicted_churn` $\rightarrow$ `Account.High_Risk_Flag__c`.
- **HubSpot**: Go to **Contacts / Companies** $\rightarrow$ **Import** $\rightarrow$ Upload `predictions_output.csv` $\rightarrow$ Map `customerID` as Primary Identifier.

### Option B: Automated REST API Push (`sync_crm.py`)

Using the Salesforce REST API (`simple_salesforce` library):

```python
"""
Salesforce CRM Sync Script
Pushes churn predictions directly into Salesforce Account custom fields.
"""
from simple_salesforce import Salesforce
import pandas as pd

# Salesforce Credentials
SF_USERNAME = 'admin@yourcompany.com'
SF_PASSWORD = 'YourPassword'
SF_TOKEN = 'YourSecurityToken'

def sync_predictions_to_salesforce(predictions_csv='predictions_output.csv'):
    sf = Salesforce(username=SF_USERNAME, password=SF_PASSWORD, security_token=SF_TOKEN)
    df = pd.read_csv(predictions_csv)

    records_to_update = []
    for idx, row in df.iterrows():
        records_to_update.append({
            'Customer_ID__c': str(row['customerID']),
            'Churn_Probability__c': float(row['churn_probability']),
            'Churn_Risk_Flag__c': bool(row['predicted_churn'])
        })

    # Bulk Upsert matching on Customer_ID__c
    results = sf.bulk.Account.upsert(records_to_update, 'Customer_ID__c')
    print(f"[SUCCESS] Synchronized {len(results)} prediction records to Salesforce CRM.")

if __name__ == '__main__':
    sync_predictions_to_salesforce()
```

---

## 4. Model Drift Monitoring & Automated Retraining Trigger

Over time, customer behavior and market conditions change, leading to **concept drift** and **data drift**. If model performance drops below the acceptable threshold ($\text{AUC} < 0.85$), the system automatically alerts engineers and triggers model retraining.

### Model Drift Monitoring Script (`monitor_drift.py`)

```python
"""
Model Performance & Drift Monitor
Evaluates current predictions AUC score. If AUC < 0.85, triggers automated alert & retraining.
"""
import os
import pandas as pd
from sklearn.metrics import roc_auc_score
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('DriftMonitor')

AUC_THRESHOLD = 0.85

def evaluate_model_drift(predictions_csv='predictions_output.csv'):
    if not os.path.exists(predictions_csv):
        logger.error(f"File '{predictions_csv}' not found.")
        return

    df = pd.read_csv(predictions_csv)

    if 'actual_churn' not in df.columns or 'churn_probability' not in df.columns:
        logger.error("Required columns missing for drift evaluation.")
        return

    # Compute current AUC-ROC on ground truth labels
    current_auc = roc_auc_score(df['actual_churn'], df['churn_probability'])
    logger.info(f"Current Model Test Set AUC-ROC: {current_auc:.4f}")

    if current_auc < AUC_THRESHOLD:
        logger.warning(f"🚨 MODEL DRIFT DETECTED: AUC ({current_auc:.4f}) dropped below threshold ({AUC_THRESHOLD})!")
        trigger_retraining_alert(current_auc)
    else:
        logger.info(f"✅ Model Health Normal (AUC {current_auc:.4f} >= {AUC_THRESHOLD}). No drift detected.")

def trigger_retraining_alert(auc_score):
    print("=" * 70)
    print(f"ALERT: Triggering Automated Model Retraining Pipeline...")
    print(f"Reason: AUC-ROC degraded to {auc_score:.4f} (Threshold: {AUC_THRESHOLD})")
    print("=" * 70)
    # Execute full retraining script
    os.system("python churn_analysis.py")

if __name__ == '__main__':
    evaluate_model_drift()
```

---

## 5. Operations & Health Summary

| Task | Command / Method | Frequency | Log File |
| :--- | :--- | :--- | :--- |
| **Pipeline Execution** | `python run_churn_pipeline.py --data customer_data.csv` | Weekly (Mondays 2 AM) | [`churn_pipeline.log`](file:///c:/Users/yashn/CustomerChurnAnalysis/churn_pipeline.log) |
| **Drift Monitoring** | `python monitor_drift.py` | Weekly (Post-Pipeline) | Console / Slack Log |
| **Tableau Refresh** | Auto-refreshes from [`churn_predictions.csv`](file:///c:/Users/yashn/CustomerChurnAnalysis/churn_predictions.csv) | Weekly | Dashboard XML |
| **CRM Sync** | `python sync_crm.py` | Weekly | Salesforce Logs |
