# 📦 Data Version Control (DVC) Guide (`DATAVERSION.md`)

This repository uses **[Data Version Control (DVC)](https://dvc.org/)** to track raw dataset versions (`data/customer_data.csv`) independently of Git code commits.

---

## 🛠️ 1. How to Pull the Latest Dataset

To pull the latest tracked dataset from your DVC remote storage and sync your local workspace:

```bash
# Pull the latest dataset tracked by data/customer_data.csv.dvc
dvc pull
```

After pulling the latest dataset, re-run the ML pipeline to retrain models and update churn prediction artifacts:

```bash
# Reprocess ML pipeline and update predictions
python churn_analysis.py

# Or run via Makefile
make pipeline
```

---

## ⚙️ 2. Configuring DVC Remote Storage

DVC supports multiple remote storage backends, including local storage folders, AWS S3 buckets, Google Cloud Storage (GCS), or Azure Blob Storage.

### Option A: Local / Shared Network Remote Storage

To configure a local directory (or shared network drive) as your DVC remote:

```bash
# Create local storage directory
mkdir -p /tmp/dvc-storage

# Add local remote named 'localremote'
dvc remote add -d localremote /tmp/dvc-storage

# Push tracked dataset to local remote
dvc push
```

### Option B: AWS S3 Bucket Remote Storage

To configure an AWS S3 bucket as your production DVC remote storage:

```bash
# Add S3 remote bucket
dvc remote add -d s3remote s3://my-churn-analytics-bucket/dvcstore

# Configure AWS region (if needed)
dvc remote modify s3remote region us-east-1

# Push tracked dataset to S3
dvc push
```

> **Note**: For AWS S3 authentication, set standard AWS credentials in your environment:
> ```bash
> export AWS_ACCESS_KEY_ID="YOUR_ACCESS_KEY"
> export AWS_SECRET_ACCESS_KEY="YOUR_SECRET_KEY"
> ```

---

## 🔄 3. Workflow for Adding New Dataset Versions

When new customer data is ingested:

1. Place the updated CSV file at `data/customer_data.csv`.
2. Update DVC tracking:
   ```bash
   dvc add data/customer_data.csv
   ```
3. Commit the updated `.dvc` tracking file to Git:
   ```bash
   git add data/customer_data.csv.dvc data/.gitignore
   git commit -m "Update dataset version via DVC"
   ```
4. Push data file to DVC remote:
   ```bash
   dvc push
   ```
