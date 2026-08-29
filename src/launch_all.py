"""
Master 1-Click Execution & Validation Launcher for Customer Churn System
Executes ML pipeline, Tableau XML builder, A/B cohort generator, and CLI runner.
Verifies project file inventory across src/, data/, models/, dashboard/, and app/.
"""

import os
import sys
import subprocess

def run_step(command, step_name):
    print(f"\n[RUNNING] {step_name}...")
    result = subprocess.run([sys.executable] + command.split(), capture_output=True, text=True)
    if result.returncode == 0:
        print(f"[SUCCESS] {step_name} completed cleanly.")
        return True
    else:
        print(f"[ERROR] {step_name} failed:\n{result.stderr}")
        return False

def main():
    print("=" * 80)
    print("      MASTER EXECUTION & VALIDATION LAUNCHER")
    print("=" * 80)

    steps = [
        ("src/churn_analysis.py", "1. Machine Learning Pipeline & High-AUC Tuning"),
        ("src/generate_tableau_workbook.py", "2. Tableau XML Workbook Builder"),
        ("src/ab_test_cohort.py", "3. A/B Testing Cohort Generator"),
        ("src/run_churn_pipeline.py --data data/customer_data.csv", "4. Production CLI Runner & Logger")
    ]

    all_passed = True
    for cmd, name in steps:
        success = run_step(cmd, name)
        if not success:
            all_passed = False

    print("\n" + "=" * 80)
    print("               FINAL WORKSPACE INVENTORY VERIFICATION")
    print("=" * 80)

    expected_files = [
        'data/customer_data.csv', 'src/churn_analysis.py', 'data/processed/churn_predictions.csv',
        'data/processed/churn_predictions_v2.csv', 'src/generate_tableau_workbook.py',
        'dashboard/Customer_Churn_Dashboard.twb', 'src/ab_test_cohort.py', 'data/processed/ab_test_control.csv',
        'data/processed/ab_test_variant.csv', 'src/run_churn_pipeline.py', 'data/processed/predictions_output.csv',
        'app/app.py', 'src/api.py', 'src/json_logger.py', 'models/model.pkl', 'models/scaler.pkl',
        'models/selector.pkl', 'README.md', 'LICENSE', 'Makefile', 'requirements.txt', 'Dockerfile'
    ]

    found_count = 0
    for f in expected_files:
        exists = os.path.exists(f)
        status = "[OK]" if exists else "[MISSING]"
        if exists:
            found_count += 1
        print(f" {status:<10} | {f}")

    print("=" * 80)
    print(f" Workspace Completeness: {found_count} / {len(expected_files)} Files Verified ({found_count/len(expected_files):.0%})")
    
    if all_passed and found_count == len(expected_files):
        print("\nALL SYSTEMS GO! Your project is 100% operational, validated, and portfolio-ready.")
        print("To launch the Streamlit app live:  streamlit run app/app.py")
    else:
        print("\nNote: Some components require attention. Check logs above.")
    print("=" * 80 + "\n")

if __name__ == '__main__':
    main()
