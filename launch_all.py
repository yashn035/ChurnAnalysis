"""
Master 1-Click Execution & Validation Launcher for Customer Churn System
Executes ML pipeline, Tableau XML builder, A/B cohort generator, and CLI runner.
Verifies all 21 project files and outputs master status report.
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
        ("churn_analysis.py", "1. Machine Learning Pipeline & High-AUC Tuning"),
        ("generate_tableau_workbook.py", "2. Tableau XML Workbook Builder"),
        ("ab_test_cohort.py", "3. A/B Testing Cohort Generator"),
        ("run_churn_pipeline.py --data customer_data.csv", "4. Production CLI Runner & Logger")
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
        'customer_data.csv', 'churn_analysis.py', 'churn_predictions.csv',
        'churn_predictions_v2.csv', 'generate_tableau_workbook.py',
        'Customer_Churn_Dashboard.twb', 'ab_test_cohort.py', 'ab_test_control.csv',
        'ab_test_variant.csv', 'run_churn_pipeline.py', 'predictions_output.csv',
        'app.py', 'README.md', 'DEPLOYMENT.md', 'requirements.txt',
        'DEMO_SCRIPT.md', 'LINKEDIN_POST.md', 'EXECUTIVE_ONE_PAGER.md',
        'MEDIUM_ARTICLE.md', 'STREAMLIT_CLOUD_GUIDE.md', 'launch_all.py'
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
        print("To launch the Streamlit app live:  streamlit run app.py")
    else:
        print("\nNote: Some components require attention. Check logs above.")
    print("=" * 80 + "\n")

if __name__ == '__main__':
    main()
