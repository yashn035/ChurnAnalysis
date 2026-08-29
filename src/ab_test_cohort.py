"""
A/B Test Cohort Generator for Customer Retention Strategy
Filters high-risk Month-to-Month subscribers (churn_probability > 0.50),
randomly partitions them 50/50 into Control and Variant groups,
exports ab_test_control.csv and ab_test_variant.csv, and prints summary metrics.
"""

import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

RANDOM_STATE = 42

def generate_ab_test_cohorts(input_csv='data/processed/churn_predictions_v2.csv'):
    if not os.path.exists(input_csv):
        input_csv = 'data/processed/churn_predictions.csv' if os.path.exists('data/processed/churn_predictions.csv') else 'churn_predictions_v2.csv'
        if not os.path.exists(input_csv):
            raise FileNotFoundError("Neither 'data/processed/churn_predictions_v2.csv' nor 'churn_predictions.csv' was found.")
            
    print(f"[INFO] Loading predictions from '{input_csv}'...")
    df = pd.read_csv(input_csv)
    
    # Filter for Month-to-Month subscribers with churn_probability > 0.50
    # Handles both string 'Month-to-month' and numeric 0
    is_month_to_month = (df['Contract'] == 'Month-to-month') | (df['Contract'] == 0) | (df['Contract'] == '0')
    is_high_risk = df['churn_probability'] > 0.50
    
    cohort = df[is_month_to_month & is_high_risk].copy()
    
    total_target_customers = len(cohort)
    print(f"[INFO] Identified {total_target_customers} high-risk Month-to-Month subscribers (P > 0.50).")
    
    if total_target_customers < 2:
        print("[WARNING] Not enough target customers (minimum 2 required for split). Lowering probability threshold to 0.40...")
        is_high_risk = df['churn_probability'] > 0.40
        cohort = df[is_month_to_month & is_high_risk].copy()
        total_target_customers = len(cohort)

    # Perform 50/50 randomized split
    control_df, variant_df = train_test_split(
        cohort,
        test_size=0.50,
        random_state=RANDOM_STATE,
        shuffle=True
    )
    
    # Add group metadata
    control_df = control_df.copy()
    variant_df = variant_df.copy()
    
    control_df['ab_group'] = 'Control'
    control_df['treatment_offer'] = 'Standard Care (No Discount)'
    
    variant_df['ab_group'] = 'Variant'
    variant_df['treatment_offer'] = '15% Monthly Discount for 1-Year Lock-In'
    
    # Export CSV files
    os.makedirs('data/processed', exist_ok=True)
    control_file = 'data/processed/ab_test_control.csv'
    variant_file = 'data/processed/ab_test_variant.csv'
    
    control_df.to_csv(control_file, index=False)
    variant_df.to_csv(variant_file, index=False)
    
    print(f"[SUCCESS] Exported {len(control_df)} customers to '{control_file}'")
    print(f"[SUCCESS] Exported {len(variant_df)} customers to '{variant_file}'\n")
    
    # Compute Summary Statistics Table
    summary_data = [
        {
            'Group': 'Control',
            'Sample Size': len(control_df),
            'Avg Churn Prob': f"{control_df['churn_probability'].mean():.4f}",
            'Avg Monthly Charge': f"${control_df['MonthlyCharges'].mean():.2f}",
            'Avg Tenure (mos)': f"{control_df['tenure'].mean():.1f}",
            'Treatment Offer': 'Standard Care'
        },
        {
            'Group': 'Variant',
            'Sample Size': len(variant_df),
            'Avg Churn Prob': f"{variant_df['churn_probability'].mean():.4f}",
            'Avg Monthly Charge': f"${variant_df['MonthlyCharges'].mean():.2f}",
            'Avg Tenure (mos)': f"{variant_df['tenure'].mean():.1f}",
            'Treatment Offer': '15% Discount (1-Yr Lock-In)'
        }
    ]
    
    summary_df = pd.DataFrame(summary_data)
    
    print("=" * 80)
    print("               A/B RETENTION TRIAL COHORT SUMMARY TABLE")
    print("=" * 80)
    print(summary_df.to_string(index=False))
    print("=" * 80)

if __name__ == '__main__':
    generate_ab_test_cohorts()
