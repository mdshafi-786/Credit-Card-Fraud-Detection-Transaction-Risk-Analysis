import pandas as pd
import numpy as np
import os

# Load the dataset
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
df = pd.read_csv(os.path.join(BASE_DIR, "transactions.csv"))

print("=" * 80)
print("DATASET OVERVIEW")
print("=" * 80)
print(f"Shape: {df.shape[0]} rows x {df.shape[1]} columns")
print(f"\nColumns ({len(df.columns)}):")
for i, col in enumerate(df.columns):
    print(f"  {i+1}. {col} - dtype: {df[col].dtype}")

print(f"\n{'=' * 80}")
print("MISSING VALUES")
print("=" * 80)
missing = df.isnull().sum()
print(missing[missing > 0] if missing.sum() > 0 else "No missing values found!")

print(f"\n{'=' * 80}")
print("BASIC STATISTICS - NUMERICAL COLUMNS")
print("=" * 80)
print(df.describe().to_string())

print(f"\n{'=' * 80}")
print("TARGET VARIABLE (is_fraud) DISTRIBUTION")
print("=" * 80)
fraud_counts = df['is_fraud'].value_counts()
print(f"Legitimate (0): {fraud_counts.get(0, 0)} ({fraud_counts.get(0, 0)/len(df)*100:.2f}%)")
print(f"Fraudulent (1): {fraud_counts.get(1, 0)} ({fraud_counts.get(1, 0)/len(df)*100:.2f}%)")
print(f"Imbalance Ratio: 1:{fraud_counts.get(0, 0)/fraud_counts.get(1, 1):.1f}")

print(f"\n{'=' * 80}")
print("CATEGORICAL COLUMNS ANALYSIS")
print("=" * 80)
cat_cols = ['merchant_category', 'customer_gender', 'transaction_city', 'transaction_state',
            'card_type', 'device_type', 'channel', 'day_of_week']
for col in cat_cols:
    print(f"\n--- {col} ---")
    vc = df[col].value_counts()
    print(f"Unique values: {df[col].nunique()}")
    print(vc.head(10).to_string())

print(f"\n{'=' * 80}")
print("FRAUD RATE BY CATEGORY")
print("=" * 80)
for col in cat_cols:
    print(f"\n--- Fraud Rate by {col} ---")
    fraud_rate = df.groupby(col)['is_fraud'].mean().sort_values(ascending=False)
    print(fraud_rate.head(10).to_string())

print(f"\n{'=' * 80}")
print("FRAUD vs NON-FRAUD COMPARISON (Numerical)")
print("=" * 80)
num_cols = ['transaction_amount_inr', 'customer_age', 'customer_income_monthly_inr',
            'customer_tenure_months', 'avg_transaction_amount_inr', 'transaction_count_30d',
            'distance_from_home_km', 'failed_attempts_24h', 'previous_fraud_count',
            'account_age_months', 'amount_to_customer_avg_ratio']
for col in num_cols:
    fraud_mean = df[df['is_fraud']==1][col].mean()
    legit_mean = df[df['is_fraud']==0][col].mean()
    print(f"{col}:")
    print(f"  Fraud Mean: {fraud_mean:.2f} | Legit Mean: {legit_mean:.2f} | Diff: {((fraud_mean-legit_mean)/legit_mean*100):.1f}%")

print(f"\n{'=' * 80}")
print("TRANSACTION HOUR ANALYSIS")
print("=" * 80)
hour_fraud = df.groupby('transaction_hour')['is_fraud'].agg(['count', 'sum', 'mean'])
hour_fraud.columns = ['total', 'fraud_count', 'fraud_rate']
print(hour_fraud.to_string())

print(f"\n{'=' * 80}")
print("INTERNATIONAL TRANSACTION ANALYSIS")
print("=" * 80)
intl = df.groupby('international_transaction')['is_fraud'].agg(['count', 'sum', 'mean'])
intl.columns = ['total', 'fraud_count', 'fraud_rate']
print(intl.to_string())

print(f"\n{'=' * 80}")
print("TRANSACTION AMOUNT RANGES")
print("=" * 80)
bins = [0, 500, 1000, 2000, 5000, 10000, 20000, 50000, float('inf')]
labels = ['0-500', '500-1K', '1K-2K', '2K-5K', '5K-10K', '10K-20K', '20K-50K', '50K+']
df['amount_range'] = pd.cut(df['transaction_amount_inr'], bins=bins, labels=labels)
amt_fraud = df.groupby('amount_range', observed=False)['is_fraud'].agg(['count', 'sum', 'mean'])
amt_fraud.columns = ['total', 'fraud_count', 'fraud_rate']
print(amt_fraud.to_string())

print(f"\n{'=' * 80}")
print("TOP FRAUD CITIES")
print("=" * 80)
city_fraud = df.groupby('transaction_city')['is_fraud'].agg(['count', 'sum', 'mean']).sort_values('sum', ascending=False)
city_fraud.columns = ['total', 'fraud_count', 'fraud_rate']
print(city_fraud.head(15).to_string())

print(f"\n{'=' * 80}")
print("CORRELATION WITH FRAUD")
print("=" * 80)
corr_with_fraud = df[num_cols + ['is_fraud']].corr()['is_fraud'].drop('is_fraud').sort_values(ascending=False)
print(corr_with_fraud.to_string())

print(f"\n{'=' * 80}")
print("UNIQUE COUNTS")
print("=" * 80)
print(f"Unique Customers: {df['customer_id'].nunique()}")
print(f"Unique Merchants: {df['merchant_id'].nunique()}")
print(f"Unique Cities: {df['transaction_city'].nunique()}")
print(f"Unique States: {df['transaction_state'].nunique()}")
print(f"Date Range: {df['transaction_datetime'].min()} to {df['transaction_datetime'].max()}")

print("\n\nANALYSIS COMPLETE!")
