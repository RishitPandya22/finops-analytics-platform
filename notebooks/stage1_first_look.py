import pandas as pd
import numpy as np

df = pd.read_csv('data/finops_data.csv')

print("=" * 60)
print("FINOPS DATASET - FIRST LOOK")
print("=" * 60)

print(f"\n Shape: {df.shape}")
print(f" Rows: {df.shape[0]:,}")
print(f" Columns: {df.shape[1]}")

print("\n Column Names & Data Types:")
print(df.dtypes)

print("\n First 5 Rows:")
print(df.head())

print("\n Missing Values:")
print(df.isnull().sum())

print("\n Numerical Summary:")
print(df.describe().round(2))

print("\n Churn Distribution:")
print(df['churned'].value_counts())
print(f" Churn Rate: {df['churned'].mean():.2%}")

print("\n Plan Distribution:")
print(df['plan'].value_counts())

print("\n Revenue by Plan:")
print(df.groupby('plan')['monthly_revenue'].mean().round(2))

print("\n Region Distribution:")
print(df['region'].value_counts())