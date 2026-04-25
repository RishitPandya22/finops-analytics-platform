import pandas as pd
import numpy as np

df = pd.read_csv('data/finops_data.csv')
df['signup_date'] = pd.to_datetime(df['signup_date'])

print("=" * 60)
print("STAGE 3 - DATA CLEANING & FEATURE ENGINEERING")
print("=" * 60)

# ── Step 1: Check & Handle Missing Values ─────────────────────────
print(f"\nMissing values before cleaning:")
print(df.isnull().sum())

df = df.dropna()
print(f"\nShape after cleaning: {df.shape}")

# ── Step 2: Remove Duplicates ──────────────────────────────────────
before = len(df)
df = df.drop_duplicates(subset='customer_id')
print(f"Duplicates removed: {before - len(df)}")

# ── Step 3: Fix Data Types ─────────────────────────────────────────
df['churned'] = df['churned'].astype(int)
df['support_tickets'] = df['support_tickets'].astype(int)
df['login_frequency'] = df['login_frequency'].astype(int)
df['num_users'] = df['num_users'].astype(int)

# ── Step 4: Encode Categorical Columns ────────────────────────────
plan_map = {'Starter': 0, 'Growth': 1, 'Professional': 2, 'Enterprise': 3}
df['plan_encoded'] = df['plan'].map(plan_map)

region_dummies = pd.get_dummies(df['region'], prefix='region')
industry_dummies = pd.get_dummies(df['industry'], prefix='industry')
payment_dummies = pd.get_dummies(df['payment_method'], prefix='payment')

df = pd.concat([df, region_dummies, industry_dummies, payment_dummies], axis=1)

# ── Step 5: New Features ───────────────────────────────────────────

# Revenue per user
df['revenue_per_user'] = (df['monthly_revenue'] / df['num_users']).round(2)

# Cost efficiency ratio
df['cost_efficiency'] = (df['gross_profit'] / df['monthly_revenue']).round(4)

# Engagement score (0-100)
max_login = df['login_frequency'].max()
max_users = df['num_users'].max()
df['engagement_score'] = (
    (df['login_frequency'] / max_login * 50) +
    (df['num_users'] / max_users * 30) +
    ((10 - df['support_tickets'].clip(0, 10)) / 10 * 20)
).round(2)

# Risk score (higher = more likely to churn)
df['risk_score'] = (
    (df['support_tickets'] * 3) +
    ((10 - df['login_frequency'].clip(0, 10)) * 2) +
    ((10 - df['nps_score']) * 2) +
    (df['tenure_months'].apply(lambda x: 10 if x < 3 else 5 if x < 6 else 0))
).round(2)

# Annual Revenue Run Rate
df['annual_revenue'] = (df['monthly_revenue'] * 12).round(2)

# Revenue segment
def revenue_segment(rev):
    if rev < 300:
        return 'Low'
    elif rev < 1000:
        return 'Medium'
    elif rev < 3000:
        return 'High'
    else:
        return 'Premium'

df['revenue_segment'] = df['monthly_revenue'].apply(revenue_segment)

# Tenure bucket
def tenure_bucket(months):
    if months <= 3:
        return 'New'
    elif months <= 12:
        return 'Growing'
    elif months <= 24:
        return 'Established'
    else:
        return 'Veteran'

df['tenure_bucket'] = df['tenure_months'].apply(tenure_bucket)

# Profitability tier
def profit_tier(margin):
    if margin >= 60:
        return 'High'
    elif margin >= 45:
        return 'Medium'
    else:
        return 'Low'

df['profitability_tier'] = df['gross_margin_pct'].apply(profit_tier)

# ── Step 6: Save Processed Data ───────────────────────────────────
df.to_csv('data/finops_processed.csv', index=False)

print("\nNew features created:")
new_features = ['revenue_per_user', 'cost_efficiency', 'engagement_score',
                'risk_score', 'annual_revenue', 'revenue_segment',
                'tenure_bucket', 'profitability_tier']
for f in new_features:
    print(f"  ✓ {f}")

print(f"\nFinal dataset shape: {df.shape}")
print(f"\nSample of new features:")
print(df[['customer_id', 'plan', 'engagement_score', 'risk_score',
          'revenue_segment', 'tenure_bucket', 'profitability_tier']].head(10))

print("\nProcessed data saved to data/finops_processed.csv")
print("\nStage 3 complete!")