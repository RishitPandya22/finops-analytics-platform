import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random

np.random.seed(42)
random.seed(42)

N = 5000

industries = ['Technology', 'Healthcare', 'Retail', 'Finance', 'Manufacturing', 'Education', 'Logistics']
plans = ['Starter', 'Growth', 'Professional', 'Enterprise']
regions = ['North America', 'Europe', 'Asia Pacific', 'Latin America', 'Middle East']
payment_methods = ['Credit Card', 'Bank Transfer', 'Invoice', 'PayPal']

plan_base_revenue = {
    'Starter': 199,
    'Growth': 599,
    'Professional': 1499,
    'Enterprise': 4999
}

plan_churn_rate = {
    'Starter': 0.30,
    'Growth': 0.18,
    'Professional': 0.10,
    'Enterprise': 0.05
}

start_date = datetime(2021, 1, 1)
end_date = datetime(2024, 1, 1)

def random_date(start, end):
    return start + timedelta(days=random.randint(0, (end - start).days))

records = []

for i in range(N):
    plan = random.choice(plans)
    industry = random.choice(industries)
    region = random.choice(regions)
    payment = random.choice(payment_methods)

    signup_date = random_date(start_date, end_date)
    tenure_months = max(1, (datetime(2024, 6, 1) - signup_date).days // 30)

    base = plan_base_revenue[plan]
    monthly_revenue = round(base * np.random.uniform(0.85, 1.25), 2)

    support_tickets = np.random.poisson(3 if plan == 'Starter' else 2 if plan == 'Growth' else 1)
    login_frequency = np.random.poisson(15 if plan in ['Professional', 'Enterprise'] else 8)
    num_users = np.random.randint(1, 5) if plan == 'Starter' else \
                np.random.randint(3, 15) if plan == 'Growth' else \
                np.random.randint(10, 50) if plan == 'Professional' else \
                np.random.randint(40, 200)

    # More realistic COGS — varies by industry
    industry_cogs = {
        'Technology': 0.32, 'Healthcare': 0.45, 'Retail': 0.55,
        'Finance': 0.35, 'Manufacturing': 0.50, 'Education': 0.38, 'Logistics': 0.48
    }
    cogs_rate = industry_cogs[industry] + np.random.uniform(-0.05, 0.05)
    cogs = round(monthly_revenue * cogs_rate, 2)
    gross_profit = round(monthly_revenue - cogs, 2)
    gross_margin = round((gross_profit / monthly_revenue) * 100, 2)

    # Churn probability based on MULTIPLE realistic factors
    churn_prob = plan_churn_rate[plan]
    if support_tickets > 4:
        churn_prob += 0.12
    if support_tickets > 7:
        churn_prob += 0.08
    if login_frequency < 4:
        churn_prob += 0.10
    if login_frequency < 2:
        churn_prob += 0.08
    if tenure_months < 3:
        churn_prob += 0.06
    if payment == 'Invoice':
        churn_prob += 0.04
    if monthly_revenue < plan_base_revenue[plan] * 0.90:
        churn_prob += 0.05
    if num_users == 1:
        churn_prob += 0.05

    churn_prob = min(churn_prob, 0.85)
    churned = 1 if random.random() < churn_prob else 0

    # NPS is now INDEPENDENT — just customer sentiment noise
    nps_score = int(np.clip(np.random.normal(7, 2.5), 0, 10))

    ltv = round(monthly_revenue * tenure_months * (1 - cogs_rate), 2)

    records.append({
        'customer_id': f'CUST-{10000 + i}',
        'signup_date': signup_date.strftime('%Y-%m-%d'),
        'plan': plan,
        'industry': industry,
        'region': region,
        'payment_method': payment,
        'tenure_months': tenure_months,
        'monthly_revenue': monthly_revenue,
        'cogs': cogs,
        'gross_profit': gross_profit,
        'gross_margin_pct': gross_margin,
        'support_tickets': support_tickets,
        'login_frequency': login_frequency,
        'num_users': num_users,
        'nps_score': nps_score,
        'ltv': ltv,
        'churned': churned
    })

df = pd.DataFrame(records)
df.to_csv('data/finops_data.csv', index=False)
print(f"Dataset generated successfully!")
print(f"Shape: {df.shape}")
print(f"Churn rate: {df['churned'].mean():.2%}")
print(f"Total Revenue: ${df['monthly_revenue'].sum():,.0f}")