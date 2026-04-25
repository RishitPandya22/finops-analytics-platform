# FinOps Analytics Platform — Professional Analytics Report

**Author:** Rishit Pandya  
**Program:** Master of Data Science, University of Adelaide  
**Date:** June 2024  
**Dataset:** Synthetic FinOps Dataset (5,000 customers)

---

## Executive Summary

This report presents a comprehensive financial operations analytics study covering revenue forecasting, customer churn prediction, and profitability analysis across a simulated SaaS business with 5,000 customers. Three machine learning models were developed and deployed via an interactive Streamlit dashboard styled as a Bloomberg-terminal interface.

Key findings include a Revenue Forecasting model achieving R² of 0.9989, a Churn Prediction model achieving AUC-ROC of 0.6291 using Random Forest classification, and a Profitability Scoring model achieving R² of 0.8754. The platform enables real-time predictions for any customer profile, empowering data-driven decisions across revenue, retention, and margin management.

---

## 1. Project Overview

### 1.1 Objectives

- Forecast monthly recurring revenue (MRR) by customer profile
- Identify customers at high risk of churning before it happens
- Score and segment customers by profitability and gross margin
- Deliver all insights via an interactive, production-grade dashboard

### 1.2 Business Context

Financial operations analytics is one of the fastest-growing disciplines in modern data science. SaaS companies, subscription businesses, and financial services firms rely on FinOps analytics to understand their revenue drivers, reduce customer churn, and optimise profitability across customer segments. This project simulates a realistic FinOps environment with industry-standard metrics including MRR, LTV, gross margin, churn rate, and ARR.

### 1.3 Tech Stack

| Component | Technology |
|---|---|
| Data Processing | Python, pandas, numpy |
| Machine Learning | scikit-learn, imbalanced-learn |
| Visualisation | plotly, matplotlib, seaborn |
| Dashboard | Streamlit |
| Deployment | Streamlit Cloud |
| Version Control | GitHub |

---

## 2. Dataset Description

### 2.1 Data Generation

A synthetic dataset of 5,000 customers was engineered to simulate realistic SaaS business operations. Rather than using a generic public dataset, this dataset was purpose-built to include the specific financial metrics required for FinOps analysis — an approach commonly used in real data science workflows when proprietary business data is unavailable.

### 2.2 Raw Features

| Feature | Type | Description |
|---|---|---|
| customer_id | String | Unique customer identifier |
| signup_date | Date | Customer acquisition date |
| plan | Categorical | Subscription tier (Starter/Growth/Professional/Enterprise) |
| industry | Categorical | Customer industry vertical |
| region | Categorical | Geographic region |
| payment_method | Categorical | Payment type |
| tenure_months | Integer | Months since signup |
| monthly_revenue | Float | Monthly recurring revenue ($) |
| cogs | Float | Cost of goods sold ($) |
| gross_profit | Float | Revenue minus COGS ($) |
| gross_margin_pct | Float | Gross margin percentage |
| support_tickets | Integer | Monthly support tickets raised |
| login_frequency | Integer | Monthly login count |
| num_users | Integer | Number of seats/users |
| nps_score | Integer | Net Promoter Score (0-10) |
| ltv | Float | Customer lifetime value ($) |
| churned | Binary | 1 = churned, 0 = retained |

### 2.3 Key Business Logic in Data Generation

- Enterprise customers have a 5% base churn rate vs 30% for Starter — reflecting real SaaS retention curves
- Industry-specific COGS rates were applied — Retail at 55%, Technology at 32% — matching real-world margin profiles
- Churn probability increases with support ticket volume, low login frequency, short tenure, and invoice payment method
- NPS scores were generated independently of churn to avoid data leakage and ensure model integrity

### 2.4 Dataset Statistics

| Metric | Value |
|---|---|
| Total Customers | 5,000 |
| Overall Churn Rate | ~28% |
| Avg Monthly Revenue | Varies by plan |
| Date Range | Jan 2021 — Jan 2024 |
| Features (raw) | 17 |
| Features (engineered) | 25+ |

---

## 3. Exploratory Data Analysis

### 3.1 Revenue Analysis

Revenue distribution follows clear plan-based tiers with Starter customers generating ~$199/month and Enterprise customers generating ~$4,999/month. Significant variance exists within each tier due to usage-based pricing simulation (±25% of base price).

Regional analysis reveals North America and Europe as the highest revenue-generating regions, consistent with real-world SaaS market distribution. Industry analysis shows Finance and Technology verticals commanding higher average revenues per customer.

### 3.2 Churn Analysis

Churn rate decreases significantly with plan tier — Starter at ~30%, Growth at ~18%, Professional at ~10%, and Enterprise at ~5%. This validates the synthetic data generation logic and mirrors real-world SaaS retention patterns.

Key churn drivers identified through EDA include support ticket volume (customers with 5+ tickets churn at significantly higher rates), login frequency (low engagement strongly correlates with churn), and short tenure (customers in their first 3 months have elevated churn risk).

### 3.3 Profitability Analysis

Gross margin varies meaningfully by industry due to industry-specific COGS rates. Technology and Finance verticals show the highest margins (68–70%), while Retail and Manufacturing show compressed margins (45–50%). Plan tier shows minimal impact on gross margin percentage, confirming that margin is primarily driven by the cost structure of serving different industries.

---

## 4. Feature Engineering

Eight new features were engineered to improve model performance and business interpretability.

### 4.1 Engineered Features

**Revenue Per User** — Monthly revenue divided by number of users. Captures pricing efficiency and identifies under-monetised accounts.

**Cost Efficiency** — Gross profit as a proportion of revenue. A direct measure of operational profitability independent of revenue scale.

**Engagement Score** — A composite score (0–100) combining login frequency (50% weight), number of users (30% weight), and inverse support tickets (20% weight). Higher scores indicate more engaged, stickier customers.

**Risk Score** — A composite churn risk indicator combining support ticket volume, low login frequency, low NPS, and short tenure. Higher scores indicate elevated churn probability before the model is even applied.

**Annual Revenue (ARR)** — Monthly revenue multiplied by 12. The standard SaaS metric used by investors and operators.

**Revenue Segment** — Categorical bucketing of customers into Low (<$300), Medium ($300–$1000), High ($1000–$3000), and Premium (>$3000) tiers.

**Tenure Bucket** — Categorical bucketing into New (≤3 months), Growing (4–12 months), Established (13–24 months), and Veteran (>24 months).

**Profitability Tier** — Classification of customers into High (≥60% margin), Medium (45–60% margin), and Low (<45% margin) profitability buckets.

---

## 5. Model Development

### 5.1 Model 1 — Revenue Forecaster

**Algorithm:** Gradient Boosting Regressor  
**Target:** Monthly Revenue ($)  

| Metric | Value |
|---|---|
| RMSE | $65.08 |
| MAE | $45.02 |
| R² | 0.9989 |

The extremely high R² reflects the strong structural relationship between plan tier and revenue in the dataset — a realistic pattern since SaaS pricing is fundamentally plan-driven. The model correctly learns that plan type is the dominant revenue predictor, with usage variations accounting for residual variance.

### 5.2 Model 2 — Churn Predictor

**Algorithm:** Random Forest Classifier (winner after comparison with Gradient Boosting)  
**Target:** Churned (Binary 0/1)  
**Class Imbalance Handling:** SMOTE oversampling applied to training set

| Model | AUC-ROC | Accuracy |
|---|---|---|
| Random Forest | **0.6291** | 0.7060 |
| Gradient Boosting | 0.6078 | 0.7770 |

Random Forest was selected as the winner based on AUC-ROC — the primary metric for imbalanced classification problems. An AUC-ROC of 0.6291 reflects the genuine difficulty of predicting churn when multiple independent behavioural signals contribute. This is a realistic and honest score — real-world churn models at SaaS companies typically range between 0.60 and 0.80.

**Top Churn Features:**
1. Plan encoded (subscription tier)
2. Number of users
3. Engagement score
4. Revenue per user
5. Cost efficiency
6. Login frequency

### 5.3 Model 3 — Profitability Scorer

**Algorithm:** Gradient Boosting Regressor  
**Target:** Gross Margin Percentage  

| Metric | Value |
|---|---|
| RMSE | 2.98% |
| MAE | 2.57% |
| R² | 0.8754 |

R² of 0.8754 indicates the model explains 87.54% of gross margin variance. Key predictors include industry (due to industry-specific COGS rates), plan tier, and revenue per user. This model is particularly valuable for flagging low-margin customers who may require pricing or cost restructuring.

---

## 6. Dashboard Features

The Streamlit dashboard was built with a Bloomberg terminal-inspired dark UI providing four core modules.

**Revenue Analysis Tab** — Revenue breakdown by plan, region, industry, and tenure with interactive Plotly charts and segment pie charts.

**Churn Intelligence Tab** — Churn rate analysis by plan, region, support tickets, and login frequency. Risk scatter plot mapping engagement score against risk score with churn overlay.

**Profitability Tab** — Gross margin analysis by industry and plan. Profitability tier distribution, LTV boxplots by plan, and cost efficiency scatter analysis.

**Live Predictor Tab** — Real-time customer profile input with instant predictions from all three models including a churn risk gauge meter with colour-coded risk levels.

---

## 7. Key Business Insights

1. **Enterprise customers are 6x more valuable than Starter customers** in LTV terms and churn at one-sixth the rate — upselling is the highest-ROI retention strategy.

2. **Support ticket volume is a leading churn indicator** — customers raising 5+ tickets per month churn at rates 30–40% higher than low-ticket customers. Early support intervention can reduce churn.

3. **Login frequency below 4 per month is a red flag** — low engagement strongly predicts churn within the next billing cycle. Automated re-engagement campaigns should trigger at this threshold.

4. **Technology and Finance verticals deliver the highest margins** — customer acquisition investment should be weighted towards these verticals for optimal portfolio profitability.

5. **New customers (tenure <3 months) are the highest churn risk** — a structured onboarding programme targeting the first 90 days would have the highest retention ROI.

---

## 8. Limitations & Future Work

### 8.1 Current Limitations

- Dataset is synthetic — while realistic patterns were engineered, real-world data would introduce additional complexity and noise
- Churn model AUC-ROC of 0.63 leaves meaningful room for improvement with richer behavioural features
- No time-series forecasting — revenue predictions are cross-sectional rather than forward-looking

### 8.2 Future Enhancements

- Integrate real transaction data via CRM or billing system APIs
- Add LSTM or Prophet-based time-series revenue forecasting
- Build a customer health score combining all three model outputs
- Add cohort analysis and retention curve visualisations
- Implement automated churn alert notifications via email or Slack

---

## 9. Conclusion

This project demonstrates an end-to-end FinOps analytics pipeline from synthetic data engineering through feature engineering, multi-model machine learning, and production deployment. The platform provides actionable intelligence across three critical business dimensions — revenue, retention, and profitability — packaged in a professional Bloomberg-style dashboard ready for business stakeholder consumption.

The combination of realistic data engineering, honest model evaluation, and polished visualisation makes this a representative example of production data science work in the financial operations domain.

---

*Report generated as part of the Master of Data Science portfolio — University of Adelaide*