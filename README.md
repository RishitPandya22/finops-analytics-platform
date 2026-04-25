# 📊 FinOps Analytics Platform

![Python](https://img.shields.io/badge/Python-3.13-blue?style=flat-square&logo=python)
![Streamlit](https://img.shields.io/badge/Streamlit-Deployed-FF4B4B?style=flat-square&logo=streamlit)
![scikit-learn](https://img.shields.io/badge/scikit--learn-ML-F7931E?style=flat-square&logo=scikit-learn)
![License](https://img.shields.io/badge/License-MIT-green?style=flat-square)

> A production-grade Financial Operations Analytics Platform featuring revenue forecasting, customer churn prediction, and profitability intelligence — deployed as a Bloomberg terminal-style interactive dashboard.

**🔗 Live Demo:** [finops-analytics-platform.streamlit.app](https://finops-analytics-platform.streamlit.app/)

---

## 🚀 Overview

This end-to-end data science project tackles three core FinOps problems faced by SaaS and subscription businesses:

- **Revenue Forecasting** — Predict monthly recurring revenue by customer profile
- **Churn Prediction** — Identify at-risk customers before they leave
- **Profitability Scoring** — Score and segment customers by gross margin

Built with a dark Bloomberg terminal UI, real-time AI predictions, and interactive Plotly visualisations across 5,000 synthetic customers with realistic business logic.

---

## 📈 Model Performance

| Model | Algorithm | Key Metric | Score |
|---|---|---|---|
| Revenue Forecaster | Gradient Boosting Regressor | R² | 0.9989 |
| Churn Predictor | Random Forest Classifier | AUC-ROC | 0.6291 |
| Profitability Scorer | Gradient Boosting Regressor | R² | 0.8754 |

---

## 🖥️ Dashboard Features

| Tab | Features |
|---|---|
| 📈 Revenue Analysis | Revenue by plan, region, industry, tenure — bar, box, pie charts |
| 🚨 Churn Intelligence | Churn by plan & region, support ticket analysis, risk scatter |
| 💰 Profitability | Margin by industry & plan, LTV distribution, cost efficiency |
| 🤖 Live Predictor | Real-time predictions from all 3 models + churn risk gauge |

---

## 🗂️ Project Structure
finops-analytics-platform/
│
├── data/
│   ├── finops_data.csv              # Raw synthetic dataset
│   └── finops_processed.csv         # Feature engineered dataset
│
├── models/
│   ├── revenue_model.pkl            # Trained revenue forecaster
│   ├── churn_model.pkl              # Trained churn predictor
│   ├── profit_model.pkl             # Trained profitability scorer
│   └── model_metrics.csv           # Saved model performance metrics
│
├── assets/                          # EDA chart exports
│
├── reports/
│   └── finops_analytics_report.md  # Full professional analytics report
│
├── generate_data.py                 # Synthetic dataset generator
├── feature_engineering.py          # Data cleaning & feature engineering
├── train_models.py                  # Model training & comparison
├── app.py                           # Streamlit dashboard
└── requirements.txt                 # Dependencies
---

## ⚙️ Run Locally

```bash
# 1. Clone the repo
git clone https://github.com/RishitPandya22E/finops-analytics-platform.git
cd finops-analytics-platform

# 2. Create virtual environment
python -m venv venv
venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Generate dataset
python generate_data.py

# 5. Feature engineering
python feature_engineering.py

# 6. Train models
python models\train_models.py

# 7. Launch dashboard
streamlit run app.py
```

---

## 🧠 Key Business Insights

- **Enterprise customers are 6x more valuable** in LTV and churn at one-sixth the rate of Starter customers
- **5+ support tickets/month** is the strongest leading churn indicator — early intervention saves revenue
- **Login frequency below 4/month** triggers elevated churn risk — automated re-engagement should fire at this threshold
- **Technology & Finance verticals** deliver the highest gross margins — ideal for CAC investment
- **First 90 days are critical** — new customers churn at the highest rate across all plans

---

## 🛠️ Tech Stack

| Layer | Technology |
|---|---|
| Language | Python 3.13 |
| Data Processing | pandas, numpy |
| Machine Learning | scikit-learn, imbalanced-learn |
| Visualisation | plotly, matplotlib, seaborn |
| Dashboard | Streamlit |
| Deployment | Streamlit Cloud |
| Version Control | GitHub Desktop |

---

## 👨‍💻 Author

**Rishit Pandya**  
Master of Data Science — University of Adelaide 🇦🇺  
[GitHub](https://github.com/RishitPandya22) · [LinkedIn](www.linkedin.com/in/
rishit-pandya-854b7928a
)

---

*Part of an end-to-end data science portfolio spanning retail analytics, housing analysis, medical AI, sports prediction, stock forecasting, and financial operations intelligence.*