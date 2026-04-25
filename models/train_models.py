import pandas as pd
import numpy as np
import joblib
import os
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import GradientBoostingRegressor, GradientBoostingClassifier, RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (mean_squared_error, r2_score, mean_absolute_error,
                             classification_report, roc_auc_score, accuracy_score)
from imblearn.over_sampling import SMOTE

os.makedirs('models', exist_ok=True)

df = pd.read_csv('data/finops_processed.csv')

# ── Drop original text columns to avoid string leaking into features ──
df = df.drop(columns=['customer_id', 'signup_date', 'plan', 'industry',
                       'region', 'payment_method', 'month',
                       'revenue_segment', 'tenure_bucket', 'profitability_tier'], errors='ignore')

print("=" * 60)
print("STAGE 4 - MODEL BUILDING & COMPARISON")
print("=" * 60)

# ── Feature columns used across all models ─────────────────────────
BASE_FEATURES = [
    'plan_encoded', 'tenure_months', 'support_tickets',
    'login_frequency', 'num_users', 'nps_score',
    'revenue_per_user', 'cost_efficiency', 'engagement_score',
    'risk_score'
]

region_cols = [c for c in df.columns if c.startswith('region_')]
industry_cols = [c for c in df.columns if c.startswith('industry_')]
payment_cols = [c for c in df.columns if c.startswith('payment_')]

ALL_FEATURES = BASE_FEATURES + region_cols + industry_cols + payment_cols

# ══════════════════════════════════════════════════════════════════
# MODEL 1 — REVENUE FORECASTER (Gradient Boosting Regressor)
# ══════════════════════════════════════════════════════════════════
print("\n[MODEL 1] Revenue Forecaster — Gradient Boosting Regressor")
print("-" * 60)

X_rev = df[ALL_FEATURES]
y_rev = df['monthly_revenue']

X_train_r, X_test_r, y_train_r, y_test_r = train_test_split(
    X_rev, y_rev, test_size=0.2, random_state=42)

revenue_model = GradientBoostingRegressor(
    n_estimators=200,
    learning_rate=0.08,
    max_depth=4,
    min_samples_split=10,
    subsample=0.85,
    random_state=42
)

revenue_model.fit(X_train_r, y_train_r)
y_pred_r = revenue_model.predict(X_test_r)

rmse_r = np.sqrt(mean_squared_error(y_test_r, y_pred_r))
mae_r = mean_absolute_error(y_test_r, y_pred_r)
r2_r = r2_score(y_test_r, y_pred_r)

print(f"  RMSE:  ${rmse_r:,.2f}")
print(f"  MAE:   ${mae_r:,.2f}")
print(f"  R²:    {r2_r:.4f}")

joblib.dump(revenue_model, 'models/revenue_model.pkl')
joblib.dump(ALL_FEATURES, 'models/revenue_features.pkl')
print("  Model saved!")

# ══════════════════════════════════════════════════════════════════
# MODEL 2 — CHURN PREDICTOR (Random Forest vs Gradient Boosting)
# ══════════════════════════════════════════════════════════════════
print("\n[MODEL 2] Churn Predictor — Model Comparison")
print("-" * 60)

X_churn = df[ALL_FEATURES]
y_churn = df['churned']

X_train_c, X_test_c, y_train_c, y_test_c = train_test_split(
    X_churn, y_churn, test_size=0.2, random_state=42, stratify=y_churn)

sm = SMOTE(random_state=42)
X_train_sm, y_train_sm = sm.fit_resample(X_train_c, y_train_c)
print(f"  After SMOTE — Class 0: {sum(y_train_sm==0)}, Class 1: {sum(y_train_sm==1)}")

rf_model = RandomForestClassifier(
    n_estimators=200,
    max_depth=8,
    min_samples_split=10,
    random_state=42,
    n_jobs=-1
)
rf_model.fit(X_train_sm, y_train_sm)
rf_pred = rf_model.predict(X_test_c)
rf_prob = rf_model.predict_proba(X_test_c)[:, 1]
rf_auc = roc_auc_score(y_test_c, rf_prob)
rf_acc = accuracy_score(y_test_c, rf_pred)

print(f"\n  Random Forest:")
print(f"    AUC-ROC:  {rf_auc:.4f}")
print(f"    Accuracy: {rf_acc:.4f}")

gb_churn_model = GradientBoostingClassifier(
    n_estimators=200,
    learning_rate=0.08,
    max_depth=4,
    subsample=0.85,
    random_state=42
)
gb_churn_model.fit(X_train_sm, y_train_sm)
gb_pred = gb_churn_model.predict(X_test_c)
gb_prob = gb_churn_model.predict_proba(X_test_c)[:, 1]
gb_auc = roc_auc_score(y_test_c, gb_prob)
gb_acc = accuracy_score(y_test_c, gb_pred)

print(f"\n  Gradient Boosting:")
print(f"    AUC-ROC:  {gb_auc:.4f}")
print(f"    Accuracy: {gb_acc:.4f}")

if gb_auc >= rf_auc:
    best_churn_model = gb_churn_model
    best_churn_name = "Gradient Boosting"
    best_auc = gb_auc
    best_acc = gb_acc
else:
    best_churn_model = rf_model
    best_churn_name = "Random Forest"
    best_auc = rf_auc
    best_acc = rf_acc

print(f"\n  Winner: {best_churn_name} (AUC-ROC: {best_auc:.4f})")

joblib.dump(best_churn_model, 'models/churn_model.pkl')
joblib.dump(ALL_FEATURES, 'models/churn_features.pkl')
joblib.dump(best_churn_name, 'models/churn_model_name.pkl')
print("  Model saved!")

# ══════════════════════════════════════════════════════════════════
# MODEL 3 — PROFITABILITY SCORER (Gradient Boosting Regressor)
# ══════════════════════════════════════════════════════════════════
print("\n[MODEL 3] Profitability Scorer — Gradient Boosting Regressor")
print("-" * 60)

profit_features = [
    'plan_encoded', 'tenure_months', 'num_users',
    'login_frequency', 'support_tickets', 'nps_score',
    'revenue_per_user', 'engagement_score', 'annual_revenue'
] + region_cols + industry_cols

X_profit = df[profit_features]
y_profit = df['gross_margin_pct']

X_train_p, X_test_p, y_train_p, y_test_p = train_test_split(
    X_profit, y_profit, test_size=0.2, random_state=42)

profit_model = GradientBoostingRegressor(
    n_estimators=200,
    learning_rate=0.08,
    max_depth=4,
    subsample=0.85,
    random_state=42
)

profit_model.fit(X_train_p, y_train_p)
y_pred_p = profit_model.predict(X_test_p)

rmse_p = np.sqrt(mean_squared_error(y_test_p, y_pred_p))
mae_p = mean_absolute_error(y_test_p, y_pred_p)
r2_p = r2_score(y_test_p, y_pred_p)

print(f"  RMSE:  {rmse_p:.4f}%")
print(f"  MAE:   {mae_p:.4f}%")
print(f"  R²:    {r2_p:.4f}")

joblib.dump(profit_model, 'models/profit_model.pkl')
joblib.dump(profit_features, 'models/profit_features.pkl')
print("  Model saved!")

# ══════════════════════════════════════════════════════════════════
# FEATURE IMPORTANCE SUMMARY
# ══════════════════════════════════════════════════════════════════
print("\n[FEATURE IMPORTANCE] Top 8 — Churn Model")
print("-" * 60)
importance_df = pd.DataFrame({
    'feature': ALL_FEATURES,
    'importance': best_churn_model.feature_importances_
}).sort_values('importance', ascending=False).head(8)
for _, row in importance_df.iterrows():
    bar = '█' * int(row['importance'] * 200)
    print(f"  {row['feature']:<25} {bar} {row['importance']:.4f}")

# ══════════════════════════════════════════════════════════════════
# SAVE MODEL METRICS FOR DASHBOARD
# ══════════════════════════════════════════════════════════════════
metrics = {
    'revenue_rmse': round(rmse_r, 2),
    'revenue_mae': round(mae_r, 2),
    'revenue_r2': round(r2_r, 4),
    'churn_auc': round(best_auc, 4),
    'churn_accuracy': round(best_acc, 4),
    'churn_model_name': best_churn_name,
    'profit_rmse': round(rmse_p, 4),
    'profit_mae': round(mae_p, 4),
    'profit_r2': round(r2_p, 4)
}

pd.DataFrame([metrics]).to_csv('models/model_metrics.csv', index=False)
print("\nModel metrics saved to models/model_metrics.csv")

print("\n" + "=" * 60)
print("ALL 3 MODELS TRAINED & SAVED SUCCESSFULLY!")
print("=" * 60)