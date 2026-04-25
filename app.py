import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import joblib
import os

st.set_page_config(
    page_title="FinOps Analytics Platform",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── CSS ────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
html, body, [class*="css"] { font-family: 'Inter', sans-serif; background-color: #0A0E1A; color: #E2E8F0; }
.main { background-color: #0A0E1A; }
section[data-testid="stSidebar"] { background-color: #0D1117; border-right: 1px solid #1E293B; }
.kpi-card { background: #111827; border: 1px solid #1E293B; border-radius: 12px; padding: 20px 24px; text-align: center; }
.kpi-label { font-size: 12px; color: #64748B; text-transform: uppercase; letter-spacing: 1.5px; font-weight: 600; margin-bottom: 8px; }
.kpi-value { font-size: 28px; font-weight: 700; color: #00D4FF; }
.kpi-delta { font-size: 12px; color: #00FF88; margin-top: 4px; }
.section-header { background: #111827; border-left: 3px solid #00D4FF; border-radius: 0 8px 8px 0; padding: 10px 16px; margin: 24px 0 16px 0; }
.section-title { font-size: 14px; font-weight: 600; color: #00D4FF; text-transform: uppercase; letter-spacing: 1.5px; margin: 0; }
.metric-row { display: flex; gap: 12px; margin-bottom: 16px; }
.model-badge { background: #0F2027; border: 1px solid #00D4FF; border-radius: 20px; padding: 4px 14px; font-size: 12px; color: #00D4FF; display: inline-block; margin: 4px; }
.risk-high { color: #FF6B6B; font-weight: 700; font-size: 20px; }
.risk-medium { color: #FFD93D; font-weight: 700; font-size: 20px; }
.risk-low { color: #00FF88; font-weight: 700; font-size: 20px; }
div[data-testid="stTab"] { background: #111827; }
.stTabs [data-baseweb="tab-list"] { background-color: #111827; border-bottom: 1px solid #1E293B; gap: 4px; }
.stTabs [data-baseweb="tab"] { background-color: transparent; color: #64748B; border-radius: 8px 8px 0 0; padding: 10px 20px; font-size: 13px; font-weight: 500; }
.stTabs [aria-selected="true"] { background-color: #0A0E1A; color: #00D4FF; border-bottom: 2px solid #00D4FF; }
.stSlider > div > div { background: #1E293B; }
div[data-testid="stMetric"] { background: #111827; border: 1px solid #1E293B; border-radius: 12px; padding: 16px; }
</style>
""", unsafe_allow_html=True)

PLOTLY_THEME = dict(
    paper_bgcolor='#0A0E1A',
    plot_bgcolor='#111827',
    font=dict(color='#E2E8F0', family='Inter'),
    xaxis=dict(gridcolor='#1E293B', linecolor='#1E293B'),
    yaxis=dict(gridcolor='#1E293B', linecolor='#1E293B'),
    colorway=['#00D4FF', '#FF6B6B', '#00FF88', '#FFD93D', '#C77DFF', '#FF9F43']
)

# ── Load Data ──────────────────────────────────────────────────────
@st.cache_data
def load_data():
    if not os.path.exists('data/finops_processed.csv'):
        os.system('python generate_data.py')
        os.system('python feature_engineering.py')
    return pd.read_csv('data/finops_processed.csv')

@st.cache_resource
def load_models():
    models = {}
    if not os.path.exists('models/churn_model.pkl'):
        os.system('python train_models.py')
    try:
        models['revenue'] = joblib.load('models/revenue_model.pkl')
        models['revenue_features'] = joblib.load('models/revenue_features.pkl')
        models['churn'] = joblib.load('models/churn_model.pkl')
        models['churn_features'] = joblib.load('models/churn_features.pkl')
        models['churn_name'] = joblib.load('models/churn_model_name.pkl')
        models['profit'] = joblib.load('models/profit_model.pkl')
        models['profit_features'] = joblib.load('models/profit_features.pkl')
        if os.path.exists('models/model_metrics.csv'):
            models['metrics'] = pd.read_csv('models/model_metrics.csv').iloc[0]
    except Exception as e:
        st.error(f"Model loading error: {e}")
    return models

df = load_data()
models = load_models()

# ── Sidebar ────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("<div style='text-align:center;padding:16px 0 8px'><span style='font-size:28px'>📊</span><br><span style='font-size:16px;font-weight:700;color:#00D4FF'>FinOps Analytics</span><br><span style='font-size:11px;color:#64748B'>PLATFORM v1.0</span></div>", unsafe_allow_html=True)
    st.markdown("<hr style='border-color:#1E293B;margin:12px 0'>", unsafe_allow_html=True)

    st.markdown("<p style='font-size:11px;color:#64748B;text-transform:uppercase;letter-spacing:1px;margin-bottom:8px'>Filters</p>", unsafe_allow_html=True)
    plan_filter = st.multiselect("Plan", options=['Starter', 'Growth', 'Professional', 'Enterprise'], default=['Starter', 'Growth', 'Professional', 'Enterprise'])
    region_filter = st.multiselect("Region", options=df['region'].unique().tolist(), default=df['region'].unique().tolist())
    churn_filter = st.selectbox("Customer Status", ["All Customers", "Active Only", "Churned Only"])

    st.markdown("<hr style='border-color:#1E293B;margin:12px 0'>", unsafe_allow_html=True)
    st.markdown("<p style='font-size:11px;color:#64748B;text-transform:uppercase;letter-spacing:1px;margin-bottom:8px'>Model Performance</p>", unsafe_allow_html=True)
    if 'metrics' in models:
        m = models['metrics']
        st.markdown(f"<div class='model-badge'>Revenue R² {m['revenue_r2']:.4f}</div>", unsafe_allow_html=True)
        st.markdown(f"<div class='model-badge'>Churn AUC {m['churn_auc']:.4f}</div>", unsafe_allow_html=True)
        st.markdown(f"<div class='model-badge'>Profit R² {m['profit_r2']:.4f}</div>", unsafe_allow_html=True)

    st.markdown("<hr style='border-color:#1E293B;margin:12px 0'>", unsafe_allow_html=True)
    st.markdown("<p style='font-size:10px;color:#334155;text-align:center'>Built by Rishit Pandya<br>MDS · University of Adelaide</p>", unsafe_allow_html=True)

# ── Apply Filters ──────────────────────────────────────────────────
filtered = df[df['plan'].isin(plan_filter) & df['region'].isin(region_filter)]
if churn_filter == "Active Only":
    filtered = filtered[filtered['churned'] == 0]
elif churn_filter == "Churned Only":
    filtered = filtered[filtered['churned'] == 1]

# ── Header ─────────────────────────────────────────────────────────
st.markdown("<h1 style='font-size:26px;font-weight:700;color:#E2E8F0;margin-bottom:4px'>FinOps Analytics Platform</h1>", unsafe_allow_html=True)
st.markdown("<p style='font-size:13px;color:#64748B;margin-bottom:24px'>Revenue Forecasting · Customer Churn · Profitability Intelligence</p>", unsafe_allow_html=True)

# ── KPI Row ────────────────────────────────────────────────────────
k1, k2, k3, k4, k5 = st.columns(5)
total_rev = filtered['monthly_revenue'].sum()
churn_rate = filtered['churned'].mean() * 100
avg_margin = filtered['gross_margin_pct'].mean()
avg_ltv = filtered['ltv'].mean()
total_customers = len(filtered)

with k1:
    st.markdown(f"<div class='kpi-card'><div class='kpi-label'>Total Revenue</div><div class='kpi-value'>${total_rev/1e6:.2f}M</div><div class='kpi-delta'>↑ Monthly ARR</div></div>", unsafe_allow_html=True)
with k2:
    st.markdown(f"<div class='kpi-card'><div class='kpi-label'>Churn Rate</div><div class='kpi-value' style='color:#FF6B6B'>{churn_rate:.1f}%</div><div class='kpi-delta' style='color:#FF6B6B'>↓ Target &lt;15%</div></div>", unsafe_allow_html=True)
with k3:
    st.markdown(f"<div class='kpi-card'><div class='kpi-label'>Avg Gross Margin</div><div class='kpi-value' style='color:#00FF88'>{avg_margin:.1f}%</div><div class='kpi-delta'>↑ Profitability</div></div>", unsafe_allow_html=True)
with k4:
    st.markdown(f"<div class='kpi-card'><div class='kpi-label'>Avg LTV</div><div class='kpi-value' style='color:#FFD93D'>${avg_ltv:,.0f}</div><div class='kpi-delta'>↑ Lifetime Value</div></div>", unsafe_allow_html=True)
with k5:
    st.markdown(f"<div class='kpi-card'><div class='kpi-label'>Customers</div><div class='kpi-value' style='color:#C77DFF'>{total_customers:,}</div><div class='kpi-delta'>↑ Active Base</div></div>", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ── Tabs ───────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs(["📈 Revenue Analysis", "🚨 Churn Intelligence", "💰 Profitability", "🤖 Live Predictor"])

# ══════════════════════════════════════════════════════════════════
# TAB 1 — REVENUE ANALYSIS
# ══════════════════════════════════════════════════════════════════
with tab1:
    st.markdown("<div class='section-header'><p class='section-title'>Revenue Overview</p></div>", unsafe_allow_html=True)

    c1, c2 = st.columns(2)
    with c1:
        rev_plan = filtered.groupby('plan')['monthly_revenue'].sum().reset_index()
        plan_order = ['Starter', 'Growth', 'Professional', 'Enterprise']
        rev_plan['plan'] = pd.Categorical(rev_plan['plan'], categories=plan_order, ordered=True)
        rev_plan = rev_plan.sort_values('plan')
        fig = px.bar(rev_plan, x='plan', y='monthly_revenue', color='plan',
                     color_discrete_sequence=['#00D4FF', '#00FF88', '#FFD93D', '#C77DFF'],
                     title='Total Revenue by Plan')
        fig.update_layout(**PLOTLY_THEME, showlegend=False, title_font_color='#E2E8F0')
        fig.update_traces(marker_line_width=0)
        st.plotly_chart(fig, use_container_width=True)

    with c2:
        rev_region = filtered.groupby('region')['monthly_revenue'].sum().reset_index().sort_values('monthly_revenue', ascending=True)
        fig = px.bar(rev_region, x='monthly_revenue', y='region', orientation='h',
                     title='Revenue by Region', color='monthly_revenue',
                     color_continuous_scale=[[0, '#1E293B'], [1, '#00D4FF']])
        fig.update_layout(**PLOTLY_THEME, coloraxis_showscale=False, title_font_color='#E2E8F0')
        fig.update_traces(marker_line_width=0)
        st.plotly_chart(fig, use_container_width=True)

    c3, c4 = st.columns(2)
    with c3:
        fig = px.box(filtered, x='plan', y='monthly_revenue', color='plan',
                     color_discrete_sequence=['#00D4FF', '#00FF88', '#FFD93D', '#C77DFF'],
                     title='Revenue Distribution by Plan', category_orders={'plan': plan_order})
        fig.update_layout(**PLOTLY_THEME, showlegend=False, title_font_color='#E2E8F0')
        st.plotly_chart(fig, use_container_width=True)

    with c4:
        rev_ind = filtered.groupby('industry')['monthly_revenue'].mean().reset_index().sort_values('monthly_revenue', ascending=False)
        fig = px.bar(rev_ind, x='industry', y='monthly_revenue',
                     title='Avg Revenue by Industry',
                     color='monthly_revenue', color_continuous_scale=[[0, '#1E293B'], [1, '#C77DFF']])
        fig.update_layout(**PLOTLY_THEME, coloraxis_showscale=False, title_font_color='#E2E8F0', xaxis_tickangle=-30)
        fig.update_traces(marker_line_width=0)
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("<div class='section-header'><p class='section-title'>Revenue Segments</p></div>", unsafe_allow_html=True)
    c5, c6 = st.columns(2)
    with c5:
        seg = filtered['revenue_segment'].value_counts().reset_index()
        seg.columns = ['segment', 'count']
        fig = px.pie(seg, names='segment', values='count', title='Customer Revenue Segments',
                     color_discrete_sequence=['#00D4FF', '#00FF88', '#FFD93D', '#C77DFF'])
        fig.update_layout(**PLOTLY_THEME, title_font_color='#E2E8F0')
        fig.update_traces(textfont_color='#E2E8F0')
        st.plotly_chart(fig, use_container_width=True)

    with c6:
        tenure = filtered.groupby('tenure_bucket')['monthly_revenue'].mean().reset_index()
        bucket_order = ['New', 'Growing', 'Established', 'Veteran']
        tenure['tenure_bucket'] = pd.Categorical(tenure['tenure_bucket'], categories=bucket_order, ordered=True)
        tenure = tenure.sort_values('tenure_bucket')
        fig = px.bar(tenure, x='tenure_bucket', y='monthly_revenue',
                     title='Avg Revenue by Tenure',
                     color='monthly_revenue', color_continuous_scale=[[0, '#1E293B'], [1, '#00FF88']])
        fig.update_layout(**PLOTLY_THEME, coloraxis_showscale=False, title_font_color='#E2E8F0')
        fig.update_traces(marker_line_width=0)
        st.plotly_chart(fig, use_container_width=True)

# ══════════════════════════════════════════════════════════════════
# TAB 2 — CHURN INTELLIGENCE
# ══════════════════════════════════════════════════════════════════
with tab2:
    st.markdown("<div class='section-header'><p class='section-title'>Churn Intelligence</p></div>", unsafe_allow_html=True)

    c1, c2 = st.columns(2)
    with c1:
        churn_plan = filtered.groupby('plan')['churned'].mean().reset_index()
        churn_plan['churn_pct'] = churn_plan['churned'] * 100
        churn_plan['plan'] = pd.Categorical(churn_plan['plan'], categories=plan_order, ordered=True)
        churn_plan = churn_plan.sort_values('plan')
        fig = px.bar(churn_plan, x='plan', y='churn_pct', color='churn_pct',
                     color_continuous_scale=[[0, '#00FF88'], [0.5, '#FFD93D'], [1, '#FF6B6B']],
                     title='Churn Rate by Plan (%)')
        fig.update_layout(**PLOTLY_THEME, coloraxis_showscale=False, title_font_color='#E2E8F0')
        fig.update_traces(marker_line_width=0)
        st.plotly_chart(fig, use_container_width=True)

    with c2:
        churn_region = filtered.groupby('region')['churned'].mean().reset_index()
        churn_region['churn_pct'] = churn_region['churned'] * 100
        fig = px.bar(churn_region.sort_values('churn_pct', ascending=True),
                     x='churn_pct', y='region', orientation='h',
                     title='Churn Rate by Region (%)',
                     color='churn_pct', color_continuous_scale=[[0, '#00FF88'], [1, '#FF6B6B']])
        fig.update_layout(**PLOTLY_THEME, coloraxis_showscale=False, title_font_color='#E2E8F0')
        fig.update_traces(marker_line_width=0)
        st.plotly_chart(fig, use_container_width=True)

    c3, c4 = st.columns(2)
    with c3:
        fig = px.histogram(filtered, x='support_tickets', color='churned',
                           barmode='overlay', title='Support Tickets vs Churn',
                           color_discrete_map={0: '#00FF88', 1: '#FF6B6B'},
                           labels={'churned': 'Churned'})
        fig.update_layout(**PLOTLY_THEME, title_font_color='#E2E8F0')
        st.plotly_chart(fig, use_container_width=True)

    with c4:
        fig = px.histogram(filtered, x='login_frequency', color='churned',
                           barmode='overlay', title='Login Frequency vs Churn',
                           color_discrete_map={0: '#00FF88', 1: '#FF6B6B'},
                           labels={'churned': 'Churned'})
        fig.update_layout(**PLOTLY_THEME, title_font_color='#E2E8F0')
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("<div class='section-header'><p class='section-title'>Risk Analysis</p></div>", unsafe_allow_html=True)
    c5, c6 = st.columns(2)
    with c5:
        fig = px.scatter(filtered, x='engagement_score', y='risk_score',
                         color='churned', size='monthly_revenue',
                         color_discrete_map={0: '#00FF88', 1: '#FF6B6B'},
                         title='Engagement Score vs Risk Score',
                         labels={'churned': 'Churned'},
                         opacity=0.6)
        fig.update_layout(**PLOTLY_THEME, title_font_color='#E2E8F0')
        st.plotly_chart(fig, use_container_width=True)

    with c6:
        churn_tenure = filtered.groupby('tenure_bucket')['churned'].mean().reset_index()
        churn_tenure['churn_pct'] = churn_tenure['churned'] * 100
        churn_tenure['tenure_bucket'] = pd.Categorical(churn_tenure['tenure_bucket'], categories=bucket_order, ordered=True)
        churn_tenure = churn_tenure.sort_values('tenure_bucket')
        fig = px.line(churn_tenure, x='tenure_bucket', y='churn_pct',
                      title='Churn Rate by Tenure Bucket',
                      markers=True, color_discrete_sequence=['#FF6B6B'])
        fig.update_layout(**PLOTLY_THEME, title_font_color='#E2E8F0')
        fig.update_traces(line_width=3, marker_size=10)
        st.plotly_chart(fig, use_container_width=True)

# ══════════════════════════════════════════════════════════════════
# TAB 3 — PROFITABILITY
# ══════════════════════════════════════════════════════════════════
with tab3:
    st.markdown("<div class='section-header'><p class='section-title'>Profitability Intelligence</p></div>", unsafe_allow_html=True)

    c1, c2 = st.columns(2)
    with c1:
        margin_ind = filtered.groupby('industry')['gross_margin_pct'].mean().reset_index().sort_values('gross_margin_pct', ascending=True)
        fig = px.bar(margin_ind, x='gross_margin_pct', y='industry', orientation='h',
                     title='Avg Gross Margin by Industry (%)',
                     color='gross_margin_pct', color_continuous_scale=[[0, '#FF6B6B'], [0.5, '#FFD93D'], [1, '#00FF88']])
        fig.update_layout(**PLOTLY_THEME, coloraxis_showscale=False, title_font_color='#E2E8F0')
        fig.update_traces(marker_line_width=0)
        st.plotly_chart(fig, use_container_width=True)

    with c2:
        margin_plan = filtered.groupby('plan')['gross_margin_pct'].mean().reset_index()
        margin_plan['plan'] = pd.Categorical(margin_plan['plan'], categories=plan_order, ordered=True)
        margin_plan = margin_plan.sort_values('plan')
        fig = px.bar(margin_plan, x='plan', y='gross_margin_pct', color='plan',
                     color_discrete_sequence=['#00D4FF', '#00FF88', '#FFD93D', '#C77DFF'],
                     title='Avg Gross Margin by Plan (%)')
        fig.update_layout(**PLOTLY_THEME, showlegend=False, title_font_color='#E2E8F0')
        fig.update_traces(marker_line_width=0)
        st.plotly_chart(fig, use_container_width=True)

    c3, c4 = st.columns(2)
    with c3:
        fig = px.scatter(filtered, x='monthly_revenue', y='gross_profit',
                         color='profitability_tier', size='ltv',
                         color_discrete_map={'High': '#00FF88', 'Medium': '#FFD93D', 'Low': '#FF6B6B'},
                         title='Revenue vs Gross Profit by Profitability Tier',
                         opacity=0.6)
        fig.update_layout(**PLOTLY_THEME, title_font_color='#E2E8F0')
        st.plotly_chart(fig, use_container_width=True)

    with c4:
        tier_count = filtered['profitability_tier'].value_counts().reset_index()
        tier_count.columns = ['tier', 'count']
        fig = px.pie(tier_count, names='tier', values='count',
                     title='Profitability Tier Distribution',
                     color='tier', color_discrete_map={'High': '#00FF88', 'Medium': '#FFD93D', 'Low': '#FF6B6B'})
        fig.update_layout(**PLOTLY_THEME, title_font_color='#E2E8F0')
        fig.update_traces(textfont_color='#E2E8F0')
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("<div class='section-header'><p class='section-title'>LTV & Cost Efficiency</p></div>", unsafe_allow_html=True)
    c5, c6 = st.columns(2)
    with c5:
        fig = px.box(filtered, x='plan', y='ltv', color='plan',
                     color_discrete_sequence=['#00D4FF', '#00FF88', '#FFD93D', '#C77DFF'],
                     title='Customer LTV Distribution by Plan',
                     category_orders={'plan': plan_order})
        fig.update_layout(**PLOTLY_THEME, showlegend=False, title_font_color='#E2E8F0')
        st.plotly_chart(fig, use_container_width=True)

    with c6:
        fig = px.scatter(filtered, x='cost_efficiency', y='ltv',
                         color='plan', size='monthly_revenue',
                         color_discrete_sequence=['#00D4FF', '#00FF88', '#FFD93D', '#C77DFF'],
                         title='Cost Efficiency vs LTV', opacity=0.6)
        fig.update_layout(**PLOTLY_THEME, title_font_color='#E2E8F0')
        st.plotly_chart(fig, use_container_width=True)

# ══════════════════════════════════════════════════════════════════
# TAB 4 — LIVE PREDICTOR
# ══════════════════════════════════════════════════════════════════
with tab4:
    st.markdown("<div class='section-header'><p class='section-title'>Live AI Predictor</p></div>", unsafe_allow_html=True)
    st.markdown("<p style='color:#64748B;font-size:13px;margin-bottom:20px'>Enter customer details below to get real-time predictions from all 3 models.</p>", unsafe_allow_html=True)

    col_l, col_r = st.columns(2)

    with col_l:
        st.markdown("<p style='color:#00D4FF;font-size:12px;font-weight:600;text-transform:uppercase;letter-spacing:1px'>Customer Profile</p>", unsafe_allow_html=True)
        pred_plan = st.selectbox("Plan", ['Starter', 'Growth', 'Professional', 'Enterprise'])
        pred_industry = st.selectbox("Industry", ['Technology', 'Healthcare', 'Retail', 'Finance', 'Manufacturing', 'Education', 'Logistics'])
        pred_region = st.selectbox("Region", ['North America', 'Europe', 'Asia Pacific', 'Latin America', 'Middle East'])
        pred_payment = st.selectbox("Payment Method", ['Credit Card', 'Bank Transfer', 'Invoice', 'PayPal'])
        pred_tenure = st.slider("Tenure (Months)", 1, 42, 12)
        pred_users = st.slider("Number of Users", 1, 200, 10)

    with col_r:
        st.markdown("<p style='color:#00D4FF;font-size:12px;font-weight:600;text-transform:uppercase;letter-spacing:1px'>Usage & Behaviour</p>", unsafe_allow_html=True)
        pred_tickets = st.slider("Support Tickets", 0, 15, 2)
        pred_logins = st.slider("Login Frequency", 0, 30, 10)
        pred_nps = st.slider("NPS Score", 0, 10, 7)

    if st.button("🚀 Run Predictions", use_container_width=True):
        plan_map = {'Starter': 0, 'Growth': 1, 'Professional': 2, 'Enterprise': 3}
        plan_enc = plan_map[pred_plan]

        base_rev = {'Starter': 199, 'Growth': 599, 'Professional': 1499, 'Enterprise': 4999}
        est_rev = base_rev[pred_plan] * 1.05
        cogs_rate = 0.40
        cost_eff = 1 - cogs_rate
        rev_per_user = est_rev / max(pred_users, 1)
        max_login = df['login_frequency'].max()
        max_users_val = df['num_users'].max()
        engagement = ((pred_logins / max_login * 50) + (pred_users / max_users_val * 30) + ((10 - min(pred_tickets, 10)) / 10 * 20))
        risk = (pred_tickets * 3) + ((10 - min(pred_logins, 10)) * 2) + ((10 - pred_nps) * 2) + (10 if pred_tenure < 3 else 5 if pred_tenure < 6 else 0)
        annual_rev = est_rev * 12

        all_regions = [c.replace('region_', '') for c in df.columns if c.startswith('region_')]
        all_industries = [c.replace('industry_', '') for c in df.columns if c.startswith('industry_')]
        all_payments = [c.replace('payment_', '') for c in df.columns if c.startswith('payment_')]

        input_dict = {
            'plan_encoded': plan_enc,
            'tenure_months': pred_tenure,
            'support_tickets': pred_tickets,
            'login_frequency': pred_logins,
            'num_users': pred_users,
            'nps_score': pred_nps,
            'revenue_per_user': rev_per_user,
            'cost_efficiency': cost_eff,
            'engagement_score': engagement,
            'risk_score': risk
        }
        for r in all_regions:
            input_dict[f'region_{r}'] = 1 if r == pred_region else 0
        for ind in all_industries:
            input_dict[f'industry_{ind}'] = 1 if ind == pred_industry else 0
        for p in all_payments:
            input_dict[f'payment_{p}'] = 1 if p == pred_payment else 0

        input_df = pd.DataFrame([input_dict])

        try:
            rev_features = models['revenue_features']
            rev_input = input_df.reindex(columns=rev_features, fill_value=0)
            predicted_revenue = models['revenue'].predict(rev_input)[0]

            churn_features = models['churn_features']
            churn_input = input_df.reindex(columns=churn_features, fill_value=0)
            churn_prob = models['churn'].predict_proba(churn_input)[0][1] * 100

            profit_features = models['profit_features']
            profit_input = input_df.reindex(columns=profit_features, fill_value=0)
            profit_margin = models['profit'].predict(profit_input)[0]

            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown("<div class='section-header'><p class='section-title'>Prediction Results</p></div>", unsafe_allow_html=True)

            r1, r2, r3 = st.columns(3)
            with r1:
                st.markdown(f"<div class='kpi-card'><div class='kpi-label'>Predicted Revenue</div><div class='kpi-value'>${predicted_revenue:,.0f}</div><div class='kpi-delta'>Monthly MRR</div></div>", unsafe_allow_html=True)
            with r2:
                risk_color = '#FF6B6B' if churn_prob > 60 else '#FFD93D' if churn_prob > 30 else '#00FF88'
                risk_label = 'HIGH RISK' if churn_prob > 60 else 'MEDIUM RISK' if churn_prob > 30 else 'LOW RISK'
                st.markdown(f"<div class='kpi-card'><div class='kpi-label'>Churn Probability</div><div class='kpi-value' style='color:{risk_color}'>{churn_prob:.1f}%</div><div class='kpi-delta' style='color:{risk_color}'>{risk_label}</div></div>", unsafe_allow_html=True)
            with r3:
                st.markdown(f"<div class='kpi-card'><div class='kpi-label'>Predicted Margin</div><div class='kpi-value' style='color:#00FF88'>{profit_margin:.1f}%</div><div class='kpi-delta'>Gross Margin %</div></div>", unsafe_allow_html=True)

            st.markdown("<br>", unsafe_allow_html=True)
            fig = go.Figure(go.Indicator(
                mode="gauge+number",
                value=churn_prob,
                title={'text': "Churn Risk Gauge", 'font': {'color': '#E2E8F0', 'size': 16}},
                number={'suffix': '%', 'font': {'color': '#E2E8F0', 'size': 32}},
                gauge={
                    'axis': {'range': [0, 100], 'tickcolor': '#64748B'},
                    'bar': {'color': risk_color},
                    'bgcolor': '#111827',
                    'bordercolor': '#1E293B',
                    'steps': [
                        {'range': [0, 30], 'color': '#0A2A1A'},
                        {'range': [30, 60], 'color': '#2A2A0A'},
                        {'range': [60, 100], 'color': '#2A0A0A'}
                    ],
                    'threshold': {'line': {'color': 'white', 'width': 2}, 'thickness': 0.75, 'value': churn_prob}
                }
            ))
            fig.update_layout(paper_bgcolor='#0A0E1A', font_color='#E2E8F0', height=300)
            st.plotly_chart(fig, use_container_width=True)

        except Exception as e:
            st.error(f"Prediction error: {e}")