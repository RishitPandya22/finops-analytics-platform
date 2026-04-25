import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

os.makedirs('assets', exist_ok=True)

plt.style.use('dark_background')
COLORS = ['#00D4FF', '#FF6B6B', '#00FF88', '#FFD93D', '#C77DFF', '#FF9F43', '#48DBFB']
BG = '#0A0E1A'
CARD = '#111827'
TEXT = '#E2E8F0'

def style_ax(ax, title):
    ax.set_facecolor(CARD)
    ax.set_title(title, color=TEXT, fontsize=13, fontweight='bold', pad=12)
    ax.tick_params(colors=TEXT)
    ax.xaxis.label.set_color(TEXT)
    ax.yaxis.label.set_color(TEXT)
    for spine in ax.spines.values():
        spine.set_edgecolor('#1E293B')

df = pd.read_csv('data/finops_data.csv')
df['signup_date'] = pd.to_datetime(df['signup_date'])
df['month'] = df['signup_date'].dt.to_period('M')

print("Running EDA... generating all charts")

# ── Chart 1: Revenue Distribution by Plan ──────────────────────────
fig, ax = plt.subplots(figsize=(10, 5), facecolor=BG)
plan_order = ['Starter', 'Growth', 'Professional', 'Enterprise']
for i, plan in enumerate(plan_order):
    data = df[df['plan'] == plan]['monthly_revenue']
    ax.hist(data, bins=30, alpha=0.75, label=plan, color=COLORS[i])
style_ax(ax, 'Revenue Distribution by Plan')
ax.set_xlabel('Monthly Revenue ($)')
ax.set_ylabel('Count')
ax.legend(facecolor=CARD, labelcolor=TEXT)
plt.tight_layout()
plt.savefig('assets/chart1_revenue_distribution.png', dpi=150, bbox_inches='tight', facecolor=BG)
plt.close()
print("Chart 1 done")

# ── Chart 2: Churn Rate by Plan ────────────────────────────────────
fig, ax = plt.subplots(figsize=(8, 5), facecolor=BG)
churn_by_plan = df.groupby('plan')['churned'].mean().reindex(plan_order) * 100
bars = ax.bar(churn_by_plan.index, churn_by_plan.values, color=COLORS[:4], edgecolor='none', width=0.5)
for bar, val in zip(bars, churn_by_plan.values):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
            f'{val:.1f}%', ha='center', va='bottom', color=TEXT, fontsize=11, fontweight='bold')
style_ax(ax, 'Churn Rate by Plan (%)')
ax.set_ylabel('Churn Rate (%)')
ax.set_ylim(0, churn_by_plan.max() + 8)
plt.tight_layout()
plt.savefig('assets/chart2_churn_by_plan.png', dpi=150, bbox_inches='tight', facecolor=BG)
plt.close()
print("Chart 2 done")

# ── Chart 3: Monthly Revenue Trend ────────────────────────────────
fig, ax = plt.subplots(figsize=(12, 5), facecolor=BG)
monthly = df.groupby('month')['monthly_revenue'].sum().reset_index()
monthly['month_str'] = monthly['month'].astype(str)
ax.plot(monthly['month_str'], monthly['monthly_revenue'], color=COLORS[0], linewidth=2.5, marker='o', markersize=4)
ax.fill_between(monthly['month_str'], monthly['monthly_revenue'], alpha=0.15, color=COLORS[0])
style_ax(ax, 'Total Monthly Revenue Trend')
ax.set_xlabel('Month')
ax.set_ylabel('Total Revenue ($)')
tick_step = max(1, len(monthly) // 8)
ax.set_xticks(range(0, len(monthly), tick_step))
ax.set_xticklabels(monthly['month_str'].iloc[::tick_step], rotation=45, ha='right')
plt.tight_layout()
plt.savefig('assets/chart3_revenue_trend.png', dpi=150, bbox_inches='tight', facecolor=BG)
plt.close()
print("Chart 3 done")

# ── Chart 4: Gross Margin by Industry ─────────────────────────────
fig, ax = plt.subplots(figsize=(10, 5), facecolor=BG)
margin_by_industry = df.groupby('industry')['gross_margin_pct'].mean().sort_values(ascending=True)
bars = ax.barh(margin_by_industry.index, margin_by_industry.values, color=COLORS[2], edgecolor='none', height=0.5)
for bar, val in zip(bars, margin_by_industry.values):
    ax.text(val + 0.2, bar.get_y() + bar.get_height()/2,
            f'{val:.1f}%', va='center', color=TEXT, fontsize=10)
style_ax(ax, 'Average Gross Margin by Industry (%)')
ax.set_xlabel('Gross Margin (%)')
plt.tight_layout()
plt.savefig('assets/chart4_margin_by_industry.png', dpi=150, bbox_inches='tight', facecolor=BG)
plt.close()
print("Chart 4 done")

# ── Chart 5: Churn vs Support Tickets ─────────────────────────────
fig, ax = plt.subplots(figsize=(8, 5), facecolor=BG)
churn_tickets = df.groupby('support_tickets')['churned'].mean() * 100
ax.bar(churn_tickets.index[:12], churn_tickets.values[:12], color=COLORS[1], edgecolor='none', width=0.6)
style_ax(ax, 'Churn Rate vs Support Tickets Raised')
ax.set_xlabel('Number of Support Tickets')
ax.set_ylabel('Churn Rate (%)')
plt.tight_layout()
plt.savefig('assets/chart5_churn_vs_tickets.png', dpi=150, bbox_inches='tight', facecolor=BG)
plt.close()
print("Chart 5 done")

# ── Chart 6: LTV by Plan (Boxplot) ────────────────────────────────
fig, ax = plt.subplots(figsize=(10, 5), facecolor=BG)
data_to_plot = [df[df['plan'] == p]['ltv'].values for p in plan_order]
bp = ax.boxplot(data_to_plot, labels=plan_order, patch_artist=True, medianprops=dict(color='white', linewidth=2))
for patch, color in zip(bp['boxes'], COLORS):
    patch.set_facecolor(color)
    patch.set_alpha(0.7)
for element in ['whiskers', 'caps', 'fliers']:
    for item in bp[element]:
        item.set_color('#4A5568')
style_ax(ax, 'Customer Lifetime Value (LTV) by Plan')
ax.set_ylabel('LTV ($)')
plt.tight_layout()
plt.savefig('assets/chart6_ltv_by_plan.png', dpi=150, bbox_inches='tight', facecolor=BG)
plt.close()
print("Chart 6 done")

# ── Chart 7: Correlation Heatmap ──────────────────────────────────
fig, ax = plt.subplots(figsize=(10, 8), facecolor=BG)
num_cols = ['monthly_revenue', 'cogs', 'gross_profit', 'gross_margin_pct',
            'support_tickets', 'login_frequency', 'num_users', 'nps_score',
            'tenure_months', 'ltv', 'churned']
corr = df[num_cols].corr()
mask = np.triu(np.ones_like(corr, dtype=bool))
sns.heatmap(corr, mask=mask, annot=True, fmt='.2f', cmap='coolwarm',
            ax=ax, linewidths=0.5, linecolor='#1E293B',
            annot_kws={'size': 9}, cbar_kws={'shrink': 0.8})
style_ax(ax, 'Feature Correlation Heatmap')
plt.tight_layout()
plt.savefig('assets/chart7_correlation_heatmap.png', dpi=150, bbox_inches='tight', facecolor=BG)
plt.close()
print("Chart 7 done")

# ── Chart 8: Revenue by Region ────────────────────────────────────
fig, ax = plt.subplots(figsize=(10, 5), facecolor=BG)
rev_region = df.groupby('region')['monthly_revenue'].sum().sort_values(ascending=False)
bars = ax.bar(rev_region.index, rev_region.values, color=COLORS[:len(rev_region)], edgecolor='none', width=0.5)
for bar, val in zip(bars, rev_region.values):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 500,
            f'${val/1000:.0f}K', ha='center', va='bottom', color=TEXT, fontsize=10, fontweight='bold')
style_ax(ax, 'Total Revenue by Region')
ax.set_ylabel('Total Revenue ($)')
ax.set_xlabel('Region')
plt.xticks(rotation=15)
plt.tight_layout()
plt.savefig('assets/chart8_revenue_by_region.png', dpi=150, bbox_inches='tight', facecolor=BG)
plt.close()
print("Chart 8 done")

# ── Chart 9: NPS Score Distribution ──────────────────────────────
fig, ax = plt.subplots(figsize=(10, 5), facecolor=BG)
churned_nps = df[df['churned'] == 1]['nps_score']
retained_nps = df[df['churned'] == 0]['nps_score']
ax.hist(retained_nps, bins=10, alpha=0.7, label='Retained', color=COLORS[2])
ax.hist(churned_nps, bins=10, alpha=0.7, label='Churned', color=COLORS[1])
style_ax(ax, 'NPS Score Distribution — Churned vs Retained')
ax.set_xlabel('NPS Score')
ax.set_ylabel('Count')
ax.legend(facecolor=CARD, labelcolor=TEXT)
plt.tight_layout()
plt.savefig('assets/chart9_nps_distribution.png', dpi=150, bbox_inches='tight', facecolor=BG)
plt.close()
print("Chart 9 done")

print("\nAll 9 charts saved to assets/ folder!")
print(f"\nKey Insights:")
print(f"  Avg Monthly Revenue:  ${df['monthly_revenue'].mean():,.2f}")
print(f"  Overall Churn Rate:   {df['churned'].mean():.2%}")
print(f"  Avg Gross Margin:     {df['gross_margin_pct'].mean():.2f}%")
print(f"  Avg LTV:              ${df['ltv'].mean():,.2f}")
print(f"  Avg NPS Score:        {df['nps_score'].mean():.2f}")