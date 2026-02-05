import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import warnings
import os
warnings.filterwarnings('ignore')

# Create output directory
output_dir = 'outputs'
os.makedirs(output_dir, exist_ok=True)

# Set style
sns.set_style("whitegrid")
sns.set_palette("husl")
plt.rcParams['figure.figsize'] = (15, 10)

# Load datasets - UPDATE THESE PATHS TO YOUR LOCAL PATHS
base_path = 'D:/cdac/hackathon/Track1_Analytics_Dataset/'
df1 = pd.read_csv(f'{base_path}annex1.csv')
df2 = pd.read_csv(f'{base_path}annex2.csv')
df3 = pd.read_csv(f'{base_path}annex3.csv')
df4 = pd.read_csv(f'{base_path}annex4.csv')

print("✓ All datasets loaded successfully!")

# Convert dates
df2['Date'] = pd.to_datetime(df2['Date'])
df3['Date'] = pd.to_datetime(df3['Date'])

# Calculate revenue
df2['Revenue'] = df2['Quantity Sold (kilo)'] * df2['Unit Selling Price (RMB/kg)']
df2['YearMonth'] = df2['Date'].dt.to_period('M')
df2['DayOfWeek'] = df2['Date'].dt.day_name()

# Merge datasets
df2_with_cat = df2.merge(df1[['Item Code', 'Category Name', 'Item Name']], on='Item Code', how='left')

print("✓ Data preprocessing complete!")

# Create comprehensive visualizations
fig = plt.figure(figsize=(20, 24))

print("Creating visualizations...")

# 1. Monthly Revenue Trend
ax1 = plt.subplot(5, 2, 1)
monthly_revenue = df2.groupby('YearMonth')['Revenue'].sum()
monthly_revenue.index = monthly_revenue.index.to_timestamp()
ax1.plot(monthly_revenue.index, monthly_revenue.values, marker='o', linewidth=2, markersize=4)
ax1.set_title('Monthly Revenue Trend', fontsize=14, fontweight='bold')
ax1.set_xlabel('Month')
ax1.set_ylabel('Revenue (¥)')
ax1.grid(True, alpha=0.3)
plt.xticks(rotation=45)

# 2. Monthly Transaction Volume
ax2 = plt.subplot(5, 2, 2)
monthly_trans = df2.groupby('YearMonth').size()
monthly_trans.index = monthly_trans.index.to_timestamp()
ax2.plot(monthly_trans.index, monthly_trans.values, marker='o', linewidth=2, markersize=4, color='orange')
ax2.set_title('Monthly Transaction Volume', fontsize=14, fontweight='bold')
ax2.set_xlabel('Month')
ax2.set_ylabel('Number of Transactions')
ax2.grid(True, alpha=0.3)
plt.xticks(rotation=45)

# 3. Sales by Category
ax3 = plt.subplot(5, 2, 3)
category_sales = df2_with_cat.groupby('Category Name')['Revenue'].sum().sort_values(ascending=True)
category_sales.plot(kind='barh', ax=ax3, color='skyblue')
ax3.set_title('Total Revenue by Category', fontsize=14, fontweight='bold')
ax3.set_xlabel('Revenue (¥)')
ax3.set_ylabel('Category')

# 4. Loss Rate Distribution
ax4 = plt.subplot(5, 2, 4)
ax4.hist(df4['Loss Rate (%)'], bins=30, color='coral', edgecolor='black', alpha=0.7)
ax4.axvline(df4['Loss Rate (%)'].mean(), color='red', linestyle='--', linewidth=2, label=f'Mean: {df4["Loss Rate (%)"].mean():.2f}%')
ax4.set_title('Loss Rate Distribution', fontsize=14, fontweight='bold')
ax4.set_xlabel('Loss Rate (%)')
ax4.set_ylabel('Frequency')
ax4.legend()
ax4.grid(True, alpha=0.3)

# 5. Day of Week Analysis
ax5 = plt.subplot(5, 2, 5)
day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
dow_revenue = df2.groupby('DayOfWeek')['Revenue'].sum().reindex(day_order)
dow_revenue.plot(kind='bar', ax=ax5, color='lightgreen', edgecolor='black')
ax5.set_title('Revenue by Day of Week', fontsize=14, fontweight='bold')
ax5.set_xlabel('Day')
ax5.set_ylabel('Revenue (¥)')
ax5.set_xticklabels(day_order, rotation=45)
ax5.grid(True, alpha=0.3)

# 6. Price Distribution
ax6 = plt.subplot(5, 2, 6)
ax6.hist(df2['Unit Selling Price (RMB/kg)'], bins=50, color='purple', alpha=0.7, edgecolor='black')
ax6.axvline(df2['Unit Selling Price (RMB/kg)'].mean(), color='red', linestyle='--', linewidth=2, label=f'Mean: ¥{df2["Unit Selling Price (RMB/kg)"].mean():.2f}')
ax6.set_title('Selling Price Distribution', fontsize=14, fontweight='bold')
ax6.set_xlabel('Price (¥/kg)')
ax6.set_ylabel('Frequency')
ax6.legend()
ax6.grid(True, alpha=0.3)

# 7. Top 10 Products by Revenue
ax7 = plt.subplot(5, 2, 7)
top_products = df2.groupby('Item Code')['Revenue'].sum().nlargest(10).sort_values(ascending=True)
top_products.plot(kind='barh', ax=ax7, color='gold')
ax7.set_title('Top 10 Products by Revenue', fontsize=14, fontweight='bold')
ax7.set_xlabel('Revenue (¥)')
ax7.set_ylabel('Item Code')

# 8. Quantity Sold Distribution
ax8 = plt.subplot(5, 2, 8)
qty_clean = df2[df2['Quantity Sold (kilo)'] > 0]['Quantity Sold (kilo)']
ax8.hist(qty_clean, bins=50, color='teal', alpha=0.7, edgecolor='black')
ax8.axvline(qty_clean.mean(), color='red', linestyle='--', linewidth=2, label=f'Mean: {qty_clean.mean():.2f} kg')
ax8.set_title('Quantity Sold Distribution (Positive Values)', fontsize=14, fontweight='bold')
ax8.set_xlabel('Quantity (kg)')
ax8.set_ylabel('Frequency')
ax8.legend()
ax8.grid(True, alpha=0.3)

# 9. Wholesale vs Selling Price
ax9 = plt.subplot(5, 2, 9)
# Sample data for visualization (merge takes too long, so aggregate first)
avg_wholesale = df3.groupby('Item Code')['Wholesale Price (RMB/kg)'].mean()
avg_selling = df2.groupby('Item Code')['Unit Selling Price (RMB/kg)'].mean()
comparison_df = pd.DataFrame({
    'Wholesale': avg_wholesale,
    'Selling': avg_selling
}).dropna()

x_pos = np.arange(min(20, len(comparison_df)))
width = 0.35
sample_items = comparison_df.head(20)

ax9.bar(x_pos - width/2, sample_items['Wholesale'], width, label='Wholesale Price', color='lightblue')
ax9.bar(x_pos + width/2, sample_items['Selling'], width, label='Selling Price', color='salmon')
ax9.set_title('Wholesale vs Selling Price (Sample 20 Items)', fontsize=14, fontweight='bold')
ax9.set_xlabel('Item Index')
ax9.set_ylabel('Price (¥/kg)')
ax9.legend()
ax9.grid(True, alpha=0.3)

# 10. Discount Analysis
ax10 = plt.subplot(5, 2, 10)
discount_data = df2.groupby('Discount (Yes/No)').agg({
    'Revenue': 'sum',
    'Item Code': 'count'
})
colors_discount = ['lightcoral', 'lightgreen']
discount_data['Item Code'].plot(kind='pie', ax=ax10, autopct='%1.1f%%', colors=colors_discount, startangle=90)
ax10.set_title('Transaction Distribution: Discount vs No Discount', fontsize=14, fontweight='bold')
ax10.set_ylabel('')

plt.tight_layout()
plt.savefig(f'{output_dir}/comprehensive_eda_visualizations.png', dpi=300, bbox_inches='tight')
print("✓ Saved: comprehensive_eda_visualizations.png")

# Create second figure for additional insights
fig2 = plt.figure(figsize=(20, 12))

# 1. Category Distribution (Pie Chart)
ax1 = plt.subplot(2, 3, 1)
category_dist = df1['Category Name'].value_counts()
ax1.pie(category_dist.values, labels=category_dist.index, autopct='%1.1f%%', startangle=90)
ax1.set_title('Product Category Distribution', fontsize=14, fontweight='bold')

# 2. Yearly Revenue Comparison
ax2 = plt.subplot(2, 3, 2)
yearly_revenue = df2.groupby(df2['Date'].dt.year)['Revenue'].sum()
yearly_revenue.plot(kind='bar', ax=ax2, color=['#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A'])
ax2.set_title('Yearly Revenue Comparison', fontsize=14, fontweight='bold')
ax2.set_xlabel('Year')
ax2.set_ylabel('Revenue (¥)')
ax2.set_xticklabels(yearly_revenue.index, rotation=0)
ax2.grid(True, alpha=0.3)

# 3. Loss Rate by Category
ax3 = plt.subplot(2, 3, 3)
df_loss_cat = df4.merge(df1[['Item Code', 'Category Name']], on='Item Code', how='left')
loss_by_cat = df_loss_cat.groupby('Category Name')['Loss Rate (%)'].mean().sort_values(ascending=True)
loss_by_cat.plot(kind='barh', ax=ax3, color='indianred')
ax3.set_title('Average Loss Rate by Category', fontsize=14, fontweight='bold')
ax3.set_xlabel('Loss Rate (%)')
ax3.set_ylabel('Category')

# 4. Transaction Volume by Hour
ax4 = plt.subplot(2, 3, 4)
df2['Hour'] = pd.to_datetime(df2['Time']).dt.hour
hourly_trans = df2.groupby('Hour').size()
ax4.plot(hourly_trans.index, hourly_trans.values, marker='o', linewidth=2, markersize=6, color='green')
ax4.set_title('Transaction Volume by Hour of Day', fontsize=14, fontweight='bold')
ax4.set_xlabel('Hour')
ax4.set_ylabel('Number of Transactions')
ax4.grid(True, alpha=0.3)
ax4.set_xticks(range(0, 24, 2))

# 5. Revenue vs Quantity Scatter
ax5 = plt.subplot(2, 3, 5)
sample_data = df2.sample(min(5000, len(df2)))
ax5.scatter(sample_data['Quantity Sold (kilo)'], sample_data['Revenue'], alpha=0.3, s=10)
ax5.set_title('Revenue vs Quantity Sold (Sample)', fontsize=14, fontweight='bold')
ax5.set_xlabel('Quantity Sold (kg)')
ax5.set_ylabel('Revenue (¥)')
ax5.grid(True, alpha=0.3)

# 6. Top Loss Items
ax6 = plt.subplot(2, 3, 6)
top_loss = df4.nlargest(15, 'Loss Rate (%)')
ax6.barh(range(len(top_loss)), top_loss['Loss Rate (%)'].values, color='darkred', alpha=0.7)
ax6.set_yticks(range(len(top_loss)))
ax6.set_yticklabels(top_loss['Item Name'].values, fontsize=8)
ax6.set_title('Top 15 Items with Highest Loss Rate', fontsize=14, fontweight='bold')
ax6.set_xlabel('Loss Rate (%)')
ax6.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(f'{output_dir}/additional_eda_insights.png', dpi=300, bbox_inches='tight')
print("✓ Saved: additional_eda_insights.png")

# Create correlation heatmap
fig3, ax = plt.subplots(figsize=(10, 8))
numerical_data = df2[['Quantity Sold (kilo)', 'Unit Selling Price (RMB/kg)', 'Revenue']].copy()
numerical_data['Hour'] = df2['Hour']
numerical_data['DayOfWeek_Num'] = df2['Date'].dt.dayofweek

corr_matrix = numerical_data.corr()
sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm', center=0, ax=ax, 
            square=True, linewidths=1, cbar_kws={"shrink": 0.8})
ax.set_title('Correlation Matrix - Sales Metrics', fontsize=16, fontweight='bold', pad=20)
plt.tight_layout()
plt.savefig(f'{output_dir}/correlation_heatmap.png', dpi=300, bbox_inches='tight')
print("✓ Saved: correlation_heatmap.png")

print("\n" + "="*80)
print("ALL VISUALIZATIONS CREATED SUCCESSFULLY!")
print("="*80)
print(f"\nGenerated Files (saved in '{output_dir}/' folder):")
print("1. comprehensive_eda_visualizations.png - Main dashboard with 10 key charts")
print("2. additional_eda_insights.png - Additional insights and analysis")
print("3. correlation_heatmap.png - Correlation analysis")
print("\n" + "="*80)