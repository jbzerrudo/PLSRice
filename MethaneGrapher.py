import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
from scipy import stats
import seaborn as sns

# Set style for publication quality
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams.update({
    'font.size': 9,
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial', 'DejaVu Sans'],
    'axes.linewidth': 1.2,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight'
})

# Read and prepare data
df = pd.read_csv(r"G:\2025\PLS2018UPLB\ANALYSIS\METADATA\NEWDATA\UYData_new.csv", encoding='cp1252')
df['Date'] = pd.to_datetime(df['Date'])
df = df.sort_values('Date').reset_index(drop=True)

# Calculate rolling averages
df['CH4_7day'] = df['CH4 mg/m^2/h'].rolling(window=7, center=True).mean()
df['CH4_30day'] = df['CH4 mg/m^2/h'].rolling(window=30, center=True).mean()

# Outlier detection
Q1 = df['CH4 mg/m^2/h'].quantile(0.25)
Q3 = df['CH4 mg/m^2/h'].quantile(0.75)
IQR = Q3 - Q1
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR
df_clean = df[(df['CH4 mg/m^2/h'] >= lower_bound) & (df['CH4 mg/m^2/h'] <= upper_bound)]

# Add seasonal information
df['Month'] = df['Date'].dt.month
df['Season'] = df['Month'].map({12: 'Winter', 1: 'Winter', 2: 'Winter',
                                3: 'Spring', 4: 'Spring', 5: 'Spring',
                                6: 'Summer', 7: 'Summer', 8: 'Summer',
                                9: 'Autumn', 10: 'Autumn', 11: 'Autumn'})

# Create subplot figure
fig = plt.figure(figsize=(12, 10))

# Create a 2x2 subplot layout
gs = fig.add_gridspec(3, 2, height_ratios=[2, 1, 1], hspace=0.3, wspace=0.25)

# Main time series plot
ax1 = fig.add_subplot(gs[0, :])
ax1.plot(df['Date'], df['CH4 mg/m^2/h'], color='lightgray', linewidth=0.3, alpha=0.5, label='Raw data')
ax1.plot(df['Date'], df['CH4_7day'], color='#3498DB', linewidth=1.2, label='7-day average')
ax1.plot(df['Date'], df['CH4_30day'], color='#E74C3C', linewidth=1.8, label='30-day average')
ax1.axhline(y=0, color='black', linestyle='--', linewidth=0.8, alpha=0.5)

ax1.set_ylabel('CH₄ flux (mg m⁻² h⁻¹)', fontweight='bold')
ax1.set_title('Methane Flux Time Series Analysis', fontsize=14, fontweight='bold', pad=20)
ax1.legend(loc='upper right')
ax1.grid(True, alpha=0.3)

# Format dates
ax1.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
ax1.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45, ha='right')

# Histogram
ax2 = fig.add_subplot(gs[1, 0])
ax2.hist(df_clean['CH4 mg/m^2/h'], bins=50, color='skyblue', alpha=0.7, edgecolor='black', linewidth=0.5)
ax2.axvline(df['CH4 mg/m^2/h'].mean(), color='red', linestyle='--', linewidth=2, label=f'Mean: {df["CH4 mg/m^2/h"].mean():.2f}')
ax2.axvline(df['CH4 mg/m^2/h'].median(), color='orange', linestyle='--', linewidth=2, label=f'Median: {df["CH4 mg/m^2/h"].median():.2f}')
ax2.set_xlabel('CH₄ flux (mg m⁻² h⁻¹)', fontweight='bold')
ax2.set_ylabel('Frequency', fontweight='bold')
ax2.set_title('Distribution (outliers removed)', fontweight='bold')
ax2.legend(fontsize=8)
ax2.grid(True, alpha=0.3)

# Box plot by season
ax3 = fig.add_subplot(gs[1, 1])
seasons_order = ['Winter', 'Spring', 'Summer', 'Autumn']
season_data = [df[df['Season'] == season]['CH4 mg/m^2/h'].dropna() for season in seasons_order]
bp = ax3.boxplot(season_data, labels=seasons_order, patch_artist=True)
colors = ['lightblue', 'lightgreen', 'lightyellow', 'lightcoral']
for patch, color in zip(bp['boxes'], colors):
    patch.set_facecolor(color)
ax3.set_ylabel('CH₄ flux (mg m⁻² h⁻¹)', fontweight='bold')
ax3.set_title('Seasonal Variation', fontweight='bold')
ax3.grid(True, alpha=0.3)
plt.setp(ax3.xaxis.get_majorticklabels(), rotation=45)

# Monthly averages
ax4 = fig.add_subplot(gs[2, :])
monthly_stats = df.groupby('Month')['CH4 mg/m^2/h'].agg(['mean', 'std', 'count'])
months_with_data = monthly_stats.index
means = monthly_stats['mean']
stds = monthly_stats['std']

ax4.errorbar(months_with_data, means, yerr=stds, fmt='o-', capsize=5, 
             color='navy', markerfacecolor='lightblue', markersize=8, linewidth=2)
ax4.set_xlabel('Month', fontweight='bold')
ax4.set_ylabel('CH₄ flux (mg m⁻² h⁻¹)', fontweight='bold')
ax4.set_title('Monthly Averages (±1 SD)', fontweight='bold')
ax4.grid(True, alpha=0.3)
ax4.set_xlim(0.5, 12.5)
ax4.set_xticks(range(1, 13))
ax4.set_xticklabels(['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                     'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'])

# Add statistics text box
stats_text = (f'Dataset Summary:\n'
              f'• Total measurements: {len(df)}\n'
              f'• Date range: {df["Date"].min().strftime("%Y-%m-%d")} to {df["Date"].max().strftime("%Y-%m-%d")}\n'
              f'• Mean flux: {df["CH4 mg/m^2/h"].mean():.2f} ± {df["CH4 mg/m^2/h"].std():.2f} mg m⁻² h⁻¹\n'
              f'• Outliers: {len(df) - len(df_clean)} ({(len(df) - len(df_clean))/len(df)*100:.1f}%)')

ax1.text(0.02, 0.02, stats_text, transform=ax1.transAxes, fontsize=8,
         verticalalignment='bottom', bbox=dict(boxstyle='round,pad=0.5', 
         facecolor='white', alpha=0.9, edgecolor='gray'))

plt.tight_layout()
plt.savefig('methane_comprehensive_analysis.pdf', dpi=300, bbox_inches='tight')
plt.savefig('methane_comprehensive_analysis.png', dpi=300, bbox_inches='tight')
plt.show()

# Print comprehensive statistics
print("COMPREHENSIVE METHANE FLUX ANALYSIS")
print("=" * 50)
print(f"Data Period: {df['Date'].min().strftime('%Y-%m-%d')} to {df['Date'].max().strftime('%Y-%m-%d')}")
print(f"Total Duration: {(df['Date'].max() - df['Date'].min()).days} days")
print(f"Measurement Frequency: ~{len(df)/(df['Date'].max() - df['Date'].min()).days:.1f} measurements/day")

print(f"\nDescriptive Statistics:")
print(f"Mean: {df['CH4 mg/m^2/h'].mean():.3f} mg m⁻² h⁻¹")
print(f"Median: {df['CH4 mg/m^2/h'].median():.3f} mg m⁻² h⁻¹")
print(f"Standard Deviation: {df['CH4 mg/m^2/h'].std():.3f} mg m⁻² h⁻¹")
print(f"Skewness: {stats.skew(df['CH4 mg/m^2/h']):.3f}")
print(f"Kurtosis: {stats.kurtosis(df['CH4 mg/m^2/h']):.3f}")

print(f"\nSeasonal Averages:")
for season in seasons_order:
    season_data = df[df['Season'] == season]['CH4 mg/m^2/h']
    if len(season_data) > 0:
        print(f"{season}: {season_data.mean():.3f} ± {season_data.std():.3f} mg m⁻² h⁻¹ (n={len(season_data)})")

# Test for seasonal differences
season_groups = [df[df['Season'] == season]['CH4 mg/m^2/h'].dropna() for season in seasons_order if len(df[df['Season'] == season]) > 0]
if len(season_groups) > 1:
    f_stat, p_value = stats.f_oneway(*season_groups)
    print(f"\nOne-way ANOVA (seasonal differences):")
    print(f"F-statistic: {f_stat:.3f}, p-value: {p_value:.6f}")
    if p_value < 0.05:
        print("Significant seasonal differences detected (p < 0.05)")
    else:
        print("No significant seasonal differences (p ≥ 0.05)")