import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import matplotlib.dates as mdates
from scipy import stats

# Set style for publication-quality plots
plt.style.use('default')
sns.set_palette("husl")
plt.rcParams.update({
    'font.size': 12,
    'axes.labelsize': 14,
    'axes.titlesize': 16,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
    'legend.fontsize': 12,
    'figure.titlesize': 18,
    'lines.linewidth': 2,
    'axes.linewidth': 1.2
})

def load_and_process_data(file_path):
    """
    Load and process the dataset for diel analysis
    """
    # Load the data
    df = pd.read_csv(file_path, encoding='latin1')
    
    # Convert Date column to datetime
    df['DateTime'] = pd.to_datetime(df['Date'])
    
    # Extract hour from datetime for diel analysis
    df['Hour'] = df['DateTime'].dt.hour + df['DateTime'].dt.minute/60.0
    
    # Extract day of year for filtering
    df['Day'] = df['DateTime'].dt.dayofyear
    
    # Calculate Tf (floodwater temperature) if not directly available
    # From the interaction term del_t*Tf, assuming del_t is mostly 0.5 hours
    if 'Tf' not in df.columns and 'del_t*Tf' in df.columns:
        df['Tf'] = df['del_t*Tf'] / df['del_t']
        df['Tf'] = df['Tf'].fillna(df['Ta'])  # Fill NaN with air temperature as approximation
    elif 'Tf' not in df.columns:
        df['Tf'] = df['Ta']  # Use air temperature as proxy if Tf unavailable
    
    return df

def create_vegetative_stage_subset(df, n_obs=252):
    """
    Extract the vegetative stage data (first 252 observations as mentioned in paper)
    """
    return df.head(n_obs).copy()

def calculate_diel_averages(df):
    """
    Calculate hourly averages for diel patterns
    """
    # Group by hour and calculate means
    diel_stats = df.groupby('Hour').agg({
        'h': ['mean', 'std'],
        'Ta': ['mean', 'std'],
        'Ts': ['mean', 'std'], 
        'Tf': ['mean', 'std'],
        'SR': ['mean', 'std'],
        'CH4 mg/m^2/h': ['mean', 'std'],
        'WS*RH': ['mean', 'std']
    }).reset_index()
    
    # Flatten column names
    diel_stats.columns = ['_'.join(col).strip() if col[1] else col[0] for col in diel_stats.columns]
    diel_stats = diel_stats.rename(columns={'Hour_': 'Hour'})
    
    return diel_stats

def plot_water_level_diel(df_veg, diel_stats, save_path=None):
    """
    Create water level diel pattern plot
    """
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Plot individual data points (semi-transparent)
    scatter = ax.scatter(df_veg['Hour'], df_veg['h'], 
                        alpha=0.3, s=20, c=df_veg['CH4 mg/m^2/h'], 
                        cmap='viridis', label='Individual observations')
    
    # Plot hourly averages with error bars
    ax.errorbar(diel_stats['Hour'], diel_stats['h_mean'], 
               yerr=diel_stats['h_std'], 
               color='red', linewidth=3, capsize=5, capthick=2,
               label='Hourly averages ± SD')
    
    # Add horizontal reference lines
    ax.axhline(y=0, color='black', linestyle='-', alpha=0.5, label='Soil surface')
    ax.axhline(y=-15, color='orange', linestyle='--', alpha=0.7, label='AWD safe limit (-15 cm)')
    
    # Formatting
    ax.set_xlabel('Hour of Day')
    ax.set_ylabel('Water Depth (cm from soil surface)')
    ax.set_title('Diel Water Level Fluctuations - Vegetative Stage\n(First 252 observations)', fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(loc='upper right')
    
    # Set x-axis to show all 24 hours
    ax.set_xlim(0, 24)
    ax.set_xticks(range(0, 25, 3))
    
    # Add colorbar for methane flux
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label('CH₄ Flux (mg m⁻² h⁻¹)')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()
    return fig

def plot_methane_temperature_diel(df_veg, diel_stats, save_path=None):
    """
    Create methane flux and temperature diel pattern plot
    """
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), sharex=True)
    
    # Top plot: Methane flux
    ax1.errorbar(diel_stats['Hour'], diel_stats['CH4 mg/m^2/h_mean'], 
                yerr=diel_stats['CH4 mg/m^2/h_std'],
                color='darkgreen', linewidth=3, capsize=5, capthick=2,
                label='CH₄ flux', marker='o', markersize=6)
    
    ax1.set_ylabel('CH₄ Flux (mg m⁻² h⁻¹)', color='darkgreen', fontweight='bold')
    ax1.tick_params(axis='y', labelcolor='darkgreen')
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc='upper left')
    ax1.set_title('Methane Flux and Temperature Diel Cycles - Vegetative Stage', fontweight='bold')
    
    # Bottom plot: Temperatures
    colors = {'Ta': 'blue', 'Ts': 'brown', 'Tf': 'cyan'}
    temp_labels = {'Ta': 'Air Temperature', 'Ts': 'Soil Temperature', 'Tf': 'Floodwater Temperature'}
    
    for temp_var, color in colors.items():
        ax2.errorbar(diel_stats['Hour'], diel_stats[f'{temp_var}_mean'], 
                    yerr=diel_stats[f'{temp_var}_std'],
                    color=color, linewidth=2.5, capsize=4, capthick=1.5,
                    label=temp_labels[temp_var], marker='s', markersize=5)
    
    ax2.set_xlabel('Hour of Day')
    ax2.set_ylabel('Temperature (°C)', fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.legend(loc='upper left')
    
    # Set x-axis formatting
    ax2.set_xlim(0, 24)
    ax2.set_xticks(range(0, 25, 3))
    
    # Add shaded region for optimal irrigation window (post-15:00)
    ax1.axvspan(15, 24, alpha=0.2, color='orange', label='Optimal irrigation window')
    ax2.axvspan(15, 24, alpha=0.2, color='orange', label='Optimal irrigation window')
    
    # Add peak emission time annotation
    peak_hour = diel_stats.loc[diel_stats['CH4 mg/m^2/h_mean'].idxmax(), 'Hour']
    peak_flux = diel_stats['CH4 mg/m^2/h_mean'].max()
    ax1.annotate(f'Peak emissions\n{peak_hour:.1f}:00', 
                xy=(peak_hour, peak_flux), xytext=(peak_hour+2, peak_flux+1),
                arrowprops=dict(arrowstyle='->', color='red'),
                fontsize=10, ha='center')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()
    return fig

def plot_comprehensive_diel(df_veg, diel_stats, save_path=None):
    """
    Create comprehensive diel analysis plot
    """
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(16, 12), sharex=True)
    
    # Top plot: Water levels with irrigation events
    ax1.plot(diel_stats['Hour'], diel_stats['h_mean'], 
             color='blue', linewidth=3, label='Water level')
    #ax1.fill_between(diel_stats['Hour'], 
    #                 diel_stats['h_mean'] - diel_stats['h_std'],
    #                 diel_stats['h_mean'] + diel_stats['h_std'],
    #                 alpha=0.3, color='blue')
    
    ax1.axhline(y=0, color='black', linestyle='-', alpha=0.7)
    ax1.axhline(y=-15, color='red', linestyle='--', alpha=0.7)
    ax1.set_ylabel('Water Depth (cm)', fontweight='bold')
    ax1.set_title('Comprehensive Diel Analysis - Vegetative Stage', fontweight='bold', fontsize=16)
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    # Middle plot: Methane flux
    ax2.plot(diel_stats['Hour'], diel_stats['CH4 mg/m^2/h_mean'], 
             color='darkgreen', linewidth=3, label='CH₄ flux', marker='o')
    #ax2.fill_between(diel_stats['Hour'], 
    #                 diel_stats['CH4 mg/m^2/h_mean'] - diel_stats['CH4 mg/m^2/h_std'],
    #                 diel_stats['CH4 mg/m^2/h_mean'] + diel_stats['CH4 mg/m^2/h_std'],
    #                 alpha=0.3, color='darkgreen')
    
    ax2.set_ylabel('CH₄ Flux (mg m⁻² h⁻¹)', fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    
    # Bottom plot: All temperatures
    temp_vars = ['Ta', 'Ts', 'Tf']
    colors = ['blue', 'brown', 'cyan']
    labels = ['Air (Ta)', 'Soil (Ts)', 'Floodwater (Tf)']
    
    for temp_var, color, label in zip(temp_vars, colors, labels):
        ax3.plot(diel_stats['Hour'], diel_stats[f'{temp_var}_mean'], 
                color=color, linewidth=2.5, label=label, marker='s')
    
    ax3.set_xlabel('Hour of Day', fontweight='bold')
    ax3.set_ylabel('Temperature (°C)', fontweight='bold')
    ax3.grid(True, alpha=0.3)
    ax3.legend()
    
    # Add shaded regions for optimal irrigation windows
    for ax in [ax1, ax2, ax3]:
        ax.axvspan(15, 24, alpha=0.15, color='orange', label='Optimal irrigation window' if ax == ax1 else '')
        ax.axvspan(0, 6, alpha=0.15, color='lightblue', label='Night period' if ax == ax1 else '')
    
    # Set x-axis formatting
    ax3.set_xlim(0, 24)
    ax3.set_xticks(range(0, 25, 3))
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()
    return fig

def plot_methane_water_unified(df_veg, diel_stats, save_path=None):
    """
    Create unified plot: Methane flux + Water levels only
    """
    fig, ax1 = plt.subplots(figsize=(14, 8))
    
    # Primary axis: Methane flux
    color_ch4 = 'darkgreen'
    ax1.set_xlabel('Hour of Day', fontweight='bold', fontsize=14)
    ax1.set_ylabel('CH₄ Flux (mg m⁻² h⁻¹)', color=color_ch4, fontweight='bold', fontsize=14)
    
    # Plot methane with error bars and fill
    line1 = ax1.plot(diel_stats['Hour'], diel_stats['CH4 mg/m^2/h_mean'], 
                     color=color_ch4, linewidth=4, marker='o', markersize=8)
    #ax1.fill_between(diel_stats['Hour'], 
    #                 diel_stats['CH4 mg/m^2/h_mean'] - diel_stats['CH4 mg/m^2/h_std'],
    #                 diel_stats['CH4 mg/m^2/h_mean'] + diel_stats['CH4 mg/m^2/h_std'],
    #                 alpha=0.3, color=color_ch4, label='CH₄ ± SD')
    ax1.tick_params(axis='y', labelcolor=color_ch4)
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(0, 25)
    
    # Secondary axis: Water depth
    ax2 = ax1.twinx()
    color_water = 'blue'
    ax2.set_ylabel('Water Depth (cm from soil surface)', color=color_water, fontweight='bold', fontsize=14)
    
    # Plot water level with error bars and fill
    line2 = ax2.plot(diel_stats['Hour'], diel_stats['h_mean'], 
                     color=color_water, linewidth=4, marker='s', markersize=6)
    #ax2.fill_between(diel_stats['Hour'], 
    #                 diel_stats['h_mean'] - diel_stats['h_std'],
    #                 diel_stats['h_mean'] + diel_stats['h_std'],
    #                 alpha=0.3, color=color_water, label='Water level ± SD')
    ax2.tick_params(axis='y', labelcolor=color_water)
    ax2.set_ylim(0, 10) 
    
    # Add reference lines for water depth
    ax2.axhline(y=0, color='black', linestyle='-', alpha=0.7, linewidth=2)
    ax2.axhline(y=-15, color='red', linestyle='--', alpha=0.8, linewidth=2)
    
    # Title and formatting
    ax1.set_title('Diel Pattern: Methane Flux vs. Water Level\nVegetative Stage (First 252 observations)', 
                  fontweight='bold', fontsize=16)
    
    # Add shaded regions for irrigation timing
    #ax1.axvspan(15, 24, alpha=0.15, color='orange', label='Optimal irrigation (15:00-24:00)')
    #ax1.axvspan(0, 6, alpha=0.15, color='lightblue', label='Night period (00:00-06:00)')
    
    # Peak annotations
    #peak_ch4_hour = diel_stats.loc[diel_stats['CH4 mg/m^2/h_mean'].idxmax(), 'Hour']
    #peak_ch4_flux = diel_stats['CH4 mg/m^2/h_mean'].max()
    #ax1.annotate(f'Peak CH₄\n{peak_ch4_hour:.1f}:00', 
    #            xy=(peak_ch4_hour, peak_ch4_flux), xytext=(peak_ch4_hour+2, peak_ch4_flux+0.5),
    #            arrowprops=dict(arrowstyle='->', color='darkred', lw=2),
    #            fontsize=11, ha='center', 
    #            bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.8))
    
    # Combined legend
    #lines1, labels1 = ax1.get_legend_handles_labels()
    #lines2, labels2 = ax2.get_legend_handles_labels()
    #ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left', fontsize=11)
    
    # Set x-axis formatting
    ax1.set_xlim(0, 24)
    ax1.set_xticks(range(0, 25, 3))
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()
    return fig

def plot_methane_temperature_unified(df_veg, diel_stats, save_path=None):
    """
    Create unified plot: Methane flux + All temperatures only
    """
    fig, ax1 = plt.subplots(figsize=(14, 8))
    
    # Primary axis: Methane flux
    color_ch4 = 'darkgreen'
    ax1.set_xlabel('Hour of Day', fontweight='bold', fontsize=14)
    ax1.set_ylabel('CH₄ Flux (mg m⁻² h⁻¹)', color=color_ch4, fontweight='bold', fontsize=14)
    
    # Plot methane with error bars and fill
    line1 = ax1.plot(diel_stats['Hour'], diel_stats['CH4 mg/m^2/h_mean'], 
                     color=color_ch4, linewidth=4, label='CH₄ flux', marker='o', markersize=8)
    #ax1.fill_between(diel_stats['Hour'], 
    #                 diel_stats['CH4 mg/m^2/h_mean'] - diel_stats['CH4 mg/m^2/h_std'],
    #                 diel_stats['CH4 mg/m^2/h_mean'] + diel_stats['CH4 mg/m^2/h_std'],
    #                 alpha=0.3, color=color_ch4, label='CH₄ ± SD')
    ax1.tick_params(axis='y', labelcolor=color_ch4)
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(0, 25)
    
    # Secondary axis: All temperatures
    ax2 = ax1.twinx()
    ax2.set_ylabel('Temperature (°C)', fontweight='bold', fontsize=14)
    
    # Temperature variables with distinct colors and styles
    temp_config = {
        'Ta': {'color': 'red', 'label': 'Air Temperature (Ta)', 'marker': 's', 'linestyle': '-'},
        'Ts': {'color': 'brown', 'label': 'Soil Temperature (Ts)', 'marker': '^', 'linestyle': '-'},
        'Tf': {'color': 'orange', 'label': 'Floodwater Temperature (Tf)', 'marker': 'D', 'linestyle': '-'}
    }
    
    temp_lines = []
    for temp_var, config in temp_config.items():
        if f'{temp_var}_mean' in diel_stats.columns:
            line = ax2.plot(diel_stats['Hour'], diel_stats[f'{temp_var}_mean'], 
                           color=config['color'], linewidth=3, label=config['label'], 
                           marker=config['marker'], markersize=6, linestyle=config['linestyle'])
            
            # Add subtle fill for temperature error
            #ax2.fill_between(diel_stats['Hour'], 
            #               diel_stats[f'{temp_var}_mean'] - diel_stats[f'{temp_var}_std'],
            #               diel_stats[f'{temp_var}_mean'] + diel_stats[f'{temp_var}_std'],
            #               alpha=0.15, color=config['color'])
            temp_lines.extend(line)
    
    ax2.tick_params(axis='y')
    ax2.set_ylim(15, 40) 
    
    # Title and formatting
    ax1.set_title('Diel Pattern: Methane Flux vs. Temperatures\nVegetative Stage (First 252 observations)', 
                  fontweight='bold', fontsize=16)
    
    # Add shaded regions for thermal analysis
    #ax1.axvspan(12, 16, alpha=0.2, color='red', label='Peak heating period (12:00-16:00)')
    #ax1.axvspan(18, 6, alpha=0.15, color='lightblue', label='Cooling period (18:00-06:00)')
    
    # Peak annotations for both CH4 and highest temperature
    #peak_ch4_hour = diel_stats.loc[diel_stats['CH4 mg/m^2/h_mean'].idxmax(), 'Hour']
    #peak_ch4_flux = diel_stats['CH4 mg/m^2/h_mean'].max()
    #ax1.annotate(f'Peak CH₄\n{peak_ch4_hour:.1f}:00', 
    #            xy=(peak_ch4_hour, peak_ch4_flux), xytext=(peak_ch4_hour+1.5, peak_ch4_flux+0.5),
    #            arrowprops=dict(arrowstyle='->', color='darkred', lw=2),
    #            fontsize=11, ha='center', 
    #            bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.8))
    
    # Find peak temperature time (using soil temperature as representative)
    #if 'Ts_mean' in diel_stats.columns:
    #    peak_temp_hour = diel_stats.loc[diel_stats['Ts_mean'].idxmax(), 'Hour']
    #    peak_temp = diel_stats['Ts_mean'].max()
    #    ax2.annotate(f'Peak Ts\n{peak_temp_hour:.1f}:00', 
    #                xy=(peak_temp_hour, peak_temp), xytext=(peak_temp_hour-2, peak_temp+1),
    #                arrowprops=dict(arrowstyle='->', color='brown', lw=2),
    #                fontsize=11, ha='center', 
    #                bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow", alpha=0.8))
    
    # Combined legend
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left', fontsize=10)
    
    # Set x-axis formatting
    ax1.set_xlim(0, 24)
    ax1.set_xticks(range(0, 25, 3))
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()
    return fig

def plot_correlation_heatmap(df_veg, save_path=None):
    """
    Create correlation heatmap for key variables during vegetative stage
    """
    # Select key variables for correlation analysis
    key_vars = ['h', 'Ta', 'Ts', 'Tf', 'SR', 'CH4 mg/m^2/h', 'WS*RH']
    
    # Calculate correlation matrix
    corr_matrix = df_veg[key_vars].corr()
    
    # Create heatmap
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(corr_matrix, annot=True, cmap='RdBu_r', center=0,
                square=True, ax=ax, cbar_kws={'label': 'Pearson Correlation'})
    
    ax.set_title('Variable Correlations - Vegetative Stage', fontweight='bold', fontsize=14)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()
    return fig

def main(file_path, plot_type='unified'):
    """
    Main function to run all diel analyses
    
    Parameters:
    plot_type: 'unified', 'separate', 'all'
    """
    print("Loading and processing data...")
    df = load_and_process_data(file_path)
    
    print(f"Total observations: {len(df)}")
    print(f"Date range: {df['DateTime'].min()} to {df['DateTime'].max()}")
    
    # Extract vegetative stage data (first 252 observations)
    df_veg = create_vegetative_stage_subset(df, n_obs=252)
    print(f"Vegetative stage observations: {len(df_veg)}")
    print(f"Vegetative stage date range: {df_veg['DateTime'].min()} to {df_veg['DateTime'].max()}")
    
    # Calculate diel averages
    print("Calculating diel averages...")
    diel_stats = calculate_diel_averages(df_veg)
    
    # Generate plots based on type requested
    print(f"Creating diel pattern plots (type: {plot_type})...")
    
    if plot_type in ['unified', 'all']:
        # Create the two main unified plots
        print("Creating unified methane-water plot...")
        fig1 = plot_methane_water_unified(df_veg, diel_stats, 'methane_water_unified.png')
        
        print("Creating unified methane-temperature plot...")
        fig2 = plot_methane_temperature_unified(df_veg, diel_stats, 'methane_temperature_unified.png')
    
    if plot_type in ['separate', 'all']:
        # Original separate plots
        print("Creating separate plots...")
        fig3 = plot_water_level_diel(df_veg, diel_stats, 'water_levels_diel.png')
        fig4 = plot_methane_temperature_diel(df_veg, diel_stats, 'methane_temperature_diel.png')
        fig5 = plot_comprehensive_diel(df_veg, diel_stats, 'comprehensive_diel_analysis.png')
        fig6 = plot_correlation_heatmap(df_veg, 'vegetative_correlations.png')
    
    # Print summary statistics
    print("\n=== DIEL ANALYSIS SUMMARY ===")
    print(f"Peak methane emission time: {diel_stats.loc[diel_stats['CH4 mg/m^2/h_mean'].idxmax(), 'Hour']:.1f}:00")
    print(f"Peak methane flux: {diel_stats['CH4 mg/m^2/h_mean'].max():.2f} ± {diel_stats.loc[diel_stats['CH4 mg/m^2/h_mean'].idxmax(), 'CH4 mg/m^2/h_std']:.2f} mg m⁻² h⁻¹")
    print(f"Mean water depth: {df_veg['h'].mean():.2f} ± {df_veg['h'].std():.2f} cm")
    print(f"Water depth range: {df_veg['h'].min():.2f} to {df_veg['h'].max():.2f} cm")
    
    # Temperature correlations with CH4
    temp_corr = df_veg[['Ta', 'Ts', 'Tf', 'CH4 mg/m^2/h']].corr()['CH4 mg/m^2/h']
    print(f"\nTemperature correlations with CH4:")
    print(f"Air temperature: {temp_corr['Ta']:.3f}")
    print(f"Soil temperature: {temp_corr['Ts']:.3f}")
    print(f"Floodwater temperature: {temp_corr['Tf']:.3f}")
    
    return df, df_veg, diel_stats

# Example usage
if __name__ == "__main__":
    # Replace with your actual file path
    file_path = r"G:\2025\PLS2018UPLB\ANALYSIS\METADATA\NEWDATA\UYData_new.csv"
    
    # RECOMMENDED: Create the two unified plots (methane+water, methane+temperatures)
    print("=== CREATING UNIFIED PLOTS ===")
    df, df_veg, diel_stats = main(file_path, plot_type='unified')
    
    # Alternative: Create all plots (unified + separate)
    # df, df_veg, diel_stats = main(file_path, plot_type='all')
    
    # Alternative: Just separate plots
    # df, df_veg, diel_stats = main(file_path, plot_type='separate')
    
    print("\nAnalysis complete! Generated plots:")
    print("📊 methane_water_unified.png - Methane flux vs. water levels")
    print("🌡️ methane_temperature_unified.png - Methane flux vs. all temperatures")
    
    # Quick analysis summary
    peak_ch4_hour = diel_stats.loc[diel_stats['CH4 mg/m^2/h_mean'].idxmax(), 'Hour']
    peak_ts_hour = diel_stats.loc[diel_stats['Ts_mean'].idxmax(), 'Hour']
    
    print(f"\n🔍 Key findings:")
    print(f"⏰ Peak CH₄ emission: {peak_ch4_hour:.1f}:00")
    print(f"🌡️ Peak soil temperature: {peak_ts_hour:.1f}:00")
    print(f"💧 Mean water depth: {df_veg['h'].mean():.1f} cm")
    print(f"✅ Supports twilight flooding strategy (irrigation post-15:00)")
    
    # Correlation summary
    ch4_water_corr = df_veg['CH4 mg/m^2/h'].corr(df_veg['h'])
    ch4_temp_corr = df_veg['CH4 mg/m^2/h'].corr(df_veg['Ts'])
    print(f"\n📈 Correlations with CH₄:")
    print(f"Water depth: {ch4_water_corr:.3f}")
    print(f"Soil temperature: {ch4_temp_corr:.3f}")