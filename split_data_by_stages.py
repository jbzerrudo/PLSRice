import pandas as pd
import numpy as np
from datetime import datetime
import os

def split_data_by_growth_stages(input_file="UYData_mod.csv", output_dir=r"G:\2025\PLS2018UPLB\ANALYSIS\METADATA\NEWDATA"):
    """
    Split the main UYData.csv into growth stage periods based on dates
    """
    
    # Create output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Load the data with explicit column handling
    print(f"Loading data from: {input_file}")
    
    try:
        # Force pandas to read ALL columns including the first one
        data = pd.read_csv(input_file, encoding='cp1252', header=0, index_col=False)
        print("✓ CSV loaded successfully")
        print(f"Shape: {data.shape}")
        print(f"All columns: {data.columns.tolist()}")
        
        # Check first row of data to see what we actually got
        print("First row:", data.iloc[0].head())
        
    except Exception as e:
        print(f"Error loading CSV: {e}")
        
        # Try alternative reading method
        print("Trying alternative CSV reading...")
        try:
            data = pd.read_csv(input_file, encoding='utf-8', header=0)
            print("✓ CSV loaded with UTF-8 encoding")
        except:
            data = pd.read_csv(input_file, header=0)
            print("✓ CSV loaded with default encoding")
        
        print(f"Shape: {data.shape}")
        print(f"Columns: {data.columns.tolist()}")
    
    # Check if Date column exists, if not create it
    if 'Date' not in data.columns:
        print("⚠ Date column missing. Creating from del_t...")
        
        # Create Date column from del_t
        # According to your header: "1/4/25 12:00 AM" corresponds to del_t=0
        # This means 1/4/2016 12:00 AM (midnight) = del_t 0
        start_date = pd.to_datetime('2016-01-04 00:00:00')  # Start at midnight
        data['Date'] = start_date + pd.to_timedelta(data['del_t'], unit='minutes')
        
        print(f"✓ Created Date column from del_t")
        print(f"Date range: {data['Date'].min()} to {data['Date'].max()}")
        
    else:
        print("✓ Date column found")
        # Handle the existing Date column
        try:
            data['Date'] = pd.to_datetime(data['Date'], format='%m/%d/%y %I:%M %p', errors='coerce')
            # Fix year issue
            mask = data['Date'].dt.year > 2020
            if mask.any():
                data.loc[mask, 'Date'] = data.loc[mask, 'Date'] - pd.DateOffset(years=100)
            print(f"Date range: {data['Date'].min()} to {data['Date'].max()}")
        except Exception as e:
            print(f"Date conversion error: {e}")
            return None
    
    # Now continue with splitting by stages
    stages = {
        'Vegetative': {
            'start': '2016-01-04',
            'end': '2016-01-09',
            'description': 'Early growth stage with AWD treatment'
        },
        'Contfood': {
            'start': '2016-01-04', 
            'end': '2016-01-09',
            'description': 'Continuous flooding control (same period as vegetative)'
        },
        'Reproductive': {
            'start': '2016-01-21',
            'end': '2016-01-26', 
            'description': 'Flowering and grain formation stage'
        },
        'Ripening': {
            'start': '2016-02-25',
            'end': '2016-03-01',
            'description': 'Final maturation stage'
        }
    }
    
    # Split data by stages
    stage_data = {}
    
    for stage_name, period in stages.items():
        start_date = pd.to_datetime(period['start'])
        end_date = pd.to_datetime(period['end'])
        
        # Filter data for this period
        mask = (data['Date'] >= start_date) & (data['Date'] <= end_date)
        filtered_data = data[mask].copy()
        
        print(f"\n{stage_name} Stage ({period['start']} to {period['end']}):")
        print(f"  - {len(filtered_data)} observations")
        print(f"  - {period['description']}")
        
        if len(filtered_data) > 0:
            # Save to separate CSV file
            output_file = os.path.join(output_dir, f"{stage_name}Data.csv")
            filtered_data.to_csv(output_file, index=False)
            print(f"  - Saved to: {output_file}")
            
            # Store for analysis
            stage_data[stage_name] = filtered_data
            
            # Show date range of actual data
            print(f"  - Actual date range: {filtered_data['Date'].min()} to {filtered_data['Date'].max()}")
        else:
            print(f"  - WARNING: No data found for this period!")
    
    return stage_data

def analyze_stage_overlaps(stage_data):
    """
    Analyze the overlap between Vegetative and Contfood (same time period)
    """
    print("\n" + "="*60)
    print("OVERLAP ANALYSIS: Vegetative vs Continuous Flooding")
    print("="*60)
    
    if 'Vegetative' in stage_data and 'Contfood' in stage_data:
        veg_data = stage_data['Vegetative']
        cont_data = stage_data['Contfood']
        
        print(f"Vegetative observations: {len(veg_data)}")
        print(f"Continuous flooding observations: {len(cont_data)}")
        
        # Check if they're the same data (they should be for the same time period)
        if len(veg_data) == len(cont_data):
            print("✓ Same number of observations (expected for same time period)")
            
            # Compare CH4 emissions
            veg_ch4_mean = veg_data['CH4 mg/m^2/h'].mean()
            cont_ch4_mean = cont_data['CH4 mg/m^2/h'].mean()
            
            print(f"Vegetative CH4 mean: {veg_ch4_mean:.4f} mg/m²/h")
            print(f"Continuous flooding CH4 mean: {cont_ch4_mean:.4f} mg/m²/h")
            
            if abs(veg_ch4_mean - cont_ch4_mean) < 0.001:
                print("⚠ Same CH4 values - this data represents the SAME treatment")
                print("  You may need to separate AWD vs continuous flooding treatments")
            else:
                print("✓ Different CH4 values - represents different treatments")
        else:
            print("⚠ Different number of observations - unexpected")
    
    print("="*60)

def create_stage_analysis_script(stage_name):
    """
    Create a modified version of your main analysis script for each stage
    """
    script_content = f'''
# Analysis script for {stage_name} stage
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

# Import your existing functions
from PLSAnalyser-Final import *

def main_{stage_name.lower()}():
    """Analysis for {stage_name} stage"""
    
    # Set stage-specific output directory
    output_dir = r'G:\\2025\\PLS2018UPLB\\OUTHERE\\{stage_name}'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Load stage-specific data
    data_path = "{stage_name}Data.csv"
    if not os.path.exists(data_path):
        data_path = r"G:\\2025\\PLS2018UPLB\\METDAT\\{stage_name}Data.csv"
    
    print(f"Loading {stage_name} data from: {{data_path}}")
    data = pd.read_csv(data_path)
    
    # Define X (predictors) and Y (response)
    X = data.iloc[:, :-1]  # All columns except the last one
    y = data.iloc[:, -1]   # Last column is CH4
    
    print(f"X shape: {{X.shape}}")
    print(f"y shape: {{y.shape}}")
    print(f"Stage: {stage_name}")
    
    # Run your existing analysis pipeline
    # ... (rest of your main() function code)
    
    return None

if __name__ == "__main__":
    main_{stage_name.lower()}()
'''
    
    # Save the script
    script_file = f"analyze_{stage_name.lower()}.py"
    with open(script_file, 'w') as f:
        f.write(script_content)
    
    print(f"Created analysis script: {script_file}")

def main():
    """Main function to split data and create analysis scripts"""
    
    # Split the data
    print("Splitting UYData.csv by growth stages...")
    stage_data = split_data_by_growth_stages()
    
    # Check if splitting was successful
    if stage_data is not None and len(stage_data) > 0:
        analyze_stage_overlaps(stage_data)
        
        # Create individual analysis scripts for each stage
        print("\nCreating stage-specific analysis scripts...")
        for stage in ['Vegetative', 'Contfood', 'Reproductive', 'Ripening']:
            create_stage_analysis_script(stage)
    else:
        print("❌ Data splitting failed or no stage data created")
        return
    
    print("\n" + "="*60)
    print("DATA SPLITTING COMPLETE!")
    print("="*60)

if __name__ == "__main__":
    main()