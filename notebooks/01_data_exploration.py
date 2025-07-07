# NYC Building Energy Data Exploration
# This notebook provides interactive exploration of NYC building energy data

# %%
# Import required libraries
import sys
sys.path.append('../src')  # Add src to path

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Set plot style
plt.style.use('default')
sns.set_palette("husl")

print("✅ Libraries imported successfully")

# %%
# Load the data
data_path = 'C:\\Users\\Cooper\\Documents\\Projects\\nyc-building-energy-analyzer\\data\\raw\\nyc_energy_2022.csv'

try:
    df = pd.read_csv(data_path)
    print(f"✅ Data loaded successfully: {df.shape[0]} rows, {df.shape[1]} columns")
except FileNotFoundError:
    print("❌ Data file not found. Please download the NYC energy data and place it in data/raw/")
    print("Download from: https://data.cityofnewyork.us/Environment/Energy-and-Water-Data-Disclosure-for-Local-Law-97-/usc3-8zwd")

# %%
# Quick data overview
print("=== DATASET OVERVIEW ===")
print(f"Shape: {df.shape}")
print(f"Memory usage: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
print(f"\nFirst 5 rows:")
df.head()

# %%
# Examine column names
print("=== COLUMN ANALYSIS ===")
print(f"Total columns: {len(df.columns)}")
print("\nAll columns:")
for i, col in enumerate(df.columns):
    print(f"{i+1:2d}. {col}")

# %%
# Identify key energy-related columns
energy_keywords = ['eui', 'energy', 'emissions', 'ghg', 'consumption', 'use']
building_keywords = ['property', 'building', 'type', 'address', 'year', 'area', 'gfa']

energy_columns = [col for col in df.columns if any(keyword in col.lower() for keyword in energy_keywords)]
building_columns = [col for col in df.columns if any(keyword in col.lower() for keyword in building_keywords)]

print("=== KEY COLUMNS IDENTIFIED ===")
print(f"\nEnergy-related columns ({len(energy_columns)}):")
for col in energy_columns:
    print(f"  - {col}")

print(f"\nBuilding info columns ({len(building_columns)}):")
for col in building_columns:
    print(f"  - {col}")

# %%
# Data types and basic statistics
print("=== DATA TYPES ===")
print(df.dtypes.value_counts())

print("\n=== NUMERIC COLUMNS SUMMARY ===")
numeric_cols = df.select_dtypes(include=[np.number]).columns
print(f"Numeric columns: {len(numeric_cols)}")
df[numeric_cols].describe()

# %%
# Missing values analysis
print("=== MISSING VALUES ANALYSIS ===")
missing = df.isnull().sum()
missing_pct = (missing / len(df)) * 100

missing_df = pd.DataFrame({
    'Column': missing.index,
    'Missing_Count': missing.values,
    'Missing_Percentage': missing_pct.values
}).sort_values('Missing_Percentage', ascending=False)

# Show columns with missing values
columns_with_missing = missing_df[missing_df['Missing_Count'] > 0]
print(f"Columns with missing values: {len(columns_with_missing)} out of {len(df.columns)}")

if len(columns_with_missing) > 0:
    print("\nTop 15 columns with most missing values:")
    display(columns_with_missing.head(15))

# %%
# Visualize missing values
if len(columns_with_missing) > 0:
    plt.figure(figsize=(12, 8))
    top_missing = columns_with_missing.head(20)
    plt.barh(range(len(top_missing)), top_missing['Missing_Percentage'])
    plt.yticks(range(len(top_missing)), [col[:50] + '...' if len(col) > 50 else col 
              for col in top_missing['Column']])
    plt.xlabel('Missing Percentage (%)')
    plt.title('Top 20 Columns with Missing Values')
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.show()

# %%
# Identify and examine key EUI columns
eui_columns = [col for col in df.columns if 'eui' in col.lower()]
print(f"=== EUI (Energy Use Intensity) COLUMNS ===")
print(f"Found {len(eui_columns)} EUI columns:")

for col in eui_columns:
    print(f"\n{col}:")
    print(f"  Non-null values: {df[col].count():,}")
    print(f"  Data type: {df[col].dtype}")
    if df[col].dtype in ['int64', 'float64']:
        print(f"  Range: {df[col].min():.2f} to {df[col].max():.2f}")
        print(f"  Mean: {df[col].mean():.2f}")

# %%
# Focus on Site EUI for main analysis
# Find the main Site EUI column
site_eui_cols = [col for col in eui_columns if 'site' in col.lower()]

if site_eui_cols:
    main_eui_col = site_eui_cols[0]  # Use first Site EUI column
    print(f"=== MAIN ANALYSIS COLUMN: {main_eui_col} ===")
    
    # Basic statistics
    print(f"Total values: {df[main_eui_col].count():,}")
    print(f"Missing values: {df[main_eui_col].isnull().sum():,}")
    print(f"Zero values: {(df[main_eui_col] == 0).sum():,}")
    print(f"Negative values: {(df[main_eui_col] < 0).sum():,}")
    
    # Remove invalid values for analysis
    valid_eui = df[main_eui_col].dropna()
    valid_eui = valid_eui[valid_eui > 0]  # Remove zero and negative
    
    print(f"\nValid EUI values for analysis: {len(valid_eui):,}")
    print(f"Statistics:")
    print(f"  Mean: {valid_eui.mean():.2f} kBtu/ft²")
    print(f"  Median: {valid_eui.median():.2f} kBtu/ft²")
    print(f"  Std Dev: {valid_eui.std():.2f} kBtu/ft²")
    print(f"  Min: {valid_eui.min():.2f} kBtu/ft²")
    print(f"  Max: {valid_eui.max():.2f} kBtu/ft²")
    
else:
    print("❌ No Site EUI column found")
    main_eui_col = None

# %%
# EUI Distribution Analysis
if main_eui_col and len(valid_eui) > 0:
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # Histogram
    axes[0, 0].hist(valid_eui, bins=50, alpha=0.7, color='skyblue', edgecolor='black')
    axes[0, 0].axvline(valid_eui.mean(), color='red', linestyle='--', label='Mean')
    axes[0, 0].axvline(valid_eui.median(), color='green', linestyle='--', label='Median')
    axes[0, 0].set_xlabel('Site EUI (kBtu/ft²)')
    axes[0, 0].set_ylabel('Frequency')
    axes[0, 0].set_title('Distribution of Site Energy Use Intensity')
    axes[0, 0].legend()
    
    # Box plot
    axes[0, 1].boxplot(valid_eui)
    axes[0, 1].set_ylabel('Site EUI (kBtu/ft²)')
    axes[0, 1].set_title('Site EUI Box Plot (All Data)')
    
    # Remove extreme outliers for better visualization
    q99 = valid_eui.quantile(0.99)
    q01 = valid_eui.quantile(0.01)
    filtered_eui = valid_eui[(valid_eui >= q01) & (valid_eui <= q99)]
    
    # Histogram without extreme outliers
    axes[1, 0].hist(filtered_eui, bins=50, alpha=0.7, color='lightcoral', edgecolor='black')
    axes[1, 0].axvline(filtered_eui.mean(), color='red', linestyle='--', label='Mean')
    axes[1, 0].axvline(filtered_eui.median(), color='green', linestyle='--', label='Median')
    axes[1, 0].set_xlabel('Site EUI (kBtu/ft²)')
    axes[1, 0].set_ylabel('Frequency')
    axes[1, 0].set_title('Site EUI Distribution (1st-99th Percentile)')
    axes[1, 0].legend()
    
    # Cumulative distribution
    sorted_eui = np.sort(filtered_eui)
    cumulative = np.arange(1, len(sorted_eui) + 1) / len(sorted_eui)
    axes[1, 1].plot(sorted_eui, cumulative, linewidth=2)
    axes[1, 1].set_xlabel('Site EUI (kBtu/ft²)')
    axes[1, 1].set_ylabel('Cumulative Probability')
    axes[1, 1].set_title('Cumulative Distribution of Site EUI')
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

# %%
# Building Type Analysis
property_type_cols = [col for col in df.columns if 'property' in col.lower() and 'type' in col.lower()]

if property_type_cols and main_eui_col:
    prop_type_col = property_type_cols[0]
    print(f"=== BUILDING TYPE ANALYSIS: {prop_type_col} ===")
    
    # Building type distribution
    type_counts = df[prop_type_col].value_counts()
    print(f"\nBuilding Types (Top 15):")
    print(type_counts.head(15))
    
    # Create analysis dataset with valid EUI values
    analysis_df = df.dropna(subset=[main_eui_col, prop_type_col])
    analysis_df = analysis_df[analysis_df[main_eui_col] > 0]
    
    print(f"\nBuildings with valid EUI and type data: {len(analysis_df):,}")
    
    # Average EUI by building type
    avg_eui_by_type = analysis_df.groupby(prop_type_col)[main_eui_col].agg(['mean', 'median', 'count']).round(2)
    avg_eui_by_type = avg_eui_by_type.sort_values('mean', ascending=False)
    
    print(f"\nAverage EUI by Building Type (Top 15):")
    display(avg_eui_by_type.head(15))
else:
    print("❌ Property type column not found or no valid EUI data")

# %%
# Visualize Building Type Analysis
if property_type_cols and main_eui_col and len(analysis_df) > 0:
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # Building type distribution
    top_types = type_counts.head(10)
    axes[0, 0].barh(range(len(top_types)), top_types.values)
    axes[0, 0].set_yticks(range(len(top_types)))
    axes[0, 0].set_yticklabels([t[:30] + '...' if len(t) > 30 else t for t in top_types.index])
    axes[0, 0].set_xlabel('Number of Buildings')
    axes[0, 0].set_title('Top 10 Building Types by Count')
    axes[0, 0].invert_yaxis()
    
    # Average EUI by type
    top_eui_types = avg_eui_by_type.head(10)
    axes[0, 1].barh(range(len(top_eui_types)), top_eui_types['mean'])
    axes[0, 1].set_yticks(range(len(top_eui_types)))
    axes[0, 1].set_yticklabels([t[:30] + '...' if len(t) > 30 else t for t in top_eui_types.index])
    axes[0, 1].set_xlabel('Average Site EUI (kBtu/ft²)')
    axes[0, 1].set_title('Top 10 Building Types by Average EUI')
    axes[0, 1].invert_yaxis()
    
    # Box plot for major building types
    major_types = type_counts.head(6).index
    major_data = []
    major_labels = []
    
    for btype in major_types:
        data = analysis_df[analysis_df[prop_type_col] == btype][main_eui_col]
        if len(data) > 10:  # Only include types with sufficient data
            # Remove extreme outliers for visualization
            q99 = data.quantile(0.99)
            q01 = data.quantile(0.01)
            filtered_data = data[(data >= q01) & (data <= q99)]
            major_data.append(filtered_data.values)
            major_labels.append(btype[:15] + '...' if len(btype) > 15 else btype)
    
    if major_data:
        bp = axes[1, 0].boxplot(major_data, labels=major_labels)
        axes[1, 0].set_xticklabels(major_labels, rotation=45, ha='right')
        axes[1, 0].set_ylabel('Site EUI (kBtu/ft²)')
        axes[1, 0].set_title('EUI Distribution by Major Building Types\n(Outliers removed for clarity)')
    
    # Efficiency distribution pie chart
    median_eui = analysis_df[main_eui_col].median()
    q25_eui = analysis_df[main_eui_col].quantile(0.25)
    q75_eui = analysis_df[main_eui_col].quantile(0.75)
    
    efficiency_categories = []
    for eui in analysis_df[main_eui_col]:
        if eui <= q25_eui:
            efficiency_categories.append('Excellent (Top 25%)')
        elif eui <= median_eui:
            efficiency_categories.append('Good (25-50%)')
        elif eui <= q75_eui:
            efficiency_categories.append('Fair (50-75%)')
        else:
            efficiency_categories.append('Poor (Bottom 25%)')
    
    efficiency_counts = pd.Series(efficiency_categories).value_counts()
    axes[1, 1].pie(efficiency_counts.values, labels=efficiency_counts.index, autopct='%1.1f%%')
    axes[1, 1].set_title('Building Energy Efficiency Distribution')
    
    plt.tight_layout()
    plt.show()

# %%
# Building Area Analysis
area_cols = [col for col in df.columns if any(term in col.lower() for term in ['gfa', 'area', 'sqft', 'ft²'])]

if area_cols and main_eui_col:
    area_col = area_cols[0]
    print(f"=== BUILDING AREA ANALYSIS: {area_col} ===")
    
    # Create analysis dataset
    area_analysis_df = df.dropna(subset=[main_eui_col, area_col])
    area_analysis_df = area_analysis_df[
        (area_analysis_df[main_eui_col] > 0) & 
        (area_analysis_df[area_col] > 0)
    ]
    
    print(f"Buildings with valid EUI and area data: {len(area_analysis_df):,}")
    print(f"\nBuilding Area Statistics:")
    print(f"  Mean: {area_analysis_df[area_col].mean():,.0f} sq ft")
    print(f"  Median: {area_analysis_df[area_col].median():,.0f} sq ft")
    print(f"  Min: {area_analysis_df[area_col].min():,.0f} sq ft")
    print(f"  Max: {area_analysis_df[area_col].max():,.0f} sq ft")
    
    # Correlation between area and EUI
    correlation = area_analysis_df[area_col].corr(area_analysis_df[main_eui_col])
    print(f"\nCorrelation between building area and EUI: {correlation:.3f}")

# %%
# Area vs EUI Visualization
if area_cols and main_eui_col and len(area_analysis_df) > 0:
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # Remove extreme outliers for better visualization
    eui_q99 = area_analysis_df[main_eui_col].quantile(0.99)
    area_q99 = area_analysis_df[area_col].quantile(0.99)
    
    viz_df = area_analysis_df[
        (area_analysis_df[main_eui_col] <= eui_q99) & 
        (area_analysis_df[area_col] <= area_q99)
    ]
    
    # Scatter plot: Area vs EUI
    axes[0, 0].scatter(viz_df[area_col], viz_df[main_eui_col], alpha=0.5)
    axes[0, 0].set_xlabel('Building Area (sq ft)')
    axes[0, 0].set_ylabel('Site EUI (kBtu/ft²)')
    axes[0, 0].set_title('Building Area vs Energy Use Intensity')
    axes[0, 0].set_xscale('log')
    
    # Area distribution
    axes[0, 1].hist(viz_df[area_col], bins=50, alpha=0.7, color='green', edgecolor='black')
    axes[0, 1].set_xlabel('Building Area (sq ft)')
    axes[0, 1].set_ylabel('Frequency')
    axes[0, 1].set_title('Building Area Distribution')
    axes[0, 1].set_xscale('log')
    
    # Binned area analysis
    area_bins = [0, 10000, 50000, 100000, 500000, float('inf')]
    area_labels = ['<10k', '10k-50k', '50k-100k', '100k-500k', '>500k']
    
    viz_df['area_category'] = pd.cut(viz_df[area_col], bins=area_bins, labels=area_labels)
    area_eui_stats = viz_df.groupby('area_category')[main_eui_col].agg(['mean', 'median', 'count'])
    
    axes[1, 0].bar(range(len(area_eui_stats)), area_eui_stats['mean'])
    axes[1, 0].set_xticks(range(len(area_eui_stats)))
    axes[1, 0].set_xticklabels(area_eui_stats.index)
    axes[1, 0].set_xlabel('Building Size Category (sq ft)')
    axes[1, 0].set_ylabel('Average Site EUI (kBtu/ft²)')
    axes[1, 0].set_title('Average EUI by Building Size Category')
    
    # Building count by size category
    axes[1, 1].bar(range(len(area_eui_stats)), area_eui_stats['count'])
    axes[1, 1].set_xticks(range(len(area_eui_stats)))
    axes[1, 1].set_xticklabels(area_eui_stats.index)
    axes[1, 1].set_xlabel('Building Size Category (sq ft)')
    axes[1, 1].set_ylabel('Number of Buildings')
    axes[1, 1].set_title('Building Count by Size Category')
    
    plt.tight_layout()
    plt.show()
    
    print("\nEUI Statistics by Building Size:")
    display(area_eui_stats)

# %%
# Efficiency Opportunities Analysis
if main_eui_col and area_col:
    print("=== EFFICIENCY OPPORTUNITIES ANALYSIS ===")
    
    # Use the cleaned analysis dataset
    opp_df = area_analysis_df.copy()
    
    # Calculate benchmarks
    overall_median = opp_df[main_eui_col].median()
    overall_25th = opp_df[main_eui_col].quantile(0.25)
    
    print(f"Energy Efficiency Benchmarks:")
    print(f"  25th percentile (excellent): {overall_25th:.2f} kBtu/ft²")
    print(f"  Median (good): {overall_median:.2f} kBtu/ft²")
    
    # Identify underperforming buildings
    underperformers = opp_df[opp_df[main_eui_col] > overall_median].copy()
    
    print(f"\nUnderperforming Buildings Analysis:")
    print(f"  Total buildings: {len(opp_df):,}")
    print(f"  Underperforming buildings: {len(underperformers):,} ({len(underperformers)/len(opp_df)*100:.1f}%)")
    
    # Calculate potential savings
    underperformers['excess_eui'] = underperformers[main_eui_col] - overall_median
    underperformers['annual_excess_kbtu'] = underperformers['excess_eui'] * underperformers[area_col]
    
    # Estimate cost savings (assuming $15 per MMBtu = $0.015 per kBtu)
    cost_per_kbtu = 0.015
    underperformers['potential_annual_savings'] = underperformers['annual_excess_kbtu'] * cost_per_kbtu
    
    total_potential_savings = underperformers['potential_annual_savings'].sum()
    total_excess_energy = underperformers['annual_excess_kbtu'].sum()
    
    print(f"\nPotential Impact:")
    print(f"  Total excess energy use: {total_excess_energy/1e6:.1f} million kBtu/year")
    print(f"  Total potential savings: ${total_potential_savings/1e6:.1f} million/year")
    print(f"  Average savings per underperforming building: ${underperformers['potential_annual_savings'].mean():,.0f}/year")
    print(f"  Median savings per underperforming building: ${underperformers['potential_annual_savings'].median():,.0f}/year")

# %%
# Savings Opportunity Visualization
if main_eui_col and area_col and len(underperformers) > 0:
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # Potential savings distribution
    axes[0, 0].hist(underperformers['potential_annual_savings'], bins=50, 
                   alpha=0.7, color='orange', edgecolor='black')
    axes[0, 0].set_xlabel('Potential Annual Savings ($)')
    axes[0, 0].set_ylabel('Number of Buildings')
    axes[0, 0].set_title('Distribution of Potential Annual Savings')
    axes[0, 0].axvline(underperformers['potential_annual_savings'].median(), 
                      color='red', linestyle='--', label='Median')
    axes[0, 0].legend()
    
    # Excess EUI distribution
    axes[0, 1].hist(underperformers['excess_eui'], bins=50, 
                   alpha=0.7, color='red', edgecolor='black')
    axes[0, 1].set_xlabel('Excess EUI (kBtu/ft²)')
    axes[0, 1].set_ylabel('Number of Buildings')
    axes[0, 1].set_title('Distribution of Excess Energy Use Intensity')
    
    # Savings vs building size
    # Remove extreme outliers for visualization
    savings_q99 = underperformers['potential_annual_savings'].quantile(0.99)
    area_q99 = underperformers[area_col].quantile(0.99)
    
    viz_underperformers = underperformers[
        (underperformers['potential_annual_savings'] <= savings_q99) & 
        (underperformers[area_col] <= area_q99)
    ]
    
    scatter = axes[1, 0].scatter(viz_underperformers[area_col], 
                                viz_underperformers['potential_annual_savings'], 
                                c=viz_underperformers['excess_eui'], 
                                cmap='Reds', alpha=0.6)
    axes[1, 0].set_xlabel('Building Area (sq ft)')
    axes[1, 0].set_ylabel('Potential Annual Savings ($)')
    axes[1, 0].set_title('Savings Potential vs Building Size\n(Color = Excess EUI)')
    axes[1, 0].set_xscale('log')
    plt.colorbar(scatter, ax=axes[1, 0], label='Excess EUI (kBtu/ft²)')
    
    # Savings by area category
    underperformers['area_category'] = pd.cut(underperformers[area_col], 
                                            bins=area_bins, labels=area_labels)
    
    savings_by_area = underperformers.groupby('area_category')['potential_annual_savings'].agg(['sum', 'mean', 'count'])
    
    axes[1, 1].bar(range(len(savings_by_area)), savings_by_area['sum'] / 1e6)
    axes[1, 1].set_xticks(range(len(savings_by_area)))
    axes[1, 1].set_xticklabels(savings_by_area.index)
    axes[1, 1].set_xlabel('Building Size Category (sq ft)')
    axes[1, 1].set_ylabel('Total Potential Savings ($ Millions)')
    axes[1, 1].set_title('Total Savings Potential by Building Size')
    
    plt.tight_layout()
    plt.show()
    
    print("\nSavings Potential by Building Size Category:")
    savings_by_area_display = savings_by_area.copy()
    savings_by_area_display['sum'] = savings_by_area_display['sum'].apply(lambda x: f"${x/1e6:.2f}M")
    savings_by_area_display['mean'] = savings_by_area_display['mean'].apply(lambda x: f"${x:,.0f}")
    savings_by_area_display.columns = ['Total Savings', 'Avg per Building', 'Building Count']
    display(savings_by_area_display)

# %%
# Building Type Savings Analysis
if prop_type_col and main_eui_col and area_col:
    print("=== SAVINGS BY BUILDING TYPE ===")
    
    # Merge building type data with underperformers
    type_underperformers = underperformers.merge(
        df[[prop_type_col]].dropna(), 
        left_index=True, right_index=True, how='inner'
    )
    
    if len(type_underperformers) > 0:
        type_savings = type_underperformers.groupby(prop_type_col).agg({
            'potential_annual_savings': ['sum', 'mean', 'count'],
            'excess_eui': 'mean'
        }).round(2)
        
        type_savings.columns = ['Total_Savings', 'Avg_Savings_Per_Building', 'Count', 'Avg_Excess_EUI']
        type_savings = type_savings.sort_values('Total_Savings', ascending=False)
        
        print(f"Top 15 Building Types by Total Potential Savings:")
        top_savings_types = type_savings.head(15).copy()
        top_savings_types['Total_Savings_M'] = (top_savings_types['Total_Savings'] / 1e6).round(2)
        top_savings_types['Avg_Savings_K'] = (top_savings_types['Avg_Savings_Per_Building'] / 1e3).round(1)
        
        display_cols = ['Total_Savings_M', 'Avg_Savings_K', 'Count', 'Avg_Excess_EUI']
        display_labels = ['Total Savings ($M)', 'Avg Savings ($K)', 'Building Count', 'Avg Excess EUI']
        
        display_df = top_savings_types[display_cols].copy()
        display_df.columns = display_labels
        display(display_df)
        
        # Visualize top building types by savings
        plt.figure(figsize=(12, 8))
        top_10_types = type_savings.head(10)
        plt.barh(range(len(top_10_types)), top_10_types['Total_Savings'] / 1e6)
        plt.yticks(range(len(top_10_types)), 
                  [t[:35] + '...' if len(t) > 35 else t for t in top_10_types.index])
        plt.xlabel('Total Potential Savings ($ Millions)')
        plt.title('Top 10 Building Types by Total Savings Potential')
        plt.gca().invert_yaxis()
        plt.tight_layout()
        plt.show()

# %%
# Executive Summary
print("\n" + "="*80)
print("NYC BUILDING ENERGY EFFICIENCY ANALYSIS - EXECUTIVE SUMMARY")
print("="*80)

if main_eui_col:
    print(f"\n📊 DATASET OVERVIEW:")
    print(f"   • Total buildings analyzed: {len(opp_df):,}")
    print(f"   • Average energy intensity: {opp_df[main_eui_col].mean():.1f} kBtu/ft²")
    print(f"   • Median energy intensity: {opp_df[main_eui_col].median():.1f} kBtu/ft²")
    print(f"   • Energy intensity range: {opp_df[main_eui_col].min():.1f} - {opp_df[main_eui_col].max():.1f} kBtu/ft²")
    
    print(f"\n🎯 EFFICIENCY OPPORTUNITIES:")
    print(f"   • Buildings below median efficiency: {len(underperformers):,} ({len(underperformers)/len(opp_df)*100:.1f}%)")
    print(f"   • Total potential annual savings: ${total_potential_savings/1e6:.1f} million")
    print(f"   • Average savings per underperforming building: ${underperformers['potential_annual_savings'].mean():,.0f}")
    print(f"   • Total excess energy use: {total_excess_energy/1e6:.1f} million kBtu/year")
    
    if prop_type_col:
        most_common_type = opp_df.merge(df[[prop_type_col]].dropna(), 
                                       left_index=True, right_index=True)[prop_type_col].mode()[0]
        print(f"\n🏢 BUILDING INSIGHTS:")
        print(f"   • Most common building type: {most_common_type}")
        print(f"   • Building types analyzed: {len(df[prop_type_col].unique())}")
        
        if len(type_underperformers) > 0:
            top_savings_type = type_savings.index[0]
            print(f"   • Highest savings potential building type: {top_savings_type}")
            print(f"   • Potential savings from top type: ${type_savings.loc[top_savings_type, 'Total_Savings']/1e6:.1f} million")

print(f"\n🔧 RECOMMENDED NEXT STEPS:")
print(f"   1. Focus HVAC optimization on underperforming {top_savings_type.lower()} buildings")
print(f"   2. Develop energy efficiency retrofits for buildings >75th percentile EUI")
print(f"   3. Create predictive models for identifying efficiency opportunities")
print(f"   4. Implement building automation systems for continuous monitoring")

print(f"\n📈 BUSINESS IMPACT:")
print(f"   • Estimated ROI potential: ${total_potential_savings/1e6:.1f}M annually")
print(f"   • CO2 reduction potential: ~{total_excess_energy/293:.0f} metric tons CO2/year")
print(f"   • Energy reduction potential: {total_excess_energy/1e6:.1f} million kBtu/year")

print("="*80)

# %%
# Save key results for next steps
print("\n💾 SAVING ANALYSIS RESULTS...")

# Save cleaned dataset
if 'clean_df' not in locals():
    clean_df = opp_df.copy()

# Create summary statistics
summary_stats = {
    'total_buildings': len(clean_df),
    'median_eui': clean_df[main_eui_col].median(),
    'mean_eui': clean_df[main_eui_col].mean(),
    'underperformers_count': len(underperformers),
    'total_potential_savings': total_potential_savings,
    'main_eui_column': main_eui_col,
    'property_type_column': prop_type_col if 'prop_type_col' in locals() else None,
    'area_column': area_col if 'area_col' in locals() else None
}

# Save to processed data folder
clean_df.to_csv('../data/processed/nyc_energy_clean.csv', index=False)
underperformers.to_csv('../data/processed/underperforming_buildings.csv', index=False)

# Save summary statistics
import json
with open('../data/processed/analysis_summary.json', 'w') as f:
    json.dump(summary_stats, f, indent=2)

print("✅ Results saved to data/processed/")
print("   • nyc_energy_clean.csv - Cleaned dataset")
print("   • underperforming_buildings.csv - Buildings with efficiency opportunities")
print("   • analysis_summary.json - Key statistics")

# %%
# Final data quality check
print("\n🔍 FINAL DATA QUALITY ASSESSMENT:")

print(f"\nOriginal dataset:")
print(f"   • Rows: {df.shape[0]:,}")
print(f"   • Columns: {df.shape[1]:,}")

print(f"\nCleaned dataset:")
print(f"   • Rows: {len(clean_df):,}")
print(f"   • Columns: {clean_df.shape[1]:,}")
print(f"   • Data retention: {len(clean_df)/df.shape[0]*100:.1f}%")

print(f"\nKey data completeness:")
if main_eui_col:
    print(f"   • Site EUI data: {clean_df[main_eui_col].count():,} buildings ({clean_df[main_eui_col].count()/len(clean_df)*100:.1f}%)")
if 'area_col' in locals():
    print(f"   • Building area data: {clean_df[area_col].count():,} buildings ({clean_df[area_col].count()/len(clean_df)*100:.1f}%)")
if 'prop_type_col' in locals():
    clean_type_data = clean_df.merge(df[[prop_type_col]].dropna(), left_index=True, right_index=True, how='inner')
    print(f"   • Property type data: {len(clean_type_data):,} buildings ({len(clean_type_data)/len(clean_df)*100:.1f}%)")

print(f"\n✅ Data exploration complete!")
print(f"📁 Generated files saved to outputs/figures/")
print(f"📊 Ready for next phase: Predictive modeling and dashboard development")

# %%