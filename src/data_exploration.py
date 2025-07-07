"""
NYC Building Energy Data Exploration
Comprehensive analysis of building energy efficiency patterns
"""

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

class NYCEnergyExplorer:
    """Class for exploring NYC building energy data"""
    
    def __init__(self, data_path: str):
        """Initialize with data path"""
        self.data_path = data_path
        self.df = None
        self.clean_df = None
        
    def load_data(self):
        """Load the NYC energy data"""
        try:
            self.df = pd.read_csv(self.data_path)
            print(f"✅ Data loaded successfully: {self.df.shape[0]} rows, {self.df.shape[1]} columns")
            return True
        except FileNotFoundError:
            print(f"❌ File not found: {self.data_path}")
            return False
        except Exception as e:
            print(f"❌ Error loading data: {e}")
            return False
    
    def basic_info(self):
        """Display basic information about the dataset"""
        if self.df is None:
            print("❌ No data loaded")
            return
            
        print("=== DATASET OVERVIEW ===")
        print(f"Shape: {self.df.shape}")
        print(f"Memory usage: {self.df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
        
        print("\n=== COLUMN INFORMATION ===")
        print(f"Total columns: {len(self.df.columns)}")
        
        # Look for key energy columns
        energy_columns = [col for col in self.df.columns if any(keyword in col.lower() 
                         for keyword in ['eui', 'energy', 'emissions', 'ghg', 'consumption'])]
        print(f"Energy-related columns ({len(energy_columns)}):")
        for col in energy_columns[:10]:  # Show first 10
            print(f"  - {col}")
        
        # Look for building info columns
        building_columns = [col for col in self.df.columns if any(keyword in col.lower() 
                           for keyword in ['property', 'building', 'type', 'address', 'year'])]
        print(f"\nBuilding info columns ({len(building_columns)}):")
        for col in building_columns[:10]:  # Show first 10
            print(f"  - {col}")
    
    def missing_values_analysis(self):
        """Analyze missing values in the dataset"""
        if self.df is None:
            return
            
        print("\n=== MISSING VALUES ANALYSIS ===")
        missing = self.df.isnull().sum()
        missing_pct = (missing / len(self.df)) * 100
        
        missing_df = pd.DataFrame({
            'Column': missing.index,
            'Missing_Count': missing.values,
            'Missing_Percentage': missing_pct.values
        }).sort_values('Missing_Percentage', ascending=False)
        
        # Show columns with missing values
        columns_with_missing = missing_df[missing_df['Missing_Count'] > 0]
        print(f"Columns with missing values: {len(columns_with_missing)}")
        
        if len(columns_with_missing) > 0:
            print("\nTop 15 columns with most missing values:")
            print(columns_with_missing.head(15).to_string(index=False))
            
            # Visualize missing values
            plt.figure(figsize=(12, 8))
            top_missing = columns_with_missing.head(20)
            plt.barh(range(len(top_missing)), top_missing['Missing_Percentage'])
            plt.yticks(range(len(top_missing)), [col[:50] + '...' if len(col) > 50 else col 
                      for col in top_missing['Column']])
            plt.xlabel('Missing Percentage (%)')
            plt.title('Top 20 Columns with Missing Values')
            plt.gca().invert_yaxis()
            plt.tight_layout()
            plt.savefig('outputs/figures/missing_values_analysis.png', dpi=300, bbox_inches='tight')
            plt.show()
    
    def identify_key_columns(self):
        """Identify and standardize key energy columns"""
        if self.df is None:
            return {}
            
        print("\n=== KEY COLUMNS IDENTIFICATION ===")
        
        # Define column mappings based on common NYC energy data column names
        column_mapping = {}
        
        # Energy Use Intensity columns
        eui_candidates = [col for col in self.df.columns if 'eui' in col.lower()]
        if eui_candidates:
            site_eui = [col for col in eui_candidates if 'site' in col.lower() and 'weather' not in col.lower()]
            weather_eui = [col for col in eui_candidates if 'weather' in col.lower()]
            
            if site_eui:
                column_mapping['site_eui'] = site_eui[0]
            if weather_eui:
                column_mapping['weather_eui'] = weather_eui[0]
        
        # Property information - look for the correct area column
        # First try to find the self-reported GFA column
        area_candidates = [col for col in self.df.columns if 'property gfa - self-reported' in col.lower()]
        if not area_candidates:
            # Fallback to calculated GFA
            area_candidates = [col for col in self.df.columns if 'property gfa' in col.lower() and 'calculated' in col.lower()]
        if not area_candidates:
            # Last resort - any GFA column with data
            area_candidates = [col for col in self.df.columns if 'gfa' in col.lower() and self.df[col].count() > 1000]
        
        if area_candidates:
            column_mapping['property_area'] = area_candidates[0]
        
        # Property type
        property_type_candidates = [col for col in self.df.columns if 'property' in col.lower() and 'type' in col.lower()]
        if property_type_candidates:
            # Prefer self-selected over calculated
            self_selected = [col for col in property_type_candidates if 'self' in col.lower()]
            if self_selected:
                column_mapping['property_type'] = self_selected[0]
            else:
                column_mapping['property_type'] = property_type_candidates[0]
        
        # Year information
        year_cols = [col for col in self.df.columns if 'year' in col.lower()]
        if year_cols:
            year_built = [col for col in year_cols if 'built' in col.lower()]
            if year_built:
                column_mapping['year_built'] = year_built[0]
        
        # Emissions
        emission_cols = [col for col in self.df.columns if any(term in col.lower() 
                        for term in ['emission', 'ghg', 'co2'])]
        if emission_cols:
            # Prefer total emissions
            total_emissions = [col for col in emission_cols if 'total' in col.lower()]
            if total_emissions:
                column_mapping['emissions'] = total_emissions[0]
            else:
                column_mapping['emissions'] = emission_cols[0]
        
        # Address/Location
        address_cols = [col for col in self.df.columns if any(term in col.lower() 
                       for term in ['address', 'street', 'location'])]
        if address_cols:
            column_mapping['address'] = address_cols[0]
        
        print("Identified key columns:")
        for key, col in column_mapping.items():
            data_count = self.df[col].count() if col in self.df.columns else 0
            print(f"  {key}: {col} ({data_count:,} non-null values)")
        
        return column_mapping
    
    def clean_data(self, column_mapping=None):
        """Clean and prepare data for analysis"""
        if self.df is None:
            return
            
        print("\n=== DATA CLEANING ===")
        
        if column_mapping is None:
            column_mapping = self.identify_key_columns()
        
        # Start with a copy
        self.clean_df = self.df.copy()
        
        # Convert numeric columns to proper data types first
        numeric_keywords = ['eui', 'energy', 'emissions', 'ghg', 'area', 'gfa', 'sqft', 'ft²', 'year']
        numeric_cols_to_convert = []
        
        for col in self.clean_df.columns:
            if any(keyword in col.lower() for keyword in numeric_keywords):
                numeric_cols_to_convert.append(col)
        
        print(f"Converting {len(numeric_cols_to_convert)} columns to numeric...")
        for col in numeric_cols_to_convert:
            if col in self.clean_df.columns:
                original_count = self.clean_df[col].count()
                self.clean_df[col] = pd.to_numeric(self.clean_df[col], errors='coerce')
                new_count = self.clean_df[col].count()
                if original_count != new_count:
                    print(f"  {col}: {original_count - new_count} values converted to NaN")
        
        # Clean specific columns if they exist
        if 'site_eui' in column_mapping:
            eui_col = column_mapping['site_eui']
            print(f"\nCleaning EUI column: {eui_col}")
            
            # Check data after conversion
            total_values = len(self.clean_df)
            non_null_values = self.clean_df[eui_col].count()
            print(f"  Total rows: {total_values:,}")
            print(f"  Non-null EUI values: {non_null_values:,}")
            
            if non_null_values > 0:
                # Remove zero and negative values
                initial_count = len(self.clean_df)
                self.clean_df = self.clean_df[
                    (self.clean_df[eui_col].notna()) & 
                    (self.clean_df[eui_col] > 0)
                ]
                print(f"  Removed {initial_count - len(self.clean_df)} rows with invalid EUI values")
                
                if len(self.clean_df) > 0:
                    # Remove extreme outliers (beyond 99th percentile)
                    q99 = self.clean_df[eui_col].quantile(0.99)
                    outliers = len(self.clean_df[self.clean_df[eui_col] > q99])
                    self.clean_df = self.clean_df[self.clean_df[eui_col] <= q99]
                    print(f"  Removed {outliers} extreme outliers (>99th percentile)")
            else:
                print(f"  ❌ No valid EUI values found in {eui_col}")
        
        # Use the correct area column - fix the column mapping
        area_candidates = [col for col in self.clean_df.columns if 'property gfa - self-reported' in col.lower()]
        if not area_candidates:
            area_candidates = [col for col in self.clean_df.columns if 'property gfa' in col.lower() and 'calculated' in col.lower()]
        
        if area_candidates:
            area_col = area_candidates[0]
            print(f"\nUsing area column: {area_col}")
            column_mapping['property_area'] = area_col  # Update mapping
            
            if area_col in self.clean_df.columns:
                area_count = self.clean_df[area_col].count()
                print(f"  Non-null area values: {area_count:,}")
                
                if area_count > 0:
                    # Remove buildings with zero or very small area
                    initial_count = len(self.clean_df)
                    self.clean_df = self.clean_df[
                        (self.clean_df[area_col].notna()) & 
                        (self.clean_df[area_col] > 1000)
                    ]  # Minimum 1000 sq ft
                    print(f"  Removed {initial_count - len(self.clean_df)} rows with invalid building area")
        else:
            print("  ❌ No suitable area column found")
        
        print(f"\nFinal cleaned dataset: {self.clean_df.shape[0]} rows")
        
        # Store column mapping for later use
        self.column_mapping = column_mapping
    
    def energy_consumption_analysis(self):
        """Analyze energy consumption patterns"""
        if self.clean_df is None:
            print("❌ No cleaned data available. Run clean_data() first.")
            return
        
        print("\n=== ENERGY CONSUMPTION ANALYSIS ===")
        
        col_map = getattr(self, 'column_mapping', {})
        
        if 'site_eui' not in col_map:
            print("❌ Site EUI column not found")
            return
        
        eui_col = col_map['site_eui']
        
        # Basic statistics
        print(f"\nSite EUI Statistics ({eui_col}):")
        print(f"  Mean: {self.clean_df[eui_col].mean():.2f} kBtu/ft²")
        print(f"  Median: {self.clean_df[eui_col].median():.2f} kBtu/ft²")
        print(f"  Std Dev: {self.clean_df[eui_col].std():.2f} kBtu/ft²")
        print(f"  Min: {self.clean_df[eui_col].min():.2f} kBtu/ft²")
        print(f"  Max: {self.clean_df[eui_col].max():.2f} kBtu/ft²")
        
        # Create visualizations
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Distribution
        axes[0, 0].hist(self.clean_df[eui_col], bins=50, alpha=0.7, color='skyblue', edgecolor='black')
        axes[0, 0].axvline(self.clean_df[eui_col].mean(), color='red', linestyle='--', label='Mean')
        axes[0, 0].axvline(self.clean_df[eui_col].median(), color='green', linestyle='--', label='Median')
        axes[0, 0].set_xlabel('Site EUI (kBtu/ft²)')
        axes[0, 0].set_ylabel('Frequency')
        axes[0, 0].set_title('Distribution of Site Energy Use Intensity')
        axes[0, 0].legend()
        
        # Box plot
        axes[0, 1].boxplot(self.clean_df[eui_col])
        axes[0, 1].set_ylabel('Site EUI (kBtu/ft²)')
        axes[0, 1].set_title('Site EUI Box Plot')
        
        # Cumulative distribution
        sorted_eui = np.sort(self.clean_df[eui_col])
        cumulative = np.arange(1, len(sorted_eui) + 1) / len(sorted_eui)
        axes[1, 0].plot(sorted_eui, cumulative, linewidth=2)
        axes[1, 0].set_xlabel('Site EUI (kBtu/ft²)')
        axes[1, 0].set_ylabel('Cumulative Probability')
        axes[1, 0].set_title('Cumulative Distribution of Site EUI')
        axes[1, 0].grid(True, alpha=0.3)
        
        # Energy efficiency categories
        percentiles = [25, 50, 75, 90]
        thresholds = [self.clean_df[eui_col].quantile(p/100) for p in percentiles]
        
        categories = []
        for eui in self.clean_df[eui_col]:
            if eui <= thresholds[0]:
                categories.append('Excellent (Top 25%)')
            elif eui <= thresholds[1]:
                categories.append('Good (25-50%)')
            elif eui <= thresholds[2]:
                categories.append('Fair (50-75%)')
            elif eui <= thresholds[3]:
                categories.append('Poor (75-90%)')
            else:
                categories.append('Very Poor (Bottom 10%)')
        
        category_counts = pd.Series(categories).value_counts()
        axes[1, 1].pie(category_counts.values, labels=category_counts.index, autopct='%1.1f%%')
        axes[1, 1].set_title('Building Energy Efficiency Categories')
        
        plt.tight_layout()
        plt.savefig('outputs/figures/energy_consumption_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        return thresholds
    
    def building_type_analysis(self):
        """Analyze energy patterns by building type"""
        if self.clean_df is None:
            return
        
        print("\n=== BUILDING TYPE ANALYSIS ===")
        
        col_map = getattr(self, 'column_mapping', {})
        
        if 'property_type' not in col_map or 'site_eui' not in col_map:
            print("❌ Required columns not found")
            return
        
        type_col = col_map['property_type']
        eui_col = col_map['site_eui']
        
        # Building type distribution
        type_counts = self.clean_df[type_col].value_counts()
        print(f"\nBuilding Types (Top 10):")
        print(type_counts.head(10))
        
        # Average EUI by building type
        avg_eui_by_type = self.clean_df.groupby(type_col)[eui_col].agg(['mean', 'median', 'count']).round(2)
        avg_eui_by_type = avg_eui_by_type.sort_values('mean', ascending=False)
        
        print(f"\nAverage EUI by Building Type (Top 15):")
        print(avg_eui_by_type.head(15))
        
        # Visualizations
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        # Building type distribution
        top_types = type_counts.head(15)
        axes[0, 0].barh(range(len(top_types)), top_types.values)
        axes[0, 0].set_yticks(range(len(top_types)))
        axes[0, 0].set_yticklabels([t[:30] + '...' if len(t) > 30 else t for t in top_types.index])
        axes[0, 0].set_xlabel('Number of Buildings')
        axes[0, 0].set_title('Top 15 Building Types by Count')
        axes[0, 0].invert_yaxis()
        
        # Average EUI by type
        top_eui_types = avg_eui_by_type.head(15)
        axes[0, 1].barh(range(len(top_eui_types)), top_eui_types['mean'])
        axes[0, 1].set_yticks(range(len(top_eui_types)))
        axes[0, 1].set_yticklabels([t[:30] + '...' if len(t) > 30 else t for t in top_eui_types.index])
        axes[0, 1].set_xlabel('Average Site EUI (kBtu/ft²)')
        axes[0, 1].set_title('Top 15 Building Types by Average EUI')
        axes[0, 1].invert_yaxis()
        
        # Box plot for major building types
        major_types = type_counts.head(8).index
        major_data = [self.clean_df[self.clean_df[type_col] == btype][eui_col].values 
                     for btype in major_types]
        
        bp = axes[1, 0].boxplot(major_data, labels=[t[:15] + '...' if len(t) > 15 else t 
                                                   for t in major_types])
        axes[1, 0].set_xticklabels(axes[1, 0].get_xticklabels(), rotation=45, ha='right')
        axes[1, 0].set_ylabel('Site EUI (kBtu/ft²)')
        axes[1, 0].set_title('EUI Distribution by Major Building Types')
        
        # Scatter plot: Building area vs EUI by type
        if 'property_area' in col_map:
            area_col = col_map['property_area']
            for i, btype in enumerate(major_types[:5]):  # Top 5 types
                data = self.clean_df[self.clean_df[type_col] == btype]
                axes[1, 1].scatter(data[area_col], data[eui_col], alpha=0.6, 
                                 label=btype[:20] + '...' if len(btype) > 20 else btype)
            
            axes[1, 1].set_xlabel('Building Area (sq ft)')
            axes[1, 1].set_ylabel('Site EUI (kBtu/ft²)')
            axes[1, 1].set_title('Building Area vs EUI by Type')
            axes[1, 1].legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            axes[1, 1].set_xscale('log')
        
        plt.tight_layout()
        plt.savefig('outputs/figures/building_type_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def efficiency_opportunities(self):
        """Identify energy efficiency opportunities and potential savings"""
        if self.clean_df is None:
            return
        
        print("\n=== EFFICIENCY OPPORTUNITIES ANALYSIS ===")
        
        col_map = getattr(self, 'column_mapping', {})
        
        if 'site_eui' not in col_map:
            return
        
        eui_col = col_map['site_eui']
        area_col = col_map.get('property_area', None)
        type_col = col_map.get('property_type', None)
        
        # Calculate efficiency benchmarks
        overall_median = self.clean_df[eui_col].median()
        overall_25th = self.clean_df[eui_col].quantile(0.25)
        
        print(f"Overall EUI Benchmarks:")
        print(f"  25th percentile (efficient): {overall_25th:.2f} kBtu/ft²")
        print(f"  Median: {overall_median:.2f} kBtu/ft²")
        
        # Identify underperforming buildings
        underperformers = self.clean_df[self.clean_df[eui_col] > overall_median].copy()
        
        # Calculate potential savings if buildings improved to median efficiency
        underperformers['excess_eui'] = underperformers[eui_col] - overall_median
        
        if area_col:
            underperformers['annual_excess_kbtu'] = (underperformers['excess_eui'] * 
                                                   underperformers[area_col])
            
            # Estimate cost savings (assuming $15 per MMBtu = $0.015 per kBtu)
            cost_per_kbtu = 0.015
            underperformers['potential_annual_savings'] = (underperformers['annual_excess_kbtu'] * 
                                                         cost_per_kbtu)
            
            total_potential_savings = underperformers['potential_annual_savings'].sum()
            total_excess_energy = underperformers['annual_excess_kbtu'].sum()
            
            print(f"\nEfficiency Opportunity Summary:")
            print(f"  Underperforming buildings: {len(underperformers):,}")
            print(f"  Total excess energy use: {total_excess_energy/1e6:.1f} million kBtu/year")
            print(f"  Total potential savings: ${total_potential_savings/1e6:.1f} million/year")
            print(f"  Average savings per building: ${underperformers['potential_annual_savings'].mean():,.0f}/year")
        
        # Savings by building type
        if type_col and area_col:
            type_savings = (underperformers.groupby(type_col)
                          .agg({
                              'potential_annual_savings': ['sum', 'mean', 'count'],
                              'excess_eui': 'mean'
                          }).round(2))
            
            type_savings.columns = ['Total_Savings', 'Avg_Savings_Per_Building', 'Count', 'Avg_Excess_EUI']
            type_savings = type_savings.sort_values('Total_Savings', ascending=False)
            
            print(f"\nTop 10 Building Types by Total Potential Savings:")
            print(type_savings.head(10))
        
        # Create visualizations
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        # Efficiency distribution
        efficiency_bins = [0, overall_25th, overall_median, 
                          self.clean_df[eui_col].quantile(0.75), 
                          self.clean_df[eui_col].max()]
        efficiency_labels = ['Excellent', 'Good', 'Fair', 'Poor']
        
        self.clean_df['efficiency_category'] = pd.cut(self.clean_df[eui_col], 
                                                    bins=efficiency_bins, 
                                                    labels=efficiency_labels)
        
        efficiency_counts = self.clean_df['efficiency_category'].value_counts()
        axes[0, 0].pie(efficiency_counts.values, labels=efficiency_counts.index, 
                      autopct='%1.1f%%', startangle=90)
        axes[0, 0].set_title('Building Efficiency Distribution')
        
        # Potential savings distribution
        if area_col:
            axes[0, 1].hist(underperformers['potential_annual_savings'], bins=50, 
                           alpha=0.7, color='orange', edgecolor='black')
            axes[0, 1].set_xlabel('Potential Annual Savings ($)')
            axes[0, 1].set_ylabel('Number of Buildings')
            axes[0, 1].set_title('Distribution of Potential Savings')
            axes[0, 1].axvline(underperformers['potential_annual_savings'].median(), 
                              color='red', linestyle='--', label='Median')
            axes[0, 1].legend()
        
        # EUI vs building size for underperformers
        if area_col:
            scatter = axes[1, 0].scatter(underperformers[area_col], underperformers[eui_col], 
                                       c=underperformers['potential_annual_savings'], 
                                       cmap='Reds', alpha=0.6)
            axes[1, 0].set_xlabel('Building Area (sq ft)')
            axes[1, 0].set_ylabel('Site EUI (kBtu/ft²)')
            axes[1, 0].set_title('Underperforming Buildings: Size vs EUI\n(Color = Potential Savings)')
            axes[1, 0].set_xscale('log')
            plt.colorbar(scatter, ax=axes[1, 0], label='Potential Savings ($)')
        
        # Top building types by total savings potential
        if type_col and area_col:
            top_savings_types = type_savings.head(10)
            axes[1, 1].barh(range(len(top_savings_types)), top_savings_types['Total_Savings'])
            axes[1, 1].set_yticks(range(len(top_savings_types)))
            axes[1, 1].set_yticklabels([t[:25] + '...' if len(t) > 25 else t 
                                       for t in top_savings_types.index])
            axes[1, 1].set_xlabel('Total Potential Savings ($)')
            axes[1, 1].set_title('Top 10 Building Types by Savings Potential')
            axes[1, 1].invert_yaxis()
        
        plt.tight_layout()
        plt.savefig('outputs/figures/efficiency_opportunities.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        return underperformers
    
    def generate_summary_report(self):
        """Generate a comprehensive summary report"""
        if self.clean_df is None:
            return
        
        print("\n" + "="*60)
        print("NYC BUILDING ENERGY EFFICIENCY ANALYSIS - SUMMARY REPORT")
        print("="*60)
        
        col_map = getattr(self, 'column_mapping', {})
        eui_col = col_map.get('site_eui', None)
        
        if eui_col:
            # Key statistics
            print(f"\nDATASET SUMMARY:")
            print(f"  Total buildings analyzed: {len(self.clean_df):,}")
            print(f"  Average energy intensity: {self.clean_df[eui_col].mean():.1f} kBtu/ft²")
            print(f"  Median energy intensity: {self.clean_df[eui_col].median():.1f} kBtu/ft²")
            print(f"  Energy intensity range: {self.clean_df[eui_col].min():.1f} - {self.clean_df[eui_col].max():.1f} kBtu/ft²")
            
            # Efficiency categories
            q25 = self.clean_df[eui_col].quantile(0.25)
            q75 = self.clean_df[eui_col].quantile(0.75)
            
            efficient_buildings = len(self.clean_df[self.clean_df[eui_col] <= q25])
            inefficient_buildings = len(self.clean_df[self.clean_df[eui_col] >= q75])
            
            print(f"\nEFFICIENCY BREAKDOWN:")
            print(f"  Highly efficient buildings (top 25%): {efficient_buildings:,} ({efficient_buildings/len(self.clean_df)*100:.1f}%)")
            print(f"  Poor performing buildings (bottom 25%): {inefficient_buildings:,} ({inefficient_buildings/len(self.clean_df)*100:.1f}%)")
            
            # Building types
            if 'property_type' in col_map:
                type_col = col_map['property_type']
                most_common_type = self.clean_df[type_col].mode()[0]
                type_count = len(self.clean_df[type_col].unique())
                
                print(f"\nBUILDING TYPES:")
                print(f"  Total building types: {type_count}")
                print(f"  Most common type: {most_common_type}")
                print(f"  Buildings of most common type: {len(self.clean_df[self.clean_df[type_col] == most_common_type]):,}")
        
        print(f"\nFILES GENERATED:")
        print(f"  - outputs/figures/missing_values_analysis.png")
        print(f"  - outputs/figures/energy_consumption_analysis.png")
        print(f"  - outputs/figures/building_type_analysis.png")
        print(f"  - outputs/figures/efficiency_opportunities.png")
        
        print(f"\nNEXT STEPS:")
        print(f"  1. Review generated visualizations")
        print(f"  2. Focus on building types with highest savings potential")
        print(f"  3. Develop HVAC optimization recommendations")
        print(f"  4. Create predictive models for energy efficiency")
        
        print("="*60)

def main():
    """Main execution function"""
    
    # Initialize explorer
    explorer = NYCEnergyExplorer('data/raw/nyc_energy_2022.csv')
    
    # Load and explore data
    if explorer.load_data():
        explorer.basic_info()
        explorer.missing_values_analysis()
        
        # Clean data
        explorer.clean_data()
        
        # Perform analyses
        explorer.energy_consumption_analysis()
        explorer.building_type_analysis()
        explorer.efficiency_opportunities()
        
        # Generate summary
        explorer.generate_summary_report()
    
    print("\n✅ Data exploration complete!")

if __name__ == "__main__":
    main()