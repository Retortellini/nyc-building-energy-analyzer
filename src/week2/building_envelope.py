"""
Building Envelope Performance Analysis
Focuses on thermal performance characteristics that impact HVAC loads
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

class EnvelopeAnalyzer:
    """
    Building envelope thermal performance analyzer
    
    Engineering Focus:
    - Heat transfer analysis through building envelope
    - Thermal bridging and insulation effectiveness
    - Window performance and solar heat gain
    - Air infiltration impact on HVAC loads
    """
    
    def __init__(self, hvac_data: pd.DataFrame):
        """Initialize with HVAC analysis results"""
        self.df = hvac_data.copy()
        self.envelope_metrics = {}
    
    def calculate_thermal_performance_indicators(self):
        """
        Calculate indicators of building envelope thermal performance
        
        Engineering Basis:
        - Heat loss coefficient estimation from energy patterns
        - Thermal mass effects on energy consumption
        - Building orientation and solar gain analysis
        """
        print("=== BUILDING ENVELOPE THERMAL ANALYSIS ===")
        
        # Find the site EUI column
        site_eui_col = None
        for col in self.df.columns:
            if 'site eui' in col.lower() and 'weather' not in col.lower():
                site_eui_col = col
                break
        
        if not site_eui_col:
            print("❌ Site EUI column not found")
            return self.df
        
        # Convert to numeric and handle weather sensitivity
        self.df[site_eui_col] = pd.to_numeric(self.df[site_eui_col], errors='coerce')
        
        if 'weather_sensitivity' in self.df.columns:
            self.df['weather_sensitivity'] = pd.to_numeric(self.df['weather_sensitivity'], errors='coerce')
            
            # Thermal Performance Index
            # High weather sensitivity + high EUI = poor envelope
            site_eui_median = self.df[site_eui_col].median()
            thermal_performance = (
                self.df['weather_sensitivity'].fillna(0) * 
                (self.df[site_eui_col].fillna(site_eui_median) / site_eui_median)
            )
            
            self.df['thermal_performance_index'] = thermal_performance
        
        # Building shape factor (proxy from area - more complex shapes have higher heat loss)
        # Assume square buildings for approximation
        area_cols = [col for col in self.df.columns if 'property gfa' in col.lower() and 'self-reported' in col.lower()]
        if area_cols:
            area_col = area_cols[0]
            self.df[area_col] = pd.to_numeric(self.df[area_col], errors='coerce')
            area = self.df[area_col].fillna(50000)  # Default reasonable area
            
            # Perimeter to area ratio (higher = more heat loss)
            perimeter_area_ratio = 4 * np.sqrt(area) / area
            self.df['envelope_exposure_factor'] = perimeter_area_ratio * 10000  # Scale for readability
        
        print(f"✅ Thermal performance indicators calculated")
        return self.df
    
    def identify_envelope_upgrade_candidates(self):
        """
        Identify buildings most likely to benefit from envelope upgrades
        
        Criteria:
        - High thermal performance index
        - Old building vintage
        - High weather sensitivity
        - Cost-effective upgrade potential
        """
        # Score buildings for envelope upgrade priority
        upgrade_score = 0
        
        # Age factor (older = higher priority)
        if 'building_vintage' in self.df.columns:
            vintage_scores = {
                'Pre-Efficiency Era': 4,
                'Early Efficiency': 3,
                'Modern Standards': 2,
                'High Performance': 1,
                'Unknown': 2
            }
            age_score = self.df['building_vintage'].map(vintage_scores).fillna(2)
            upgrade_score += age_score
        
        # Performance factor
        thermal_score = pd.qcut(self.df['thermal_performance_index'], 5, 
                               labels=[1, 2, 3, 4, 5]).astype(float)
        upgrade_score += thermal_score
        
        # Size factor (larger buildings = higher total savings)
        if 'Property GFA - Self-Reported (ft²)' in self.df.columns:
            size_score = pd.qcut(self.df['Property GFA - Self-Reported (ft²)'], 3,
                                labels=[1, 2, 3]).astype(float)
            upgrade_score += size_score
        
        self.df['envelope_upgrade_priority'] = upgrade_score
        
        # Identify top candidates
        top_candidates = self.df[self.df['envelope_upgrade_priority'] >= 7]
        
        print(f"\n🏗️ ENVELOPE UPGRADE CANDIDATES:")
        print(f"   High priority buildings: {len(top_candidates):,}")
        
        if len(top_candidates) > 0:
            avg_eui = top_candidates['Site EUI (kBtu/ft²)'].mean()
            avg_sensitivity = top_candidates['weather_sensitivity'].mean()
            
            print(f"   Average EUI: {avg_eui:.1f} kBtu/ft²")
            print(f"   Average weather sensitivity: {avg_sensitivity:.1f}%")
            
            # Estimate upgrade potential
            efficient_baseline = self.df['Site EUI (kBtu/ft²)'].quantile(0.25)
            potential_reduction = avg_eui - efficient_baseline
            
            if potential_reduction > 0:
                avg_area = top_candidates['Property GFA - Self-Reported (ft²)'].median()
                annual_savings = potential_reduction * avg_area * 0.015
                total_savings = annual_savings * len(top_candidates)
                
                print(f"   Potential EUI reduction: {potential_reduction:.1f} kBtu/ft²")
                print(f"   Estimated savings: ${annual_savings:,.0f}/building/year")
                print(f"   Total opportunity: ${total_savings/1e6:.1f}M/year")
        
        return top_candidates
    
    def analyze_vintage_performance_gaps(self):
        """
        Quantify performance gaps between building vintages
        Focus on envelope-related energy consumption patterns
        """
        if 'building_vintage' not in self.df.columns:
            print("❌ Building vintage data not available")
            return None
        
        print(f"\n🏢 VINTAGE PERFORMANCE GAP ANALYSIS:")
        
        vintage_analysis = self.df.groupby('building_vintage').agg({
            'Site EUI (kBtu/ft²)': ['mean', 'median', 'std', 'count'],
            'weather_sensitivity': ['mean', 'median'],
            'thermal_performance_index': 'mean',
            'Property GFA - Self-Reported (ft²)': 'median'
        }).round(2)
        
        # Flatten column names
        vintage_analysis.columns = [
            'EUI_Mean', 'EUI_Median', 'EUI_Std', 'Building_Count',
            'Weather_Sens_Mean', 'Weather_Sens_Median', 
            'Thermal_Index', 'Median_Area'
        ]
        
        # Calculate modernization potential
        best_vintage_eui = vintage_analysis['EUI_Median'].min()
        vintage_analysis['Modernization_Potential'] = (
            vintage_analysis['EUI_Median'] - best_vintage_eui
        )
        
        # Estimate costs and savings for each vintage
        vintage_analysis['Annual_Savings_Potential'] = (
            vintage_analysis['Modernization_Potential'] * 
            vintage_analysis['Median_Area'] * 0.015
        )
        
        vintage_analysis['Total_Vintage_Opportunity'] = (
            vintage_analysis['Annual_Savings_Potential'] * 
            vintage_analysis['Building_Count'] / 1e6
        )
        
        print("Vintage Performance Summary:")
        display_cols = ['EUI_Median', 'Weather_Sens_Mean', 'Building_Count', 
                       'Modernization_Potential', 'Total_Vintage_Opportunity']
        print(vintage_analysis[display_cols])
        
        return vintage_analysis

def main():
    """Demonstration of building envelope analysis"""
    try:
        # Load HVAC analysis results
        hvac_data = pd.read_csv('data/processed/hvac_analysis_results.csv')
        
        analyzer = EnvelopeAnalyzer(hvac_data)
        analyzer.calculate_thermal_performance_indicators()
        analyzer.identify_envelope_upgrade_candidates()
        analyzer.analyze_vintage_performance_gaps()
        
    except FileNotFoundError:
        print("❌ HVAC analysis results not found. Run HVAC analysis first.")

if __name__ == "__main__":
    main()