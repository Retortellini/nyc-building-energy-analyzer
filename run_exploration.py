"""
Quick runner script for NYC Building Energy Data Exploration
Run this after downloading the NYC energy data
"""

import sys
from pathlib import Path

# Add src to path
sys.path.append('src')

def check_setup():
    """Check if everything is set up correctly"""
    print("🔍 Checking project setup...")
    
    # Check data file
    data_file = Path('data/raw/nyc_energy_2022.csv')
    if not data_file.exists():
        print("❌ Data file not found!")
        print("Please download NYC energy data from:")
        print("https://data.cityofnewyork.us/Environment/Energy-and-Water-Data-Disclosure-for-Local-Law-97-/usc3-8zwd")
        print(f"Save as: {data_file}")
        return False
    
    # Check output directories
    output_dirs = ['outputs/figures', 'data/processed']
    for dir_path in output_dirs:
        Path(dir_path).mkdir(parents=True, exist_ok=True)
    
    print("✅ Setup check complete!")
    return True

def run_basic_exploration():
    """Run basic data exploration"""
    print("\n🚀 Running basic data exploration...")
    
    try:
        from data_exploration import NYCEnergyExplorer
        
        # Initialize explorer
        explorer = NYCEnergyExplorer('data/raw/nyc_energy_2022.csv')
        
        # Run exploration
        if explorer.load_data():
            print("📊 Starting analysis...")
            explorer.basic_info()
            explorer.missing_values_analysis()
            explorer.clean_data()
            explorer.energy_consumption_analysis()
            explorer.building_type_analysis()
            explorer.efficiency_opportunities()
            explorer.generate_summary_report()
            
            print("\n✅ Basic exploration complete!")
            print("📁 Check outputs/figures/ for visualizations")
            
        else:
            print("❌ Failed to load data")
            
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("Make sure you're in the project root directory")
    except Exception as e:
        print(f"❌ Error during exploration: {e}")

def run_notebook_exploration():
    """Launch Jupyter notebook for interactive exploration"""
    print("\n📓 Launching Jupyter notebook for interactive exploration...")
    
    try:
        import subprocess
        subprocess.run(['jupyter', 'notebook', 'notebooks/01_data_exploration.ipynb'])
    except FileNotFoundError:
        print("❌ Jupyter not found. Install with: pip install jupyter")
    except Exception as e:
        print(f"❌ Error launching notebook: {e}")

def main():
    """Main execution"""
    print("NYC Building Energy Analysis - Data Exploration")
    print("=" * 50)
    
    if not check_setup():
        return
    
    print("\nChoose exploration method:")
    print("1. Run automated analysis (Python script)")
    print("2. Launch interactive notebook (Jupyter)")
    print("3. Both")
    
    choice = input("\nEnter choice (1, 2, or 3): ").strip()
    
    if choice in ['1', '3']:
        run_basic_exploration()
    
    if choice in ['2', '3']:
        run_notebook_exploration()
    
    if choice not in ['1', '2', '3']:
        print("Invalid choice. Running automated analysis...")
        run_basic_exploration()

if __name__ == "__main__":
    main()
