"""
Test script to verify VSCode and Python setup
"""

import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

def test_imports():
    """Test that all required packages are installed"""
    packages = [
        'pandas', 'numpy', 'matplotlib', 'seaborn', 
        'plotly', 'streamlit', 'sklearn', 'requests'
    ]
    
    print("Testing package imports...")
    for package in packages:
        try:
            __import__(package)
            print(f"✅ {package} - OK")
        except ImportError:
            print(f"❌ {package} - FAILED")

def test_project_structure():
    """Test that project folders exist"""
    required_folders = [
        'data/raw', 'data/processed', 'src', 'notebooks', 
        'outputs/figures', 'outputs/reports', 'tests', 'docs'
    ]
    
    print("\nTesting project structure...")
    for folder in required_folders:
        if Path(folder).exists():
            print(f"✅ {folder} - OK")
        else:
            print(f"❌ {folder} - MISSING")

def test_data_creation():
    """Test creating and saving sample data"""
    print("\nTesting data operations...")
    
    # Create sample data
    sample_data = pd.DataFrame({
        'building_id': range(1, 101),
        'energy_use': np.random.normal(100, 20, 100),
        'building_type': np.random.choice(['Office', 'Residential', 'Retail'], 100)
    })
    
    # Save to outputs
    output_path = Path('outputs/test_data.csv')
    sample_data.to_csv(output_path, index=False)
    
    if output_path.exists():
        print("✅ Data creation and saving - OK")
    else:
        print("❌ Data creation and saving - FAILED")

def main():
    print("=== VSCode Setup Test ===")
    print(f"Python version: {sys.version}")
    print(f"Current working directory: {Path.cwd()}")
    
    test_imports()
    test_project_structure()
    test_data_creation()
    
    print("\n=== Setup Test Complete ===")

if __name__ == "__main__":
    main()