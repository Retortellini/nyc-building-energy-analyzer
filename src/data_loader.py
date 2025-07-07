"""
Data loading and initial processing for NYC Building Energy Analysis
"""

import pandas as pd
import numpy as np
from pathlib import Path

def load_nyc_energy_data(file_path: str) -> pd.DataFrame:
    """
    Load NYC building energy data from CSV file
    
    Args:
        file_path (str): Path to the CSV file
        
    Returns:
        pd.DataFrame: Loaded and initially cleaned data
    """
    try:
        df = pd.read_csv(file_path)
        print(f"Successfully loaded {len(df)} records")
        print(f"Columns: {list(df.columns)}")
        return df
    except FileNotFoundError:
        print(f"File not found: {file_path}")
        return pd.DataFrame()
    except Exception as e:
        print(f"Error loading data: {e}")
        return pd.DataFrame()

def get_data_info(df: pd.DataFrame) -> dict:
    """
    Get basic information about the dataset
    
    Args:
        df (pd.DataFrame): The dataset
        
    Returns:
        dict: Summary statistics and info
    """
    info = {
        'shape': df.shape,
        'missing_values': df.isnull().sum().to_dict(),
        'data_types': df.dtypes.to_dict(),
        'numeric_columns': df.select_dtypes(include=[np.number]).columns.tolist(),
        'categorical_columns': df.select_dtypes(include=['object']).columns.tolist()
    }
    return info

if __name__ == "__main__":
    # Test the functions
    print("Data loader module - ready for use!")