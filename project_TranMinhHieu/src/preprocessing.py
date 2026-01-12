"""
Data Preprocessing Module
Handles data cleaning, missing value treatment, and initial data preparation.
"""

import pandas as pd
import numpy as np
from pathlib import Path


def load_data(filepath='data/global_disaster_response_2018_2024.csv'):
    """
    Load the disaster response dataset.
    
    Parameters:
    -----------
    filepath : str
        Path to the CSV file
        
    Returns:
    --------
    pd.DataFrame
        Loaded dataset
    """
    try:
        df = pd.read_csv(filepath)
        print(f"Data loaded successfully: {df.shape[0]} rows, {df.shape[1]} columns")
        return df
    except FileNotFoundError:
        print(f"Error: File not found at {filepath}")
        return None
    except Exception as e:
        print(f"Error loading data: {str(e)}")
        return None


def check_missing_values(df):
    """
    Check for missing values in the dataset.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe
        
    Returns:
    --------
    pd.DataFrame
        Summary of missing values
    """
    missing = pd.DataFrame({
        'Column': df.columns,
        'Missing_Count': df.isnull().sum().values,
        'Missing_Percentage': (df.isnull().sum().values / len(df) * 100).round(2)
    })
    missing = missing[missing['Missing_Count'] > 0].sort_values('Missing_Count', ascending=False)
    
    if len(missing) > 0:
        print("\nMissing Values Summary:")
        print(missing.to_string(index=False))
    else:
        print("\nNo missing values found in the dataset.")
    
    return missing


def handle_missing_values(df, strategy='mean'):
    """
    Handle missing values in the dataset.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe
    strategy : str
        Strategy for handling missing values ('mean', 'median', 'drop')
        
    Returns:
    --------
    pd.DataFrame
        Dataframe with handled missing values
    """
    df_clean = df.copy()
    
    numeric_cols = df_clean.select_dtypes(include=[np.number]).columns
    
    if strategy == 'mean':
        df_clean[numeric_cols] = df_clean[numeric_cols].fillna(df_clean[numeric_cols].mean())
    elif strategy == 'median':
        df_clean[numeric_cols] = df_clean[numeric_cols].fillna(df_clean[numeric_cols].median())
    elif strategy == 'drop':
        df_clean = df_clean.dropna()
    
    # Fill categorical missing values with mode or 'Unknown'
    categorical_cols = df_clean.select_dtypes(include=['object']).columns
    for col in categorical_cols:
        if df_clean[col].isnull().any():
            df_clean[col] = df_clean[col].fillna(df_clean[col].mode()[0] if not df_clean[col].mode().empty else 'Unknown')
    
    print(f"\nMissing values handled using '{strategy}' strategy")
    return df_clean


def convert_data_types(df):
    """
    Convert data types to appropriate formats.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe
        
    Returns:
    --------
    pd.DataFrame
        Dataframe with corrected data types
    """
    df_converted = df.copy()
    
    # Convert date column to datetime
    if 'date' in df_converted.columns:
        df_converted['date'] = pd.to_datetime(df_converted['date'], errors='coerce')
        print("Date column converted to datetime format")
    
    return df_converted


def remove_duplicates(df):
    """
    Remove duplicate rows from the dataset.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe
        
    Returns:
    --------
    pd.DataFrame
        Dataframe without duplicates
    """
    initial_rows = len(df)
    df_no_duplicates = df.drop_duplicates()
    duplicates_removed = initial_rows - len(df_no_duplicates)
    
    if duplicates_removed > 0:
        print(f"\nRemoved {duplicates_removed} duplicate rows")
    else:
        print("\nNo duplicate rows found")
    
    return df_no_duplicates


def preprocess_data(filepath='data/global_disaster_response_2018_2024.csv', save_output=True):
    """
    Complete preprocessing pipeline.
    
    Parameters:
    -----------
    filepath : str
        Path to the input CSV file
    save_output : bool
        Whether to save the preprocessed data
        
    Returns:
    --------
    pd.DataFrame
        Preprocessed dataframe
    """
    print("="*50)
    print("Starting Data Preprocessing")
    print("="*50)
    
    # Load data
    df = load_data(filepath)
    if df is None:
        return None
    
    # Check for missing values
    check_missing_values(df)
    
    # Handle missing values
    df = handle_missing_values(df, strategy='mean')
    
    # Convert data types
    df = convert_data_types(df)
    
    # Remove duplicates
    df = remove_duplicates(df)
    
    # Save preprocessed data
    if save_output:
        output_path = filepath.replace('.csv', '_preprocessed.csv')
        df.to_csv(output_path, index=False)
        print(f"\nPreprocessed data saved to: {output_path}")
    
    print("\n" + "="*50)
    print("Preprocessing Complete")
    print("="*50)
    
    return df


if __name__ == "__main__":
    # Run preprocessing
    df_preprocessed = preprocess_data()
    if df_preprocessed is not None:
        print(f"\nFinal dataset shape: {df_preprocessed.shape}")
        print("\nFirst few rows:")
        print(df_preprocessed.head())
