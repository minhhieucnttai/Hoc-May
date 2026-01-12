"""
Feature Engineering Module
Creates new features and prepares data for modeling.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler


def create_temporal_features(df):
    """
    Create temporal features from date column.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe with date column
        
    Returns:
    --------
    pd.DataFrame
        Dataframe with new temporal features
    """
    df_fe = df.copy()
    
    if 'date' in df_fe.columns:
        df_fe['date'] = pd.to_datetime(df_fe['date'])
        df_fe['year'] = df_fe['date'].dt.year
        df_fe['month'] = df_fe['date'].dt.month
        df_fe['quarter'] = df_fe['date'].dt.quarter
        df_fe['day_of_year'] = df_fe['date'].dt.dayofyear
        df_fe['season'] = df_fe['month'].apply(lambda x: 
            'Winter' if x in [12, 1, 2] else
            'Spring' if x in [3, 4, 5] else
            'Summer' if x in [6, 7, 8] else
            'Fall'
        )
        
        print("Temporal features created: year, month, quarter, day_of_year, season")
    
    return df_fe


def create_severity_index(df):
    """
    Create a severity index based on multiple impact factors.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe
        
    Returns:
    --------
    pd.DataFrame
        Dataframe with severity index
    """
    df_fe = df.copy()
    
    # Normalize the impact factors
    scaler = StandardScaler()
    
    impact_features = ['casualties', 'affected_population', 'economic_impact_usd']
    available_features = [f for f in impact_features if f in df_fe.columns]
    
    if available_features:
        df_normalized = pd.DataFrame(
            scaler.fit_transform(df_fe[available_features]),
            columns=[f + '_normalized' for f in available_features],
            index=df_fe.index
        )
        
        # Calculate severity index as weighted average
        weights = {
            'casualties_normalized': 0.4,
            'affected_population_normalized': 0.3,
            'economic_impact_usd_normalized': 0.3
        }
        
        df_fe['severity_index'] = sum(
            df_normalized[col] * weight 
            for col, weight in weights.items() 
            if col in df_normalized.columns
        )
        
        # Normalize to 0-10 scale
        df_fe['severity_index'] = (
            (df_fe['severity_index'] - df_fe['severity_index'].min()) / 
            (df_fe['severity_index'].max() - df_fe['severity_index'].min()) * 10
        )
        
        print("Severity index created (scale 0-10)")
    
    return df_fe


def create_response_features(df):
    """
    Create features related to response effectiveness.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe
        
    Returns:
    --------
    pd.DataFrame
        Dataframe with response features
    """
    df_fe = df.copy()
    
    # Response time categories
    if 'response_time_hours' in df_fe.columns:
        df_fe['response_speed'] = pd.cut(
            df_fe['response_time_hours'],
            bins=[0, 24, 48, 72, np.inf],
            labels=['Very Fast', 'Fast', 'Moderate', 'Slow']
        )
        print("Response speed categories created")
    
    # Response effectiveness categories
    if 'response_effectiveness' in df_fe.columns:
        df_fe['response_quality'] = pd.cut(
            df_fe['response_effectiveness'],
            bins=[0, 0.5, 0.7, 0.85, 1.0],
            labels=['Poor', 'Fair', 'Good', 'Excellent']
        )
        print("Response quality categories created")
    
    # Create efficiency metric (effectiveness / response_time)
    if 'response_effectiveness' in df_fe.columns and 'response_time_hours' in df_fe.columns:
        df_fe['response_efficiency'] = df_fe['response_effectiveness'] / (df_fe['response_time_hours'] + 1)
        print("Response efficiency metric created")
    
    return df_fe


def create_impact_ratios(df):
    """
    Create ratio features for impact analysis.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe
        
    Returns:
    --------
    pd.DataFrame
        Dataframe with impact ratios
    """
    df_fe = df.copy()
    
    # Casualty rate (casualties per affected population)
    if 'casualties' in df_fe.columns and 'affected_population' in df_fe.columns:
        df_fe['casualty_rate'] = df_fe['casualties'] / (df_fe['affected_population'] + 1)
        print("Casualty rate created")
    
    # Economic impact per person
    if 'economic_impact_usd' in df_fe.columns and 'affected_population' in df_fe.columns:
        df_fe['economic_impact_per_capita'] = df_fe['economic_impact_usd'] / (df_fe['affected_population'] + 1)
        print("Economic impact per capita created")
    
    # Economic impact per casualty
    if 'economic_impact_usd' in df_fe.columns and 'casualties' in df_fe.columns:
        df_fe['economic_impact_per_casualty'] = df_fe['economic_impact_usd'] / (df_fe['casualties'] + 1)
        print("Economic impact per casualty created")
    
    return df_fe


def encode_categorical_features(df, categorical_cols=None):
    """
    Encode categorical features using Label Encoding.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe
    categorical_cols : list, optional
        List of categorical columns to encode
        
    Returns:
    --------
    tuple
        (encoded dataframe, dictionary of label encoders)
    """
    df_fe = df.copy()
    encoders = {}
    
    if categorical_cols is None:
        categorical_cols = ['disaster_type', 'country', 'region', 'season', 
                          'response_speed', 'response_quality']
    
    available_cols = [col for col in categorical_cols if col in df_fe.columns]
    
    for col in available_cols:
        le = LabelEncoder()
        df_fe[col + '_encoded'] = le.fit_transform(df_fe[col].astype(str))
        encoders[col] = le
        print(f"Encoded {col}: {len(le.classes_)} unique values")
    
    return df_fe, encoders


def create_aggregated_features(df):
    """
    Create aggregated features based on groupings.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe
        
    Returns:
    --------
    pd.DataFrame
        Dataframe with aggregated features
    """
    df_fe = df.copy()
    
    # Average casualties by disaster type
    if 'disaster_type' in df_fe.columns and 'casualties' in df_fe.columns:
        disaster_avg_casualties = df_fe.groupby('disaster_type')['casualties'].transform('mean')
        df_fe['avg_casualties_by_type'] = disaster_avg_casualties
        print("Average casualties by disaster type created")
    
    # Average response time by region
    if 'region' in df_fe.columns and 'response_time_hours' in df_fe.columns:
        region_avg_response = df_fe.groupby('region')['response_time_hours'].transform('mean')
        df_fe['avg_response_time_by_region'] = region_avg_response
        print("Average response time by region created")
    
    # Average effectiveness by country
    if 'country' in df_fe.columns and 'response_effectiveness' in df_fe.columns:
        country_avg_effectiveness = df_fe.groupby('country')['response_effectiveness'].transform('mean')
        df_fe['avg_effectiveness_by_country'] = country_avg_effectiveness
        print("Average effectiveness by country created")
    
    return df_fe


def engineer_features(df, encode_categoricals=True):
    """
    Complete feature engineering pipeline.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe
    encode_categoricals : bool
        Whether to encode categorical features
        
    Returns:
    --------
    tuple
        (engineered dataframe, encoders dictionary)
    """
    print("\n" + "="*50)
    print("FEATURE ENGINEERING")
    print("="*50 + "\n")
    
    df_engineered = df.copy()
    
    # Create temporal features
    print("Creating temporal features...")
    df_engineered = create_temporal_features(df_engineered)
    
    # Create severity index
    print("\nCreating severity index...")
    df_engineered = create_severity_index(df_engineered)
    
    # Create response features
    print("\nCreating response features...")
    df_engineered = create_response_features(df_engineered)
    
    # Create impact ratios
    print("\nCreating impact ratios...")
    df_engineered = create_impact_ratios(df_engineered)
    
    # Create aggregated features
    print("\nCreating aggregated features...")
    df_engineered = create_aggregated_features(df_engineered)
    
    # Encode categorical features
    encoders = {}
    if encode_categoricals:
        print("\nEncoding categorical features...")
        df_engineered, encoders = encode_categorical_features(df_engineered)
    
    print("\n" + "="*50)
    print("FEATURE ENGINEERING COMPLETE")
    print("="*50)
    print(f"\nTotal features: {len(df_engineered.columns)}")
    
    return df_engineered, encoders


if __name__ == "__main__":
    # Load and preprocess data
    from preprocessing import load_data, preprocess_data
    
    df = load_data('data/global_disaster_response_2018_2024.csv')
    if df is not None:
        df = preprocess_data('data/global_disaster_response_2018_2024.csv', save_output=False)
        df_engineered, encoders = engineer_features(df)
        
        print("\nEngineered features:")
        print(df_engineered.columns.tolist())
        print("\nSample data:")
        print(df_engineered.head())
