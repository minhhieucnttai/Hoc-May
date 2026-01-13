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
    
    # Use existing severity_index if available
    if 'severity_index' not in df_fe.columns:
        # Normalize the impact factors
        scaler = StandardScaler()
        
        impact_features = ['casualties', 'economic_loss_usd']
        available_features = [f for f in impact_features if f in df_fe.columns]
        
        if available_features:
            df_normalized = pd.DataFrame(
                scaler.fit_transform(df_fe[available_features]),
                columns=[f + '_normalized' for f in available_features],
                index=df_fe.index
            )
            
            # Calculate severity index as weighted average
            weights = {
                'casualties_normalized': 0.5,
                'economic_loss_usd_normalized': 0.5
            }
            
            df_fe['severity_index_calc'] = sum(
                df_normalized[col] * weight 
                for col, weight in weights.items() 
                if col in df_normalized.columns
            )
            
            # Normalize to 0-10 scale
            min_val = df_fe['severity_index_calc'].min()
            max_val = df_fe['severity_index_calc'].max()
            
            if max_val != min_val:
                df_fe['severity_index_calc'] = (
                    (df_fe['severity_index_calc'] - min_val) / 
                    (max_val - min_val) * 10
                )
            else:
                df_fe['severity_index_calc'] = 5.0
            
            print("Severity index calculated (scale 0-10)")
    else:
        print("Severity index already exists in dataset")
    
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
    
    # Response efficiency score categories
    if 'response_efficiency_score' in df_fe.columns:
        df_fe['response_quality'] = pd.cut(
            df_fe['response_efficiency_score'],
            bins=[0, 25, 50, 75, 100],
            labels=['Poor', 'Fair', 'Good', 'Excellent']
        )
        print("Response quality categories created")
    
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
    
    # Casualty rate (casualties per response hour)
    if 'casualties' in df_fe.columns and 'response_time_hours' in df_fe.columns:
        df_fe['casualty_per_hour'] = df_fe['casualties'] / (df_fe['response_time_hours'] + 1)
        print("Casualty per hour rate created")
    
    # Economic impact per casualty
    if 'economic_loss_usd' in df_fe.columns and 'casualties' in df_fe.columns:
        df_fe['economic_impact_per_casualty'] = df_fe['economic_loss_usd'] / (df_fe['casualties'] + 1)
        print("Economic impact per casualty created")
    
    # Aid to economic loss ratio
    if 'aid_amount_usd' in df_fe.columns and 'economic_loss_usd' in df_fe.columns:
        df_fe['aid_coverage_ratio'] = df_fe['aid_amount_usd'] / (df_fe['economic_loss_usd'] + 1)
        print("Aid coverage ratio created")
    
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
    
    # Average response time by country
    if 'country' in df_fe.columns and 'response_time_hours' in df_fe.columns:
        country_avg_response = df_fe.groupby('country')['response_time_hours'].transform('mean')
        df_fe['avg_response_time_by_country'] = country_avg_response
        print("Average response time by country created")
    
    # Average efficiency by disaster type
    if 'disaster_type' in df_fe.columns and 'response_efficiency_score' in df_fe.columns:
        disaster_avg_efficiency = df_fe.groupby('disaster_type')['response_efficiency_score'].transform('mean')
        df_fe['avg_efficiency_by_type'] = disaster_avg_efficiency
        print("Average efficiency by disaster type created")
    
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
