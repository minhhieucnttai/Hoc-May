"""
Module Tạo và Chọn Đặc trưng (Feature Engineering)
==================================================
Chứa các hàm tạo đặc trưng mới và chọn đặc trưng quan trọng.

Các chức năng chính:
- Tạo các đặc trưng tương tác
- Tạo các tỷ lệ (ratio features)
- Chọn đặc trưng dựa trên importance
"""

import pandas as pd
import numpy as np
from typing import List, Optional


def create_ratio_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Tạo các đặc trưng tỷ lệ (ratio features).
    
    Các đặc trưng được tạo:
    - loss_per_casualty: Thiệt hại kinh tế trên mỗi ca thương vong
    - aid_per_hour: Số tiền viện trợ trên mỗi giờ phản ứng
    - severity_response_ratio: Tỷ lệ độ nghiêm trọng và thời gian phản ứng
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame đầu vào
        
    Returns:
    --------
    pd.DataFrame
        DataFrame với các đặc trưng mới
    """
    df = df.copy()
    
    # Loss per casualty
    if 'economic_loss_usd' in df.columns and 'casualties' in df.columns:
        df['loss_per_casualty'] = df['economic_loss_usd'] / (df['casualties'] + 1)
    
    # Aid per hour
    if 'aid_amount_usd' in df.columns and 'response_time_hours' in df.columns:
        df['aid_per_hour'] = df['aid_amount_usd'] / (df['response_time_hours'] + 1)
    
    # Severity response ratio
    if 'severity_index' in df.columns and 'response_time_hours' in df.columns:
        df['severity_response_ratio'] = df['severity_index'] / (df['response_time_hours'] + 1)
    
    return df


def create_interaction_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Tạo các đặc trưng tương tác giữa các biến.
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame đầu vào
        
    Returns:
    --------
    pd.DataFrame
        DataFrame với các đặc trưng tương tác
    """
    df = df.copy()
    
    # Severity x Economic loss
    if 'severity_index' in df.columns and 'economic_loss_usd' in df.columns:
        df['severity_x_loss'] = df['severity_index'] * np.log1p(df['economic_loss_usd'])
    
    # Response efficiency x Aid
    if 'response_efficiency_score' in df.columns and 'aid_amount_usd' in df.columns:
        df['efficiency_x_aid'] = df['response_efficiency_score'] * np.log1p(df['aid_amount_usd'])
    
    return df


def create_geo_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Tạo các đặc trưng địa lý.
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame đầu vào
        
    Returns:
    --------
    pd.DataFrame
        DataFrame với các đặc trưng địa lý
    """
    df = df.copy()
    
    if 'latitude' in df.columns and 'longitude' in df.columns:
        # Distance from equator
        df['distance_from_equator'] = np.abs(df['latitude'])
        
        # Hemisphere indicator
        df['is_northern_hemisphere'] = (df['latitude'] >= 0).astype(int)
    
    return df


def create_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Tạo các đặc trưng thời gian bổ sung.
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame đầu vào (đã có year, month từ preprocessing)
        
    Returns:
    --------
    pd.DataFrame
        DataFrame với các đặc trưng thời gian
    """
    df = df.copy()
    
    if 'month' in df.columns:
        # Season (mùa)
        df['season'] = df['month'].apply(lambda x: 
            'Winter' if x in [12, 1, 2] else
            'Spring' if x in [3, 4, 5] else
            'Summer' if x in [6, 7, 8] else 'Fall')
        
        # Quarter
        df['quarter'] = (df['month'] - 1) // 3 + 1
    
    return df


def engineer_features(df: pd.DataFrame, 
                      create_ratios: bool = True,
                      create_interactions: bool = True,
                      create_geo: bool = True,
                      create_time: bool = True) -> pd.DataFrame:
    """
    Pipeline tạo đặc trưng hoàn chỉnh.
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame đầu vào
    create_ratios : bool
        Có tạo ratio features không
    create_interactions : bool
        Có tạo interaction features không
    create_geo : bool
        Có tạo geo features không
    create_time : bool
        Có tạo time features không
        
    Returns:
    --------
    pd.DataFrame
        DataFrame với các đặc trưng mới
    """
    df = df.copy()
    
    if create_ratios:
        df = create_ratio_features(df)
    
    if create_interactions:
        df = create_interaction_features(df)
    
    if create_geo:
        df = create_geo_features(df)
    
    if create_time:
        df = create_time_features(df)
    
    return df


def select_features_by_importance(feature_importance: dict, 
                                  threshold: float = 0.01) -> List[str]:
    """
    Chọn đặc trưng dựa trên importance score.
    
    Parameters:
    -----------
    feature_importance : dict
        Dictionary {feature_name: importance_score}
    threshold : float
        Ngưỡng importance tối thiểu
        
    Returns:
    --------
    List[str]
        Danh sách các đặc trưng được chọn
    """
    selected = [f for f, imp in feature_importance.items() if imp >= threshold]
    return selected


def remove_correlated_features(df: pd.DataFrame, 
                               threshold: float = 0.95) -> pd.DataFrame:
    """
    Loại bỏ các đặc trưng có tương quan cao.
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame đầu vào
    threshold : float
        Ngưỡng tương quan để loại bỏ
        
    Returns:
    --------
    pd.DataFrame
        DataFrame sau khi loại bỏ đặc trưng trùng lặp
    """
    # Chỉ xét các cột số
    numeric_df = df.select_dtypes(include=[np.number])
    
    # Tính ma trận tương quan
    corr_matrix = numeric_df.corr().abs()
    
    # Tìm các cặp có tương quan cao
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    
    # Các cột cần loại bỏ
    to_drop = [column for column in upper.columns if any(upper[column] > threshold)]
    
    return df.drop(columns=to_drop)


if __name__ == "__main__":
    print("Module feature_engineering.py loaded successfully!")
