"""
Module Tiền xử lý dữ liệu (Data Preprocessing)
==============================================
Chứa các hàm xử lý dữ liệu thô trước khi đưa vào mô hình học máy.

Các chức năng chính:
- Đọc và làm sạch dữ liệu
- Xử lý dữ liệu thời gian
- Xử lý biến số (log transform)
- Xử lý giá trị thiếu
"""

import pandas as pd
import numpy as np
from typing import Tuple, List


def load_data(filepath: str) -> pd.DataFrame:
    """
    Đọc dữ liệu từ file CSV.
    
    Parameters:
    -----------
    filepath : str
        Đường dẫn đến file CSV
        
    Returns:
    --------
    pd.DataFrame
        DataFrame chứa dữ liệu thô
    """
    df = pd.read_csv(filepath)
    return df


def process_datetime_features(df: pd.DataFrame, date_column: str = 'date') -> pd.DataFrame:
    """
    Xử lý và trích xuất đặc trưng từ cột thời gian.
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame đầu vào
    date_column : str
        Tên cột chứa dữ liệu thời gian
        
    Returns:
    --------
    pd.DataFrame
        DataFrame với các đặc trưng thời gian mới (year, month)
    """
    df = df.copy()
    if date_column in df.columns:
        df[date_column] = pd.to_datetime(df[date_column])
        df['year'] = df[date_column].dt.year
        df['month'] = df[date_column].dt.month
        df = df.drop(columns=[date_column])
    return df


def apply_log_transform(df: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
    """
    Áp dụng log transform cho các cột có phân bố lệch.
    
    Sử dụng log1p (log(1+x)) để xử lý các giá trị = 0.
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame đầu vào
    columns : List[str]
        Danh sách các cột cần transform
        
    Returns:
    --------
    pd.DataFrame
        DataFrame với các cột đã được log transform
    """
    df = df.copy()
    for col in columns:
        if col in df.columns:
            df[f'{col}_log'] = np.log1p(df[col])
    return df


def handle_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """
    Xử lý các giá trị thiếu trong dữ liệu.
    
    - Biến số: điền bằng median
    - Biến phân loại: điền bằng mode
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame đầu vào
        
    Returns:
    --------
    pd.DataFrame
        DataFrame đã xử lý giá trị thiếu
    """
    df = df.copy()
    
    # Xử lý biến số
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        if df[col].isnull().any():
            df[col] = df[col].fillna(df[col].median())
    
    # Xử lý biến phân loại
    categorical_cols = df.select_dtypes(include=['object']).columns
    for col in categorical_cols:
        if df[col].isnull().any():
            df[col] = df[col].fillna(df[col].mode()[0])
    
    return df


def preprocess_data(df: pd.DataFrame, 
                    target_column: str = 'recovery_days',
                    log_transform_cols: List[str] = None) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Pipeline tiền xử lý dữ liệu hoàn chỉnh.
    
    Thực hiện các bước:
    1. Xử lý dữ liệu thời gian
    2. Áp dụng log transform
    3. Xử lý giá trị thiếu
    4. Tách features và target
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame đầu vào
    target_column : str
        Tên cột mục tiêu
    log_transform_cols : List[str]
        Danh sách các cột cần log transform
        
    Returns:
    --------
    Tuple[pd.DataFrame, pd.Series]
        (X, y) - Features và target
    """
    if log_transform_cols is None:
        log_transform_cols = ['economic_loss_usd', 'aid_amount_usd']
    
    # Xử lý thời gian
    df = process_datetime_features(df)
    
    # Log transform
    df = apply_log_transform(df, log_transform_cols)
    
    # Xử lý giá trị thiếu
    df = handle_missing_values(df)
    
    # Tách features và target
    y = df[target_column]
    X = df.drop(columns=[target_column])
    
    return X, y


def get_categorical_features(X: pd.DataFrame) -> List[str]:
    """
    Lấy danh sách các cột phân loại (categorical features).
    
    Parameters:
    -----------
    X : pd.DataFrame
        DataFrame features
        
    Returns:
    --------
    List[str]
        Danh sách tên các cột phân loại
    """
    return X.select_dtypes(include=['object']).columns.tolist()


if __name__ == "__main__":
    # Test module
    print("Module preprocessing.py loaded successfully!")
