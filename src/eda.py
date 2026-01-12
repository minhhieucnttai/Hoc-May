"""
Module Phân tích Khám phá Dữ liệu (Exploratory Data Analysis - EDA)
====================================================================
Chứa các hàm phân tích và trực quan hóa dữ liệu.

Các chức năng chính:
- Thống kê mô tả dữ liệu
- Phân tích phân bố
- Phân tích tương quan
- Trực quan hóa dữ liệu
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Optional


def get_data_summary(df: pd.DataFrame) -> dict:
    """
    Tạo bảng tóm tắt thông tin về dữ liệu.
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame cần phân tích
        
    Returns:
    --------
    dict
        Dictionary chứa các thông tin tóm tắt
    """
    summary = {
        'n_rows': len(df),
        'n_cols': len(df.columns),
        'numeric_cols': df.select_dtypes(include=[np.number]).columns.tolist(),
        'categorical_cols': df.select_dtypes(include=['object']).columns.tolist(),
        'missing_values': df.isnull().sum().to_dict(),
        'dtypes': df.dtypes.to_dict()
    }
    return summary


def plot_distribution(df: pd.DataFrame, column: str, figsize: tuple = (10, 4)):
    """
    Vẽ biểu đồ phân bố của một biến số.
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame chứa dữ liệu
    column : str
        Tên cột cần vẽ
    figsize : tuple
        Kích thước biểu đồ
    """
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    
    # Histogram
    axes[0].hist(df[column], bins=50, edgecolor='black', alpha=0.7)
    axes[0].set_xlabel(column)
    axes[0].set_ylabel('Tần suất')
    axes[0].set_title(f'Phân bố của {column}')
    
    # Boxplot
    axes[1].boxplot(df[column].dropna())
    axes[1].set_ylabel(column)
    axes[1].set_title(f'Boxplot của {column}')
    
    plt.tight_layout()
    return fig


def plot_target_distribution(df: pd.DataFrame, target_col: str = 'recovery_days'):
    """
    Vẽ biểu đồ phân bố của biến mục tiêu.
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame chứa dữ liệu
    target_col : str
        Tên cột mục tiêu
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    
    # Histogram với KDE
    sns.histplot(df[target_col], kde=True, ax=axes[0], color='steelblue')
    axes[0].set_xlabel('Số ngày phục hồi (Recovery Days)')
    axes[0].set_ylabel('Tần suất')
    axes[0].set_title('Phân bố số ngày phục hồi')
    
    # Boxplot
    sns.boxplot(x=df[target_col], ax=axes[1], color='steelblue')
    axes[1].set_xlabel('Số ngày phục hồi (Recovery Days)')
    axes[1].set_title('Boxplot số ngày phục hồi')
    
    plt.tight_layout()
    return fig


def plot_correlation_matrix(df: pd.DataFrame, figsize: tuple = (12, 10)):
    """
    Vẽ ma trận tương quan giữa các biến số.
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame chứa dữ liệu
    figsize : tuple
        Kích thước biểu đồ
    """
    # Chỉ lấy các cột số
    numeric_df = df.select_dtypes(include=[np.number])
    
    # Tính ma trận tương quan
    corr_matrix = numeric_df.corr()
    
    # Vẽ heatmap
    fig, ax = plt.subplots(figsize=figsize)
    sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='RdBu_r', 
                center=0, ax=ax, square=True)
    ax.set_title('Ma trận tương quan giữa các biến số')
    
    plt.tight_layout()
    return fig


def plot_categorical_distribution(df: pd.DataFrame, column: str, top_n: int = 10):
    """
    Vẽ biểu đồ phân bố của biến phân loại.
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame chứa dữ liệu
    column : str
        Tên cột cần vẽ
    top_n : int
        Số lượng category hiển thị
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    value_counts = df[column].value_counts().head(top_n)
    value_counts.plot(kind='bar', ax=ax, color='steelblue', edgecolor='black')
    
    ax.set_xlabel(column)
    ax.set_ylabel('Số lượng')
    ax.set_title(f'Phân bố của {column} (Top {top_n})')
    plt.xticks(rotation=45, ha='right')
    
    plt.tight_layout()
    return fig


def plot_target_by_category(df: pd.DataFrame, 
                            category_col: str, 
                            target_col: str = 'recovery_days',
                            top_n: int = 10):
    """
    Vẽ biểu đồ so sánh biến mục tiêu theo các category.
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame chứa dữ liệu
    category_col : str
        Tên cột phân loại
    target_col : str
        Tên cột mục tiêu
    top_n : int
        Số lượng category hiển thị
    """
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Lấy top categories
    top_categories = df[category_col].value_counts().head(top_n).index
    df_filtered = df[df[category_col].isin(top_categories)]
    
    # Boxplot
    sns.boxplot(data=df_filtered, x=category_col, y=target_col, ax=ax)
    ax.set_xlabel(category_col)
    ax.set_ylabel('Số ngày phục hồi')
    ax.set_title(f'Số ngày phục hồi theo {category_col}')
    plt.xticks(rotation=45, ha='right')
    
    plt.tight_layout()
    return fig


def plot_scatter_with_target(df: pd.DataFrame, 
                             feature: str, 
                             target: str = 'recovery_days'):
    """
    Vẽ scatter plot giữa một feature và target.
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame chứa dữ liệu
    feature : str
        Tên feature
    target : str
        Tên biến mục tiêu
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    ax.scatter(df[feature], df[target], alpha=0.3, s=10)
    ax.set_xlabel(feature)
    ax.set_ylabel(target)
    ax.set_title(f'Mối quan hệ giữa {feature} và {target}')
    
    plt.tight_layout()
    return fig


def perform_eda(df: pd.DataFrame, target_col: str = 'recovery_days') -> dict:
    """
    Thực hiện phân tích EDA hoàn chỉnh.
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame cần phân tích
    target_col : str
        Tên cột mục tiêu
        
    Returns:
    --------
    dict
        Dictionary chứa kết quả phân tích
    """
    results = {}
    
    # Thống kê mô tả
    results['summary'] = get_data_summary(df)
    results['describe'] = df.describe().to_dict()
    
    # Thống kê biến mục tiêu
    results['target_stats'] = {
        'mean': df[target_col].mean(),
        'median': df[target_col].median(),
        'std': df[target_col].std(),
        'min': df[target_col].min(),
        'max': df[target_col].max()
    }
    
    return results


if __name__ == "__main__":
    print("Module eda.py loaded successfully!")
