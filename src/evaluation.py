"""
Module Đánh giá Mô hình (Model Evaluation)
==========================================
Chứa các hàm đánh giá hiệu suất mô hình hồi quy.

Các chỉ số đánh giá:
- MAE (Mean Absolute Error)
- RMSE (Root Mean Squared Error)
- R² Score
- MAPE (Mean Absolute Percentage Error)

Lưu ý: Đây là bài toán HỒI QUY, không sử dụng:
- Confusion Matrix
- Precision/Recall/F1
- ROC-AUC
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from typing import Dict, List, Optional
import shap


def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """
    Tính các chỉ số đánh giá mô hình hồi quy.
    
    Parameters:
    -----------
    y_true : np.ndarray
        Giá trị thực
    y_pred : np.ndarray
        Giá trị dự đoán
        
    Returns:
    --------
    Dict[str, float]
        Dictionary chứa các chỉ số
    """
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_true, y_pred)
    
    # MAPE (tránh chia cho 0)
    mask = y_true != 0
    mape = np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100
    
    return {
        'MAE': mae,
        'MSE': mse,
        'RMSE': rmse,
        'R2': r2,
        'MAPE': mape
    }


def print_evaluation_report(metrics: Dict[str, float], model_name: str = "CatBoost"):
    """
    In báo cáo đánh giá mô hình.
    
    Parameters:
    -----------
    metrics : Dict[str, float]
        Các chỉ số đánh giá
    model_name : str
        Tên mô hình
    """
    print(f"\n{'='*50}")
    print(f"BÁO CÁO ĐÁNH GIÁ MÔ HÌNH: {model_name}")
    print(f"{'='*50}")
    print(f"MAE (Mean Absolute Error):     {metrics['MAE']:.4f} ngày")
    print(f"RMSE (Root Mean Squared Error): {metrics['RMSE']:.4f} ngày")
    print(f"R² Score:                       {metrics['R2']:.4f}")
    print(f"MAPE (Mean Absolute % Error):  {metrics['MAPE']:.2f}%")
    print(f"{'='*50}\n")


def plot_actual_vs_predicted(y_true: np.ndarray, 
                             y_pred: np.ndarray,
                             title: str = "Actual vs Predicted"):
    """
    Vẽ biểu đồ so sánh giá trị thực và dự đoán.
    
    Parameters:
    -----------
    y_true : np.ndarray
        Giá trị thực
    y_pred : np.ndarray
        Giá trị dự đoán
    title : str
        Tiêu đề biểu đồ
    """
    fig, ax = plt.subplots(figsize=(10, 8))
    
    ax.scatter(y_true, y_pred, alpha=0.3, s=10, color='steelblue')
    
    # Đường chéo (perfect prediction)
    min_val = min(y_true.min(), y_pred.min())
    max_val = max(y_true.max(), y_pred.max())
    ax.plot([min_val, max_val], [min_val, max_val], 'r--', label='Perfect Prediction')
    
    ax.set_xlabel('Giá trị thực (Actual Recovery Days)', fontsize=12)
    ax.set_ylabel('Giá trị dự đoán (Predicted Recovery Days)', fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.legend()
    
    # Thêm R² vào biểu đồ
    r2 = r2_score(y_true, y_pred)
    ax.text(0.05, 0.95, f'R² = {r2:.4f}', transform=ax.transAxes, fontsize=12,
            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat'))
    
    plt.tight_layout()
    return fig


def plot_residuals(y_true: np.ndarray, y_pred: np.ndarray):
    """
    Vẽ biểu đồ phân bố residuals (sai số).
    
    Parameters:
    -----------
    y_true : np.ndarray
        Giá trị thực
    y_pred : np.ndarray
        Giá trị dự đoán
    """
    residuals = y_true - y_pred
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Histogram of residuals
    axes[0].hist(residuals, bins=50, edgecolor='black', alpha=0.7, color='steelblue')
    axes[0].axvline(x=0, color='red', linestyle='--', label='Zero Error')
    axes[0].set_xlabel('Residual (Actual - Predicted)')
    axes[0].set_ylabel('Tần suất')
    axes[0].set_title('Phân bố Residuals')
    axes[0].legend()
    
    # Residuals vs Predicted
    axes[1].scatter(y_pred, residuals, alpha=0.3, s=10, color='steelblue')
    axes[1].axhline(y=0, color='red', linestyle='--')
    axes[1].set_xlabel('Giá trị dự đoán')
    axes[1].set_ylabel('Residual')
    axes[1].set_title('Residuals vs Predicted Values')
    
    plt.tight_layout()
    return fig


def plot_feature_importance(importance_df: pd.DataFrame, top_n: int = 15):
    """
    Vẽ biểu đồ Feature Importance.
    
    Parameters:
    -----------
    importance_df : pd.DataFrame
        DataFrame chứa feature và importance
    top_n : int
        Số lượng features hiển thị
    """
    fig, ax = plt.subplots(figsize=(10, 8))
    
    top_features = importance_df.head(top_n)
    
    bars = ax.barh(range(len(top_features)), top_features['importance'], color='steelblue')
    ax.set_yticks(range(len(top_features)))
    ax.set_yticklabels(top_features['feature'])
    ax.invert_yaxis()
    ax.set_xlabel('Importance Score')
    ax.set_title(f'Top {top_n} Feature Importance')
    
    plt.tight_layout()
    return fig


def compute_shap_values(model, X: pd.DataFrame, sample_size: int = 1000):
    """
    Tính SHAP values để giải thích mô hình.
    
    Parameters:
    -----------
    model : CatBoostRegressor
        Mô hình đã train
    X : pd.DataFrame
        Features
    sample_size : int
        Số lượng samples để tính SHAP
        
    Returns:
    --------
    shap.Explanation
        SHAP values
    """
    # Lấy sample để tăng tốc
    if len(X) > sample_size:
        X_sample = X.sample(n=sample_size, random_state=42)
    else:
        X_sample = X
    
    explainer = shap.TreeExplainer(model)
    shap_values = explainer(X_sample)
    
    return shap_values


def plot_shap_summary(shap_values, X: pd.DataFrame):
    """
    Vẽ biểu đồ SHAP summary.
    
    Parameters:
    -----------
    shap_values : shap.Explanation
        SHAP values
    X : pd.DataFrame
        Features
    """
    fig, ax = plt.subplots(figsize=(10, 8))
    shap.summary_plot(shap_values, X, show=False)
    plt.title('SHAP Feature Importance')
    plt.tight_layout()
    return fig


def compare_models(models_results: Dict[str, Dict]) -> pd.DataFrame:
    """
    So sánh hiệu suất nhiều mô hình.
    
    Parameters:
    -----------
    models_results : Dict[str, Dict]
        Dictionary {model_name: metrics}
        
    Returns:
    --------
    pd.DataFrame
        Bảng so sánh các mô hình
    """
    comparison_data = []
    
    for model_name, metrics in models_results.items():
        row = {'Model': model_name}
        row.update(metrics)
        comparison_data.append(row)
    
    df_comparison = pd.DataFrame(comparison_data)
    df_comparison = df_comparison.sort_values('RMSE')
    
    return df_comparison


def plot_model_comparison(comparison_df: pd.DataFrame, metric: str = 'RMSE'):
    """
    Vẽ biểu đồ so sánh các mô hình.
    
    Parameters:
    -----------
    comparison_df : pd.DataFrame
        DataFrame chứa kết quả so sánh
    metric : str
        Chỉ số để so sánh
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    colors = ['green' if m == comparison_df[metric].min() else 'steelblue' 
              for m in comparison_df[metric]]
    
    bars = ax.bar(comparison_df['Model'], comparison_df[metric], color=colors)
    ax.set_xlabel('Mô hình')
    ax.set_ylabel(metric)
    ax.set_title(f'So sánh {metric} giữa các mô hình')
    plt.xticks(rotation=45, ha='right')
    
    # Thêm giá trị trên mỗi cột
    for bar, val in zip(bars, comparison_df[metric]):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                f'{val:.2f}', ha='center', va='bottom', fontsize=10)
    
    plt.tight_layout()
    return fig


def evaluate_model(model, X_test: pd.DataFrame, y_test: pd.Series, 
                   model_name: str = "CatBoost") -> Dict:
    """
    Pipeline đánh giá mô hình hoàn chỉnh.
    
    Parameters:
    -----------
    model : CatBoostRegressor
        Mô hình đã train
    X_test : pd.DataFrame
        Test features
    y_test : pd.Series
        Test target
    model_name : str
        Tên mô hình
        
    Returns:
    --------
    Dict
        Kết quả đánh giá và các biểu đồ
    """
    # Dự đoán
    y_pred = model.predict(X_test)
    
    # Tính metrics
    metrics = calculate_metrics(y_test.values, y_pred)
    
    # In báo cáo
    print_evaluation_report(metrics, model_name)
    
    return {
        'metrics': metrics,
        'y_pred': y_pred,
        'model_name': model_name
    }


if __name__ == "__main__":
    print("Module evaluation.py loaded successfully!")
    print("Sử dụng các chỉ số hồi quy: MAE, RMSE, R², MAPE")
