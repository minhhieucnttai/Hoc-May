"""
Module Huấn luyện Mô hình CatBoost (Model Training)
===================================================
Chứa các hàm huấn luyện mô hình CatBoost Regressor cho bài toán
dự đoán số ngày phục hồi sau thảm họa.

Tác giả: Trần Minh Hiếu
Mô hình: CatBoost Regressor

Lý do chọn CatBoost:
- Xử lý tốt biến phân loại (country, disaster_type) 
- Không cần One-Hot Encoding
- Hiệu suất cao với dataset vừa-lớn (50k dòng)
- Ít overfitting với Ordered Boosting
- Hỗ trợ GPU acceleration
"""

import pandas as pd
import numpy as np
from catboost import CatBoostRegressor, Pool
from sklearn.model_selection import train_test_split, cross_val_score, RandomizedSearchCV
import joblib
from typing import Tuple, Dict, List, Optional


def get_catboost_model(params: Optional[Dict] = None) -> CatBoostRegressor:
    """
    Khởi tạo mô hình CatBoost với các tham số mặc định hoặc tùy chỉnh.
    
    Parameters:
    -----------
    params : Optional[Dict]
        Tham số tùy chỉnh cho mô hình
        
    Returns:
    --------
    CatBoostRegressor
        Mô hình CatBoost
    """
    default_params = {
        'loss_function': 'RMSE',
        'iterations': 500,
        'learning_rate': 0.05,
        'depth': 6,
        'l2_leaf_reg': 3,
        'random_seed': 42,
        'verbose': False
    }
    
    if params:
        default_params.update(params)
    
    return CatBoostRegressor(**default_params)


def prepare_data_for_catboost(X: pd.DataFrame, 
                               y: pd.Series,
                               cat_features: List[str],
                               test_size: float = 0.2,
                               random_state: int = 42) -> Tuple:
    """
    Chuẩn bị dữ liệu cho CatBoost.
    
    Parameters:
    -----------
    X : pd.DataFrame
        Features
    y : pd.Series
        Target
    cat_features : List[str]
        Danh sách tên các cột categorical
    test_size : float
        Tỷ lệ dữ liệu test
    random_state : int
        Random seed
        
    Returns:
    --------
    Tuple
        (X_train, X_test, y_train, y_test, cat_feature_indices)
    """
    # Tách train/test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    
    # Lấy indices của categorical features
    cat_feature_indices = [X.columns.get_loc(col) for col in cat_features if col in X.columns]
    
    return X_train, X_test, y_train, y_test, cat_feature_indices


def train_baseline_model(X_train: pd.DataFrame, 
                         y_train: pd.Series,
                         cat_features: List[int]) -> CatBoostRegressor:
    """
    Huấn luyện mô hình baseline (không tuning).
    
    Parameters:
    -----------
    X_train : pd.DataFrame
        Training features
    y_train : pd.Series
        Training target
    cat_features : List[int]
        Indices của categorical features
        
    Returns:
    --------
    CatBoostRegressor
        Mô hình đã train
    """
    model = CatBoostRegressor(
        loss_function='RMSE',
        iterations=300,
        learning_rate=0.1,
        depth=6,
        verbose=False,
        random_seed=42
    )
    
    model.fit(X_train, y_train, cat_features=cat_features)
    return model


def hyperparameter_tuning(X_train: pd.DataFrame,
                          y_train: pd.Series,
                          cat_features: List[int],
                          n_iter: int = 20,
                          cv: int = 3) -> Tuple[CatBoostRegressor, Dict]:
    """
    Tối ưu siêu tham số với RandomizedSearchCV.
    
    Các siêu tham số quan trọng:
    - iterations: số cây
    - learning_rate: tốc độ học
    - depth: độ sâu cây
    - l2_leaf_reg: regularization
    
    Parameters:
    -----------
    X_train : pd.DataFrame
        Training features
    y_train : pd.Series
        Training target
    cat_features : List[int]
        Indices của categorical features
    n_iter : int
        Số lần thử nghiệm
    cv : int
        Số folds cross-validation
        
    Returns:
    --------
    Tuple[CatBoostRegressor, Dict]
        (best_model, best_params)
    """
    param_grid = {
        'iterations': [300, 500, 800],
        'learning_rate': [0.01, 0.05, 0.1],
        'depth': [4, 6, 8, 10],
        'l2_leaf_reg': [1, 3, 5, 7],
        'bagging_temperature': [0, 0.5, 1]
    }
    
    base_model = CatBoostRegressor(
        loss_function='RMSE',
        verbose=False,
        random_seed=42,
        cat_features=cat_features
    )
    
    search = RandomizedSearchCV(
        base_model,
        param_grid,
        n_iter=n_iter,
        cv=cv,
        scoring='neg_root_mean_squared_error',
        random_state=42,
        n_jobs=-1
    )
    
    search.fit(X_train, y_train)
    
    return search.best_estimator_, search.best_params_


def train_optimized_model(X_train: pd.DataFrame,
                          y_train: pd.Series,
                          X_val: pd.DataFrame = None,
                          y_val: pd.Series = None,
                          cat_features: List[int] = None,
                          params: Dict = None,
                          verbose: int = 100) -> CatBoostRegressor:
    """
    Huấn luyện mô hình tối ưu với early stopping.
    
    Parameters:
    -----------
    X_train : pd.DataFrame
        Training features
    y_train : pd.Series
        Training target
    X_val : pd.DataFrame
        Validation features (optional)
    y_val : pd.Series
        Validation target (optional)
    cat_features : List[int]
        Indices của categorical features
    params : Dict
        Tham số mô hình
    verbose : int
        Verbosity level (0=silent, 100=print every 100 iterations)
        
    Returns:
    --------
    CatBoostRegressor
        Mô hình đã train
    """
    if params is None:
        params = {
            'iterations': 500,
            'learning_rate': 0.05,
            'depth': 6,
            'l2_leaf_reg': 3
        }
    
    model = CatBoostRegressor(
        loss_function='RMSE',
        verbose=verbose,
        random_seed=42,
        **params
    )
    
    if X_val is not None and y_val is not None:
        model.fit(
            X_train, y_train,
            cat_features=cat_features,
            eval_set=(X_val, y_val),
            early_stopping_rounds=50
        )
    else:
        model.fit(X_train, y_train, cat_features=cat_features)
    
    return model


def cross_validate_model(model: CatBoostRegressor,
                         X: pd.DataFrame,
                         y: pd.Series,
                         cv: int = 5) -> Dict:
    """
    Đánh giá mô hình với cross-validation.
    
    Parameters:
    -----------
    model : CatBoostRegressor
        Mô hình CatBoost
    X : pd.DataFrame
        Features
    y : pd.Series
        Target
    cv : int
        Số folds
        
    Returns:
    --------
    Dict
        Kết quả cross-validation
    """
    scores = cross_val_score(
        model, X, y,
        cv=cv,
        scoring='neg_root_mean_squared_error'
    )
    
    return {
        'cv_rmse_mean': -scores.mean(),
        'cv_rmse_std': scores.std(),
        'cv_scores': -scores
    }


def get_feature_importance(model: CatBoostRegressor,
                           feature_names: List[str]) -> pd.DataFrame:
    """
    Lấy feature importance từ mô hình.
    
    Parameters:
    -----------
    model : CatBoostRegressor
        Mô hình đã train
    feature_names : List[str]
        Danh sách tên features
        
    Returns:
    --------
    pd.DataFrame
        DataFrame chứa feature importance
    """
    importance = model.feature_importances_
    
    df_importance = pd.DataFrame({
        'feature': feature_names,
        'importance': importance
    }).sort_values('importance', ascending=False)
    
    return df_importance


def save_model(model: CatBoostRegressor, filepath: str):
    """
    Lưu mô hình ra file.
    
    Parameters:
    -----------
    model : CatBoostRegressor
        Mô hình cần lưu
    filepath : str
        Đường dẫn file
    """
    model.save_model(filepath)


def load_model(filepath: str) -> CatBoostRegressor:
    """
    Load mô hình từ file.
    
    Parameters:
    -----------
    filepath : str
        Đường dẫn file
        
    Returns:
    --------
    CatBoostRegressor
        Mô hình đã load
    """
    model = CatBoostRegressor()
    model.load_model(filepath)
    return model


if __name__ == "__main__":
    print("Module model_TranMinhHieu.py loaded successfully!")
    print("Sử dụng CatBoost Regressor cho bài toán dự đoán số ngày phục hồi")
