"""
Source Package - Recovery Days Prediction
==========================================
Mã nguồn cho dự án dự đoán số ngày phục hồi sau thảm họa.

Modules:
- preprocessing: Tiền xử lý dữ liệu
- eda: Phân tích khám phá dữ liệu
- feature_engineering: Tạo đặc trưng mới
- model_TranMinhHieu: Huấn luyện mô hình CatBoost
- evaluation: Đánh giá mô hình
- app: Script chính
"""

from .preprocessing import load_data, preprocess_data, get_categorical_features
from .eda import perform_eda, get_data_summary
from .feature_engineering import engineer_features
from .model_TranMinhHieu import (
    get_catboost_model,
    train_baseline_model,
    hyperparameter_tuning,
    train_optimized_model,
    save_model,
    load_model
)
from .evaluation import calculate_metrics, evaluate_model

__version__ = "1.0.0"
__author__ = "Trần Minh Hiếu"
