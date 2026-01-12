"""
Script chính - Demo Pipeline Machine Learning
=============================================
Dự đoán số ngày phục hồi sau thảm họa (Recovery Days Prediction)

Pipeline:
1. Load dữ liệu
2. Tiền xử lý (Preprocessing)
3. Phân tích khám phá (EDA)
4. Tạo đặc trưng (Feature Engineering)
5. Huấn luyện mô hình (Model Training)
6. Đánh giá (Evaluation)
7. Giải thích mô hình (SHAP)

Tác giả: Trần Minh Hiếu
"""

import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# Import các module
from preprocessing import load_data, preprocess_data, get_categorical_features
from eda import perform_eda, plot_target_distribution, plot_correlation_matrix
from feature_engineering import engineer_features
from model_TranMinhHieu import (
    prepare_data_for_catboost,
    train_baseline_model,
    hyperparameter_tuning,
    train_optimized_model,
    cross_validate_model,
    get_feature_importance,
    save_model
)
from evaluation import (
    evaluate_model,
    plot_actual_vs_predicted,
    plot_residuals,
    plot_feature_importance,
    compare_models,
    plot_model_comparison
)


def main():
    """
    Chạy pipeline machine learning hoàn chỉnh.
    """
    print("="*60)
    print("DỰ ĐOÁN SỐ NGÀY PHỤC HỒI SAU THẢM HỌA")
    print("(Recovery Days Prediction After Global Disasters)")
    print("="*60)
    
    # =========================================================
    # 1. LOAD DỮ LIỆU
    # =========================================================
    print("\n[1] LOAD DỮ LIỆU...")
    
    # Kiểm tra file dữ liệu
    data_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'global_disaster_response_2018_2024.csv')
    
    if not os.path.exists(data_path):
        print(f"Không tìm thấy file dữ liệu tại: {data_path}")
        print("Đang tạo dữ liệu mẫu...")
        df = create_sample_data()
    else:
        df = load_data(data_path)
    
    print(f"Đã load {len(df)} bản ghi")
    print(f"Các cột: {df.columns.tolist()}")
    
    # =========================================================
    # 2. PHÂN TÍCH KHÁM PHÁ (EDA)
    # =========================================================
    print("\n[2] PHÂN TÍCH KHÁM PHÁ DỮ LIỆU (EDA)...")
    
    eda_results = perform_eda(df)
    print(f"Số dòng: {eda_results['summary']['n_rows']}")
    print(f"Số cột: {eda_results['summary']['n_cols']}")
    print(f"Biến số: {eda_results['summary']['numeric_cols']}")
    print(f"Biến phân loại: {eda_results['summary']['categorical_cols']}")
    
    # Thống kê target
    print(f"\nThống kê biến mục tiêu (recovery_days):")
    print(f"  - Mean: {eda_results['target_stats']['mean']:.2f}")
    print(f"  - Median: {eda_results['target_stats']['median']:.2f}")
    print(f"  - Std: {eda_results['target_stats']['std']:.2f}")
    print(f"  - Min: {eda_results['target_stats']['min']:.2f}")
    print(f"  - Max: {eda_results['target_stats']['max']:.2f}")
    
    # =========================================================
    # 3. TIỀN XỬ LÝ DỮ LIỆU
    # =========================================================
    print("\n[3] TIỀN XỬ LÝ DỮ LIỆU...")
    
    X, y = preprocess_data(df, target_column='recovery_days')
    print(f"Features shape: {X.shape}")
    print(f"Target shape: {y.shape}")
    
    # =========================================================
    # 4. TẠO ĐẶC TRƯNG (FEATURE ENGINEERING)
    # =========================================================
    print("\n[4] TẠO ĐẶC TRƯNG MỚI...")
    
    X = engineer_features(X)
    print(f"Số features sau engineering: {X.shape[1]}")
    print(f"Các features mới: {[c for c in X.columns if '_' in c and c not in df.columns][:5]}...")
    
    # Lấy danh sách categorical features
    cat_features = get_categorical_features(X)
    print(f"Categorical features: {cat_features}")
    
    # =========================================================
    # 5. CHUẨN BỊ DỮ LIỆU CHO MÔ HÌNH
    # =========================================================
    print("\n[5] CHIA DỮ LIỆU TRAIN/TEST...")
    
    X_train, X_test, y_train, y_test, cat_feature_indices = prepare_data_for_catboost(
        X, y, cat_features, test_size=0.2, random_state=42
    )
    
    print(f"Train set: {len(X_train)} samples")
    print(f"Test set: {len(X_test)} samples")
    
    # =========================================================
    # 6. HUẤN LUYỆN MÔ HÌNH BASELINE
    # =========================================================
    print("\n[6] HUẤN LUYỆN MÔ HÌNH BASELINE...")
    
    baseline_model = train_baseline_model(X_train, y_train, cat_feature_indices)
    print("Đã huấn luyện mô hình baseline!")
    
    # Đánh giá baseline
    baseline_results = evaluate_model(baseline_model, X_test, y_test, "CatBoost Baseline")
    
    # =========================================================
    # 7. TỐI ƯU SIÊU THAM SỐ
    # =========================================================
    print("\n[7] TỐI ƯU SIÊU THAM SỐ (Hyperparameter Tuning)...")
    print("Đang thực hiện RandomizedSearchCV (có thể mất vài phút)...")
    
    best_model, best_params = hyperparameter_tuning(
        X_train, y_train, cat_feature_indices, n_iter=10, cv=3
    )
    
    print(f"Best parameters: {best_params}")
    
    # Đánh giá mô hình tối ưu
    optimized_results = evaluate_model(best_model, X_test, y_test, "CatBoost Optimized")
    
    # =========================================================
    # 8. CROSS-VALIDATION
    # =========================================================
    print("\n[8] CROSS-VALIDATION...")
    
    cv_results = cross_validate_model(best_model, X, y, cv=5)
    print(f"CV RMSE Mean: {cv_results['cv_rmse_mean']:.4f}")
    print(f"CV RMSE Std: {cv_results['cv_rmse_std']:.4f}")
    
    # =========================================================
    # 9. FEATURE IMPORTANCE
    # =========================================================
    print("\n[9] FEATURE IMPORTANCE...")
    
    importance_df = get_feature_importance(best_model, X.columns.tolist())
    print("\nTop 10 Important Features:")
    print(importance_df.head(10).to_string(index=False))
    
    # =========================================================
    # 10. LƯU MÔ HÌNH
    # =========================================================
    print("\n[10] LƯU MÔ HÌNH...")
    
    model_path = os.path.join(os.path.dirname(__file__), '..', 'models', 'catboost_model.cbm')
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    save_model(best_model, model_path)
    print(f"Đã lưu mô hình tại: {model_path}")
    
    # =========================================================
    # KẾT LUẬN
    # =========================================================
    print("\n" + "="*60)
    print("KẾT LUẬN")
    print("="*60)
    print(f"""
Mô hình CatBoost Regressor đã được huấn luyện thành công!

Kết quả đánh giá trên tập test:
- MAE: {optimized_results['metrics']['MAE']:.4f} ngày
- RMSE: {optimized_results['metrics']['RMSE']:.4f} ngày
- R² Score: {optimized_results['metrics']['R2']:.4f}

Mô hình có khả năng dự đoán số ngày phục hồi sau thảm họa
với độ chính xác cao, phù hợp để hỗ trợ ra quyết định
trong công tác ứng phó thảm họa.
""")
    
    return best_model, importance_df, optimized_results


def create_sample_data(n_samples: int = 50002) -> pd.DataFrame:
    """
    Tạo dữ liệu mẫu cho việc demo.
    
    Parameters:
    -----------
    n_samples : int
        Số lượng bản ghi
        
    Returns:
    --------
    pd.DataFrame
        DataFrame mẫu
    """
    np.random.seed(42)
    
    # Danh sách các quốc gia
    countries = ['USA', 'Japan', 'China', 'India', 'Brazil', 'Germany', 'UK', 
                 'France', 'Australia', 'Canada', 'Mexico', 'Indonesia', 
                 'Philippines', 'Bangladesh', 'Pakistan', 'Nigeria', 'Egypt',
                 'Vietnam', 'Thailand', 'South Korea']
    
    # Loại thảm họa
    disaster_types = ['Earthquake', 'Flood', 'Tornado', 'Hurricane', 'Wildfire',
                      'Tsunami', 'Drought', 'Volcanic Eruption', 'Landslide', 'Storm']
    
    # Tạo dữ liệu
    data = {
        'date': pd.date_range('2018-01-01', periods=n_samples, freq='h')[:n_samples],
        'country': np.random.choice(countries, n_samples),
        'disaster_type': np.random.choice(disaster_types, n_samples),
        'severity_index': np.random.uniform(1, 10, n_samples),
        'casualties': np.random.exponential(100, n_samples).astype(int),
        'economic_loss_usd': np.random.exponential(1e6, n_samples),
        'response_time_hours': np.random.exponential(24, n_samples),
        'aid_amount_usd': np.random.exponential(5e5, n_samples),
        'response_efficiency_score': np.random.uniform(0, 1, n_samples),
        'latitude': np.random.uniform(-90, 90, n_samples),
        'longitude': np.random.uniform(-180, 180, n_samples)
    }
    
    df = pd.DataFrame(data)
    
    # Tạo recovery_days dựa trên các features (tạo mối quan hệ phi tuyến)
    df['recovery_days'] = (
        10 + 
        df['severity_index'] * 5 + 
        np.log1p(df['economic_loss_usd']) * 0.5 +
        df['response_time_hours'] * 0.3 -
        np.log1p(df['aid_amount_usd']) * 0.2 -
        df['response_efficiency_score'] * 10 +
        np.random.normal(0, 10, n_samples)
    )
    
    # Đảm bảo recovery_days >= 1
    df['recovery_days'] = df['recovery_days'].clip(lower=1)
    
    return df


if __name__ == "__main__":
    model, importance, results = main()
