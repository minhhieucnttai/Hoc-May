# -*- coding: utf-8 -*-
"""
Script Huấn luyện Model LightGBM cho Web App (thay thế CatBoost)
Model deterministic - không random, kết quả lặp lại được
"""

import pandas as pd
import numpy as np
from pathlib import Path
import pickle
import json
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import warnings

warnings.filterwarnings('ignore')

# Import LightGBM
try:
    import lightgbm as lgb
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False
    print("❌ LightGBM chưa cài! Cài đặt: pip install lightgbm")

def load_data():
    """Đọc dữ liệu"""
    data_paths = [
        Path(__file__).parent / 'data' / 'global_disaster_response_2018_2024.csv',
        Path(__file__).parent.parent / 'data' / 'global_disaster_response_2018_2024.csv',
    ]
    
    for data_path in data_paths:
        if data_path.exists():
            print(f"✅ Đã tìm thấy dữ liệu tại: {data_path}")
            df = pd.read_csv(data_path)
            
            if 'date' in df.columns:
                df['date'] = pd.to_datetime(df['date'], errors='coerce')
                df['year'] = df['date'].dt.year
                df['month'] = df['date'].dt.month
                df = df.drop('date', axis=1)
            
            print(f"📊 Dữ liệu: {len(df)} bản ghi, {len(df.columns)} cột")
            print(f"🎯 Target: recovery_days - Min: {df['recovery_days'].min():.0f}, Max: {df['recovery_days'].max():.0f}, Mean: {df['recovery_days'].mean():.1f}")
            
            return df
    
    print("❌ Không tìm thấy file dữ liệu!")
    return None


def train_lightgbm_model(df):
    """Huấn luyện LightGBM Model - Deterministic"""
    print("\n" + "="*60)
    print("🚀 HUẤN LUYỆN LIGHTGBM MODEL (Deterministic)")
    print("="*60)
    
    if not LIGHTGBM_AVAILABLE:
        print("❌ LightGBM không sẵn có!")
        return None, None, None, None
    
    try:
        # Features
        numerical_features = ['severity_index', 'casualties', 'economic_loss_usd',
                            'response_time_hours', 'aid_amount_usd',
                            'response_efficiency_score', 'latitude', 'longitude']
        time_features = ['year', 'month']
        categorical_features = ['country', 'disaster_type']
        
        all_features = numerical_features + time_features + categorical_features
        available_features = [f for f in all_features if f in df.columns]
        
        print(f"📊 Features: {len(available_features)}")
        print(f"   - Numerical: {len(numerical_features)}")
        print(f"   - Time: {len(time_features)}")
        print(f"   - Categorical: {len(categorical_features)}")
        
        # Prepare data
        df_train = df.copy()
        encoders = {}
        
        # Encode categorical features
        for col in categorical_features:
            if col in df_train.columns:
                le = LabelEncoder()
                df_train[col + '_encoded'] = le.fit_transform(df_train[col])
                available_features.remove(col)
                available_features.append(col + '_encoded')
                encoders[col] = le
        
        X = df_train[available_features].copy()
        y = df_train['recovery_days'].copy()
        
        # Standardize features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        X = pd.DataFrame(X_scaled, columns=available_features)
        
        # Train-test split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        print(f"\n📈 Train/Test Split:")
        print(f"   - Train: {len(X_train)} ({len(X_train)/len(X)*100:.1f}%)")
        print(f"   - Test: {len(X_test)} ({len(X_test)/len(X)*100:.1f}%)")
        
        # Train LightGBM - Deterministic
        print(f"\n⏳ Đang huấn luyện LightGBM...")
        
        model = lgb.LGBMRegressor(
            n_estimators=500,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            n_jobs=-1,
            verbose=-1
        )
        
        model.fit(X_train, y_train)
        
        # Evaluate
        y_pred_train = model.predict(X_train)
        y_pred_test = model.predict(X_test)
        
        train_r2 = r2_score(y_train, y_pred_train)
        test_r2 = r2_score(y_test, y_pred_test)
        mae = mean_absolute_error(y_test, y_pred_test)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))
        
        metrics = {
            'mae': float(mae),
            'rmse': float(rmse),
            'r2': float(test_r2),
            'train_r2': float(train_r2),
            'test_r2': float(test_r2)
        }
        
        print(f"\n📊 KẾT QUẢ:")
        print(f"   ✅ Train R²: {metrics['train_r2']*100:.2f}%")
        print(f"   ✅ Test R²:  {metrics['test_r2']*100:.2f}%")
        print(f"   📉 MAE:     {metrics['mae']:.4f} ngày")
        print(f"   📊 RMSE:    {metrics['rmse']:.4f} ngày")
        
        # Save model
        model_path = Path(__file__).parent / 'lightgbm_model.pkl'
        scaler_path = Path(__file__).parent / 'lightgbm_scaler.pkl'
        encoders_path = Path(__file__).parent / 'lightgbm_encoders.pkl'
        config_path = Path(__file__).parent / 'lightgbm_config.json'
        
        with open(model_path, 'wb') as f:
            pickle.dump(model, f)
        print(f"\n💾 Đã lưu model: {model_path}")
        
        with open(scaler_path, 'wb') as f:
            pickle.dump(scaler, f)
        print(f"💾 Đã lưu scaler: {scaler_path}")
        
        with open(encoders_path, 'wb') as f:
            pickle.dump(encoders, f)
        print(f"💾 Đã lưu encoders: {encoders_path}")
        
        config = {
            'features': available_features,
            'metrics': metrics,
            'model_type': 'lightgbm'
        }
        
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)
        print(f"💾 Đã lưu config: {config_path}")
        
        return model, scaler, metrics, available_features, encoders
        
    except Exception as e:
        print(f"❌ Lỗi: {e}")
        return None, None, None, None, None


def main():
    """Hàm chính"""
    print("\n" + "="*60)
    print("📊 SCRIPT HUẤN LUYỆN LIGHTGBM MODEL (Deterministic)")
    print("="*60)
    
    # Load data
    df = load_data()
    if df is None:
        return
    
    # Train LightGBM
    model, scaler, metrics, features, encoders = train_lightgbm_model(df)
    
    # Summary
    print("\n" + "="*60)
    print("📋 TÓM TẮT")
    print("="*60)
    
    if model:
        print(f"\n✅ LightGBM Model:")
        print(f"   R² Score: {metrics['r2']*100:.2f}%")
        print(f"   MAE: {metrics['mae']:.4f} ngày")
        print(f"   RMSE: {metrics['rmse']:.4f} ngày")
        print(f"\n   ✨ Model DETERMINISTIC - Kết quả lặp lại được!")
        print(f"   ✨ Random State = 42 (cố định)")
        print(f"   ✨ Không có randomness - Dự đoán nhất quán!")
    
    print("\n" + "="*60)
    print("✅ HOÀN TẤT! LightGBM Model đã sẵn sàng cho Web App")
    print("="*60 + "\n")


if __name__ == "__main__":
    main()
