# -*- coding: utf-8 -*-
"""
Utility: Quick Train Script - Huấn luyện nhanh XGBoost
Dùng khi muốn train nhanh model mà không cần toàn bộ config
"""

import pandas as pd
import numpy as np
from pathlib import Path
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from xgboost import XGBRegressor
import warnings

warnings.filterwarnings('ignore')


def quick_train():
    """Train XGBoost model quickly"""
    
    print("⚡ QUICK TRAIN XGBoost")
    print("="*50)
    
    # Load data
    data_paths = [
        Path(__file__).parent / 'data' / 'global_disaster_response_2018_2024.csv',
        Path(__file__).parent.parent / 'data' / 'global_disaster_response_2018_2024.csv',
    ]
    
    df = None
    for path in data_paths:
        if path.exists():
            df = pd.read_csv(path)
            print(f"✅ Loaded data from {path}")
            break
    
    if df is None:
        print("❌ Data file not found!")
        return
    
    # Prepare data
    df['date'] = pd.to_datetime(df['date'], errors='coerce')
    df['year'] = df['date'].dt.year
    df['month'] = df['date'].dt.month
    df = df.drop('date', axis=1)
    
    numerical_features = ['severity_index', 'casualties', 'economic_loss_usd',
                         'response_time_hours', 'aid_amount_usd',
                         'response_efficiency_score', 'latitude', 'longitude']
    time_features = ['year', 'month']
    categorical_features = ['country', 'disaster_type']
    
    features = numerical_features + time_features + categorical_features
    available_features = [f for f in features if f in df.columns]
    
    # Encode categorical
    df_train = df.copy()
    encoders = {}
    
    for col in categorical_features:
        if col in df_train.columns:
            le = LabelEncoder()
            df_train[col + '_encoded'] = le.fit_transform(df_train[col])
            available_features.remove(col)
            available_features.append(col + '_encoded')
            encoders[col] = le
    
    X = df_train[available_features].copy()
    y = df_train['recovery_days'].copy()
    
    # Scale
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X = pd.DataFrame(X_scaled, columns=available_features)
    
    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    print(f"📊 Train/Test: {len(X_train)}/{len(X_test)}")
    
    # Train
    print("⏳ Training...")
    model = XGBRegressor(n_estimators=300, max_depth=5, learning_rate=0.05, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)
    
    # Evaluate
    y_pred = model.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    
    print(f"\n📊 Results:")
    print(f"   R²: {r2*100:.2f}%")
    print(f"   MAE: {mae:.4f} days")
    print(f"   RMSE: {rmse:.4f} days")
    
    # Save
    with open(Path(__file__).parent / 'xgboost_model.pkl', 'wb') as f:
        pickle.dump(model, f)
    with open(Path(__file__).parent / 'xgboost_scaler.pkl', 'wb') as f:
        pickle.dump(scaler, f)
    with open(Path(__file__).parent / 'xgboost_encoders.pkl', 'wb') as f:
        pickle.dump(encoders, f)
    
    print(f"\n✅ Model saved!")


if __name__ == "__main__":
    quick_train()
