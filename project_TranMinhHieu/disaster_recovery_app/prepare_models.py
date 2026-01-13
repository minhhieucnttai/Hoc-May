# -*- coding: utf-8 -*-
"""
Script để chuẩn bị dữ liệu và train các mô hình
================================================
Chạy script này để tạo:
- X_test.csv: Features của tập test
- y_test.csv: Target của tập test
- model_catboost.pkl: Mô hình CatBoost
- model_rf.pkl: Mô hình Random Forest
- model_xgb.pkl: Mô hình XGBoost

Usage:
    python prepare_models.py
"""

import pandas as pd
import numpy as np
import pickle
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor

# Import các thư viện ML
try:
    from catboost import CatBoostRegressor
    HAS_CATBOOST = True
except ImportError:
    HAS_CATBOOST = False
    print("Warning: CatBoost not installed")

try:
    import xgboost as xgb
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False
    print("Warning: XGBoost not installed")


def load_and_preprocess_data(data_path):
    """Load và tiền xử lý dữ liệu."""
    print("Loading data...")
    df = pd.read_csv(data_path)
    
    # Convert date to datetime with explicit format detection
    if 'date' in df.columns:
        # Try multiple date formats
        date_formats = ['%d/%m/%Y', '%Y-%m-%d', '%m/%d/%Y']
        for fmt in date_formats:
            try:
                df['date'] = pd.to_datetime(df['date'], format=fmt, errors='coerce')
                if df['date'].notna().sum() > len(df) * 0.9:  # If >90% parsed successfully
                    break
            except (ValueError, TypeError):
                continue
        df['year'] = df['date'].dt.year
        df['month'] = df['date'].dt.month
    
    # Handle missing values with imputation for numeric columns
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        if df[col].isnull().any():
            df[col] = df[col].fillna(df[col].median())
    
    # Drop rows with remaining missing values in categorical columns
    df = df.dropna()
    
    print(f"Data shape: {df.shape}")
    return df


def prepare_features(df):
    """Chuẩn bị features cho modeling."""
    # Define features
    numeric_features = [
        'severity_index', 'casualties', 'economic_loss_usd',
        'response_time_hours', 'aid_amount_usd', 'response_efficiency_score',
        'latitude', 'longitude'
    ]
    
    categorical_features = ['country', 'disaster_type']
    
    # Add time features if available
    if 'year' in df.columns:
        numeric_features.append('year')
    if 'month' in df.columns:
        numeric_features.append('month')
    
    # Encode categorical features
    label_encoders = {}
    for col in categorical_features:
        le = LabelEncoder()
        df[col + '_encoded'] = le.fit_transform(df[col].astype(str))
        label_encoders[col] = le
        numeric_features.append(col + '_encoded')
    
    # Target
    target = 'recovery_days'
    
    X = df[numeric_features].copy()
    y = df[target].copy()
    
    return X, y, label_encoders


def train_catboost(X_train, y_train, cat_features=None):
    """Train CatBoost model."""
    if not HAS_CATBOOST:
        return None
    
    print("Training CatBoost...")
    model = CatBoostRegressor(
        iterations=300,
        learning_rate=0.1,
        depth=6,
        l2_leaf_reg=3,
        random_state=42,
        verbose=False
    )
    model.fit(X_train, y_train)
    return model


def train_random_forest(X_train, y_train):
    """Train Random Forest model."""
    print("Training Random Forest...")
    model = RandomForestRegressor(
        n_estimators=100,
        max_depth=10,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1
    )
    model.fit(X_train, y_train)
    return model


def train_xgboost(X_train, y_train):
    """Train XGBoost model."""
    if not HAS_XGBOOST:
        return None
    
    print("Training XGBoost...")
    model = xgb.XGBRegressor(
        n_estimators=100,
        max_depth=6,
        learning_rate=0.1,
        random_state=42
    )
    model.fit(X_train, y_train)
    return model


def main():
    # Paths
    base_path = Path(__file__).parent
    data_paths = [
        base_path.parent / "data" / "global_disaster_response_2018_2024.csv",
        base_path.parent / "web" / "data" / "global_disaster_response_2018_2024.csv",
        base_path.parent / "src" / "data" / "global_disaster_response_2018_2024.csv",
    ]
    
    # Find data file
    data_path = None
    for p in data_paths:
        if p.exists():
            data_path = p
            break
    
    if data_path is None:
        print("Error: Data file not found!")
        return
    
    print(f"Using data from: {data_path}")
    
    # Load and preprocess data
    df = load_and_preprocess_data(data_path)
    
    # Prepare features
    X, y, encoders = prepare_features(df)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    print(f"Training set: {X_train.shape}")
    print(f"Test set: {X_test.shape}")
    
    # Save test data
    X_test.to_csv(base_path / "X_test.csv", index=False)
    pd.DataFrame({'recovery_days': y_test}).to_csv(base_path / "y_test.csv", index=False)
    print("Saved X_test.csv and y_test.csv")
    
    # Train models
    models = {}
    
    # CatBoost
    cat_model = train_catboost(X_train, y_train)
    if cat_model is not None:
        models['CatBoost'] = cat_model
        with open(base_path / "model_catboost.pkl", 'wb') as f:
            pickle.dump(cat_model, f)
        print("Saved model_catboost.pkl")
    
    # Random Forest
    rf_model = train_random_forest(X_train, y_train)
    models['Random Forest'] = rf_model
    with open(base_path / "model_rf.pkl", 'wb') as f:
        pickle.dump(rf_model, f)
    print("Saved model_rf.pkl")
    
    # XGBoost
    xgb_model = train_xgboost(X_train, y_train)
    if xgb_model is not None:
        models['XGBoost'] = xgb_model
        with open(base_path / "model_xgb.pkl", 'wb') as f:
            pickle.dump(xgb_model, f)
        print("Saved model_xgb.pkl")
    
    # Evaluate models
    print("\n" + "="*50)
    print("MODEL EVALUATION")
    print("="*50)
    
    from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
    
    for name, model in models.items():
        y_pred = model.predict(X_test)
        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        r2 = r2_score(y_test, y_pred)
        print(f"\n{name}:")
        print(f"  MAE:  {mae:.2f}")
        print(f"  RMSE: {rmse:.2f}")
        print(f"  R²:   {r2:.4f}")
    
    print("\n✅ Done! All files created successfully.")


if __name__ == "__main__":
    main()
