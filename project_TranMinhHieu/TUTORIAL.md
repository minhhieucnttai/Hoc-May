# Hướng Dẫn Chi Tiết Từng Bước / Step-by-Step Tutorial

## Bài 1: Bắt Đầu Với Dữ Liệu / Lesson 1: Getting Started with Data

### Mục Tiêu / Objectives:
- Hiểu cấu trúc dữ liệu / Understand data structure
- Tải và khám phá dữ liệu / Load and explore data
- Xử lý dữ liệu cơ bản / Basic data processing

### Bước 1.1: Tải Dữ Liệu / Load Data

```python
import sys
sys.path.append('src')

from preprocessing import load_data

# Tải dữ liệu / Load data
df = load_data('data/global_disaster_response_2018_2024.csv')

# Xem thông tin cơ bản / View basic info
print(f"Số dòng / Rows: {len(df)}")
print(f"Số cột / Columns: {len(df.columns)}")
print("\nTên các cột / Column names:")
print(df.columns.tolist())
```

**Output mong đợi / Expected output:**
```
Data loaded successfully: 35 rows, 9 columns
Số dòng / Rows: 35
Số cột / Columns: 9

Tên các cột / Column names:
['date', 'disaster_type', 'country', 'region', 'affected_population', 
 'casualties', 'response_time_hours', 'response_effectiveness', 'economic_impact_usd']
```

### Bước 1.2: Khám Phá Dữ Liệu / Explore Data

```python
# Xem 5 dòng đầu / View first 5 rows
print("\n5 dòng đầu tiên / First 5 rows:")
print(df.head())

# Thống kê mô tả / Descriptive statistics
print("\nThống kê / Statistics:")
print(df.describe())

# Kiểm tra giá trị thiếu / Check missing values
print("\nGiá trị thiếu / Missing values:")
print(df.isnull().sum())

# Các loại thiên tai / Disaster types
print("\nCác loại thiên tai / Disaster types:")
print(df['disaster_type'].value_counts())
```

### Bước 1.3: Tiền Xử Lý / Preprocessing

```python
from preprocessing import preprocess_data

# Tiền xử lý đầy đủ / Full preprocessing
df_clean = preprocess_data('data/global_disaster_response_2018_2024.csv', save_output=False)

# So sánh trước và sau / Compare before and after
print(f"\nTrước xử lý / Before: {len(df)} rows")
print(f"Sau xử lý / After: {len(df_clean)} rows")
```

---

## Bài 2: Kỹ Thuật Đặc Trưng / Lesson 2: Feature Engineering

### Mục Tiêu / Objectives:
- Tạo đặc trưng từ dữ liệu gốc / Create features from raw data
- Hiểu các loại đặc trưng / Understand feature types
- Mã hóa dữ liệu phân loại / Encode categorical data

### Bước 2.1: Tạo Đặc Trưng Thời Gian / Create Temporal Features

```python
from feature_engineering import create_temporal_features

# Tạo đặc trưng thời gian / Create temporal features
df_temporal = create_temporal_features(df_clean)

# Xem các cột mới / View new columns
new_cols = ['year', 'month', 'quarter', 'day_of_year', 'season']
print("\nĐặc trưng thời gian mới / New temporal features:")
print(df_temporal[new_cols].head())

# Phân tích theo mùa / Analyze by season
print("\nThiên tai theo mùa / Disasters by season:")
print(df_temporal['season'].value_counts())
```

### Bước 2.2: Tạo Chỉ Số Mức Độ / Create Severity Index

```python
from feature_engineering import create_severity_index

# Tạo severity index / Create severity index
df_severity = create_severity_index(df_temporal)

# Xem phân phối / View distribution
print("\nPhân phối Severity Index / Severity Index distribution:")
print(df_severity['severity_index'].describe())

# Top 5 thiên tai nghiêm trọng nhất / Top 5 most severe disasters
print("\nTop 5 thiên tai nghiêm trọng nhất / Top 5 most severe disasters:")
top5 = df_severity.nlargest(5, 'severity_index')[['date', 'disaster_type', 'country', 'severity_index']]
print(top5)
```

### Bước 2.3: Tạo Tất Cả Đặc Trưng / Create All Features

```python
from feature_engineering import engineer_features

# Tạo tất cả đặc trưng / Engineer all features
df_engineered, encoders = engineer_features(df_clean, encode_categoricals=True)

print(f"\nTổng số đặc trưng / Total features: {len(df_engineered.columns)}")
print("\nCác đặc trưng mới / New features:")
for col in df_engineered.columns:
    if col not in df_clean.columns:
        print(f"  - {col}")
```

---

## Bài 3: Huấn Luyện Mô Hình Cơ Bản / Lesson 3: Basic Model Training

### Mục Tiêu / Objectives:
- Huấn luyện mô hình đơn giản / Train a simple model
- Đánh giá hiệu suất / Evaluate performance
- Hiểu metrics / Understand metrics

### Bước 3.1: Chuẩn Bị Dữ Liệu / Prepare Data

```python
from model_TranMinhHieu import DisasterResponseModel

# Khởi tạo mô hình / Initialize model
model = DisasterResponseModel(model_type='random_forest', task='regression')

# Chọn đặc trưng / Select features
feature_cols = [
    'affected_population',
    'casualties', 
    'economic_impact_usd',
    'response_time_hours'
]

# Chuẩn bị dữ liệu / Prepare data
X, y = model.prepare_features(df_engineered, 'response_effectiveness', feature_cols)

print(f"\nKích thước X / X shape: {X.shape}")
print(f"Kích thước y / y shape: {y.shape}")
print(f"\nĐặc trưng sử dụng / Features used:")
for i, feat in enumerate(feature_cols, 1):
    print(f"  {i}. {feat}")
```

### Bước 3.2: Huấn Luyện / Training

```python
# Huấn luyện mô hình / Train model
results = model.train(X, y, test_size=0.2, scale_features=True)

# Hiển thị kết quả / Display results
print("\n" + "="*50)
print("KẾT QUẢ HUẤN LUYỆN / TRAINING RESULTS")
print("="*50)
print(f"Training Score (R²): {results['train_score']:.4f}")
print(f"Testing Score (R²):  {results['test_score']:.4f}")

# Giải thích / Explanation
if results['test_score'] > 0.8:
    print("\n✅ Mô hình rất tốt! / Excellent model!")
elif results['test_score'] > 0.6:
    print("\n✓ Mô hình tốt / Good model!")
else:
    print("\n⚠ Mô hình cần cải thiện / Model needs improvement")
```

### Bước 3.3: Dự Đoán / Prediction

```python
import numpy as np

# Lấy dữ liệu test / Get test data
X_test = results['X_test']
y_test = results['y_test']

# Dự đoán / Predict
y_pred = model.predict(X_test)

# So sánh một vài giá trị / Compare some values
print("\nSo sánh Thực tế vs Dự đoán / Actual vs Predicted:")
print("="*50)
for i in range(min(5, len(y_test))):
    actual = y_test.iloc[i] if hasattr(y_test, 'iloc') else y_test[i]
    predicted = y_pred[i]
    error = abs(actual - predicted)
    print(f"Sample {i+1}: Actual={actual:.3f}, Predicted={predicted:.3f}, Error={error:.3f}")
```

---

## Bài 4: Đánh Giá Mô Hình / Lesson 4: Model Evaluation

### Mục Tiêu / Objectives:
- Sử dụng các metrics đánh giá / Use evaluation metrics
- Tạo visualizations / Create visualizations
- Hiểu kết quả / Understand results

### Bước 4.1: Tính Metrics / Calculate Metrics

```python
from evaluation import evaluate_regression_model

# Đánh giá / Evaluate
metrics = evaluate_regression_model(y_test, y_pred, model_name='Random Forest')

# Giải thích metrics / Explain metrics
print("\n" + "="*50)
print("GIẢI THÍCH METRICS / METRICS EXPLANATION")
print("="*50)
print(f"""
MSE (Mean Squared Error): {metrics['MSE']:.4f}
  → Sai số bình phương trung bình / Average squared error
  → Càng nhỏ càng tốt / Lower is better

RMSE (Root MSE): {metrics['RMSE']:.4f}
  → Căn bậc hai của MSE / Square root of MSE
  → Cùng đơn vị với target / Same unit as target

MAE (Mean Absolute Error): {metrics['MAE']:.4f}
  → Sai số tuyệt đối trung bình / Average absolute error
  → Dễ hiểu hơn MSE / More interpretable than MSE

R² Score: {metrics['R2']:.4f}
  → Tỷ lệ phương sai được giải thích / Variance explained
  → Từ 0 đến 1, càng cao càng tốt / From 0 to 1, higher is better

MAPE: {metrics['MAPE']:.2f}%
  → Sai số phần trăm / Percentage error
  → Dễ hiểu nhất / Most interpretable
""")
```

### Bước 4.2: Visualizations

```python
from evaluation import plot_regression_results
import matplotlib.pyplot as plt

# Tạo plots / Create plots
plot_regression_results(y_test, y_pred, model_name='Random Forest', save_fig=True)

print("\n✓ Plots đã được lưu vào outputs/ / Plots saved to outputs/")
```

### Bước 4.3: Feature Importance

```python
# Lấy feature importance / Get feature importance
importance_df = model.get_feature_importance(top_n=10)

# Vẽ biểu đồ / Plot
from evaluation import plot_feature_importance
plot_feature_importance(importance_df, model_name='Random Forest', save_fig=True)

# Phân tích / Analysis
print("\nPHÂN TÍCH ĐỘ QUAN TRỌNG / IMPORTANCE ANALYSIS:")
print("="*50)
top_feature = importance_df.iloc[0]
print(f"Đặc trưng quan trọng nhất / Most important feature: {top_feature['feature']}")
print(f"Độ quan trọng / Importance: {top_feature['importance']:.4f}")
```

---

## Bài 5: So Sánh Nhiều Mô Hình / Lesson 5: Comparing Multiple Models

### Mục Tiêu / Objectives:
- Huấn luyện nhiều mô hình / Train multiple models
- So sánh hiệu suất / Compare performance
- Chọn mô hình tốt nhất / Select best model

### Bước 5.1: Huấn Luyện Nhiều Mô Hình / Train Multiple Models

```python
from model_TranMinhHieu import train_multiple_models

# Huấn luyện tất cả mô hình / Train all models
models, results = train_multiple_models(X, y, task='regression')

print("\nĐÃ HUẤN LUYỆN CÁC MÔ HÌNH / MODELS TRAINED:")
for model_name in models.keys():
    print(f"  ✓ {model_name}")
```

### Bước 5.2: So Sánh / Compare

```python
from evaluation import compare_models
import pandas as pd

# So sánh / Compare
comparison_df = compare_models(results)

# Hiển thị bảng so sánh / Display comparison table
print("\nBẢNG SO SÁNH MÔ HÌNH / MODEL COMPARISON TABLE:")
print("="*60)
print(comparison_df.to_string(index=False))

# Tìm mô hình tốt nhất / Find best model
best_idx = comparison_df['Test Score'].idxmax()
best_model = comparison_df.loc[best_idx, 'Model']
best_score = comparison_df.loc[best_idx, 'Test Score']

print("\n" + "="*60)
print(f"MÔ HÌNH TốT NHẤT / BEST MODEL: {best_model}")
print(f"Test Score: {best_score:.4f}")
print("="*60)
```

### Bước 5.3: Lưu Mô Hình Tốt Nhất / Save Best Model

```python
# Lấy mô hình tốt nhất / Get best model
best_model_obj = models[best_model.lower().replace(' ', '_')]

# Lưu / Save
best_model_obj.save_model(f'models/best_model_{best_model.replace(" ", "_")}.pkl')
print(f"\n✓ Đã lưu mô hình tốt nhất / Best model saved!")
```

---

## Bài 6: Sử Dụng Web Dashboard / Lesson 6: Using Web Dashboard

### Mục Tiêu / Objectives:
- Khởi động web app / Launch web app
- Khám phá các tính năng / Explore features
- Sử dụng các công cụ tương tác / Use interactive tools

### Bước 6.1: Khởi Động / Launch

```bash
# Trong terminal / In terminal
streamlit run web/app.py
```

### Bước 6.2: Các Tính Năng Chính / Main Features

**Overview Page:**
- Xem tổng quan metrics / View overall metrics
- Kiểm tra thống kê nhanh / Check quick statistics
- Xem thiên tai gần đây / View recent disasters

**Data Explorer:**
```
1. Chọn năm quan tâm / Select years of interest
2. Lọc theo loại thiên tai / Filter by disaster type
3. Chọn khu vực / Select regions
4. Xem dữ liệu đã lọc / View filtered data
5. Tải xuống CSV / Download CSV
```

**Visualizations:**
- Biểu đồ phân phối / Distribution charts
- Xu hướng theo thời gian / Temporal trends
- Phân tích khu vực / Regional analysis
- Đánh giá tác động / Impact assessment

**Statistics:**
- Thống kê mô tả / Descriptive statistics
- Ma trận tương quan / Correlation matrix
- Phân tích phân phối / Distribution analysis

**Predictions:**
```python
# Nhập các tham số:
- Loại thiên tai / Disaster type
- Khu vực / Region
- Dân số bị ảnh hưởng / Affected population
- Casualties
- Tác động kinh tế / Economic impact
- Thời gian phản ứng / Response time

# Nhận kết quả dự đoán / Get prediction result
```

---

## Bài 7: Tùy Chỉnh và Mở Rộng / Lesson 7: Customization and Extension

### Mục Tiêu / Objectives:
- Thêm dữ liệu mới / Add new data
- Tùy chỉnh mô hình / Customize models
- Tạo đặc trưng mới / Create new features

### Bước 7.1: Thêm Dữ Liệu Mới / Add New Data

```python
import pandas as pd

# Tạo dữ liệu mới / Create new data
new_data = pd.DataFrame({
    'date': ['2024-12-01'],
    'disaster_type': ['Earthquake'],
    'country': ['Japan'],
    'region': ['Asia'],
    'affected_population': [100000],
    'casualties': [200],
    'response_time_hours': [18],
    'response_effectiveness': [0.88],
    'economic_impact_usd': [3000000000]
})

# Thêm vào dữ liệu hiện tại / Append to existing data
df_combined = pd.concat([df, new_data], ignore_index=True)

# Lưu / Save
df_combined.to_csv('data/global_disaster_response_updated.csv', index=False)
print(f"✓ Đã thêm {len(new_data)} dòng mới / Added {len(new_data)} new rows")
```

### Bước 7.2: Tạo Đặc Trưng Tùy Chỉnh / Create Custom Features

```python
def create_custom_feature(df):
    """Tạo đặc trưng tùy chỉnh / Create custom feature"""
    
    # Ví dụ: Tỷ lệ hiệu quả/chi phí / Example: Efficiency/cost ratio
    df['efficiency_cost_ratio'] = (
        df['response_effectiveness'] / 
        (df['economic_impact_usd'] / 1e9 + 1)  # Tránh chia 0 / Avoid division by 0
    )
    
    # Ví dụ: Chỉ số phản ứng nhanh / Example: Fast response indicator
    df['fast_response'] = (df['response_time_hours'] <= 24).astype(int)
    
    return df

# Áp dụng / Apply
df_custom = create_custom_feature(df_engineered.copy())
print("✓ Đã tạo đặc trưng tùy chỉnh / Custom features created:")
print("  - efficiency_cost_ratio")
print("  - fast_response")
```

### Bước 7.3: Tùy Chỉnh Hyperparameters

```python
from sklearn.ensemble import RandomForestRegressor

# Tạo mô hình với tham số tùy chỉnh / Create model with custom parameters
custom_rf = RandomForestRegressor(
    n_estimators=200,           # Số cây / Number of trees
    max_depth=15,               # Độ sâu tối đa / Max depth
    min_samples_split=10,       # Min samples to split
    min_samples_leaf=4,         # Min samples in leaf
    max_features='sqrt',        # Features per split
    random_state=42,
    n_jobs=-1                   # Dùng tất cả CPU / Use all CPUs
)

# Huấn luyện / Train
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
custom_rf.fit(X_train, y_train)

# Đánh giá / Evaluate
score = custom_rf.score(X_test, y_test)
print(f"\nCustom Model Test Score: {score:.4f}")
```

---

## Bài 8: Best Practices và Tips

### Mục Tiêu / Objectives:
- Học các best practices / Learn best practices
- Tối ưu hóa hiệu suất / Optimize performance
- Tránh lỗi thường gặp / Avoid common mistakes

### Best Practices:

**1. Luôn kiểm tra dữ liệu / Always check data:**
```python
# Kiểm tra missing values / Check missing values
print(df.isnull().sum())

# Kiểm tra duplicates / Check duplicates
print(f"Duplicates: {df.duplicated().sum()}")

# Kiểm tra data types / Check data types
print(df.dtypes)
```

**2. Chia dữ liệu đúng cách / Split data properly:**
```python
# Luôn dùng random_state để reproducibility
X_train, X_test, y_train, y_test = train_test_split(
    X, y, 
    test_size=0.2, 
    random_state=42  # ← Quan trọng! / Important!
)
```

**3. Scale features khi cần / Scale features when needed:**
```python
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)  # Chỉ transform, không fit! / Only transform, don't fit!
```

**4. Cross-validation:**
```python
# Luôn dùng cross-validation để đánh giá / Always use cross-validation
from sklearn.model_selection import cross_val_score

scores = cross_val_score(model, X, y, cv=5)
print(f"CV Score: {scores.mean():.4f} (+/- {scores.std() * 2:.4f})")
```

**5. Lưu và load models:**
```python
# Lưu cả scaler và model / Save both scaler and model
import pickle

# Lưu / Save
with open('models/full_pipeline.pkl', 'wb') as f:
    pickle.dump({'model': model, 'scaler': scaler}, f)

# Load / Load
with open('models/full_pipeline.pkl', 'rb') as f:
    pipeline = pickle.load(f)
    loaded_model = pipeline['model']
    loaded_scaler = pipeline['scaler']
```

### Common Mistakes to Avoid:

**❌ Không nên:**
```python
# 1. Fit scaler trên toàn bộ dữ liệu / Don't fit scaler on all data
scaler.fit(X)  # ❌ Data leakage!

# 2. Quên random_state / Forget random_state
train_test_split(X, y, test_size=0.2)  # ❌ Không reproducible / Not reproducible

# 3. Dùng test set để tune / Use test set for tuning
# Tune trên validation set, chỉ dùng test set cuối cùng / Tune on validation, use test at end
```

**✅ Nên:**
```python
# 1. Fit scaler chỉ trên training data / Fit scaler only on training data
scaler.fit(X_train)  # ✅

# 2. Luôn dùng random_state / Always use random_state
train_test_split(X, y, test_size=0.2, random_state=42)  # ✅

# 3. Dùng validation set riêng / Use separate validation set
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5)
```

---

## Bài 9: Troubleshooting Guide

### Vấn Đề Thường Gặp / Common Issues:

**1. ImportError: No module named 'xyz'**
```bash
# Giải pháp / Solution:
pip install -r requirements.txt
```

**2. FileNotFoundError: data/xxx.csv**
```python
# Kiểm tra path / Check path
import os
print(os.getcwd())  # Xem thư mục hiện tại / See current directory

# Di chuyển đến đúng thư mục / Navigate to correct directory
cd project_TranMinhHieu
```

**3. Model performance kém / Poor model performance**
```python
# Giải pháp / Solutions:
# 1. Thêm features / Add more features
# 2. Tune hyperparameters
# 3. Thử mô hình khác / Try different models
# 4. Thu thập thêm dữ liệu / Collect more data
```

**4. Memory Error**
```python
# Giải pháp / Solutions:
# 1. Giảm n_estimators trong Random Forest
model = RandomForestRegressor(n_estimators=50)  # Thay vì 100 / Instead of 100

# 2. Dùng fewer features
feature_cols = feature_cols[:5]  # Chỉ dùng 5 features đầu / Only use first 5

# 3. Downsample data
df_sample = df.sample(frac=0.5)  # Dùng 50% data
```

**5. Streamlit không khởi động / Streamlit won't start**
```bash
# Giải pháp / Solutions:
# 1. Kiểm tra port / Check port
streamlit run web/app.py --server.port 8502

# 2. Clear cache / Xóa cache
streamlit cache clear

# 3. Reinstall / Cài lại
pip uninstall streamlit
pip install streamlit
```

---

## Tổng Kết / Conclusion

Bạn đã học được / You have learned:

✅ Tải và xử lý dữ liệu / Load and process data
✅ Tạo đặc trưng kỹ thuật / Engineer features
✅ Huấn luyện mô hình ML / Train ML models
✅ Đánh giá và so sánh mô hình / Evaluate and compare models
✅ Sử dụng web dashboard / Use web dashboard
✅ Tùy chỉnh và mở rộng / Customize and extend
✅ Best practices / Best practices
✅ Troubleshooting / Troubleshooting

**Bước tiếp theo / Next steps:**
1. Thực hành với dữ liệu riêng / Practice with your own data
2. Thử nghiệm các mô hình khác / Experiment with other models
3. Tùy chỉnh dashboard / Customize the dashboard
4. Tạo đặc trưng mới / Create new features

**Happy coding! / Chúc bạn code vui vẻ! 🚀**
