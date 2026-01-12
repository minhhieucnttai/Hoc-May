# Tài Liệu Chi Tiết / Detailed Documentation

## Mục Lục / Table of Contents

1. [Giới Thiệu Tổng Quan / Overview](#overview)
2. [Kiến Trúc Hệ Thống / System Architecture](#architecture)
3. [API Documentation](#api-documentation)
4. [Hướng Dẫn Sử Dụng / Usage Guide](#usage-guide)
5. [Ví Dụ / Examples](#examples)
6. [Câu Hỏi Thường Gặp / FAQ](#faq)

---

## Giới Thiệu Tổng Quan / Overview

Dự án phân tích dữ liệu thiên tai toàn cầu từ 2018-2024, sử dụng Machine Learning để dự đoán và phân tích hiệu quả ứng phó thảm họa.

**Global Disaster Response Analysis** project analyzes worldwide disaster data from 2018-2024, using Machine Learning to predict and analyze disaster response effectiveness.

### Tính Năng Chính / Key Features

- 📊 **Phân tích dữ liệu tự động** / Automated data analysis
- 🤖 **4 mô hình Machine Learning** / 4 ML models (Random Forest, Gradient Boosting, Linear Regression, Decision Tree)
- 📈 **30+ đặc trưng kỹ thuật** / 30+ engineered features
- 🌐 **Dashboard tương tác Streamlit** / Interactive Streamlit dashboard
- 📉 **Đánh giá và trực quan hóa toàn diện** / Comprehensive evaluation and visualization

---

## Kiến Trúc Hệ Thống / System Architecture

### Luồng Dữ Liệu / Data Flow

```
CSV Data → Preprocessing → Feature Engineering → Model Training → Evaluation
                                                        ↓
                                                   Web Dashboard
```

### Cấu Trúc Module / Module Structure

```
project_TranMinhHieu/
├── data/                           # Dữ liệu / Data files
├── src/                            # Mã nguồn / Source code
│   ├── preprocessing.py            # Tiền xử lý / Preprocessing
│   ├── eda.py                      # Phân tích khám phá / EDA
│   ├── feature_engineering.py      # Kỹ thuật đặc trưng / Feature engineering
│   ├── model_TranMinhHieu.py       # Mô hình ML / ML models
│   ├── evaluation.py               # Đánh giá / Evaluation
│   └── main.py                     # Pipeline chính / Main pipeline
└── web/                            # Ứng dụng web / Web app
    └── app.py                      # Streamlit dashboard
```

---

## API Documentation

### 1. preprocessing.py

#### `load_data(filepath)`
Tải dữ liệu từ file CSV / Load data from CSV file

**Parameters:**
- `filepath` (str): Đường dẫn đến file CSV / Path to CSV file

**Returns:**
- `pd.DataFrame`: DataFrame chứa dữ liệu / DataFrame containing data

**Example:**
```python
from preprocessing import load_data
df = load_data('data/global_disaster_response_2018_2024.csv')
```

#### `preprocess_data(filepath, save_output=True)`
Pipeline tiền xử lý hoàn chỉnh / Complete preprocessing pipeline

**Parameters:**
- `filepath` (str): Đường dẫn đến file dữ liệu / Path to data file
- `save_output` (bool): Có lưu kết quả hay không / Whether to save output

**Returns:**
- `pd.DataFrame`: DataFrame đã được làm sạch / Cleaned DataFrame

**Chức năng / Features:**
- Xử lý giá trị thiếu / Handle missing values
- Chuyển đổi kiểu dữ liệu / Convert data types
- Loại bỏ trùng lặp / Remove duplicates

**Example:**
```python
from preprocessing import preprocess_data
df_clean = preprocess_data('data/global_disaster_response_2018_2024.csv')
```

---

### 2. feature_engineering.py

#### `engineer_features(df, encode_categoricals=True)`
Tạo đặc trưng kỹ thuật / Engineer features

**Parameters:**
- `df` (pd.DataFrame): DataFrame đầu vào / Input DataFrame
- `encode_categoricals` (bool): Có mã hóa biến phân loại hay không / Whether to encode categorical variables

**Returns:**
- `tuple`: (DataFrame với đặc trưng mới, dictionary của encoders) / (DataFrame with new features, dictionary of encoders)

**Các đặc trưng được tạo / Features Created:**

1. **Đặc trưng thời gian / Temporal Features:**
   - `year`: Năm / Year
   - `month`: Tháng / Month
   - `quarter`: Quý / Quarter
   - `season`: Mùa (Spring, Summer, Fall, Winter)
   - `day_of_year`: Ngày trong năm / Day of year

2. **Chỉ số mức độ nghiêm trọng / Severity Index:**
   - `severity_index`: Chỉ số 0-10 kết hợp casualties, affected_population, economic_impact
   - Formula: Weighted average của 3 yếu tố được chuẩn hóa

3. **Đặc trưng phản ứng / Response Features:**
   - `response_speed`: Phân loại tốc độ (Very Fast, Fast, Moderate, Slow)
   - `response_quality`: Chất lượng phản ứng (Poor, Fair, Good, Excellent)
   - `response_efficiency`: Hiệu suất = effectiveness / response_time

4. **Tỷ lệ tác động / Impact Ratios:**
   - `casualty_rate`: Tỷ lệ casualties/affected_population
   - `economic_impact_per_capita`: Tác động kinh tế/người
   - `economic_impact_per_casualty`: Tác động kinh tế/casualties

5. **Đặc trưng tổng hợp / Aggregated Features:**
   - `avg_casualties_by_type`: Casualties trung bình theo loại thiên tai
   - `avg_response_time_by_region`: Thời gian phản ứng TB theo vùng
   - `avg_effectiveness_by_country`: Hiệu quả TB theo quốc gia

**Example:**
```python
from feature_engineering import engineer_features
df_engineered, encoders = engineer_features(df_clean, encode_categoricals=True)
print(f"Created {len(df_engineered.columns)} features")
```

---

### 3. model_TranMinhHieu.py

#### Class: `DisasterResponseModel`

**Khởi tạo / Initialization:**
```python
model = DisasterResponseModel(model_type='random_forest', task='regression')
```

**Parameters:**
- `model_type` (str): Loại mô hình / Model type
  - `'random_forest'`: Random Forest (mặc định / default)
  - `'gradient_boosting'`: Gradient Boosting
  - `'linear'`: Linear Regression
  - `'decision_tree'`: Decision Tree
- `task` (str): Loại nhiệm vụ / Task type
  - `'regression'`: Hồi quy (mặc định / default)
  - `'classification'`: Phân loại

#### Methods:

**`prepare_features(df, target_col, feature_cols=None)`**
Chuẩn bị dữ liệu cho mô hình / Prepare data for modeling

**`train(X, y, test_size=0.2, scale_features=True)`**
Huấn luyện mô hình / Train the model

**Returns:**
```python
{
    'X_train': Training features,
    'X_test': Test features,
    'y_train': Training targets,
    'y_test': Test targets,
    'train_score': Training score (R² for regression),
    'test_score': Test score (R² for regression)
}
```

**`cross_validate(X, y, cv=5)`**
Kiểm chứng chéo / Cross-validation

**`predict(X)`**
Dự đoán / Make predictions

**`get_feature_importance(top_n=10)`**
Lấy độ quan trọng của đặc trưng / Get feature importance

**`save_model(filepath)` / `load_model(filepath)`**
Lưu/Tải mô hình / Save/Load model

**Example:**
```python
from model_TranMinhHieu import DisasterResponseModel

# Khởi tạo mô hình / Initialize model
model = DisasterResponseModel(model_type='random_forest', task='regression')

# Chuẩn bị dữ liệu / Prepare data
feature_cols = ['affected_population', 'casualties', 'economic_impact_usd', 'response_time_hours']
X, y = model.prepare_features(df_engineered, 'response_effectiveness', feature_cols)

# Huấn luyện / Train
results = model.train(X, y, test_size=0.2)
print(f"Test R²: {results['test_score']:.4f}")

# Kiểm chứng chéo / Cross-validate
cv_results = model.cross_validate(X, y, cv=5)

# Độ quan trọng đặc trưng / Feature importance
importance = model.get_feature_importance(top_n=10)

# Lưu mô hình / Save model
model.save_model('models/my_model.pkl')
```

---

### 4. evaluation.py

#### `evaluate_regression_model(y_true, y_pred, model_name='Model')`
Đánh giá mô hình hồi quy / Evaluate regression model

**Metrics Calculated:**
- MSE (Mean Squared Error)
- RMSE (Root Mean Squared Error)
- MAE (Mean Absolute Error)
- R² Score
- MAPE (Mean Absolute Percentage Error)

**Example:**
```python
from evaluation import evaluate_regression_model
metrics = evaluate_regression_model(y_test, y_pred, 'Random Forest')
```

#### `plot_regression_results(y_true, y_pred, model_name, save_fig=False)`
Vẽ kết quả hồi quy / Plot regression results

**Tạo 3 biểu đồ / Creates 3 plots:**
1. Actual vs Predicted scatter plot
2. Residuals plot
3. Residuals distribution histogram

#### `compare_models(results_dict, metric='test_score')`
So sánh nhiều mô hình / Compare multiple models

**Example:**
```python
from model_TranMinhHieu import train_multiple_models
from evaluation import compare_models

models, results = train_multiple_models(X, y, task='regression')
comparison_df = compare_models(results)
```

---

### 5. Streamlit Web App (web/app.py)

#### Cấu trúc Dashboard / Dashboard Structure

**6 Trang / 6 Pages:**

1. **Overview** - Tổng quan / Overview
   - Key metrics (Total disasters, casualties, affected population, economic impact)
   - Quick statistics
   - Recent disasters table

2. **Data Explorer** - Khám phá dữ liệu / Data exploration
   - Interactive filters (year, disaster type, region)
   - Filterable data table
   - CSV export

3. **Visualizations** - Trực quan hóa / Visualizations
   - Distribution charts
   - Temporal trends
   - Regional distribution
   - Impact analysis
   - Response effectiveness

4. **Statistics** - Thống kê / Statistics
   - Descriptive statistics
   - Correlation matrix
   - Distribution analysis
   - Box plots

5. **Predictions** - Dự đoán / Predictions
   - Interactive predictor
   - Parameter input
   - Gauge visualization

6. **About** - Giới thiệu / About
   - Project information
   - Technologies used
   - Usage guide

#### Chạy ứng dụng / Run the app:
```bash
streamlit run web/app.py
```

---

## Hướng Dẫn Sử Dụng / Usage Guide

### Bước 1: Cài Đặt / Step 1: Installation

```bash
# Cài đặt dependencies / Install dependencies
pip install -r requirements.txt
```

### Bước 2: Chạy Pipeline Phân Tích / Step 2: Run Analysis Pipeline

```bash
# Di chuyển vào thư mục dự án / Navigate to project directory
cd project_TranMinhHieu

# Chạy pipeline đầy đủ / Run full pipeline
python src/main.py
```

**Kết quả tạo ra / Outputs generated:**
- `data/global_disaster_response_2018_2024_preprocessed.csv`
- `data/global_disaster_response_2018_2024_engineered.csv`
- `models/*.pkl` - Trained models
- `outputs/*.png` - Visualization images
- `outputs/model_comparison.csv`

### Bước 3: Khởi Động Web Dashboard / Step 3: Launch Web Dashboard

```bash
streamlit run web/app.py
```

Mở trình duyệt tại / Open browser at: `http://localhost:8501`

---

## Ví Dụ / Examples

### Example 1: Phân Tích Đơn Giản / Simple Analysis

```python
import sys
sys.path.append('src')

from preprocessing import load_data, preprocess_data
from feature_engineering import engineer_features
from model_TranMinhHieu import DisasterResponseModel

# Tải và xử lý dữ liệu / Load and process data
df = load_data('data/global_disaster_response_2018_2024.csv')
df_clean = preprocess_data('data/global_disaster_response_2018_2024.csv', save_output=False)
df_engineered, encoders = engineer_features(df_clean)

# Huấn luyện mô hình / Train model
model = DisasterResponseModel(model_type='random_forest')
X, y = model.prepare_features(df_engineered, 'response_effectiveness', 
                               ['casualties', 'affected_population', 'response_time_hours'])
results = model.train(X, y)

print(f"Model trained! Test R²: {results['test_score']:.4f}")
```

### Example 2: So Sánh Nhiều Mô Hình / Compare Multiple Models

```python
from model_TranMinhHieu import train_multiple_models
from evaluation import compare_models

# Huấn luyện nhiều mô hình / Train multiple models
models, results = train_multiple_models(X, y, task='regression')

# So sánh hiệu suất / Compare performance
comparison_df = compare_models(results)
print(comparison_df)
```

### Example 3: Dự Đoán Tùy Chỉnh / Custom Prediction

```python
import numpy as np

# Tạo dữ liệu mới / Create new data
new_data = np.array([[100000, 500, 36]])  # [affected_population, casualties, response_time]

# Dự đoán / Predict
prediction = model.predict(new_data)
print(f"Predicted response effectiveness: {prediction[0]:.2f}")
```

### Example 4: Trực Quan Hóa Đặc Trưng / Feature Visualization

```python
from evaluation import plot_feature_importance

# Lấy và vẽ độ quan trọng / Get and plot importance
importance_df = model.get_feature_importance(top_n=15)
plot_feature_importance(importance_df, model_name='Random Forest', save_fig=True)
```

---

## Câu Hỏi Thường Gặp / FAQ

### Q1: Làm thế nào để thêm dữ liệu mới? / How to add new data?

**A:** Thêm dữ liệu vào file CSV với cùng định dạng:
```
date,disaster_type,country,region,affected_population,casualties,response_time_hours,response_effectiveness,economic_impact_usd
2024-12-01,Earthquake,Japan,Asia,50000,100,24,0.85,2000000000
```

### Q2: Làm thế nào để thay đổi tham số mô hình? / How to change model parameters?

**A:** Chỉnh sửa trong `model_TranMinhHieu.py`:
```python
self.model = RandomForestRegressor(
    n_estimators=200,      # Thay đổi từ 100 / Change from 100
    max_depth=15,          # Thay đổi từ 10 / Change from 10
    random_state=42
)
```

### Q3: Làm thế nào để sử dụng mô hình đã lưu? / How to use saved model?

**A:**
```python
from model_TranMinhHieu import DisasterResponseModel

# Tải mô hình / Load model
model = DisasterResponseModel.load_model('models/my_model.pkl')

# Sử dụng / Use
predictions = model.predict(X_new)
```

### Q4: Làm thế nào để thay đổi cổng Streamlit? / How to change Streamlit port?

**A:**
```bash
streamlit run web/app.py --server.port 8502
```

### Q5: Lỗi "Module not found"? / "Module not found" error?

**A:** Đảm bảo bạn đang ở đúng thư mục / Ensure you're in correct directory:
```bash
cd project_TranMinhHieu
python src/main.py
```

Hoặc thêm path / Or add path:
```python
import sys
sys.path.append('src')
```

### Q6: Làm thế nào để tùy chỉnh visualizations? / How to customize visualizations?

**A:** Chỉnh sửa trong `eda.py` hoặc `evaluation.py`:
```python
# Thay đổi kích thước figure / Change figure size
plt.figure(figsize=(16, 8))  # Thay đổi từ (12, 6)

# Thay đổi màu / Change colors
plt.plot(x, y, color='red')  # Thay vì 'blue'

# Thay đổi style / Change style
sns.set_style("darkgrid")  # Thay vì "whitegrid"
```

### Q7: Dữ liệu cần định dạng gì? / What data format is required?

**A:** CSV với các cột sau / CSV with following columns:
- `date`: YYYY-MM-DD
- `disaster_type`: Text
- `country`: Text
- `region`: Text
- `affected_population`: Integer
- `casualties`: Integer
- `response_time_hours`: Float
- `response_effectiveness`: Float (0-1)
- `economic_impact_usd`: Float

### Q8: Làm thế nào để xuất báo cáo? / How to export reports?

**A:** Các visualizations tự động lưu vào `outputs/` khi chạy `main.py` với `save_figs=True`

---

## Tài Liệu Kỹ Thuật / Technical Documentation

### Dependencies Version Requirements

| Package | Version | Purpose |
|---------|---------|---------|
| pandas | ≥1.5.0 | Data manipulation |
| numpy | ≥1.23.0 | Numerical operations |
| scikit-learn | ≥1.2.0 | Machine learning |
| matplotlib | ≥3.6.0 | Static plots |
| seaborn | ≥0.12.0 | Statistical visualization |
| streamlit | ≥1.25.0 | Web framework |
| plotly | ≥5.14.0 | Interactive charts |

### Performance Optimization Tips

1. **Tăng tốc huấn luyện / Speed up training:**
   ```python
   model = RandomForestRegressor(n_jobs=-1)  # Sử dụng tất cả CPU cores
   ```

2. **Giảm memory usage:**
   ```python
   df = pd.read_csv('data.csv', dtype={'casualties': 'int32'})  # Dùng int32 thay vì int64
   ```

3. **Cache Streamlit:**
   ```python
   @st.cache_data
   def load_data():
       return pd.read_csv('data.csv')
   ```

---

## Liên Hệ & Hỗ Trợ / Contact & Support

**Author:** Tran Minh Hieu

Để được hỗ trợ, vui lòng tham khảo:
- README.md
- SETUP_INSTRUCTIONS.md
- Tài liệu này / This documentation

---

**Ngày cập nhật / Last updated:** 2026-01-12
**Phiên bản / Version:** 1.0.0
