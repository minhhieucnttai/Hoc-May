# API Reference - Tài Liệu Tham Khảo API

Tài liệu tham khảo đầy đủ cho tất cả các functions và classes trong dự án.
Complete API reference for all functions and classes in the project.

---

## Table of Contents / Mục Lục

1. [preprocessing.py](#preprocessingpy)
2. [eda.py](#edapy)
3. [feature_engineering.py](#feature_engineeringpy)
4. [model_TranMinhHieu.py](#model_tranminhhieupy)
5. [evaluation.py](#evaluationpy)
6. [main.py](#mainpy)

---

## preprocessing.py

### Functions

#### `load_data(filepath='data/global_disaster_response_2018_2024.csv')`

Tải dữ liệu từ file CSV / Load data from CSV file.

**Parameters:**
- `filepath` (str, optional): Đường dẫn đến file CSV. Mặc định là 'data/global_disaster_response_2018_2024.csv' / Path to CSV file. Default is 'data/global_disaster_response_2018_2024.csv'

**Returns:**
- `pd.DataFrame` or `None`: DataFrame chứa dữ liệu, hoặc None nếu lỗi / DataFrame containing data, or None if error

**Raises:**
- `FileNotFoundError`: Nếu file không tồn tại / If file doesn't exist
- `Exception`: Các lỗi khác khi đọc file / Other errors when reading file

**Example:**
```python
from preprocessing import load_data
df = load_data('data/global_disaster_response_2018_2024.csv')
if df is not None:
    print(f"Loaded {len(df)} records")
```

---

#### `check_missing_values(df)`

Kiểm tra giá trị thiếu trong dataset / Check for missing values in dataset.

**Parameters:**
- `df` (pd.DataFrame): DataFrame đầu vào / Input DataFrame

**Returns:**
- `pd.DataFrame`: Bảng tóm tắt giá trị thiếu / Summary table of missing values

**Columns trong output / Columns in output:**
- `Column`: Tên cột / Column name
- `Missing_Count`: Số giá trị thiếu / Count of missing values
- `Missing_Percentage`: Phần trăm giá trị thiếu / Percentage of missing values

**Example:**
```python
from preprocessing import check_missing_values
missing_summary = check_missing_values(df)
print(missing_summary)
```

---

#### `handle_missing_values(df, strategy='mean')`

Xử lý giá trị thiếu / Handle missing values.

**Parameters:**
- `df` (pd.DataFrame): DataFrame đầu vào / Input DataFrame
- `strategy` (str, optional): Chiến lược xử lý / Strategy for handling
  - `'mean'`: Điền bằng giá trị trung bình / Fill with mean (default)
  - `'median'`: Điền bằng trung vị / Fill with median
  - `'drop'`: Loại bỏ dòng có giá trị thiếu / Drop rows with missing values

**Returns:**
- `pd.DataFrame`: DataFrame đã xử lý / DataFrame with handled missing values

**Example:**
```python
from preprocessing import handle_missing_values
df_clean = handle_missing_values(df, strategy='median')
```

---

#### `convert_data_types(df)`

Chuyển đổi kiểu dữ liệu / Convert data types.

**Parameters:**
- `df` (pd.DataFrame): DataFrame đầu vào / Input DataFrame

**Returns:**
- `pd.DataFrame`: DataFrame với kiểu dữ liệu đúng / DataFrame with correct data types

**Conversions performed / Chuyển đổi thực hiện:**
- Cột 'date' → datetime / Column 'date' → datetime

**Example:**
```python
from preprocessing import convert_data_types
df_converted = convert_data_types(df)
```

---

#### `remove_duplicates(df)`

Loại bỏ dòng trùng lặp / Remove duplicate rows.

**Parameters:**
- `df` (pd.DataFrame): DataFrame đầu vào / Input DataFrame

**Returns:**
- `pd.DataFrame`: DataFrame không có trùng lặp / DataFrame without duplicates

**Example:**
```python
from preprocessing import remove_duplicates
df_unique = remove_duplicates(df)
```

---

#### `preprocess_data(filepath='data/global_disaster_response_2018_2024.csv', save_output=True)`

Pipeline tiền xử lý hoàn chỉnh / Complete preprocessing pipeline.

**Parameters:**
- `filepath` (str, optional): Đường dẫn file đầu vào / Input file path
- `save_output` (bool, optional): Có lưu kết quả hay không / Whether to save output. Default True

**Returns:**
- `pd.DataFrame`: DataFrame đã được tiền xử lý / Preprocessed DataFrame

**Pipeline steps / Các bước pipeline:**
1. Load data / Tải dữ liệu
2. Check missing values / Kiểm tra giá trị thiếu
3. Handle missing values / Xử lý giá trị thiếu
4. Convert data types / Chuyển đổi kiểu dữ liệu
5. Remove duplicates / Loại bỏ trùng lặp
6. Save output (optional) / Lưu kết quả (tùy chọn)

**Example:**
```python
from preprocessing import preprocess_data
df_preprocessed = preprocess_data('data/my_data.csv', save_output=True)
```

---

## eda.py

### Functions

#### `set_plot_style()`

Thiết lập style cho plots / Set plotting style.

**Parameters:** None

**Returns:** None

**Side effects:**
- Thay đổi seaborn style thành 'whitegrid' / Changes seaborn style to 'whitegrid'
- Đặt kích thước figure mặc định (12, 6) / Sets default figure size to (12, 6)
- Đặt font size 10 / Sets font size to 10

---

#### `basic_statistics(df)`

Hiển thị thống kê cơ bản / Display basic statistics.

**Parameters:**
- `df` (pd.DataFrame): DataFrame đầu vào / Input DataFrame

**Returns:** None

**Output includes:**
- Dataset shape / Kích thước dataset
- Column names and types / Tên cột và kiểu dữ liệu
- Numerical statistics / Thống kê số
- Categorical statistics / Thống kê phân loại

**Example:**
```python
from eda import basic_statistics
basic_statistics(df)
```

---

#### `plot_disaster_types(df, save_fig=False)`

Vẽ phân phối loại thiên tai / Plot distribution of disaster types.

**Parameters:**
- `df` (pd.DataFrame): DataFrame đầu vào / Input DataFrame
- `save_fig` (bool, optional): Có lưu figure hay không / Whether to save figure

**Returns:** None

**Side effects:**
- Hiển thị bar plot / Displays bar plot
- Lưu vào 'outputs/disaster_types_distribution.png' nếu save_fig=True / Saves to 'outputs/disaster_types_distribution.png' if save_fig=True

---

#### `plot_temporal_trends(df, save_fig=False)`

Vẽ xu hướng theo thời gian / Plot temporal trends.

**Parameters:**
- `df` (pd.DataFrame): DataFrame với cột 'date' / DataFrame with 'date' column
- `save_fig` (bool, optional): Có lưu figure hay không / Whether to save figure

**Returns:** None

**Requirements:**
- DataFrame phải có cột 'date' / DataFrame must have 'date' column

**Charts created / Biểu đồ tạo ra:**
1. Number of disasters by year / Số thiên tai theo năm
2. Disaster types by year (stacked) / Loại thiên tai theo năm (xếp chồng)

---

#### `plot_geographical_distribution(df, save_fig=False)`

Vẽ phân phối địa lý / Plot geographical distribution.

**Parameters:**
- `df` (pd.DataFrame): DataFrame đầu vào / Input DataFrame
- `save_fig` (bool, optional): Có lưu figure hay không / Whether to save figure

**Returns:** None

**Charts created:**
1. Distribution by region / Phân phối theo vùng
2. Top 10 countries / Top 10 quốc gia

---

#### `plot_impact_analysis(df, save_fig=False)`

Vẽ phân tích tác động / Plot impact analysis.

**Parameters:**
- `df` (pd.DataFrame): DataFrame đầu vào / Input DataFrame
- `save_fig` (bool, optional): Có lưu figure hay không / Whether to save figure

**Returns:** None

**Charts created:**
1. Casualties by disaster type / Casualties theo loại
2. Affected population by type / Dân số bị ảnh hưởng theo loại
3. Economic impact by type / Tác động kinh tế theo loại
4. Response effectiveness by type / Hiệu quả phản ứng theo loại

---

#### `plot_correlation_matrix(df, save_fig=False)`

Vẽ ma trận tương quan / Plot correlation matrix.

**Parameters:**
- `df` (pd.DataFrame): DataFrame đầu vào / Input DataFrame
- `save_fig` (bool, optional): Có lưu figure hay không / Whether to save figure

**Returns:** None

**Note:** Chỉ tính tương quan cho các cột số / Only calculates correlation for numeric columns

---

#### `perform_eda(df, save_figs=False)`

Thực hiện EDA đầy đủ / Perform complete exploratory data analysis.

**Parameters:**
- `df` (pd.DataFrame): DataFrame đầu vào / Input DataFrame
- `save_figs` (bool, optional): Có lưu tất cả figures hay không / Whether to save all figures

**Returns:** None

**Steps performed:**
1. Basic statistics / Thống kê cơ bản
2. Disaster types plot / Biểu đồ loại thiên tai
3. Temporal trends / Xu hướng thời gian
4. Geographical distribution / Phân phối địa lý
5. Impact analysis / Phân tích tác động
6. Correlation matrix / Ma trận tương quan

**Example:**
```python
from eda import perform_eda
perform_eda(df, save_figs=True)
```

---

## feature_engineering.py

### Functions

#### `create_temporal_features(df)`

Tạo đặc trưng thời gian / Create temporal features.

**Parameters:**
- `df` (pd.DataFrame): DataFrame với cột 'date' / DataFrame with 'date' column

**Returns:**
- `pd.DataFrame`: DataFrame với đặc trưng thời gian mới / DataFrame with new temporal features

**Features created:**
- `year`: Năm / Year (int)
- `month`: Tháng / Month (1-12)
- `quarter`: Quý / Quarter (1-4)
- `day_of_year`: Ngày trong năm / Day of year (1-366)
- `season`: Mùa / Season ('Winter', 'Spring', 'Summer', 'Fall')

**Example:**
```python
from feature_engineering import create_temporal_features
df_temporal = create_temporal_features(df)
print(df_temporal[['year', 'month', 'season']].head())
```

---

#### `create_severity_index(df)`

Tạo chỉ số mức độ nghiêm trọng / Create severity index.

**Parameters:**
- `df` (pd.DataFrame): DataFrame đầu vào / Input DataFrame

**Returns:**
- `pd.DataFrame`: DataFrame với cột 'severity_index' / DataFrame with 'severity_index' column

**Formula:**
```
severity_index = weighted_average(
    casualties_normalized * 0.4,
    affected_population_normalized * 0.3,
    economic_impact_normalized * 0.3
)
```

**Scale:** 0-10 (0 = least severe, 10 = most severe)

**Example:**
```python
from feature_engineering import create_severity_index
df_severity = create_severity_index(df)
print(df_severity['severity_index'].describe())
```

---

#### `create_response_features(df)`

Tạo đặc trưng phản ứng / Create response features.

**Parameters:**
- `df` (pd.DataFrame): DataFrame đầu vào / Input DataFrame

**Returns:**
- `pd.DataFrame`: DataFrame với đặc trưng phản ứng / DataFrame with response features

**Features created:**
- `response_speed`: Categorical ('Very Fast', 'Fast', 'Moderate', 'Slow')
  - Very Fast: ≤ 24 hours
  - Fast: 24-48 hours
  - Moderate: 48-72 hours
  - Slow: > 72 hours
- `response_quality`: Categorical ('Poor', 'Fair', 'Good', 'Excellent')
  - Poor: 0-0.5
  - Fair: 0.5-0.7
  - Good: 0.7-0.85
  - Excellent: 0.85-1.0
- `response_efficiency`: Float (effectiveness / (response_time + 1))

---

#### `create_impact_ratios(df)`

Tạo tỷ lệ tác động / Create impact ratios.

**Parameters:**
- `df` (pd.DataFrame): DataFrame đầu vào / Input DataFrame

**Returns:**
- `pd.DataFrame`: DataFrame với tỷ lệ tác động / DataFrame with impact ratios

**Features created:**
- `casualty_rate`: casualties / (affected_population + 1)
- `economic_impact_per_capita`: economic_impact / (affected_population + 1)
- `economic_impact_per_casualty`: economic_impact / (casualties + 1)

**Note:** +1 được thêm vào mẫu số để tránh chia cho 0 / +1 is added to denominators to avoid division by zero

---

#### `encode_categorical_features(df, categorical_cols=None)`

Mã hóa đặc trưng phân loại / Encode categorical features.

**Parameters:**
- `df` (pd.DataFrame): DataFrame đầu vào / Input DataFrame
- `categorical_cols` (list, optional): Danh sách cột cần mã hóa / List of columns to encode

**Returns:**
- `tuple`: (DataFrame đã mã hóa, dictionary của encoders) / (Encoded DataFrame, dictionary of encoders)

**Default columns encoded:**
- disaster_type
- country
- region
- season
- response_speed
- response_quality

**Example:**
```python
from feature_engineering import encode_categorical_features
df_encoded, encoders = encode_categorical_features(df)

# Decode sau này / Decode later
original_value = encoders['disaster_type'].inverse_transform([0])
```

---

#### `create_aggregated_features(df)`

Tạo đặc trưng tổng hợp / Create aggregated features.

**Parameters:**
- `df` (pd.DataFrame): DataFrame đầu vào / Input DataFrame

**Returns:**
- `pd.DataFrame`: DataFrame với đặc trưng tổng hợp / DataFrame with aggregated features

**Features created:**
- `avg_casualties_by_type`: Casualties trung bình theo loại thiên tai / Average casualties by disaster type
- `avg_response_time_by_region`: Thời gian phản ứng TB theo vùng / Average response time by region
- `avg_effectiveness_by_country`: Hiệu quả TB theo quốc gia / Average effectiveness by country

---

#### `engineer_features(df, encode_categoricals=True)`

Pipeline kỹ thuật đặc trưng đầy đủ / Complete feature engineering pipeline.

**Parameters:**
- `df` (pd.DataFrame): DataFrame đầu vào / Input DataFrame
- `encode_categoricals` (bool, optional): Có mã hóa biến phân loại hay không / Whether to encode categorical variables

**Returns:**
- `tuple`: (DataFrame đã kỹ thuật, dictionary của encoders) / (Engineered DataFrame, dictionary of encoders)

**Pipeline steps:**
1. Create temporal features / Tạo đặc trưng thời gian
2. Create severity index / Tạo chỉ số mức độ
3. Create response features / Tạo đặc trưng phản ứng
4. Create impact ratios / Tạo tỷ lệ tác động
5. Create aggregated features / Tạo đặc trưng tổng hợp
6. Encode categorical features (optional) / Mã hóa (tùy chọn)

**Example:**
```python
from feature_engineering import engineer_features
df_engineered, encoders = engineer_features(df, encode_categoricals=True)
print(f"Total features: {len(df_engineered.columns)}")
```

---

## model_TranMinhHieu.py

### Class: DisasterResponseModel

Mô hình chính cho dự đoán phản ứng thiên tai / Main model class for disaster response prediction.

#### `__init__(model_type='random_forest', task='regression')`

Khởi tạo mô hình / Initialize model.

**Parameters:**
- `model_type` (str, optional): Loại mô hình / Model type
  - `'random_forest'`: Random Forest (default)
  - `'gradient_boosting'`: Gradient Boosting
  - `'linear'`: Linear Regression
  - `'decision_tree'`: Decision Tree
- `task` (str, optional): Loại nhiệm vụ / Task type
  - `'regression'`: Hồi quy (default)
  - `'classification'`: Phân loại

**Attributes:**
- `model`: Mô hình scikit-learn / Scikit-learn model
- `scaler`: StandardScaler object
- `feature_names`: Danh sách tên đặc trưng / List of feature names
- `is_fitted`: Boolean, mô hình đã được huấn luyện chưa / Boolean, whether model is fitted

**Example:**
```python
from model_TranMinhHieu import DisasterResponseModel
model = DisasterResponseModel(model_type='random_forest', task='regression')
```

---

#### `prepare_features(df, target_col, feature_cols=None)`

Chuẩn bị đặc trưng và target / Prepare features and target.

**Parameters:**
- `df` (pd.DataFrame): DataFrame đầu vào / Input DataFrame
- `target_col` (str): Tên cột target / Target column name
- `feature_cols` (list, optional): Danh sách tên đặc trưng / List of feature names

**Returns:**
- `tuple`: (X, y) - Features và target / Features and target

**Example:**
```python
X, y = model.prepare_features(df, 'response_effectiveness', ['casualties', 'response_time_hours'])
```

---

#### `train(X, y, test_size=0.2, scale_features=True)`

Huấn luyện mô hình / Train model.

**Parameters:**
- `X` (pd.DataFrame or np.array): Features
- `y` (pd.Series or np.array): Target
- `test_size` (float, optional): Tỷ lệ test set / Test set proportion (default 0.2)
- `scale_features` (bool, optional): Có scale features hay không / Whether to scale features (default True)

**Returns:**
- `dict`: Dictionary chứa kết quả / Dictionary containing results
  - `X_train`: Training features (scaled)
  - `X_test`: Test features (scaled)
  - `y_train`: Training targets
  - `y_test`: Test targets
  - `train_score`: Training score (R² for regression)
  - `test_score`: Test score (R² for regression)

**Example:**
```python
results = model.train(X, y, test_size=0.2, scale_features=True)
print(f"Test R²: {results['test_score']:.4f}")
```

---

#### `cross_validate(X, y, cv=5)`

Kiểm chứng chéo / Cross-validation.

**Parameters:**
- `X` (pd.DataFrame or np.array): Features
- `y` (pd.Series or np.array): Target
- `cv` (int, optional): Số folds (default 5)

**Returns:**
- `dict`: Dictionary chứa scores / Dictionary containing scores
  - `scores`: Array của scores cho mỗi fold / Array of scores for each fold
  - `mean`: Score trung bình / Mean score
  - `std`: Độ lệch chuẩn / Standard deviation

**Example:**
```python
cv_results = model.cross_validate(X, y, cv=5)
print(f"CV Score: {cv_results['mean']:.4f} (+/- {cv_results['std']*2:.4f})")
```

---

#### `predict(X)`

Dự đoán / Make predictions.

**Parameters:**
- `X` (pd.DataFrame or np.array): Features

**Returns:**
- `np.array`: Predictions

**Raises:**
- `ValueError`: Nếu mô hình chưa được huấn luyện / If model not trained

**Example:**
```python
predictions = model.predict(X_new)
```

---

#### `get_feature_importance(top_n=10)`

Lấy độ quan trọng đặc trưng / Get feature importance.

**Parameters:**
- `top_n` (int, optional): Số đặc trưng top (default 10)

**Returns:**
- `pd.DataFrame` or `None`: DataFrame với importance, hoặc None nếu không có / DataFrame with importance, or None if not available

**Note:** Chỉ hoạt động với tree-based models / Only works with tree-based models

**Example:**
```python
importance = model.get_feature_importance(top_n=15)
print(importance)
```

---

#### `save_model(filepath='models/disaster_response_model.pkl')`

Lưu mô hình / Save model.

**Parameters:**
- `filepath` (str, optional): Đường dẫn file / File path

**Returns:** None

**Saves:**
- Model
- Scaler
- Feature names
- Model type
- Task type

**Example:**
```python
model.save_model('models/my_model.pkl')
```

---

#### `load_model(filepath)` (classmethod)

Tải mô hình / Load model.

**Parameters:**
- `filepath` (str): Đường dẫn file / File path

**Returns:**
- `DisasterResponseModel`: Mô hình đã tải / Loaded model

**Example:**
```python
model = DisasterResponseModel.load_model('models/my_model.pkl')
```

---

### Functions

#### `train_multiple_models(X, y, task='regression')`

Huấn luyện nhiều mô hình / Train multiple models.

**Parameters:**
- `X` (pd.DataFrame): Features
- `y` (pd.Series): Target
- `task` (str, optional): 'regression' or 'classification'

**Returns:**
- `tuple`: (Dictionary của models, Dictionary của results) / (Dictionary of models, Dictionary of results)

**Models trained:**
- Random Forest
- Gradient Boosting
- Linear Regression (regression) / Logistic Regression (classification)
- Decision Tree (regression only)

**Example:**
```python
from model_TranMinhHieu import train_multiple_models
models, results = train_multiple_models(X, y, task='regression')
for name, result in results.items():
    print(f"{name}: {result['test_score']:.4f}")
```

---

## evaluation.py

### Functions

#### `evaluate_regression_model(y_true, y_pred, model_name='Model')`

Đánh giá mô hình hồi quy / Evaluate regression model.

**Parameters:**
- `y_true` (array-like): Giá trị thực / True values
- `y_pred` (array-like): Giá trị dự đoán / Predicted values
- `model_name` (str, optional): Tên mô hình / Model name

**Returns:**
- `dict`: Dictionary của metrics
  - `MSE`: Mean Squared Error
  - `RMSE`: Root Mean Squared Error
  - `MAE`: Mean Absolute Error
  - `R2`: R² Score
  - `MAPE`: Mean Absolute Percentage Error (%)

**Example:**
```python
from evaluation import evaluate_regression_model
metrics = evaluate_regression_model(y_test, y_pred, 'Random Forest')
```

---

#### `evaluate_classification_model(y_true, y_pred, model_name='Model')`

Đánh giá mô hình phân loại / Evaluate classification model.

**Parameters:**
- `y_true` (array-like): Labels thực / True labels
- `y_pred` (array-like): Labels dự đoán / Predicted labels
- `model_name` (str, optional): Tên mô hình / Model name

**Returns:**
- `dict`: Dictionary của metrics
  - `Accuracy`: Độ chính xác
  - `Precision`: Độ chính xác dương tính
  - `Recall`: Độ nhạy
  - `F1-Score`: F1 score

---

#### `plot_regression_results(y_true, y_pred, model_name='Model', save_fig=False)`

Vẽ kết quả hồi quy / Plot regression results.

**Parameters:**
- `y_true` (array-like): Giá trị thực / True values
- `y_pred` (array-like): Giá trị dự đoán / Predicted values
- `model_name` (str, optional): Tên mô hình / Model name
- `save_fig` (bool, optional): Có lưu figure hay không / Whether to save figure

**Returns:** None

**Plots created:**
1. Actual vs Predicted scatter plot
2. Residuals plot
3. Residuals distribution

---

#### `plot_confusion_matrix(y_true, y_pred, labels=None, model_name='Model', save_fig=False)`

Vẽ confusion matrix / Plot confusion matrix.

**Parameters:**
- `y_true` (array-like): Labels thực / True labels
- `y_pred` (array-like): Labels dự đoán / Predicted labels
- `labels` (list, optional): Class labels
- `model_name` (str, optional): Tên mô hình / Model name
- `save_fig` (bool, optional): Có lưu figure hay không / Whether to save figure

**Returns:** None

---

#### `plot_feature_importance(feature_importance_df, model_name='Model', top_n=15, save_fig=False)`

Vẽ feature importance / Plot feature importance.

**Parameters:**
- `feature_importance_df` (pd.DataFrame): DataFrame với importance
- `model_name` (str, optional): Tên mô hình / Model name
- `top_n` (int, optional): Số features hiển thị (default 15)
- `save_fig` (bool, optional): Có lưu figure hay không / Whether to save figure

**Returns:** None

---

#### `compare_models(results_dict, metric='test_score')`

So sánh nhiều mô hình / Compare multiple models.

**Parameters:**
- `results_dict` (dict): Dictionary của results từ các mô hình / Dictionary of results from models
- `metric` (str, optional): Metric để so sánh / Metric to compare (default 'test_score')

**Returns:**
- `pd.DataFrame`: Bảng so sánh / Comparison table

**Example:**
```python
from evaluation import compare_models
comparison = compare_models(results)
print(comparison)
```

---

#### `generate_evaluation_report(model, X_test, y_test, model_name='Model', task='regression')`

Tạo báo cáo đánh giá đầy đủ / Generate comprehensive evaluation report.

**Parameters:**
- `model`: Mô hình đã huấn luyện / Trained model
- `X_test`: Test features
- `y_test`: Test targets
- `model_name` (str, optional): Tên mô hình / Model name
- `task` (str, optional): 'regression' or 'classification'

**Returns:**
- `dict`: Dictionary của metrics

**Side effects:**
- Prints metrics
- Creates and saves plots
- Creates feature importance plot

**Example:**
```python
from evaluation import generate_evaluation_report
metrics = generate_evaluation_report(model, X_test, y_test, 'Random Forest', 'regression')
```

---

## main.py

### Function

#### `main()`

Hàm chính thực thi pipeline / Main function executing the pipeline.

**Parameters:** None

**Returns:** None

**Pipeline steps:**
1. Load and preprocess data / Tải và tiền xử lý dữ liệu
2. Perform EDA / Thực hiện EDA
3. Engineer features / Kỹ thuật đặc trưng
4. Train model (Response Effectiveness) / Huấn luyện mô hình
5. Evaluate model / Đánh giá mô hình
6. Compare multiple models / So sánh nhiều mô hình
7. Train secondary model (Casualty Prediction) / Huấn luyện mô hình phụ

**Outputs created:**
- `data/global_disaster_response_2018_2024_preprocessed.csv`
- `data/global_disaster_response_2018_2024_engineered.csv`
- `models/response_effectiveness_rf_model.pkl`
- `models/casualty_prediction_rf_model.pkl`
- `outputs/*.png` (visualization images)
- `outputs/model_comparison.csv`

**Example:**
```python
from main import main
main()
```

---

## Notes / Ghi Chú

### Data Type Conventions / Quy ước kiểu dữ liệu

- `df`, `DataFrame`: pandas DataFrame
- `X`: Features (DataFrame or numpy array)
- `y`: Target (Series or numpy array)
- `array-like`: Có thể là list, numpy array, pandas Series, etc.

### Error Handling / Xử lý lỗi

Tất cả functions xử lý lỗi một cách graceful và in ra thông báo / All functions handle errors gracefully and print messages:
- FileNotFoundError: Khi file không tìm thấy / When file not found
- ValueError: Khi tham số không hợp lệ / When invalid parameters
- KeyError: Khi cột không tồn tại / When column doesn't exist

### Best Practices / Thực hành tốt nhất

1. Luôn kiểm tra output của load_data() / Always check output of load_data():
   ```python
   df = load_data('data.csv')
   if df is None:
       print("Error loading data")
       return
   ```

2. Lưu intermediate results / Save intermediate results:
   ```python
   df_preprocessed = preprocess_data('data.csv', save_output=True)
   ```

3. Dùng cross-validation / Use cross-validation:
   ```python
   cv_results = model.cross_validate(X, y, cv=5)
   ```

4. Luôn lưu trained models / Always save trained models:
   ```python
   model.save_model('models/my_model.pkl')
   ```

---

**Last updated / Cập nhật lần cuối:** 2026-01-12
**Version / Phiên bản:** 1.0.0
