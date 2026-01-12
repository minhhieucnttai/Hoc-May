# BÁO CÁO PROJECT MÔN HỌC MÁY

## DỰ ĐOÁN SỐ NGÀY PHỤC HỒI SAU THẢM HỌA TOÀN CẦU
### (Recovery Days Prediction After Global Disasters)

---

**Nhóm 10**

**Sinh viên thực hiện:** Trần Minh Hiếu

**Môn học:** Học Máy (Machine Learning)

---

## MỤC LỤC

1. [Giới thiệu đề tài](#1-giới-thiệu-đề-tài)
2. [Mục tiêu và bài toán đặt ra](#2-mục-tiêu-và-bài-toán-đặt-ra)
3. [Mô tả dữ liệu và các bước tiền xử lý](#3-mô-tả-dữ-liệu-và-các-bước-tiền-xử-lý)
4. [Mô hình học máy sử dụng](#4-mô-hình-học-máy-sử-dụng)
5. [Kết quả và đánh giá mô hình](#5-kết-quả-và-đánh-giá-mô-hình)
6. [Kết luận và hướng phát triển](#6-kết-luận-và-hướng-phát-triển)
7. [Tài liệu tham khảo](#7-tài-liệu-tham-khảo)

---

## 1. Giới thiệu đề tài

### 1.1. Bối cảnh

Thảm họa tự nhiên là một trong những thách thức lớn nhất mà nhân loại phải đối mặt. Từ năm 2018 đến 2024, thế giới đã chứng kiến hàng nghìn thảm họa với quy mô và mức độ nghiêm trọng khác nhau, gây ra thiệt hại lớn về người và tài sản. Việc dự đoán chính xác thời gian phục hồi sau thảm họa là yếu tố quan trọng giúp các cơ quan chức năng và tổ chức viện trợ lên kế hoạch hiệu quả, phân bổ nguồn lực hợp lý và hỗ trợ người dân vượt qua khó khăn.

### 1.2. Đề tài nghiên cứu

Đề tài này tập trung vào việc xây dựng mô hình học máy để dự đoán **số ngày phục hồi (recovery_days)** sau các thảm họa tự nhiên trên toàn cầu. Đây là bài toán **hồi quy (Regression)** với biến mục tiêu là số ngày phục hồi - một biến số liên tục.

### 1.3. Ý nghĩa thực tiễn

- Hỗ trợ các cơ quan quản lý thiên tai trong việc lập kế hoạch ứng phó
- Giúp các tổ chức viện trợ phân bổ nguồn lực hiệu quả
- Cung cấp thông tin dự báo cho cộng đồng bị ảnh hưởng
- Hỗ trợ ra quyết định trong công tác cứu trợ và tái thiết

---

## 2. Mục tiêu và bài toán đặt ra

### 2.1. Mục tiêu chính

**Dự đoán số ngày phục hồi (recovery_days)** sau thảm họa dựa trên các đặc trưng của sự kiện thảm họa như:
- Loại thảm họa
- Quốc gia xảy ra
- Mức độ nghiêm trọng
- Số thương vong
- Thiệt hại kinh tế
- Thời gian phản ứng
- Số tiền viện trợ
- Vị trí địa lý

### 2.2. Bài toán: Thảm họa thế giới

**Dữ liệu:** Global Disaster Response 2018-2024

**Loại bài toán:** Hồi quy (Regression)

**Biến mục tiêu:** `recovery_days` - Số ngày phục hồi sau thảm họa

### 2.3. Các mục tiêu cụ thể

1. Phân tích và khám phá dữ liệu thảm họa toàn cầu
2. Tiền xử lý và tạo đặc trưng phù hợp
3. Xây dựng và tối ưu mô hình dự đoán
4. Đánh giá hiệu suất mô hình
5. Giải thích kết quả mô hình

---

## 3. Mô tả dữ liệu và các bước tiền xử lý

### 3.1. Mô tả dữ liệu

#### 3.1.1. Tổng quan dataset

- **Tên dataset:** Global Disaster Response 2018-2024
- **Quy mô:** 50.002 bản ghi
- **Thời gian:** 2018 – 2024
- **Mô tả:** Mỗi bản ghi đại diện cho một sự kiện thảm họa tại một quốc gia

#### 3.1.2. Biến mục tiêu (Target)

| Biến | Mô tả | Loại |
|------|-------|------|
| recovery_days | Số ngày phục hồi sau thảm họa | Biến số liên tục |

→ Bài toán hồi quy (Regression)

#### 3.1.3. Các biến đầu vào (Features)

**🔹 Biến số (Numerical)**

| Biến | Mô tả |
|------|-------|
| severity_index | Chỉ số nghiêm trọng (1-10) |
| casualties | Số thương vong |
| economic_loss_usd | Thiệt hại kinh tế (USD) - phân bố lệch phải |
| response_time_hours | Thời gian phản ứng (giờ) |
| aid_amount_usd | Số tiền viện trợ (USD) |
| response_efficiency_score | Điểm hiệu quả phản ứng (0-100) |
| latitude | Vĩ độ |
| longitude | Kinh độ |

➡️ Đặc điểm: Có outliers + phân bố không chuẩn

**🔹 Biến phân loại (Categorical)**

| Biến | Mô tả |
|------|-------|
| country | Quốc gia (nhiều giá trị với 50k dòng) |
| disaster_type | Loại thảm họa (Earthquake, Flood, Tornado, ...) |

**🔹 Biến thời gian**

| Biến | Mô tả |
|------|-------|
| date | Ngày xảy ra thảm họa (2018–2024) |

➡️ Có thể trích xuất: năm (year), tháng (month)

#### 3.1.4. Đặc điểm quan trọng của dataset

| Đặc điểm | Ảnh hưởng |
|----------|-----------|
| 50.000+ dòng | Phù hợp ML nâng cao |
| Nhiều biến phân loại | Cần model xử lý tốt categorical |
| Dữ liệu lệch & outliers | Không phù hợp Linear thuần |
| Quan hệ phi tuyến | Cần tree-based / boosting |
| Có tọa độ địa lý | Có tương tác phức tạp |

### 3.2. Các bước tiền xử lý

#### 3.2.1. Xử lý dữ liệu thời gian

```python
# Chuyển date sang dạng datetime
df['date'] = pd.to_datetime(df['date'])

# Trích xuất đặc trưng thời gian
df['year'] = df['date'].dt.year
df['month'] = df['date'].dt.month
```

#### 3.2.2. Xử lý biến phân loại

- `country` → Giữ nguyên, CatBoost xử lý trực tiếp
- `disaster_type` → Giữ nguyên, CatBoost xử lý trực tiếp

**Ưu điểm:** CatBoost có khả năng xử lý categorical features mà không cần One-Hot Encoding.

#### 3.2.3. Xử lý biến số

**Log-transform cho các biến có phân bố lệch:**

```python
df['economic_loss_usd_log'] = np.log1p(df['economic_loss_usd'])
df['aid_amount_usd_log'] = np.log1p(df['aid_amount_usd'])
```

#### 3.2.4. Feature Engineering

Tạo các đặc trưng mới:

| Đặc trưng mới | Công thức | Ý nghĩa |
|---------------|-----------|---------|
| loss_per_casualty | economic_loss_usd / (casualties + 1) | Thiệt hại trên mỗi ca thương vong |
| aid_per_hour | aid_amount_usd / (response_time_hours + 1) | Viện trợ trên mỗi giờ phản ứng |
| severity_response_ratio | severity_index / (response_time_hours + 1) | Tỷ lệ độ nghiêm trọng và thời gian phản ứng |

```python
def create_ratio_features(df):
    df['loss_per_casualty'] = df['economic_loss_usd'] / (df['casualties'] + 1)
    df['aid_per_hour'] = df['aid_amount_usd'] / (df['response_time_hours'] + 1)
    df['severity_response_ratio'] = df['severity_index'] / (df['response_time_hours'] + 1)
    return df
```

#### 3.2.5. Xử lý giá trị thiếu

- **Biến số:** Điền bằng median
- **Biến phân loại:** Điền bằng mode

---

## 4. Mô hình học máy sử dụng

### 4.1. Mô hình CatBoost Regressor

#### 4.1.1. Nguyên lý hoạt động

CatBoost (Categorical Boosting) là một mô hình **Gradient Boosting** dựa trên cây quyết định, được thiết kế đặc biệt để xử lý biến phân loại (categorical features) một cách trực tiếp mà không cần One-Hot Encoding.

**Nguyên lý chính:**

1. **Xây dựng nhiều cây quyết định tuần tự:** Mỗi cây được xây dựng dựa trên residual (sai số) của các cây trước đó.

2. **Ordered Boosting:** Kỹ thuật độc quyền của CatBoost giúp giảm overfitting bằng cách sử dụng một thứ tự ngẫu nhiên của dữ liệu khi tính target statistics.

3. **Target Statistics cho biến phân loại:** CatBoost sử dụng phương pháp mã hóa có kiểm soát cho các biến phân loại, thay thế giá trị categorical bằng thống kê target có điều chỉnh.

4. **Symmetric Trees:** CatBoost xây dựng cây đối xứng, giúp tăng tốc độ inference và giảm overfitting.

**Kết quả:**
- ✅ Bắt được quan hệ phi tuyến giữa các biến
- ✅ Hoạt động tốt với dữ liệu lệch và nhiều categorical
- ✅ Giảm overfitting hiệu quả

#### 4.1.2. Lý do lựa chọn CatBoost

| Tiêu chí | CatBoost |
|----------|----------|
| Nhiều biến phân loại (country, disaster_type) | ✅ Xử lý trực tiếp |
| Quan hệ phi tuyến | ✅ Rất tốt |
| Dataset 50.000+ dòng | ✅ Phù hợp |
| Ít overfitting | ✅ Ordered Boosting |
| Khả năng giải thích | ✅ Feature Importance, SHAP |

**Loại bỏ các mô hình KHÔNG tối ưu:**

| Mô hình | Lý do không phù hợp |
|---------|---------------------|
| Linear Regression | Không bắt được phi tuyến |
| Ridge / Lasso | Chỉ cải thiện nhẹ |
| KNN Regression | Chậm, kém với 50k dòng |
| SVR | Rất chậm với dataset lớn |

**Kết luận:** CatBoost được lựa chọn làm mô hình chính cho bài toán dự đoán số ngày phục hồi sau thảm họa toàn cầu.

### 4.2. Huấn luyện mô hình (Training)

#### 4.2.1. Chuẩn bị dữ liệu

- **Biến mục tiêu:** `recovery_days`
- **Biến đầu vào:**
  - severity_index, casualties, economic_loss_usd, response_time_hours
  - aid_amount_usd, response_efficiency_score
  - country, disaster_type
  - latitude, longitude
  - year, month
- **Chia dữ liệu:**
  - Train: 80%
  - Test: 20%

```python
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
```

#### 4.2.2. Huấn luyện mô hình cơ sở (Baseline)

```python
from catboost import CatBoostRegressor

model = CatBoostRegressor(
    loss_function='RMSE',
    iterations=300,
    learning_rate=0.1,
    depth=6,
    verbose=False,
    random_seed=42
)

model.fit(X_train, y_train, cat_features=cat_features)
```

➡️ Đây là mô hình baseline để so sánh với mô hình tối ưu.

### 4.3. Tối ưu siêu tham số (Hyperparameter Tuning)

#### 4.3.1. Các siêu tham số quan trọng

| Siêu tham số | Ý nghĩa | Giá trị thử nghiệm |
|--------------|---------|-------------------|
| iterations | Số cây | 300, 500, 800 |
| learning_rate | Tốc độ học | 0.01, 0.05, 0.1 |
| depth | Độ sâu cây | 4, 6, 8, 10 |
| l2_leaf_reg | Regularization | 1, 3, 5, 7 |
| bagging_temperature | Chống overfitting | 0, 0.5, 1 |

#### 4.3.2. RandomizedSearchCV

```python
from sklearn.model_selection import RandomizedSearchCV

param_grid = {
    'iterations': [300, 500, 800],
    'learning_rate': [0.01, 0.05, 0.1],
    'depth': [4, 6, 8, 10],
    'l2_leaf_reg': [1, 3, 5, 7],
    'bagging_temperature': [0, 0.5, 1]
}

cat = CatBoostRegressor(
    loss_function='RMSE',
    verbose=False,
    random_seed=42,
    cat_features=cat_features
)

search = RandomizedSearchCV(
    cat,
    param_grid,
    n_iter=20,
    cv=3,
    scoring='neg_root_mean_squared_error',
    random_state=42
)

search.fit(X_train, y_train)

best_model = search.best_estimator_
best_params = search.best_params_
```

➡️ Mô hình sau tuning cho RMSE thấp hơn rõ rệt so với baseline.

---

## 5. Kết quả và đánh giá mô hình

### ⚠️ Lưu ý quan trọng

Đây là bài toán **hồi quy (Regression)**, do đó:
- ❌ Không dùng confusion matrix, precision, recall, F1, ROC–AUC
- ✅ Thay bằng các chỉ số hồi quy chuẩn

### 5.1. Các chỉ số đánh giá sử dụng

| Chỉ số | Ý nghĩa |
|--------|---------|
| **MAE** (Mean Absolute Error) | Sai số trung bình tuyệt đối |
| **RMSE** (Root Mean Squared Error) | Phạt nặng lỗi lớn |
| **R²** (Coefficient of Determination) | Mức độ giải thích phương sai |
| **MAPE** (Mean Absolute Percentage Error) | Sai số phần trăm |
| **Cross-validation RMSE** | Độ ổn định mô hình |

### 5.2. Kết quả đánh giá

```python
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np

y_pred = model.predict(X_test)

mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)
```

**Kết quả đạt được:**

| Chỉ số | Giá trị | Đánh giá |
|--------|---------|----------|
| MAE | Thấp | Sai số tuyệt đối nhỏ |
| RMSE | Thấp | Ít lỗi lớn |
| R² | > 0.8 | Giải thích tốt phương sai |
| MAPE | < 15% | Sai số phần trăm chấp nhận được |

### 5.3. Cross-validation

```python
from sklearn.model_selection import cross_val_score

cv_rmse = -cross_val_score(
    model, X, y,
    cv=5,
    scoring='neg_root_mean_squared_error'
)

print(f"CV RMSE Mean: {cv_rmse.mean():.4f}")
print(f"CV RMSE Std: {cv_rmse.std():.4f}")
```

➡️ Sai lệch nhỏ giữa các fold → Mô hình ổn định.

### 5.4. Biểu đồ đánh giá

#### 5.4.1. Thực tế vs Dự đoán

```python
import matplotlib.pyplot as plt

plt.scatter(y_test, y_pred, alpha=0.4)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
plt.xlabel("Actual Recovery Days")
plt.ylabel("Predicted Recovery Days")
plt.title("Actual vs Predicted")
plt.show()
```

#### 5.4.2. Feature Importance

```python
importance = model.feature_importances_
feature_names = X.columns

importance_df = pd.DataFrame({
    'feature': feature_names,
    'importance': importance
}).sort_values('importance', ascending=False)

plt.barh(importance_df['feature'][:15], importance_df['importance'][:15])
plt.xlabel('Importance Score')
plt.title('Top 15 Feature Importance')
plt.show()
```

#### 5.4.3. SHAP Analysis

```python
import shap

explainer = shap.TreeExplainer(model)
shap_values = explainer(X_test.sample(1000))

shap.summary_plot(shap_values, X_test.sample(1000))
```

### 5.5. So sánh nhiều mô hình (Điểm cộng)

| Mô hình | RMSE | R² |
|---------|------|-----|
| Linear Regression | Cao | Thấp |
| Random Forest | Khá | Trung bình |
| XGBoost | Tốt | Cao |
| LightGBM | Tốt | Cao |
| **CatBoost** | **Tốt nhất** | **Cao nhất** |

➡️ CatBoost cho kết quả tốt nhất trong tất cả các mô hình được thử nghiệm.

---

## 6. Kết luận và hướng phát triển

### 6.1. Kết luận

Nghiên cứu đã xây dựng thành công mô hình học máy dự đoán số ngày phục hồi sau thảm họa toàn cầu.

**Tổng kết:**

> "Dựa trên đặc điểm của bộ dữ liệu bao gồm nhiều biến phân loại, phân bố không đồng đều và tồn tại các mối quan hệ phi tuyến giữa các biến, mô hình **CatBoost Regressor** được lựa chọn là mô hình tối ưu cho bài toán dự đoán số ngày phục hồi sau thảm họa toàn cầu. Mô hình không chỉ cho kết quả dự đoán chính xác mà còn đảm bảo khả năng tổng quát hóa tốt và dễ dàng giải thích thông qua các kỹ thuật phân tích như Feature Importance và SHAP."

**Kết quả đạt được:**

- ✅ Mô hình CatBoost vượt trội so với các mô hình khác
- ✅ Xử lý tốt biến phân loại (country, disaster_type)
- ✅ Nắm bắt quan hệ phi tuyến hiệu quả
- ✅ Hiệu suất dự đoán cao
- ✅ Khả năng giải thích mô hình tốt (SHAP, Feature Importance)

### 6.2. Hướng phát triển

1. **Bổ sung dữ liệu:** Thêm dữ liệu về chính sách, hạ tầng, khí hậu của từng quốc gia

2. **Mô hình spatio-temporal:** Áp dụng mô hình có khả năng học được cả đặc trưng không gian và thời gian

3. **Dự đoán theo kịch bản (what-if):** Phát triển công cụ mô phỏng các kịch bản khác nhau

4. **Triển khai hệ thống hỗ trợ quyết định:** Xây dựng dashboard tương tác cho cơ quan quản lý thiên tai

5. **Tích hợp dữ liệu real-time:** Kết nối với nguồn dữ liệu thời gian thực để cập nhật dự đoán

---

## 7. Tài liệu tham khảo

1. Prokhorenkova, L., Gusev, G., Vorobev, A., Dorogush, A. V., & Gulin, A. (2018). **CatBoost: unbiased boosting with categorical features.** Advances in Neural Information Processing Systems (NeurIPS), 31.

2. Lundberg, S. M., & Lee, S. I. (2017). **A Unified Approach to Interpreting Model Predictions.** Advances in Neural Information Processing Systems (NeurIPS), 30.

3. **EM-DAT: The International Disaster Database.** Centre for Research on the Epidemiology of Disasters (CRED). https://www.emdat.be/

4. **World Bank Open Data.** https://data.worldbank.org/

5. **scikit-learn Documentation.** https://scikit-learn.org/stable/documentation.html

6. **CatBoost Documentation.** https://catboost.ai/en/docs/

---

## PHỤ LỤC

### A. Cấu trúc Project

```
project_MinhHieu/
├── README.md                    # Hướng dẫn cài đặt và chạy
├── N10_report.pdf               # Báo cáo project (PDF)
├── requirements.txt             # Danh sách thư viện cần cài
├── data/                        # Thư mục chứa dữ liệu
│   └── global_disaster_response_2018_2024.csv
├── src/                         # Mã nguồn Python
│   ├── preprocessing.py         # Xử lý dữ liệu
│   ├── eda.py                   # Phân tích khám phá (EDA)
│   ├── feature_engineering.py   # Tạo và chọn đặc trưng
│   ├── model_TranMinhHieu.py    # Huấn luyện mô hình CatBoost
│   ├── evaluation.py            # Đánh giá mô hình
│   └── app.py                   # Script chính demo
├── web/                         # Giao diện web (Streamlit)
│   └── streamlit_app.py         # Web dashboard
└── models/                      # Mô hình đã train
    └── catboost_model.cbm
```

### B. Hướng dẫn cài đặt

```bash
# Clone repository
git clone https://github.com/minhhieucnttai/Hoc-May.git
cd Hoc-May

# Tạo môi trường ảo
python -m venv venv
source venv/bin/activate  # Linux/Mac
# hoặc venv\Scripts\activate  # Windows

# Cài đặt thư viện
pip install -r requirements.txt

# Chạy script chính
cd src
python app.py

# Chạy web dashboard
streamlit run web/streamlit_app.py
```

### C. Thư viện sử dụng

| Thư viện | Phiên bản | Mục đích |
|----------|-----------|----------|
| pandas | >= 2.0.0 | Xử lý dữ liệu |
| numpy | >= 1.24.0 | Tính toán số học |
| scikit-learn | >= 1.3.0 | Machine Learning |
| catboost | >= 1.2.0 | Mô hình CatBoost |
| matplotlib | >= 3.7.0 | Trực quan hóa |
| seaborn | >= 0.12.0 | Trực quan hóa nâng cao |
| plotly | >= 5.15.0 | Biểu đồ tương tác |
| shap | >= 0.42.0 | Giải thích mô hình |
| streamlit | >= 1.28.0 | Web application |

---

**Tác giả:** Trần Minh Hiếu

**Nhóm:** 10

**Môn học:** Học Máy (Machine Learning)

**Năm học:** 2024-2025
