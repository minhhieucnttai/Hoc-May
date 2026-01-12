# 🌍 Dự Đoán Số Ngày Phục Hồi Sau Thảm Họa
## (Recovery Days Prediction After Global Disasters)

### Machine Learning Project - Môn Học Máy

---

## 📋 Giới thiệu

Project này xây dựng mô hình Machine Learning để dự đoán **số ngày phục hồi (recovery_days)** sau các thảm họa tự nhiên trên toàn cầu. Đây là bài toán **hồi quy (Regression)** sử dụng **CatBoost Regressor**.

**Tác giả:** Trần Minh Hiếu

---

## 📁 Cấu trúc Project

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

---

## 🚀 Hướng dẫn cài đặt

### Yêu cầu hệ thống
- Python 3.8+
- pip (Python package manager)

### Các bước cài đặt

1. **Clone repository:**
```bash
git clone https://github.com/minhhieucnttai/Hoc-May.git
cd Hoc-May
```

2. **Tạo môi trường ảo (khuyến nghị):**
```bash
python -m venv venv

# Windows
venv\Scripts\activate

# Linux/Mac
source venv/bin/activate
```

3. **Cài đặt thư viện:**
```bash
pip install -r requirements.txt
```

---

## 🎯 Cách chạy Project

### 1. Chạy Pipeline chính (Script)
```bash
cd src
python app.py
```

### 2. Chạy Web Dashboard (Streamlit)
```bash
streamlit run web/streamlit_app.py
```
Sau đó mở trình duyệt tại: `http://localhost:8501`

---

## 📊 Mô tả dữ liệu

### Dataset: Global Disaster Response 2018-2024
- **Quy mô:** ~50,000 bản ghi
- **Thời gian:** 2018 - 2024
- **Biến mục tiêu:** `recovery_days` (số ngày phục hồi)

### Các biến đầu vào:

| Biến | Loại | Mô tả |
|------|------|-------|
| date | Datetime | Ngày xảy ra thảm họa |
| country | Categorical | Quốc gia |
| disaster_type | Categorical | Loại thảm họa (Earthquake, Flood, ...) |
| severity_index | Numerical | Chỉ số nghiêm trọng (1-10) |
| casualties | Numerical | Số thương vong |
| economic_loss_usd | Numerical | Thiệt hại kinh tế (USD) |
| response_time_hours | Numerical | Thời gian phản ứng (giờ) |
| aid_amount_usd | Numerical | Số tiền viện trợ (USD) |
| response_efficiency_score | Numerical | Điểm hiệu quả phản ứng |
| latitude, longitude | Numerical | Tọa độ địa lý |

---

## 🤖 Mô hình Machine Learning

### Mô hình chính: **CatBoost Regressor**

#### Lý do chọn CatBoost:
- ✅ Xử lý tốt biến phân loại (country, disaster_type) - không cần One-Hot Encoding
- ✅ Bắt được quan hệ phi tuyến giữa các biến
- ✅ Hiệu suất cao với dataset vừa-lớn (50k dòng)
- ✅ Ít overfitting nhờ Ordered Boosting
- ✅ Hỗ trợ giải thích mô hình (Feature Importance, SHAP)

#### Hyperparameter Tuning:
```python
param_grid = {
    'iterations': [300, 500, 800],
    'learning_rate': [0.01, 0.05, 0.1],
    'depth': [4, 6, 8, 10],
    'l2_leaf_reg': [1, 3, 5, 7]
}
```

---

## 📈 Đánh giá mô hình

### Các chỉ số đánh giá (Regression Metrics):

| Chỉ số | Ý nghĩa |
|--------|---------|
| **MAE** | Mean Absolute Error - Sai số tuyệt đối trung bình |
| **RMSE** | Root Mean Squared Error - Phạt nặng lỗi lớn |
| **R²** | Coefficient of Determination - Mức độ giải thích phương sai |
| **MAPE** | Mean Absolute Percentage Error - Sai số phần trăm |

> ⚠️ **Lưu ý:** Đây là bài toán HỒI QUY, không sử dụng Confusion Matrix, Precision/Recall/F1, ROC-AUC.

---

## 🌐 Web Dashboard

Web app được xây dựng với **Streamlit**, bao gồm:

1. **📊 Tổng quan dữ liệu** - Thống kê mô tả dataset
2. **📈 Phân tích EDA** - Biểu đồ phân bố, tương quan
3. **🤖 Huấn luyện mô hình** - Tùy chỉnh tham số và train
4. **🎯 Dự đoán** - Nhập thông tin và xem kết quả
5. **📋 Về Project** - Thông tin chi tiết

---

## 📚 Tài liệu tham khảo

1. Prokhorenkova et al., *CatBoost: unbiased boosting with categorical features*, NeurIPS, 2018
2. Lundberg & Lee, *A Unified Approach to Interpreting Model Predictions*, NeurIPS, 2017
3. EM-DAT: The International Disaster Database
4. World Bank Open Data
5. scikit-learn Documentation

---

## 📄 License

This project is for educational purposes - Machine Learning Course Project.

---

## 👨‍💻 Liên hệ

**Trần Minh Hiếu**
- GitHub: [minhhieucnttai](https://github.com/minhhieucnttai)