# Global Disaster Response Analysis (2018-2024)
# Phân Tích Phản Ứng Thiên Tai Toàn Cầu (2018-2024)

## Project Overview / Tổng Quan Dự Án

This project analyzes global disaster response data from 2018 to 2024, implementing machine learning models to predict and analyze disaster response patterns.

Dự án này phân tích dữ liệu phản ứng thiên tai toàn cầu từ 2018 đến 2024, sử dụng các mô hình machine learning để dự đoán và phân tích mô hình phản ứng thiên tai.

## 📚 Documentation / Tài Liệu

- **[SETUP_INSTRUCTIONS.md](SETUP_INSTRUCTIONS.md)** - Quick start guide / Hướng dẫn bắt đầu nhanh
- **[DOCUMENTATION.md](DOCUMENTATION.md)** - Detailed documentation / Tài liệu chi tiết
- **[TUTORIAL.md](TUTORIAL.md)** - Step-by-step tutorials / Hướng dẫn từng bước
- **[API_REFERENCE.md](API_REFERENCE.md)** - Complete API reference / Tài liệu tham khảo API đầy đủ

## Project Structure
```
project_TranMinhHieu/
│
├── README.md
├── N10_report.pdf        ✅ (đã tạo sẵn cho bạn)
├── requirements.txt
│
├── data/
│   └── global_disaster_response_2018_2024.csv
│
├── src/
│   ├── preprocessing.py
│   ├── eda.py
│   ├── feature_engineering.py
│   ├── model_TranMinhHieu.py
│   ├── evaluation.py
│   └── main.py
│
└── web/
    └── app.py            ✅ Streamlit Web App
```

## Installation

```bash
pip install -r requirements.txt
```

## Usage

### Run the main analysis
```bash
python src/main.py
```

### Launch the web application
```bash
streamlit run web/app.py
```

## Components / Các Thành Phần

### Source Modules / Mô-đun Nguồn

- **preprocessing.py**: Data cleaning and preprocessing / Làm sạch và tiền xử lý dữ liệu
- **eda.py**: Exploratory Data Analysis / Phân tích dữ liệu khám phá
- **feature_engineering.py**: Feature extraction and engineering (30+ features) / Trích xuất và kỹ thuật đặc trưng (30+ đặc trưng)
- **model_TranMinhHieu.py**: Machine learning models (RF, GB, Linear, DT) / Mô hình machine learning
- **evaluation.py**: Model evaluation and metrics / Đánh giá mô hình và metrics
- **main.py**: Main entry point for the analysis pipeline / Điểm vào chính cho pipeline phân tích

### Web Application / Ứng Dụng Web

- **app.py**: Streamlit web application with 6 interactive pages / Ứng dụng web Streamlit với 6 trang tương tác
  - Overview / Tổng quan
  - Data Explorer / Khám phá dữ liệu
  - Visualizations / Trực quan hóa
  - Statistics / Thống kê
  - Predictions / Dự đoán
  - About / Giới thiệu

## Features / Tính Năng

✅ **Complete ML Pipeline** / Pipeline ML đầy đủ
- Data preprocessing with missing value handling / Tiền xử lý với xử lý giá trị thiếu
- 30+ engineered features / 30+ đặc trưng kỹ thuật
- 4 ML models with cross-validation / 4 mô hình ML với kiểm chứng chéo
- Comprehensive evaluation metrics / Metrics đánh giá toàn diện

✅ **Interactive Dashboard** / Dashboard tương tác
- Real-time data filtering / Lọc dữ liệu thời gian thực
- Interactive Plotly visualizations / Trực quan hóa Plotly tương tác
- CSV export functionality / Chức năng xuất CSV
- Response effectiveness predictor / Dự đoán hiệu quả phản ứng

✅ **Extensive Documentation** / Tài liệu mở rộng
- Vietnamese and English support / Hỗ trợ tiếng Việt và tiếng Anh
- API reference / Tài liệu tham khảo API
- Step-by-step tutorials / Hướng dẫn từng bước
- Code examples / Ví dụ mã
