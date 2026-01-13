# -*- coding: utf-8 -*-
"""
Web Dashboard - Streamlit Application
=====================================
Giao diện web trực quan để dự đoán số ngày phục hồi sau thảm họa.

Chức năng:
- Hiển thị Actual vs Predicted Recovery Days
- Trực quan Feature Importance
- Trình bày SHAP Explainability
- So sánh nhiều mô hình
- Thể hiện tính minh bạch – ứng dụng thực tế

Chạy ứng dụng:
    streamlit run app.py

Tác giả: Trần Minh Hiếu
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import shap
import pickle
import os
from pathlib import Path

# =========================================================
# CẤU HÌNH TRANG
# =========================================================
st.set_page_config(page_title="Disaster Recovery Prediction", layout="wide")

# =========================================================
# CÁC HÀM HỖ TRỢ
# =========================================================
@st.cache_data
def load_test_data():
    """Load dữ liệu test từ file CSV."""
    base_path = Path(__file__).parent
    X_test = pd.read_csv(base_path / "X_test.csv")
    y_test = pd.read_csv(base_path / "y_test.csv").values.ravel()
    return X_test, y_test


@st.cache_resource
def load_models():
    """Load các mô hình đã train."""
    base_path = Path(__file__).parent
    models = {}
    
    model_files = {
        "CatBoost": "model_catboost.pkl",
        "Random Forest": "model_rf.pkl",
        "XGBoost": "model_xgb.pkl"
    }
    
    for name, filename in model_files.items():
        filepath = base_path / filename
        if filepath.exists():
            with open(filepath, 'rb') as f:
                models[name] = pickle.load(f)
    
    return models


# =========================================================
# LOAD DỮ LIỆU VÀ MÔ HÌNH
# =========================================================
try:
    X_test, y_test = load_test_data()
    models = load_models()
    data_loaded = True
except Exception as e:
    st.error(f"❌ Lỗi khi load dữ liệu: {e}")
    data_loaded = False

# =========================================================
# SIDEBAR - CHỌN MÔ HÌNH
# =========================================================
st.sidebar.title("Chọn mô hình")

if data_loaded and models:
    model_name = st.sidebar.selectbox(
        "Mô hình dự đoán",
        list(models.keys())
    )
    
    model = models[model_name]
    y_pred = model.predict(X_test)
else:
    st.sidebar.warning("⚠️ Chưa load được mô hình")
    model_name = None
    model = None
    y_pred = None

st.sidebar.markdown("---")
st.sidebar.markdown("""
### 📊 Thông tin
- **Dataset**: Global Disaster Response 2018-2024
- **Target**: Recovery Days (số ngày phục hồi)
- **Task**: Regression
""")

# =========================================================
# HEADER CHÍNH
# =========================================================
st.title("🌍 Dự Đoán Số Ngày Phục Hồi Sau Thảm Họa")
st.markdown("""
<p style="font-size: 1.1rem; color: #666;">
    Machine Learning Project - Sử dụng CatBoost, Random Forest, XGBoost để dự đoán recovery_days
</p>
""", unsafe_allow_html=True)

if not data_loaded:
    st.warning("⚠️ Vui lòng chuẩn bị dữ liệu và mô hình trước khi sử dụng ứng dụng.")
    st.stop()

# =========================================================
# 5.4. BIỂU ĐỒ ĐÁNH GIÁ
# =========================================================
st.header("5.4. Biểu đồ đánh giá")

# =========================================================
# 5.4.1. BIỂU ĐỒ THỰC TẾ VS DỰ ĐOÁN
# =========================================================
st.subheader("5.4.1. Thực tế vs Dự đoán")

if y_pred is not None:
    col1, col2 = st.columns([2, 1])
    
    with col1:
        fig, ax = plt.subplots(figsize=(7, 6))
        ax.scatter(y_test, y_pred, alpha=0.4)
        ax.plot([y_test.min(), y_test.max()],
                [y_test.min(), y_test.max()],
                "r--", linewidth=2)
        
        ax.set_xlabel("Actual Recovery Days")
        ax.set_ylabel("Predicted Recovery Days")
        ax.set_title(f"Actual vs Predicted Recovery Days ({model_name})")
        
        st.pyplot(fig)
        plt.close()
    
    with col2:
        # Hiển thị metrics
        from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
        
        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        r2 = r2_score(y_test, y_pred)
        
        st.metric("MAE", f"{mae:.2f} ngày")
        st.metric("RMSE", f"{rmse:.2f} ngày")
        st.metric("R² Score", f"{r2:.4f}")
    
    st.markdown("""
    📌 **Ý nghĩa**: Biểu đồ cho thấy mức độ phù hợp giữa giá trị dự đoán và thực tế. 
    Các điểm càng gần đường y = x thì mô hình dự đoán càng chính xác.
    """)
else:
    st.warning("⚠️ Chưa có dữ liệu dự đoán")

st.markdown("---")

# =========================================================
# 5.4.2. FEATURE IMPORTANCE
# =========================================================
st.subheader("5.4.2. Feature Importance")

if model is not None:
    try:
        # Lấy feature importance dựa trên loại model
        if model_name == "CatBoost":
            importances = model.get_feature_importance()
        elif hasattr(model, 'feature_importances_'):
            importances = model.feature_importances_
        else:
            importances = None
        
        if importances is not None:
            feature_names = X_test.columns
            
            fi_df = pd.DataFrame({
                "Feature": feature_names,
                "Importance": importances
            }).sort_values(by="Importance", ascending=False)
            
            col1, col2 = st.columns([2, 1])
            
            with col1:
                fig, ax = plt.subplots(figsize=(8, 6))
                top_n = min(10, len(fi_df))
                ax.barh(fi_df["Feature"][:top_n], fi_df["Importance"][:top_n])
                ax.invert_yaxis()
                ax.set_title(f"Top {top_n} Feature Importance ({model_name})")
                ax.set_xlabel("Importance")
                
                st.pyplot(fig)
                plt.close()
            
            with col2:
                st.markdown("### Top Features")
                st.dataframe(fi_df.head(10), use_container_width=True)
        else:
            st.info("ℹ️ Feature Importance không khả dụng cho mô hình này")
    except Exception as e:
        st.warning(f"⚠️ Không thể hiển thị Feature Importance: {e}")
else:
    st.info("Feature Importance chi tiết nhất được trình bày với CatBoost")

st.markdown("---")

# =========================================================
# SHAP EXPLAINABILITY
# =========================================================
st.subheader("🔍 SHAP Explainability")

if model is not None and model_name == "CatBoost":
    try:
        st.markdown("""
        **SHAP (SHapley Additive exPlanations)** giúp giải thích đóng góp của từng đặc trưng 
        vào kết quả dự đoán, làm tăng tính minh bạch và khả năng ứng dụng thực tế của mô hình.
        """)
        
        with st.spinner("Đang tính SHAP values..."):
            # Sử dụng một subset nhỏ để tính nhanh hơn
            sample_size = min(100, len(X_test))
            X_sample = X_test.sample(n=sample_size, random_state=42)
            
            explainer = shap.Explainer(model)
            shap_values = explainer(X_sample)
            
            fig, ax = plt.subplots(figsize=(10, 6))
            shap.summary_plot(shap_values, X_sample, show=False)
            st.pyplot(fig)
            plt.close()
        
        st.markdown("""
        📌 **Giải thích**: 
        - Mỗi điểm đại diện cho một mẫu dữ liệu
        - Màu đỏ = giá trị đặc trưng cao, Màu xanh = giá trị đặc trưng thấp
        - Vị trí trên trục X cho biết tác động đến dự đoán (dương/âm)
        """)
    except Exception as e:
        st.warning(f"⚠️ Không thể hiển thị SHAP: {e}")
else:
    st.info("ℹ️ SHAP Explainability được hiển thị tốt nhất với mô hình CatBoost. Vui lòng chọn CatBoost từ sidebar.")

st.markdown("---")

# =========================================================
# 5.5. SO SÁNH NHIỀU MÔ HÌNH
# =========================================================
st.header("5.5. So sánh nhiều mô hình")

comparison_df = pd.DataFrame({
    "Mô hình": ["Linear Regression", "Random Forest", "XGBoost", "LightGBM", "CatBoost"],
    "RMSE": ["cao", "khá", "tốt", "tốt", "tốt nhất"],
    "R²": ["thấp", "trung bình", "cao", "cao", "cao nhất"]
})

st.table(comparison_df)

# Nếu có nhiều model, hiển thị so sánh thực tế
if len(models) > 1:
    st.subheader("So sánh chi tiết các mô hình")
    
    comparison_results = []
    for name, m in models.items():
        pred = m.predict(X_test)
        mae = mean_absolute_error(y_test, pred)
        rmse = np.sqrt(mean_squared_error(y_test, pred))
        r2 = r2_score(y_test, pred)
        comparison_results.append({
            "Mô hình": name,
            "MAE": f"{mae:.2f}",
            "RMSE": f"{rmse:.2f}",
            "R²": f"{r2:.4f}"
        })
    
    comparison_detail_df = pd.DataFrame(comparison_results)
    st.dataframe(comparison_detail_df, use_container_width=True)

st.markdown("---")

# =========================================================
# 6. KẾT LUẬN VÀ HƯỚNG PHÁT TRIỂN
# =========================================================
st.header("6. Kết luận và hướng phát triển")

st.markdown("""
### Kết luận
Nghiên cứu đã xây dựng thành công mô hình học máy dự đoán số ngày phục hồi sau thảm họa toàn cầu.
Kết quả cho thấy **CatBoost Regressor vượt trội nhất** nhờ khả năng xử lý biến phân loại,
mô hình hóa quan hệ phi tuyến và đạt hiệu suất dự đoán cao.

### Hướng phát triển
- Bổ sung dữ liệu chính sách và hạ tầng
- Áp dụng mô hình spatio-temporal
- Dự đoán theo kịch bản *what-if*
- Triển khai hệ thống hỗ trợ quyết định
""")

# =========================================================
# FOOTER
# =========================================================
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #888; padding: 1rem;">
    <p>📚 Machine Learning Project - Disaster Recovery Prediction</p>
    <p>Tác giả: Trần Minh Hiếu | © 2024</p>
</div>
""", unsafe_allow_html=True)
