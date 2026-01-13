# -*- coding: utf-8 -*-
"""
Backup: Single Model App (XGBoost Only)
Phiên bản app chỉ có 1 model duy nhất, trước khi thêm LightGBM
"""

import streamlit as st
import pandas as pd
import numpy as np
import pickle
from pathlib import Path
import json
from sklearn.preprocessing import StandardScaler
import plotly.express as px
import plotly.graph_objects as go
import warnings

warnings.filterwarnings('ignore')

st.set_page_config(
    page_title="Dự đoán Thời gian Khôi phục Thảm họa",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    body { background-color: #f5f5f5; }
    .stTabs [data-baseweb="tab-list"] button [data-testid="stMarkdownContainer"] p { font-size: 1.1em; }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_xgboost_model():
    """Load XGBoost model and components"""
    try:
        model_path = Path(__file__).parent / 'xgboost_model.pkl'
        scaler_path = Path(__file__).parent / 'xgboost_scaler.pkl'
        encoders_path = Path(__file__).parent / 'xgboost_encoders.pkl'
        config_path = Path(__file__).parent / 'xgboost_config.json'
        
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
        
        with open(scaler_path, 'rb') as f:
            scaler = pickle.load(f)
        
        with open(encoders_path, 'rb') as f:
            encoders = pickle.load(f)
        
        with open(config_path, 'r') as f:
            config = json.load(f)
        
        return model, scaler, encoders, config
    except Exception as e:
        st.error(f"❌ Lỗi load model: {e}")
        return None, None, None, None


@st.cache_data
def load_data():
    """Load training data"""
    try:
        data_paths = [
            Path(__file__).parent / 'data' / 'global_disaster_response_2018_2024.csv',
            Path(__file__).parent.parent / 'data' / 'global_disaster_response_2018_2024.csv',
        ]
        
        for data_path in data_paths:
            if data_path.exists():
                df = pd.read_csv(data_path)
                if 'date' in df.columns:
                    df['date'] = pd.to_datetime(df['date'], errors='coerce')
                return df
        return None
    except Exception as e:
        st.error(f"❌ Lỗi load dữ liệu: {e}")
        return None


def make_prediction(model, scaler, encoders, features, input_data):
    """Make prediction using XGBoost"""
    try:
        X = pd.DataFrame([input_data])
        X_scaled = scaler.transform(X[features])
        prediction = model.predict(X_scaled)[0]
        return prediction
    except Exception as e:
        st.error(f"❌ Lỗi dự đoán: {e}")
        return None


def main():
    """Main function"""
    st.title("🌍 Dự đoán Thời gian Khôi phục Thảm họa")
    st.write("*Single Model Version - XGBoost Only (Backup)*")
    
    model, scaler, encoders, config = load_xgboost_model()
    df = load_data()
    
    if model is None:
        st.error("❌ Model không sẵn có!")
        return
    
    features = config.get('features', [])
    
    # Sidebar
    with st.sidebar:
        st.header("⚙️ Menu")
        page = st.radio("Chọn trang:", [
            "📊 Tổng quan",
            "🔍 Khám phá dữ liệu",
            "📈 Trực quan hóa",
            "🤖 Thông tin Model",
            "🔮 Dự đoán",
            "ℹ️ Về ứng dụng"
        ])
    
    # Pages
    if page == "📊 Tổng quan":
        st.header("📊 Tổng quan")
        if df is not None:
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("📦 Bản ghi", len(df))
            col2.metric("🌍 Quốc gia", df['country'].nunique() if 'country' in df.columns else 0)
            col3.metric("⚠️ Loại thảm họa", df['disaster_type'].nunique() if 'disaster_type' in df.columns else 0)
            col4.metric("📅 Năm", f"{int(df['date'].dt.year.min())}-{int(df['date'].dt.year.max())}" if 'date' in df.columns else "N/A")
    
    elif page == "🔍 Khám phá dữ liệu":
        st.header("🔍 Khám phá dữ liệu")
        if df is not None:
            st.write(f"**Tổng cộng:** {len(df)} bản ghi")
            st.dataframe(df.head(100), use_container_width=True)
    
    elif page == "📈 Trực quan hóa":
        st.header("📈 Trực quan hóa")
        if df is not None:
            fig = px.histogram(df, x='recovery_days', nbins=50, 
                             title='📊 Phân bố Thời gian Khôi phục')
            st.plotly_chart(fig, use_container_width=True)
    
    elif page == "🤖 Thông tin Model":
        st.header("🤖 Thông tin Model - XGBoost")
        st.write("**Model Type:** XGBoost Regressor")
        st.write("**Random State:** 42 (Deterministic)")
        
        if config:
            metrics = config.get('metrics', {})
            col1, col2, col3 = st.columns(3)
            col1.metric("R² Score", f"{metrics.get('r2', 0)*100:.2f}%")
            col2.metric("MAE", f"{metrics.get('mae', 0):.4f} ngày")
            col3.metric("RMSE", f"{metrics.get('rmse', 0):.4f} ngày")
    
    elif page == "🔮 Dự đoán":
        st.header("🔮 Dự đoán Thời gian Khôi phục")
        
        col1, col2 = st.columns(2)
        
        input_data = {}
        
        with col1:
            st.write("**Thông tin sự kiện**")
            input_data['severity_index'] = st.slider("Mức độ nghiêm trọng (1-10)", 1.0, 10.0, 5.0)
            input_data['casualties'] = st.number_input("Số người thiệt mạng", 0, 100000, 100)
            input_data['economic_loss_usd'] = st.number_input("Tổn thất kinh tế (USD)", 0, 10000000000, 1000000)
        
        with col2:
            st.write("**Phản ứng và hiệu suất**")
            input_data['response_time_hours'] = st.number_input("Thời gian phản ứng (giờ)", 0, 1000, 24)
            input_data['aid_amount_usd'] = st.number_input("Hỗ trợ (USD)", 0, 10000000000, 1000000)
            input_data['response_efficiency_score'] = st.slider("Hiệu suất phản ứng (0-100)", 0.0, 100.0, 50.0)
        
        col3, col4 = st.columns(2)
        with col3:
            input_data['latitude'] = st.number_input("Vĩ độ", -90.0, 90.0, 0.0)
            input_data['longitude'] = st.number_input("Kinh độ", -180.0, 180.0, 0.0)
        
        with col4:
            input_data['year'] = st.number_input("Năm", 2018, 2024, 2024)
            input_data['month'] = st.number_input("Tháng", 1, 12, 1)
        
        if st.button("🔮 Dự đoán", key="predict_btn"):
            prediction = make_prediction(model, scaler, encoders, features, input_data)
            if prediction:
                st.success(f"✅ **Dự đoán:** {prediction:.1f} ngày khôi phục")
                st.info(f"💡 Model XGBoost dự đoán thời gian khôi phục xấp xỉ {int(prediction)} ngày")
    
    elif page == "ℹ️ Về ứng dụng":
        st.header("ℹ️ Về ứng dụng")
        st.write("**Phiên bản:** Single Model (XGBoost only) - BACKUP")
        st.write("**Trạng thái:** Phiên bản cũ - không còn sử dụng")
        st.info("💡 Đây là phiên bản chỉ có 1 model XGBoost. Ứng dụng hiện tại sử dụng cả XGBoost và LightGBM để so sánh.")


if __name__ == "__main__":
    main()
