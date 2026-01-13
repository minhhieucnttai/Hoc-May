# -*- coding: utf-8 -*-
"""
Backup: App Random Forest Version
Dùng để so sánh hoặc khôi phục nếu cần
"""

import streamlit as st
import pandas as pd
import numpy as np
import pickle
from pathlib import Path
import json
import plotly.express as px
import plotly.graph_objects as go

# Cấu hình Streamlit
st.set_page_config(
    page_title="Dự đoán Thời gian Khôi phục Thảm họa - Random Forest",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Add custom CSS
st.markdown("""
<style>
    body { background-color: #f5f5f5; }
    .stTabs [data-baseweb="tab-list"] button [data-testid="stMarkdownContainer"] p { font-size: 1.1em; }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_model_and_config():
    """Load trained model and configuration"""
    try:
        model_path = Path(__file__).parent / 'random_forest_model.pkl'
        config_path = Path(__file__).parent / 'random_forest_config.json'
        encoders_path = Path(__file__).parent / 'random_forest_encoders.pkl'
        
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
        
        with open(config_path, 'r') as f:
            config = json.load(f)
        
        with open(encoders_path, 'rb') as f:
            encoders = pickle.load(f)
        
        return model, config, encoders
    except Exception as e:
        st.error(f"❌ Lỗi load model: {e}")
        return None, None, None


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


def main():
    """Main function"""
    st.title("🌍 Dự đoán Thời gian Khôi phục Thảm họa")
    st.write("*Random Forest Model Version (Backup)*")
    
    model, config, encoders = load_model_and_config()
    df = load_data()
    
    if model is None:
        st.error("❌ Model không sẵn có!")
        return
    
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
            col1, col2, col3 = st.columns(3)
            col1.metric("📦 Tổng bản ghi", len(df))
            col2.metric("🌍 Quốc gia", df['country'].nunique())
            col3.metric("⚠️ Loại thảm họa", df['disaster_type'].nunique())
    
    elif page == "🔍 Khám phá dữ liệu":
        st.header("🔍 Khám phá dữ liệu")
        if df is not None:
            st.dataframe(df.head(100), use_container_width=True)
    
    elif page == "📈 Trực quan hóa":
        st.header("📈 Trực quan hóa")
        if df is not None:
            fig = px.histogram(df, x='recovery_days', nbins=50, title='Phân bố Thời gian Khôi phục')
            st.plotly_chart(fig, use_container_width=True)
    
    elif page == "🤖 Thông tin Model":
        st.header("🤖 Thông tin Model")
        st.write("**Model Type:** Random Forest (Backup)")
        if config:
            st.json(config)
    
    elif page == "🔮 Dự đoán":
        st.header("🔮 Dự đoán Thời gian Khôi phục")
        
        col1, col2 = st.columns(2)
        with col1:
            severity = st.slider("Mức độ nghiêm trọng", 1.0, 10.0, 5.0)
            casualties = st.number_input("Số người thiệt mạng", 0, 100000, 100)
            economic_loss = st.number_input("Tổn thất kinh tế (USD)", 0, 1000000000, 1000000)
        
        with col2:
            response_time = st.number_input("Thời gian phản ứng (giờ)", 0, 1000, 24)
            aid_amount = st.number_input("Hỗ trợ (USD)", 0, 1000000000, 1000000)
            efficiency = st.slider("Hiệu suất phản ứng", 0.0, 100.0, 50.0)
        
        if st.button("🔮 Dự đoán"):
            st.success("✅ Dự đoán: ~10 ngày")
    
    elif page == "ℹ️ Về ứng dụng":
        st.header("ℹ️ Về ứng dụng")
        st.write("**Phiên bản:** Backup - Random Forest")
        st.write("**Trạng thái:** Không còn sử dụng")
        st.info("💡 Đây là phiên bản cũ dùng Random Forest. Ứng dụng hiện tại sử dụng XGBoost và LightGBM.")


if __name__ == "__main__":
    main()
