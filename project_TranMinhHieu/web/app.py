import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
import sys
import os

# Thêm đường dẫn src vào path
sys.path.insert(0, os.path.join(os.path.dirname("project_TranMinhHieu/web/data/global_disaster_response_2018_2024.csv"), '..', 'src'))

from preprocessing import load_data, preprocess_data, get_categorical_features
from eda import perform_eda, get_data_summary
from feature_engineering import engineer_features
from model_TranMinhHieu import (
    prepare_data_for_catboost,
    train_optimized_model,
    get_feature_importance,
    load_model,
    save_model
)
from evaluation import calculate_metrics, plot_actual_vs_predicted

# =========================================================
# CẤU HÌNH TRANG
# =========================================================
st.set_page_config(
    page_title="Recovery Days Prediction",
    page_icon="🌍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =========================================================
# CSS TÙY CHỈNH
# =========================================================
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1E88E5;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        text-align: center;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 24px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        padding-top: 10px;
        padding-bottom: 10px;
    }
</style>
""", unsafe_allow_html=True)


# =========================================================
# HÀM TẠO DỮ LIỆU MẪU
# =========================================================
@st.cache_data
def create_sample_data(n_samples: int = 10000) -> pd.DataFrame:
    """Tạo dữ liệu mẫu cho demo."""
    np.random.seed(42)
    
    countries = ['USA', 'Japan', 'China', 'India', 'Brazil', 'Germany', 'UK', 
                 'France', 'Australia', 'Canada', 'Mexico', 'Indonesia', 
                 'Philippines', 'Bangladesh', 'Pakistan']
    
    disaster_types = ['Earthquake', 'Flood', 'Tornado', 'Hurricane', 'Wildfire',
                      'Tsunami', 'Drought', 'Volcanic Eruption', 'Landslide', 'Storm']
    
    data = {
        'date': pd.date_range('2018-01-01', periods=n_samples, freq='h')[:n_samples],
        'country': np.random.choice(countries, n_samples),
        'disaster_type': np.random.choice(disaster_types, n_samples),
        'severity_index': np.random.uniform(1, 10, n_samples),
        'casualties': np.random.exponential(100, n_samples).astype(int),
        'economic_loss_usd': np.random.exponential(1e6, n_samples),
        'response_time_hours': np.random.exponential(24, n_samples),
        'aid_amount_usd': np.random.exponential(5e5, n_samples),
        'response_efficiency_score': np.random.uniform(0, 1, n_samples),
        'latitude': np.random.uniform(-90, 90, n_samples),
        'longitude': np.random.uniform(-180, 180, n_samples)
    }
    
    df = pd.DataFrame(data)
    
    df['recovery_days'] = (
        10 + 
        df['severity_index'] * 5 + 
        np.log1p(df['economic_loss_usd']) * 0.5 +
        df['response_time_hours'] * 0.3 -
        np.log1p(df['aid_amount_usd']) * 0.2 -
        df['response_efficiency_score'] * 10 +
        np.random.normal(0, 10, n_samples)
    ).clip(lower=1)
    
    return df


# =========================================================
# SIDEBAR
# =========================================================
st.sidebar.title("🌍 Recovery Days Prediction")
st.sidebar.markdown("---")

# Chọn nguồn dữ liệu
data_source = st.sidebar.radio(
    "📁 Nguồn dữ liệu:",
    ["Dữ liệu mẫu", "Tải lên file CSV"]
)

if data_source == "Tải lên file CSV":
    uploaded_file = st.sidebar.file_uploader("Chọn file CSV", type=['csv'])
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
    else:
        st.sidebar.warning("Vui lòng tải lên file CSV hoặc sử dụng dữ liệu mẫu")
        df = create_sample_data()
else:
    df = create_sample_data()

st.sidebar.markdown("---")
st.sidebar.markdown("""
### 📊 Thông tin Dataset
""")
st.sidebar.write(f"- Số bản ghi: {len(df):,}")
st.sidebar.write(f"- Số features: {len(df.columns)}")


# =========================================================
# HEADER CHÍNH
# =========================================================
st.markdown('<h1 class="main-header">🌍 Dự Đoán Số Ngày Phục Hồi Sau Thảm Họa</h1>', unsafe_allow_html=True)
st.markdown("""
<p style="text-align: center; font-size: 1.1rem; color: #666;">
    Machine Learning Project - Sử dụng CatBoost Regressor để dự đoán recovery_days
</p>
""", unsafe_allow_html=True)

# =========================================================
# TABS CHÍNH
# =========================================================
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📊 Tổng quan dữ liệu",
    "📈 Phân tích EDA", 
    "🤖 Huấn luyện mô hình",
    "🎯 Dự đoán",
    "📋 Về Project"
])

# =========================================================
# TAB 1: TỔNG QUAN DỮ LIỆU
# =========================================================
with tab1:
    st.header("📊 Tổng quan dữ liệu")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Số bản ghi", f"{len(df):,}")
    with col2:
        st.metric("Số features", len(df.columns))
    with col3:
        st.metric("Recovery Days (Mean)", f"{df['recovery_days'].mean():.1f}")
    with col4:
        st.metric("Recovery Days (Median)", f"{df['recovery_days'].median():.1f}")
    
    st.markdown("---")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📋 Mẫu dữ liệu")
        st.dataframe(df.head(10), use_container_width=True)
    
    with col2:
        st.subheader("📈 Thống kê mô tả")
        st.dataframe(df.describe(), use_container_width=True)
    
    st.markdown("---")
    st.subheader("🔍 Kiểu dữ liệu các cột")
    dtype_df = pd.DataFrame({
        'Column': df.columns,
        'Type': df.dtypes.values,
        'Non-Null': df.notnull().sum().values,
        'Missing': df.isnull().sum().values
    })
    st.dataframe(dtype_df, use_container_width=True)


# =========================================================
# TAB 2: PHÂN TÍCH EDA
# =========================================================
with tab2:
    st.header("📈 Phân tích khám phá dữ liệu (EDA)")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Phân bố Recovery Days")
        fig = px.histogram(df, x='recovery_days', nbins=50, 
                          color_discrete_sequence=['steelblue'])
        fig.update_layout(
            xaxis_title="Số ngày phục hồi",
            yaxis_title="Tần suất"
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("Boxplot Recovery Days")
        fig = px.box(df, y='recovery_days', color_discrete_sequence=['steelblue'])
        fig.update_layout(yaxis_title="Số ngày phục hồi")
        st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("---")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Recovery Days theo Disaster Type")
        fig = px.box(df, x='disaster_type', y='recovery_days', 
                    color='disaster_type')
        fig.update_layout(xaxis_tickangle=-45, showlegend=False)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("Recovery Days theo Country")
        country_mean = df.groupby('country')['recovery_days'].mean().sort_values(ascending=False)
        fig = px.bar(x=country_mean.index, y=country_mean.values,
                    labels={'x': 'Country', 'y': 'Mean Recovery Days'})
        fig.update_layout(xaxis_tickangle=-45)
        st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("---")
    
    st.subheader("Ma trận tương quan")
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    corr_matrix = df[numeric_cols].corr()
    fig = px.imshow(corr_matrix, text_auto='.2f', aspect='auto',
                   color_continuous_scale='RdBu_r')
    st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("---")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Scatter: Severity vs Recovery Days")
        fig = px.scatter(df.sample(min(1000, len(df))), 
                        x='severity_index', y='recovery_days',
                        opacity=0.5)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("Scatter: Economic Loss vs Recovery Days")
        df_sample = df.sample(min(1000, len(df)))
        fig = px.scatter(df_sample, 
                        x=np.log1p(df_sample['economic_loss_usd']), 
                        y='recovery_days',
                        opacity=0.5,
                        labels={'x': 'Log(Economic Loss USD)'})
        st.plotly_chart(fig, use_container_width=True)


# =========================================================
# TAB 3: HUẤN LUYỆN MÔ HÌNH
# =========================================================
with tab3:
    st.header("🤖 Huấn luyện mô hình CatBoost")
    
    st.markdown("""
    ### Mô hình: CatBoost Regressor
    
    **Lý do chọn CatBoost:**
    - ✅ Xử lý tốt biến phân loại (country, disaster_type)
    - ✅ Không cần One-Hot Encoding
    - ✅ Hiệu suất cao với dataset vừa-lớn
    - ✅ Ít overfitting với Ordered Boosting
    """)
    
    st.markdown("---")
    
    col1, col2 = st.columns(2)
    
    with col1:
        test_size = st.slider("Test size (%)", 10, 40, 20) / 100
        iterations = st.slider("Số iterations", 100, 1000, 300)
        learning_rate = st.select_slider("Learning rate", 
                                        options=[0.01, 0.03, 0.05, 0.1, 0.2],
                                        value=0.1)
    
    with col2:
        depth = st.slider("Depth", 4, 12, 6)
        l2_leaf_reg = st.slider("L2 regularization", 1, 10, 3)
    
    if st.button("🚀 Huấn luyện mô hình", type="primary"):
        with st.spinner("Đang huấn luyện mô hình..."):
            # Tiền xử lý
            X, y = preprocess_data(df, target_column='recovery_days')
            X = engineer_features(X)
            cat_features = get_categorical_features(X)
            
            # Chia dữ liệu
            X_train, X_test, y_train, y_test, cat_indices = prepare_data_for_catboost(
                X, y, cat_features, test_size=test_size, random_state=42
            )
            
            # Huấn luyện
            params = {
                'iterations': iterations,
                'learning_rate': learning_rate,
                'depth': depth,
                'l2_leaf_reg': l2_leaf_reg
            }
            
            model = train_optimized_model(
                X_train, y_train,
                cat_features=cat_indices,
                params=params
            )
            
            # Dự đoán và đánh giá
            y_pred = model.predict(X_test)
            metrics = calculate_metrics(y_test.values, y_pred)
            
            # Lưu vào session state
            st.session_state['model'] = model
            st.session_state['X'] = X
            st.session_state['cat_features'] = cat_features
            
            st.success("✅ Đã huấn luyện xong mô hình!")
            
            # Hiển thị kết quả
            st.markdown("### 📊 Kết quả đánh giá")
            
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("MAE", f"{metrics['MAE']:.2f}")
            with col2:
                st.metric("RMSE", f"{metrics['RMSE']:.2f}")
            with col3:
                st.metric("R² Score", f"{metrics['R2']:.4f}")
            with col4:
                st.metric("MAPE", f"{metrics['MAPE']:.2f}%")
            
            st.markdown("---")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Actual vs Predicted")
                fig = px.scatter(x=y_test, y=y_pred, opacity=0.3,
                               labels={'x': 'Actual', 'y': 'Predicted'})
                fig.add_trace(go.Scatter(
                    x=[y_test.min(), y_test.max()],
                    y=[y_test.min(), y_test.max()],
                    mode='lines',
                    name='Perfect Prediction',
                    line=dict(color='red', dash='dash')
                ))
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                st.subheader("Feature Importance")
                importance_df = get_feature_importance(model, X.columns.tolist())
                fig = px.bar(importance_df.head(15), x='importance', y='feature',
                           orientation='h')
                fig.update_layout(yaxis={'categoryorder': 'total ascending'})
                st.plotly_chart(fig, use_container_width=True)


# =========================================================
# TAB 4: DỰ ĐOÁN
# =========================================================
with tab4:
    st.header("🎯 Dự đoán số ngày phục hồi")
    
    st.markdown("""
    Nhập thông tin về thảm họa để dự đoán số ngày phục hồi:
    """)
    
    col1, col2 = st.columns(2)
    
    with col1:
        country = st.selectbox("Quốc gia", df['country'].unique())
        disaster_type = st.selectbox("Loại thảm họa", df['disaster_type'].unique())
        severity_index = st.slider("Chỉ số nghiêm trọng (1-10)", 1.0, 10.0, 5.0)
        casualties = st.number_input("Số thương vong", 0, 10000, 100)
        economic_loss = st.number_input("Thiệt hại kinh tế (USD)", 0, 100000000, 1000000)
    
    with col2:
        response_time = st.slider("Thời gian phản ứng (giờ)", 1.0, 168.0, 24.0)
        aid_amount = st.number_input("Số tiền viện trợ (USD)", 0, 50000000, 500000)
        efficiency_score = st.slider("Điểm hiệu quả phản ứng (0-1)", 0.0, 1.0, 0.5)
        latitude = st.slider("Vĩ độ", -90.0, 90.0, 0.0)
        longitude = st.slider("Kinh độ", -180.0, 180.0, 0.0)
    
    if st.button("🔮 Dự đoán", type="primary"):
        if 'model' not in st.session_state:
            st.warning("⚠️ Vui lòng huấn luyện mô hình trước (Tab 'Huấn luyện mô hình')")
        else:
            # Tạo input data
            input_data = pd.DataFrame({
                'country': [country],
                'disaster_type': [disaster_type],
                'severity_index': [severity_index],
                'casualties': [casualties],
                'economic_loss_usd': [economic_loss],
                'response_time_hours': [response_time],
                'aid_amount_usd': [aid_amount],
                'response_efficiency_score': [efficiency_score],
                'latitude': [latitude],
                'longitude': [longitude],
                'year': [2024],
                'month': [6]
            })
            
            # Feature engineering
            # Disable time feature creation since we already have year/month from input
            input_data = engineer_features(input_data, create_time=False)
            
            # Log transform
            input_data['economic_loss_usd_log'] = np.log1p(input_data['economic_loss_usd'])
            input_data['aid_amount_usd_log'] = np.log1p(input_data['aid_amount_usd'])
            
            # Đảm bảo có đủ các cột như training data
            X_train = st.session_state['X']
            for col in X_train.columns:
                if col not in input_data.columns:
                    if X_train[col].dtype == 'object':
                        input_data[col] = X_train[col].mode()[0]
                    else:
                        input_data[col] = 0
            
            # Sắp xếp lại cột
            input_data = input_data[X_train.columns]
            
            # Dự đoán
            model = st.session_state['model']
            prediction = model.predict(input_data)[0]
            
            st.markdown("---")
            st.markdown(f"""
            <div style="text-align: center; padding: 2rem; background-color: #e8f4f8; border-radius: 1rem;">
                <h2 style="color: #1E88E5;">Kết quả dự đoán</h2>
                <h1 style="font-size: 4rem; color: #0D47A1;">{prediction:.1f}</h1>
                <h3>ngày phục hồi</h3>
            </div>
            """, unsafe_allow_html=True)
            
            # Phân tích
            st.markdown("---")
            st.subheader("📝 Phân tích")
            
            if prediction < 30:
                st.success("🟢 Dự kiến phục hồi NHANH (< 1 tháng)")
            elif prediction < 90:
                st.warning("🟡 Dự kiến phục hồi TRUNG BÌNH (1-3 tháng)")
            else:
                st.error("🔴 Dự kiến phục hồi CHẬM (> 3 tháng)")


# =========================================================
# TAB 5: VỀ PROJECT
# =========================================================
with tab5:
    st.header("📋 Về Project")
    
    st.markdown("""
    ## Dự đoán số ngày phục hồi sau thảm họa toàn cầu
    
    ### 1. Giới thiệu
    
    Project này xây dựng mô hình Machine Learning để dự đoán **số ngày phục hồi (recovery_days)** 
    sau các thảm họa tự nhiên trên toàn cầu. Đây là bài toán **hồi quy (Regression)**.
    
    ### 2. Dataset
    
    - **Nguồn**: Global Disaster Response 2018-2024
    - **Quy mô**: ~50,000 bản ghi
    - **Biến mục tiêu**: recovery_days
    
    ### 3. Mô hình được chọn: CatBoost Regressor
    
    **Lý do chọn:**
    - ✅ Xử lý tốt biến phân loại (country, disaster_type) - không cần One-Hot Encoding
    - ✅ Bắt được quan hệ phi tuyến giữa các biến
    - ✅ Hiệu suất cao với dataset vừa-lớn (50k dòng)
    - ✅ Ít overfitting nhờ Ordered Boosting
    - ✅ Hỗ trợ giải thích mô hình (Feature Importance, SHAP)
    
    ### 4. Pipeline
    
    ```
    Data → Preprocessing → EDA → Feature Engineering → Training → Evaluation
    ```
    
    ### 5. Đánh giá mô hình
    
    Sử dụng các chỉ số **hồi quy**:
    - **MAE** (Mean Absolute Error)
    - **RMSE** (Root Mean Squared Error)
    - **R² Score**
    - **MAPE** (Mean Absolute Percentage Error)
    
    ### 6. Tác giả
    
    **Trần Minh Hiếu**
    
    ---
    
    ### 📚 Tài liệu tham khảo
    
    1. Prokhorenkova et al., *CatBoost: unbiased boosting with categorical features*, NeurIPS, 2018
    2. Lundberg & Lee, *A Unified Approach to Interpreting Model Predictions*, NeurIPS, 2017
    3. EM-DAT: The International Disaster Database
    4. World Bank Open Data
    """)


# =========================================================
# FOOTER
# =========================================================
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #888; padding: 1rem;">
    <p>📚 Machine Learning Project - Recovery Days Prediction</p>
    <p>Tác giả: Trần Minh Hiếu | © 2024</p>
</div>
""", unsafe_allow_html=True)
