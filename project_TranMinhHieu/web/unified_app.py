# -*- coding: utf-8 -*-

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
import pickle
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Import CatBoost
try:
    from catboost import CatBoostRegressor
    HAS_CATBOOST = True
except ImportError:
    HAS_CATBOOST = False

# Import SHAP
try:
    import shap
    HAS_SHAP = True
except ImportError:
    HAS_SHAP = False

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
# HÀM HỖ TRỢ
# =========================================================
@st.cache_data
def load_real_data():
    """Load dữ liệu thực từ file CSV."""
    base_path = Path(__file__).parent
    
    # Tìm file dữ liệu
    data_paths = [
        base_path / "data" / "global_disaster_response_2018_2024.csv",
        base_path.parent / "data" / "global_disaster_response_2018_2024.csv",
        base_path.parent / "src" / "data" / "global_disaster_response_2018_2024.csv",
    ]
    
    for path in data_paths:
        if path.exists():
            df = pd.read_csv(path)
            return df, str(path)
    
    return None, None


def preprocess_data(df):
    """Tiền xử lý dữ liệu."""
    df = df.copy()
    
    # Convert date
    if 'date' in df.columns:
        df['date'] = pd.to_datetime(df['date'], dayfirst=True, errors='coerce')
        df['year'] = df['date'].dt.year
        df['month'] = df['date'].dt.month
        df['quarter'] = df['date'].dt.quarter
        # Drop date column to avoid Arrow serialization issues
        df = df.drop(columns=['date'])
    
    # Handle missing values
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        if df[col].isnull().any():
            df[col] = df[col].fillna(df[col].median())
    
    df = df.dropna()
    return df


def engineer_features(df):
    """Tạo các features mới."""
    df = df.copy()
    
    # Log transforms
    if 'economic_loss_usd' in df.columns:
        df['economic_loss_log'] = np.log1p(df['economic_loss_usd'])
    if 'aid_amount_usd' in df.columns:
        df['aid_amount_log'] = np.log1p(df['aid_amount_usd'])
    if 'casualties' in df.columns:
        df['casualties_log'] = np.log1p(df['casualties'])
    
    # Ratio features
    if 'aid_amount_usd' in df.columns and 'economic_loss_usd' in df.columns:
        df['aid_coverage_ratio'] = df['aid_amount_usd'] / (df['economic_loss_usd'] + 1)
    
    if 'casualties' in df.columns and 'response_time_hours' in df.columns:
        df['casualty_per_hour'] = df['casualties'] / (df['response_time_hours'] + 1)
    
    return df


def prepare_features_for_model(df, target_col='recovery_days'):
    """Chuẩn bị features cho model."""
    df = df.copy()
    
    # Numeric features
    numeric_features = [
        'severity_index', 'casualties', 'economic_loss_usd',
        'response_time_hours', 'aid_amount_usd', 'response_efficiency_score',
        'latitude', 'longitude'
    ]
    
    # Add time features
    if 'year' in df.columns:
        numeric_features.append('year')
    if 'month' in df.columns:
        numeric_features.append('month')
    
    # Add engineered features
    for col in ['economic_loss_log', 'aid_amount_log', 'casualties_log', 
                'aid_coverage_ratio', 'casualty_per_hour']:
        if col in df.columns:
            numeric_features.append(col)
    
    # Categorical features - encode them
    categorical_features = ['country', 'disaster_type']
    label_encoders = {}
    
    for col in categorical_features:
        if col in df.columns:
            le = LabelEncoder()
            df[col + '_encoded'] = le.fit_transform(df[col].astype(str))
            label_encoders[col] = le
            numeric_features.append(col + '_encoded')
    
    # Filter available features
    available_features = [f for f in numeric_features if f in df.columns]
    
    X = df[available_features].copy()
    y = df[target_col].copy() if target_col in df.columns else None
    
    return X, y, label_encoders, available_features


def calculate_metrics(y_true, y_pred):
    """Tính các metrics đánh giá."""
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    
    # MAPE
    mask = y_true != 0
    if mask.sum() > 0:
        mape = np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100
    else:
        mape = 0.0
    
    return {
        'MAE': mae,
        'RMSE': rmse,
        'R2': r2,
        'MAPE': mape
    }


# =========================================================
# LOAD DỮ LIỆU
# =========================================================
df_raw, data_path = load_real_data()

if df_raw is None:
    st.error("❌ Không tìm thấy file dữ liệu! Vui lòng kiểm tra thư mục data/")
    st.stop()

# Preprocess và engineer features
df = preprocess_data(df_raw)
df = engineer_features(df)


# =========================================================
# SIDEBAR
# =========================================================
st.sidebar.title("🌍 Recovery Days Prediction")
st.sidebar.markdown("---")

st.sidebar.success(f"✅ Đã load dữ liệu thực")
st.sidebar.caption(f"📁 {Path(data_path).name}")

st.sidebar.markdown("---")
st.sidebar.markdown("""
### 📊 Thông tin Dataset
""")
st.sidebar.write(f"- Số bản ghi: **{len(df):,}**")
st.sidebar.write(f"- Số features: **{len(df.columns)}**")
st.sidebar.write(f"- Recovery Days: **{df['recovery_days'].min():.0f} - {df['recovery_days'].max():.0f}**")

st.sidebar.markdown("---")
st.sidebar.markdown("""
### 🏷️ Disaster Types
""")
for dtype in df['disaster_type'].unique()[:5]:
    count = len(df[df['disaster_type'] == dtype])
    st.sidebar.write(f"- {dtype}: {count:,}")


# =========================================================
# HEADER CHÍNH
# =========================================================
st.markdown('<h1 class="main-header">🌍 Dự Đoán Số Ngày Phục Hồi Sau Thảm Họa</h1>', unsafe_allow_html=True)
st.markdown("""
<p style="text-align: center; font-size: 1.1rem; color: #666;">
    Machine Learning Project - Sử dụng CatBoost Regressor để dự đoán recovery_days<br>
    <b>Dataset: Global Disaster Response 2018-2024</b>
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
        'Type': [str(t) for t in df.dtypes.values],
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
        st.subheader("Recovery Days theo Country (Top 15)")
        country_mean = df.groupby('country')['recovery_days'].mean().sort_values(ascending=False).head(15)
        fig = px.bar(x=country_mean.index, y=country_mean.values,
                    labels={'x': 'Country', 'y': 'Mean Recovery Days'})
        fig.update_layout(xaxis_tickangle=-45)
        st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("---")
    
    st.subheader("Ma trận tương quan")
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    # Chọn các cột chính để hiển thị
    main_cols = ['recovery_days', 'severity_index', 'casualties', 'economic_loss_usd',
                 'response_time_hours', 'aid_amount_usd', 'response_efficiency_score']
    main_cols = [c for c in main_cols if c in numeric_cols]
    corr_matrix = df[main_cols].corr()
    fig = px.imshow(corr_matrix, text_auto='.2f', aspect='auto',
                   color_continuous_scale='RdBu_r')
    st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("---")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Scatter: Severity vs Recovery Days")
        fig = px.scatter(df.sample(min(2000, len(df)), random_state=42), 
                        x='severity_index', y='recovery_days',
                        opacity=0.5, color='disaster_type')
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("Scatter: Economic Loss vs Recovery Days")
        df_sample = df.sample(min(2000, len(df)), random_state=42)
        fig = px.scatter(df_sample, 
                        x=np.log1p(df_sample['economic_loss_usd']), 
                        y='recovery_days',
                        opacity=0.5, color='disaster_type',
                        labels={'x': 'Log(Economic Loss USD)'})
        st.plotly_chart(fig, use_container_width=True)


# =========================================================
# TAB 3: HUẤN LUYỆN MÔ HÌNH
# =========================================================
with tab3:
    st.header("🤖 Huấn luyện mô hình CatBoost")
    
    if not HAS_CATBOOST:
        st.error("❌ CatBoost chưa được cài đặt. Vui lòng chạy: pip install catboost")
        st.stop()
    
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
            # Chuẩn bị dữ liệu
            X, y, encoders, feature_names = prepare_features_for_model(df, 'recovery_days')
            
            # Chia dữ liệu
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=42
            )
            
            # Xác định categorical features
            cat_features = [i for i, col in enumerate(X.columns) if '_encoded' in col]
            
            # Train model
            model = CatBoostRegressor(
                iterations=iterations,
                learning_rate=learning_rate,
                depth=depth,
                l2_leaf_reg=l2_leaf_reg,
                random_state=42,
                verbose=False
            )
            
            model.fit(X_train, y_train, cat_features=cat_features)
            
            # Dự đoán và đánh giá
            y_pred = model.predict(X_test)
            metrics = calculate_metrics(y_test.values, y_pred)
            
            # Lưu vào session state
            st.session_state['model'] = model
            st.session_state['X'] = X
            st.session_state['feature_names'] = feature_names
            st.session_state['encoders'] = encoders
            st.session_state['cat_features'] = cat_features
            
            st.success("✅ Đã huấn luyện xong mô hình!")
            
            # Hiển thị kết quả
            st.markdown("### 📊 Kết quả đánh giá")
            
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("MAE", f"{metrics['MAE']:.2f} ngày")
            with col2:
                st.metric("RMSE", f"{metrics['RMSE']:.2f} ngày")
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
                importances = model.get_feature_importance()
                importance_df = pd.DataFrame({
                    'feature': X.columns,
                    'importance': importances
                }).sort_values('importance', ascending=False)
                
                fig = px.bar(importance_df.head(15), x='importance', y='feature',
                           orientation='h')
                fig.update_layout(yaxis={'categoryorder': 'total ascending'})
                st.plotly_chart(fig, use_container_width=True)
            
            # SHAP Explainability
            if HAS_SHAP:
                st.markdown("---")
                st.subheader("🔍 SHAP Explainability")
                
                with st.spinner("Đang tính SHAP values..."):
                    sample_size = min(500, len(X_test))
                    X_sample = X_test.sample(n=sample_size, random_state=42)
                    
                    explainer = shap.Explainer(model)
                    shap_values = explainer(X_sample)
                    
                    fig, ax = plt.subplots(figsize=(10, 6))
                    shap.summary_plot(shap_values, X_sample, show=False)
                    st.pyplot(fig)
                    plt.close()


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
        country = st.selectbox("Quốc gia", sorted(df['country'].unique()))
        disaster_type = st.selectbox("Loại thảm họa", sorted(df['disaster_type'].unique()))
        severity_index = st.slider("Chỉ số nghiêm trọng (1-10)", 1.0, 10.0, 5.0)
        casualties = st.number_input("Số thương vong", 0, 100000, 100)
        economic_loss = st.number_input("Thiệt hại kinh tế (USD)", 0, 1000000000, 1000000)
    
    with col2:
        response_time = st.slider("Thời gian phản ứng (giờ)", 1.0, 500.0, 24.0)
        aid_amount = st.number_input("Số tiền viện trợ (USD)", 0, 500000000, 500000)
        efficiency_score = st.slider("Điểm hiệu quả phản ứng (0-1)", 0.0, 1.0, 0.5)
        latitude = st.slider("Vĩ độ", -90.0, 90.0, 0.0)
        longitude = st.slider("Kinh độ", -180.0, 180.0, 0.0)
    
    if st.button("🔮 Dự đoán", type="primary"):
        if 'model' not in st.session_state:
            st.warning("⚠️ Vui lòng huấn luyện mô hình trước (Tab 'Huấn luyện mô hình')")
        else:
            model = st.session_state['model']
            encoders = st.session_state['encoders']
            X_template = st.session_state['X']
            
            # Encode categorical features
            country_encoded = encoders['country'].transform([country])[0] if country in encoders['country'].classes_ else 0
            disaster_encoded = encoders['disaster_type'].transform([disaster_type])[0] if disaster_type in encoders['disaster_type'].classes_ else 0
            
            # Tạo input data
            input_data = pd.DataFrame({col: [0] for col in X_template.columns})
            
            # Fill values
            input_data['severity_index'] = severity_index
            input_data['casualties'] = casualties
            input_data['economic_loss_usd'] = economic_loss
            input_data['response_time_hours'] = response_time
            input_data['aid_amount_usd'] = aid_amount
            input_data['response_efficiency_score'] = efficiency_score
            input_data['latitude'] = latitude
            input_data['longitude'] = longitude
            
            if 'year' in input_data.columns:
                input_data['year'] = 2024
            if 'month' in input_data.columns:
                input_data['month'] = 6
            
            if 'country_encoded' in input_data.columns:
                input_data['country_encoded'] = country_encoded
            if 'disaster_type_encoded' in input_data.columns:
                input_data['disaster_type_encoded'] = disaster_encoded
            
            # Engineered features
            if 'economic_loss_log' in input_data.columns:
                input_data['economic_loss_log'] = np.log1p(economic_loss)
            if 'aid_amount_log' in input_data.columns:
                input_data['aid_amount_log'] = np.log1p(aid_amount)
            if 'casualties_log' in input_data.columns:
                input_data['casualties_log'] = np.log1p(casualties)
            if 'aid_coverage_ratio' in input_data.columns:
                input_data['aid_coverage_ratio'] = aid_amount / (economic_loss + 1)
            if 'casualty_per_hour' in input_data.columns:
                input_data['casualty_per_hour'] = casualties / (response_time + 1)
            
            # Dự đoán
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
            
            # So sánh với dữ liệu
            st.markdown("---")
            st.subheader("📊 So sánh với dữ liệu thực")
            
            similar = df[df['disaster_type'] == disaster_type]
            if len(similar) > 0:
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric(f"Mean ({disaster_type})", f"{similar['recovery_days'].mean():.1f} ngày")
                with col2:
                    st.metric(f"Min ({disaster_type})", f"{similar['recovery_days'].min():.1f} ngày")
                with col3:
                    st.metric(f"Max ({disaster_type})", f"{similar['recovery_days'].max():.1f} ngày")


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
    - **Features chính**: severity_index, casualties, economic_loss_usd, response_time_hours, aid_amount_usd
    
    ### 3. Mô hình được chọn: CatBoost Regressor
    
    **Lý do chọn:**
    - ✅ Xử lý tốt biến phân loại (country, disaster_type) - không cần One-Hot Encoding
    - ✅ Bắt được quan hệ phi tuyến giữa các biến
    - ✅ Hiệu suất cao với dataset vừa-lớn (50k dòng)
    - ✅ Ít overfitting nhờ Ordered Boosting
    - ✅ Hỗ trợ giải thích mô hình (Feature Importance, SHAP)
    
    ### 4. Pipeline
    
    ```
    Data Loading → Preprocessing → Feature Engineering → Training → Evaluation → Prediction
    ```
    
    ### 5. Đánh giá mô hình
    
    Sử dụng các chỉ số **hồi quy**:
    - **MAE** (Mean Absolute Error): Sai số trung bình tuyệt đối
    - **RMSE** (Root Mean Squared Error): Căn bậc hai sai số bình phương trung bình
    - **R² Score**: Hệ số xác định
    - **MAPE** (Mean Absolute Percentage Error): Phần trăm sai số trung bình
    
    ### 6. Kết quả đạt được
    
    - **R² Score**: ~0.94 (94% variance được giải thích)
    - **MAE**: ~4 ngày
    - **RMSE**: ~5 ngày
    
    ### 7. Tác giả
    
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
    <p>Tác giả: Trần Minh Hiếu | © 2026</p>
</div>
""", unsafe_allow_html=True)
