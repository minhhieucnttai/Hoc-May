# -*- coding: utf-8 -*-
"""
Unified Web Dashboard - Streamlit Application
==============================================
Ứng dụng tích hợp đầy đủ tính năng dự đoán số ngày phục hồi sau thảm họa.

Tính năng:
- Tổng quan dữ liệu và EDA
- Dự đoán với XGBoost & LightGBM
- Dữ liệu chính sách và hạ tầng
- Mô hình không gian-thời gian
- Dự đoán kịch bản What-If
- Hệ thống hỗ trợ quyết định

Chạy: streamlit run unified_app.py
Tác giả: Trần Minh Hiếu
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import pickle
import json
import warnings
from pathlib import Path
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

warnings.filterwarnings('ignore')

# =========================================================
# CẤU HÌNH TRANG
# =========================================================
st.set_page_config(
    page_title="Hệ Thống Dự Đoán Phục Hồi Thảm Họa",
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
        font-size: 2.2rem;
        font-weight: bold;
        color: #1E88E5;
        text-align: center;
        margin-bottom: 1rem;
        padding: 1rem;
        background: linear-gradient(90deg, #e3f2fd 0%, #bbdefb 100%);
        border-radius: 10px;
    }
    .sub-header {
        font-size: 1rem;
        color: #666;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    .info-box {
        background-color: #e8f4f8;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #1E88E5;
        margin: 1rem 0;
    }
    .warning-box {
        background-color: #fff3e0;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #ff9800;
        margin: 1rem 0;
    }
    .success-box {
        background-color: #e8f5e9;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #4caf50;
        margin: 1rem 0;
    }
    .decision-card {
        background: white;
        padding: 1.5rem;
        border-radius: 10px;
        border: 2px solid #1E88E5;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# =========================================================
# DỮ LIỆU CHÍNH SÁCH VÀ HẠ TẦNG
# =========================================================
POLICY_DATA = {
    'United States': {'policy_score': 85, 'infrastructure_score': 90, 'early_warning': True, 'emergency_fund_usd': 50e9},
    'Japan': {'policy_score': 95, 'infrastructure_score': 95, 'early_warning': True, 'emergency_fund_usd': 30e9},
    'Germany': {'policy_score': 88, 'infrastructure_score': 92, 'early_warning': True, 'emergency_fund_usd': 20e9},
    'France': {'policy_score': 82, 'infrastructure_score': 85, 'early_warning': True, 'emergency_fund_usd': 15e9},
    'United Kingdom': {'policy_score': 80, 'infrastructure_score': 88, 'early_warning': True, 'emergency_fund_usd': 12e9},
    'China': {'policy_score': 75, 'infrastructure_score': 80, 'early_warning': True, 'emergency_fund_usd': 40e9},
    'India': {'policy_score': 60, 'infrastructure_score': 55, 'early_warning': True, 'emergency_fund_usd': 10e9},
    'Brazil': {'policy_score': 55, 'infrastructure_score': 60, 'early_warning': False, 'emergency_fund_usd': 5e9},
    'Indonesia': {'policy_score': 50, 'infrastructure_score': 50, 'early_warning': True, 'emergency_fund_usd': 3e9},
    'Philippines': {'policy_score': 55, 'infrastructure_score': 45, 'early_warning': True, 'emergency_fund_usd': 2e9},
    'Mexico': {'policy_score': 58, 'infrastructure_score': 62, 'early_warning': True, 'emergency_fund_usd': 4e9},
    'Australia': {'policy_score': 85, 'infrastructure_score': 88, 'early_warning': True, 'emergency_fund_usd': 8e9},
    'Spain': {'policy_score': 75, 'infrastructure_score': 78, 'early_warning': True, 'emergency_fund_usd': 6e9},
    'Italy': {'policy_score': 72, 'infrastructure_score': 75, 'early_warning': True, 'emergency_fund_usd': 7e9},
    'South Korea': {'policy_score': 88, 'infrastructure_score': 90, 'early_warning': True, 'emergency_fund_usd': 12e9},
    'Canada': {'policy_score': 82, 'infrastructure_score': 85, 'early_warning': True, 'emergency_fund_usd': 10e9},
    'Russia': {'policy_score': 65, 'infrastructure_score': 70, 'early_warning': True, 'emergency_fund_usd': 8e9},
    'Turkey': {'policy_score': 60, 'infrastructure_score': 65, 'early_warning': True, 'emergency_fund_usd': 5e9},
    'South Africa': {'policy_score': 50, 'infrastructure_score': 55, 'early_warning': False, 'emergency_fund_usd': 2e9},
    'Nigeria': {'policy_score': 40, 'infrastructure_score': 35, 'early_warning': False, 'emergency_fund_usd': 1e9},
    'Greece': {'policy_score': 68, 'infrastructure_score': 70, 'early_warning': True, 'emergency_fund_usd': 3e9},
}

# =========================================================
# HÀM LOAD DỮ LIỆU
# =========================================================
@st.cache_data
def load_data():
    """Load dữ liệu từ file CSV."""
    paths = [
        Path(__file__).parent / 'data' / 'global_disaster_response_2018_2024.csv',
        Path(__file__).parent.parent / 'data' / 'global_disaster_response_2018_2024.csv'
    ]
    for p in paths:
        if p.exists():
            df = pd.read_csv(p)
            if 'date' in df.columns:
                df['date'] = pd.to_datetime(df['date'], errors='coerce', dayfirst=True)
                df['year'] = df['date'].dt.year
                df['month'] = df['date'].dt.month
                df['day'] = df['date'].dt.day
            return df
    return None

@st.cache_resource
def load_models():
    """Load các mô hình đã train."""
    models = {}
    base_path = Path(__file__).parent
    
    # Load XGBoost
    try:
        xgb_model = pickle.load(open(base_path / 'xgboost_model.pkl', 'rb'))
        xgb_scaler = pickle.load(open(base_path / 'xgboost_scaler.pkl', 'rb'))
        xgb_encoders = pickle.load(open(base_path / 'xgboost_encoders.pkl', 'rb'))
        xgb_config = json.load(open(base_path / 'xgboost_config.json'))
        models['XGBoost'] = {'model': xgb_model, 'scaler': xgb_scaler, 'encoders': xgb_encoders, 'config': xgb_config}
    except Exception as e:
        st.warning(f"Không load được XGBoost: {e}")
    
    # Load LightGBM
    try:
        lgb_model = pickle.load(open(base_path / 'lightgbm_model.pkl', 'rb'))
        lgb_scaler = pickle.load(open(base_path / 'lightgbm_scaler.pkl', 'rb'))
        lgb_encoders = pickle.load(open(base_path / 'lightgbm_encoders.pkl', 'rb'))
        lgb_config = json.load(open(base_path / 'lightgbm_config.json'))
        models['LightGBM'] = {'model': lgb_model, 'scaler': lgb_scaler, 'encoders': lgb_encoders, 'config': lgb_config}
    except Exception as e:
        st.warning(f"Không load được LightGBM: {e}")
    
    return models

def predict_recovery(models, model_name, country, disaster_type, input_data):
    """Dự đoán số ngày phục hồi."""
    if model_name not in models:
        return None
    
    m = models[model_name]
    model, scaler, encoders, config = m['model'], m['scaler'], m['encoders'], m['config']
    
    d = input_data.copy()
    d['country_encoded'] = encoders['country'].transform([country])[0]
    d['disaster_type_encoded'] = encoders['disaster_type'].transform([disaster_type])[0]
    
    features = config.get('features', [])
    X = pd.DataFrame([[d.get(f, 0) for f in features]], columns=features)
    X_scaled = scaler.transform(X)
    
    return model.predict(X_scaled)[0]

def get_policy_factor(country):
    """Lấy hệ số điều chỉnh dựa trên chính sách và hạ tầng."""
    if country in POLICY_DATA:
        policy = POLICY_DATA[country]
        # Điểm trung bình chính sách và hạ tầng (0-100)
        avg_score = (policy['policy_score'] + policy['infrastructure_score']) / 2
        # Hệ số điều chỉnh: điểm cao = giảm thời gian phục hồi
        factor = 1 - (avg_score - 50) / 200  # Factor từ 0.75 đến 1.25
        return factor, policy
    return 1.0, None

# =========================================================
# TRANG: TỔNG QUAN
# =========================================================
def page_overview(df):
    st.markdown('<div class="main-header">📊 Tổng Quan Hệ Thống</div>', unsafe_allow_html=True)
    
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("📦 Tổng bản ghi", f"{len(df):,}")
    col2.metric("🌍 Quốc gia", df['country'].nunique())
    col3.metric("⚡ Loại thảm họa", df['disaster_type'].nunique())
    col4.metric("📅 Giai đoạn", f"{int(df['year'].min())}-{int(df['year'].max())}")
    
    st.divider()
    
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("📈 Thống kê Recovery Days")
        stats = df['recovery_days'].describe()
        st.dataframe(stats.round(2), use_container_width=True)
    
    with col2:
        st.subheader("🗺️ Phân bố theo quốc gia")
        country_counts = df['country'].value_counts().head(10)
        fig = px.bar(x=country_counts.index, y=country_counts.values, 
                    labels={'x': 'Quốc gia', 'y': 'Số lượng'})
        st.plotly_chart(fig, use_container_width=True)
    
    st.subheader("📋 Dữ liệu mẫu")
    st.dataframe(df.head(20), use_container_width=True, hide_index=True)

# =========================================================
# TRANG: PHÂN TÍCH EDA
# =========================================================
def page_eda(df):
    st.markdown('<div class="main-header">📈 Phân Tích Khám Phá Dữ Liệu (EDA)</div>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Phân bố Recovery Days")
        fig = px.histogram(df, x='recovery_days', nbins=50, color_discrete_sequence=['#1E88E5'])
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("Box Plot Recovery Days")
        fig = px.box(df, y='recovery_days', color_discrete_sequence=['#ff7043'])
        st.plotly_chart(fig, use_container_width=True)
    
    st.divider()
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Recovery theo Loại Thảm Họa")
        disaster_avg = df.groupby('disaster_type')['recovery_days'].mean().sort_values(ascending=False)
        fig = px.bar(x=disaster_avg.index, y=disaster_avg.values,
                    labels={'x': 'Loại thảm họa', 'y': 'TB ngày phục hồi'})
        fig.update_layout(xaxis_tickangle=-45)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("Recovery theo Năm")
        yearly_avg = df.groupby('year')['recovery_days'].mean()
        fig = px.line(x=yearly_avg.index, y=yearly_avg.values, markers=True,
                     labels={'x': 'Năm', 'y': 'TB ngày phục hồi'})
        st.plotly_chart(fig, use_container_width=True)
    
    st.divider()
    st.subheader("Ma trận tương quan")
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    corr = df[numeric_cols].corr()
    fig = px.imshow(corr, text_auto='.2f', color_continuous_scale='RdBu_r')
    st.plotly_chart(fig, use_container_width=True)

# =========================================================
# TRANG: CHÍNH SÁCH VÀ HẠ TẦNG
# =========================================================
def page_policy():
    st.markdown('<div class="main-header">🏛️ Dữ Liệu Chính Sách & Hạ Tầng</div>', unsafe_allow_html=True)
    
    st.markdown("""
    <div class="info-box">
    <b>Giới thiệu:</b> Dữ liệu chính sách và hạ tầng được sử dụng để điều chỉnh dự đoán,
    phản ánh khả năng ứng phó và phục hồi của từng quốc gia.
    </div>
    """, unsafe_allow_html=True)
    
    # Chuyển đổi thành DataFrame
    policy_df = pd.DataFrame([
        {'Quốc gia': k, 'Điểm Chính Sách': v['policy_score'], 
         'Điểm Hạ Tầng': v['infrastructure_score'],
         'Hệ thống Cảnh báo Sớm': '✅' if v['early_warning'] else '❌',
         'Quỹ Khẩn Cấp (tỷ USD)': v['emergency_fund_usd'] / 1e9}
        for k, v in POLICY_DATA.items()
    ])
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📊 Bảng Dữ Liệu")
        st.dataframe(policy_df.sort_values('Điểm Chính Sách', ascending=False), 
                    use_container_width=True, hide_index=True)
    
    with col2:
        st.subheader("📈 So Sánh Điểm Số")
        fig = go.Figure()
        fig.add_trace(go.Bar(name='Chính sách', x=policy_df['Quốc gia'], y=policy_df['Điểm Chính Sách']))
        fig.add_trace(go.Bar(name='Hạ tầng', x=policy_df['Quốc gia'], y=policy_df['Điểm Hạ Tầng']))
        fig.update_layout(barmode='group', xaxis_tickangle=-45)
        st.plotly_chart(fig, use_container_width=True)
    
    st.divider()
    st.subheader("🗺️ Bản Đồ Điểm Chính Sách")
    
    # Tạo map data với ISO codes
    country_iso = {
        'United States': 'USA', 'Japan': 'JPN', 'Germany': 'DEU', 'France': 'FRA',
        'United Kingdom': 'GBR', 'China': 'CHN', 'India': 'IND', 'Brazil': 'BRA',
        'Indonesia': 'IDN', 'Philippines': 'PHL', 'Mexico': 'MEX', 'Australia': 'AUS',
        'Spain': 'ESP', 'Italy': 'ITA', 'South Korea': 'KOR', 'Canada': 'CAN',
        'Russia': 'RUS', 'Turkey': 'TUR', 'South Africa': 'ZAF', 'Nigeria': 'NGA', 'Greece': 'GRC'
    }
    policy_df['ISO'] = policy_df['Quốc gia'].map(country_iso)
    
    fig = px.choropleth(policy_df, locations='ISO', color='Điểm Chính Sách',
                       hover_name='Quốc gia', color_continuous_scale='Viridis',
                       projection='natural earth')
    st.plotly_chart(fig, use_container_width=True)

# =========================================================
# TRANG: MÔ HÌNH KHÔNG GIAN-THỜI GIAN
# =========================================================
def page_spatiotemporal(df):
    st.markdown('<div class="main-header">🌐 Mô Hình Không Gian - Thời Gian</div>', unsafe_allow_html=True)
    
    st.markdown("""
    <div class="info-box">
    <b>Spatio-Temporal Analysis:</b> Phân tích mối quan hệ giữa vị trí địa lý, thời gian 
    và thời gian phục hồi sau thảm họa.
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🗺️ Bản đồ Thảm Họa")
        sample = df.sample(min(500, len(df)), random_state=42)
        fig = px.scatter_geo(sample, lat='latitude', lon='longitude', 
                            color='recovery_days', size='severity_index',
                            hover_data=['country', 'disaster_type'],
                            color_continuous_scale='RdYlGn_r',
                            projection='natural earth')
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("📅 Xu hướng theo Thời Gian")
        monthly = df.groupby(['year', 'month']).agg({
            'recovery_days': 'mean',
            'severity_index': 'mean'
        }).reset_index()
        monthly['date'] = pd.to_datetime(monthly[['year', 'month']].assign(day=1))
        
        fig = px.line(monthly, x='date', y='recovery_days', 
                     title='Trung bình Recovery Days theo tháng')
        st.plotly_chart(fig, use_container_width=True)
    
    st.divider()
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🌡️ Heatmap: Vĩ độ vs Thời gian")
        df['lat_bin'] = pd.cut(df['latitude'], bins=10, labels=False)
        heatmap_data = df.groupby(['year', 'lat_bin'])['recovery_days'].mean().reset_index()
        heatmap_pivot = heatmap_data.pivot(index='lat_bin', columns='year', values='recovery_days')
        fig = px.imshow(heatmap_pivot, labels={'color': 'Recovery Days'},
                       color_continuous_scale='RdYlGn_r')
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("📊 Phân tích theo Mùa")
        df['season'] = df['month'].apply(lambda m: 'Xuân' if m in [3,4,5] else 
                                        'Hè' if m in [6,7,8] else 
                                        'Thu' if m in [9,10,11] else 'Đông')
        season_data = df.groupby('season')['recovery_days'].mean()
        fig = px.bar(x=season_data.index, y=season_data.values,
                    labels={'x': 'Mùa', 'y': 'TB ngày phục hồi'})
        st.plotly_chart(fig, use_container_width=True)

# =========================================================
# TRANG: DỰ ĐOÁN
# =========================================================
def page_prediction(df, models):
    st.markdown('<div class="main-header">🎯 Dự Đoán Số Ngày Phục Hồi</div>', unsafe_allow_html=True)
    
    if not models:
        st.error("❌ Không load được models!")
        return
    
    st.markdown('<div class="success-box">✅ Sẵn sàng: XGBoost & LightGBM</div>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📝 Thông tin Thảm Họa")
        country = st.selectbox("🌍 Quốc gia:", sorted(df['country'].unique()))
        disaster = st.selectbox("⚡ Loại thảm họa:", sorted(df['disaster_type'].unique()))
        severity = st.slider("📊 Mức độ nghiêm trọng (1-10):", 1, 10, 5)
        casualties = st.number_input("👥 Số thương vong:", 0, 100000, 100)
        loss = st.number_input("💰 Thiệt hại kinh tế (USD):", 0, 100000000, 1000000)
    
    with col2:
        st.subheader("📝 Thông tin Phản Ứng")
        resp_time = st.slider("⏱️ Thời gian phản ứng (giờ):", 0, 48, 12)
        aid = st.number_input("💵 Viện trợ (USD):", 0, 10000000, 500000)
        eff = st.slider("📈 Hiệu quả phản ứng (0-100):", 0, 100, 70)
        year = st.number_input("📅 Năm:", 2018, 2030, 2024)
        month = st.slider("📅 Tháng:", 1, 12, 6)
        lat = st.number_input("🌐 Latitude:", -90.0, 90.0, 0.0)
        lon = st.number_input("🌐 Longitude:", -180.0, 180.0, 0.0)
    
    use_policy = st.checkbox("🏛️ Áp dụng hệ số chính sách/hạ tầng", value=True)
    
    if st.button("🔮 Dự Đoán!", use_container_width=True, type="primary"):
        input_data = {
            'severity_index': severity, 'casualties': casualties,
            'economic_loss_usd': loss, 'response_time_hours': resp_time,
            'aid_amount_usd': aid, 'response_efficiency_score': eff,
            'year': year, 'month': month, 'latitude': lat, 'longitude': lon
        }
        
        pred_xgb = predict_recovery(models, 'XGBoost', country, disaster, input_data)
        pred_lgb = predict_recovery(models, 'LightGBM', country, disaster, input_data)
        
        # Áp dụng hệ số chính sách
        policy_factor, policy_info = get_policy_factor(country)
        if use_policy and policy_info:
            pred_xgb_adj = pred_xgb * policy_factor
            pred_lgb_adj = pred_lgb * policy_factor
        else:
            pred_xgb_adj, pred_lgb_adj = pred_xgb, pred_lgb
        
        st.divider()
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown(f"""
            <div style='text-align:center;background:#ffe6e6;padding:20px;border-radius:10px;'>
            <h4>🔷 XGBoost</h4>
            <h2 style='color:#d9534f;'>{pred_xgb_adj:.1f}</h2><p>ngày</p>
            </div>""", unsafe_allow_html=True)
        
        with col2:
            st.markdown(f"""
            <div style='text-align:center;background:#e6f3ff;padding:20px;border-radius:10px;'>
            <h4>🔹 LightGBM</h4>
            <h2 style='color:#5cb85c;'>{pred_lgb_adj:.1f}</h2><p>ngày</p>
            </div>""", unsafe_allow_html=True)
        
        with col3:
            avg = (pred_xgb_adj + pred_lgb_adj) / 2
            st.markdown(f"""
            <div style='text-align:center;background:#f0f8ff;padding:20px;border-radius:10px;border:2px solid #1E88E5;'>
            <h4>📊 Trung Bình</h4>
            <h2 style='color:#1E88E5;'>{avg:.1f}</h2><p>ngày</p>
            </div>""", unsafe_allow_html=True)
        
        if policy_info:
            st.info(f"🏛️ Điều chỉnh theo chính sách {country}: Điểm CS={policy_info['policy_score']}, HT={policy_info['infrastructure_score']}, Hệ số={policy_factor:.3f}")

# =========================================================
# TRANG: KỊCH BẢN WHAT-IF
# =========================================================
def page_whatif(df, models):
    st.markdown('<div class="main-header">�� Dự Đoán Kịch Bản What-If</div>', unsafe_allow_html=True)
    
    st.markdown("""
    <div class="info-box">
    <b>What-If Analysis:</b> Phân tích ảnh hưởng của các yếu tố khác nhau đến thời gian phục hồi.
    Thay đổi các tham số để xem tác động.
    </div>
    """, unsafe_allow_html=True)
    
    if not models:
        st.error("❌ Models chưa sẵn sàng!")
        return
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📝 Kịch bản Cơ Sở")
        base_country = st.selectbox("Quốc gia:", sorted(df['country'].unique()), key='base_country')
        base_disaster = st.selectbox("Loại thảm họa:", sorted(df['disaster_type'].unique()), key='base_disaster')
        base_severity = st.slider("Mức độ:", 1, 10, 5, key='base_sev')
        base_resp_time = st.slider("Thời gian phản ứng (giờ):", 0, 48, 24, key='base_resp')
        base_eff = st.slider("Hiệu quả (%):", 0, 100, 50, key='base_eff')
    
    with col2:
        st.subheader("🔄 Kịch bản What-If")
        whatif_severity = st.slider("Mức độ:", 1, 10, base_severity, key='wf_sev')
        whatif_resp_time = st.slider("Thời gian phản ứng (giờ):", 0, 48, base_resp_time, key='wf_resp')
        whatif_eff = st.slider("Hiệu quả (%):", 0, 100, base_eff, key='wf_eff')
        whatif_aid = st.number_input("Viện trợ tăng thêm (USD):", 0, 10000000, 0, key='wf_aid')
    
    if st.button("📊 So Sánh Kịch Bản", use_container_width=True, type="primary"):
        base_input = {
            'severity_index': base_severity, 'casualties': 100,
            'economic_loss_usd': 1000000, 'response_time_hours': base_resp_time,
            'aid_amount_usd': 500000, 'response_efficiency_score': base_eff,
            'year': 2024, 'month': 6, 'latitude': 0, 'longitude': 0
        }
        
        whatif_input = base_input.copy()
        whatif_input.update({
            'severity_index': whatif_severity,
            'response_time_hours': whatif_resp_time,
            'response_efficiency_score': whatif_eff,
            'aid_amount_usd': 500000 + whatif_aid
        })
        
        base_pred = predict_recovery(models, 'XGBoost', base_country, base_disaster, base_input)
        whatif_pred = predict_recovery(models, 'XGBoost', base_country, base_disaster, whatif_input)
        
        diff = whatif_pred - base_pred
        pct_change = (diff / base_pred) * 100
        
        st.divider()
        
        col1, col2, col3 = st.columns(3)
        
        col1.metric("Kịch bản Cơ Sở", f"{base_pred:.1f} ngày")
        col2.metric("Kịch bản What-If", f"{whatif_pred:.1f} ngày", delta=f"{diff:+.1f} ngày")
        col3.metric("Thay đổi", f"{pct_change:+.1f}%")
        
        # Phân tích độ nhạy
        st.divider()
        st.subheader("📈 Phân Tích Độ Nhạy")
        
        # Thay đổi hiệu quả phản ứng
        eff_range = range(10, 100, 10)
        eff_predictions = []
        for e in eff_range:
            test_input = base_input.copy()
            test_input['response_efficiency_score'] = e
            pred = predict_recovery(models, 'XGBoost', base_country, base_disaster, test_input)
            eff_predictions.append(pred)
        
        fig = px.line(x=list(eff_range), y=eff_predictions, markers=True,
                     labels={'x': 'Hiệu quả phản ứng (%)', 'y': 'Ngày phục hồi'})
        fig.update_layout(title='Tác động của Hiệu quả Phản ứng')
        st.plotly_chart(fig, use_container_width=True)

# =========================================================
# TRANG: HỆ THỐNG HỖ TRỢ QUYẾT ĐỊNH
# =========================================================
def page_decision_support(df, models):
    st.markdown('<div class="main-header">🎛️ Hệ Thống Hỗ Trợ Quyết Định</div>', unsafe_allow_html=True)
    
    st.markdown("""
    <div class="info-box">
    <b>Decision Support System (DSS):</b> Cung cấp khuyến nghị hành động dựa trên phân tích dữ liệu
    và dự đoán mô hình để tối ưu hóa quá trình phục hồi.
    </div>
    """, unsafe_allow_html=True)
    
    if not models:
        st.error("❌ Models chưa sẵn sàng!")
        return
    
    st.subheader("📝 Nhập Thông Tin Tình Huống")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        country = st.selectbox("Quốc gia:", sorted(df['country'].unique()), key='dss_country')
        disaster = st.selectbox("Loại thảm họa:", sorted(df['disaster_type'].unique()), key='dss_disaster')
    
    with col2:
        severity = st.slider("Mức độ nghiêm trọng:", 1, 10, 5, key='dss_sev')
        casualties = st.number_input("Thương vong:", 0, 10000, 100, key='dss_cas')
    
    with col3:
        economic_loss = st.number_input("Thiệt hại (USD):", 0, 100000000, 1000000, key='dss_loss')
        current_eff = st.slider("Hiệu quả hiện tại (%):", 0, 100, 50, key='dss_eff')
    
    if st.button("🔍 Phân Tích & Đề Xuất", use_container_width=True, type="primary"):
        # Dự đoán hiện tại
        current_input = {
            'severity_index': severity, 'casualties': casualties,
            'economic_loss_usd': economic_loss, 'response_time_hours': 24,
            'aid_amount_usd': 500000, 'response_efficiency_score': current_eff,
            'year': 2024, 'month': 6, 'latitude': 0, 'longitude': 0
        }
        
        current_pred = predict_recovery(models, 'XGBoost', country, disaster, current_input)
        
        # Tính toán các kịch bản cải thiện
        scenarios = []
        
        # Kịch bản 1: Tăng hiệu quả
        improved_input = current_input.copy()
        improved_input['response_efficiency_score'] = min(100, current_eff + 20)
        pred1 = predict_recovery(models, 'XGBoost', country, disaster, improved_input)
        scenarios.append(('Tăng hiệu quả +20%', pred1, current_pred - pred1))
        
        # Kịch bản 2: Giảm thời gian phản ứng
        improved_input = current_input.copy()
        improved_input['response_time_hours'] = max(1, 24 - 12)
        pred2 = predict_recovery(models, 'XGBoost', country, disaster, improved_input)
        scenarios.append(('Giảm thời gian phản ứng 12h', pred2, current_pred - pred2))
        
        # Kịch bản 3: Tăng viện trợ
        improved_input = current_input.copy()
        improved_input['aid_amount_usd'] = 1000000
        pred3 = predict_recovery(models, 'XGBoost', country, disaster, improved_input)
        scenarios.append(('Tăng viện trợ gấp đôi', pred3, current_pred - pred3))
        
        st.divider()
        
        # Hiển thị kết quả
        col1, col2 = st.columns([1, 2])
        
        with col1:
            st.subheader("📊 Dự đoán Hiện tại")
            st.markdown(f"""
            <div style='text-align:center;background:#fff3e0;padding:30px;border-radius:10px;'>
            <h2 style='color:#ff9800;'>{current_pred:.1f}</h2>
            <p>ngày phục hồi</p>
            </div>
            """, unsafe_allow_html=True)
            
            # Đánh giá mức độ
            if current_pred < 30:
                st.success("🟢 Phục hồi NHANH")
            elif current_pred < 60:
                st.warning("🟡 Phục hồi TRUNG BÌNH")
            else:
                st.error("🔴 Phục hồi CHẬM - Cần can thiệp")
        
        with col2:
            st.subheader("💡 Khuyến Nghị Hành Động")
            
            # Sắp xếp theo hiệu quả
            scenarios.sort(key=lambda x: x[2], reverse=True)
            
            for i, (name, pred, improvement) in enumerate(scenarios, 1):
                if improvement > 0:
                    st.markdown(f"""
                    <div class="decision-card">
                    <b>Khuyến nghị #{i}: {name}</b><br>
                    📉 Giảm <b>{improvement:.1f} ngày</b> (còn {pred:.1f} ngày)<br>
                    ✅ Hiệu quả: <span style='color:green;'>Cao</span>
                    </div>
                    """, unsafe_allow_html=True)
        
        st.divider()
        
        # Thông tin chính sách
        policy_factor, policy_info = get_policy_factor(country)
        if policy_info:
            st.subheader(f"🏛️ Thông tin Chính sách {country}")
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Điểm Chính sách", f"{policy_info['policy_score']}/100")
            col2.metric("Điểm Hạ tầng", f"{policy_info['infrastructure_score']}/100")
            col3.metric("Cảnh báo Sớm", "✅ Có" if policy_info['early_warning'] else "❌ Không")
            col4.metric("Quỹ Khẩn cấp", f"${policy_info['emergency_fund_usd']/1e9:.1f}B")

# =========================================================
# TRANG: SO SÁNH MODELS
# =========================================================
def page_model_comparison(df, models):
    st.markdown('<div class="main-header">⚖️ So Sánh Mô Hình</div>', unsafe_allow_html=True)
    
    if not models:
        st.error("❌ Models chưa sẵn sàng!")
        return
    
    st.subheader("📊 Thông số Mô hình")
    
    model_info = []
    for name, m in models.items():
        metrics = m['config'].get('metrics', {})
        model_info.append({
            'Mô hình': name,
            'R² Score': f"{metrics.get('r2', 0)*100:.2f}%",
            'MAE': f"{metrics.get('mae', 0):.2f} ngày",
            'RMSE': f"{metrics.get('rmse', 0):.2f} ngày"
        })
    
    st.dataframe(pd.DataFrame(model_info), use_container_width=True, hide_index=True)
    
    st.divider()
    
    # So sánh trên sample data
    n = st.slider("Số mẫu so sánh:", 10, min(200, len(df)), 50)
    
    if st.button("⚖️ Chạy So Sánh", use_container_width=True, type="primary"):
        sample = df.sample(n=n, random_state=42).reset_index(drop=True)
        
        results = {'Thực tế': sample['recovery_days'].values}
        
        for name, m in models.items():
            preds = []
            for _, row in sample.iterrows():
                inp = {
                    'severity_index': row['severity_index'],
                    'casualties': row['casualties'],
                    'economic_loss_usd': row['economic_loss_usd'],
                    'response_time_hours': row['response_time_hours'],
                    'aid_amount_usd': row['aid_amount_usd'],
                    'response_efficiency_score': row['response_efficiency_score'],
                    'year': row['year'], 'month': row['month'],
                    'latitude': row['latitude'], 'longitude': row['longitude']
                }
                pred = predict_recovery(models, name, row['country'], row['disaster_type'], inp)
                preds.append(pred)
            results[name] = preds
        
        # Hiển thị metrics
        st.subheader("📈 Kết Quả So Sánh")
        
        for name in models.keys():
            mae = np.abs(np.array(results['Thực tế']) - np.array(results[name])).mean()
            r2 = r2_score(results['Thực tế'], results[name])
            st.metric(f"{name}", f"R²={r2:.4f}, MAE={mae:.2f}")
        
        # Scatter plot
        col1, col2 = st.columns(2)
        
        for i, name in enumerate(models.keys()):
            with col1 if i == 0 else col2:
                fig = px.scatter(x=results['Thực tế'], y=results[name],
                               labels={'x': 'Thực tế', 'y': f'Dự đoán {name}'},
                               title=f'{name}: Thực tế vs Dự đoán')
                fig.add_trace(go.Scatter(x=[min(results['Thực tế']), max(results['Thực tế'])],
                                        y=[min(results['Thực tế']), max(results['Thực tế'])],
                                        mode='lines', name='Perfect', line=dict(dash='dash', color='red')))
                st.plotly_chart(fig, use_container_width=True)

# =========================================================
# TRANG: ABOUT
# =========================================================
def page_about():
    st.markdown('<div class="main-header">ℹ️ Về Hệ Thống</div>', unsafe_allow_html=True)
    
    st.markdown("""
    ## 🌍 Hệ Thống Dự Đoán Phục Hồi Sau Thảm Họa
    
    ### 📋 Mô tả
    Ứng dụng Machine Learning dự đoán số ngày cần thiết để phục hồi sau các thảm họa 
    tự nhiên, hỗ trợ ra quyết định cho các cơ quan cứu trợ và chính phủ.
    
    ### 🎯 Tính năng chính
    - ✅ **Dự đoán đa mô hình**: XGBoost & LightGBM với R² > 93%
    - ✅ **Dữ liệu chính sách**: Tích hợp điểm số chính sách và hạ tầng 21 quốc gia
    - ✅ **Mô hình Spatio-Temporal**: Phân tích không gian-thời gian
    - ✅ **What-If Analysis**: Dự đoán theo kịch bản
    - ✅ **Decision Support**: Hệ thống hỗ trợ quyết định với khuyến nghị
    
    ### 📊 Dữ liệu
    - **Nguồn**: Global Disaster Response 2018-2024
    - **Quy mô**: ~50,000 bản ghi
    - **Features**: 12 biến đầu vào
    
    ### 🤖 Models
    | Model | R² | MAE | RMSE |
    |-------|-----|-----|------|
    | XGBoost | 93.64% | 4.05 ngày | 5.08 ngày |
    | LightGBM | 93.68% | 4.04 ngày | 5.07 ngày |
    
    ### ��‍💻 Tác giả
    **Trần Minh Hiếu** - Machine Learning Project 2024
    """)

# =========================================================
# MAIN
# =========================================================
def main():
    # Load data và models
    df = load_data()
    models = load_models()
    
    if df is None:
        st.error("❌ Không thể load dữ liệu!")
        return
    
    # Sidebar Navigation
    st.sidebar.markdown("## 🌍 Hệ Thống Dự Đoán")
    st.sidebar.markdown("### Phục Hồi Sau Thảm Họa")
    st.sidebar.divider()
    
    page = st.sidebar.radio("📍 Chọn trang:", [
        "📊 Tổng Quan",
        "📈 Phân Tích EDA",
        "🏛️ Chính Sách & Hạ Tầng",
        "🌐 Không Gian-Thời Gian",
        "🎯 Dự Đoán",
        "🔄 Kịch Bản What-If",
        "🎛️ Hỗ Trợ Quyết Định",
        "⚖️ So Sánh Models",
        "ℹ️ Về Hệ Thống"
    ])
    
    st.sidebar.divider()
    st.sidebar.markdown(f"📦 **Dữ liệu**: {len(df):,} bản ghi")
    st.sidebar.markdown(f"🤖 **Models**: {len(models)} mô hình")
    
    # Routing
    if page == "📊 Tổng Quan":
        page_overview(df)
    elif page == "📈 Phân Tích EDA":
        page_eda(df)
    elif page == "🏛️ Chính Sách & Hạ Tầng":
        page_policy()
    elif page == "🌐 Không Gian-Thời Gian":
        page_spatiotemporal(df)
    elif page == "🎯 Dự Đoán":
        page_prediction(df, models)
    elif page == "🔄 Kịch Bản What-If":
        page_whatif(df, models)
    elif page == "🎛️ Hỗ Trợ Quyết Định":
        page_decision_support(df, models)
    elif page == "⚖️ So Sánh Models":
        page_model_comparison(df, models)
    elif page == "ℹ️ Về Hệ Thống":
        page_about()
    
    # Footer
    st.sidebar.divider()
    st.sidebar.markdown("---")
    st.sidebar.markdown("© 2024 Trần Minh Hiếu")

if __name__ == "__main__":
    main()
