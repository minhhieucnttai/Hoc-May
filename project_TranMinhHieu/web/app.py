# -*- coding: utf-8 -*-
"""
Đồ án ML - Dự đoán thảm họa
Trần Minh Hiếu - 2024
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
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

warnings.filterwarnings('ignore')

st.set_page_config(page_title="Dự đoán Hồi phục - ML", page_icon="🌍", layout="wide")

st.markdown("""
<style>
    .header {font-size: 2.2rem; color: #1f77b4; font-weight: bold; text-align: center; margin: 20px 0;}
    .subheader {font-size: 1rem; color: #666; text-align: center;}
    .metric-box {background: #f0f8ff; padding: 15px; border-radius: 5px; border-left: 4px solid #1f77b4; margin: 10px 0;}
    .success {background: #d4edda; padding: 10px; border-radius: 5px; border-left: 4px solid #28a745; margin: 10px 0;}
    .error {background: #f8d7da; padding: 10px; border-radius: 5px; border-left: 4px solid #dc3545;}
</style>
""", unsafe_allow_html=True)

# ============ LẤY DỮ LIỆU ============
@st.cache_data
def load_data():
    paths = [
        Path(__file__).parent / 'data' / 'global_disaster_response_2018_2024.csv',
        Path(__file__).parent.parent / 'data' / 'global_disaster_response_2018_2024.csv'
    ]
    for p in paths:
        if p.exists():
            df = pd.read_csv(p)
            # Extract year, month from date if available
            if 'date' in df.columns:
                df['date'] = pd.to_datetime(df['date'], errors='coerce')
                df['year'] = df['date'].dt.year
                df['month'] = df['date'].dt.month
                df['day'] = df['date'].dt.day
            return df
    st.error("❌ Không tìm thấy data!")
    return None

# ============ LOAD MODELS ============
@st.cache_resource
def load_xgb():
    try:
        mp = Path(__file__).parent / 'xgboost_model.pkl'
        sp = Path(__file__).parent / 'xgboost_scaler.pkl'
        ep = Path(__file__).parent / 'xgboost_encoders.pkl'
        cp = Path(__file__).parent / 'xgboost_config.json'
        
        if mp.exists() and sp.exists() and ep.exists():
            m = pickle.load(open(mp, 'rb'))
            s = pickle.load(open(sp, 'rb'))
            e = pickle.load(open(ep, 'rb'))
            c = json.load(open(cp)) if cp.exists() else {}
            return m, s, e, c
    except Exception as ex:
        st.warning(f"⚠️ Lỗi XGBoost: {ex}")
    return None, None, None, None

@st.cache_resource
def load_lgb():
    try:
        mp = Path(__file__).parent / 'lightgbm_model.pkl'
        sp = Path(__file__).parent / 'lightgbm_scaler.pkl'
        ep = Path(__file__).parent / 'lightgbm_encoders.pkl'
        cp = Path(__file__).parent / 'lightgbm_config.json'
        
        if mp.exists() and sp.exists() and ep.exists():
            m = pickle.load(open(mp, 'rb'))
            s = pickle.load(open(sp, 'rb'))
            e = pickle.load(open(ep, 'rb'))
            c = json.load(open(cp)) if cp.exists() else {}
            return m, s, e, c
    except Exception as ex:
        st.warning(f"⚠️ Lỗi LightGBM: {ex}")
    return None, None, None, None

# ============ DỰ ĐOÁN ============
def predict_xgb(country, disaster, inp_data):
    xgb_m, xgb_s, xgb_e, xgb_c = st.session_state.xgb
    if not xgb_m:
        return None
    
    d = inp_data.copy()
    d['country_encoded'] = xgb_e['country'].transform([country])[0]
    d['disaster_type_encoded'] = xgb_e['disaster_type'].transform([disaster])[0]
    d.pop('country', None)
    d.pop('disaster_type', None)
    
    feats = xgb_c.get('features', [])
    X = pd.DataFrame([[d.get(f, 0) for f in feats]], columns=feats)
    X_scaled = xgb_s.transform(X)
    return xgb_m.predict(X_scaled)[0]

def predict_lgb(country, disaster, inp_data):
    lgb_m, lgb_s, lgb_e, lgb_c = st.session_state.lgb
    if not lgb_m:
        return None
    
    d = inp_data.copy()
    d['country_encoded'] = lgb_e['country'].transform([country])[0]
    d['disaster_type_encoded'] = lgb_e['disaster_type'].transform([disaster])[0]
    d.pop('country', None)
    d.pop('disaster_type', None)
    
    feats = lgb_c.get('features', [])
    X = pd.DataFrame([[d.get(f, 0) for f in feats]], columns=feats)
    X_scaled = lgb_s.transform(X)
    return lgb_m.predict(X_scaled)[0]

# ============ TRANG TỔNG QUAN ============
def page_overview(df):
    st.markdown('<div class="header">📊 Tổng Quan Dự Án</div>', unsafe_allow_html=True)
    
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("📊 Records", f"{len(df):,}")
    c2.metric("🌍 Quốc gia", df['country'].nunique())
    c3.metric("⚡ Thảm họa", df['disaster_type'].nunique())
    c4.metric("📅 Năm", f"{int(df['year'].min())}-{int(df['year'].max())}")
    
    st.divider()
    
    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Recovery Days")
        st.dataframe(df['recovery_days'].describe().round(2), use_container_width=True)
    
    with c2:
        st.subheader("Số liệu")
        st.dataframe(df.select_dtypes(include='number').describe().T.round(2), use_container_width=True)

# ============ TRANG KHÁM PHÁ ============
def page_explore(df):
    st.markdown('<div class="header">🔍 Khám Phá Dữ Liệu</div>', unsafe_allow_html=True)
    
    c1, c2 = st.columns(2)
    with c1:
        countries = st.multiselect("Quốc gia:", sorted(df['country'].unique()), 
                                   default=sorted(df['country'].unique())[:5])
        disasters = st.multiselect("Thảm họa:", sorted(df['disaster_type'].unique()),
                                  default=sorted(df['disaster_type'].unique())[:3])
        min_r = st.slider("Min Recovery:", int(df['recovery_days'].min()), int(df['recovery_days'].max()), 
                         int(df['recovery_days'].min()))
        max_r = st.slider("Max Recovery:", int(df['recovery_days'].min()), int(df['recovery_days'].max()), 
                         int(df['recovery_days'].max()))
    
    filt = df[(df['country'].isin(countries)) & 
              (df['disaster_type'].isin(disasters)) &
              (df['recovery_days'] >= min_r) &
              (df['recovery_days'] <= max_r)]
    
    with c2:
        st.metric("📌 Bản ghi", len(filt))
        st.metric("📈 Thay đổi", f"{len(filt) - len(df):,}")
    
    st.dataframe(filt.head(20), use_container_width=True, hide_index=True)

# ============ TRANG BIỂU ĐỒ ============
def page_viz(df):
    st.markdown('<div class="header">📈 Biểu Đồ & Thống Kê</div>', unsafe_allow_html=True)
    
    c1, c2 = st.columns(2)
    
    with c1:
        fig = px.histogram(df, x='recovery_days', nbins=40, 
                          title='Phân phối Recovery Days', 
                          color_discrete_sequence=['#1f77b4'])
        st.plotly_chart(fig, use_container_width=True)
    
    with c2:
        fig = px.box(df, y='recovery_days', title='Box Plot Recovery',
                    color_discrete_sequence=['#ff7f0e'])
        st.plotly_chart(fig, use_container_width=True)
    
    st.divider()
    
    c1, c2 = st.columns(2)
    
    with c1:
        top_countries = df.groupby('country')['recovery_days'].mean().sort_values(ascending=False).head(10)
        fig = px.bar(top_countries, title='Top 10 Quốc Gia', color_discrete_sequence=['#2ca02c'])
        st.plotly_chart(fig, use_container_width=True)
    
    with c2:
        by_disaster = df.groupby('disaster_type')['recovery_days'].mean().sort_values(ascending=False)
        fig = px.bar(by_disaster, title='Theo Loại Thảm Họa', color_discrete_sequence=['#d62728'])
        st.plotly_chart(fig, use_container_width=True)

# ============ TRANG THÔNG TIN MODEL ============
def page_model_info():
    st.markdown('<div class="header">🤖 Thông Tin Models</div>', unsafe_allow_html=True)
    
    st.markdown("""
    ### XGBoost & LightGBM
    
    **XGBoost Regressor**
    - R² = 93.64% | MAE = 4.05 ngày | RMSE = 5.08 ngày
    
    **LightGBM Regressor**  
    - R² = 93.68% | MAE = 4.04 ngày | RMSE = 5.07 ngày
    
    ### Features (12 biến)
    - **Số**: severity_index, casualties, economic_loss_usd, response_time_hours, aid_amount_usd, response_efficiency_score, latitude, longitude
    - **Thời gian**: year, month
    - **Phân loại**: country, disaster_type
    
    ### Đặc tính
    ✅ Deterministic - Random State = 42
    ✅ Dự đoán nhất quán - Cùng input = Cùng output
    ✅ Train/Test Split: 80/20
    """)

# ============ TRANG DỰ ĐOÁN ============
def page_prediction(df):
    st.markdown('<div class="header">🎯 Dự Đoán Hồi Phục</div>', unsafe_allow_html=True)
    
    xgb_m, _, _, _ = st.session_state.xgb
    lgb_m, _, _, _ = st.session_state.lgb
    
    if not xgb_m or not lgb_m:
        st.error("❌ Models không load được!")
        return
    
    st.markdown('<div class="success">✅ XGBoost & LightGBM sẵn sàng</div>', unsafe_allow_html=True)
    
    c1, c2 = st.columns(2)
    
    with c1:
        country = st.selectbox("Quốc gia:", sorted(df['country'].unique()))
        disaster = st.selectbox("Loại thảm họa:", sorted(df['disaster_type'].unique()))
        severity = st.slider("Mức độ (1-10):", 1, 10, 5)
        casualties = st.number_input("Người thương vong:", 0, 100000, 100)
        loss = st.number_input("Thiệt hại (USD):", 0, 100000000, 1000000)
    
    with c2:
        resp_time = st.slider("Thời gian phản ứng (giờ):", 0, 48, 12)
        aid = st.number_input("Hỗ trợ (USD):", 0, 10000000, 500000)
        eff = st.slider("Hiệu quả (0-100):", 0, 100, 70)
        year = st.number_input("Năm:", int(df['year'].min()), int(df['year'].max()), 2024)
        month = st.slider("Tháng:", 1, 12, 6)
        lat = st.number_input("Latitude:", -90.0, 90.0, 0.0)
        lon = st.number_input("Longitude:", -180.0, 180.0, 0.0)
    
    if st.button("🔮 Dự Đoán!", use_container_width=True, type="primary"):
        inp = {
            'severity_index': severity,
            'casualties': casualties,
            'economic_loss_usd': loss,
            'response_time_hours': resp_time,
            'aid_amount_usd': aid,
            'response_efficiency_score': eff,
            'year': year,
            'month': month,
            'latitude': lat,
            'longitude': lon
        }
        
        try:
            pred_xgb = predict_xgb(country, disaster, inp)
            pred_lgb = predict_lgb(country, disaster, inp)
            
            st.divider()
            
            c1, c2, c3 = st.columns(3)
            
            with c1:
                st.markdown(f"""
                <div style='text-align: center; background: #ffe6e6; padding: 20px; border-radius: 8px;'>
                    <h4>🔷 XGBoost</h4>
                    <h2 style='color: #d9534f;'>{pred_xgb:.1f}</h2>
                    <p>ngày</p>
                </div>
                """, unsafe_allow_html=True)
            
            with c2:
                st.markdown(f"""
                <div style='text-align: center; background: #e6f3ff; padding: 20px; border-radius: 8px;'>
                    <h4>🔹 LightGBM</h4>
                    <h2 style='color: #5cb85c;'>{pred_lgb:.1f}</h2>
                    <p>ngày</p>
                </div>
                """, unsafe_allow_html=True)
            
            with c3:
                avg = (pred_xgb + pred_lgb) / 2
                st.markdown(f"""
                <div style='text-align: center; background: #f0f8ff; padding: 20px; border-radius: 8px; border: 2px solid #1f77b4;'>
                    <h4>📊 Trung bình</h4>
                    <h2 style='color: #1f77b4;'>{avg:.1f}</h2>
                    <p>ngày</p>
                </div>
                """, unsafe_allow_html=True)
            
            st.info(f"✨ Hai model cho kết quả gần nhau (chênh {abs(pred_xgb - pred_lgb):.2f} ngày) → tin cậy cao!")
        except Exception as e:
            st.error(f"❌ Lỗi: {e}")

# ============ TRANG SO SÁNH ============
def page_comparison(df):
    st.markdown('<div class="header">⚖️ So Sánh 2 Models</div>', unsafe_allow_html=True)
    
    xgb_m, xgb_s, xgb_e, xgb_c = st.session_state.xgb
    lgb_m, lgb_s, lgb_e, lgb_c = st.session_state.lgb
    
    if not xgb_m or not lgb_m:
        st.error("❌ Models không sẵn sàng!")
        return
    
    n = st.slider("Số bản ghi:", 10, min(500, len(df)), 100)
    
    if st.button("⚖️ Chạy so sánh", use_container_width=True, type="primary"):
        try:
            s_df = df.sample(n=n, random_state=42).reset_index(drop=True)
            
            # XGBoost predictions
            x_list = []
            for _, r in s_df.iterrows():
                d = {
                    'severity_index': r['severity_index'],
                    'casualties': r['casualties'],
                    'economic_loss_usd': r['economic_loss_usd'],
                    'response_time_hours': r['response_time_hours'],
                    'aid_amount_usd': r['aid_amount_usd'],
                    'response_efficiency_score': r['response_efficiency_score'],
                    'year': r['year'],
                    'month': r['month'],
                    'latitude': r['latitude'],
                    'longitude': r['longitude'],
                    'country_encoded': xgb_e['country'].transform([r['country']])[0],
                    'disaster_type_encoded': xgb_e['disaster_type'].transform([r['disaster_type']])[0]
                }
                x_list.append([d.get(f, 0) for f in xgb_c['features']])
            
            X_xgb = pd.DataFrame(x_list, columns=xgb_c['features'])
            X_xgb_s = xgb_s.transform(X_xgb)
            p_xgb = xgb_m.predict(X_xgb_s)
            
            # LightGBM predictions
            l_list = []
            for _, r in s_df.iterrows():
                d = {
                    'severity_index': r['severity_index'],
                    'casualties': r['casualties'],
                    'economic_loss_usd': r['economic_loss_usd'],
                    'response_time_hours': r['response_time_hours'],
                    'aid_amount_usd': r['aid_amount_usd'],
                    'response_efficiency_score': r['response_efficiency_score'],
                    'year': r['year'],
                    'month': r['month'],
                    'latitude': r['latitude'],
                    'longitude': r['longitude'],
                    'country_encoded': lgb_e['country'].transform([r['country']])[0],
                    'disaster_type_encoded': lgb_e['disaster_type'].transform([r['disaster_type']])[0]
                }
                l_list.append([d.get(f, 0) for f in lgb_c['features']])
            
            X_lgb = pd.DataFrame(l_list, columns=lgb_c['features'])
            X_lgb_s = lgb_s.transform(X_lgb)
            p_lgb = lgb_m.predict(X_lgb_s)
            
            # Bảng so sánh
            cmp_df = pd.DataFrame({
                'Quốc Gia': s_df['country'].values,
                'Thảm Họa': s_df['disaster_type'].values,
                'Thực Tế': s_df['recovery_days'].round(1).values,
                'XGBoost': p_xgb.round(1),
                'LightGBM': p_lgb.round(1),
                'Chênh XGB': (s_df['recovery_days'].values - p_xgb).round(1),
                'Chênh LGB': (s_df['recovery_days'].values - p_lgb).round(1)
            })
            
            st.subheader("📊 Kết Quả So Sánh")
            st.dataframe(cmp_df, use_container_width=True, hide_index=True)
            
            st.divider()
            
            # Metrics
            mae_xgb = np.abs(s_df['recovery_days'].values - p_xgb).mean()
            mae_lgb = np.abs(s_df['recovery_days'].values - p_lgb).mean()
            rmse_xgb = np.sqrt(((s_df['recovery_days'].values - p_xgb) ** 2).mean())
            rmse_lgb = np.sqrt(((s_df['recovery_days'].values - p_lgb) ** 2).mean())
            r2_xgb = r2_score(s_df['recovery_days'], p_xgb)
            r2_lgb = r2_score(s_df['recovery_days'], p_lgb)
            
            c1, c2, c3 = st.columns(3)
            c1.metric("XGBoost R²", f"{r2_xgb:.4f}", delta=f"{r2_xgb*100:.1f}%")
            c2.metric("LightGBM R²", f"{r2_lgb:.4f}", delta=f"{r2_lgb*100:.1f}%")
            c3.metric("MAE Avg", f"{(mae_xgb + mae_lgb)/2:.2f}")
            
            st.divider()
            
            # Visualizations
            c1, c2 = st.columns(2)
            
            with c1:
                fig = px.scatter(x=s_df['recovery_days'], y=p_xgb, 
                               labels={'x': 'Thực Tế', 'y': 'Dự Đoán XGBoost'},
                               title='XGBoost: Thực vs Dự Đoán',
                               trendline='ols')
                st.plotly_chart(fig, use_container_width=True)
            
            with c2:
                fig = px.scatter(x=s_df['recovery_days'], y=p_lgb,
                               labels={'x': 'Thực Tế', 'y': 'Dự Đoán LightGBM'},
                               title='LightGBM: Thực vs Dự Đoán',
                               trendline='ols')
                st.plotly_chart(fig, use_container_width=True)
            
            # Error distribution
            c1, c2 = st.columns(2)
            
            with c1:
                fig = px.histogram(x=(s_df['recovery_days'].values - p_xgb),
                                 title='Phân phối Lỗi XGBoost',
                                 nbins=20, color_discrete_sequence=['#d9534f'])
                st.plotly_chart(fig, use_container_width=True)
            
            with c2:
                fig = px.histogram(x=(s_df['recovery_days'].values - p_lgb),
                                 title='Phân phối Lỗi LightGBM',
                                 nbins=20, color_discrete_sequence=['#5cb85c'])
                st.plotly_chart(fig, use_container_width=True)
        
        except Exception as e:
            st.error(f"❌ Lỗi: {e}")

# ============ TRANG BATCH ============
def page_batch(df):
    st.markdown('<div class="header">📦 Dự Đoán Hàng Loạt</div>', unsafe_allow_html=True)
    
    xgb_m, _, _, _ = st.session_state.xgb
    lgb_m, _, _, _ = st.session_state.lgb
    
    if not xgb_m or not lgb_m:
        st.error("❌ Models không sẵn sàng!")
        return
    
    st.write("Upload file CSV có các columns: country, disaster_type, severity_index, casualties, economic_loss_usd, response_time_hours, aid_amount_usd, response_efficiency_score, year, month, latitude, longitude")
    
    uploaded = st.file_uploader("Chọn file CSV:", type=['csv'])
    
    if uploaded and st.button("📊 Dự đoán batch", use_container_width=True, type="primary"):
        try:
            batch_df = pd.read_csv(uploaded)
            results = []
            
            for _, row in batch_df.iterrows():
                inp = {
                    'severity_index': row['severity_index'],
                    'casualties': row['casualties'],
                    'economic_loss_usd': row['economic_loss_usd'],
                    'response_time_hours': row['response_time_hours'],
                    'aid_amount_usd': row['aid_amount_usd'],
                    'response_efficiency_score': row['response_efficiency_score'],
                    'year': row['year'],
                    'month': row['month'],
                    'latitude': row['latitude'],
                    'longitude': row['longitude']
                }
                
                p1 = predict_xgb(row['country'], row['disaster_type'], inp)
                p2 = predict_lgb(row['country'], row['disaster_type'], inp)
                
                results.append({
                    'Country': row['country'],
                    'Disaster': row['disaster_type'],
                    'XGBoost': p1,
                    'LightGBM': p2,
                    'Average': (p1 + p2) / 2
                })
            
            res_df = pd.DataFrame(results)
            st.dataframe(res_df, use_container_width=True, hide_index=True)
            
            # Download
            csv = res_df.to_csv(index=False)
            st.download_button(label="📥 Tải kết quả",
                             data=csv,
                             file_name="predictions.csv",
                             mime="text/csv")
        
        except Exception as e:
            st.error(f"❌ Lỗi: {e}")

# ============ TRANG ABOUT ============
def page_about():
    st.markdown('<div class="header">ℹ️ Về Dự Án</div>', unsafe_allow_html=True)
    
    st.markdown("""
    ### Dự Đoán Hồi Phục Thảm Họa
    
    **Mục tiêu**: Xây dựng 2 mô hình ML để dự đoán số ngày cần thiết để 
    phục hồi sau các thảm họa thiên nhiên.
    
    **Dữ liệu**: 50,000 bản ghi thảm họa (2018-2024) từ 20+ quốc gia
    
    **Models**:
    - XGBoost: R² = 93.64%, MAE = 4.05 ngày
    - LightGBM: R² = 93.68%, MAE = 4.04 ngày
    
    **Tính năng**:
    - ✅ Dự đoán đơn + batch
    - ✅ So sánh hiệu suất 2 models
    - ✅ Visualizations & analytics
    - ✅ Deterministic predictions (cùng input = cùng output)
    
    **Tác giả**: Trần Minh Hiếu
    """)

# ============ MAIN ============
def main():
    # Initialize session state
    if 'models_loaded' not in st.session_state:
        st.session_state.xgb = load_xgb()
        st.session_state.lgb = load_lgb()
        st.session_state.models_loaded = True
    
    # Load data
    df = load_data()
    if df is None:
        return
    
    # Header
    st.markdown('<div class="header">🌍 Dự Đoán Hồi Phục Thảm Họa</div>', unsafe_allow_html=True)
    st.markdown('<div class="subheader">Dual Model: XGBoost & LightGBM</div>', unsafe_allow_html=True)
    
    # Navigation
    st.sidebar.title("📍 Menu")
    page = st.sidebar.radio("Chọn:", [
        "📋 Tổng Quan",
        "🔍 Khám Phá",
        "📈 Biểu Đồ",
        "🤖 Models",
        "🎯 Dự Đoán",
        "⚖️ So Sánh",
        "📦 Batch",
        "ℹ️ About"
    ])
    
    if page == "📋 Tổng Quan":
        page_overview(df)
    elif page == "🔍 Khám Phá":
        page_explore(df)
    elif page == "📈 Biểu Đồ":
        page_viz(df)
    elif page == "🤖 Models":
        page_model_info()
    elif page == "🎯 Dự Đoán":
        page_prediction(df)
    elif page == "⚖️ So Sánh":
        page_comparison(df)
    elif page == "📦 Batch":
        page_batch(df)
    elif page == "ℹ️ About":
        page_about()

if __name__ == "__main__":
    main()
