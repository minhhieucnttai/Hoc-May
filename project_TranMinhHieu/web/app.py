"""
Streamlit Web Application
Interactive dashboard for disaster response analysis and prediction.
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
import sys

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / 'src'))

# Set page configuration
st.set_page_config(
    page_title="Disaster Response Analysis",
    page_icon="🌍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #ff7f0e;
        margin-top: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        text-align: center;
    }
    </style>
    """, unsafe_allow_html=True)


@st.cache_data
def load_data():
    """Load the disaster response dataset."""
    data_path = Path(__file__).parent.parent / 'data' / 'global_disaster_response_2018_2024.csv'
    try:
        df = pd.read_csv(data_path)
        df['date'] = pd.to_datetime(df['date'])
        df['year'] = df['date'].dt.year
        df['month'] = df['date'].dt.month
        return df
    except FileNotFoundError:
        st.error(f"Data file not found at {data_path}")
        return None


@st.cache_data
def load_engineered_data():
    """Load engineered dataset if available."""
    data_path = Path(__file__).parent.parent / 'data' / 'global_disaster_response_2018_2024_engineered.csv'
    try:
        return pd.read_csv(data_path)
    except FileNotFoundError:
        return None


def main():
    """Main application function."""
    
    # Header
    st.markdown('<h1 class="main-header">🌍 Global Disaster Response Analysis Dashboard</h1>', 
                unsafe_allow_html=True)
    st.markdown("### Analyzing disaster response patterns from 2018-2024")
    
    # Load data
    df = load_data()
    
    if df is None:
        st.error("Failed to load data. Please ensure the data file exists.")
        return
    
    # Sidebar
    st.sidebar.title("📊 Navigation")
    page = st.sidebar.radio(
        "Select a page:",
        ["Overview", "Data Explorer", "Visualizations", "Statistics", "Predictions", "About"]
    )
    
    # Overview Page
    if page == "Overview":
        show_overview(df)
    
    # Data Explorer Page
    elif page == "Data Explorer":
        show_data_explorer(df)
    
    # Visualizations Page
    elif page == "Visualizations":
        show_visualizations(df)
    
    # Statistics Page
    elif page == "Statistics":
        show_statistics(df)
    
    # Predictions Page
    elif page == "Predictions":
        show_predictions(df)
    
    # About Page
    elif page == "About":
        show_about()


def show_overview(df):
    """Display overview page."""
    st.header("📈 Overview")
    
    # Key Metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Disasters", len(df))
    
    with col2:
        st.metric("Total Casualties", f"{df['casualties'].sum():,.0f}")
    
    with col3:
        st.metric("Affected Population", f"{df['affected_population'].sum():,.0f}")
    
    with col4:
        st.metric("Economic Impact", f"${df['economic_impact_usd'].sum()/1e9:.2f}B")
    
    # Quick Stats
    st.markdown("---")
    st.subheader("Quick Statistics")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**Date Range:**", f"{df['date'].min().date()} to {df['date'].max().date()}")
        st.write("**Number of Countries:**", df['country'].nunique())
        st.write("**Number of Regions:**", df['region'].nunique())
        st.write("**Disaster Types:**", df['disaster_type'].nunique())
    
    with col2:
        st.write("**Average Response Time:**", f"{df['response_time_hours'].mean():.1f} hours")
        st.write("**Average Response Effectiveness:**", f"{df['response_effectiveness'].mean():.2f}")
        st.write("**Average Casualties:**", f"{df['casualties'].mean():.0f}")
        st.write("**Average Affected Population:**", f"{df['affected_population'].mean():.0f}")
    
    # Recent Disasters
    st.markdown("---")
    st.subheader("📅 Recent Disasters")
    recent = df.nlargest(5, 'date')[['date', 'disaster_type', 'country', 'casualties', 
                                       'affected_population', 'response_effectiveness']]
    st.dataframe(recent, use_container_width=True)


def show_data_explorer(df):
    """Display data explorer page."""
    st.header("🔍 Data Explorer")
    
    # Filters
    st.sidebar.subheader("Filters")
    
    # Year filter
    years = sorted(df['year'].unique())
    selected_years = st.sidebar.multiselect("Select Years", years, default=years)
    
    # Disaster type filter
    disaster_types = sorted(df['disaster_type'].unique())
    selected_types = st.sidebar.multiselect("Select Disaster Types", disaster_types, 
                                            default=disaster_types)
    
    # Region filter
    regions = sorted(df['region'].unique())
    selected_regions = st.sidebar.multiselect("Select Regions", regions, default=regions)
    
    # Filter data
    filtered_df = df[
        (df['year'].isin(selected_years)) &
        (df['disaster_type'].isin(selected_types)) &
        (df['region'].isin(selected_regions))
    ]
    
    st.write(f"Showing {len(filtered_df)} out of {len(df)} records")
    
    # Display filtered data
    st.dataframe(filtered_df, use_container_width=True)
    
    # Download button
    csv = filtered_df.to_csv(index=False)
    st.download_button(
        label="📥 Download Filtered Data as CSV",
        data=csv,
        file_name="filtered_disaster_data.csv",
        mime="text/csv"
    )


def show_visualizations(df):
    """Display visualizations page."""
    st.header("📊 Visualizations")
    
    # Disaster Types Distribution
    st.subheader("Distribution of Disaster Types")
    fig = px.bar(df['disaster_type'].value_counts(), 
                 title="Number of Disasters by Type",
                 labels={'value': 'Count', 'index': 'Disaster Type'},
                 color_discrete_sequence=['#1f77b4'])
    st.plotly_chart(fig, use_container_width=True)
    
    # Temporal Trends
    st.subheader("Temporal Trends")
    yearly_data = df.groupby('year').agg({
        'disaster_type': 'count',
        'casualties': 'sum',
        'affected_population': 'sum',
        'economic_impact_usd': 'sum'
    }).reset_index()
    yearly_data.columns = ['Year', 'Number of Disasters', 'Total Casualties', 
                          'Total Affected Population', 'Total Economic Impact']
    
    fig = px.line(yearly_data, x='Year', y='Number of Disasters',
                  title="Number of Disasters Over Time",
                  markers=True)
    st.plotly_chart(fig, use_container_width=True)
    
    # Regional Distribution
    st.subheader("Regional Distribution")
    region_data = df.groupby('region').size().reset_index(name='count')
    fig = px.pie(region_data, values='count', names='region',
                 title="Disasters by Region")
    st.plotly_chart(fig, use_container_width=True)
    
    # Impact Analysis
    st.subheader("Impact Analysis")
    col1, col2 = st.columns(2)
    
    with col1:
        impact_by_type = df.groupby('disaster_type')['casualties'].sum().sort_values(ascending=True)
        fig = px.bar(impact_by_type, orientation='h',
                     title="Total Casualties by Disaster Type",
                     labels={'value': 'Total Casualties', 'disaster_type': 'Disaster Type'},
                     color_discrete_sequence=['#ff7f0e'])
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        economic_by_type = df.groupby('disaster_type')['economic_impact_usd'].sum().sort_values(ascending=True)
        fig = px.bar(economic_by_type / 1e9, orientation='h',
                     title="Total Economic Impact by Disaster Type (Billion USD)",
                     labels={'value': 'Economic Impact (B USD)', 'disaster_type': 'Disaster Type'},
                     color_discrete_sequence=['#2ca02c'])
        st.plotly_chart(fig, use_container_width=True)
    
    # Response Effectiveness
    st.subheader("Response Effectiveness Analysis")
    response_by_type = df.groupby('disaster_type')['response_effectiveness'].mean().sort_values(ascending=True)
    fig = px.bar(response_by_type, orientation='h',
                 title="Average Response Effectiveness by Disaster Type",
                 labels={'value': 'Avg Response Effectiveness', 'disaster_type': 'Disaster Type'},
                 color_discrete_sequence=['#9467bd'])
    st.plotly_chart(fig, use_container_width=True)


def show_statistics(df):
    """Display statistics page."""
    st.header("📈 Statistical Analysis")
    
    # Descriptive Statistics
    st.subheader("Descriptive Statistics")
    st.dataframe(df.describe(), use_container_width=True)
    
    # Correlation Matrix
    st.subheader("Correlation Matrix")
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    corr_matrix = df[numeric_cols].corr()
    
    fig = px.imshow(corr_matrix,
                    labels=dict(color="Correlation"),
                    x=corr_matrix.columns,
                    y=corr_matrix.columns,
                    color_continuous_scale='RdBu_r',
                    aspect='auto')
    fig.update_layout(title="Correlation Matrix of Numerical Features")
    st.plotly_chart(fig, use_container_width=True)
    
    # Distribution Analysis
    st.subheader("Distribution Analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        selected_col = st.selectbox("Select a numerical column:", numeric_cols)
        fig = px.histogram(df, x=selected_col, nbins=30,
                          title=f"Distribution of {selected_col}")
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        fig = px.box(df, y=selected_col,
                     title=f"Box Plot of {selected_col}")
        st.plotly_chart(fig, use_container_width=True)


def show_predictions(df):
    """Display predictions page."""
    st.header("🔮 Response Effectiveness Predictor")
    
    st.write("Use this tool to predict response effectiveness based on disaster parameters.")
    
    col1, col2 = st.columns(2)
    
    with col1:
        disaster_type = st.selectbox("Disaster Type", df['disaster_type'].unique())
        region = st.selectbox("Region", df['region'].unique())
        affected_population = st.number_input("Affected Population", 
                                              min_value=0, value=50000, step=1000)
        casualties = st.number_input("Casualties", min_value=0, value=100, step=10)
    
    with col2:
        economic_impact = st.number_input("Economic Impact (USD)", 
                                         min_value=0, value=1000000000, step=1000000)
        response_time = st.number_input("Response Time (hours)", 
                                       min_value=1, value=24, step=1)
    
    if st.button("Predict Response Effectiveness"):
        # Simple rule-based prediction for demo
        # In production, this would use the trained model
        
        base_effectiveness = 0.7
        
        # Adjust based on response time
        if response_time <= 12:
            time_factor = 0.2
        elif response_time <= 24:
            time_factor = 0.15
        elif response_time <= 48:
            time_factor = 0.1
        else:
            time_factor = 0
        
        # Adjust based on severity
        severity = casualties / (affected_population + 1)
        severity_factor = -0.1 if severity > 0.01 else 0
        
        predicted_effectiveness = max(0.5, min(1.0, base_effectiveness + time_factor + severity_factor))
        
        st.success(f"Predicted Response Effectiveness: {predicted_effectiveness:.2%}")
        
        # Show gauge chart
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=predicted_effectiveness * 100,
            domain={'x': [0, 1], 'y': [0, 1]},
            title={'text': "Response Effectiveness"},
            gauge={
                'axis': {'range': [None, 100]},
                'bar': {'color': "darkblue"},
                'steps': [
                    {'range': [0, 50], 'color': "lightgray"},
                    {'range': [50, 70], 'color': "gray"},
                    {'range': [70, 85], 'color': "lightblue"},
                    {'range': [85, 100], 'color': "blue"}
                ],
                'threshold': {
                    'line': {'color': "red", 'width': 4},
                    'thickness': 0.75,
                    'value': 90
                }
            }
        ))
        st.plotly_chart(fig, use_container_width=True)


def show_about():
    """Display about page."""
    st.header("ℹ️ About This Project")
    
    st.markdown("""
    ### Global Disaster Response Analysis (2018-2024)
    
    This interactive dashboard provides comprehensive analysis of global disaster response data 
    from 2018 to 2024. The project aims to:
    
    - **Analyze** patterns and trends in disaster occurrences
    - **Evaluate** response effectiveness across different disaster types and regions
    - **Predict** response effectiveness based on various parameters
    - **Visualize** key insights through interactive charts and graphs
    
    #### Project Structure
    
    ```
    project_TranMinhHieu/
    │
    ├── README.md
    ├── N10_report.pdf
    ├── requirements.txt
    │
    ├── data/
    │   └── global_disaster_response_2018_2024.csv
    │
    ├── src/
    │   ├── preprocessing.py          # Data cleaning and preparation
    │   ├── eda.py                    # Exploratory data analysis
    │   ├── feature_engineering.py    # Feature creation
    │   ├── model_TranMinhHieu.py     # Machine learning models
    │   ├── evaluation.py             # Model evaluation
    │   └── main.py                   # Main pipeline
    │
    └── web/
        └── app.py                    # This Streamlit application
    ```
    
    #### Technologies Used
    
    - **Python** - Core programming language
    - **Pandas & NumPy** - Data manipulation
    - **Scikit-learn** - Machine learning
    - **Matplotlib & Seaborn** - Visualization
    - **Streamlit** - Web application framework
    - **Plotly** - Interactive charts
    
    #### Author
    
    **Tran Minh Hieu**
    
    #### How to Use
    
    1. **Overview** - View key metrics and recent disasters
    2. **Data Explorer** - Filter and explore the dataset
    3. **Visualizations** - Interactive charts and graphs
    4. **Statistics** - Statistical analysis and correlations
    5. **Predictions** - Predict response effectiveness
    
    ---
    
    📧 For questions or feedback, please contact the project maintainer.
    """)


if __name__ == "__main__":
    main()
