# Setup Instructions

## Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Run the Complete Analysis Pipeline
```bash
python src/main.py
```

This will:
- Load and preprocess the disaster data
- Perform exploratory data analysis
- Engineer features
- Train multiple ML models
- Generate evaluation reports and visualizations
- Save trained models

### 3. Launch the Web Application
```bash
streamlit run web/app.py
```

The app will open in your browser at `http://localhost:8501`

## Project Components

### Data Files
- `data/global_disaster_response_2018_2024.csv` - Main dataset with 35 disaster records

### Source Modules
- `src/preprocessing.py` - Data cleaning and validation
- `src/eda.py` - Exploratory data analysis with visualizations
- `src/feature_engineering.py` - Feature creation (30+ features)
- `src/model_TranMinhHieu.py` - ML models (RF, GB, Linear, DT)
- `src/evaluation.py` - Model evaluation and metrics
- `src/main.py` - Complete pipeline orchestration

### Web Application
- `web/app.py` - Interactive Streamlit dashboard with 6 pages:
  - Overview - Key metrics and statistics
  - Data Explorer - Interactive filtering
  - Visualizations - Charts and graphs
  - Statistics - Correlation and distributions
  - Predictions - Response effectiveness predictor
  - About - Project information

## Output Files

After running `src/main.py`, the following will be generated:

- `data/global_disaster_response_2018_2024_preprocessed.csv` - Cleaned data
- `data/global_disaster_response_2018_2024_engineered.csv` - Feature-engineered data
- `models/*.pkl` - Trained machine learning models
- `outputs/*.png` - Visualization images
- `outputs/model_comparison.csv` - Model performance comparison

## System Requirements

- Python 3.7+
- pandas >= 1.5.0
- numpy >= 1.23.0
- scikit-learn >= 1.2.0
- matplotlib >= 3.6.0
- seaborn >= 0.12.0
- streamlit >= 1.25.0
- plotly >= 5.14.0

## Troubleshooting

### Missing Dependencies
```bash
pip install -r requirements.txt --upgrade
```

### Data Not Found
Ensure you're running commands from the `project_TranMinhHieu` directory.

### Port Already in Use (Streamlit)
```bash
streamlit run web/app.py --server.port 8502
```

## Next Steps

1. Review generated visualizations in `outputs/` folder
2. Check model performance in `outputs/model_comparison.csv`
3. Experiment with the web application
4. Modify parameters in `src/main.py` for custom analysis

## Contact

For questions or issues, please refer to the main README.md file.
