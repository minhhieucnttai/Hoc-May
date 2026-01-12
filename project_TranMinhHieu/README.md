# Global Disaster Response Analysis (2018-2024)

## Project Overview
This project analyzes global disaster response data from 2018 to 2024, implementing machine learning models to predict and analyze disaster response patterns.

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

## Components

- **preprocessing.py**: Data cleaning and preprocessing
- **eda.py**: Exploratory Data Analysis
- **feature_engineering.py**: Feature extraction and engineering
- **model_TranMinhHieu.py**: Machine learning model implementation
- **evaluation.py**: Model evaluation and metrics
- **main.py**: Main entry point for the analysis pipeline
- **app.py**: Streamlit web application for interactive visualization
