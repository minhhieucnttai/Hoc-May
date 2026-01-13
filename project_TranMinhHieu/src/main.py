"""
Main Module
Orchestrates the entire disaster response analysis pipeline.
"""

import sys
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Add src to path for imports
sys.path.append(str(Path(__file__).parent))

from preprocessing import load_data, preprocess_data
from eda import perform_eda
from feature_engineering import engineer_features
from model_TranMinhHieu import DisasterResponseModel, train_multiple_models
from evaluation import generate_evaluation_report, compare_models


def main():
    """
    Main execution function for the disaster response analysis pipeline.
    """
    print("\n" + "="*70)
    print(" "*15 + "DISASTER RESPONSE ANALYSIS PIPELINE")
    print("="*70 + "\n")
    
    # Change to project root directory
    project_root = Path(__file__).parent.parent
    import os
    os.chdir(project_root)
    
    # Create necessary directories
    Path('outputs').mkdir(exist_ok=True)
    Path('models').mkdir(exist_ok=True)
    
    # Step 1: Load and Preprocess Data
    print("\n" + "="*70)
    print("STEP 1: DATA LOADING AND PREPROCESSING")
    print("="*70)
    
    data_path = 'data/global_disaster_response_2018_2024.csv'
    df = load_data(data_path)
    
    if df is None:
        print("Error: Could not load data. Exiting...")
        return
    
    df_clean = preprocess_data(data_path, save_output=True)
    
    # Step 2: Exploratory Data Analysis
    print("\n" + "="*70)
    print("STEP 2: EXPLORATORY DATA ANALYSIS")
    print("="*70)
    
    perform_eda(df_clean, save_figs=True)
    
    # Step 3: Feature Engineering
    print("\n" + "="*70)
    print("STEP 3: FEATURE ENGINEERING")
    print("="*70)
    
    df_engineered, encoders = engineer_features(df_clean, encode_categoricals=True)
    
    # Save engineered data
    df_engineered.to_csv('data/global_disaster_response_2018_2024_engineered.csv', index=False)
    print("\nEngineered data saved to: data/global_disaster_response_2018_2024_engineered.csv")
    
    # Step 4: Model Training - Recovery Days Prediction
    print("\n" + "="*70)
    print("STEP 4: MODEL TRAINING - RECOVERY DAYS PREDICTION")
    print("="*70)
    
    # Define features for modeling
    feature_cols = [
        'severity_index', 'casualties', 'economic_loss_usd',
        'response_time_hours', 'aid_amount_usd', 'response_efficiency_score',
        'casualty_per_hour', 'aid_coverage_ratio'
    ]
    
    # Filter to only available features
    available_features = [col for col in feature_cols if col in df_engineered.columns]
    
    print(f"\nUsing {len(available_features)} features for modeling:")
    for feat in available_features:
        print(f"  - {feat}")
    
    # Train primary model
    print("\n" + "-"*70)
    print("Training Primary Model: Random Forest Regressor")
    print("-"*70)
    
    model = DisasterResponseModel(model_type='random_forest', task='regression')
    X, y = model.prepare_features(df_engineered, 'recovery_days', available_features)
    results = model.train(X, y, test_size=0.2)
    
    # Cross-validation
    cv_results = model.cross_validate(X, y, cv=5)
    
    # Feature importance
    feature_importance = model.get_feature_importance(top_n=10)
    
    # Save the model
    model.save_model('models/recovery_days_rf_model.pkl')
    
    # Step 5: Model Evaluation
    print("\n" + "="*70)
    print("STEP 5: MODEL EVALUATION")
    print("="*70)
    
    metrics = generate_evaluation_report(
        model, results['X_test'], results['y_test'],
        model_name='Random_Forest_Recovery_Days',
        task='regression'
    )
    
    # Step 6: Compare Multiple Models
    print("\n" + "="*70)
    print("STEP 6: COMPARING MULTIPLE MODELS")
    print("="*70)
    
    models, model_results = train_multiple_models(X, y, task='regression')
    comparison_df = compare_models(model_results)
    
    # Save comparison results
    comparison_df.to_csv('outputs/model_comparison.csv', index=False)
    print("\nModel comparison saved to: outputs/model_comparison.csv")
    
    # Step 7: Additional Prediction - Casualty Prediction
    print("\n" + "="*70)
    print("STEP 7: ADDITIONAL MODEL - CASUALTY PREDICTION")
    print("="*70)
    
    casualty_features = [
        'severity_index', 'economic_loss_usd', 'response_time_hours',
        'aid_amount_usd', 'response_efficiency_score', 'casualty_per_hour'
    ]
    
    available_casualty_features = [col for col in casualty_features if col in df_engineered.columns]
    
    casualty_model = DisasterResponseModel(model_type='random_forest', task='regression')
    X_casualty, y_casualty = casualty_model.prepare_features(
        df_engineered, 'casualties', available_casualty_features
    )
    casualty_results = casualty_model.train(X_casualty, y_casualty, test_size=0.2)
    
    casualty_model.save_model('models/casualties_prediction_rf_model.pkl')
    
    casualty_metrics = generate_evaluation_report(
        casualty_model, casualty_results['X_test'], casualty_results['y_test'],
        model_name='Random_Forest_Casualties_Prediction',
        task='regression'
    )
    
    # Final Summary
    print("\n" + "="*70)
    print("PIPELINE EXECUTION SUMMARY")
    print("="*70)
    
    print("\n[OK] Data loaded and preprocessed successfully")
    print("[OK] Exploratory data analysis completed")
    print("[OK] Feature engineering completed")
    print("[OK] Models trained and evaluated")
    print("\nOutput Files:")
    print("  - Preprocessed data: data/global_disaster_response_2018_2024_preprocessed.csv")
    print("  - Engineered data: data/global_disaster_response_2018_2024_engineered.csv")
    print("  - Model 1: models/recovery_days_rf_model.pkl")
    print("  - Model 2: models/casualties_prediction_rf_model.pkl")
    print("  - Visualizations: outputs/*.png")
    print("  - Model comparison: outputs/model_comparison.csv")
    
    print("\n" + "="*70)
    print("PIPELINE COMPLETED SUCCESSFULLY!")
    print("="*70 + "\n")
    
    print("Next steps:")
    print("  1. Review the generated visualizations in the 'outputs' folder")
    print("  2. Analyze the model comparison results")
    print("  3. Launch the web app: streamlit run web/app.py")
    print()


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\nError occurred during pipeline execution: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
