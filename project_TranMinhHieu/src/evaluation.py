import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    mean_squared_error, mean_absolute_error, r2_score,
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report
)


def evaluate_regression_model(y_true, y_pred, model_name='Model'):
    """
    Evaluate regression model performance.
    
    Parameters:
    -----------
    y_true : array-like
        True target values
    y_pred : array-like
        Predicted target values
    model_name : str
        Name of the model
        
    Returns:
    --------
    dict
        Dictionary of evaluation metrics
    """
    print("\n" + "="*50)
    print(f"REGRESSION EVALUATION - {model_name}")
    print("="*50)
    
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    
    # Mean Absolute Percentage Error (handle zero values)
    mask = y_true != 0
    if mask.sum() > 0:
        mape = np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100
    else:
        mape = 0.0
    
    metrics = {
        'MSE': mse,
        'RMSE': rmse,
        'MAE': mae,
        'R2': r2,
        'MAPE': mape
    }
    
    print(f"\nMean Squared Error (MSE): {mse:.4f}")
    print(f"Root Mean Squared Error (RMSE): {rmse:.4f}")
    print(f"Mean Absolute Error (MAE): {mae:.4f}")
    print(f"R² Score: {r2:.4f}")
    print(f"Mean Absolute Percentage Error (MAPE): {mape:.2f}%")
    
    return metrics


def evaluate_classification_model(y_true, y_pred, model_name='Model'):
    """
    Evaluate classification model performance.
    
    Parameters:
    -----------
    y_true : array-like
        True target values
    y_pred : array-like
        Predicted target values
    model_name : str
        Name of the model
        
    Returns:
    --------
    dict
        Dictionary of evaluation metrics
    """
    print("\n" + "="*50)
    print(f"CLASSIFICATION EVALUATION - {model_name}")
    print("="*50)
    
    accuracy = accuracy_score(y_true, y_pred)
    
    # Handle multiclass vs binary
    average_method = 'weighted'
    
    precision = precision_score(y_true, y_pred, average=average_method, zero_division=0)
    recall = recall_score(y_true, y_pred, average=average_method, zero_division=0)
    f1 = f1_score(y_true, y_pred, average=average_method, zero_division=0)
    
    metrics = {
        'Accuracy': accuracy,
        'Precision': precision,
        'Recall': recall,
        'F1-Score': f1
    }
    
    print(f"\nAccuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1-Score: {f1:.4f}")
    
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred, zero_division=0))
    
    return metrics


def plot_regression_results(y_true, y_pred, model_name='Model', save_fig=False):
    """
    Plot regression model results.
    
    Parameters:
    -----------
    y_true : array-like
        True target values
    y_pred : array-like
        Predicted target values
    model_name : str
        Name of the model
    save_fig : bool
        Whether to save the figure
    """
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    # Actual vs Predicted
    axes[0].scatter(y_true, y_pred, alpha=0.6, edgecolors='k', s=80)
    axes[0].plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 
                 'r--', lw=2, label='Perfect Prediction')
    axes[0].set_xlabel('Actual Values', fontsize=12)
    axes[0].set_ylabel('Predicted Values', fontsize=12)
    axes[0].set_title(f'Actual vs Predicted - {model_name}', fontsize=14, fontweight='bold')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Residuals plot
    residuals = y_true - y_pred
    axes[1].scatter(y_pred, residuals, alpha=0.6, edgecolors='k', s=80)
    axes[1].axhline(y=0, color='r', linestyle='--', lw=2)
    axes[1].set_xlabel('Predicted Values', fontsize=12)
    axes[1].set_ylabel('Residuals', fontsize=12)
    axes[1].set_title(f'Residual Plot - {model_name}', fontsize=14, fontweight='bold')
    axes[1].grid(True, alpha=0.3)
    
    # Residuals distribution
    axes[2].hist(residuals, bins=30, edgecolor='black', alpha=0.7)
    axes[2].axvline(x=0, color='r', linestyle='--', lw=2)
    axes[2].set_xlabel('Residuals', fontsize=12)
    axes[2].set_ylabel('Frequency', fontsize=12)
    axes[2].set_title(f'Residuals Distribution - {model_name}', fontsize=14, fontweight='bold')
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_fig:
        plt.savefig(f'outputs/{model_name}_regression_evaluation.png', dpi=300, bbox_inches='tight')
    
    plt.show()


def plot_confusion_matrix(y_true, y_pred, labels=None, model_name='Model', save_fig=False):
    """
    Plot confusion matrix for classification.
    
    Parameters:
    -----------
    y_true : array-like
        True target values
    y_pred : array-like
        Predicted target values
    labels : list, optional
        Class labels
    model_name : str
        Name of the model
    save_fig : bool
        Whether to save the figure
    """
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=labels, yticklabels=labels,
                cbar_kws={'label': 'Count'})
    plt.title(f'Confusion Matrix - {model_name}', fontsize=14, fontweight='bold')
    plt.ylabel('True Label', fontsize=12)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.tight_layout()
    
    if save_fig:
        plt.savefig(f'outputs/{model_name}_confusion_matrix.png', dpi=300, bbox_inches='tight')
    
    plt.show()


def plot_feature_importance(feature_importance_df, model_name='Model', top_n=15, save_fig=False):
    """
    Plot feature importance.
    
    Parameters:
    -----------
    feature_importance_df : pd.DataFrame
        Dataframe with feature names and importance scores
    model_name : str
        Name of the model
    top_n : int
        Number of top features to display
    save_fig : bool
        Whether to save the figure
    """
    if feature_importance_df is None or len(feature_importance_df) == 0:
        print("No feature importance data available")
        return
    
    top_features = feature_importance_df.head(top_n)
    
    plt.figure(figsize=(10, 8))
    plt.barh(range(len(top_features)), top_features['importance'].values, color='steelblue')
    plt.yticks(range(len(top_features)), top_features['feature'].values)
    plt.xlabel('Importance', fontsize=12)
    plt.ylabel('Feature', fontsize=12)
    plt.title(f'Top {top_n} Feature Importances - {model_name}', fontsize=14, fontweight='bold')
    plt.gca().invert_yaxis()
    plt.grid(axis='x', alpha=0.3)
    plt.tight_layout()
    
    if save_fig:
        plt.savefig(f'outputs/{model_name}_feature_importance.png', dpi=300, bbox_inches='tight')
    
    plt.show()


def compare_models(results_dict, metric='test_score'):
    """
    Compare multiple models.
    
    Parameters:
    -----------
    results_dict : dict
        Dictionary with model names as keys and results as values
    metric : str
        Metric to compare ('test_score', 'train_score', etc.)
    """
    print("\n" + "="*50)
    print("MODEL COMPARISON")
    print("="*50)
    
    comparison_data = {
        'Model': [],
        'Train Score': [],
        'Test Score': []
    }
    
    for model_name, results in results_dict.items():
        comparison_data['Model'].append(model_name)
        comparison_data['Train Score'].append(results.get('train_score', 0))
        comparison_data['Test Score'].append(results.get('test_score', 0))
    
    comparison_df = pd.DataFrame(comparison_data)
    print("\n", comparison_df.to_string(index=False))
    
    # Plot comparison
    fig, ax = plt.subplots(figsize=(12, 6))
    x = np.arange(len(comparison_df))
    width = 0.35
    
    ax.bar(x - width/2, comparison_df['Train Score'], width, label='Train Score', alpha=0.8)
    ax.bar(x + width/2, comparison_df['Test Score'], width, label='Test Score', alpha=0.8)
    
    ax.set_xlabel('Model', fontsize=12)
    ax.set_ylabel('Score', fontsize=12)
    ax.set_title('Model Performance Comparison', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(comparison_df['Model'], rotation=45, ha='right')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    return comparison_df


def generate_evaluation_report(model, X_test, y_test, model_name='Model', task='regression'):
    """
    Generate comprehensive evaluation report.
    
    Parameters:
    -----------
    model : trained model object
        The trained model with predict method
    X_test : array-like
        Test features
    y_test : array-like
        Test target
    model_name : str
        Name of the model
    task : str
        Type of task ('regression' or 'classification')
        
    Returns:
    --------
    dict
        Evaluation metrics
    """
    print("\n" + "="*50)
    print(f"GENERATING EVALUATION REPORT - {model_name}")
    print("="*50)
    
    # Make predictions
    y_pred = model.predict(X_test)
    
    # Evaluate based on task
    if task == 'regression':
        metrics = evaluate_regression_model(y_test, y_pred, model_name)
        plot_regression_results(y_test, y_pred, model_name, save_fig=True)
    else:
        metrics = evaluate_classification_model(y_test, y_pred, model_name)
        plot_confusion_matrix(y_test, y_pred, model_name=model_name, save_fig=True)
    
    # Feature importance if available
    if hasattr(model, 'get_feature_importance'):
        feature_imp = model.get_feature_importance()
        if feature_imp is not None:
            plot_feature_importance(feature_imp, model_name, save_fig=True)
    
    return metrics


if __name__ == "__main__":
    # Example usage
    from preprocessing import load_data, preprocess_data
    from feature_engineering import engineer_features
    from model_TranMinhHieu import DisasterResponseModel
    from pathlib import Path
    
    # Create outputs directory
    Path('outputs').mkdir(exist_ok=True)
    
    print("Loading and preparing data...")
    df = load_data('data/global_disaster_response_2018_2024.csv')
    
    if df is not None:
        df = preprocess_data('data/global_disaster_response_2018_2024.csv', save_output=False)
        df_engineered, encoders = engineer_features(df, encode_categoricals=True)
        
        # Train and evaluate model
        model = DisasterResponseModel(model_type='random_forest', task='regression')
        
        feature_cols = [
            'severity_index', 'casualties', 'economic_loss_usd',
            'response_time_hours', 'aid_amount_usd', 'response_efficiency_score',
            'casualty_per_hour', 'aid_coverage_ratio'
        ]
        
        X, y = model.prepare_features(df_engineered, 'recovery_days', feature_cols)
        results = model.train(X, y)
        
        # Generate evaluation report
        metrics = generate_evaluation_report(
            model, results['X_test'], results['y_test'],
            model_name='Random Forest', task='regression'
        )
