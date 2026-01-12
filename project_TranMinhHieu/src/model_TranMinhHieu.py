"""
Machine Learning Model Module
Implements and trains models for disaster response prediction.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.preprocessing import StandardScaler
import pickle
from pathlib import Path


class DisasterResponseModel:
    """
    Main model class for disaster response prediction.
    """
    
    def __init__(self, model_type='random_forest', task='regression'):
        """
        Initialize the model.
        
        Parameters:
        -----------
        model_type : str
            Type of model ('random_forest', 'gradient_boosting', 'linear', 'decision_tree')
        task : str
            Type of task ('regression' or 'classification')
        """
        self.model_type = model_type
        self.task = task
        self.model = None
        self.scaler = StandardScaler()
        self.feature_names = None
        self.is_fitted = False
        
        self._initialize_model()
    
    def _initialize_model(self):
        """Initialize the appropriate model based on type and task."""
        if self.task == 'regression':
            if self.model_type == 'random_forest':
                self.model = RandomForestRegressor(
                    n_estimators=100,
                    max_depth=10,
                    min_samples_split=5,
                    min_samples_leaf=2,
                    random_state=42,
                    n_jobs=-1
                )
            elif self.model_type == 'gradient_boosting':
                self.model = GradientBoostingRegressor(
                    n_estimators=100,
                    max_depth=5,
                    learning_rate=0.1,
                    random_state=42
                )
            elif self.model_type == 'linear':
                self.model = LinearRegression()
            elif self.model_type == 'decision_tree':
                self.model = DecisionTreeRegressor(
                    max_depth=10,
                    min_samples_split=5,
                    random_state=42
                )
        else:  # classification
            if self.model_type == 'random_forest':
                self.model = RandomForestClassifier(
                    n_estimators=100,
                    max_depth=10,
                    min_samples_split=5,
                    random_state=42,
                    n_jobs=-1
                )
            elif self.model_type == 'logistic':
                self.model = LogisticRegression(max_iter=1000, random_state=42)
    
    def prepare_features(self, df, target_col, feature_cols=None):
        """
        Prepare features and target for modeling.
        
        Parameters:
        -----------
        df : pd.DataFrame
            Input dataframe
        target_col : str
            Name of target column
        feature_cols : list, optional
            List of feature column names
            
        Returns:
        --------
        tuple
            (X, y) features and target
        """
        if feature_cols is None:
            # Select numeric features automatically
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            feature_cols = [col for col in numeric_cols if col != target_col]
        
        # Handle missing values in features
        X = df[feature_cols].fillna(df[feature_cols].mean())
        y = df[target_col]
        
        self.feature_names = feature_cols
        
        return X, y
    
    def train(self, X, y, test_size=0.2, scale_features=True):
        """
        Train the model.
        
        Parameters:
        -----------
        X : pd.DataFrame or np.array
            Features
        y : pd.Series or np.array
            Target
        test_size : float
            Proportion of data for testing
        scale_features : bool
            Whether to scale features
            
        Returns:
        --------
        dict
            Training results with train/test splits
        """
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42
        )
        
        # Scale features
        if scale_features:
            X_train_scaled = self.scaler.fit_transform(X_train)
            X_test_scaled = self.scaler.transform(X_test)
        else:
            X_train_scaled = X_train
            X_test_scaled = X_test
        
        # Train model
        print(f"\nTraining {self.model_type} model for {self.task}...")
        self.model.fit(X_train_scaled, y_train)
        self.is_fitted = True
        
        # Calculate scores
        train_score = self.model.score(X_train_scaled, y_train)
        test_score = self.model.score(X_test_scaled, y_test)
        
        print(f"Training Score: {train_score:.4f}")
        print(f"Testing Score: {test_score:.4f}")
        
        results = {
            'X_train': X_train_scaled,
            'X_test': X_test_scaled,
            'y_train': y_train,
            'y_test': y_test,
            'train_score': train_score,
            'test_score': test_score
        }
        
        return results
    
    def cross_validate(self, X, y, cv=5):
        """
        Perform cross-validation.
        
        Parameters:
        -----------
        X : pd.DataFrame or np.array
            Features
        y : pd.Series or np.array
            Target
        cv : int
            Number of cross-validation folds
            
        Returns:
        --------
        dict
            Cross-validation scores
        """
        X_scaled = self.scaler.fit_transform(X)
        
        scores = cross_val_score(self.model, X_scaled, y, cv=cv, 
                                scoring='r2' if self.task == 'regression' else 'accuracy')
        
        print(f"\nCross-Validation Scores ({cv}-fold):")
        print(f"Scores: {scores}")
        print(f"Mean: {scores.mean():.4f} (+/- {scores.std() * 2:.4f})")
        
        return {
            'scores': scores,
            'mean': scores.mean(),
            'std': scores.std()
        }
    
    def predict(self, X):
        """
        Make predictions.
        
        Parameters:
        -----------
        X : pd.DataFrame or np.array
            Features
            
        Returns:
        --------
        np.array
            Predictions
        """
        if not self.is_fitted:
            raise ValueError("Model must be trained before making predictions")
        
        X_scaled = self.scaler.transform(X)
        return self.model.predict(X_scaled)
    
    def get_feature_importance(self, top_n=10):
        """
        Get feature importance (for tree-based models).
        
        Parameters:
        -----------
        top_n : int
            Number of top features to return
            
        Returns:
        --------
        pd.DataFrame
            Feature importance dataframe
        """
        if not self.is_fitted:
            raise ValueError("Model must be trained first")
        
        if hasattr(self.model, 'feature_importances_'):
            importance_df = pd.DataFrame({
                'feature': self.feature_names,
                'importance': self.model.feature_importances_
            }).sort_values('importance', ascending=False).head(top_n)
            
            print("\nTop Feature Importances:")
            print(importance_df.to_string(index=False))
            
            return importance_df
        else:
            print("Feature importance not available for this model type")
            return None
    
    def save_model(self, filepath='models/disaster_response_model.pkl'):
        """
        Save the trained model.
        
        Parameters:
        -----------
        filepath : str
            Path to save the model
        """
        if not self.is_fitted:
            raise ValueError("Model must be trained before saving")
        
        # Create directory if it doesn't exist
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        
        model_data = {
            'model': self.model,
            'scaler': self.scaler,
            'feature_names': self.feature_names,
            'model_type': self.model_type,
            'task': self.task
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
        
        print(f"\nModel saved to: {filepath}")
    
    @classmethod
    def load_model(cls, filepath='models/disaster_response_model.pkl'):
        """
        Load a trained model.
        
        Parameters:
        -----------
        filepath : str
            Path to the saved model
            
        Returns:
        --------
        DisasterResponseModel
            Loaded model instance
        """
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)
        
        instance = cls(model_type=model_data['model_type'], task=model_data['task'])
        instance.model = model_data['model']
        instance.scaler = model_data['scaler']
        instance.feature_names = model_data['feature_names']
        instance.is_fitted = True
        
        print(f"Model loaded from: {filepath}")
        return instance


def train_multiple_models(X, y, task='regression'):
    """
    Train multiple models and compare performance.
    
    Parameters:
    -----------
    X : pd.DataFrame
        Features
    y : pd.Series
        Target
    task : str
        Type of task ('regression' or 'classification')
        
    Returns:
    --------
    dict
        Dictionary of trained models and their scores
    """
    print("\n" + "="*50)
    print("TRAINING MULTIPLE MODELS")
    print("="*50)
    
    models = {}
    results = {}
    
    model_types = ['random_forest', 'gradient_boosting', 'linear', 'decision_tree'] if task == 'regression' else ['random_forest', 'logistic']
    
    for model_type in model_types:
        print(f"\n{'='*50}")
        print(f"Training {model_type}...")
        print('='*50)
        
        model = DisasterResponseModel(model_type=model_type, task=task)
        train_results = model.train(X, y)
        
        models[model_type] = model
        results[model_type] = {
            'train_score': train_results['train_score'],
            'test_score': train_results['test_score']
        }
    
    # Print comparison
    print("\n" + "="*50)
    print("MODEL COMPARISON")
    print("="*50)
    comparison_df = pd.DataFrame(results).T
    print(comparison_df)
    
    # Find best model
    best_model_name = comparison_df['test_score'].idxmax()
    print(f"\nBest Model: {best_model_name}")
    print(f"Test Score: {comparison_df.loc[best_model_name, 'test_score']:.4f}")
    
    return models, results


if __name__ == "__main__":
    # Example usage
    from preprocessing import load_data, preprocess_data
    from feature_engineering import engineer_features
    
    print("Loading and preparing data...")
    df = load_data('data/global_disaster_response_2018_2024.csv')
    
    if df is not None:
        df = preprocess_data('data/global_disaster_response_2018_2024.csv', save_output=False)
        df_engineered, encoders = engineer_features(df, encode_categoricals=True)
        
        # Train model to predict response effectiveness
        print("\n" + "="*50)
        print("PREDICTING RESPONSE EFFECTIVENESS")
        print("="*50)
        
        model = DisasterResponseModel(model_type='random_forest', task='regression')
        
        feature_cols = [
            'affected_population', 'casualties', 'economic_impact_usd',
            'response_time_hours', 'severity_index', 'casualty_rate',
            'disaster_type_encoded', 'region_encoded'
        ]
        
        X, y = model.prepare_features(df_engineered, 'response_effectiveness', feature_cols)
        results = model.train(X, y)
        
        # Cross-validation
        model.cross_validate(X, y)
        
        # Feature importance
        model.get_feature_importance()
        
        # Save model
        model.save_model()
