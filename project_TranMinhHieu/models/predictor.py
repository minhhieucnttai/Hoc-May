
import pandas as pd
import numpy as np
from catboost import CatBoostRegressor
import json

class RecoveryDaysPredictor:
    def __init__(self, model_path, config_path):
        # Load model
        self.model = CatBoostRegressor()
        self.model.load_model(model_path)

        # Load config
        with open(config_path, 'r', encoding='utf-8') as f:
            self.config = json.load(f)

        self.feature_cols = self.config['feature_cols']
        self.categorical_features = self.config['categorical_features']

    def preprocess(self, disaster_data):
        """Tiền xử lý dữ liệu đầu vào."""
        df_input = pd.DataFrame([disaster_data])

        # Log-transform
        df_input['economic_loss_usd_log'] = np.log1p(df_input['economic_loss_usd'])
        df_input['aid_amount_usd_log'] = np.log1p(df_input['aid_amount_usd'])

        # Feature engineering
        df_input['loss_per_casualty'] = df_input['economic_loss_usd'] / (df_input['casualties'] + 1)
        df_input['aid_per_hour'] = df_input['aid_amount_usd'] / (df_input['response_time_hours'] + 1)
        df_input['severity_response_ratio'] = df_input['severity_index'] / (df_input['response_efficiency_score'] + 1) * 100
        df_input['aid_efficiency'] = df_input['aid_amount_usd'] / (df_input['economic_loss_usd'] + 1)

        return df_input[self.feature_cols]

    def predict(self, disaster_data):
        """Dự đoán số ngày hồi phục."""
        X_input = self.preprocess(disaster_data)
        prediction = self.model.predict(X_input)[0]
        return max(1, round(prediction, 1))  # Tối thiểu 1 ngày

    def get_feature_importance(self):
        """Lấy feature importance."""
        return self.model.get_feature_importance(prettified=True)

    def get_model_info(self):
        """Lấy thông tin model."""
        return {
            'iterations': self.model.tree_count_,
            'learning_rate': self.model.get_params().get('learning_rate', 'N/A'),
            'depth': self.model.get_params().get('depth', 'N/A'),
            'features_count': len(self.feature_cols),
            'categorical_features_count': len(self.categorical_features)
        }
