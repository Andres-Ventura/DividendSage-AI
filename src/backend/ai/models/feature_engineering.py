import pandas as pd
from sklearn.preprocessing import StandardScaler

class FeatureEngineering:
    def __init__(self):
        self.scaler = StandardScaler()

    def transform(self, data):
        # Example feature engineering logic
        features = data[['dividend_amount', 'adjusted_close']]
        scaled_features = self.scaler.fit_transform(features)
        return pd.DataFrame(scaled_features, columns=features.columns)
