import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

from backend.ai.models.dividend_forecast import DividendForecastModel
from backend.ai.models.feature_engineering import FeatureEngineering

class DividendInsightsTransform:
    """
    A class to process, transform, and derive insights from dividend data.
    """

    def __init__(self):
        self.feature_engineer = FeatureEngineering()
        self.forecast_model = DividendForecastModel()

    def preprocess_data(self, data_sources):
        """
        Preprocess and merge data from multiple sources into a unified format.

        Args:
            data_sources (list of DataFrames): List of raw data sources.

        Returns:
            pd.DataFrame: Unified and cleaned dataset.
        """
        processed_frames = []

        for source in data_sources:
            if isinstance(source, dict):
                df = pd.DataFrame(source)
            elif isinstance(source, pd.DataFrame):
                df = source.copy()
            else:
                raise ValueError("Unsupported data source type")

            df['date'] = pd.to_datetime(df['date'], errors='coerce')
            df = df.dropna(subset=['date'])
            df = df.sort_values('date')
            df.rename(columns={
                'dividend_amount': 'dividend',
                'adjusted_close': 'price'
            }, inplace=True)
            processed_frames.append(df)

        unified_data = pd.concat(processed_frames, ignore_index=True)
        return unified_data

    def feature_engineering(self, data):
        """
        Perform feature engineering on the cleaned dataset.

        Args:
            data (pd.DataFrame): Cleaned dataset.

        Returns:
            pd.DataFrame: Dataset with engineered features.
        """
        data['year'] = data['date'].dt.year
        data['month'] = data['date'].dt.month
        data['rolling_avg_3m'] = data['dividend'].rolling(3).mean()
        data['volatility_3m'] = data['dividend'].rolling(3).std()
        data['dividend_lag_1'] = data['dividend'].shift(1)
        return data.dropna()

    def train_model(self, data):
        """
        Train the forecasting model.

        Args:
            data (pd.DataFrame): Dataset with features and target variable.
        """
        X = self.feature_engineer.transform(data[['price', 'dividend']])
        y = data['dividend']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        self.forecast_model.train(X_train, y_train)
        mse = self.forecast_model.evaluate(X_test, y_test)
        print(f"Model trained. MSE: {mse}")

    def predict_dividends(self, data):
        """
        Predict future dividends.

        Args:
            data (pd.DataFrame): Data for prediction.

        Returns:
            pd.DataFrame: Data with predictions.
        """
        X = self.feature_engineer.transform(data[['price', 'dividend']])
        data['predicted_dividend'] = self.forecast_model.predict(X)
        return data

    def derive_insights(self, data):
        """
        Derive actionable insights from dividend predictions.

        Args:
            data (pd.DataFrame): Data with predictions.

        Returns:
            pd.DataFrame: Data with insights.
        """
        data['dividend_growth'] = data['dividend'] / data['dividend_lag_1'] - 1
        data['high_growth'] = data['dividend_growth'] > 0.1
        return data
