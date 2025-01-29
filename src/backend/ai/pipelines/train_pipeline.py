from ai.models.dividend_forecast import DividendForecastModel
from ai.models.feature_engineering import FeatureEngineering
from sklearn.model_selection import train_test_split
import pandas as pd

class TrainPipeline:
    def __init__(self):
        self.feature_engineer = FeatureEngineering()
        self.forecast_model = DividendForecastModel()

    def run(self, data):
        # Preprocessing and feature engineering
        transformed_data = self.feature_engineer.transform(data)

        # Splitting data
        X = transformed_data
        y = data['future_dividend']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Training model
        self.forecast_model.train(X_train, y_train)

        # Evaluating model
        mse = self.forecast_model.evaluate(X_test, y_test)
        print(f"Model Mean Squared Error: {mse}")
