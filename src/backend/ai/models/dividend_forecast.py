import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error


class DividendForecastModel:
    def __init__(self):
        self.model = RandomForestRegressor()
        self.is_fitted = False

    def train(self, X_train, y_train):
        self.model.fit(X_train, y_train)
        self.is_fitted = True

    def predict(self, X_test):
        if not self.is_fitted:
            raise ValueError(
                "Model has not been trained yet. Call train() before making predictions."
            )
        return self.model.predict(X_test)

    def evaluate(self, X_test, y_test):
        if not self.is_fitted:
            raise ValueError(
                "Model has not been trained yet. Call train() before evaluating."
            )
        predictions = self.model.predict(X_test)
        return mean_squared_error(y_test, predictions)
