import pandas as pd
from sklearn.preprocessing import StandardScaler


class FeatureEngineering:
    def __init__(self):
        self.scaler = StandardScaler()

    def transform(self, data):
        """
        Transform raw data into features for model training.

        Args:
            data (pd.DataFrame): DataFrame containing 'price' and 'dividend' columns

        Returns:
            pd.DataFrame: Transformed features
        """
        features = pd.DataFrame()

        # Basic features from price and dividend
        features["price"] = data["price"]
        features["dividend"] = data["dividend"]

        # Calculate rolling statistics
        features["price_ma"] = data["price"].rolling(window=3, min_periods=1).mean()
        features["dividend_ma"] = (
            data["dividend"].rolling(window=3, min_periods=1).mean()
        )

        # Calculate momentum indicators
        features["price_momentum"] = data["price"].pct_change(periods=3).fillna(0)
        features["dividend_momentum"] = data["dividend"].pct_change(periods=3).fillna(0)

        # Calculate volatility
        features["price_volatility"] = (
            data["price"].rolling(window=3, min_periods=1).std().fillna(0)
        )
        features["dividend_volatility"] = (
            data["dividend"].rolling(window=3, min_periods=1).std().fillna(0)
        )

        # Calculate dividend yield
        features["dividend_yield"] = (data["dividend"] / data["price"]).fillna(0)

        return features
