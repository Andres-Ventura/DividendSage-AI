import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import logging

from backend.ai.models.dividend_forecast import DividendForecastModel
from backend.ai.models.feature_engineering import FeatureEngineering


class DividendInsightsTransform:
    """
    A class to process, transform, and derive insights from dividend data.
    """

    def __init__(self):
        self.feature_engineer = FeatureEngineering()
        self.forecast_model = DividendForecastModel()

    def preprocess_data(self, data):
        """
        Preprocess and merge data from Alpha Vantage, Yahoo Finance, and uploaded data into a unified format.

        Args:
            data (dict): Raw data containing alpha_vantage, yahoo_finance, and uploaded_data.

        Returns:
            pd.DataFrame: Unified and cleaned dataset with dates, prices, dividends, and financial metrics.
        """
        logging.basicConfig(level=logging.DEBUG)
        logger = logging.getLogger(__name__)

        # Initialize list to hold processed DataFrames
        processed_dfs = []

        # Process dividends from all sources
        dividend_dfs = []

        # Alpha Vantage Dividends
        alpha_dividends = data["alpha_vantage"]["dividends"]["dividend_history"]

        if alpha_dividends:
            alpha_div_df = pd.DataFrame(alpha_dividends)
            alpha_div_df["date"] = pd.to_datetime(alpha_div_df["date"], errors="coerce")
            alpha_div_df = alpha_div_df.dropna(subset=["date"])
            if not alpha_div_df.empty:
                alpha_div_df = alpha_div_df[["date", "dividend_amount"]].rename(
                    columns={"dividend_amount": "dividend"}
                )
                dividend_dfs.append(alpha_div_df)

        # Yahoo Finance Dividends
        yahoo_div_series = data["yahoo_finance"]["dividends_and_splits"]["dividends"]

        if not yahoo_div_series.empty:
            yahoo_div_df = yahoo_div_series.reset_index()
            yahoo_div_df.columns = ["date", "dividend"]
            yahoo_div_df["date"] = pd.to_datetime(yahoo_div_df["date"], errors="coerce")
            yahoo_div_df = yahoo_div_df.dropna(subset=["date"])
            dividend_dfs.append(yahoo_div_df)

        # Combine all dividends
        dividends = (
            pd.concat(dividend_dfs, ignore_index=True)
            .drop_duplicates("date")
            .sort_values("date")
            if dividend_dfs
            else pd.DataFrame(columns=["date", "dividend"])
        )

        # Process Splits to adjust historical data
        splits_data = data["yahoo_finance"]["dividends_and_splits"]["splits"]

        # Initialize empty DataFrame
        splits_df = pd.DataFrame(columns=["date", "split_ratio"])

        # Check if splits_data exists and is not empty
        if isinstance(splits_data, pd.Series) and not splits_data.empty:
            splits_df = splits_data.reset_index()
            splits_df.columns = ["date", "split_ratio"]
        elif isinstance(splits_data, dict) and splits_data:
            splits_df = pd.DataFrame.from_dict(
                splits_data, orient="index"
            ).reset_index()
            splits_df.columns = ["date", "split_ratio"]

        # Handle timezone-aware timestamps and date conversion
        if not splits_df.empty:
            splits_df["date"] = pd.to_datetime(splits_df["date"], errors="coerce")
            splits_df["date"] = (
                splits_df["date"].dt.tz_localize(None)
                if hasattr(splits_df["date"].dt, "tz_localize")
                else splits_df["date"]
            )
            splits_df = splits_df.dropna(subset=["date"]).sort_values(
                "date", ascending=False
            )

        for _, split in splits_df.iterrows():
            split_date = split["date"]

            # Convert split_date to timezone-naive if it has timezone info
            if hasattr(split_date, "tz"):
                split_date = split_date.tz_localize(None)

            ratio = split["split_ratio"]

            if ratio > 0:
                # Convert both dates to timezone-naive for comparison
                dividends["date"] = dividends["date"].dt.tz_localize(None)
                dividends.loc[dividends["date"] < split_date, "dividend"] /= ratio

        # Process price data from all sources
        price_dfs = []

        # Alpha Vantage Daily Adjusted Prices
        alpha_daily = data["alpha_vantage"]["daily_adjusted"]["daily_data"]

        if alpha_daily:
            alpha_price_df = pd.DataFrame(alpha_daily)
            alpha_price_df["date"] = pd.to_datetime(
                alpha_price_df["date"], errors="coerce"
            )
            alpha_price_df = alpha_price_df.dropna(subset=["date"])
            if "adjusted_close" in alpha_price_df.columns:
                alpha_price_df = alpha_price_df[["date", "adjusted_close"]].rename(
                    columns={"adjusted_close": "price"}
                )
                price_dfs.append(alpha_price_df)

        # Yahoo Finance Historical Prices (Open)
        yahoo_price_dict = data["yahoo_finance"]["historical"]["Open"]

        if yahoo_price_dict:
            # Convert timestamp-keyed dict to dataframe
            yahoo_price_df = pd.DataFrame.from_dict(
                yahoo_price_dict, orient="index"
            ).reset_index()
            yahoo_price_df.columns = ["date", "price"]

            # Convert timezone-aware timestamps to naive datetime
            yahoo_price_df["date"] = yahoo_price_df["date"].dt.tz_localize(None)

            # Handle date conversion and filtering
            yahoo_price_df["date"] = pd.to_datetime(
                yahoo_price_df["date"], errors="coerce"
            )
            yahoo_price_df = yahoo_price_df.dropna(subset=["date"])
            yahoo_price_df = yahoo_price_df[yahoo_price_df["price"] > 0]

            price_dfs.append(yahoo_price_df)

        # Combine all prices
        prices = (
            pd.concat(price_dfs, ignore_index=True)
            .drop_duplicates("date")
            .sort_values("date")
            if price_dfs
            else pd.DataFrame(columns=["date", "price"])
        )

        if not splits_data.empty and not prices.empty:
            for _, split in splits_df.iterrows():
                split_date = split["date"]
                ratio = split["split_ratio"]
                if ratio > 0:
                    prices.loc[prices["date"] < split_date, "price"] /= ratio

        unified_data = (
            pd.merge(prices, dividends, on="date", how="outer")
            .sort_values("date")
            .reset_index(drop=True)
        )

        uploaded_df = pd.DataFrame(data["uploaded_data"])

        if not uploaded_df.empty:
            uploaded_df["date"] = pd.to_datetime(uploaded_df["period"], errors="coerce")
            uploaded_df = uploaded_df.dropna(subset=["date"])
            if "amount" in uploaded_df.columns and "units" in uploaded_df.columns:
                if uploaded_df["units"].iloc[0] == "millions":
                    uploaded_df["amount"] *= 1e6
            # Pivot if multiple metrics exist (example uses same period, so handle accordingly)
            uploaded_df = uploaded_df.drop(columns=["period", "units"], errors="ignore")
            uploaded_df.rename(columns={"amount": "uploaded_amount"}, inplace=True)
            unified_data = pd.merge(unified_data, uploaded_df, on="date", how="left")

        # Process Yahoo Finance Earnings Data
        # Yearly Earnings
        yahoo_price_dict = data["yahoo_finance"]["historical"]["Open"]

        if yahoo_price_dict:
            yahoo_price_df = pd.DataFrame.from_dict(
                yahoo_price_dict, orient="index"
            ).reset_index()
            yahoo_price_df.columns = ["date", "price"]

            # Ensure timezone-naive datetime
            yahoo_price_df["date"] = pd.to_datetime(
                yahoo_price_df["date"]
            ).dt.tz_localize(None)
            yahoo_price_df = yahoo_price_df.dropna(subset=["date"])
            yahoo_price_df = yahoo_price_df[yahoo_price_df["price"] > 0]
            price_dfs.append(yahoo_price_df)

        # Quarterly Earnings (Reported EPS)
        quarterly_earnings = data["yahoo_finance"]["earnings"]["quarterly_earnings"]
        
        if quarterly_earnings:
            q_earnings_df = pd.DataFrame.from_dict(
                quarterly_earnings, orient="index"
            ).reset_index()
            q_earnings_df.columns = [
                "date",
                "eps_estimate",
                "reported_eps",
                "surprise_pct",
            ]
            q_earnings_df = q_earnings_df[["date", "reported_eps"]]

            # Ensure timezone-naive datetime
            q_earnings_df["date"] = pd.to_datetime(
                q_earnings_df["date"]
            ).dt.tz_localize(None)
            q_earnings_df = q_earnings_df.dropna(subset=["date"])

            unified_data = pd.merge(unified_data, q_earnings_df, on="date", how="left")

        # Fill missing dividends with 0 and forward fill missing prices for continuity
        unified_data["dividend"] = unified_data["dividend"].fillna(0)
        unified_data["price"] = unified_data["price"].ffill()
        
        return unified_data

    def feature_engineering(self, data):
        """
        Perform feature engineering on the cleaned dataset.

        Args:
            data (pd.DataFrame): Cleaned dataset.

        Returns:
            pd.DataFrame: Dataset with engineered features.
        """
        data["year"] = data["date"].dt.year
        data["month"] = data["date"].dt.month
        data["rolling_avg_3m"] = data["dividend"].rolling(3).mean()
        data["volatility_3m"] = data["dividend"].rolling(3).std()
        data["dividend_lag_1"] = data["dividend"].shift(1)
        return data.dropna()

    def train_model(self, data):
        """
        Train the forecasting model.

        Args:
            data (pd.DataFrame): Dataset with features and target variable.
        """
        X = self.feature_engineer.transform(data[["price", "dividend"]])
        y = data["dividend"]
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        self.forecast_model.train(X_train, y_train)
        mse = self.forecast_model.evaluate(X_test, y_test)

    def predict_dividends(self, data):
        """
        Predict future dividends.

        Args:
            data (pd.DataFrame): Data for prediction.

        Returns:
            pd.DataFrame: Data with predictions.
        """
        X = self.feature_engineer.transform(data[["price", "dividend"]])
        data["predicted_dividend"] = self.forecast_model.predict(X)
        return data

    def derive_insights(self, data):
        """
        Derive actionable insights from dividend predictions.

        Args:
            data (pd.DataFrame): Data with predictions.

        Returns:
            pd.DataFrame: Data with insights.
        """
        data["dividend_growth"] = data["dividend"] / data["dividend_lag_1"] - 1
        data["high_growth"] = data["dividend_growth"] > 0.1
        return data
