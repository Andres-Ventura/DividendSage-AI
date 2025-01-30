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
        logging.info("Starting data preprocessing")

        # Log raw data from all sources
        logging.info("Raw Alpha Vantage Data:")
        logging.info(f"- Daily Adjusted: {data['alpha_vantage']['daily_adjusted']}")
        logging.info(f"- Dividends: {data['alpha_vantage']['dividends']}")

        logging.info("Raw Yahoo Finance Data:")
        logging.info(f"- Historical: {data['yahoo_finance']['historical']}")
        logging.info(
            f"- Dividends and Splits: {data['yahoo_finance']['dividends_and_splits']}"
        )
        if "earnings" in data["yahoo_finance"]:
            logging.info(f"- Earnings: {data['yahoo_finance']['earnings']}")

        logging.info("Raw Uploaded Data:")
        logging.info(f"- Data: {data['uploaded_data']}")

        # Process dividends from all sources
        dividend_dfs = []

        # Alpha Vantage Dividends
        logging.debug(
            f"Alpha Vantage dividends data: {data['alpha_vantage']['dividends']}"
        )
        alpha_dividends = data["alpha_vantage"]["dividends"]["dividend_history"]

        if alpha_dividends:
            logging.info("Processing Alpha Vantage dividends")
            alpha_div_df = pd.DataFrame(alpha_dividends)
            alpha_div_df["date"] = pd.to_datetime(
                alpha_div_df["date"], errors="coerce"
            ).dt.tz_localize(None)
            alpha_div_df = alpha_div_df.dropna(subset=["date"])
            if not alpha_div_df.empty:
                logging.debug(
                    f"Alpha Vantage dividend shape before processing: {alpha_div_df.shape}"
                )
                alpha_div_df = alpha_div_df[["date", "amount"]].rename(
                    columns={"amount": "dividend"}
                )
                dividend_dfs.append(alpha_div_df)
                logging.debug(
                    f"Alpha Vantage dividend shape after processing: {alpha_div_df.shape}"
                )

        # Yahoo Finance Dividends
        logging.debug(
            f"Yahoo Finance dividends data type: {type(data['yahoo_finance']['dividends_and_splits']['dividends'])}"
        )
        yahoo_div_series = data["yahoo_finance"]["dividends_and_splits"]["dividends"]

        if not yahoo_div_series.empty:
            logging.info("Processing Yahoo Finance dividends")
            yahoo_div_df = yahoo_div_series.reset_index()
            yahoo_div_df.columns = ["date", "dividend"]
            yahoo_div_df["date"] = pd.to_datetime(
                yahoo_div_df["date"], errors="coerce"
            ).dt.tz_localize(None)
            yahoo_div_df = yahoo_div_df.dropna(subset=["date"])
            dividend_dfs.append(yahoo_div_df)
            logging.debug(f"Yahoo Finance dividend shape: {yahoo_div_df.shape}")

        # Combine all dividends
        dividends = (
            pd.concat(dividend_dfs, ignore_index=True)
            .drop_duplicates("date")
            .sort_values("date")
            if dividend_dfs
            else pd.DataFrame(columns=["date", "dividend"])
        )
        logging.info(f"Combined dividends shape: {dividends.shape}")

        # Process Splits
        logging.debug(
            f"Splits data type: {type(data['yahoo_finance']['dividends_and_splits']['splits'])}"
        )
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
            splits_df["date"] = splits_df["date"].dt.tz_localize(None)
            splits_df = splits_df.dropna(subset=["date"]).sort_values(
                "date", ascending=True
            )  # ✅ Oldest first
            splits_df["cumulative_ratio"] = splits_df[
                "split_ratio"
            ].cumprod()  # Track cumulative split impact

        for _, split in splits_df.iterrows():
            split_date = split["date"]
            ratio = split["split_ratio"]
            if ratio > 0:
                # Adjust dividends BEFORE this split date by the split ratio
                mask = dividends["date"] < split_date
                dividends.loc[
                    mask, "dividend"
                ] /= ratio  # ✅ Dividends adjusted once per split chronologically

        # Process price data
        logging.info("Processing price data")
        price_dfs = []

        # Alpha Vantage Daily Adjusted Prices
        logging.debug(
            f"Alpha Vantage daily data: {bool(data['alpha_vantage']['daily_adjusted']['daily_data'])}"
        )
        alpha_daily = data["alpha_vantage"]["daily_adjusted"]["daily_data"]

        if alpha_daily:
            logging.info("Processing Alpha Vantage prices")
            alpha_price_df = pd.DataFrame(alpha_daily)
            logging.debug(f"Alpha Vantage price columns: {alpha_price_df.columns}")
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
        logging.debug(
            f"Yahoo Finance price data type: {type(data['yahoo_finance']['historical']['Open'])}"
        )
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
            ).dt.tz_localize(None)

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
        logging.info(
            f"Unified data shape after price/dividend merge: {unified_data.shape}"
        )

        # Process uploaded data
        uploaded_df = pd.DataFrame(data["uploaded_data"])
        logging.debug(f"Uploaded data shape: {uploaded_df.shape}")

        if not uploaded_df.empty:
            logging.info("Processing uploaded data")
            uploaded_df["date"] = pd.to_datetime(uploaded_df["period"], errors="coerce")
            uploaded_df = uploaded_df.dropna(subset=["date"])
            if "amount" in uploaded_df.columns and "units" in uploaded_df.columns:
                if uploaded_df["units"].iloc[0] == "millions":
                    uploaded_df["amount"] *= 1e6
            # Pivot if multiple metrics exist (example uses same period, so handle accordingly)
            uploaded_df = uploaded_df.drop(columns=["period", "units"], errors="ignore")
            uploaded_df.rename(columns={"amount": "uploaded_amount"}, inplace=True)
            unified_data = pd.merge(unified_data, uploaded_df, on="date", how="left")

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
        if "earnings" in data["yahoo_finance"]:
            quarterly_earnings = data["yahoo_finance"]["earnings"].get(
                "quarterly_earnings"
            )

            if quarterly_earnings:
                try:
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

                    unified_data = pd.merge(
                        unified_data, q_earnings_df, on="date", how="left"
                    )
                    logging.info("Successfully merged quarterly earnings data")

                except Exception as e:
                    logging.warning(f"Failed to process quarterly earnings: {str(e)}")
            else:
                logging.info("No earnings data available in Yahoo Finance response")

        # Fill missing dividends with 0 and forward fill missing prices for continuity
        unified_data["dividend"] = unified_data["dividend"].fillna(0)
        unified_data["price"] = unified_data["price"].ffill().bfill()

        logging.info(f"Final unified data shape: {unified_data.shape}")
        logging.debug(f"Final unified data columns: {unified_data.columns}")
        return unified_data

    def feature_engineering(self, data):
        """Perform feature engineering with improved NaN handling"""
        # Keep original data for reference
        data = data.copy()

        # Date-based features
        data["year"] = data["date"].dt.year
        data["month"] = data["date"].dt.month

        # Rolling calculations with minimum 1 period
        data["rolling_avg_3m"] = data["dividend"].rolling(3, min_periods=1).mean()
        data["volatility_3m"] = data["dividend"].rolling(3, min_periods=1).std()

        # Lag feature with forward fill
        data["dividend_lag_1"] = data["dividend"].shift(1).fillna(0)

        # Only drop rows with NaN in critical features
        return data.dropna(subset=["rolling_avg_3m", "volatility_3m"])

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

    def derive_insights(
        self, data, growth_threshold=0.1, yield_threshold=0.03, window=3
    ):
        """
        Derive actionable insights with predictive modeling for future metrics.
        """
        # Ensure chronological order and data types
        data = data.sort_values("date").reset_index(drop=True)
        data["dividend"] = pd.to_numeric(data["dividend"], errors="coerce")
        data["price"] = pd.to_numeric(data["price"], errors="coerce")

        # Create future dates for predictions (5 years of quarterly data)
        last_date = data["date"].max()
        future_dates = pd.date_range(
            start=last_date + pd.DateOffset(months=3),
            periods=20,  # 5 years * 4 quarters
            freq="QE",
        )

        # Initialize predictive models
        from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
        from sklearn.preprocessing import StandardScaler

        # Prepare historical features for prediction
        def prepare_features(df):
            features = pd.DataFrame()
            features["year"] = df["date"].dt.year
            features["quarter"] = df["date"].dt.quarter
            features["price_momentum"] = pd.to_numeric(
                df["price"].pct_change(4), errors="coerce"
            ).fillna(0)
            features["dividend_momentum"] = pd.to_numeric(
                df["dividend"].pct_change(4), errors="coerce"
            ).fillna(0)
            if "reported_eps" in df.columns:
                features["eps_momentum"] = pd.to_numeric(
                    df["reported_eps"].pct_change(4), errors="coerce"
                ).fillna(0)
            return features

        historical_features = prepare_features(data)
        future_features = pd.DataFrame(
            {
                "year": future_dates.year,
                "quarter": future_dates.quarter,
                "price_momentum": [historical_features["price_momentum"].mean()]
                * len(future_dates),
                "dividend_momentum": [historical_features["dividend_momentum"].mean()]
                * len(future_dates),
            }
        )

        scaler = StandardScaler()
        historical_features_scaled = scaler.fit_transform(historical_features)
        future_features_scaled = scaler.transform(future_features)

        # Train models and predict future values
        predictions = pd.DataFrame(index=future_dates)

        # Dividend predictions
        dividend_model = GradientBoostingRegressor(random_state=42)
        dividend_model.fit(historical_features_scaled, data["dividend"])
        predictions["dividend"] = dividend_model.predict(future_features_scaled)

        # Price predictions
        price_model = GradientBoostingRegressor(random_state=42)
        price_model.fit(historical_features_scaled, data["price"])
        predictions["price"] = price_model.predict(future_features_scaled)

        # Reset index to make date a column
        predictions = predictions.reset_index().rename(columns={"index": "date"})

        # Combine historical and predicted data
        combined_data = pd.concat(
            [data[["date", "dividend", "price"]], predictions]
        ).reset_index(drop=True)

        # Calculate metrics with proper type handling
        combined_data["dividend_yield"] = (
            pd.to_numeric(combined_data["dividend"], errors="coerce")
            / pd.to_numeric(combined_data["price"], errors="coerce")
        ).fillna(0)

        combined_data["dividend_growth"] = (
            pd.to_numeric(combined_data["dividend"], errors="coerce").pct_change(4)
        ).fillna(0)

        combined_data["rolling_avg_growth"] = (
            combined_data["dividend_growth"].rolling(window=4, min_periods=1).mean()
        ).fillna(0)

        combined_data["growth_volatility"] = (
            combined_data["dividend_growth"].rolling(window=4, min_periods=1).std()
        ).fillna(0)

        # Calculate boolean metrics with explicit type conversion
        combined_data["yield_alert"] = (
            combined_data["dividend_yield"] < yield_threshold
        ).astype(int)

        combined_data["high_growth"] = (
            combined_data["dividend_growth"] > growth_threshold
        ).astype(int)

        combined_data["negative_growth"] = (
            combined_data["dividend_growth"] < 0
        ).astype(int)

        combined_data["growth_stable"] = (
            combined_data["growth_volatility"]
            < combined_data["growth_volatility"].quantile(0.25)
        ).astype(int)

        # Calculate composite score with proper type handling
        combined_data["dividend_score"] = (
            (1 - combined_data["yield_alert"]) * 0.1
            + combined_data["rolling_avg_growth"].rank(pct=True) * 0.6
            + combined_data["dividend_yield"].rank(pct=True) * 0.3
        ).round(2)

        combined_data["growth_momentum"] = (
            combined_data["dividend_growth"].diff() > 0
        ).astype(int)

        combined_data["yield_trend"] = (
            combined_data["dividend_yield"].pct_change() > 0
        ).astype(int)

        # Calculate confidence intervals
        volatility = combined_data["growth_volatility"].mean()
        predictions["projection_low"] = predictions["dividend"] * (1 - volatility)
        predictions["projection_high"] = predictions["dividend"] * (1 + volatility)

        # Store future projections with proper date handling
        annual_predictions = (
            predictions.set_index("date")
            .resample("Y")
            .agg(
                {
                    "dividend": "sum",
                    "projection_low": "sum",
                    "projection_high": "sum",
                    "price": "mean",
                }
            )
        )

        combined_data["future_projections"] = {
            "quarterly_detail": predictions.to_dict("records"),
            "annual_summary": annual_predictions.to_dict("index"),
            "metadata": {
                "model_type": "GradientBoosting",
                "confidence_interval": float(volatility),
                "base_dividend": float(data["dividend"].iloc[-1]),
                "base_price": float(data["price"].iloc[-1]),
            },
        }

        return combined_data
