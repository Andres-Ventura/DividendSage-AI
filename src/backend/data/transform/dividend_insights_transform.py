import pandas as pd
import numpy as np
from sklearn.discriminant_analysis import StandardScaler
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from pandas.tseries.frequencies import to_offset
import logging
from prophet import Prophet

from backend.ai.models.dividend_forecast import DividendForecastModel
from backend.ai.models.feature_engineering import FeatureEngineering


class DividendInsightsTransform:
    def __init__(self):
        # Set up logging at initialization
        self.logger = logging.getLogger(__name__)
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
            )
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
            self.logger.setLevel(logging.INFO)

        self.feature_engineer = FeatureEngineering()
        self.forecast_model = DividendForecastModel()
        self.logger.info("DividendInsightsTransform initialized")

    def detect_dividend_frequency(self, data):
        self.logger.debug(f"Detecting dividend frequency from {len(data)} records")

        if len(data) < 2:
            self.logger.warning("Insufficient data to detect dividend frequency")
            return "Unknown"

        dates = pd.to_datetime(data["date"]).sort_values()
        date_diffs = dates.diff().dropna()

        if date_diffs.empty:
            self.logger.warning("No date differences found for frequency detection")
            return "Unknown"

        mode_freq = date_diffs.mode().iloc[0]
        freq = pd.tseries.frequencies.to_offset(mode_freq).normalize().name
        self.logger.info(f"Detected dividend frequency: {freq}")
        return freq

    # def preprocess_data(self, data):
    #     """
    #     Preprocess and merge data from Alpha Vantage, Yahoo Finance, and uploaded data into a unified format.

    #     Args:
    #         data (dict): Raw data containing alpha_vantage, yahoo_finance, and uploaded_data.

    #     Returns:
    #         pd.DataFrame: Unified and cleaned dataset with dates, prices, dividends, and financial metrics.
    #     """
    #     logging.basicConfig(level=logging.DEBUG)
    #     logging.info("Starting data preprocessing")

    #     dividend_dfs = []

    #     alpha_dividends = data["alpha_vantage"]["dividends"]["dividend_history"]

    #     if alpha_dividends:
    #         logging.info("Processing Alpha Vantage dividends")
    #         alpha_div_df = pd.DataFrame(alpha_dividends)
    #         alpha_div_df["date"] = pd.to_datetime(
    #             alpha_div_df["date"], errors="coerce"
    #         ).dt.tz_localize(None)
    #         alpha_div_df = alpha_div_df.dropna(subset=["date"])
    #         if not alpha_div_df.empty:
    #             logging.debug(
    #                 f"Alpha Vantage dividend shape before processing: {alpha_div_df.shape}"
    #             )
    #             alpha_div_df = alpha_div_df[["date", "amount"]].rename(
    #                 columns={"amount": "dividend"}
    #             )
    #             dividend_dfs.append(alpha_div_df)
    #             logging.debug(
    #                 f"Alpha Vantage dividend shape after processing: {alpha_div_df.shape}"
    #             )

    #     # Yahoo Finance Dividends
    #     logging.debug(
    #         f"Yahoo Finance dividends data type: {type(data['yahoo_finance']['dividends_and_splits']['dividends'])}"
    #     )
    #     yahoo_div_series = data["yahoo_finance"]["dividends_and_splits"]["dividends"]

    #     if not yahoo_div_series.empty:
    #         logging.info("Processing Yahoo Finance dividends")
    #         yahoo_div_df = yahoo_div_series.reset_index()
    #         yahoo_div_df.columns = ["date", "dividend"]
    #         yahoo_div_df["date"] = pd.to_datetime(
    #             yahoo_div_df["date"], errors="coerce"
    #         ).dt.tz_localize(None)
    #         yahoo_div_df = yahoo_div_df.dropna(subset=["date"])
    #         dividend_dfs.append(yahoo_div_df)
    #         logging.debug(f"Yahoo Finance dividend shape: {yahoo_div_df.shape}")

    #     # Combine all dividends
    #     dividends = (
    #         pd.concat(dividend_dfs, ignore_index=True)
    #         .drop_duplicates("date")
    #         .sort_values("date")
    #         if dividend_dfs
    #         else pd.DataFrame(columns=["date", "dividend"])
    #     )

    #     logging.info(f"Combined dividends shape: {dividends.shape}")

    #     logging.debug(
    #         f"Splits data type: {type(data['yahoo_finance']['dividends_and_splits']['splits'])}"
    #     )

    #     # Add frequency detection
    #     div_freq = self.detect_dividend_frequency(dividends)

    #     data["annualization_factor"] = {"quarterly": 4, "monthly": 12, "annual": 1}.get(
    #         div_freq, 4
    #     )

    #     splits_data = data["yahoo_finance"]["dividends_and_splits"]["splits"]

    #     splits_df = pd.DataFrame(columns=["date", "split_ratio"])

    #     splits_df = splits_df.sort_values("date")
    #     cumulative_ratio = 1.0
    #     for idx, row in reversed(list(splits_df.iterrows())):
    #         cumulative_ratio *= row["split_ratio"]
    #         mask = dividends["date"] < row["date"]
    #         dividends.loc[mask, "dividend"] /= row["split_ratio"]

    #     if isinstance(splits_data, pd.Series) and not splits_data.empty:
    #         splits_df = splits_data.reset_index()
    #         splits_df.columns = ["date", "split_ratio"]
    #     elif isinstance(splits_data, dict) and splits_data:
    #         splits_df = pd.DataFrame.from_dict(
    #             splits_data, orient="index"
    #         ).reset_index()
    #         splits_df.columns = ["date", "split_ratio"]

    #     # Handle timezone-aware timestamps and date conversion
    #     if not splits_df.empty:
    #         splits_df["date"] = splits_df["date"].dt.tz_localize(None)
    #         splits_df = splits_df.dropna(subset=["date"]).sort_values(
    #             "date", ascending=True
    #         )  # ✅ Oldest first
    #         splits_df["cumulative_ratio"] = splits_df[
    #             "split_ratio"
    #         ].cumprod()  # Track cumulative split impact

    #     for _, split in splits_df.iterrows():
    #         split_date = split["date"]
    #         ratio = split["split_ratio"]
    #         if ratio > 0:
    #             # Adjust dividends BEFORE this split date by the split ratio
    #             mask = dividends["date"] < split_date
    #             dividends.loc[
    #                 mask, "dividend"
    #             ] /= ratio  # ✅ Dividends adjusted once per split chronologically

    #     # Process price data
    #     logging.info("Processing price data")
    #     price_dfs = []

    #     # Alpha Vantage Daily Adjusted Prices
    #     logging.debug(
    #         f"Alpha Vantage daily data: {bool(data['alpha_vantage']['daily_adjusted']['daily_data'])}"
    #     )
    #     alpha_daily = data["alpha_vantage"]["daily_adjusted"]["daily_data"]

    #     if alpha_daily:
    #         logging.info("Processing Alpha Vantage prices")
    #         alpha_price_df = pd.DataFrame(alpha_daily)
    #         logging.debug(f"Alpha Vantage price columns: {alpha_price_df.columns}")
    #         alpha_price_df["date"] = pd.to_datetime(
    #             alpha_price_df["date"], errors="coerce"
    #         )
    #         alpha_price_df = alpha_price_df.dropna(subset=["date"])
    #         if "adjusted_close" in alpha_price_df.columns:
    #             alpha_price_df = alpha_price_df[["date", "adjusted_close"]].rename(
    #                 columns={"adjusted_close": "price"}
    #             )
    #             price_dfs.append(alpha_price_df)

    #     # Yahoo Finance Historical Prices (Open)
    #     logging.debug(
    #         f"Yahoo Finance price data type: {type(data['yahoo_finance']['historical']['Open'])}"
    #     )
    #     yahoo_price_dict = data["yahoo_finance"]["historical"]["Open"]

    #     if yahoo_price_dict:
    #         # Convert timestamp-keyed dict to dataframe
    #         yahoo_price_df = pd.DataFrame.from_dict(
    #             yahoo_price_dict, orient="index"
    #         ).reset_index()
    #         yahoo_price_df.columns = ["date", "price"]

    #         # Convert timezone-aware timestamps to naive datetime
    #         yahoo_price_df["date"] = yahoo_price_df["date"].dt.tz_localize(None)

    #         # Handle date conversion and filtering
    #         yahoo_price_df["date"] = pd.to_datetime(
    #             yahoo_price_df["date"], errors="coerce"
    #         ).dt.tz_localize(None)

    #         yahoo_price_df = yahoo_price_df.dropna(subset=["date"])
    #         yahoo_price_df = yahoo_price_df[yahoo_price_df["price"] > 0]

    #         price_dfs.append(yahoo_price_df)

    #     # Combine all prices
    #     prices = (
    #         pd.concat(price_dfs, ignore_index=True)
    #         .drop_duplicates("date")
    #         .sort_values("date")
    #         if price_dfs
    #         else pd.DataFrame(columns=["date", "price"])
    #     )

    #     if not splits_data.empty and not prices.empty:
    #         for _, split in splits_df.iterrows():
    #             split_date = split["date"]
    #             ratio = split["split_ratio"]
    #             if ratio > 0:
    #                 prices.loc[prices["date"] < split_date, "price"] /= ratio

    #     unified_data = (
    #         pd.merge(prices, dividends, on="date", how="outer")
    #         .sort_values("date")
    #         .reset_index(drop=True)
    #     )
    #     logging.info(
    #         f"Unified data shape after price/dividend merge: {unified_data.shape}"
    #     )

    #     # Process uploaded data
    #     uploaded_df = pd.DataFrame(data["uploaded_data"])
    #     logging.debug(f"Uploaded data shape: {uploaded_df.shape}")

    #     if not uploaded_df.empty:
    #         logging.info("Processing uploaded data")

    #         uploaded_df["date"] = pd.to_datetime(uploaded_df["period"], errors="coerce")

    #         uploaded_df = uploaded_df.dropna(subset=["date"])
    #         if "amount" in uploaded_df.columns and "units" in uploaded_df.columns:
    #             if uploaded_df["units"].iloc[0] == "millions":
    #                 uploaded_df["amount"] *= 1e6

    #         uploaded_df = uploaded_df.drop(columns=["period", "units"], errors="ignore")
    #         uploaded_df.rename(columns={"amount": "uploaded_amount"}, inplace=True)
    #         unified_data = pd.merge(unified_data, uploaded_df, on="date", how="left")

    #     # Yearly Earnings
    #     yahoo_price_dict = data["yahoo_finance"]["historical"]["Open"]

    #     if yahoo_price_dict:
    #         yahoo_price_df = pd.DataFrame.from_dict(
    #             yahoo_price_dict, orient="index"
    #         ).reset_index()
    #         yahoo_price_df.columns = ["date", "price"]

    #         # Ensure timezone-naive datetime
    #         yahoo_price_df["date"] = pd.to_datetime(
    #             yahoo_price_df["date"]
    #         ).dt.tz_localize(None)
    #         yahoo_price_df = yahoo_price_df.dropna(subset=["date"])
    #         yahoo_price_df = yahoo_price_df[yahoo_price_df["price"] > 0]
    #         price_dfs.append(yahoo_price_df)

    #     # Quarterly Earnings (Reported EPS)
    #     if "earnings" in data["yahoo_finance"]:
    #         quarterly_earnings = data["yahoo_finance"]["earnings"].get(
    #             "quarterly_earnings"
    #         )

    #         if quarterly_earnings:
    #             try:
    #                 q_earnings_df = pd.DataFrame.from_dict(
    #                     quarterly_earnings, orient="index"
    #                 ).reset_index()
    #                 q_earnings_df.columns = [
    #                     "date",
    #                     "eps_estimate",
    #                     "reported_eps",
    #                     "surprise_pct",
    #                 ]
    #                 q_earnings_df = q_earnings_df[["date", "reported_eps"]]

    #                 # Ensure timezone-naive datetime
    #                 q_earnings_df["date"] = pd.to_datetime(
    #                     q_earnings_df["date"]
    #                 ).dt.tz_localize(None)
    #                 q_earnings_df = q_earnings_df.dropna(subset=["date"])

    #                 unified_data = pd.merge(
    #                     unified_data, q_earnings_df, on="date", how="left"
    #                 )
    #                 logging.info("Successfully merged quarterly earnings data")

    #             except Exception as e:
    #                 logging.warning(f"Failed to process quarterly earnings: {str(e)}")
    #         else:
    #             logging.info("No earnings data available in Yahoo Finance response")

    #     # Fill missing dividends with 0 and forward fill missing prices for continuity
    #     unified_data["dividend"] = unified_data["dividend"].fillna(0)
    #     unified_data["price"] = unified_data["price"].ffill().bfill()

    #     logging.info(f"Final unified data shape: {unified_data.shape}")
    #     logging.debug(f"Final unified data columns: {unified_data.columns}")
    #     return unified_data

    def preprocess_data(self, data):
        logging.basicConfig(level=logging.DEBUG)
        logging.info("Starting data preprocessing")

        dividend_dfs = []
        price_dfs = []

        # Process Alpha Vantage dividends
        alpha_dividends = data["alpha_vantage"]["dividends"]["dividend_history"]
        if isinstance(alpha_dividends, (list, dict)):
            logging.info("Processing Alpha Vantage dividends")
            alpha_div_df = pd.DataFrame(alpha_dividends)
            if not alpha_div_df.empty and "date" in alpha_div_df.columns:
                alpha_div_df["date"] = pd.to_datetime(
                    alpha_div_df["date"], errors="coerce"
                ).dt.tz_localize(None)
                alpha_div_df = alpha_div_df.dropna(subset=["date"])
                if not alpha_div_df.empty:
                    alpha_div_df = alpha_div_df[["date", "amount"]].rename(
                        columns={"amount": "dividend"}
                    )
                    dividend_dfs.append(alpha_div_df)

        # Process Yahoo Finance dividends
        yahoo_div_series = data["yahoo_finance"]["dividends_and_splits"]["dividends"]
        if isinstance(yahoo_div_series, pd.Series) and not yahoo_div_series.empty:
            logging.info("Processing Yahoo Finance dividends")
            yahoo_div_df = yahoo_div_series.reset_index()
            yahoo_div_df.columns = ["date", "dividend"]
            yahoo_div_df["date"] = pd.to_datetime(
                yahoo_div_df["date"], errors="coerce"
            ).dt.tz_localize(None)
            yahoo_div_df = yahoo_div_df.dropna(subset=["date"])
            dividend_dfs.append(yahoo_div_df)

        # Process Alpha Vantage prices
        alpha_daily = data["alpha_vantage"]["daily_adjusted"]["daily_data"]
        if isinstance(alpha_daily, (list, dict)):
            logging.info("Processing Alpha Vantage prices")
            alpha_price_df = pd.DataFrame(alpha_daily)
            if not alpha_price_df.empty and "date" in alpha_price_df.columns:
                alpha_price_df["date"] = pd.to_datetime(
                    alpha_price_df["date"], errors="coerce"
                ).dt.tz_localize(None)
                alpha_price_df = alpha_price_df.dropna(subset=["date"])
                if "adjusted_close" in alpha_price_df.columns:
                    alpha_price_df = alpha_price_df[["date", "adjusted_close"]].rename(
                        columns={"adjusted_close": "price"}
                    )
                    price_dfs.append(alpha_price_df)

        # Process Yahoo Finance prices
        yahoo_price_dict = data["yahoo_finance"]["historical"]["Open"]
        if isinstance(yahoo_price_dict, dict) and yahoo_price_dict:
            logging.info("Processing Yahoo Finance prices")
            yahoo_price_df = pd.DataFrame.from_dict(
                yahoo_price_dict, orient="index"
            ).reset_index()
            yahoo_price_df.columns = ["date", "price"]
            yahoo_price_df["date"] = pd.to_datetime(
                yahoo_price_df["date"]
            ).dt.tz_localize(None)
            yahoo_price_df = yahoo_price_df.dropna(subset=["date"])
            yahoo_price_df = yahoo_price_df[yahoo_price_df["price"] > 0]
            price_dfs.append(yahoo_price_df)

        # Combine dividends and prices
        if dividend_dfs:
            dividends = pd.concat(dividend_dfs, ignore_index=True)
            dividends = dividends.drop_duplicates("date").sort_values("date")
        else:
            dividends = pd.DataFrame(columns=["date", "dividend"])

        if price_dfs:
            prices = pd.concat(price_dfs, ignore_index=True)
            prices = prices.drop_duplicates("date").sort_values("date")
        else:
            prices = pd.DataFrame(columns=["date", "price"])

        # Merge prices and dividends
        unified_data = pd.merge(prices, dividends, on="date", how="outer").sort_values(
            "date"
        )

        # Fill missing values
        unified_data["dividend"] = unified_data["dividend"].fillna(0)
        unified_data["price"] = unified_data["price"].ffill().bfill()

        logging.info(f"Final unified data shape: {unified_data.shape}")
        logging.debug(f"Final unified data columns: {unified_data.columns}")
        
        
        print(f"UNIFIED_DATA === {unified_data}")

        return unified_data

    def train_prophet_model(self, data):
        df = data[["date", "dividend"]].rename(columns={"date": "ds", "dividend": "y"})
        model = Prophet(seasonality_mode="multiplicative")
        model.fit(df)
        return model

    def feature_engineering(self, data):
        """Feature engineering with enhanced logging"""
        self.logger.info("Starting feature engineering")
        try:

            original_shape = data.shape
            data = data.copy()

            # Log each feature creation step
            self.logger.debug("Creating date-based features")
            data["year"] = data["date"].dt.year
            data["month"] = data["date"].dt.month

            self.logger.debug("Calculating rolling averages")
            data["rolling_avg_3m"] = (
                data["dividend"].rolling(3, min_periods=3).mean().ffill()
            )

            self.logger.debug("Calculating volatility metrics")
            data["volatility_3m"] = data["dividend"].rolling(3, min_periods=1).std()

            data["rolling_avg_3m"] = data["rolling_avg_3m"].fillna(0.0)
            data["volatility_3m"] = data["volatility_3m"].fillna(0.0)

            self.logger.info(
                f"Feature engineering complete. "
                f"Shape change: {original_shape} → {data.shape}"
            )

            return data

        except Exception as e:
            self.logger.error(f"Feature engineering failed: {str(e)}", exc_info=True)
            raise

    def train_model(self, data):
        """Model training with enhanced error handling and data validation"""
        self.logger.info("Starting model training")

        try:
            # Validate input data
            required_columns = ["price", "dividend"]
            missing_columns = [col for col in required_columns if col not in data.columns]
            if missing_columns:
                raise ValueError(
                    f"Missing required columns for training: {missing_columns}"
                )

            # Prepare feature DataFrame
            self.logger.debug("Preparing features for model training")
            features_df = data[required_columns].copy()

            # Handle missing values
            features_df.fillna(0, inplace=True)

            # Apply feature engineering
            X = self.feature_engineer.transform(features_df)
            y = features_df["dividend"].values

            # Ensure X has correct shape
            expected_feature_count = X.shape[1]
            column_names = [f"feature_{i}" for i in range(expected_feature_count)]
            X = pd.DataFrame(X, index=features_df.index, columns=column_names)

            self.logger.debug(f"Feature matrix shape after transformation: {X.shape}")

            # Ensure no NaNs or infinite values remain
            if np.isnan(X.to_numpy()).any() or np.isinf(X.to_numpy()).any():
                self.logger.warning(
                    "NaN or infinite values detected in features. Applying fixes..."
                )
                X = np.nan_to_num(X.to_numpy(), posinf=1e6, neginf=-1e6, nan=0)
                X = pd.DataFrame(X, index=features_df.index, columns=column_names)

            # Scale features
            self.logger.debug("Scaling features")
            self.scaler = StandardScaler()
            X = self.scaler.fit_transform(X)

            # Split data and train
            self.logger.info("Splitting data and training model")
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42
            )

            self.forecast_model.train(X_train, y_train)
            mse = self.forecast_model.evaluate(X_test, y_test)
            self.logger.info(f"Model training complete. MSE: {mse:.4f}")

        except Exception as e:
            self.logger.error("Model training failed", exc_info=True)
            raise RuntimeError(f"Model training error: {str(e)}") from e

    def prepare_features(self, df):
        """
    Enhanced feature preparation with proper error handling
    """
        try:
            features = pd.DataFrame()

            # Ensure required columns exist
            required_columns = ["date", "price", "dividend"]
            if not all(col in df.columns for col in required_columns):
                raise ValueError(f"Missing required columns: {required_columns}")

            df["date"] = pd.to_datetime(df["date"])  # Ensure date is in datetime format

            # Basic time-based features
            features["year"] = df["date"].dt.year
            features["quarter"] = df["date"].dt.quarter
            features["month"] = df["date"].dt.month

            # Price-based features
            features["price_momentum"] = df["price"].pct_change(4).fillna(0)
            features["price_volatility"] = df["price"].rolling(window=4).std().fillna(0)

            # Dividend-based features
            features["dividend_momentum"] = df["dividend"].pct_change(4).fillna(0)
            features["dividend_yield"] = (df["dividend"] * 4 / df["price"]).fillna(0)
            features["payout_trend"] = features["dividend_yield"].pct_change().fillna(0)

            return features

        except Exception as e:
            logging.error(f"Error in prepare_features: {str(e)}")
            raise

    def derive_insights(self, data, growth_threshold=0.15, yield_threshold=0.025, window=5):
        """
    Derive actionable insights with predictive modeling for future metrics.
    Enhanced error handling and type checking added.
    """
        try:
            if not isinstance(data, pd.DataFrame):
                raise ValueError("Input data must be a pandas DataFrame")

            if data.empty:
                raise ValueError("Input DataFrame is empty")

            logging.debug(f"DataFrame columns before processing: {data.columns.tolist()}")

        # Ensure required columns exist
            required_columns = ["date", "dividend", "price"]
            missing_columns = [col for col in required_columns if col not in data.columns]

            if missing_columns:
                raise ValueError(f"Missing required columns: {missing_columns}")

        # Sort by date and reset index
            data = data.sort_values("date").reset_index(drop=True)

        # Extract last date and create future date range
            last_date = pd.to_datetime(data["date"]).max()
            future_dates = pd.date_range(
            start=last_date + pd.DateOffset(months=3), 
            periods=8,  # 2 years of quarterly predictions
            freq="Q"
            )

        # Train Prophet model on historical data
            prophet_model = self.train_prophet_model(data)
        
        # Create future dataframe for Prophet
            future_df = pd.DataFrame({'ds': future_dates})
        
        # Generate forecast without bounds
            forecast = prophet_model.predict(future_df)
        
        # Extract predictions and convert negative values to their absolute values
            predictions = pd.DataFrame({
            'date': forecast['ds'],
            'dividend': np.abs(forecast['yhat']),  # Convert negative predictions to positive
            'price': None
            })
        
        # Estimate future prices using simple trend extrapolation
            historical_price_changes = data['price'].pct_change().mean()
            last_price = data['price'].iloc[-1]
        
            for i, _ in enumerate(predictions.index):
                predictions.loc[predictions.index[i], 'price'] = last_price * (1 + historical_price_changes) ** (i + 1)

        # Add engineered features for the predictions
            predictions['year'] = predictions['date'].dt.year
            predictions['month'] = predictions['date'].dt.month
            predictions['rolling_avg_3m'] = predictions['dividend'].rolling(3, min_periods=1).mean()
            predictions['volatility_3m'] = predictions['dividend'].rolling(3, min_periods=1).std()

        # Combine historical and predicted data
            full_data = pd.concat([
            data,
            predictions
            ]).reset_index(drop=True)

        # Fill any missing values in the engineered columns
            full_data['rolling_avg_3m'] = full_data['rolling_avg_3m'].fillna(method='ffill')
            full_data['volatility_3m'] = full_data['volatility_3m'].fillna(method='ffill')

        # Sort by date
            full_data = full_data.sort_values('date').reset_index(drop=True)

            return full_data

        except Exception as e:
            logging.error(f"Error in derive_insights: {str(e)}")
            raise