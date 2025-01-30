from typing import Optional, Dict, Any, Tuple
import pandas as pd
from backend.data.combined_data_provider import CombinedDataProvider
from backend.data.transform.dividend_insights_transform import DividendInsightsTransform
import asyncio
import signal
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings
import logging
import matplotlib.dates as mdates


class DividendInsightsLoader:
    """
    A class to load transformed dividend insights into a CSV file, train the model, and generate insights.
    """

    def __init__(self, output_path):
        self.output_path = output_path
        self.transformer = DividendInsightsTransform()
        self.data_provider = CombinedDataProvider()

        # Suppress matplotlib font manager warnings
        logging.getLogger("matplotlib.font_manager").setLevel(logging.ERROR)
        warnings.filterwarnings("ignore", category=UserWarning, module="matplotlib")

        # Set matplotlib to use a simple style that doesn't depend on system fonts
        plt.style.use("seaborn-v0_8-darkgrid")

        # Create output directory if it doesn't exist
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    async def load_data_and_train_model(
        self, symbol: str, uploaded_file: Optional[str] = None
    ):
        """
        Fetch, transform, and load data into a CSV file, then train the model.
        """
        try:
            print(f"Starting data fetch for symbol: {symbol}")
            raw_data = await self.data_provider.fetch_combined_data(
                symbol, uploaded_file
            )
            self._log_dataframe_status("Raw data", raw_data)

            print("Starting data preprocessing...")
            unified_data = await self.transformer.preprocess_data(raw_data)
            self._log_dataframe_status("Preprocessing", unified_data)
            self._validate_dataframe(unified_data, "Data preprocessing")

            print("Starting feature engineering...")
            engineered_data = await self.transformer.feature_engineering(unified_data)
            self._log_dataframe_status("Feature engineering", engineered_data)
            self._validate_dataframe(engineered_data, "Feature engineering")

            print(f"Saving data to {self.output_path}...")
            engineered_data.to_csv(self.output_path, index=False, float_format="%.10g")
            print(f"Data successfully saved to '{self.output_path}'.")

            print("Starting model training...")
            self.transformer.train_model(engineered_data)
            print("Model training completed successfully.")

        except Exception as e:
            print(f"ERROR in load_data_and_train_model: {str(e)}")
            print(f"Error type: {type(e).__name__}")
            raise

    async def generate_insights(
        self, symbol: str, uploaded_file: Optional[str] = None
    ) -> Tuple[list, str, str]:
        """
        Generate insights and save both CSV and chart visualization.
        Returns: Tuple of (insights_data, csv_path, chart_path)
        """
        try:
            print(f"Starting insights generation for symbol: {symbol}")
            raw_data = await self.data_provider.fetch_combined_data(
                symbol, uploaded_file
            )
            print("Raw data fetched for insights generation")

            unified_data = await self._process_data_async(raw_data)
            engineered_data = await self._engineer_features_async(unified_data)

            # Clean any infinite or NaN values before filtering
            engineered_data = engineered_data.replace(
                [float("inf"), float("-inf")], None
            )

            # Filter for dividend periods
            dividend_periods = engineered_data[engineered_data["dividend"] > 0].copy()

            print("Deriving insights...")
            insights_data = await self._derive_insights_async(dividend_periods)
            print("Insights data:", insights_data)  # Debug print

            if isinstance(insights_data, pd.DataFrame):
                # Convert date column to datetime if it isn't already
                insights_data["date"] = pd.to_datetime(insights_data["date"])

                # Get the last historical date
                last_historical_date = dividend_periods["date"].max()

                # Split the data using boolean indexing
                historical_mask = insights_data["date"] <= pd.to_datetime(
                    last_historical_date
                )
                historical_data = insights_data[historical_mask].copy()
                future_data = insights_data[~historical_mask].copy()

                # Create paths for outputs
                base_path = Path(self.output_path).parent
                csv_path = base_path / f"{symbol}_insights.csv"
                chart_path = base_path / f"{symbol}_insights_chart.png"

                # Save to CSV with clear separation
                insights_df = pd.concat(
                    [
                        historical_data,
                        pd.DataFrame({"date": [pd.NaT]}),  # Add separator row
                        future_data.assign(
                            data_type="predicted"
                        ),  # Mark predicted data
                    ]
                )

                # Format dates for CSV
                insights_df["date"] = insights_df["date"].dt.strftime("%Y-%m-%d")

                insights_df.to_csv(csv_path, index=False, float_format="%.10g")
                print(f"Insights saved to CSV: {csv_path}")

                # Generate visualization
                self._generate_insights_chart(
                    historical_data, future_data, symbol, chart_path
                )
                print(f"Chart saved to: {chart_path}")

                return insights_data.to_dict("records"), str(csv_path), str(chart_path)
            else:
                raise ValueError("Insights generation did not return a valid DataFrame")

        except Exception as e:
            print(f"ERROR in generate_insights: {str(e)}")
            print(f"Error type: {type(e)}")
            import traceback

            traceback.print_exc()
            raise

    def _validate_dataframe(self, df: Optional[pd.DataFrame], stage: str):
        """Validate that the DataFrame is not None or empty."""
        if df is None or df.empty:
            raise ValueError(f"{stage} resulted in empty or None data")

    def _log_dataframe_status(self, stage: str, df: Optional[pd.DataFrame]):
        """Log the status of a DataFrame at a given processing stage."""
        print(
            f"{stage} shape: {df.shape if isinstance(df, pd.DataFrame) else 'Not a DataFrame'}"
        )

    async def _process_data_async(self, raw_data: pd.DataFrame) -> pd.DataFrame:
        """Process data asynchronously using asyncio.to_thread."""
        return await asyncio.to_thread(self.transformer.preprocess_data, raw_data)

    async def _engineer_features_async(
        self, unified_data: pd.DataFrame
    ) -> pd.DataFrame:
        """Engineer features asynchronously using asyncio.to_thread."""
        return await asyncio.to_thread(
            self.transformer.feature_engineering, unified_data
        )

    async def _derive_insights_async(self, dividend_periods: pd.DataFrame):
        """Derive insights asynchronously using asyncio.to_thread."""
        insights = await asyncio.to_thread(
            self.transformer.derive_insights,
            dividend_periods,
            growth_threshold=0.15,
            yield_threshold=0.025,
            window=5,
        )

        # Clean any non-JSON-serializable values
        if isinstance(insights, pd.DataFrame):
            insights = insights.replace([float("inf"), float("-inf")], None)
            insights = insights.infer_objects(copy=False)

        return insights

    def _generate_insights_chart(
        self,
        historical_data: pd.DataFrame,
        future_data: pd.DataFrame,
        symbol: str,
        chart_path: Path,
    ) -> None:
        """
        Generate an enhanced visualization showing both historical and predicted values.
        """
        plt.switch_backend("Agg")

        # Create subplot grid
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 12), height_ratios=[2, 1])

        # Ensure dates are datetime
        historical_data["date"] = pd.to_datetime(historical_data["date"])
        future_data["date"] = pd.to_datetime(future_data["date"])

        # Plot 1: Dividends and Projections
        ax1.plot(
            historical_data["date"],
            historical_data["dividend"],
            label="Historical Dividends",
            color="blue",
            marker="o",
        )
        ax1.plot(
            future_data["date"],
            future_data["dividend"],
            label="Projected Dividends",
            color="red",
            linestyle="--",
        )

        # Add confidence intervals if available
        if (
            "projection_low" in future_data.columns
            and "projection_high" in future_data.columns
        ):
            ax1.fill_between(
                future_data["date"],
                future_data["projection_low"],
                future_data["projection_high"],
                alpha=0.2,
                color="red",
                label="Projection Confidence Interval",
            )

        ax1.set_title(f"Dividend History and Projections - {symbol}")
        ax1.set_ylabel("Dividend Amount ($)")
        ax1.legend()

        # Plot 2: Metrics Dashboard
        metrics = ["dividend_yield", "dividend_score", "growth_momentum"]
        colors = ["green", "purple", "orange"]

        for metric, color in zip(metrics, colors):
            if metric in historical_data.columns:
                ax2.plot(
                    historical_data["date"],
                    historical_data[metric],
                    label=metric.replace("_", " ").title(),
                    color=color,
                )
            if metric in future_data.columns:
                ax2.plot(
                    future_data["date"],
                    future_data[metric],
                    linestyle="--",
                    color=color,
                )

        ax2.set_title("Key Metrics Overview")
        ax2.set_ylabel("Metric Value")
        ax2.legend()

        # Formatting
        for ax in [ax1, ax2]:
            ax.grid(True)
            ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
            ax.xaxis.set_major_locator(mdates.AutoDateLocator())
            plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha="right")

        plt.tight_layout()
        plt.savefig(chart_path, dpi=300, bbox_inches="tight")
        plt.close()


def handle_shutdown(signum, frame):
    print("\nShutting down gracefully...")
    # Perform any cleanup here
    exit(0)


signal.signal(signal.SIGINT, handle_shutdown)
