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

            print("Training model...")
            self.transformer.train_model(engineered_data)
            print("Model training completed")

            engineered_data["date"] = pd.to_datetime(engineered_data["date"])

            engineered_data = engineered_data.replace(
                [float("inf"), float("-inf")], None
            )

            dividend_periods = engineered_data[engineered_data["dividend"] > 0].copy()
            dividend_periods["date"] = pd.to_datetime(dividend_periods["date"])

            print(f"dividend_periods {dividend_periods}")

            insights_data = await self._derive_insights_async(dividend_periods)

            if isinstance(insights_data, pd.DataFrame):
                insights_data["date"] = pd.to_datetime(insights_data["date"])
                cutoff_date = dividend_periods["date"].max()

                historical_data = insights_data[
                    insights_data["date"] <= cutoff_date
                ].copy()
                future_data = insights_data[insights_data["date"] > cutoff_date].copy()

                base_path = Path(self.output_path).parent
                csv_path = base_path / f"{symbol}_insights.csv"
                chart_path = base_path / f"{symbol}_insights_chart.png"

                # Save to CSV with clear separation between historical and predicted data
                insights_df = pd.concat(
                    [
                        historical_data,
                        pd.DataFrame({"date": [None]}),  # Add separator row
                        future_data.assign(
                            data_type="predicted"
                        ),
                    ]
                )

                # Ensure all dates are formatted consistently
                insights_df["date"] = pd.to_datetime(insights_df["date"]).dt.strftime(
                    "%Y-%m-%d"
                )

                insights_df.to_csv(csv_path, index=False, float_format="%.10g")
                print(f"Insights saved to CSV: {csv_path}")

                # Generate enhanced visualization
                self._generate_insights_chart(
                    historical_data, future_data, symbol, chart_path
                )
                print(f"Chart saved to: {chart_path}")

                return insights_data.to_dict("records"), str(csv_path), str(chart_path)
            else:
                raise ValueError("Insights generation did not return a valid DataFrame")

        except Exception as e:
            print(f"ERROR in generate_insights: {str(e)}")
            raise

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
    Generate an enhanced visualization showing both historical and predicted dividend values.
    """
        plt.switch_backend("Agg")
    
    # Create figure with single plot (removing metrics dashboard for clarity)
        fig, ax = plt.subplots(figsize=(15, 8))
    
    # Ensure dates are datetime
        historical_data["date"] = pd.to_datetime(historical_data["date"])
        future_data["date"] = pd.to_datetime(future_data["date"])
    
    # Plot historical dividends
        ax.plot(
        historical_data["date"],
        historical_data["dividend"],
        label="Historical Dividends",
        color="blue",
        marker="o",
        markersize=6,
        linewidth=2,
        )
    
    # Plot predicted dividends
        ax.plot(
        future_data["date"],
        future_data["dividend"],
        label="Projected Dividends",
        color="red",
        linestyle="--",
        marker="s",
        markersize=6,
        linewidth=2,
        )
    
    # Enhance the plot
        ax.set_title(f"Dividend History and Projections - {symbol.upper()}", fontsize=14, pad=20)
        ax.set_ylabel("Dividend Amount ($)", fontsize=12)
        ax.set_xlabel("Date", fontsize=12)
    
    # Improve grid
        ax.grid(True, linestyle='--', alpha=0.7)
    
    # Format x-axis
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
        ax.xaxis.set_major_locator(mdates.MonthLocator(interval=6))
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha="right")
    
    # Add legend with better positioning
        ax.legend(loc='upper left', frameon=True, framealpha=0.9)
    
    # Adjust layout to prevent label cutoff
        plt.tight_layout()
    
    # Save with high DPI
        plt.savefig(chart_path, dpi=300, bbox_inches="tight")
        plt.close()


def handle_shutdown(signum, frame):
    print("\nShutting down gracefully...")
    # Perform any cleanup here
    exit(0)


signal.signal(signal.SIGINT, handle_shutdown)
