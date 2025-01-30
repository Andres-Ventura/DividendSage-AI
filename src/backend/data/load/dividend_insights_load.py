from typing import Optional
import pandas as pd
from backend.data.combined_data_provider import CombinedDataProvider
from backend.data.transform.dividend_insights_transform import DividendInsightsTransform

class DividendInsightsLoader:
    """
    A class to load transformed dividend insights into a CSV file, train the model, and generate insights.
    """

    def __init__(self, output_path):
        self.output_path = output_path
        self.transformer = DividendInsightsTransform()
        self.data_provider = CombinedDataProvider()

    async def load_data_and_train_model(self, symbol: str, uploaded_file: Optional[str] = None):
        """
        Fetch, transform, and load data into a CSV file, then train the model.
        """
        try:
            # Fetch data from the combined provider
            raw_data = await self.data_provider.fetch_combined_data(symbol, uploaded_file)

            # Preprocess and transform the data
            unified_data = await self.transformer.preprocess_data(raw_data)

            if unified_data is None or unified_data.empty:
                raise ValueError("Data preprocessing failed")

            engineered_data = await self.transformer.feature_engineering(unified_data)
            
            if engineered_data is None or engineered_data.empty:
                raise ValueError("Feature engineering failed")

            # Save data to a CSV file
            engineered_data.to_csv(self.output_path, index=False)
            
            print(f"Data successfully saved to '{self.output_path}'.")
            
            print("Starting model training...")
            self.transformer.train_model(engineered_data)
            print("Model training completed.")

        except Exception as e:
            print(f"Error in load_data_and_train_model: {str(e)}")
            raise  # Re-raise the exception for the global handler to catch

    def generate_insights(self, symbol: str, uploaded_file: Optional[str] = None):
        """
        Generate insights from the trained model.
        """
        raw_data = self.data_provider.fetch_combined_data(symbol, uploaded_file)
        unified_data = self.transformer.preprocess_data(raw_data)
        engineered_data = self.transformer.feature_engineering(unified_data)

        # Derive insights after model training
        insights = self.transformer.derive_insights(engineered_data)
        return insights
