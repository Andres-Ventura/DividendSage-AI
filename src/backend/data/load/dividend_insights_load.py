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

    def load_data_and_train_model(self, symbol: str, uploaded_file: Optional[str] = None):
        """
        Fetch, transform, and load data into a CSV file, then train the model.
        """
        # Fetch data from the combined provider
        raw_data = self.data_provider.fetch_combined_data(symbol, uploaded_file)

        # Preprocess and transform the data
        unified_data = self.transformer.preprocess_data(raw_data)
        engineered_data = self.transformer.feature_engineering(unified_data)

        # Save data to a CSV file
        engineered_data.to_csv(self.output_path, index=False)
        print(f"Data successfully saved to '{self.output_path}'.")

        # Train the model
        print("Starting model training...")
        self.transformer.train_model(engineered_data)
        print("Model training completed.")

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