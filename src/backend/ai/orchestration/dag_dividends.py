from ai.pipelines.train_pipeline import TrainPipeline
import pandas as pd

class DAGDividends:
    def __init__(self, data_source):
        self.data_source = data_source
        self.pipeline = TrainPipeline()

    def execute(self):
        # Load data from source
        data = self.load_data(self.data_source)

        # Run pipeline
        self.pipeline.run(data)

    def load_data(self, source):
        # Example: Assume source is a CSV file path
        return pd.read_csv(source)