from pathlib import Path
from data_ingestion import DataIngestion
from train import CreditScoreModelTrainer
from evaluation import ModelEvaluator

class PredictionPipeline:
    """The master pipeline orchestrating ingestion, training, evaluation, and approval."""
    
    def __init__(self, raw_data_path: str | Path, accuracy_threshold: float = 0.7):
        self.base_dir = Path(__file__).parent
        self.raw_data_path = Path(raw_data_path)
        self.ingested_dir = self.base_dir / "ingested"
        self.accuracy_threshold = accuracy_threshold
        
        # Core components instantiation
        self.ingestor = DataIngestion(self.raw_data_path, self.ingested_dir)
        self.trainer = CreditScoreModelTrainer()
        self.evaluator = ModelEvaluator()

    def execute(self):
        print("🚀 Executing Credit Score Prediction Pipeline...")
        
        # 1. Handle raw data ingestion safely
        ingested_file_path = self.ingestor.run()
        
        # 2. Extract, transform, split, and train model properties
        run_id, x_test, y_test = self.trainer.run(ingested_file_path)
        
        # 3. Handle telemetry evaluations and metrics verification
        accuracy, precision, recall = self.evaluator.run(run_id, x_test, y_test)
        
        # 4. Final conditional release assessment
        print("\n--- Deployment Approval Decision ---")
        if accuracy >= self.accuracy_threshold:
            print("🎉 Success: Model metrics pass QA. Approved for deployment!")
        else:
            print(f"❌ Rejected: Model accuracy ({accuracy:.3f}) falls short of threshold ({self.accuracy_threshold})")


if __name__ == "__main__":
    # Point directly to your primary relative data source file
    DATA_INPUT = Path(__file__).parent / "data_A.csv"
    
    # Initialize and execute
    pipeline = PredictionPipeline(raw_data_path=DATA_INPUT, accuracy_threshold=0.7)
    pipeline.execute()