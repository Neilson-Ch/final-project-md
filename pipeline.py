from pathlib import Path
from data_ingestion import DataIngestion
from train import CreditScoreModelTrainer
from evaluation import ModelEvaluator

class PredictionPipeline:
    """The master pipeline orchestrating ingestion, training, evaluation, and approval."""
    
    def __init__(self, raw_data_path: str | Path, f1_threshold: float = 0.7):
        self.base_dir = Path(__file__).parent
        self.raw_data_path = Path(raw_data_path)
        self.ingested_dir = self.base_dir / "ingested"
        self.f1_threshold = f1_threshold
        
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
        accuracy, precision, recall, f1_weighted = (self.evaluator.run(run_id, x_test, y_test))
        
        # 4. Final conditional release assessment
        print("\n--- Deployment Approval Decision ---")
        if f1_weighted >= self.f1_threshold:
            print("🎉 Success: Model metrics pass QA. Approved for deployment!")
        else:
            print(f"❌ Rejected: Model f1 weighted ({f1_weighted:.3f}) falls short of threshold ({self.f1_threshold})")


if __name__ == "__main__":
    # Point directly to your primary relative data source file
    DATA_INPUT = Path(__file__).parent / "data_A.csv"
    
    # Initialize and execute
    pipeline = PredictionPipeline(raw_data_path=DATA_INPUT, f1_threshold=0.7)
    pipeline.execute()
