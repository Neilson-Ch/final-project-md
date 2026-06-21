from pathlib import Path
import pandas as pd

class DataIngestion:
    """Handles creating ingestion directories, loading, and validating raw data."""
    
    def __init__(self, input_path: str | Path, output_dir: str | Path):
        self.input_file = Path(input_path)
        self.output_dir = Path(output_dir)
        self.output_file = self.output_dir / "data_A.csv"

    def run(self) -> Path:
        print("--- Step 1: Data Ingestion ---")
        # Ensure output folder exists
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Read raw data
        df = pd.read_csv(self.input_file)

        # Basic validation
        assert not df.empty, "Dataset is empty"

        # Save ingested data
        df.to_csv(self.output_file, index=False)
        print(f"✅ Data ingested from {self.input_file} → {self.output_file}")
        
        return self.output_file