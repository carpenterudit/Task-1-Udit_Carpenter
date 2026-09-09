from pathlib import Path
import pandas as pd

class DatasetLoader:
    def __init__(self, input_path: str | Path, target_column: str | None = None):
        self.input_path = Path(input_path)
        self.target_column = target_column

    def load(self) -> tuple[pd.DataFrame, dict]:
        if not self.input_path.exists():
            raise FileNotFoundError(f"CSV file not found: {self.input_path}")
        if self.input_path.suffix.lower() != ".csv":
            raise ValueError("Only CSV input files are supported.")
        df = pd.read_csv(self.input_path)
        if df.empty:
            raise ValueError("Dataset is empty.")
        timestamp_candidates = [c for c in df.columns if "time" in c.lower() or "date" in c.lower()]
        metadata = {
            "rows": len(df),
            "columns": len(df.columns),
            "numeric_columns": df.select_dtypes(include="number").columns.tolist(),
            "categorical_columns": df.select_dtypes(include=["object", "category", "bool"]).columns.tolist(),
            "timestamp_candidates": timestamp_candidates,
            "target_detected": self.target_column if self.target_column in df.columns else None,
            "duplicates": int(df.duplicated().sum()),
        }
        return df, metadata
