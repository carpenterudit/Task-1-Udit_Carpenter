from pathlib import Path
import pandas as pd

def validate_input_file(path: str | Path) -> None:
    p = Path(path)
    if not p.exists(): raise FileNotFoundError(str(p))
    if p.suffix.lower() != ".csv": raise ValueError("Only CSV files are accepted.")

def basic_integrity(df: pd.DataFrame) -> list[str]:
    errors=[]
    if df.empty: errors.append("Dataset is empty")
    if df.columns.duplicated().any(): errors.append("Duplicate column names detected")
    return errors
