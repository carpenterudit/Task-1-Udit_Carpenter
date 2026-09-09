from pathlib import Path
import pandas as pd
import yaml
class LocalFeatureStore:
    """Small offline feature-store demonstration with a Feast-compatible conceptual contract."""
    def __init__(self, registry_path="feature_store/feature_definitions.yaml"):
        self.registry_path=Path(registry_path)
    def definitions(self): return yaml.safe_load(self.registry_path.read_text()) if self.registry_path.exists() else {}
    def get_offline_features(self, dataset_path="data/processed/final_dataset.csv", columns=None):
        df=pd.read_csv(dataset_path); return df[columns] if columns else df
