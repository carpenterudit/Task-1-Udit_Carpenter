from pathlib import Path
import yaml
class FeatureRegistry:
    def __init__(self,path): self.path=Path(path)
    def save(self, definitions):
        self.path.parent.mkdir(parents=True,exist_ok=True)
        payload={"version":"1.0","entities":["customer_id"],"features":definitions,"offline_store":{"path":"data/processed/final_dataset.csv"},"online_store":{"mode":"local-demonstration","note":"Architecture is compatible with Feast; no fake online API is created."},"feast_compatibility":True}
        self.path.write_text(yaml.safe_dump(payload,sort_keys=False),encoding="utf-8")
    def load(self): return yaml.safe_load(self.path.read_text(encoding="utf-8")) if self.path.exists() else {}
