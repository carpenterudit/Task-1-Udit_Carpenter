from pathlib import Path
import json, time
class ReportManager:
    def __init__(self, root: Path): self.root=Path(root); self.root.mkdir(parents=True,exist_ok=True)
    def save_json(self,name,data): (self.root/name).write_text(json.dumps(data,indent=2,default=str),encoding="utf-8")
