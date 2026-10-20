import json
from pathlib import Path
from datetime import datetime

class RunSession:
    def __init__(self, disease: str, base_dir: str = "results"):
        self.timestamp = datetime.now()
        # Generate ID: e.g., OPA-2026-89412
        self.run_id = f"OPA-{self.timestamp.strftime('%Y')}-{self.timestamp.strftime('%S%f')[:5]}"
        self.disease = disease
        self.root_dir = Path(base_dir) / self.run_id
        self._init_workspace()

    def _init_workspace(self):
        """Creates the isolated folder tree for this analysis run."""
        subdirs = [
            "input", "qc", "processed", "models", 
            "shap", "biomarkers", "therapeutics", "figures", "reports"
        ]
        for folder in subdirs:
            (self.root_dir / folder).mkdir(parents=True, exist_ok=True)
            
        # Log run metadata
        metadata = {
            "run_id": self.run_id,
            "disease": self.disease,
            "created_at": self.timestamp.isoformat(),
            "status": "INITIALIZED"
        }
        with open(self.root_dir / "run_metadata.json", "w") as f:
            json.dump(metadata, f, indent=4)