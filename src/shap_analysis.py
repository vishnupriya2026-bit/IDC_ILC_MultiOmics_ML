import sys
from pathlib import Path
import json

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
import numpy as np

def run_explainability(
    model, 
    X: pd.DataFrame, 
    run_dir: Path
) -> dict:
    """
    Computes feature importance values and saves explainability metrics into shap/ directory.
    """
    shap_dir = run_dir / "shap"
    shap_dir.mkdir(parents=True, exist_ok=True)

    top_features = []

    # Check for tree-based feature importances
    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
        feature_imp_df = pd.DataFrame({
            "feature": X.columns,
            "importance": importances
        }).sort_values(by="importance", ascending=False)

        top_features = feature_imp_df.head(10).to_dict(orient="records")

    # Check for linear coefficients
    elif hasattr(model, "coef_"):
        coefs = np.abs(model.coef_[0])
        feature_imp_df = pd.DataFrame({
            "feature": X.columns,
            "importance": coefs
        }).sort_values(by="importance", ascending=False)

        top_features = feature_imp_df.head(10).to_dict(orient="records")

    explainability_summary = {
        "top_features": top_features
    }

    with open(shap_dir / "feature_importance.json", "w") as f:
        json.dump(explainability_summary, f, indent=4)

    return explainability_summary