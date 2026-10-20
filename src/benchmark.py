import sys
from pathlib import Path
import json

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

def run_benchmarking(
    feature_matrix: pd.DataFrame, 
    target_series: pd.Series, 
    run_dir: Path
) -> dict:
    """
    Evaluates ML algorithms using stratified cross-validation and selects 
    the best-performing model. Saves benchmarking metrics into models/ directory.
    """
    models_dir = run_dir / "models"
    models_dir.mkdir(parents=True, exist_ok=True)
    
    # Simple, robust classifiers
    classifiers = {
        "Random_Forest": RandomForestClassifier(n_estimators=50, random_state=42),
        "Logistic_Regression": LogisticRegression(max_iter=500, random_state=42)
    }

    metrics_results = {}
    best_score = -1.0
    best_model_name = None
    best_model_obj = None

    # Determine CV splits based on sample size
    n_samples = len(target_series)
    cv_splits = min(3, n_samples) if n_samples >= 4 else 2

    cv = StratifiedKFold(n_splits=cv_splits, shuffle=True, random_state=42)

    for name, clf in classifiers.items():
        try:
            scores = cross_validate(
                clf, feature_matrix, target_series, 
                cv=cv, scoring=['accuracy', 'f1_weighted']
            )
            mean_acc = float(scores['test_accuracy'].mean())
            mean_f1 = float(scores['test_f1_weighted'].mean())

            metrics_results[name] = {
                "mean_accuracy": round(mean_acc, 4),
                "mean_f1": round(mean_f1, 4)
            }

            if mean_f1 > best_score:
                best_score = mean_f1
                best_model_name = name
                best_model_obj = clf

        except Exception as e:
            metrics_results[name] = {"error": str(e)}

    # Fit best model on full cohort
    if best_model_obj is not None:
        best_model_obj.fit(feature_matrix, target_series)

    benchmark_summary = {
        "selected_model": best_model_name,
        "best_f1_score": best_score,
        "evaluations": metrics_results
    }

    with open(models_dir / "benchmark_summary.json", "w") as f:
        json.dump(benchmark_summary, f, indent=4)

    return {
        "summary": benchmark_summary,
        "fitted_model": best_model_obj,
        "model_name": best_model_name
    }