import numpy as np
from sklearn.metrics import roc_curve

def optimize_youden_threshold(y_true, y_pred_probs):
    """
    Computes Youden's Index J across ROC thresholds and identifies optimal p*.
    """
    fpr, tpr, thresholds = roc_curve(y_true, y_pred_probs)
    j_scores = tpr - fpr
    
    # Find index that maximizes Youden's J
    best_idx = np.argmax(j_scores)
    best_threshold = thresholds[best_idx]
    best_youden = j_scores[best_idx]
    
    # Default cutoff at p = 0.50
    default_idx = np.argmin(np.abs(thresholds - 0.50))
    default_youden = j_scores[default_idx]
    
    print("\n=== Threshold Calibration Analysis ===")
    print(f"Default Cutoff (p = 0.50) Youden Index J: {default_youden:.4f}")
    print(f"Optimal Cutoff Threshold (p*):             {best_threshold:.4f}")
    print(f"Optimized Youden Index (J*):              {best_youden:.4f}")
    print(f"Sensitivity at p*:                        {tpr[best_idx]*100:.2f}%")
    print(f"Specificity at p*:                        {(1 - fpr[best_idx])*100:.2f}%\n")
    
    return best_threshold, best_youden

if __name__ == "__main__":
    # Test simulation
    np.random.seed(42)
    y_true_mock = np.random.binomial(1, 0.3, size=100)
    y_probs_mock = np.random.uniform(0, 1, size=100)
    optimize_youden_threshold(y_true_mock, y_probs_mock)
