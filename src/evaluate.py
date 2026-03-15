import numpy as np
import json
import matplotlib.pyplot as plt
import shap
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error
from src.config import OUTPUT_DIR

def evaluate_model(model, X_test, y_test):
    """
    Compute RMSE, MAPE, and R2 on test set.
    Returns dict of metrics.
    """
    y_pred_log = model.predict(X_test)

    # Metrics on log scale
    rmse_log = mean_squared_error(y_test, y_pred_log) ** 0.5

    # Back-transform for interpretable metrics
    y_test_usd  = np.expm1(y_test)
    y_pred_usd  = np.expm1(y_pred_log)
    mape        = mean_absolute_percentage_error(y_test_usd, y_pred_usd) * 100

    # R2 on log scale
    ss_res = np.sum((y_test - y_pred_log) ** 2)
    ss_tot = np.sum((y_test - y_test.mean()) ** 2)
    r2     = 1 - ss_res / ss_tot

    metrics = {
        "rmse_log_scale"    : round(rmse_log, 4),
        "mape_pct"          : round(mape, 2),
        "r2_log_scale"      : round(r2, 4),
        "n_test_samples"    : len(y_test),
    }

    print("\n=== MODEL EVALUATION ===")
    for k, v in metrics.items():
        print(f"  {k:25s}: {v}")

    # Save metrics
    with open(OUTPUT_DIR / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    return metrics, y_pred_log


def plot_shap(model, X_test, feature_cols):
    """Generate and save SHAP summary plot."""
    print("\nGenerating SHAP values...")
    explainer   = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test)

    plt.figure(figsize=(10, 8))
    shap.summary_plot(
        shap_values, X_test,
        feature_names=feature_cols,
        show=False,
        plot_size=None
    )
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "shap_summary.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"SHAP plot saved: {OUTPUT_DIR / 'shap_summary.png'}")
    return shap_values


def evaluate_coverage(y_pis, y_test):
    """
    Evaluate conformal prediction interval coverage.
    y_pis shape: (n, 2, 1) from MAPIE
    """
    lower = y_pis[:, 0, 0]
    upper = y_pis[:, 1, 0]
    coverage = np.mean((y_test >= lower) & (y_test <= upper))
    avg_width = np.mean(upper - lower)

    print(f"\n=== CONFORMAL PREDICTION ===")
    print(f"  Empirical coverage : {coverage*100:.2f}%  (target: 90%)")
    print(f"  Avg interval width : {avg_width:.4f} (log scale)")

    return {
        "coverage_pct"      : round(coverage * 100, 2),
        "avg_interval_width": round(avg_width, 4)
    }