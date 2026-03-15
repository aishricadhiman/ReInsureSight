"""
# ReinsureSight — Geospatial Flood Loss Intelligence for Reinsurance Underwriting
Run the full pipeline end to end.
"""
from src.preprocess     import load_and_preprocess, get_X_y
from src.train          import train_lgbm
from src.evaluate       import evaluate_model, plot_shap, evaluate_coverage
from src.predict        import train_mapie, predict_with_uncertainty
from src.protection_gap import compute_protection_gap, build_protection_gap_map
import pandas as pd
from src.config         import FULL_FILE, OUTPUT_DIR

def main():
    print("=" * 50)
    print("ReinsureSight Pipeline")
    print("=" * 50)

    # Step 1: Preprocess
    print("\n[1/5] Preprocessing...")
    train_df, val_df, test_df, feature_cols, encoders, full_model_df = (
        load_and_preprocess()
    )

    X_train, y_train = get_X_y(train_df, feature_cols)
    X_val,   y_val   = get_X_y(val_df,   feature_cols)
    X_test,  y_test  = get_X_y(test_df,  feature_cols)

    # Step 2: Train
    print("\n[2/5] Training LightGBM...")
    best_model, study = train_lgbm(X_train, y_train, X_val, y_val, n_trials=50)

    # Step 3: Evaluate
    print("\n[3/5] Evaluating...")
    metrics, y_pred_log = evaluate_model(best_model, X_val, y_val)
    plot_shap(best_model, X_val, feature_cols)

    # Step 4: Conformal prediction
    print("\n[4/5] Fitting MAPIE conformal wrapper...")
    mapie_model         = train_mapie(best_model, X_train, y_train)
    y_pred, y_pis       = mapie_model.predict(X_val, alpha=0.10)
    coverage_metrics    = evaluate_coverage(y_pis, y_val)

    # Step 5: Protection gap map
    print("\n[5/5] Building protection gap map...")
    # Replace the full_df block in main.py with this
    full_df = pd.read_csv(FULL_FILE)

# damage column may already exist or need computing
    if "damage_million_usd" not in full_df.columns:
        if "total_damage_adj" in full_df.columns:
            full_df["damage_million_usd"] = full_df["total_damage_adj"] / 1000
        else:
            full_df["damage_million_usd"] = 0

    full_df = full_df.dropna(subset=["latitude", "longitude"]).copy()
    full_df = compute_protection_gap(full_df)
    m, country_agg, top10 = build_protection_gap_map(full_df)

    print("\n" + "=" * 50)
    print("Pipeline complete.")
    print(f"Outputs saved to: {OUTPUT_DIR}")
    print("=" * 50)

if __name__ == "__main__":
    main()