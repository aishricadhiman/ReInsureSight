import pandas as pd
import numpy as np
import json
from sklearn.preprocessing import LabelEncoder
from src.config import (
    TRAIN_FILE, TARGET_RAW, TARGET_LOG,
    CAT_COLS, TRAIN_CUTOFF, FEATURES_PATH, FEATURE_COLS
)

def load_and_preprocess():
    """
    Load flood_model_final.csv, engineer features,
    encode categoricals, return train/val/test splits.
    """
    df = pd.read_csv(TRAIN_FILE)
    print(f"Loaded: {df.shape}")

    # ── Basic cleaning ────────────────────────────────────────────────────────
    df = df[df[TARGET_RAW].notna()].copy()
    df = df[df[TARGET_RAW] > 0].copy()
    df = df.sort_values("year").reset_index(drop=True)

    # ── Log transform target ──────────────────────────────────────────────────
    df[TARGET_LOG] = np.log1p(df[TARGET_RAW])

    # ── Fill remaining nulls ──────────────────────────────────────────────────
    num_cols = ["precip_30d_mm", "precip_7d_mm", "soil_moisture",
                "temp_2m_c", "ndvi_pre", "total_deaths",
                "no_affected", "total_affected", "duration_days"]

    for col in num_cols:
        if col in df.columns:
            df[col] = df.groupby("region")[col].transform(
                lambda x: x.fillna(x.median())
            )
            df[col] = df[col].fillna(df[col].median())

    # ── Encode categoricals ───────────────────────────────────────────────────
    encoders = {}
    for col in CAT_COLS:
        if col in df.columns:
            le = LabelEncoder()
            df[f"{col}_enc"] = le.fit_transform(
                df[col].fillna("Unknown").astype(str)
            )
            encoders[col] = le

    # ── Final feature list ────────────────────────────────────────────────────
    feature_cols = [f for f in FEATURE_COLS if f in df.columns]

    # Save feature list for app to use
    with open(FEATURES_PATH, "w") as f:
        json.dump(feature_cols, f)
    print(f"Features: {len(feature_cols)} cols")

    # ── Temporal splits ───────────────────────────────────────────────────────
    train_df = df[df["year"] <= TRAIN_CUTOFF].copy()
    val_df   = df[(df["year"] > TRAIN_CUTOFF) &
                  (df["year"] <= 2024)].copy()
    test_df  = df[df["year"] >= 2025].copy()

    print(f"Train : {len(train_df)} rows (2000–{TRAIN_CUTOFF})")
    print(f"Val   : {len(val_df)} rows (2023–2024)")
    print(f"Test  : {len(test_df)} rows (2025–2026, preliminary)")

    return train_df, val_df, test_df, feature_cols, encoders, df


def get_X_y(df, feature_cols):
    """Extract feature matrix and log target from dataframe."""
    X = df[feature_cols].copy()
    y = df[TARGET_LOG].copy()
    return X, y