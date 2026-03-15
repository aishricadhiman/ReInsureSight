import os
from pathlib import Path

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR    = Path(__file__).resolve().parent.parent
DATA_DIR    = BASE_DIR / "data"
MODEL_DIR   = BASE_DIR / "models"
OUTPUT_DIR  = BASE_DIR / "outputs"

TRAIN_FILE  = DATA_DIR / "flood_model_final.csv"
FULL_FILE   = DATA_DIR / "flood_full_pregee.csv"

MODEL_PATH  = MODEL_DIR / "lgbm_model.pkl"
MAPIE_PATH  = MODEL_DIR / "mapie_model.pkl"
FEATURES_PATH = MODEL_DIR / "feature_cols.json"

# ── Train / test split year ───────────────────────────────────────────────────
TRAIN_CUTOFF = 2022   # train: 2000-2022 | val: 2023-2024 | test: 2025+

# ── Target ────────────────────────────────────────────────────────────────────
TARGET_RAW = "damage_million_usd"
TARGET_LOG = "log_damage"

# ── Categorical columns ───────────────────────────────────────────────────────
CAT_COLS = ["flood_subtype", "season", "region", "subregion"]

# ── Feature columns (set after preprocessing) ────────────────────────────────
FEATURE_COLS = [
    # Geography
    "latitude", "longitude",
    # Physical hazard (GEE)
    "precip_30d_mm", "precip_7d_mm", "soil_moisture",
    "temp_2m_c", "ndvi_pre",
    # Flood characteristics
    "duration_days",
    # Exposure
    "total_deaths", "no_affected", "total_affected",
    # Vulnerability
    "insurance_penetration", "gdp_per_capita",
    # Temporal
    "year",
    # Encoded categoricals (added during preprocessing)
    "flood_subtype_enc", "season_enc", "region_enc", "subregion_enc",
    # Flag
    "coord_is_centroid",
]

# ── Model hyperparameters (defaults — Optuna will override) ───────────────────
LGBM_PARAMS = {
    "n_estimators"     : 500,
    "learning_rate"    : 0.05,
    "num_leaves"       : 31,
    "min_child_samples": 20,
    "subsample"        : 0.8,
    "colsample_bytree" : 0.8,
    "random_state"     : 42,
    "n_jobs"           : -1,
}

# ── Conformal prediction coverage ────────────────────────────────────────────
ALPHA = 0.10   # 90% coverage

# ── Risk tiers ────────────────────────────────────────────────────────────────
RISK_TIERS = {
    "Low"     : (0,    50),
    "Medium"  : (50,   500),
    "High"    : (500,  5000),
    "Extreme" : (5000, float("inf"))
}