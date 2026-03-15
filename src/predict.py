import pickle
import numpy as np
import json
from mapie.regression import MapieRegressor
from src.config import MODEL_PATH, MAPIE_PATH, FEATURES_PATH, ALPHA, RISK_TIERS, OUTPUT_DIR

def train_mapie(lgbm_model, X_train, y_train):
    """Wrap LightGBM in MAPIE for conformal prediction."""
    print("Fitting MAPIE conformal wrapper...")
    mapie = MapieRegressor(lgbm_model, method="plus", cv=5)
    mapie.fit(X_train, y_train)

    with open(MAPIE_PATH, "wb") as f:
        pickle.dump(mapie, f)
    print(f"MAPIE model saved: {MAPIE_PATH}")
    return mapie


def load_models():
    """Load both models and feature list."""
    with open(MODEL_PATH, "rb") as f:
        lgbm_model = pickle.load(f)
    with open(MAPIE_PATH, "rb") as f:
        mapie_model = pickle.load(f)
    with open(FEATURES_PATH, "r") as f:
        feature_cols = json.load(f)
    return lgbm_model, mapie_model, feature_cols


def predict_with_uncertainty(mapie_model, X, alpha=ALPHA):
    """
    Returns point estimate and confidence intervals in USD millions.
    """
    y_pred_log, y_pis = mapie_model.predict(X, alpha=alpha)

    lower_log = y_pis[:, 0, 0]
    upper_log = y_pis[:, 1, 0]

    # Back-transform from log scale
    point     = np.expm1(y_pred_log)
    lower_usd = np.expm1(lower_log)
    upper_usd = np.expm1(upper_log)

    return point, lower_usd, upper_usd


def get_risk_tier(damage_million_usd):
    """Classify predicted loss into risk tier."""
    for tier, (low, high) in RISK_TIERS.items():
        if low <= damage_million_usd < high:
            return tier
    return "Extreme"