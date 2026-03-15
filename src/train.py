import lightgbm as lgb
import optuna
import pickle
import numpy as np
from sklearn.metrics import mean_squared_error
from src.config import MODEL_PATH, LGBM_PARAMS

optuna.logging.set_verbosity(optuna.logging.WARNING)

def train_lgbm(X_train, y_train, X_val, y_val, n_trials=50):
    """
    Train LightGBM with Optuna hyperparameter tuning.
    Returns best model.
    """
    def objective(trial):
        params = {
            "n_estimators"     : trial.suggest_int("n_estimators", 200, 1000),
            "learning_rate"    : trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
            "num_leaves"       : trial.suggest_int("num_leaves", 20, 100),
            "min_child_samples": trial.suggest_int("min_child_samples", 10, 50),
            "subsample"        : trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree" : trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "reg_alpha"        : trial.suggest_float("reg_alpha", 0.0, 1.0),
            "reg_lambda"       : trial.suggest_float("reg_lambda", 0.0, 1.0),
            "random_state"     : 42,
            "n_jobs"           : -1,
            "verbose"          : -1,
        }
        model = lgb.LGBMRegressor(**params)
        model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            callbacks=[lgb.early_stopping(50, verbose=False),
                       lgb.log_evaluation(period=-1)]
        )
        preds = model.predict(X_val)
        return mean_squared_error(y_val, preds) ** 0.5

    print(f"Running Optuna ({n_trials} trials)...")
    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=n_trials)

    print(f"Best RMSE (log scale): {study.best_value:.4f}")
    print(f"Best params: {study.best_params}")

    # Retrain on best params
    best_params = {**study.best_params, "random_state": 42,
                   "n_jobs": -1, "verbose": -1}
    best_model = lgb.LGBMRegressor(**best_params)
    best_model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        callbacks=[lgb.early_stopping(50, verbose=False),
                   lgb.log_evaluation(period=-1)]
    )

    # Save model
    with open(MODEL_PATH, "wb") as f:
        pickle.dump(best_model, f)
    print(f"Model saved: {MODEL_PATH}")

    return best_model, study