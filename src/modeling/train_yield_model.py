"""
train_yield_model.py
--------------------
Trains an XGBoost regressor to predict final batch titer (mg/L)
from early-run (≤48h) process parameter features.

Pipeline
--------
1. Load feature matrix from data/processed/early_features.csv
2. Split into train / test (80/20, stratified by batch_type)
3. 5-fold cross-validated XGBoost training
4. Evaluate on held-out test set (RMSE, MAE, R²)
5. Save model artifact to data/processed/yield_model.joblib
6. Output feature importances (gain-based)

Usage
-----
    python src/modeling/train_yield_model.py
"""

import numpy as np
import pandas as pd
import yaml
import joblib
from pathlib import Path

from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import xgboost as xgb

CONFIG_PATH = Path(__file__).parents[2] / "config" / "params.yaml"
FEATURES_PATH = Path(__file__).parents[2] / "data" / "processed" / "early_features.csv"
MODEL_DIR = Path(__file__).parents[2] / "data" / "processed"


def load_data(path: Path) -> tuple[pd.DataFrame, pd.Series, list[str]]:
    df = pd.read_csv(path)
    feature_cols = [c for c in df.columns if "__" in c]  # pattern: param__stat
    X = df[feature_cols].fillna(df[feature_cols].median())
    y = df["final_titer_mg_L"]
    return X, y, feature_cols


def train(config: dict) -> None:
    xgb_params = config["modeling"]["xgb_params"]
    cv_folds = config["modeling"]["cv_folds"]

    X, y, feature_cols = load_data(FEATURES_PATH)
    print(f"Loaded feature matrix: {X.shape}")

    # Train/test split — preserve batch_type distribution
    df = pd.read_csv(FEATURES_PATH)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        random_state=xgb_params["random_state"],
        stratify=df["batch_type"],
    )
    print(f"Train: {len(X_train)} | Test: {len(X_test)}")

    # Model pipeline
    model = Pipeline([
        ("scaler", StandardScaler()),
        ("xgb", xgb.XGBRegressor(
            n_estimators=xgb_params["n_estimators"],
            max_depth=xgb_params["max_depth"],
            learning_rate=xgb_params["learning_rate"],
            subsample=xgb_params["subsample"],
            colsample_bytree=xgb_params["colsample_bytree"],
            random_state=xgb_params["random_state"],
            verbosity=0,
            objective="reg:squarederror",
        )),
    ])

    # Cross-validation on training set
    kf = KFold(n_splits=cv_folds, shuffle=True, random_state=42)
    cv_rmse = -cross_val_score(
        model, X_train, y_train,
        cv=kf,
        scoring="neg_root_mean_squared_error",
    )
    print(f"\nCV RMSE ({cv_folds}-fold): {cv_rmse.mean():.2f} ± {cv_rmse.std():.2f} mg/L")

    # Final fit on full training data
    model.fit(X_train, y_train)

    # Test set evaluation
    y_pred = model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print(f"\nTest Set Performance")
    print(f"  RMSE : {rmse:.2f} mg/L")
    print(f"  MAE  : {mae:.2f} mg/L")
    print(f"  R²   : {r2:.4f}")

    # Feature importances (XGBoost gain)
    xgb_model = model.named_steps["xgb"]
    importances = pd.DataFrame({
        "feature": feature_cols,
        "importance_gain": xgb_model.feature_importances_,
    }).sort_values("importance_gain", ascending=False)

    print(f"\nTop 10 Features (by gain):")
    print(importances.head(10).to_string(index=False))

    # Save artifacts
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    joblib.dump(model, MODEL_DIR / "yield_model.joblib")
    importances.to_csv(MODEL_DIR / "feature_importances.csv", index=False)

    # Save test predictions for dashboard
    test_results = pd.DataFrame({
        "y_true": y_test.values,
        "y_pred": y_pred,
    })
    test_results.to_csv(MODEL_DIR / "test_predictions.csv", index=False)

    print(f"\nModel saved → {MODEL_DIR / 'yield_model.joblib'}")
    print(f"Feature importances → {MODEL_DIR / 'feature_importances.csv'}")


if __name__ == "__main__":
    config = yaml.safe_load(open(CONFIG_PATH))
    train(config)
