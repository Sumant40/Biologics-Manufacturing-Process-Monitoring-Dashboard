"""
evaluate_model.py
-----------------
Post-training model evaluation using SHAP values for explainability.

Outputs
-------
- SHAP summary plot saved to docs/shap_summary.png
- Residual analysis: predicted vs actual, residual distribution
- Per-batch-type RMSE breakdown

Usage
-----
    python src/modeling/evaluate_model.py
"""

import numpy as np
import pandas as pd
import yaml
import joblib
import shap
import matplotlib
matplotlib.use("Agg")  # non-interactive backend for server/CI use
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.metrics import mean_squared_error, r2_score

CONFIG_PATH = Path(__file__).parents[2] / "config" / "params.yaml"
MODEL_DIR = Path(__file__).parents[2] / "data" / "processed"
DOCS_DIR = Path(__file__).parents[2] / "docs"
FEATURES_PATH = MODEL_DIR / "early_features.csv"


def load_artifacts():
    model = joblib.load(MODEL_DIR / "yield_model.joblib")
    importances = pd.read_csv(MODEL_DIR / "feature_importances.csv")
    return model, importances


def shap_analysis(model, X: pd.DataFrame, feature_names: list[str]) -> None:
    """Compute SHAP values and save summary plot."""
    xgb_model = model.named_steps["xgb"]
    scaler = model.named_steps["scaler"]
    X_scaled = scaler.transform(X)

    explainer = shap.TreeExplainer(xgb_model)
    shap_values = explainer.shap_values(X_scaled)

    DOCS_DIR.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(10, 7))
    shap.summary_plot(
        shap_values,
        X_scaled,
        feature_names=feature_names,
        show=False,
        max_display=20,
    )
    plt.tight_layout()
    plt.savefig(DOCS_DIR / "shap_summary.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"SHAP summary plot saved → {DOCS_DIR / 'shap_summary.png'}")

    # SHAP importance table
    shap_importance = pd.DataFrame({
        "feature": feature_names,
        "mean_abs_shap": np.abs(shap_values).mean(axis=0),
    }).sort_values("mean_abs_shap", ascending=False)
    shap_importance.to_csv(DOCS_DIR / "shap_importance.csv", index=False)

    print("\nTop 10 Features by SHAP:")
    print(shap_importance.head(10).to_string(index=False))


def residual_analysis(model, X: pd.DataFrame, y: pd.Series) -> None:
    """Predicted vs actual and residual distribution."""
    y_pred = model.predict(X)
    residuals = y.values - y_pred
    rmse = np.sqrt(mean_squared_error(y, y_pred))
    r2 = r2_score(y, y_pred)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    axes[0].scatter(y_pred, y.values, alpha=0.6, edgecolors="k", linewidths=0.3)
    lims = [min(y_pred.min(), y.min()) * 0.95, max(y_pred.max(), y.max()) * 1.05]
    axes[0].plot(lims, lims, "r--", linewidth=1.5, label="Perfect fit")
    axes[0].set_xlabel("Predicted Titer (mg/L)")
    axes[0].set_ylabel("Actual Titer (mg/L)")
    axes[0].set_title(f"Predicted vs Actual  |  RMSE={rmse:.1f}, R²={r2:.3f}")
    axes[0].legend()

    axes[1].hist(residuals, bins=20, color="steelblue", edgecolor="white", linewidth=0.5)
    axes[1].axvline(0, color="red", linestyle="--")
    axes[1].set_xlabel("Residual (mg/L)")
    axes[1].set_ylabel("Count")
    axes[1].set_title("Residual Distribution")

    plt.tight_layout()
    plt.savefig(DOCS_DIR / "residuals.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Residual plots saved → {DOCS_DIR / 'residuals.png'}")


if __name__ == "__main__":
    config = yaml.safe_load(open(CONFIG_PATH))
    model, importances = load_artifacts()

    df = pd.read_csv(FEATURES_PATH)
    feature_cols = [c for c in df.columns if "__" in c]
    X = df[feature_cols].fillna(df[feature_cols].median())
    y = df["final_titer_mg_L"]

    shap_analysis(model, X, feature_cols)
    residual_analysis(model, X, y)
    print("\nEvaluation complete.")
