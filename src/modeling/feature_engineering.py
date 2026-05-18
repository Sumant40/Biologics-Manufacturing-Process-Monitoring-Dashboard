"""
feature_engineering.py
-----------------------
Extracts batch-level features from the first N hours of a run
(default 48h) to use as inputs for the yield prediction model.

For each process parameter we compute:
  - mean       : average level over the early window
  - std        : variability / consistency
  - slope      : linear trend (OLS) — captures drift
  - min / max  : excursion bounds
  - auc        : trapezoidal area under curve (exposure metric)

These 6 statistics × 9 parameters = 54 features per batch.

Usage
-----
    from src.modeling.feature_engineering import extract_early_features

    df_features = extract_early_features(df_measurements, cutoff_h=48)
"""

import numpy as np
import pandas as pd
import yaml
from pathlib import Path
import sqlalchemy as sa

CONFIG_PATH = Path(__file__).parents[2] / "config" / "params.yaml"


def _linear_slope(t: np.ndarray, y: np.ndarray) -> float:
    """OLS slope of y ~ t (no intercept needed for trend detection)."""
    if len(t) < 2:
        return 0.0
    t_c = t - t.mean()
    denom = (t_c**2).sum()
    if denom == 0:
        return 0.0
    return float((t_c * y).sum() / denom)


def _extract_features_one_batch(
    group: pd.DataFrame,
    params: list[str],
    cutoff_h: float,
) -> dict:
    """Compute summary statistics for one batch within the early window."""
    early = group[group["time_h"] <= cutoff_h].copy()

    features: dict = {}
    for param in params:
        y = early[param].dropna().values
        t = early.loc[early[param].notna(), "time_h"].values

        if len(y) == 0:
            for stat in ["mean", "std", "slope", "min", "max", "auc"]:
                features[f"{param}__{stat}"] = np.nan
            continue

        features[f"{param}__mean"] = float(np.mean(y))
        features[f"{param}__std"] = float(np.std(y, ddof=1)) if len(y) > 1 else 0.0
        features[f"{param}__slope"] = _linear_slope(t, y)
        features[f"{param}__min"] = float(np.min(y))
        features[f"{param}__max"] = float(np.max(y))
        features[f"{param}__auc"] = float(np.trapezoid(y, t)) if len(t) > 1 else 0.0

    return features


def extract_early_features(
    df: pd.DataFrame,
    cutoff_h: float = 48,
    params: list[str] | None = None,
) -> pd.DataFrame:
    """
    Extract early-run features from time-series measurement DataFrame.

    Parameters
    ----------
    df       : DataFrame with columns [batch_id, time_h, *params]
    cutoff_h : Hour cutoff for 'early run' window (default 48h)
    params   : Process parameters to summarise (excludes titer_mg_L)

    Returns
    -------
    DataFrame with one row per batch and feature columns.
    """
    config = yaml.safe_load(open(CONFIG_PATH))
    if params is None:
        params = [p for p in config["process_parameters"] if p != "titer_mg_L"]

    records = []
    for batch_id, group in df.groupby("batch_id"):
        feat = _extract_features_one_batch(group, params, cutoff_h)
        feat["batch_id"] = batch_id
        records.append(feat)

    df_feat = pd.DataFrame(records)

    # Reorder: batch_id first
    cols = ["batch_id"] + [c for c in df_feat.columns if c != "batch_id"]
    return df_feat[cols]


# ------------------------------------------------------------------
# CLI: extract features and save to processed/
# ------------------------------------------------------------------

if __name__ == "__main__":
    config = yaml.safe_load(open(CONFIG_PATH))
    db_path = Path(__file__).parents[2] / config["database"]["path"]
    engine = sa.create_engine(f"sqlite:///{db_path}")

    cutoff = config["modeling"]["early_run_cutoff_hours"]
    params = [p for p in config["process_parameters"] if p != "titer_mg_L"]

    df_meas = pd.read_sql("SELECT * FROM measurements", engine)
    df_batches = pd.read_sql("SELECT batch_id, batch_type, final_titer_mg_L FROM batches", engine)

    df_feat = extract_early_features(df_meas, cutoff_h=cutoff, params=params)

    # Merge yield label
    df_feat = df_feat.merge(df_batches, on="batch_id")

    out_path = Path(__file__).parents[2] / "data" / "processed" / "early_features.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df_feat.to_csv(out_path, index=False)

    print(f"Feature matrix: {df_feat.shape[0]} batches × {df_feat.shape[1]} columns")
    print(f"Saved to {out_path}")
    print(df_feat.describe().T[["mean", "std", "min", "max"]].head(15))
