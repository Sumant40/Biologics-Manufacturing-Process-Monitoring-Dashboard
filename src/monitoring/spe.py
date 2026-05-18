"""
spe.py
------
Squared Prediction Error (SPE) monitor — also called the Q statistic.

SPE captures process variation not represented in the retained PCs.
A high SPE indicates the batch is behaving in a structurally new way
not seen in reference data (e.g. sensor fault, new failure mode).

Together T² + SPE give complete MSPC coverage:
  T²  → deviations *within* the PCA model subspace
  SPE → deviations *outside* the PCA model subspace

UCL derivation
--------------
Jackson & Mudholkar (1979) approximation:

    SPE_UCL = θ₁ [c_α √(2θ₂h₀²/θ₁) + 1 + θ₂h₀(h₀-1)/θ₁²]^(1/h₀)

where θ₁, θ₂, θ₃ are functions of residual eigenvalues and h₀ = 1 - 2θ₁θ₃/3θ₂².

Usage
-----
    from src.monitoring.spe import SPEMonitor

    monitor = SPEMonitor(alpha=0.05)
    monitor.fit(spe_ref)
    flags = monitor.flag(spe_new)
"""

import numpy as np
import pandas as pd
import yaml
import joblib
from pathlib import Path
from scipy.stats import norm

CONFIG_PATH = Path(__file__).parents[2] / "config" / "params.yaml"
MODEL_DIR = Path(__file__).parents[2] / "data" / "processed"


class SPEMonitor:
    """
    Computes UCL for SPE using the Jackson-Mudholkar approximation.

    Parameters
    ----------
    alpha : float
        Significance level (e.g. 0.05 → 95% UCL).
    """

    def __init__(self, alpha: float = 0.05):
        self.alpha = alpha
        self.ucl_: float | None = None
        self._theta: tuple | None = None

    # ------------------------------------------------------------------
    # Fit on reference SPE values
    # ------------------------------------------------------------------

    def fit(self, spe_ref: np.ndarray, residual_eigenvalues: np.ndarray) -> "SPEMonitor":
        """
        Compute SPE UCL using Jackson-Mudholkar approximation.

        Parameters
        ----------
        spe_ref              : SPE values from reference batch time-points
        residual_eigenvalues : eigenvalues of the *residual* (discarded) PCs
                               (available from pca.explained_variance_[n_components:])
        """
        lam = residual_eigenvalues  # residual eigenvalues

        theta1 = np.sum(lam)
        theta2 = np.sum(lam**2)
        theta3 = np.sum(lam**3)

        h0 = 1 - (2 * theta1 * theta3) / (3 * theta2**2)
        c_alpha = norm.ppf(1 - self.alpha)

        if h0 <= 0:
            # Fallback: percentile method
            self.ucl_ = float(np.percentile(spe_ref, (1 - self.alpha) * 100))
        else:
            term = (c_alpha * np.sqrt(2 * theta2 * h0**2) / theta1) + 1 + (theta2 * h0 * (h0 - 1)) / (theta1**2)
            self.ucl_ = float(theta1 * term ** (1 / h0))

        self._theta = (theta1, theta2, theta3)
        print(f"SPE UCL ({(1-self.alpha)*100:.0f}%) = {self.ucl_:.4f}")
        return self

    # ------------------------------------------------------------------
    # Monitoring
    # ------------------------------------------------------------------

    def flag(self, spe: np.ndarray) -> np.ndarray:
        """Returns True for observations exceeding the UCL."""
        self._check_fitted()
        return spe > self.ucl_

    def summary(self, spe: np.ndarray, batch_ids: list[str]) -> pd.DataFrame:
        """Per-batch SPE summary (max, mean, OOC flag)."""
        self._check_fitted()
        df = pd.DataFrame({"batch_id": batch_ids, "spe": spe})
        agg = (
            df.groupby("batch_id")["spe"]
            .agg(["max", "mean"])
            .rename(columns={"max": "spe_max", "mean": "spe_mean"})
            .reset_index()
        )
        agg["ucl"] = self.ucl_
        agg["ooc"] = agg["spe_max"] > self.ucl_
        return agg.sort_values("spe_max", ascending=False)

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, name: str = "spe_monitor") -> Path:
        MODEL_DIR.mkdir(parents=True, exist_ok=True)
        path = MODEL_DIR / f"{name}.joblib"
        joblib.dump(self, path)
        print(f"SPE monitor saved → {path}")
        return path

    @classmethod
    def load(cls, name: str = "spe_monitor") -> "SPEMonitor":
        return joblib.load(MODEL_DIR / f"{name}.joblib")

    def _check_fitted(self) -> None:
        if self.ucl_ is None:
            raise RuntimeError("Call .fit() before flagging.")
