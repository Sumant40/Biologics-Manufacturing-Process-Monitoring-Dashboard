"""
hotelling_t2.py
---------------
Computes Hotelling's T² statistic for each observation/batch and derives
the upper control limit (UCL) at a given significance level (α).

Theory
------
For a PCA model with n reference observations and A retained PCs:

    T²_i = t_i · Λ⁻¹ · t_i'

where t_i is the score vector (1 × A) and Λ = diag(eigenvalues).

The Phase-1 UCL follows an F-distribution:

    UCL = [A(n²-1) / n(n-A)] · F_{A, n-A, α}

Phase-2 UCL for new observations:

    UCL = [A(n+1)(n-1) / n(n-A)] · F_{A, n-A, α}

Usage
-----
    from src.monitoring.hotelling_t2 import HotellingT2

    monitor = HotellingT2(n_components=5, alpha=0.05)
    monitor.fit(T_ref)           # T_ref: scores from reference batches
    t2_new = monitor.score(T_new)
    flags = monitor.flag(T_new)  # True = out-of-control
"""

import numpy as np
import pandas as pd
import yaml
import sqlalchemy as sa
import joblib
from pathlib import Path
from scipy.stats import f as f_dist

CONFIG_PATH = Path(__file__).parents[2] / "config" / "params.yaml"
MODEL_DIR = Path(__file__).parents[2] / "data" / "processed"


class HotellingT2:
    """
    Hotelling's T² monitor for MSPC Phase-1 and Phase-2 control.

    Parameters
    ----------
    n_components : int
        Number of PC scores used (A).
    alpha : float
        Significance level for UCL (e.g. 0.05 → 95% limit).
    """

    def __init__(self, n_components: int = 5, alpha: float = 0.05):
        self.A = n_components
        self.alpha = alpha
        self.eigenvalues_: np.ndarray | None = None
        self.n_ref_: int | None = None
        self.ucl_phase1_: float | None = None
        self.ucl_phase2_: float | None = None

    # ------------------------------------------------------------------
    # Fit (Phase 1)
    # ------------------------------------------------------------------

    def fit(self, T_ref: np.ndarray) -> "HotellingT2":
        """
        Estimate eigenvalues and compute Phase-1 UCL from reference scores.

        Parameters
        ----------
        T_ref : array of shape (n_ref, A) — PC scores for reference batches
        """
        n, A = T_ref.shape
        assert A == self.A, f"Expected {self.A} PCs, got {A}"

        self.eigenvalues_ = np.var(T_ref, axis=0, ddof=1)  # variance of each PC
        self.n_ref_ = n

        # Phase-1 UCL (Jackson & Mudholkar, 1979)
        f_crit = f_dist.ppf(1 - self.alpha, A, n - A)
        self.ucl_phase1_ = (A * (n**2 - 1)) / (n * (n - A)) * f_crit

        # Phase-2 UCL (for new observations)
        self.ucl_phase2_ = (A * (n + 1) * (n - 1)) / (n * (n - A)) * f_crit

        print(
            f"T² monitor fitted | n_ref={n} | A={A} | "
            f"UCL (P1)={self.ucl_phase1_:.2f} | UCL (P2)={self.ucl_phase2_:.2f}"
        )
        return self

    # ------------------------------------------------------------------
    # Scoring
    # ------------------------------------------------------------------

    def score(self, T: np.ndarray) -> np.ndarray:
        """
        Compute T² values for each row of T.

        Parameters
        ----------
        T : array of shape (n_obs, A)

        Returns
        -------
        t2 : array of shape (n_obs,)
        """
        self._check_fitted()
        # Mahalanobis-style: sum of (t_ia² / lambda_a) over PCs
        t2 = np.sum(T**2 / self.eigenvalues_, axis=1)
        return t2

    def flag(self, T: np.ndarray, phase: int = 2) -> np.ndarray:
        """
        Returns boolean mask: True = observation exceeds UCL (out-of-control).

        Parameters
        ----------
        T     : PC scores
        phase : 1 (reference monitoring) or 2 (new batch monitoring)
        """
        t2 = self.score(T)
        ucl = self.ucl_phase1_ if phase == 1 else self.ucl_phase2_
        return t2 > ucl

    def summary(self, T: np.ndarray, batch_ids: list[str], phase: int = 2) -> pd.DataFrame:
        """
        Aggregate T² per batch (max over all time-points) with OOC flag.

        Parameters
        ----------
        T         : PC scores, one row per time-point
        batch_ids : list of batch_id per row (same length as T)
        phase     : 1 or 2 for UCL selection
        """
        t2 = self.score(T)
        ucl = self.ucl_phase1_ if phase == 1 else self.ucl_phase2_

        df = pd.DataFrame({"batch_id": batch_ids, "t2": t2})
        agg = (
            df.groupby("batch_id")["t2"]
            .agg(["max", "mean"])
            .rename(columns={"max": "t2_max", "mean": "t2_mean"})
            .reset_index()
        )
        agg["ucl"] = ucl
        agg["ooc"] = agg["t2_max"] > ucl
        return agg.sort_values("t2_max", ascending=False)

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, name: str = "hotelling_t2") -> Path:
        MODEL_DIR.mkdir(parents=True, exist_ok=True)
        path = MODEL_DIR / f"{name}.joblib"
        joblib.dump(self, path)
        print(f"T² monitor saved → {path}")
        return path

    @classmethod
    def load(cls, name: str = "hotelling_t2") -> "HotellingT2":
        path = MODEL_DIR / f"{name}.joblib"
        return joblib.load(path)

    def _check_fitted(self) -> None:
        if self.eigenvalues_ is None:
            raise RuntimeError("Call .fit() before scoring.")


# ------------------------------------------------------------------
# CLI: fit on reference scores and report batch-level OOC summary
# ------------------------------------------------------------------

if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(Path(__file__).parents[2] / "src"))
    from monitoring.pca_engine import PCAEngine

    config = yaml.safe_load(open(CONFIG_PATH))
    db_path = Path(__file__).parents[2] / config["database"]["path"]
    engine_db = sa.create_engine(f"sqlite:///{db_path}")

    params = [p for p in config["process_parameters"] if p != "titer_mg_L"]

    query = f"""
        SELECT b.batch_id, b.batch_type, m.{', m.'.join(params)}
        FROM measurements m
        JOIN batches b ON m.batch_id = b.batch_id
    """
    df_all = pd.read_sql(query, engine_db).dropna()

    # Load fitted PCA engine
    pca_engine = PCAEngine.load()

    X_all = df_all[params].values
    T_all, spe_all = pca_engine.transform(X_all)

    # Reference scores (normal batches only)
    mask_ref = df_all["batch_type"] == "normal"
    T_ref = T_all[mask_ref]

    monitor = HotellingT2(
        n_components=config["mspc"]["n_components"],
        alpha=config["mspc"]["alpha"],
    )
    monitor.fit(T_ref)

    # Batch-level OOC summary
    summary = monitor.summary(T_all, df_all["batch_id"].tolist())
    ooc_batches = summary[summary["ooc"]]
    print(f"\nOut-of-control batches detected: {len(ooc_batches)}")
    print(summary.head(20).to_string(index=False))

    monitor.save()
