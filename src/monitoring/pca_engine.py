"""
pca_engine.py
-------------
Phase-1 PCA fitting on reference (normal) batches.
Phase-2 projection of new batches onto the reference model.

The module computes:
  - Loadings (P matrix)
  - Scores (T matrix)
  - Explained variance per PC
  - Reconstructed data for SPE calculation

Typical usage
-------------
    from src.monitoring.pca_engine import PCAEngine

    engine = PCAEngine(n_components=5)
    engine.fit(X_ref)                     # Phase 1
    scores, spe = engine.transform(X_new) # Phase 2
"""

import numpy as np
import pandas as pd
import yaml
import joblib
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

CONFIG_PATH = Path(__file__).parents[2] / "config" / "params.yaml"
MODEL_DIR = Path(__file__).parents[2] / "data" / "processed"


class PCAEngine:
    """
    Wrapper around sklearn PCA with MSPC-specific utilities.

    Attributes
    ----------
    n_components : int
        Number of PCs to retain.
    scaler : StandardScaler
        Mean-centering and unit-variance scaling (critical before PCA in MSPC).
    pca : sklearn PCA object
    """

    def __init__(self, n_components: int = 5):
        self.n_components = n_components
        self.scaler = StandardScaler()
        self.pca = PCA(n_components=n_components)
        self._fitted = False

    # ------------------------------------------------------------------
    # Fitting (Phase 1 — reference batch population)
    # ------------------------------------------------------------------

    def fit(self, X: np.ndarray) -> "PCAEngine":
        """
        Fit scaler + PCA on reference data X (observations × variables).

        Parameters
        ----------
        X : array of shape (n_samples, n_features)
            Each row is one time-point from one reference batch.
        """
        X_scaled = self.scaler.fit_transform(X)
        self.pca.fit(X_scaled)
        self._fitted = True
        print(
            f"PCA fitted | {self.n_components} PCs | "
            f"Cumulative variance explained: "
            f"{self.pca.explained_variance_ratio_.cumsum()[-1]*100:.1f}%"
        )
        return self

    # ------------------------------------------------------------------
    # Transform (Phase 2 — new / incoming batches)
    # ------------------------------------------------------------------

    def transform(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        Project X into PC space and compute SPE residuals.

        Returns
        -------
        T : PC scores, shape (n_samples, n_components)
        spe : Squared Prediction Error per sample, shape (n_samples,)
        """
        self._check_fitted()
        X_scaled = self.scaler.transform(X)
        T = self.pca.transform(X_scaled)
        X_reconstructed = self.pca.inverse_transform(T)
        residuals = X_scaled - X_reconstructed
        spe = np.sum(residuals**2, axis=1)
        return T, spe

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    def explained_variance_table(self, feature_names: list[str] | None = None) -> pd.DataFrame:
        """Returns a DataFrame of explained variance per PC."""
        self._check_fitted()
        ev = self.pca.explained_variance_ratio_
        return pd.DataFrame(
            {
                "PC": [f"PC{i+1}" for i in range(self.n_components)],
                "Explained Variance (%)": ev * 100,
                "Cumulative (%)": ev.cumsum() * 100,
            }
        )

    def loadings_table(self, feature_names: list[str]) -> pd.DataFrame:
        """Returns loadings (PC × variable) as a tidy DataFrame."""
        self._check_fitted()
        loadings = self.pca.components_  # shape (n_components, n_features)
        return pd.DataFrame(
            loadings,
            index=[f"PC{i+1}" for i in range(self.n_components)],
            columns=feature_names,
        )

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, name: str = "pca_engine") -> Path:
        MODEL_DIR.mkdir(parents=True, exist_ok=True)
        path = MODEL_DIR / f"{name}.joblib"
        joblib.dump(self, path)
        print(f"PCA engine saved → {path}")
        return path

    @classmethod
    def load(cls, name: str = "pca_engine") -> "PCAEngine":
        path = MODEL_DIR / f"{name}.joblib"
        engine = joblib.load(path)
        print(f"PCA engine loaded ← {path}")
        return engine

    def _check_fitted(self) -> None:
        if not self._fitted:
            raise RuntimeError("Call .fit() before .transform().")


# ------------------------------------------------------------------
# CLI convenience
# ------------------------------------------------------------------

if __name__ == "__main__":
    import sqlalchemy as sa

    config = yaml.safe_load(open(CONFIG_PATH))
    db_path = Path(__file__).parents[2] / config["database"]["path"]
    engine_db = sa.create_engine(f"sqlite:///{db_path}")

    params = config["process_parameters"]
    params_no_titer = [p for p in params if p != "titer_mg_L"]

    # Load reference (normal) batch measurements
    query = f"""
        SELECT m.{', m.'.join(params_no_titer)}
        FROM measurements m
        JOIN batches b ON m.batch_id = b.batch_id
        WHERE b.batch_type = 'normal'
    """
    df_ref = pd.read_sql(query, engine_db)
    X_ref = df_ref[params_no_titer].dropna().values

    pca_engine = PCAEngine(n_components=config["mspc"]["n_components"])
    pca_engine.fit(X_ref)

    print("\nExplained Variance:")
    print(pca_engine.explained_variance_table().to_string(index=False))

    pca_engine.save()
