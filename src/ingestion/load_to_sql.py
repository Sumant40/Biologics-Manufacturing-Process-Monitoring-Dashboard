"""
load_to_sql.py
--------------
Loads simulated bioreactor CSVs into a SQLite database with two tables:

  batches        — one row per batch (metadata + final titer as yield label)
  measurements   — time-series process parameters per batch

Usage
-----
    python src/ingestion/load_to_sql.py
"""

import pandas as pd
import sqlalchemy as sa
import yaml
from pathlib import Path

CONFIG_PATH = Path(__file__).parents[2] / "config" / "params.yaml"
RAW_DATA_PATH = Path(__file__).parents[2] / "data" / "raw" / "bioreactor_runs.csv"


def load_config() -> dict:
    with open(CONFIG_PATH) as f:
        return yaml.safe_load(f)


def build_engine(db_path: Path) -> sa.Engine:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    return sa.create_engine(f"sqlite:///{db_path}", echo=False)


def create_schema(engine: sa.Engine) -> None:
    """Create tables if they don't exist."""
    ddl = """
    CREATE TABLE IF NOT EXISTS batches (
        batch_id        TEXT PRIMARY KEY,
        batch_type      TEXT NOT NULL,           -- normal | marginal | ooc
        n_timepoints    INTEGER,
        final_titer_mg_L REAL,                   -- yield label (day-14 titer)
        created_at      TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    );

    CREATE TABLE IF NOT EXISTS measurements (
        id              INTEGER PRIMARY KEY AUTOINCREMENT,
        batch_id        TEXT NOT NULL REFERENCES batches(batch_id),
        time_h          REAL NOT NULL,
        pH              REAL,
        DO_pct          REAL,
        temperature_C   REAL,
        agitation_rpm   REAL,
        feed_rate_A_mL_h REAL,
        feed_rate_B_mL_h REAL,
        glucose_g_L     REAL,
        lactate_g_L     REAL,
        VCD_e6_mL       REAL,
        viability_pct   REAL,
        titer_mg_L      REAL
    );

    CREATE INDEX IF NOT EXISTS idx_measurements_batch_time
        ON measurements(batch_id, time_h);
    """
    with engine.connect() as conn:
        for stmt in ddl.strip().split(";"):
            stmt = stmt.strip()
            if stmt:
                conn.execute(sa.text(stmt))
        conn.commit()
    print("Schema created / verified.")


def load_data(engine: sa.Engine, df: pd.DataFrame) -> None:
    """Populate batches and measurements tables from the raw DataFrame."""

    # --- Build batches table ---
    final_titer = (
        df.groupby("batch_id")["titer_mg_L"].last().reset_index()
        .rename(columns={"titer_mg_L": "final_titer_mg_L"})
    )
    batch_meta = (
        df.groupby("batch_id")[["batch_type"]]
        .first()
        .reset_index()
        .merge(final_titer, on="batch_id")
    )
    batch_meta["n_timepoints"] = df.groupby("batch_id").size().values

    batch_meta.to_sql("batches", engine, if_exists="replace", index=False)
    print(f"Inserted {len(batch_meta)} rows into `batches`.")

    # --- Build measurements table ---
    measure_cols = [
        "batch_id", "time_h", "pH", "DO_pct", "temperature_C",
        "agitation_rpm", "feed_rate_A_mL_h", "feed_rate_B_mL_h",
        "glucose_g_L", "lactate_g_L", "VCD_e6_mL", "viability_pct", "titer_mg_L",
    ]
    df[measure_cols].to_sql("measurements", engine, if_exists="replace", index=False)
    print(f"Inserted {len(df)} rows into `measurements`.")


def verify(engine: sa.Engine) -> None:
    with engine.connect() as conn:
        n_batches = conn.execute(sa.text("SELECT COUNT(*) FROM batches")).scalar()
        n_meas = conn.execute(sa.text("SELECT COUNT(*) FROM measurements")).scalar()
        types = conn.execute(
            sa.text("SELECT batch_type, COUNT(*) as n FROM batches GROUP BY batch_type")
        ).fetchall()
    print(f"\nVerification → batches: {n_batches}, measurements: {n_meas}")
    for row in types:
        print(f"  {row[0]}: {row[1]} batches")


if __name__ == "__main__":
    config = load_config()
    db_path = Path(__file__).parents[2] / config["database"]["path"]
    engine = build_engine(db_path)

    print(f"Loading data from {RAW_DATA_PATH} ...")
    df = pd.read_csv(RAW_DATA_PATH)

    create_schema(engine)
    load_data(engine, df)
    verify(engine)
    print(f"\nDatabase ready at: {db_path}")
