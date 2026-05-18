"""
simulate_data.py
----------------
Generates synthetic fed-batch CHO bioreactor run data.

Produces 80 batches across 14-day runs sampled every 4 hours.
Batches are tagged as: normal | marginal | out-of-control (OOC).

Output: data/raw/bioreactor_runs.csv
"""

import numpy as np
import pandas as pd
import yaml
from pathlib import Path

CONFIG_PATH = Path(__file__).parents[2] / "config" / "params.yaml"
OUTPUT_PATH = Path(__file__).parents[2] / "data" / "raw" / "bioreactor_runs.csv"


def load_config() -> dict:
    with open(CONFIG_PATH) as f:
        return yaml.safe_load(f)


def simulate_batch(
    batch_id: str,
    batch_type: str,
    rng: np.random.Generator,
    timepoints: np.ndarray,
) -> pd.DataFrame:
    """
    Simulate one fed-batch CHO culture run.

    Parameters
    ----------
    batch_id   : unique batch identifier string
    batch_type : 'normal' | 'marginal' | 'ooc'
    rng        : seeded numpy random generator
    timepoints : array of hours, e.g. [0, 4, 8, ..., 336]

    Returns
    -------
    DataFrame with columns: batch_id, batch_type, time_h, [process params]
    """
    n = len(timepoints)

    # --- Perturbation multipliers by batch type ---
    noise_scale = {"normal": 1.0, "marginal": 2.5, "ooc": 6.0}[batch_type]
    drift = {"normal": 0.0, "marginal": 0.03, "ooc": 0.15}[batch_type]

    # pH: controlled around 7.1, slight drift for marginal/OOC
    pH = (
        7.1
        + drift * np.linspace(0, -1, n)
        + rng.normal(0, 0.02 * noise_scale, n)
    )

    # Dissolved oxygen %: starts ~60%, drops as cells grow, recovers with agitation
    DO_base = 60 * np.exp(-0.005 * timepoints) + 20
    DO_pct = DO_base + rng.normal(0, 1.5 * noise_scale, n)
    DO_pct = np.clip(DO_pct, 5, 100)

    # Temperature: setpoint 36.5°C, slight excursion for OOC
    temp_shift = drift * 2 * np.sin(np.pi * timepoints / 168)
    temperature_C = 36.5 + temp_shift + rng.normal(0, 0.05 * noise_scale, n)

    # Agitation RPM: ramps up as VCD increases
    agitation_rpm = 150 + 50 * (timepoints / 336) + rng.normal(0, 3 * noise_scale, n)

    # Feed rates (mL/h): bolus-like increases over time
    feed_A = np.maximum(0, 5 + 0.15 * timepoints + rng.normal(0, 0.5 * noise_scale, n))
    feed_B = np.maximum(0, 2 + 0.08 * timepoints + rng.normal(0, 0.3 * noise_scale, n))

    # Glucose: fed to maintain ~4 g/L
    glucose = np.clip(
        4 + rng.normal(0, 0.4 * noise_scale, n) - drift * timepoints / 100,
        0.5,
        10,
    )

    # Lactate: accumulates then decreases (metabolic shift)
    lactate_profile = 2.5 * np.exp(-((timepoints - 120) ** 2) / (2 * 60**2)) + 0.5
    lactate = lactate_profile + rng.normal(0, 0.15 * noise_scale, n)
    lactate = np.maximum(0, lactate)

    # VCD (10^6 cells/mL): logistic growth
    k = 0.03 - drift * 0.008
    vcd_max = 20 - noise_scale * 2 + rng.normal(0, 1)
    VCD = vcd_max / (1 + np.exp(-k * (timepoints - 120)))
    VCD += rng.normal(0, 0.3 * noise_scale, n)
    VCD = np.maximum(0, VCD)

    # Viability %: high early, declines in late culture
    viability = np.clip(
        98 - 0.1 * timepoints + drift * (-5) + rng.normal(0, 0.8 * noise_scale, n),
        30,
        100,
    )

    # Titer (mg/L): accumulates with VCD; OOC batches have suppressed titer
    titer_scale = {"normal": 1.0, "marginal": 0.85, "ooc": 0.5}[batch_type]
    titer = np.maximum(
        0,
        titer_scale * 800 * (timepoints / 336) ** 1.5
        + rng.normal(0, 10 * noise_scale, n),
    )

    df = pd.DataFrame(
        {
            "batch_id": batch_id,
            "batch_type": batch_type,
            "time_h": timepoints,
            "pH": pH,
            "DO_pct": DO_pct,
            "temperature_C": temperature_C,
            "agitation_rpm": agitation_rpm,
            "feed_rate_A_mL_h": feed_A,
            "feed_rate_B_mL_h": feed_B,
            "glucose_g_L": glucose,
            "lactate_g_L": lactate,
            "VCD_e6_mL": VCD,
            "viability_pct": viability,
            "titer_mg_L": titer,
        }
    )
    return df


def run(config: dict) -> pd.DataFrame:
    sim = config["simulation"]
    rng = np.random.default_rng(sim["random_seed"])

    duration_h = sim["run_duration_days"] * 24
    timepoints = np.arange(0, duration_h + sim["sample_interval_hours"], sim["sample_interval_hours"])

    batch_meta = (
        [("normal", i) for i in range(sim["n_normal"])]
        + [("marginal", i) for i in range(sim["n_marginal"])]
        + [("ooc", i) for i in range(sim["n_ooc"])]
    )

    frames = []
    for batch_type, idx in batch_meta:
        batch_id = f"{batch_type[:3].upper()}-{idx+1:03d}"
        df = simulate_batch(batch_id, batch_type, rng, timepoints)
        frames.append(df)

    all_batches = pd.concat(frames, ignore_index=True)
    return all_batches


if __name__ == "__main__":
    config = load_config()
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    data = run(config)
    data.to_csv(OUTPUT_PATH, index=False)

    n_batches = data["batch_id"].nunique()
    print(f"Simulated {n_batches} batches → {OUTPUT_PATH}")
    print(data.groupby("batch_type")["batch_id"].nunique())
