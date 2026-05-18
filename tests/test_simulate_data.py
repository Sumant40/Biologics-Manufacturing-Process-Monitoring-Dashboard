"""
test_simulate_data.py
---------------------
Tests for the bioreactor data simulator.
"""

import numpy as np
import pandas as pd
import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).parents[1]))

from ingestion.simulate_data import simulate_batch, run

TIMEPOINTS = np.arange(0, 340, 4)
RNG = np.random.default_rng(99)

REQUIRED_COLS = [
    "batch_id", "batch_type", "time_h", "pH", "DO_pct", "temperature_C",
    "agitation_rpm", "feed_rate_A_mL_h", "feed_rate_B_mL_h",
    "glucose_g_L", "lactate_g_L", "VCD_e6_mL", "viability_pct", "titer_mg_L",
]


@pytest.mark.parametrize("batch_type", ["normal", "marginal", "ooc"])
def test_batch_columns_present(batch_type):
    df = simulate_batch("TST-001", batch_type, RNG, TIMEPOINTS)
    for col in REQUIRED_COLS:
        assert col in df.columns, f"Missing column: {col}"


@pytest.mark.parametrize("batch_type", ["normal", "marginal", "ooc"])
def test_ph_range(batch_type):
    df = simulate_batch("TST-001", batch_type, RNG, TIMEPOINTS)
    assert df["pH"].between(5.5, 8.5).all(), "pH out of physiological range"


@pytest.mark.parametrize("batch_type", ["normal", "marginal", "ooc"])
def test_do_range(batch_type):
    df = simulate_batch("TST-001", batch_type, RNG, TIMEPOINTS)
    assert df["DO_pct"].between(0, 105).all()


def test_titer_monotone_increase_normal():
    """Titer should be non-decreasing on average for normal batches."""
    df = simulate_batch("TST-NOR", "normal", RNG, TIMEPOINTS)
    titer = df["titer_mg_L"].values
    # First quartile mean < last quartile mean
    q = len(titer) // 4
    assert titer[:q].mean() < titer[-q:].mean()


def test_ooc_lower_titer_than_normal():
    """OOC batches should produce lower titer than normal batches."""
    rng = np.random.default_rng(7)
    normal = simulate_batch("NOR", "normal", rng, TIMEPOINTS)
    ooc = simulate_batch("OOC", "ooc", rng, TIMEPOINTS)
    assert ooc["titer_mg_L"].iloc[-1] < normal["titer_mg_L"].iloc[-1]


def test_run_returns_correct_n_batches():
    config = {
        "simulation": {
            "n_batches": 10,
            "run_duration_days": 14,
            "sample_interval_hours": 4,
            "n_normal": 6,
            "n_marginal": 3,
            "n_ooc": 1,
            "random_seed": 42,
        }
    }
    df = run(config)
    assert df["batch_id"].nunique() == 10


def test_no_nulls_in_core_params():
    df = simulate_batch("TST-001", "normal", RNG, TIMEPOINTS)
    core = ["pH", "temperature_C", "VCD_e6_mL", "titer_mg_L"]
    assert df[core].isna().sum().sum() == 0
