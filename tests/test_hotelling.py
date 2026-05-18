"""
test_hotelling.py
-----------------
Unit tests for the HotellingT2 monitor.
"""

import numpy as np
import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))
from monitoring.hotelling_t2 import HotellingT2


@pytest.fixture
def reference_scores():
    """100 reference observations across 3 PCs, drawn from N(0,1)."""
    rng = np.random.default_rng(42)
    return rng.normal(size=(100, 3))


def test_fit_sets_ucl(reference_scores):
    monitor = HotellingT2(n_components=3, alpha=0.05)
    monitor.fit(reference_scores)
    assert monitor.ucl_phase1_ > 0
    assert monitor.ucl_phase2_ > 0


def test_ucl_phase2_gt_phase1(reference_scores):
    monitor = HotellingT2(n_components=3).fit(reference_scores)
    # Phase-2 UCL should be larger (more conservative for new obs)
    assert monitor.ucl_phase2_ >= monitor.ucl_phase1_


def test_false_alarm_rate(reference_scores):
    """At α=0.05, roughly ≤5% of reference obs should exceed UCL."""
    monitor = HotellingT2(n_components=3, alpha=0.05)
    monitor.fit(reference_scores)
    flags = monitor.flag(reference_scores, phase=1)
    false_alarm_rate = flags.mean()
    assert false_alarm_rate <= 0.10, f"FAR too high: {false_alarm_rate:.2f}"


def test_ooc_detected():
    """Observations far from origin (high T²) must be flagged as OOC."""
    rng = np.random.default_rng(0)
    T_ref = rng.normal(size=(200, 3))
    monitor = HotellingT2(n_components=3, alpha=0.05).fit(T_ref)

    T_ooc = np.full((5, 3), 10.0)  # extreme outliers
    flags = monitor.flag(T_ooc, phase=2)
    assert flags.all(), "All OOC observations should be flagged"


def test_score_shape(reference_scores):
    monitor = HotellingT2(n_components=3).fit(reference_scores)
    t2 = monitor.score(reference_scores)
    assert t2.shape == (100,)


def test_unfitted_raises():
    monitor = HotellingT2(n_components=3)
    with pytest.raises(RuntimeError):
        monitor.score(np.ones((5, 3)))


def test_summary_columns(reference_scores):
    monitor = HotellingT2(n_components=3).fit(reference_scores)
    batch_ids = [f"B{i:03d}" for i in range(100)]
    summary = monitor.summary(reference_scores, batch_ids)
    assert set(["batch_id", "t2_max", "t2_mean", "ucl", "ooc"]).issubset(summary.columns)
