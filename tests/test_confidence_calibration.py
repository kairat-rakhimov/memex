"""Tests for the pure math of the confidence-calibration harness.

The LLM judge / elicitor are live LLM calls (exercised by running the script);
here we pin the deterministic bucketing / ECE / spread / subsample logic.
"""

from __future__ import annotations

import importlib
import sys
from pathlib import Path

_EXPERIMENTS_DIR = Path(__file__).resolve().parents[1] / "experiments"
if str(_EXPERIMENTS_DIR) not in sys.path:
    sys.path.insert(0, str(_EXPERIMENTS_DIR))

cal = importlib.import_module("eval_confidence_calibration")
Observation = cal.Observation
Statement = cal.Statement


def test_bucketize_bands_and_rates() -> None:
    obs = [
        Observation(0.95, True),
        Observation(0.92, False),
        Observation(0.80, True),
        Observation(0.60, False),
        Observation(0.30, False),
    ]
    buckets = cal.bucketize(obs)
    by_band = {(b.lo, b.hi): b for b in buckets}

    top = by_band[(0.9, 1.0)]
    assert top.n == 2
    assert top.faithful_rate == 0.5  # one of two faithful
    assert abs(top.mean_pred - 0.935) < 1e-9

    mid = by_band[(0.7, 0.9)]
    assert mid.n == 1 and mid.faithful_rate == 1.0

    low = by_band[(0.0, 0.5)]
    assert low.n == 1 and low.faithful_rate == 0.0


def test_bucketize_includes_one_point_zero_in_top_band() -> None:
    buckets = cal.bucketize([Observation(1.0, True)])
    top = [b for b in buckets if (b.lo, b.hi) == (0.9, 1.0)][0]
    assert top.n == 1


def test_empty_bands_report_none() -> None:
    buckets = cal.bucketize([Observation(0.95, True)])
    empties = [b for b in buckets if b.n == 0]
    assert empties and all(b.mean_pred is None and b.faithful_rate is None for b in empties)


def test_expected_calibration_error() -> None:
    # one band, mean_pred 0.9, actual 0.5 → ECE = |0.9-0.5| = 0.4
    obs = [Observation(0.9, True), Observation(0.9, False)]
    buckets = cal.bucketize(obs)
    assert abs(cal.expected_calibration_error(buckets) - 0.4) < 1e-9


def test_expected_calibration_error_zero_when_no_data() -> None:
    assert cal.expected_calibration_error(cal.bucketize([])) == 0.0


def test_spread_stats_degenerate_vs_spread() -> None:
    degenerate = cal.spread_stats([0.99, 0.99, 1.0, 0.98])
    assert degenerate["stdev"] < 0.05
    assert degenerate["histogram"][-1] == 4  # all in the top 0.9-1.0 bin

    spread = cal.spread_stats([0.1, 0.4, 0.6, 0.9])
    assert spread["stdev"] > degenerate["stdev"]
    assert spread["n"] == 4 and spread["min"] == 0.1 and spread["max"] == 0.9


def test_spread_stats_empty() -> None:
    assert cal.spread_stats([]) == {"n": 0}


def test_subsample_even_stride_and_passthrough() -> None:
    items = [Statement(text=str(i), predicted=0.5, excerpt="") for i in range(100)]
    assert cal._subsample(items, 200) is items  # fewer than limit → unchanged
    sampled = cal._subsample(items, 10)
    assert len(sampled) == 10
    # even stride, not a head slice
    assert [int(s.text) for s in sampled] == [0, 10, 20, 30, 40, 50, 60, 70, 80, 90]
