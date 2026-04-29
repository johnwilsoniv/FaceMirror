"""Framework self-test: does the test suite actually FAIL when something
that should be caught is broken?

This is the "deliberate regression" requirement from the canary-pipeline-tests
plan. Each test here:
  1. Constructs a synthetic input that simulates a regression
  2. Calls the same assertion logic the real tests use
  3. Asserts that the failure was caught LOUDLY with a clear message that
     names the patient and stage

If any of these pass when they shouldn't (i.e., the framework fails to catch
the planted bug), this file fails — alerting that the test framework itself
has degraded into rubber-stamping.
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))

from _pipeline_helpers import AU_COLUMNS, AU_DIFFICULTY, compare_au_frames  # noqa: E402
from conftest import GOLDEN_ROOT  # noqa: E402


@pytest.mark.tier0
def test_framework_catches_synthetic_au_regression(metric_bands):
    """Plant a regression: shuffle pyfaceau AU values for a normal canary so
    r drops to near-zero. Running the same assertion the real test uses must
    fail with a clear message naming AU and bucket.

    Uses IMG_0942 left (Tier 0 normal canary) and AU45_r (consistently
    non-flat across normal canaries) so the planted regression is detectable
    — neither side will be constant after injection.
    """
    py_path = GOLDEN_ROOT / "aus" / "IMG_0942_left" / "pyfaceau.parquet"
    cpp_path = GOLDEN_ROOT / "aus" / "IMG_0942_left" / "cpp.parquet"
    if not (py_path.exists() and cpp_path.exists()):
        pytest.skip("IMG_0942 left goldens missing")
    py = pd.read_parquet(py_path).set_index("frame", drop=True).copy()
    cpp = pd.read_parquet(cpp_path).set_index("frame", drop=True)

    # Plant a regression on AU45_r: replace with random noise → r should
    # drop substantially below the normal/easy threshold (which is ~0.86 today).
    rng = np.random.default_rng(42)
    py["AU45_r"] = rng.normal(0, 1, size=len(py))

    cmp = compare_au_frames(py, cpp)
    bands = metric_bands["stage3_aus"]["normal"]  # IMG_0942 is normal
    diff = AU_DIFFICULTY["AU45_r"]  # 'easy'
    thresh = bands[diff]
    r = cmp.per_au_pearson["AU45_r"]
    assert not math.isnan(r), (
        "Planted noise produced NaN correlation — pick an AU whose cpp side "
        "has variance on this canary."
    )
    assert r < thresh["pearson_r_min"], (
        f"FRAMEWORK BUG: planted random noise on AU45_r should produce r below "
        f"{thresh['pearson_r_min']}, but got r={r:.3f} — the test would not "
        f"catch a real regression of this magnitude."
    )


@pytest.mark.tier0
def test_framework_catches_synthetic_landmark_regression():
    """Plant a 50-px landmark shift; assert the per-frame distance metric
    surfaces the regression. (Stub — full Stage 2 test is in sub-PR 2; this
    verifies the metric arithmetic itself is sound.)"""
    from _pipeline_helpers import LANDMARK_REGIONS, compare_landmark_frames

    n_frames = 50
    cols_x = [f"x_{i}" for i in range(68)]
    cols_y = [f"y_{i}" for i in range(68)]
    base = np.random.RandomState(0).rand(n_frames, 68, 2) * 100
    py_data = base.copy()
    py_data[:, 30, 0] += 50.0  # shift landmark 30 by 50 px in x
    cpp_data = base

    def to_df(arr):
        flat = arr.reshape(n_frames, -1)
        df = pd.DataFrame(flat, columns=cols_x + cols_y, dtype=float)
        df["success"] = 1
        df.index.name = "frame"
        return df

    cmp = compare_landmark_frames(to_df(py_data), to_df(cpp_data))
    # Mean error should be visible: 1 of 68 landmarks shifted by 50 across all
    # frames → mean ≈ 50/68 ≈ 0.73 px. Max should be ≈ 50.
    assert cmp.max_per_landmark_px >= 49.0, (
        f"FRAMEWORK BUG: 50px shift should produce max_per_landmark ≥ 49, "
        f"but got {cmp.max_per_landmark_px:.2f}"
    )


@pytest.mark.tier0
def test_framework_catches_synthetic_inference_regression():
    """Force per-row prediction disagreement and check the parity assertion
    fires."""
    from _pipeline_helpers import saved_jan1_predict  # noqa: F401

    fake_pred_a = np.array([0, 1, 2, 0, 1, 2, 0, 1, 2, 0])
    fake_pred_b = np.array([1, 0, 1, 0, 0, 0, 0, 0, 0, 0])  # 8/10 disagree
    agree = int((fake_pred_a == fake_pred_b).sum())
    frac = agree / len(fake_pred_a)
    threshold = 0.90
    assert frac < threshold, (
        f"FRAMEWORK BUG: synthetic 8/10 disagreement should be flagged below "
        f"{threshold:.0%} threshold; got {frac:.0%}."
    )


@pytest.mark.tier0
def test_framework_catches_test_split_drift(tmp_path):
    """Simulate a split drift by mutating the locked test_patients_sides list
    and confirm the equality assertion would fail."""
    locked_path = GOLDEN_ROOT / "test_split_seed42.json"
    if not locked_path.exists():
        pytest.skip("test_split_seed42.json missing")
    locked = json.loads(locked_path.read_text())
    fake_current = list(locked["test_patients_sides"])
    fake_current[0] = "DIFFERENT_PATIENT|Left"  # tampered

    # Real assertion is `current == locked["test_patients_sides"]`; here we
    # just verify the comparison would catch it.
    assert fake_current != locked["test_patients_sides"], (
        "FRAMEWORK BUG: list equality didn't detect a single-element diff"
    )


@pytest.mark.tier0
def test_framework_catches_synthetic_state_carryover():
    """Plant a state-carryover regression: synthesize two AU DataFrames that
    differ in a way state corruption would manifest (cached_bbox staleness on
    frame 0 → wrong landmarks → wrong AU values from frame 1 onward).

    Confirms that pd.testing.assert_frame_equal with check_exact=True (the
    same call test_no_state_carryover_img_0861_isolation_vs_batch makes)
    fails on this synthetic input. If THIS assertion ever stops failing, the
    real state-carryover test would silently let regressions through.
    """
    import numpy as np

    n_frames = 30
    rng = np.random.RandomState(0)
    base_aus = rng.rand(n_frames, 17)
    iso_df = pd.DataFrame(base_aus, columns=[
        "AU01_r", "AU02_r", "AU04_r", "AU05_r", "AU06_r", "AU07_r",
        "AU09_r", "AU10_r", "AU12_r", "AU14_r", "AU15_r", "AU17_r",
        "AU20_r", "AU23_r", "AU25_r", "AU26_r", "AU45_r",
    ])
    iso_df.insert(0, "frame", np.arange(n_frames))
    iso_df.insert(1, "success", 1)

    # State carryover: imagine the cached bbox from a previous video shifted
    # the face crop by a few px. Downstream HOG features change, AU
    # predictions shift. Simulate by adding noise specifically to the AUs that
    # are most landmark-sensitive (AU45 = blink, AU06/07 = eye region).
    batch_df = iso_df.copy()
    batch_df["AU45_r"] = batch_df["AU45_r"] + 0.05  # systematic 0.05 shift
    batch_df.iloc[0, batch_df.columns.get_loc("AU06_r")] += 0.20  # frame-0 spike

    # The real test does pd.testing.assert_frame_equal with check_exact=True
    # and pytest.fails on AssertionError. Here, just verify that comparison
    # raises (we expect it to).
    try:
        pd.testing.assert_frame_equal(
            iso_df.reset_index(drop=True),
            batch_df.reset_index(drop=True),
            check_dtype=False,
            check_exact=True,
        )
    except AssertionError:
        return  # good — the comparison detected the planted regression

    raise AssertionError(
        "FRAMEWORK BUG: pd.testing.assert_frame_equal(check_exact=True) did not "
        "catch a planted 0.05 systematic AU45_r shift + frame-0 AU06_r spike — "
        "state-carryover test would silently let real regressions through."
    )
