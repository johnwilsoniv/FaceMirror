"""Tier 0 — deterministic invariants. Must pass on every commit.
Total runtime budget: ≤ 30 seconds.

These tests check things that have NO legitimate stochasticity. If any of
them fail, it's a real regression — there is no "noise" excuse.

What's covered:
  - Saved Jan 1 model: predictions on locked features must equal locked array.
    Catches sklearn/xgboost upgrade breaking pickle compatibility.
  - prepare_data_generalized + train_test_split(random_state=42) test split:
    locked patient IDs must match exactly. Catches re-introduction of
    row-order non-determinism (the bug that originally cost weeks).
  - Peak frame stability for canaries: locked peak frames in combined_results.csv
    must match what's there now within ±0 (exact, since we generated the
    goldens FROM combined_results.csv; ±3 frames is the Tier 1 cross-extractor
    tolerance, not this).
  - Pyfaceau AU CSV byte-identity for Tier 0 canaries (IMG_0942 + IMG_2380):
    re-load current pyfaceau output, compare to the golden parquet — every
    AU value at every frame must match exactly. Catches per-video state
    carryover (the original IMG_0861 bug) and any silent change to the
    pyfaceau pipeline that affects existing recorded outputs.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))

from _pipeline_helpers import (  # noqa: E402
    AU_COLUMNS,
    load_cpp_aus,
    load_pyfaceau_aus,
    prepare_mid_features,
    saved_jan1_predict,
    stable_dataframe,
)
from conftest import (  # noqa: E402
    GOLDEN_ROOT,
    PYFACEAU_COMBINED_CSV,
    TIER0_CANARIES,
    Canary,
    parametrize_canaries_sides,
)


# ---------------------------------------------------------------------------
# 1. Saved Jan 1 model predictions on locked features must equal locked array
# ---------------------------------------------------------------------------


@pytest.mark.tier0
@pytest.mark.requires_jan1_model
@pytest.mark.parametrize("source", ["pyfaceau", "cpp"])
def test_saved_jan1_predictions_match_locked(jan1_model, source):
    """Re-apply saved model to locked golden features → must equal locked
    predictions byte-for-byte.

    Failure mode this catches: sklearn/xgboost upgrade silently changes
    classifier output on identical input (broken model load, scaler RNG
    interaction, etc.).
    """
    feats_path = GOLDEN_ROOT / f"features_{source}.parquet"
    preds_path = GOLDEN_ROOT / f"predictions_{source}.json"
    if not feats_path.exists() or not preds_path.exists():
        pytest.skip(f"goldens missing for {source}; run update_goldens.py first")
    df = pd.read_parquet(feats_path)
    locked = json.loads(preds_path.read_text())

    feats_df = df.drop(columns=["_patient_id", "_side", "_y"])
    y_pred = saved_jan1_predict(feats_df, jan1_model)

    assert y_pred.tolist() == locked["y_pred"], (
        f"Saved Jan 1 model predictions on {source} features changed:\n"
        f"  locked:  {locked['y_pred']}\n"
        f"  current: {y_pred.tolist()}"
    )


# ---------------------------------------------------------------------------
# 2. train_test_split(random_state=42) test partition must be the locked set
# ---------------------------------------------------------------------------


@pytest.mark.tier0
def test_test_split_seed42_partition_unchanged():
    """Re-run prepare_data_generalized + train_test_split(42); check that the
    Patient ID|Side identifiers in the test partition exactly match the
    locked set.

    Failure mode this catches: re-introduction of row-order
    non-determinism in prepare_data_generalized (the cc0b1a8d sort regression).
    Also catches changes to test_size or stratify behavior.
    """
    from sklearn.model_selection import train_test_split

    locked_path = GOLDEN_ROOT / "test_split_seed42.json"
    if not locked_path.exists():
        pytest.skip("test_split_seed42.json missing; run update_goldens.py first")
    locked = json.loads(locked_path.read_text())

    feats, y, meta = prepare_mid_features(PYFACEAU_COMBINED_CSV)
    X_tr, X_te, y_tr, y_te, m_tr, m_te = train_test_split(
        feats, y, meta,
        test_size=locked["test_size"],
        random_state=locked["random_state"],
        stratify=y,
    )
    current_ids = sorted(f"{r['Patient ID']}|{r['Side']}" for _, r in m_te.iterrows())

    assert current_ids == locked["test_patients_sides"], (
        "Test partition under random_state=42 has changed.\n"
        f"  locked n={len(locked['test_patients_sides'])}, current n={len(current_ids)}\n"
        f"  symmetric diff: {set(current_ids) ^ set(locked['test_patients_sides'])}"
    )
    assert int(len(X_te)) == locked["n_test"]
    assert int(len(X_tr)) == locked["n_train"]


# ---------------------------------------------------------------------------
# 3. Canary peak frames stable in current combined_results.csv
# ---------------------------------------------------------------------------


@pytest.mark.tier0
def test_canary_peak_frames_in_combined_results_match_locked():
    """For canary patients, every action's peak frame in current pyfaceau
    combined_results.csv must equal the locked value (or both must be NaN/missing).

    Failure mode this catches: silent drift in
    facial_au_analyzer.find_peak_frame() or in the underlying AU CSVs that
    shifts which frame is the peak. ±3 tolerance is for cross-extractor
    comparison (Tier 1); this is a strict equality test.
    """
    locked_path = GOLDEN_ROOT / "peak_frames.json"
    if not locked_path.exists():
        pytest.skip("peak_frames.json missing; run update_goldens.py first")
    locked = json.loads(locked_path.read_text())["pyfaceau_peak_frames"]

    df = pd.read_csv(PYFACEAU_COMBINED_CSV, low_memory=False)
    actual: dict[str, dict[str, int | None]] = {}
    canary_ids = set(locked.keys())
    for _, row in df.iterrows():
        pid = row.get("Patient ID")
        if pid not in canary_ids:
            continue
        peaks = {}
        for col in df.columns:
            if col.endswith("_Max Frame"):
                action = col.split("_")[0]
                v = row[col]
                peaks[action] = None if pd.isna(v) else int(v)
        actual[pid] = peaks

    diffs = []
    for pid, locked_peaks in locked.items():
        if pid not in actual:
            diffs.append(f"{pid}: missing from current combined_results.csv")
            continue
        for action, locked_v in locked_peaks.items():
            cur_v = actual[pid].get(action)
            if locked_v != cur_v:
                diffs.append(f"{pid} {action}: locked={locked_v} current={cur_v}")

    assert not diffs, "Peak frame drift in current combined_results.csv:\n  " + "\n  ".join(diffs[:20])


# ---------------------------------------------------------------------------
# 4. Pyfaceau AU CSV byte-identical for Tier 0 canaries
# ---------------------------------------------------------------------------


@pytest.mark.tier0
@pytest.mark.requires_video
@parametrize_canaries_sides(tier=0)
def test_pyfaceau_au_csv_matches_golden_for_tier0(canary: Canary, side: str):
    """For Tier 0 canaries, current pyfaceau AU CSV must equal the golden
    parquet.

    Failure mode this catches: anyone modifies pyfaceau (or the per-video
    state-reset logic) in a way that changes existing recorded outputs.
    The IMG_0861 state-carryover bug would have failed this test loudly.
    """
    py_csv = canary.pyfaceau_csv(side)
    golden_path = GOLDEN_ROOT / "aus" / f"{canary.id}_{side}" / "pyfaceau.parquet"
    if not golden_path.exists():
        pytest.skip(f"golden missing at {golden_path}")
    if not py_csv.exists():
        pytest.skip(f"pyfaceau CSV missing at {py_csv}")

    current = stable_dataframe(load_pyfaceau_aus(py_csv).reset_index())
    golden = pd.read_parquet(golden_path)

    pd.testing.assert_frame_equal(
        current.reset_index(drop=True),
        golden.reset_index(drop=True),
        check_dtype=False,
        check_exact=True,  # strict byte-equality on all AU values
    )


# ---------------------------------------------------------------------------
# 5. C++ AU CSV byte-identical for Tier 0 canaries (catches accidental edits
#    to S2O Coded Files OF/)
# ---------------------------------------------------------------------------


@pytest.mark.tier0
@pytest.mark.requires_cpp_csv
@parametrize_canaries_sides(tier=0)
def test_cpp_au_csv_matches_golden_for_tier0(canary: Canary, side: str):
    cpp_csv = canary.cpp_csv(side)
    golden_path = GOLDEN_ROOT / "aus" / f"{canary.id}_{side}" / "cpp.parquet"
    if not golden_path.exists() or not cpp_csv.exists():
        pytest.skip(f"missing golden or source for {canary.id} {side}")
    current = stable_dataframe(load_cpp_aus(cpp_csv).reset_index())
    golden = pd.read_parquet(golden_path)
    pd.testing.assert_frame_equal(
        current.reset_index(drop=True),
        golden.reset_index(drop=True),
        check_dtype=False,
        check_exact=True,
    )
