"""Tier 1 — full quality comparison vs C++ gold standard on all 10 canary
patients. Total runtime budget: ≤ 10 minutes (mostly waiting on parquet I/O
and prepare_data_generalized).

Per-stage tests (canary patients are paralyzed and normal mixed):
    test_au_quality_vs_cpp        — Stage 3: per-AU Pearson r + MAE thresholds,
                                    bucketed by difficulty (easy/medium/hard)
                                    and by patient severity (normal/paralyzed).
    test_peak_frame_agreement     — Stage 4: regenerate-and-compare. Locked
                                    pyfaceau peak frames vs C++-derived peak
                                    frames; ≤3 frames diff for ≥80% of cells.
    test_engineered_features_drift— Stage 5: per-feature |py − cpp|; hard cap
                                    + soft warning thresholds.
    test_inference_parity_per_zone— Stage 6a: saved Jan 1 model on pyfaceau
                                    features vs same model on cpp features;
                                    ≥90% per-patient agreement.

Stages 1-2 (bbox + landmarks) require pyfaceau-side instrumentation that's
deferred to Sub-PR 2; placeholders are present here marked as xfail until
that lands.
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

from _pipeline_helpers import (  # noqa: E402
    AU_COLUMNS,
    AU_DIFFICULTY,
    compare_au_frames,
    compare_bbox_frames,
    compare_landmark_frames,
    derive_bbox_from_landmarks,
    load_pyfaceau_landmarks,
    saved_jan1_predict,
)
from conftest import (  # noqa: E402
    GOLDEN_ROOT,
    Canary,
    parametrize_canaries,
    parametrize_canaries_sides,
)


# ---------------------------------------------------------------------------
# Stage 3 — AU intensities vs C++ gold
# ---------------------------------------------------------------------------


@pytest.mark.tier1
@parametrize_canaries_sides(tier=1)
def test_au_quality_vs_cpp(canary: Canary, side: str, metric_bands):
    """For each canary × side, frame-pair the pyfaceau AU output against C++
    and assert per-AU Pearson r ≥ band and MAE ≤ band. Bands depend on AU
    difficulty bucket and patient severity (normal vs paralyzed)."""
    py_path = GOLDEN_ROOT / "aus" / f"{canary.id}_{side}" / "pyfaceau.parquet"
    cpp_path = GOLDEN_ROOT / "aus" / f"{canary.id}_{side}" / "cpp.parquet"
    if not py_path.exists() or not cpp_path.exists():
        pytest.skip(f"goldens missing for {canary.id} {side}")

    py = pd.read_parquet(py_path).set_index("frame", drop=True)
    cpp = pd.read_parquet(cpp_path).set_index("frame", drop=True)
    cmp = compare_au_frames(py, cpp)

    bucket = canary.threshold_bucket  # "normal" or "paralyzed"
    bands = metric_bands["stage3_aus"][bucket]

    failures: list[str] = []
    for au in AU_COLUMNS:
        difficulty = AU_DIFFICULTY[au]
        if difficulty == "informational":
            continue  # AU05/AU09: rare, no threshold
        thresh = bands[difficulty]
        r = cmp.per_au_pearson.get(au, float("nan"))
        mae = cmp.per_au_mae.get(au, float("nan"))

        # NaN r is allowed if the AU is essentially flat in either side
        # (constant input → undefined correlation, not a regression). MAE
        # still applies — a flat AU should have low MAE if both extractors
        # produce zeros.
        r_ok = math.isnan(r) or r >= thresh["pearson_r_min"]
        mae_ok = math.isnan(mae) or mae <= thresh["mae_max"]
        if not r_ok:
            failures.append(
                f"{au} ({difficulty}): r={r:.3f} < {thresh['pearson_r_min']:.2f}"
            )
        if not mae_ok:
            failures.append(
                f"{au} ({difficulty}): mae={mae:.3f} > {thresh['mae_max']:.2f}"
            )

    if failures:
        pytest.fail(
            f"AU quality regressions for {canary.id} {side} ({bucket} bucket):\n  "
            + "\n  ".join(failures)
        )


# ---------------------------------------------------------------------------
# Stage 4 — peak frame agreement
# ---------------------------------------------------------------------------


@pytest.mark.tier1
@parametrize_canaries(tier=1)
def test_peak_frame_agreement(canary: Canary, metric_bands):
    """Peak frames in current pyfaceau combined_results.csv vs C++ combined_results
    must agree within ±N frames for at least M% of (action × side) cells.

    Compares current locked pyfaceau peak frames against current locked C++
    peak frames (both stored in golden/peak_frames.json). The cross-extractor
    test exercises a different concern than Tier 0 (which just verifies the
    pyfaceau side hasn't drifted from its own snapshot).
    """
    locked = json.loads((GOLDEN_ROOT / "peak_frames.json").read_text())
    py_peaks_all = locked["pyfaceau_peak_frames"]
    cpp_peaks_all = locked.get("cpp_peak_frames", {})
    if canary.id not in py_peaks_all or canary.id not in cpp_peaks_all:
        pytest.skip(f"peak frames missing for {canary.id} in one source")

    bands = metric_bands["stage4_peak_frames"]["shared"]
    max_diff = bands["max_abs_frame_diff"]
    min_frac = bands["fraction_within_tolerance_min"]

    py_peaks = py_peaks_all[canary.id]
    cpp_peaks = cpp_peaks_all[canary.id]

    cell_results: list[tuple[str, int | None]] = []  # (action, diff or None)
    for action, py_v in py_peaks.items():
        cpp_v = cpp_peaks.get(action)
        if py_v is None or cpp_v is None:
            cell_results.append((action, None))
            continue
        cell_results.append((action, abs(int(py_v) - int(cpp_v))))

    valid = [r for r in cell_results if r[1] is not None]
    if not valid:
        pytest.skip(f"{canary.id}: no comparable peak frames")
    within = sum(1 for _, d in valid if d <= max_diff)
    frac = within / len(valid)

    out_of_tolerance = [(a, d) for a, d in valid if d > max_diff]
    assert frac >= min_frac, (
        f"{canary.id}: peak-frame agreement {within}/{len(valid)} = {frac:.2%} "
        f"< {min_frac:.0%} threshold (max_diff={max_diff} frames)\n"
        f"  out of tolerance: {out_of_tolerance[:10]}"
    )


# ---------------------------------------------------------------------------
# Stage 5 — engineered features drift
# ---------------------------------------------------------------------------


@pytest.mark.tier1
def test_engineered_features_drift(metric_bands):
    """For each canary × side row, |py_feature - cpp_feature| must:
      - never exceed `single_feature_drift_hard_cap` (hard fail)
      - be ≤ `soft_warning_threshold` for all but at most
        `max_features_above_warning` features (per row)

    Compares the locked features_pyfaceau.parquet vs features_cpp.parquet.
    """
    py_path = GOLDEN_ROOT / "features_pyfaceau.parquet"
    cpp_path = GOLDEN_ROOT / "features_cpp.parquet"
    if not py_path.exists() or not cpp_path.exists():
        pytest.skip("feature parquets missing")

    py_df = pd.read_parquet(py_path)
    cpp_df = pd.read_parquet(cpp_path)
    bands = metric_bands["stage5_features"]["shared"]
    hard_cap = bands["single_feature_drift_hard_cap"]
    soft = bands["soft_warning_threshold"]
    max_warn = bands["max_features_above_warning"]

    # Align rows by (_patient_id, _side)
    key_cols = ["_patient_id", "_side"]
    feature_cols = [c for c in py_df.columns if c not in {*key_cols, "_y"}]
    merged = py_df.merge(cpp_df, on=key_cols, suffixes=("_py", "_cpp"))
    if len(merged) == 0:
        pytest.fail("No overlapping (patient, side) rows in features parquets")

    hard_fails: list[str] = []
    warn_overflow: list[str] = []

    for _, row in merged.iterrows():
        pid, side = row["_patient_id"], row["_side"]
        bad_features: list[tuple[str, float]] = []
        warning_features: list[tuple[str, float]] = []
        for f in feature_cols:
            a = row[f"{f}_py"]
            b = row[f"{f}_cpp"]
            if pd.isna(a) or pd.isna(b):
                continue
            d = abs(float(a) - float(b))
            if d > hard_cap:
                bad_features.append((f, d))
            elif d > soft:
                warning_features.append((f, d))
        if bad_features:
            hard_fails.append(
                f"{pid} {side}: {len(bad_features)} feature(s) exceed hard cap "
                f"{hard_cap}: {bad_features[:5]}"
            )
        if len(warning_features) > max_warn:
            warn_overflow.append(
                f"{pid} {side}: {len(warning_features)} features exceed soft warning "
                f"{soft} (allowed {max_warn})"
            )

    if hard_fails:
        pytest.fail("Hard feature-drift failures:\n  " + "\n  ".join(hard_fails))
    if warn_overflow:
        pytest.fail("Too many soft-warning features:\n  " + "\n  ".join(warn_overflow))


# ---------------------------------------------------------------------------
# Stage 6a — saved-model inference parity (pyfaceau vs C++ features)
# ---------------------------------------------------------------------------


@pytest.mark.tier1
@pytest.mark.requires_jan1_model
def test_inference_parity_pyfaceau_vs_cpp(jan1_model, metric_bands):
    """Saved Jan 1 Mid Face model applied to pyfaceau features vs same model
    on C++-derived features must agree on ≥ X% of canary (patient, side) rows.

    This catches a different class of regression than the AU-quality test:
    even when AU r is high, downstream feature engineering can amplify small
    differences such that the boundary classifier flips. We want to fail
    that case visibly.
    """
    py_path = GOLDEN_ROOT / "features_pyfaceau.parquet"
    cpp_path = GOLDEN_ROOT / "features_cpp.parquet"
    if not py_path.exists() or not cpp_path.exists():
        pytest.skip("feature parquets missing")

    py_df = pd.read_parquet(py_path)
    cpp_df = pd.read_parquet(cpp_path)

    meta_cols = ["_patient_id", "_side", "_y"]
    py_X = py_df.drop(columns=meta_cols)
    cpp_X = cpp_df.drop(columns=meta_cols)
    py_pred = saved_jan1_predict(py_X, jan1_model)
    cpp_pred = saved_jan1_predict(cpp_X, jan1_model)

    # Must align by (patient, side); features parquets are produced in the
    # same order so np-comparison is sufficient, but assert the alignment
    # explicitly to avoid silent reorderings.
    py_keys = list(zip(py_df["_patient_id"], py_df["_side"]))
    cpp_keys = list(zip(cpp_df["_patient_id"], cpp_df["_side"]))
    assert py_keys == cpp_keys, "Feature parquets row order has diverged"

    agree = int((py_pred == cpp_pred).sum())
    total = len(py_pred)
    frac = agree / total
    threshold = metric_bands["stage6a_inference_parity"]["shared"]["agreement_min"]
    disagreements = [
        f"{pid} {side}: py={py_p} cpp={cpp_p}"
        for (pid, side), py_p, cpp_p in zip(py_keys, py_pred, cpp_pred)
        if py_p != cpp_p
    ]
    assert frac >= threshold, (
        f"Inference parity {agree}/{total}={frac:.2%} below threshold {threshold:.0%}\n"
        f"  Disagreements: {disagreements}"
    )


# ---------------------------------------------------------------------------
# Stage 1 — bbox quality vs C++ (derived from landmarks since C++ CSV doesn't
# expose face_detection bbox columns)
# ---------------------------------------------------------------------------


@pytest.mark.tier1
@parametrize_canaries_sides(tier=1)
def test_bbox_quality_vs_cpp(canary: Canary, side: str, metric_bands):
    """For each canary × side: pyfaceau bbox vs C++-derived bbox (axis-aligned
    rect of the 68 C++ landmarks). Catches face-detection regressions on
    pyfaceau side. Different (more permissive) thresholds for paralyzed faces.

    Skipped if the pyfaceau landmark parquet hasn't been generated yet
    (run instrument_pyfaceau.py first).
    """
    py_path = GOLDEN_ROOT / "landmarks" / f"{canary.id}_{side}" / "pyfaceau.parquet"
    cpp_path = GOLDEN_ROOT / "landmarks" / f"{canary.id}_{side}" / "cpp.parquet"
    if not py_path.exists():
        pytest.skip(
            f"pyfaceau landmarks not captured for {canary.id} {side}; "
            f"run: python tests/instrument_pyfaceau.py --canary {canary.id} --side {side}"
        )
    if not cpp_path.exists():
        pytest.skip(f"C++ landmarks missing for {canary.id} {side}")

    py = load_pyfaceau_landmarks(py_path)
    cpp = pd.read_parquet(cpp_path).set_index("frame", drop=True)
    cpp = cpp[~cpp.index.duplicated(keep="first")]
    cpp_with_bbox = derive_bbox_from_landmarks(cpp)

    cmp = compare_bbox_frames(py, cpp_with_bbox)
    bands = metric_bands["stage1_bbox"][canary.threshold_bucket]

    failures = []
    if not (cmp.median_iou >= bands["median_iou_min"]):
        failures.append(f"median_iou={cmp.median_iou:.3f} < {bands['median_iou_min']:.3f}")
    if "p10_iou_min" in bands and not (cmp.p10_iou >= bands["p10_iou_min"]):
        failures.append(f"p10_iou={cmp.p10_iou:.3f} < {bands['p10_iou_min']:.3f}")
    if not (cmp.median_center_diff_px <= bands["median_center_diff_max_px"]):
        failures.append(
            f"median_center_diff={cmp.median_center_diff_px:.2f}px > "
            f"{bands['median_center_diff_max_px']:.2f}px"
        )
    if "p90_center_diff_max_px" in bands and not (cmp.p90_center_diff_px <= bands["p90_center_diff_max_px"]):
        failures.append(
            f"p90_center_diff={cmp.p90_center_diff_px:.2f}px > "
            f"{bands['p90_center_diff_max_px']:.2f}px"
        )
    if not (cmp.success_rate_py >= bands["success_rate_min"]):
        failures.append(
            f"success_rate_py={cmp.success_rate_py:.3f} < {bands['success_rate_min']:.3f}"
        )
    if failures:
        pytest.fail(
            f"BBox quality regression for {canary.id} {side} ({canary.threshold_bucket}):\n  "
            + "\n  ".join(failures)
        )


# ---------------------------------------------------------------------------
# Stage 2 — landmark quality vs C++
# ---------------------------------------------------------------------------


@pytest.mark.tier1
@parametrize_canaries_sides(tier=1)
def test_landmark_quality_vs_cpp(canary: Canary, side: str, metric_bands):
    """For each canary × side: per-frame pyfaceau 68 landmarks vs C++ 68
    landmarks. Asserts mean / p95 / max per-landmark Euclidean distance is
    within bands (separated by patient severity)."""
    py_path = GOLDEN_ROOT / "landmarks" / f"{canary.id}_{side}" / "pyfaceau.parquet"
    cpp_path = GOLDEN_ROOT / "landmarks" / f"{canary.id}_{side}" / "cpp.parquet"
    if not py_path.exists():
        pytest.skip(
            f"pyfaceau landmarks not captured for {canary.id} {side}; "
            f"run: python tests/instrument_pyfaceau.py --canary {canary.id} --side {side}"
        )
    if not cpp_path.exists():
        pytest.skip(f"C++ landmarks missing for {canary.id} {side}")

    py = load_pyfaceau_landmarks(py_path)
    cpp = pd.read_parquet(cpp_path).set_index("frame", drop=True)
    cpp = cpp[~cpp.index.duplicated(keep="first")]
    cmp = compare_landmark_frames(py, cpp)
    bands = metric_bands["stage2_landmarks"][canary.threshold_bucket]

    failures = []
    if not (cmp.mean_per_landmark_px <= bands["mean_max_px"]):
        failures.append(
            f"mean per-landmark={cmp.mean_per_landmark_px:.2f}px > {bands['mean_max_px']:.2f}px"
        )
    if not (cmp.p95_per_landmark_px <= bands["p95_max_px"]):
        failures.append(
            f"p95 per-landmark={cmp.p95_per_landmark_px:.2f}px > {bands['p95_max_px']:.2f}px"
        )
    if not (cmp.max_per_landmark_px <= bands["max_max_px"]):
        failures.append(
            f"max per-landmark={cmp.max_per_landmark_px:.2f}px > {bands['max_max_px']:.2f}px"
        )
    if failures:
        per_region = ", ".join(f"{r}={v:.1f}px" for r, v in cmp.per_region_mean_px.items())
        pytest.fail(
            f"Landmark quality regression for {canary.id} {side} ({canary.threshold_bucket}):\n  "
            + "\n  ".join(failures)
            + f"\n  per-region means: {per_region}"
        )
