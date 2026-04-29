"""Stage 7 — production prediction pipeline test.

Stages 6a/6b test inference using prepare_data_generalized → saved-model
predict (which is the TRAINING-side feature pipeline). This file tests the
PRODUCTION inference path: the same one the GUI invokes per patient via
`ParalysisDetector(zone).detect(row_data, side)`.

The two paths can drift: training uses extract_features() (vectorized over
all patients), production uses extract_features_for_detection() (single-row).
A bug in either path that doesn't touch the other won't be caught by Stage
6a — Stage 7 closes that gap.

For each canary × side × zone (mid/upper/lower), call ParalysisDetector
and compare the predicted severity string ('Normal' | 'Partial' |
'Complete' | 'Error') against a locked golden severity stored in
golden/production_predictions.json.

Failure modes this catches:
  - extract_features_for_detection() drifts from extract_features()
  - production scaler.transform() output shifts (sklearn compat issue)
  - production model.predict() output shifts (xgboost compat issue)
  - GUI/orchestration silently substitutes the wrong model file
  - ZONE_CONFIG['filenames'] paths point at wrong artifacts

Runtime: ~2 sec per canary × side × zone (60 calls total in Tier 1).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
import pytest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))

from conftest import (  # noqa: E402
    GOLDEN_ROOT,
    PYFACEAU_COMBINED_CSV,
    Canary,
    parametrize_canaries,
)


@pytest.fixture(scope="session")
def production_detectors():
    """Load production ParalysisDetector for each zone — one-time cost."""
    from paralysis_detector import ParalysisDetector  # noqa: E402

    return {zone: ParalysisDetector(zone) for zone in ("mid", "upper", "lower")}


@pytest.fixture(scope="session")
def canary_rows() -> dict[str, dict]:
    """Read each canary's row from the current pyfaceau combined_results.csv.
    Returns {patient_id: dict}."""
    df = pd.read_csv(PYFACEAU_COMBINED_CSV, low_memory=False)
    return {
        row["Patient ID"]: row.to_dict()
        for _, row in df.iterrows()
        if row.get("Patient ID")
    }


@pytest.mark.tier1
@parametrize_canaries(tier=1)
def test_production_prediction_matches_golden(
    canary: Canary, production_detectors, canary_rows
):
    """For each canary × side × zone: call the production detector on the
    canary's combined_results row; compare predicted severity to the locked
    golden severity stored at update-goldens time.

    Tests the actual GUI prediction path, not the training-time path.
    """
    golden_path = GOLDEN_ROOT / "production_predictions.json"
    if not golden_path.exists():
        pytest.skip(
            f"production_predictions.json missing; run "
            f"`python tests/update_goldens.py --stage production_predictions "
            f"--reason '...'` first."
        )
    if canary.id not in canary_rows:
        pytest.skip(f"{canary.id} not in current combined_results.csv")
    row = canary_rows[canary.id]

    locked = json.loads(golden_path.read_text())
    canary_locked = locked.get(canary.id, {})
    if not canary_locked:
        pytest.skip(f"no locked predictions for {canary.id}")

    failures: list[str] = []
    per_call: list[str] = []
    for zone, detector in production_detectors.items():
        for side in ("left", "right"):
            key = f"{zone}_{side}"
            expected = canary_locked.get(key)
            if expected is None:
                continue  # not locked for this zone-side
            result_str, conf, details = detector.detect(row, side)
            per_call.append(f"{key}: pred={result_str}, conf={conf:.3f}")
            if result_str != expected:
                failures.append(
                    f"{key}: locked={expected!r} current={result_str!r} "
                    f"(conf={conf:.3f})"
                )
    if failures:
        pytest.fail(
            f"Production-path prediction regression for {canary.id}:\n  "
            + "\n  ".join(failures)
            + "\n\nFull current results:\n  "
            + "\n  ".join(per_call)
        )
