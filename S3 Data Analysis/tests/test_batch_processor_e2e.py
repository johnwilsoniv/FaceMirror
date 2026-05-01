"""End-to-end test for facial_au_batch_processor.py.

The framework's existing tests lock per-canary peak frames + engineered
features as they appear in the CURRENT combined_results.csv. They do NOT
test the code path that PRODUCES combined_results.csv. So if someone
modifies facial_au_analyzer.find_peak_frame() (or any other peak-finding /
aggregation logic in facial_au_batch_processor.py), our locked goldens
silently stay valid until somebody manually re-runs main.py --batch.

This test closes that loop: run FacialAUBatchProcessor on a 2-canary
subset (using their existing S2O AU CSVs as input), and assert the
resulting combined_results.csv subset rows match the locked golden
parquet at golden/batch_processor_subset.parquet.

Note we lock the BATCH PROCESSOR OUTPUT directly, not what's currently
in S3O combined_results.csv. The two CAN drift (e.g. config tweaks
between runs); this test catches drift in the CODE PATH, regardless of
the production-CSV state.

Failure modes this catches:
  - find_peak_frame() implementation change shifting peak detection
  - Baseline frame logic change (BL action determination)
  - Per-action AU aggregation change
  - calculate_normalized_values() change (max(0, peak - baseline))
  - Action-name parsing changes (the 'action 0' warning in current logs)

Runtime: ~10-15s.
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import pandas as pd
import pytest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))

from conftest import CANARIES_BY_ID, GOLDEN_ROOT  # noqa: E402

# Use Tier 0 normal + Tier 0 paralyzed canaries.
SUBSET_CANARY_IDS = ["IMG_0942", "IMG_2380"]


@pytest.mark.tier1
@pytest.mark.slow
@pytest.mark.requires_video
def test_batch_processor_combined_results_matches_locked():
    """Run FacialAUBatchProcessor on the 2-canary subset, compare its
    combined_results.csv subset rows to the locked golden parquet.

    Skips cleanly if golden hasn't been generated. Generate via:
        python tests/update_goldens.py --stage batch_processor_subset \\
            --reason 'initial baseline'
    """
    from facial_au_batch_processor import FacialAUBatchProcessor

    golden_path = GOLDEN_ROOT / "batch_processor_subset.parquet"
    if not golden_path.exists():
        pytest.skip(
            "batch_processor_subset.parquet missing; run "
            "`python tests/update_goldens.py --stage batch_processor_subset "
            "--reason '...'` first."
        )
    golden = pd.read_parquet(golden_path).sort_values("Patient ID").reset_index(drop=True)

    canaries = [CANARIES_BY_ID[cid] for cid in SUBSET_CANARY_IDS]
    for c in canaries:
        for side in ("left", "right"):
            if not c.pyfaceau_csv(side).exists():
                pytest.skip(f"missing pyfaceau CSV: {c.id} {side}")

    with tempfile.TemporaryDirectory() as tmp_dir_str:
        tmp_dir = Path(tmp_dir_str)
        processor = FacialAUBatchProcessor(output_dir=str(tmp_dir))
        for c in canaries:
            processor.add_patient(
                left_csv=str(c.pyfaceau_csv("left")),
                right_csv=str(c.pyfaceau_csv("right")),
                video_path=str(c.video("left")) if c.video("left").exists() else None,
                patient_id=c.id,
            )
        result = processor.process_all(extract_frames=False, max_workers=1)
        assert result is not None, "process_all returned None — check log"

        produced_csv = tmp_dir / "combined_results.csv"
        assert produced_csv.exists(), f"combined_results.csv not written at {produced_csv}"
        produced = pd.read_csv(produced_csv, low_memory=False)
        produced = produced[produced["Patient ID"].isin(SUBSET_CANARY_IDS)]
        produced = produced.sort_values("Patient ID").reset_index(drop=True)

    # Drop columns that legitimately vary across runs (timestamps/paths).
    skip_cols = {
        "Processing Status", "Processing Time", "Output Path",
        "Frame Path", "Timestamp", "Output Folder",
    }
    produced = produced[[c for c in produced.columns if c not in skip_cols]]

    # Align column order to golden's
    common_cols = [c for c in golden.columns if c in produced.columns]
    missing_in_produced = [c for c in golden.columns if c not in produced.columns]
    extra_in_produced = [c for c in produced.columns if c not in golden.columns]
    if missing_in_produced or extra_in_produced:
        pytest.fail(
            "batch_processor output schema drift:\n"
            f"  Columns in golden missing from produced: {missing_in_produced[:10]}\n"
            f"  Columns in produced not in golden:       {extra_in_produced[:10]}"
        )

    produced = produced[common_cols]
    if "Patient ID" in produced.columns:
        produced["Patient ID"] = produced["Patient ID"].astype(str)

    # Per-cell comparison with NaN equality + 1e-6 numeric tolerance
    assert len(produced) == len(golden), (
        f"Row count mismatch: produced={len(produced)} golden={len(golden)}"
    )
    failures: list[str] = []
    for col in common_cols:
        for i in range(len(golden)):
            exp = golden.iloc[i][col]
            got = produced.iloc[i][col]
            pid = golden.iloc[i]["Patient ID"]
            if pd.isna(exp) and pd.isna(got):
                continue
            if pd.isna(exp) or pd.isna(got):
                failures.append(f"{pid}/{col}: golden={exp!r} produced={got!r}")
                continue
            try:
                if abs(float(exp) - float(got)) > 1e-6:
                    failures.append(
                        f"{pid}/{col}: golden={exp} produced={got} (diff={abs(float(exp)-float(got)):.6f})"
                    )
                continue
            except (TypeError, ValueError):
                pass
            if str(exp) != str(got):
                failures.append(f"{pid}/{col}: golden={exp!r} produced={got!r}")

    if failures:
        pytest.fail(
            f"facial_au_batch_processor output drifted from locked golden "
            f"for {SUBSET_CANARY_IDS}:\n  "
            + "\n  ".join(failures[:20])
            + (f"\n  ... and {len(failures) - 20} more" if len(failures) > 20 else "")
            + "\n\nIf this is a deliberate change, re-lock with:\n"
            + "  python tests/update_goldens.py --stage batch_processor_subset --reason '...'"
        )
