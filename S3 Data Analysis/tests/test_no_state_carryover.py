"""State-carryover regression test — direct insurance against the IMG_0861
bug class that originally cost weeks of investigation.

The historical bug
-------------------
Pre-Apr-2026 pyfaceau accumulated per-video state on the Pipeline instance:
  - cached_bbox (face-tracking shortcut from previous video's last frame)
  - frames_since_detection counter
  - stored_features list (two-pass AU re-prediction holdover)
  - running_median dual-histogram (per-AU neutral-expression baseline)
  - online_au_correction percentile tracker
  - pyclnf CLNF temporal state (template tracking, scale-cache)
  - pyclnf GPU cache buffers

When the same Pipeline was reused for a sequence of videos (e.g. batch
processing), state from earlier videos contaminated later ones. The
canonical observation: IMG_0861 in isolation produced ~0.7 px landmark
error vs C++; IMG_0861 as the 7th video in a batch produced ~266 px error
on the same frames.

The fix
-------
Pipeline._reset_per_video_state() is now called at the start of every
process_video. This test verifies that the reset is intact: process the
historical canary (IMG_0861 left) twice — once isolated, once as the 2nd
video in a 2-video batch — and assert byte-identical AU output.

Why position-2 not position-7
-----------------------------
The bug accumulates with each prior video, but it begins after the FIRST
prior video (cached_bbox is set after video 1, then reused on the first
frame of video 2). So position-2 is sufficient signal at much lower
runtime. Position-7 is the original observation site; the lower-bound
position to detect the bug is position-2.

Test runtime: ~75 sec (3 videos × ~30 frames each at ~1.2 fps).
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))

from conftest import CANARIES_BY_ID, DETERMINISM_ENV, S1O_VIDEOS, Canary  # noqa: E402

# Frame budget per video used in the carryover test. Small enough to keep the
# test under 90 seconds; large enough to give cached_bbox + running_median
# corruption visible signal in the output AU stream.
MAX_FRAMES_FOR_CARRYOVER_TEST = 30

# The historical canary patient (the one whose 240-px landmark error in
# batch position 7 originally exposed the bug).
HISTORICAL_CANARY_ID = "IMG_0861"
# The video that goes BEFORE the canary in the 2-video batch run.
# Pick something different (different patient + different signal profile)
# so any cross-video state contamination has visible structure.
PRECEDING_VIDEO_CANARY_ID = "IMG_0942"  # normal canary, very different feature profile


@pytest.fixture(scope="module")
def pyfaceau_pipeline_factory():
    """Yield a callable that produces a fresh OpenFaceProcessor each call.
    Module-scoped so we initialize the heavy weights once."""
    # Heavy import; defer until needed
    from pyfaceau.processor import OpenFaceProcessor  # noqa

    def _make():
        return OpenFaceProcessor(verbose=False)

    return _make


def _process(pipeline, video_path: Path, max_frames: int) -> pd.DataFrame:
    """Run a Pipeline on one video, return the DataFrame of per-frame
    pyfaceau output (frame, success, AU01_r..AU45_r). Bypasses the
    process_video CSV write to keep the test fully in-memory."""
    df = pipeline.pipeline.process_video(
        str(video_path),
        output_csv=None,        # don't write a CSV
        max_frames=max_frames,
    )
    if "frame" not in df.columns:
        # Fall back: derive frame index from row position
        df = df.reset_index().rename(columns={"index": "frame"})
    return df.reset_index(drop=True)


@pytest.mark.tier1
@pytest.mark.slow
@pytest.mark.requires_video
def test_no_state_carryover_img_0861_isolation_vs_batch(pyfaceau_pipeline_factory):
    """The IMG_0861 historical canary: AU output processing it isolated must
    equal AU output processing it as 2nd video in a batch.

    Failure mode this catches:
      - Anyone removes Pipeline._reset_per_video_state() or breaks any of
        its cleanup steps (cached_bbox, running_median, stored_features,
        pyclnf temporal state, etc.)
      - A new piece of per-video state is introduced that the reset doesn't
        cover (the test's strength is that ANY state-carryover regression
        produces non-byte-identical AU output)
    """
    canary = CANARIES_BY_ID[HISTORICAL_CANARY_ID]
    preceding = CANARIES_BY_ID[PRECEDING_VIDEO_CANARY_ID]
    side = "left"

    canary_video = canary.video(side)
    preceding_video = preceding.video(side)
    if not canary_video.exists():
        pytest.skip(f"missing canary video at {canary_video}")
    if not preceding_video.exists():
        pytest.skip(f"missing preceding video at {preceding_video}")

    # Run A: isolation. Fresh Pipeline, just IMG_0861.
    iso_processor = pyfaceau_pipeline_factory()
    iso_df = _process(iso_processor, canary_video, MAX_FRAMES_FOR_CARRYOVER_TEST)

    # Run B: batch. Fresh Pipeline, process preceding video first, then IMG_0861
    # WITHOUT building a new Pipeline. If state-reset works, the second
    # process_video call will reset all per-video state internally.
    batch_processor = pyfaceau_pipeline_factory()
    _ = _process(batch_processor, preceding_video, MAX_FRAMES_FOR_CARRYOVER_TEST)
    batch_df = _process(batch_processor, canary_video, MAX_FRAMES_FOR_CARRYOVER_TEST)

    # Sanity: should have processed the same frame count
    assert len(iso_df) == len(batch_df), (
        f"Different frame counts (iso={len(iso_df)}, batch={len(batch_df)}) — "
        f"max_frames not honoring identically?"
    )

    # Strict byte-equality on every column. If the state-reset is broken,
    # the second video in the batch will have wrong cached_bbox, leading to
    # wrong landmarks → wrong HOG → wrong AU values from frame 1 onward.
    try:
        pd.testing.assert_frame_equal(
            iso_df.reset_index(drop=True),
            batch_df.reset_index(drop=True),
            check_dtype=False,
            check_exact=True,
        )
    except AssertionError as e:
        # Make the failure message specifically actionable for this bug class
        au_cols = [c for c in iso_df.columns if c.startswith("AU") and c.endswith("_r")]
        if au_cols:
            diffs = {}
            for c in au_cols:
                if c in batch_df.columns:
                    a = iso_df[c].astype(float)
                    b = batch_df[c].astype(float)
                    d = (a - b).abs()
                    if d.max() > 0:
                        diffs[c] = (float(d.max()), float(d.mean()))
            top = sorted(diffs.items(), key=lambda kv: -kv[1][0])[:5]
            top_summary = "  " + "\n  ".join(
                f"{c}: max diff={mx:.4f}, mean diff={mn:.4f}" for c, (mx, mn) in top
            )
        else:
            top_summary = "  (no AU columns to summarize)"

        pytest.fail(
            "State-carryover REGRESSION: pyfaceau output for "
            f"{HISTORICAL_CANARY_ID} {side} differs between isolation and "
            "batch position 2.\n\n"
            "This means Pipeline._reset_per_video_state() either was removed, "
            "broke, or is missing coverage of a newly-added piece of "
            "per-video state.\n\n"
            "The original IMG_0861 manifestation of this bug class produced "
            "~266 px landmark error → wildly wrong AU values. Top per-AU "
            "diffs (current run):\n"
            f"{top_summary}\n\n"
            f"See pyfaceau/pipeline.py:_reset_per_video_state() for the "
            f"required cleanup steps.\n\n"
            f"Original assert message:\n  {e}"
        )


@pytest.mark.tier2
@pytest.mark.slow
@pytest.mark.requires_video
def test_no_state_carryover_full_canary_batch(pyfaceau_pipeline_factory):
    """Stronger version of the isolation-vs-batch test: process ALL Tier 0
    canaries in batch (shared Pipeline) and assert each one matches its
    isolation-mode output. Catches state-carryover that begins at any
    position, not just position 2.

    Tier 2 because it processes 4 videos × 2 runs = 8 video starts ≈ 4 min.
    """
    canaries_in_batch = [
        CANARIES_BY_ID["IMG_0942"],   # normal — Tier 0 canary
        CANARIES_BY_ID["IMG_2380"],   # paralyzed — Tier 0 canary
        CANARIES_BY_ID["IMG_0861"],   # historical canary
        CANARIES_BY_ID["IMG_0422"],   # additional normal for variance
    ]
    side = "left"
    max_frames = MAX_FRAMES_FOR_CARRYOVER_TEST

    # Skip if any video missing
    for c in canaries_in_batch:
        if not c.video(side).exists():
            pytest.skip(f"missing video for {c.id} {side}")

    # Run A: isolation pass (fresh Pipeline per video)
    isolation: dict[str, pd.DataFrame] = {}
    for c in canaries_in_batch:
        proc = pyfaceau_pipeline_factory()
        isolation[c.id] = _process(proc, c.video(side), max_frames)

    # Run B: batch pass (single Pipeline reused across all)
    batch_processor = pyfaceau_pipeline_factory()
    batch: dict[str, pd.DataFrame] = {}
    for c in canaries_in_batch:
        batch[c.id] = _process(batch_processor, c.video(side), max_frames)

    # Compare per-canary
    failures: list[str] = []
    for c in canaries_in_batch:
        try:
            pd.testing.assert_frame_equal(
                isolation[c.id].reset_index(drop=True),
                batch[c.id].reset_index(drop=True),
                check_dtype=False,
                check_exact=True,
            )
        except AssertionError as e:
            failures.append(f"{c.id}: {str(e).splitlines()[0]}")
    if failures:
        pytest.fail(
            "State-carryover regression in batch processing across multiple "
            "canaries:\n  " + "\n  ".join(failures)
        )
