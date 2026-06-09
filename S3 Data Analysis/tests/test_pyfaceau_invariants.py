"""Cross-run determinism + GPU-mode invariants for pyfaceau.

Two distinct test families:

1. test_pyfaceau_run_to_run_determinism — runs pyfaceau on the same canary
   video twice (fresh Pipeline each time) and asserts byte-identical AU
   output. Verifies the FOUNDATIONAL ASSUMPTION that the state-carryover
   test (test_no_state_carryover.py) relies on. If pyfaceau ever introduces
   a non-deterministic path (timestamp-seeded RNG, parallel scheduling
   timing, GPU memory residue), this catches it directly — and prevents
   the state-carryover test from silently rubber-stamping non-determinism
   as "no carryover".

2. test_clnf_config_use_gpu_enabled / test_pyfaceau_gpu_divergence_within_band
   — guards the production "use_gpu=True" state. The v1316 dataset was built
   with GPU CLNF on Windows-CUDA (confirmed bit-exact, MAE=0 / r=1.0 on every
   canary x side; see S3 Data Analysis/LIDO_PART_A_WINDOWS_RESULTS.md), so
   flipping to CPU would diverge from the dataset (CPU is close but not
   bit-exact, mean r 0.86-0.98 vs the v1316 goldens). An older comment claimed
   CPU was "38% better on paralyzed faces" (grid_sample vs cv2.warpAffine);
   that was re-validated under pyfaceau 1.3.16 on paralyzed canaries and
   REFUTED — GPU vs CPU correlate equally with the C++ ground truth (r-delta
   in [-0.009, +0.003], FP-drift). The Tier 1 test still locks the CPU-vs-GPU
   divergence bounds so an unrelated change that moves them fails loudly.
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

from conftest import CANARIES_BY_ID, GOLDEN_ROOT, Canary  # noqa: E402

# Use the Tier 0 normal canary for both tests (small, well-behaved video).
DETERMINISM_CANARY_ID = "IMG_0942"
GPU_BASELINE_CANARY_ID = "IMG_0942"
MAX_FRAMES = 30  # keeps each run ≤ ~30s


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def cpu_processor_factory():
    """Returns a callable that builds a fresh OpenFaceProcessor with GPU
    disabled (the production state)."""
    from pyfaceau.processor import OpenFaceProcessor  # heavy import

    def _make():
        return OpenFaceProcessor(verbose=False)

    return _make


def _process(processor, video_path: Path, max_frames: int) -> pd.DataFrame:
    """Run pyfaceau on one video; return per-frame DataFrame (frame, success,
    AU01_r..AU45_r)."""
    df = processor.pipeline.process_video(
        str(video_path),
        output_csv=None,
        max_frames=max_frames,
    )
    if "frame" not in df.columns:
        df = df.reset_index().rename(columns={"index": "frame"})
    return df.reset_index(drop=True)


# ---------------------------------------------------------------------------
# 1. Cross-run determinism
# ---------------------------------------------------------------------------


@pytest.mark.tier1
@pytest.mark.slow
@pytest.mark.requires_video
def test_pyfaceau_run_to_run_determinism(cpu_processor_factory):
    """Run pyfaceau on IMG_0942 left twice, FRESH Pipeline each time.
    AU output must be byte-identical between the two runs.

    Why this matters: test_no_state_carryover assumes pyfaceau is bit-exact
    deterministic on identical input. If that assumption breaks, the
    state-carryover test silently rubber-stamps non-determinism as
    "no carryover" because both isolation and batch outputs would just
    be different runs of a non-deterministic process.
    """
    canary = CANARIES_BY_ID[DETERMINISM_CANARY_ID]
    video = canary.video("left")
    if not video.exists():
        pytest.skip(f"missing video at {video}")

    proc_a = cpu_processor_factory()
    df_a = _process(proc_a, video, MAX_FRAMES)

    proc_b = cpu_processor_factory()
    df_b = _process(proc_b, video, MAX_FRAMES)

    assert len(df_a) == len(df_b), (
        f"Different frame counts: a={len(df_a)} b={len(df_b)} (max_frames not honored?)"
    )
    try:
        pd.testing.assert_frame_equal(
            df_a.reset_index(drop=True),
            df_b.reset_index(drop=True),
            check_dtype=False,
            check_exact=True,
        )
    except AssertionError as e:
        # Make the failure message specifically actionable
        au_cols = [c for c in df_a.columns if c.startswith("AU") and c.endswith("_r")]
        diffs = []
        for c in au_cols:
            if c in df_b.columns:
                a = df_a[c].astype(float)
                b = df_b[c].astype(float)
                d = (a - b).abs()
                if d.max() > 0:
                    diffs.append((c, float(d.max()), float(d.mean())))
        diffs.sort(key=lambda kv: -kv[1])
        diff_summary = "\n  ".join(
            f"{c}: max={mx:.4f} mean={mn:.4f}" for c, mx, mn in diffs[:5]
        )
        pytest.fail(
            "Pyfaceau is no longer DETERMINISTIC across runs.\n\n"
            "Two fresh-Pipeline runs of the same video produced different "
            "AU output. This breaks the foundational assumption of "
            "test_no_state_carryover.py — which compares isolation vs batch "
            "and assumes both runs are reproducible.\n\n"
            "Likely causes:\n"
            "  - timestamp-seeded RNG introduced somewhere\n"
            "  - parallel-scheduling timing affecting reduction order\n"
            "  - GPU memory residue (if GPU was just re-enabled)\n"
            "  - non-deterministic numpy/PyTorch operation\n\n"
            f"Top per-AU diffs:\n  {diff_summary}\n\n"
            f"Original assert: {e}"
        )


# ---------------------------------------------------------------------------
# 2. GPU disabled invariant + divergence baseline
# ---------------------------------------------------------------------------


@pytest.mark.tier0
def test_clnf_config_use_gpu_enabled():
    """CLNF_CONFIG['use_gpu'] must be True in production — it is the config the
    v1316 dataset was built with (GPU CLNF on Windows-CUDA), confirmed bit-exact
    (MAE=0 / r=1.0) on every canary x side. Flipping to CPU would diverge from
    the dataset (CPU is close but not bit-exact, mean r 0.86-0.98 vs the v1316
    goldens). See S3 Data Analysis/LIDO_PART_A_WINDOWS_RESULTS.md.

    HISTORY: an older comment pinned use_gpu=False, claiming CPU was "~38%
    better on paralyzed faces" (PyTorch grid_sample vs cv2.warpAffine). That was
    re-validated under pyfaceau 1.3.16 on the paralyzed canaries and REFUTED:
    GPU and CPU correlate equally with the C++ ground truth (GPU-CPU r-delta in
    [-0.009, +0.003] — FP-drift, not a 38% effect). The accuracy reason to
    prefer CPU is gone; GPU is the validated production path (and matches v1316).
    """
    from pyfaceau.config import CLNF_CONFIG

    assert CLNF_CONFIG.get("use_gpu") is True, (
        "CLNF use_gpu flipped to False — this DIVERGES from the v1316 dataset "
        "(built with use_gpu=True; see LIDO_PART_A_WINDOWS_RESULTS.md)."
    )


@pytest.mark.tier1
@pytest.mark.slow
@pytest.mark.requires_video
def test_pyfaceau_gpu_divergence_within_band():
    """Run pyfaceau on IMG_0942 left in CPU mode AND with use_gpu=True
    forced on; assert the per-AU MAE between the two stays within the band
    locked at golden-creation time.

    Today's expected behavior:
      - GPU mode IS BROKEN on paralyzed faces (grid_sample issue)
      - GPU and CPU produce DIFFERENT AU outputs even on normal faces
      - The divergence is bounded — we lock the upper bound today

    Future scenarios this test covers:
      a) Someone "fixes" GPU and divergence drops to ~0 → test fails as
         "good news"; re-lock the band tighter and rejoice
      b) GPU regresses further (e.g. BatchedCEN is broken again) →
         divergence grows beyond the locked upper bound; test fails LOUDLY
      c) CPU regresses → same as (b) since it's a delta test

    Bands are stored at golden/gpu_divergence_baseline.json.
    """
    from pyfaceau.processor import OpenFaceProcessor  # noqa
    from pyfaceau import pipeline as pp_module
    from pyfaceau.config import CLNF_CONFIG

    canary = CANARIES_BY_ID[GPU_BASELINE_CANARY_ID]
    video = canary.video("left")
    if not video.exists():
        pytest.skip(f"missing video at {video}")

    baseline_path = GOLDEN_ROOT / "gpu_divergence_baseline.json"
    if not baseline_path.exists():
        pytest.skip(
            "No gpu_divergence_baseline.json yet — run "
            "`python tests/update_goldens.py --stage gpu_divergence "
            "--reason '...'` first."
        )
    baseline = json.loads(baseline_path.read_text())
    upper_band = baseline["max_acceptable_per_au_mae"]
    observed = baseline.get("observed_per_au_mae", {})

    # CPU run (production config)
    cpu_proc = OpenFaceProcessor(verbose=False)
    cpu_df = _process(cpu_proc, video, MAX_FRAMES)

    # GPU run — flip CLNF_CONFIG temporarily and rebuild Pipeline
    saved_use_gpu = CLNF_CONFIG.get("use_gpu", False)
    try:
        CLNF_CONFIG["use_gpu"] = True
        gpu_proc = OpenFaceProcessor(verbose=False)
        gpu_df = _process(gpu_proc, video, MAX_FRAMES)
    finally:
        CLNF_CONFIG["use_gpu"] = saved_use_gpu

    # Compare per-AU MAE on success rows (frame-paired)
    common = cpu_df["frame"].isin(gpu_df["frame"]) & gpu_df["frame"].isin(cpu_df["frame"])
    cpu_ok = cpu_df.loc[common & (cpu_df["success"] == 1) & (gpu_df["success"] == 1)]
    gpu_ok = gpu_df.loc[common & (cpu_df["success"] == 1) & (gpu_df["success"] == 1)]
    if len(cpu_ok) == 0:
        pytest.skip("No frames where both CPU and GPU succeeded")

    au_cols = [c for c in cpu_df.columns if c.startswith("AU") and c.endswith("_r")]
    per_au_mae: dict[str, float] = {}
    failures: list[str] = []
    for au in au_cols:
        a = cpu_ok[au].astype(float).to_numpy()
        b = gpu_ok[au].astype(float).to_numpy()
        mae = float(np.mean(np.abs(a - b)))
        per_au_mae[au] = mae
        upper = upper_band.get(au, 999.0)
        if mae > upper:
            failures.append(f"{au}: mae={mae:.4f} > locked upper {upper:.4f}")
    if failures:
        delta_summary = "\n  ".join(
            f"{au}: today={per_au_mae[au]:.4f}, locked_max={upper_band.get(au, 'n/a')}"
            for au in sorted(per_au_mae)
        )
        pytest.fail(
            "GPU vs CPU divergence is now OUTSIDE the locked band.\n\n"
            "Either GPU got worse, CPU regressed, or the BatchedCEN "
            "bit-equivalence rewrite broke. Today's per-AU MAE:\n  "
            + delta_summary
            + "\n\nFailures:\n  "
            + "\n  ".join(failures)
        )

    # Negative case: if divergence dropped substantially below the OBSERVED
    # baseline at golden-creation (not the inflated upper_band) for many
    # AUs, that's good news but warrants a band re-lock so we keep tight
    # detection of future regressions.
    if observed:
        substantially_tighter = []
        for au in per_au_mae:
            obs = observed.get(au)
            if obs is None or obs == 0:
                continue
            # threshold: today's MAE < 50% of observed baseline AND the
            # absolute drop is at least 0.001 (filters out floor-precision
            # noise on already-tiny values)
            if per_au_mae[au] < 0.5 * obs and (obs - per_au_mae[au]) >= 0.001:
                substantially_tighter.append((au, per_au_mae[au], obs))
        if len(substantially_tighter) >= 5:
            sample = "\n  ".join(
                f"{au}: today={now:.4f}, baseline={obs:.4f}"
                for au, now, obs in substantially_tighter[:5]
            )
            pytest.fail(
                f"GPU vs CPU divergence dropped substantially below the "
                f"locked baseline for {len(substantially_tighter)} AUs. "
                f"Probably good news (GPU got fixed?) but the band needs to "
                f"be re-locked tighter — re-run `update_goldens.py --stage "
                f"gpu_divergence` and commit.\n\nSample:\n  {sample}"
            )
