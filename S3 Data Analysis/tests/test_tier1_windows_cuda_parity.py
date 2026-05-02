"""Tier 1 — Windows + CUDA parity validation.

Purpose
-------
Confirm that pyfaceau running on Windows with NVIDIA CUDA produces AU
intensities that are consistent with:

    (a) the macOS reference run that powered the manuscript
        (`pyfaceau.parquet` per canary × side), and
    (b) the C++ OpenFace 2.2 ground truth
        (`cpp.parquet` per canary × side).

Numerical identity across (CPU/MPS/CUDA) backends is impossible — different
BLAS kernels and cuDNN algorithms produce floats that differ by ~1e-6. So
this test uses the same Pearson r ≥ band, MAE ≤ band tolerance approach as
`test_tier1_quality_vs_cpp.py`, just with the Windows-CUDA output as the
left-hand side.

How to populate the Windows-CUDA goldens
----------------------------------------
On a Windows machine with the canary corpus mounted at SPLITFACE_BASE and
CUDA available::

    $env:SPLITFACE_BASE = "$env:USERPROFILE/Documents/SplitFace"
    cd "S3 Data Analysis"
    python tests/update_goldens.py --stage windows_cuda_aus --reason "fresh CUDA install"

(Do NOT point SPLITFACE_BASE at iCloud Drive on Windows -- the Windows
iCloud client's files-on-demand mode hangs OpenCV's ffmpeg reader on first
access. Copy the canary subdirs to a non-cloud-synced local path first;
the repo root has fetch_canaries.ps1 to grab them from a Mac over SMB.)

The ``windows_cuda_aus`` stage (in update_goldens.py) runs pyfaceau LIVE on
each canary video using onnxruntime-gpu CUDAExecutionProvider + pyclnf
``use_gpu=True``, and writes
``tests/golden/aus/<id>_<side>/pyfaceau_windows_cuda.parquet`` for each.
The tests below skip cleanly if those parquets are missing — CI on
macOS / Linux runners will not produce them.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import pandas as pd
import pytest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))

from _pipeline_helpers import (  # noqa: E402
    AU_COLUMNS,
    AU_DIFFICULTY,
    compare_au_frames,
)
from conftest import (  # noqa: E402
    GOLDEN_ROOT,
    Canary,
    parametrize_canaries_sides,
)


def _windows_cuda_parquet(canary: Canary, side: str) -> Path:
    return GOLDEN_ROOT / "aus" / f"{canary.id}_{side}" / "pyfaceau_windows_cuda.parquet"


def _mac_pyfaceau_parquet(canary: Canary, side: str) -> Path:
    return GOLDEN_ROOT / "aus" / f"{canary.id}_{side}" / "pyfaceau.parquet"


def _cpp_parquet(canary: Canary, side: str) -> Path:
    return GOLDEN_ROOT / "aus" / f"{canary.id}_{side}" / "cpp.parquet"


def _assert_au_bands(cmp_result, bands, label: str) -> None:
    """Lift the failure formatting from test_tier1_quality_vs_cpp so we get
    the same diagnostic surface area for the Windows-CUDA path."""
    failures: list[str] = []
    for au in AU_COLUMNS:
        difficulty = AU_DIFFICULTY[au]
        if difficulty == "informational":
            continue
        thresh = bands[difficulty]
        r = cmp_result.per_au_pearson.get(au, float("nan"))
        mae = cmp_result.per_au_mae.get(au, float("nan"))
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
    assert not failures, f"{label} AU parity failures:\n  " + "\n  ".join(failures)


@pytest.mark.tier1
@parametrize_canaries_sides(tier=1)
def test_windows_cuda_vs_cpp_ground_truth(canary: Canary, side: str, metric_bands):
    """Manuscript-relevant: Windows-CUDA pyfaceau vs C++ OpenFace 2.2.

    This is the primary validation gate. If this passes, the Windows-CUDA
    build is good enough for clinical use — it tracks the same C++ reference
    that the published model was trained against.
    """
    win = _windows_cuda_parquet(canary, side)
    cpp = _cpp_parquet(canary, side)
    if not win.exists():
        pytest.skip(
            f"Windows-CUDA golden missing for {canary.id} {side} — "
            "regenerate locally on a Windows machine and commit"
        )
    if not cpp.exists():
        pytest.skip(f"C++ ground truth missing for {canary.id} {side}")

    py_df = pd.read_parquet(win).set_index("frame", drop=True)
    cpp_df = pd.read_parquet(cpp).set_index("frame", drop=True)
    cmp = compare_au_frames(py_df, cpp_df)

    bands = metric_bands["stage3_aus"][canary.threshold_bucket]
    _assert_au_bands(cmp, bands, label=f"WIN-CUDA vs C++ ({canary.id} {side})")


@pytest.mark.tier1
@parametrize_canaries_sides(tier=1)
def test_windows_cuda_vs_macos_pyfaceau(canary: Canary, side: str, metric_bands):
    """Cross-platform parity: Windows-CUDA pyfaceau vs macOS pyfaceau golden.

    Tighter than vs-C++ because both sides are the same Python code with
    different numeric backends. We re-use the same band table because the
    'normal' bucket already encodes the per-AU difficulty appropriately —
    if Windows-CUDA passes vs C++, this should pass comfortably.
    """
    win = _windows_cuda_parquet(canary, side)
    mac = _mac_pyfaceau_parquet(canary, side)
    if not win.exists():
        pytest.skip(
            f"Windows-CUDA golden missing for {canary.id} {side} — "
            "regenerate locally on a Windows machine and commit"
        )
    if not mac.exists():
        pytest.skip(f"macOS pyfaceau golden missing for {canary.id} {side}")

    win_df = pd.read_parquet(win).set_index("frame", drop=True)
    mac_df = pd.read_parquet(mac).set_index("frame", drop=True)
    cmp = compare_au_frames(win_df, mac_df)

    # 'normal' bucket regardless of severity — we're comparing the same code,
    # not measuring clinical accuracy here.
    bands = metric_bands["stage3_aus"]["normal"]
    _assert_au_bands(cmp, bands, label=f"WIN-CUDA vs macOS ({canary.id} {side})")
