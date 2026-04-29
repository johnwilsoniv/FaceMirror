"""Stage-by-stage helpers used by both update_goldens.py and test files.

Why this module exists:
  - The tests want to load pyfaceau outputs and C++ outputs in the same shape
    (per-frame DataFrames with a known column subset). Different formats need
    different reading code; this module hides that.
  - The golden builder wants to call the same loaders and write parquets
    deterministically. Same readers + same write logic = goldens that the test
    suite reads with byte-identical comparisons.
  - The metric calculations (per-AU Pearson r, per-landmark MAE, etc.) live
    here once instead of being copy-pasted across multiple test files.
"""

from __future__ import annotations

import hashlib
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

# Standard 17 AUs that both pyfaceau and C++ output (regression form)
AU_COLUMNS: list[str] = [
    "AU01_r", "AU02_r", "AU04_r", "AU05_r", "AU06_r", "AU07_r",
    "AU09_r", "AU10_r", "AU12_r", "AU14_r", "AU15_r", "AU17_r",
    "AU20_r", "AU23_r", "AU25_r", "AU26_r", "AU45_r",
]

# Per-AU difficulty buckets — derived from RETRAINING_REPRODUCIBILITY.md
# (AU17/AU25/AU26 have lowest pearson r vs C++ gold; AU01/02/12/45 are easiest).
AU_DIFFICULTY: dict[str, str] = {
    "AU01_r": "easy", "AU02_r": "easy", "AU12_r": "easy", "AU45_r": "easy",
    "AU04_r": "medium", "AU06_r": "medium", "AU07_r": "medium", "AU10_r": "medium",
    "AU14_r": "hard", "AU15_r": "hard", "AU17_r": "hard",
    "AU20_r": "hard", "AU23_r": "hard", "AU25_r": "hard", "AU26_r": "hard",
    # AU05 / AU09 are sparse and informational only
    "AU05_r": "informational", "AU09_r": "informational",
}

# 68-landmark region groupings for region-stratified MAE metrics
LANDMARK_REGIONS: dict[str, list[int]] = {
    "contour": list(range(0, 17)),
    "brows":   list(range(17, 27)),
    "nose":    list(range(27, 36)),
    "eyes":    list(range(36, 48)),
    "mouth":   list(range(48, 68)),
}


# ---------------------------------------------------------------------------
# Per-frame loaders
# ---------------------------------------------------------------------------


def _dedupe_by_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Some C++ CSVs have multiple rows per frame (face_id≠1 entries when the
    detector re-initializes). Keep the FIRST row per frame — that's the primary
    face. Without this, .loc[common_index] explodes to >n rows."""
    if df.index.is_unique:
        return df
    return df[~df.index.duplicated(keep="first")]


def load_pyfaceau_aus(csv_path: Path) -> pd.DataFrame:
    """Load pyfaceau per-frame AU CSV. Returns DataFrame indexed by frame
    with columns: success, AU01_r..AU45_r."""
    df = pd.read_csv(csv_path)
    df.columns = [c.strip() for c in df.columns]
    keep = ["frame", "success"] + [c for c in AU_COLUMNS if c in df.columns]
    df = df[keep].copy()
    df = df.set_index("frame", drop=True)
    return _dedupe_by_frame(df)


def load_cpp_aus(csv_path: Path) -> pd.DataFrame:
    """Load C++ per-frame AU CSV (also has landmarks/pose, but we drop those
    here)."""
    df = pd.read_csv(csv_path)
    df.columns = [c.strip() for c in df.columns]
    keep = ["frame", "success"] + [c for c in AU_COLUMNS if c in df.columns]
    df = df[keep].copy()
    df = df.set_index("frame", drop=True)
    return _dedupe_by_frame(df)


def load_cpp_landmarks(csv_path: Path) -> pd.DataFrame:
    """Load 68 2D landmarks per frame from a C++ AU CSV. Returns DataFrame
    indexed by frame with columns x_0..x_67, y_0..y_67."""
    df = pd.read_csv(csv_path)
    df.columns = [c.strip() for c in df.columns]
    landmark_cols = [c for c in df.columns if c.startswith(("x_", "y_"))]
    if not landmark_cols:
        raise ValueError(f"{csv_path} has no x_*/y_* landmark columns")
    df = df[["frame", "success"] + landmark_cols].copy()
    df = df.set_index("frame", drop=True)
    return _dedupe_by_frame(df)


def load_cpp_head_pose(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    df.columns = [c.strip() for c in df.columns]
    pose_cols = [c for c in df.columns if c.startswith("pose_")]
    df = df[["frame", "success"] + pose_cols].copy()
    df = df.set_index("frame", drop=True)
    return _dedupe_by_frame(df)


# ---------------------------------------------------------------------------
# Per-frame metrics
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class AUComparison:
    n_frames_compared: int
    per_au_pearson: dict[str, float]   # NaN if either side is constant
    per_au_mae: dict[str, float]
    per_au_sparsity_py: dict[str, float]   # fraction of frames AU>0.5
    per_au_sparsity_cpp: dict[str, float]


def compare_au_frames(py_df: pd.DataFrame, cpp_df: pd.DataFrame) -> AUComparison:
    """Inner-join on frame, compute per-AU Pearson r + MAE on success rows."""
    common_frames = py_df.index.intersection(cpp_df.index)
    py = py_df.loc[common_frames]
    cpp = cpp_df.loc[common_frames]
    # Restrict to frames where BOTH extractors succeeded
    success_mask = (py["success"].astype(int) == 1) & (cpp["success"].astype(int) == 1)
    py_ok = py.loc[success_mask]
    cpp_ok = cpp.loc[success_mask]

    per_pearson: dict[str, float] = {}
    per_mae: dict[str, float] = {}
    per_spar_py: dict[str, float] = {}
    per_spar_cpp: dict[str, float] = {}
    for au in AU_COLUMNS:
        if au not in py_ok.columns or au not in cpp_ok.columns:
            continue
        a = pd.to_numeric(py_ok[au], errors="coerce").to_numpy()
        b = pd.to_numeric(cpp_ok[au], errors="coerce").to_numpy()
        valid = ~(np.isnan(a) | np.isnan(b))
        if valid.sum() < 5:
            per_pearson[au] = float("nan")
            per_mae[au] = float("nan")
        else:
            av, bv = a[valid], b[valid]
            if av.std() == 0 or bv.std() == 0:
                per_pearson[au] = float("nan")
            else:
                per_pearson[au] = float(np.corrcoef(av, bv)[0, 1])
            per_mae[au] = float(np.mean(np.abs(av - bv)))
        per_spar_py[au] = float(np.mean(a[~np.isnan(a)] > 0.5)) if (~np.isnan(a)).any() else float("nan")
        per_spar_cpp[au] = float(np.mean(b[~np.isnan(b)] > 0.5)) if (~np.isnan(b)).any() else float("nan")

    return AUComparison(
        n_frames_compared=int(success_mask.sum()),
        per_au_pearson=per_pearson,
        per_au_mae=per_mae,
        per_au_sparsity_py=per_spar_py,
        per_au_sparsity_cpp=per_spar_cpp,
    )


@dataclass(frozen=True)
class LandmarkComparison:
    n_frames_compared: int
    mean_per_landmark_px: float       # avg over all 68 landmarks, all frames
    p95_per_landmark_px: float
    max_per_landmark_px: float
    per_region_mean_px: dict[str, float]


def compare_landmark_frames(py_df: pd.DataFrame, cpp_df: pd.DataFrame) -> LandmarkComparison:
    """Compare per-frame 68 landmarks. Both DataFrames must have x_0..x_67 and
    y_0..y_67 columns indexed by frame.
    """
    common_frames = py_df.index.intersection(cpp_df.index)
    py = py_df.loc[common_frames]
    cpp = cpp_df.loc[common_frames]
    success_mask = (py["success"].astype(int) == 1) & (cpp["success"].astype(int) == 1)
    py_ok = py.loc[success_mask]
    cpp_ok = cpp.loc[success_mask]
    n = len(py_ok)
    if n == 0:
        return LandmarkComparison(0, float("nan"), float("nan"), float("nan"), {})

    # Build (n_frames, 68, 2) arrays
    def _xy(df: pd.DataFrame) -> np.ndarray:
        x = df[[f"x_{i}" for i in range(68)]].to_numpy()
        y = df[[f"y_{i}" for i in range(68)]].to_numpy()
        return np.stack([x, y], axis=-1)  # (n, 68, 2)

    a = _xy(py_ok)
    b = _xy(cpp_ok)
    diffs = np.linalg.norm(a - b, axis=-1)  # (n, 68)
    per_region: dict[str, float] = {}
    for region, idxs in LANDMARK_REGIONS.items():
        per_region[region] = float(diffs[:, idxs].mean())
    return LandmarkComparison(
        n_frames_compared=n,
        mean_per_landmark_px=float(diffs.mean()),
        p95_per_landmark_px=float(np.percentile(diffs, 95)),
        max_per_landmark_px=float(diffs.max()),
        per_region_mean_px=per_region,
    )


# ---------------------------------------------------------------------------
# Engineered features + predictions (Stages 5/6a)
# ---------------------------------------------------------------------------


def prepare_mid_features(combined_csv: Path) -> tuple[pd.DataFrame, np.ndarray, pd.DataFrame]:
    """Run prepare_data_generalized for the Mid zone using combined_csv as input.
    Returns (features_df, targets, metadata_df).

    Note: this momentarily monkey-patches paralysis_config.INPUT_FILES to point
    at combined_csv, calls the helper, then restores. Caller should be okay
    with that — same approach used by the H1 driver scripts.
    """
    import paralysis_config
    from paralysis_utils import prepare_data_generalized

    saved = paralysis_config.INPUT_FILES.get("results_csv")
    paralysis_config.INPUT_FILES["results_csv"] = str(combined_csv)
    try:
        feats, y, meta = prepare_data_generalized(
            zone_key="mid",
            results_file_path=str(combined_csv),
            expert_file_path=paralysis_config.INPUT_FILES.get("expert_key_csv"),
        )
    finally:
        paralysis_config.INPUT_FILES["results_csv"] = saved
    return feats, y, meta


def saved_jan1_predict(features_df: pd.DataFrame, model_bundle: dict) -> np.ndarray:
    """Apply scaler.transform → model.predict using the loaded Jan 1 model bundle."""
    feats = model_bundle["features"]
    missing = [f for f in feats if f not in features_df.columns]
    if missing:
        raise KeyError(f"missing features in input frame: {missing}")
    X = features_df[feats]
    X_scaled = model_bundle["scaler"].transform(X)
    return model_bundle["model"].predict(X_scaled)


# ---------------------------------------------------------------------------
# Determinism utilities
# ---------------------------------------------------------------------------


def file_sha256(path: Path) -> str:
    """SHA256 hex of a file's bytes — used to lock golden files."""
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def write_checksums(golden_dir: Path) -> dict[str, str]:
    """Walk golden_dir, compute SHA256 for every file (excluding checksums.json
    and golden_history.md), write to checksums.json, return the mapping.
    """
    sums: dict[str, str] = {}
    for p in sorted(golden_dir.rglob("*")):
        if not p.is_file():
            continue
        if p.name in ("checksums.json", "golden_history.md", ".gitkeep"):
            continue
        rel = p.relative_to(golden_dir).as_posix()
        sums[rel] = file_sha256(p)
    out = golden_dir / "checksums.json"
    out.write_text(json.dumps(sums, indent=2, sort_keys=True) + "\n")
    return sums


def stable_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Return a DataFrame with deterministic column order + reset index (so
    parquet writes are byte-identical on re-runs)."""
    cols = sorted(df.columns)
    out = df[cols].copy()
    if isinstance(out.index, pd.RangeIndex):
        return out
    return out.reset_index()


# ---------------------------------------------------------------------------
# Misc
# ---------------------------------------------------------------------------


def ensure_s3_on_path(s3_root: Path) -> None:
    """Insert S3 Data Analysis on sys.path so the unpickled
    OrdinalBinaryClassifier from saved Jan 1 model can resolve its module."""
    s = str(s3_root)
    if s not in sys.path:
        sys.path.insert(0, s)
