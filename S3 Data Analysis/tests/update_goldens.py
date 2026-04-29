"""Generate (or regenerate) the golden reference files used by the tests.

Usage:
    python tests/update_goldens.py --stage all --reason "initial baseline"
    python tests/update_goldens.py --stage aus --reason "pyfaceau v0.3.2 upgrade"

Stages
------
    aus           — per-canary frame-paired AU CSVs (pyfaceau + C++)
    landmarks     — per-canary frame-paired landmarks (C++ only for now;
                    pyfaceau-side instrumentation is sub-PR 2)
    peak_frames   — locked peak frames per (canary, side, action)
    features      — engineered Mid Face features for canary patients
                    (both pyfaceau and C++ sources)
    predictions   — saved Jan 1 model predictions on canary patients
    test_split    — locked test patient IDs at random_state=42
    metric_bands  — initial metric thresholds calibrated from current values
                    + headroom (this should be reviewed manually after first
                    write — see comment in the function)
    all           — runs every stage above

Determinism
-----------
The script writes parquet (zstd compressed) for tabular goldens, json for
small key/value goldens, and npy for arrays. Every run produces byte-identical
output for the same input — no embedded timestamps in the files themselves.
SHA256 digests of every golden file are written to checksums.json so test
suites can fail loudly if anyone hand-edits a golden.

After every stage, an entry is appended to golden_history.md:
    - run timestamp (UTC)
    - git commit SHA at write time
    - argparse args (especially --reason)
    - SHA256 of pip-freeze of the active venv (so we can detect library drift)
"""

from __future__ import annotations

import argparse
import datetime as dt
import importlib
import json
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

# Path bootstrapping so we can import pipeline modules + the tests package
HERE = Path(__file__).resolve().parent
S3_ROOT = HERE.parent
sys.path.insert(0, str(S3_ROOT))
sys.path.insert(0, str(HERE))

from _pipeline_helpers import (  # noqa: E402
    AU_COLUMNS,
    AU_DIFFICULTY,
    LANDMARK_REGIONS,
    compare_au_frames,
    compare_bbox_frames,
    compare_landmark_frames,
    derive_bbox_from_landmarks,
    file_sha256,
    load_cpp_aus,
    load_cpp_landmarks,
    load_pyfaceau_aus,
    load_pyfaceau_landmarks,
    prepare_mid_features,
    saved_jan1_predict,
    stable_dataframe,
    write_checksums,
)
from conftest import (  # noqa: E402
    CANARIES,
    CANARY_DATA_ROOT,
    CPP_COMBINED_CSV,
    GOLDEN_ROOT,
    JAN1_MODEL_DIR,
    PYFACEAU_COMBINED_CSV,
    S3_ROOT as CONFTEST_S3_ROOT,
)

assert CONFTEST_S3_ROOT == S3_ROOT, "conftest path mismatch — fix conftest.py"


# ---------------------------------------------------------------------------
# Stage implementations
# ---------------------------------------------------------------------------


def stage_aus(args: argparse.Namespace) -> list[Path]:
    """Per-canary, per-side: write parquet snapshot of FULL pyfaceau AU CSV
    and FULL C++ AU CSV (deduplicated by frame, see _dedupe_by_frame).

    These goldens enable two distinct test patterns:
      - Tier 0 byte-equality: current CSV must reproduce the snapshot exactly
        (catches anyone modifying pyfaceau in a way that affects existing
        recorded outputs). Strict shape + value equality.
      - Tier 1 quality: live compare_au_frames against current outputs to
        get per-AU Pearson r and MAE; pass/fail bands live in metric_bands.yaml,
        not in these snapshots.

    Inner-joining at golden-write time would make Tier 0 brittle (any frame
    that one extractor sees but the other doesn't would force a golden
    refresh). So we save the full snapshots and let the Tier 1 comparator
    handle the join.
    """
    written: list[Path] = []
    out_dir = GOLDEN_ROOT / "aus"
    out_dir.mkdir(parents=True, exist_ok=True)
    for c in CANARIES:
        for side in ("left", "right"):
            py_csv = c.pyfaceau_csv(side)
            cpp_csv = c.cpp_csv(side)
            if not py_csv.exists():
                print(f"  SKIP {c.id} {side}: pyfaceau CSV missing at {py_csv}")
                continue
            if not cpp_csv.exists():
                print(f"  SKIP {c.id} {side}: C++ CSV missing at {cpp_csv}")
                continue
            py = load_pyfaceau_aus(py_csv)
            cpp = load_cpp_aus(cpp_csv)

            sub = out_dir / f"{c.id}_{side}"
            sub.mkdir(exist_ok=True)
            py_out = sub / "pyfaceau.parquet"
            cpp_out = sub / "cpp.parquet"
            stable_dataframe(py.reset_index()).to_parquet(py_out, index=False, compression="zstd")
            stable_dataframe(cpp.reset_index()).to_parquet(cpp_out, index=False, compression="zstd")
            written.extend([py_out, cpp_out])

            # quick summary so the operator can sanity-check the run
            cmp = compare_au_frames(py, cpp)
            print(
                f"  {c.id:>22s} {side:5s}  n={cmp.n_frames_compared:4d}  "
                f"mid_AUs r=AU06:{cmp.per_au_pearson.get('AU06_r', float('nan')):+.3f} "
                f"AU07:{cmp.per_au_pearson.get('AU07_r', float('nan')):+.3f} "
                f"AU45:{cmp.per_au_pearson.get('AU45_r', float('nan')):+.3f}"
            )
    return written


def stage_landmarks(args: argparse.Namespace) -> list[Path]:
    """Write C++ landmarks per canary × side. The pyfaceau landmark parquet
    is produced by `instrument_pyfaceau.py` (slow, run separately); this
    stage just snapshots the C++ side and reports whether the pyfaceau
    counterpart exists.
    """
    written: list[Path] = []
    out_dir = GOLDEN_ROOT / "landmarks"
    out_dir.mkdir(parents=True, exist_ok=True)
    have_pyfaceau = 0
    missing_pyfaceau: list[str] = []
    for c in CANARIES:
        for side in ("left", "right"):
            cpp_csv = c.cpp_csv(side)
            if not cpp_csv.exists():
                print(f"  SKIP {c.id} {side}: C++ CSV missing")
                continue
            try:
                cpp_lm = load_cpp_landmarks(cpp_csv)
            except ValueError as e:
                print(f"  SKIP {c.id} {side}: {e}")
                continue
            sub = out_dir / f"{c.id}_{side}"
            sub.mkdir(exist_ok=True)
            cpp_out = sub / "cpp.parquet"
            stable_dataframe(cpp_lm.reset_index()).to_parquet(cpp_out, index=False, compression="zstd")
            written.append(cpp_out)
            py_out = sub / "pyfaceau.parquet"
            if py_out.exists():
                have_pyfaceau += 1
                py_lm = load_pyfaceau_landmarks(py_out)
                # quick sanity report
                cmp = compare_landmark_frames(
                    py_lm[["success"] + [f"x_{i}" for i in range(68)] + [f"y_{i}" for i in range(68)]],
                    derive_bbox_from_landmarks(cpp_lm)[["success"] + [f"x_{i}" for i in range(68)] + [f"y_{i}" for i in range(68)]]
                ) if "x_0" in py_lm.columns else None
                marker = f"  py mean={cmp.mean_per_landmark_px:.2f}px max={cmp.max_per_landmark_px:.2f}px" if cmp else ""
            else:
                missing_pyfaceau.append(f"{c.id}_{side}")
                marker = "  (no pyfaceau side — run instrument_pyfaceau.py)"
            print(f"  {c.id:>22s} {side:5s}  cpp landmarks: {cpp_lm.shape}{marker}")
    if missing_pyfaceau:
        print(f"\n  NOTE: {len(missing_pyfaceau)} pyfaceau landmark parquets missing:")
        print(f"    {missing_pyfaceau[:6]}{'...' if len(missing_pyfaceau) > 6 else ''}")
        print(f"    Generate with: python tests/instrument_pyfaceau.py --all")
    print(f"\n  Have pyfaceau-side parquets: {have_pyfaceau}/20 ({100*have_pyfaceau/20:.0f}%)")
    return written


def stage_peak_frames(args: argparse.Namespace) -> list[Path]:
    """Snapshot per-action peak frames from current combined_results.csv (pyfaceau)
    and combined_results_OF_v2.csv (C++) for the canary patients."""
    written: list[Path] = []
    if not PYFACEAU_COMBINED_CSV.exists():
        print(f"  SKIP: {PYFACEAU_COMBINED_CSV} missing — run main.py --batch first")
        return written

    py_combined = pd.read_csv(PYFACEAU_COMBINED_CSV, low_memory=False)
    cpp_combined = (
        pd.read_csv(CPP_COMBINED_CSV, low_memory=False)
        if CPP_COMBINED_CSV.exists()
        else None
    )
    canary_ids = {c.id for c in CANARIES}

    def _extract(df: pd.DataFrame) -> dict:
        out: dict = {}
        for _, row in df.iterrows():
            pid = row.get("Patient ID")
            if pid not in canary_ids:
                continue
            peaks = {}
            for col in df.columns:
                if col.endswith("_Max Frame"):
                    action = col.split("_")[0]
                    val = row[col]
                    peaks[action] = None if pd.isna(val) else int(val)
            out[pid] = peaks
        return out

    obj = {
        "pyfaceau_source_csv": PYFACEAU_COMBINED_CSV.as_posix(),
        "cpp_source_csv": CPP_COMBINED_CSV.as_posix() if cpp_combined is not None else None,
        "pyfaceau_peak_frames": _extract(py_combined),
        "cpp_peak_frames": _extract(cpp_combined) if cpp_combined is not None else {},
    }
    out_path = GOLDEN_ROOT / "peak_frames.json"
    out_path.write_text(json.dumps(obj, indent=2, sort_keys=True) + "\n")
    written.append(out_path)
    print(f"  Locked peak frames for {len(obj['pyfaceau_peak_frames'])} canaries (pyfaceau)")
    return written


def stage_features(args: argparse.Namespace) -> list[Path]:
    """Run prepare_data_generalized for the Mid zone on both pyfaceau and C++
    combined_results CSVs; save the engineered-features rows for the canary
    patients (both sides) as parquet."""
    written: list[Path] = []

    feats_py, y_py, meta_py = prepare_mid_features(PYFACEAU_COMBINED_CSV)
    feats_cpp, y_cpp, meta_cpp = (
        prepare_mid_features(CPP_COMBINED_CSV)
        if CPP_COMBINED_CSV.exists()
        else (None, None, None)
    )

    canary_ids = {c.id for c in CANARIES}

    def _extract_canary_rows(features: pd.DataFrame, meta: pd.DataFrame, y: np.ndarray):
        meta = meta.reset_index(drop=True)
        features = features.reset_index(drop=True)
        mask = meta["Patient ID"].isin(canary_ids)
        return (
            features.loc[mask].copy(),
            meta.loc[mask].copy(),
            y[mask.values],
        )

    py_X, py_meta, py_y = _extract_canary_rows(feats_py, meta_py, y_py)
    py_meta = py_meta.reset_index(drop=True)
    py_X = py_X.reset_index(drop=True)
    py_X.insert(0, "_patient_id", py_meta["Patient ID"].values)
    py_X.insert(1, "_side", py_meta["Side"].values)
    py_X.insert(2, "_y", py_y)
    out = GOLDEN_ROOT / "features_pyfaceau.parquet"
    stable_dataframe(py_X).to_parquet(out, index=False, compression="zstd")
    written.append(out)
    print(f"  features_pyfaceau: {py_X.shape}")

    if feats_cpp is not None:
        cpp_X, cpp_meta, cpp_y = _extract_canary_rows(feats_cpp, meta_cpp, y_cpp)
        cpp_meta = cpp_meta.reset_index(drop=True)
        cpp_X = cpp_X.reset_index(drop=True)
        cpp_X.insert(0, "_patient_id", cpp_meta["Patient ID"].values)
        cpp_X.insert(1, "_side", cpp_meta["Side"].values)
        cpp_X.insert(2, "_y", cpp_y)
        out = GOLDEN_ROOT / "features_cpp.parquet"
        stable_dataframe(cpp_X).to_parquet(out, index=False, compression="zstd")
        written.append(out)
        print(f"  features_cpp:      {cpp_X.shape}")

    return written


def stage_predictions(args: argparse.Namespace) -> list[Path]:
    """Apply the saved Jan 1 Mid Face model to canary features (pyfaceau and
    C++) and lock the per-(patient,side) predictions."""
    import joblib

    if not JAN1_MODEL_DIR.exists():
        print(f"  SKIP: Jan 1 model dir not present at {JAN1_MODEL_DIR}")
        return []
    bundle = {
        "model": joblib.load(JAN1_MODEL_DIR / "mid_face_model.pkl"),
        "scaler": joblib.load(JAN1_MODEL_DIR / "mid_face_scaler.pkl"),
        "features": [
            line.strip()
            for line in (JAN1_MODEL_DIR / "mid_face_features.list").read_text().splitlines()
            if line.strip()
        ],
    }
    written: list[Path] = []
    for source_label, parquet_name in (("pyfaceau", "features_pyfaceau.parquet"), ("cpp", "features_cpp.parquet")):
        in_path = GOLDEN_ROOT / parquet_name
        if not in_path.exists():
            print(f"  SKIP {source_label}: {in_path} missing — run --stage features first")
            continue
        df = pd.read_parquet(in_path)
        meta_cols = ["_patient_id", "_side", "_y"]
        feature_df = df.drop(columns=meta_cols)
        y_pred = saved_jan1_predict(feature_df, bundle)
        obj = {
            "patients": df["_patient_id"].tolist(),
            "sides": df["_side"].tolist(),
            "y_true": df["_y"].astype(int).tolist(),
            "y_pred": y_pred.astype(int).tolist(),
        }
        out_path = GOLDEN_ROOT / f"predictions_{source_label}.json"
        out_path.write_text(json.dumps(obj, indent=2, sort_keys=True) + "\n")
        written.append(out_path)
        agree = int((y_pred == df["_y"].values).sum())
        print(f"  predictions_{source_label}: {agree}/{len(y_pred)} agree with expert")
    return written


def stage_test_split(args: argparse.Namespace) -> list[Path]:
    """Lock the random_state=42 test partition's patient/side identifiers
    (full dataset, not just canaries — this is what the training pipeline
    actually splits over)."""
    from sklearn.model_selection import train_test_split

    feats, y, meta = prepare_mid_features(PYFACEAU_COMBINED_CSV)
    X_tr, X_te, y_tr, y_te, m_tr, m_te = train_test_split(
        feats, y, meta, test_size=0.25, random_state=42, stratify=y
    )
    obj = {
        "test_size": 0.25,
        "random_state": 42,
        "stratify": "by target",
        "n_total": int(len(feats)),
        "n_train": int(len(X_tr)),
        "n_test": int(len(X_te)),
        "test_patients_sides": sorted(
            f"{r['Patient ID']}|{r['Side']}" for _, r in m_te.iterrows()
        ),
        "test_targets": [int(v) for v in y_te],
    }
    out_path = GOLDEN_ROOT / "test_split_seed42.json"
    out_path.write_text(json.dumps(obj, indent=2, sort_keys=True) + "\n")
    print(f"  test_split_seed42: {obj['n_test']} samples in test set")
    return [out_path]


def stage_metric_bands(args: argparse.Namespace) -> list[Path]:
    """Empirically calibrate per-stage thresholds from the current golden
    files. Bands = "today's observed values + headroom" so freshly-generated
    goldens immediately pass Tier 1. Future regressions below today's level
    fail the suite.

    Per-stage calibration policy:
      Stage 3 (AUs): for each (severity, difficulty) bucket, take the
        WORST observed Pearson r across all canary × side × AU cells in that
        bucket, subtract a small relaxation (-0.05). Take WORST observed MAE,
        add relaxation (+0.10). NaN r values are excluded (constant-AU
        cases are not regressions).
      Stage 4 (peak frames): worst observed agreement fraction - 0.05;
        max_abs_frame_diff held at 3.
      Stage 5 (features): worst observed |py - cpp| across all canary rows +
        0.10 (per-canary-row hard cap). Soft warning at 0.6× the hard cap.
        max_features_above_warning at the worst observed count + 2.
      Stage 6a: worst observed inference agreement - 0.05 (or 0.5 floor,
        whichever higher).
      Stage 6b: not auto-calibrated (sub-PR 3 — needs retrain measurements).

    Stages 1-2 (bbox/landmarks) use placeholder bands until pyfaceau-side
    capture lands (sub-PR 2).
    """
    import yaml as _yaml

    out: dict = {
        "_meta": {
            "policy": (
                "Initial bands auto-calibrated from current goldens (worst "
                "observed value + headroom). Updated whenever update_goldens.py "
                "--stage metric_bands runs. The framework catches FUTURE drift "
                "below today's level. Tightening toward the manuscript-era gold "
                "standard is a deliberate separate task — see "
                "RETRAINING_REPRODUCIBILITY.md."
            ),
            "calibration_relaxations": {
                "pearson_r_subtract": 0.05,
                "mae_add": 0.10,
                "feature_diff_add": 0.10,
                "agreement_subtract": 0.05,
                "max_features_above_warning_add": 2,
            },
        },
        "stage1_bbox": {
            "_status": "calibrated empirically when pyfaceau parquets exist; placeholder bands otherwise",
            "normal":    {"median_iou_min": 0.85, "median_center_diff_max_px": 5.0, "success_rate_min": 0.99},
            "paralyzed": {"median_iou_min": 0.80, "median_center_diff_max_px": 7.0, "success_rate_min": 0.95},
        },
        "stage2_landmarks": {
            "_status": "calibrated empirically when pyfaceau parquets exist; placeholder bands otherwise",
            "normal":    {"mean_max_px": 2.5, "p95_max_px": 6.0, "max_max_px": 10.0},
            "paralyzed": {"mean_max_px": 4.5, "p95_max_px": 10.0, "max_max_px": 15.0},
        },
    }

    # ------ Stages 1 + 2 calibration (only if pyfaceau parquets exist) ------
    landmarks_dir = GOLDEN_ROOT / "landmarks"
    bbox_obs: dict[str, list] = {"normal": [], "paralyzed": []}
    lm_obs: dict[str, list] = {"normal": [], "paralyzed": []}
    if landmarks_dir.exists():
        for c in CANARIES:
            for side in ("left", "right"):
                sub = landmarks_dir / f"{c.id}_{side}"
                py_p = sub / "pyfaceau.parquet"
                cpp_p = sub / "cpp.parquet"
                if not (py_p.exists() and cpp_p.exists()):
                    continue
                py = load_pyfaceau_landmarks(py_p)
                cpp = pd.read_parquet(cpp_p).set_index("frame", drop=True)
                cpp = cpp[~cpp.index.duplicated(keep="first")]
                # Bbox: pyfaceau has bbox cols directly; C++ doesn't, derive from landmarks
                if all(c_ in py.columns for c_ in ["bbox_x1", "bbox_y1", "bbox_x2", "bbox_y2"]):
                    cpp_with_bbox = derive_bbox_from_landmarks(cpp)
                    bcmp = compare_bbox_frames(py, cpp_with_bbox)
                    bbox_obs[c.threshold_bucket].append(bcmp)
                # Landmarks: both have x_0..x_67/y_0..y_67
                if "x_0" in py.columns and "x_0" in cpp.columns:
                    lcmp = compare_landmark_frames(py, cpp)
                    lm_obs[c.threshold_bucket].append(lcmp)
    if any(bbox_obs.values()):
        out["stage1_bbox"] = {
            "_status": "auto-calibrated from pyfaceau-side parquets present in tests/golden/landmarks/",
        }
        for sev in ("normal", "paralyzed"):
            obs = bbox_obs[sev]
            if not obs:
                continue
            worst_iou_med = min(o.median_iou for o in obs if not np.isnan(o.median_iou))
            worst_p10_iou = min(o.p10_iou for o in obs if not np.isnan(o.p10_iou))
            worst_med_center = max(o.median_center_diff_px for o in obs if not np.isnan(o.median_center_diff_px))
            worst_p90_center = max(o.p90_center_diff_px for o in obs if not np.isnan(o.p90_center_diff_px))
            worst_succ = min(o.success_rate_py for o in obs)
            out["stage1_bbox"][sev] = {
                "median_iou_min": float(round(max(0.0, worst_iou_med - 0.05), 3)),
                "p10_iou_min": float(round(max(0.0, worst_p10_iou - 0.10), 3)),
                "median_center_diff_max_px": float(round(worst_med_center + 2.0, 2)),
                "p90_center_diff_max_px": float(round(worst_p90_center + 5.0, 2)),
                "success_rate_min": float(round(max(0.0, worst_succ - 0.02), 3)),
                "_observed_worst_median_iou": float(round(worst_iou_med, 3)),
                "_observed_worst_med_center_diff": float(round(worst_med_center, 2)),
                "_observed_worst_success_rate": float(round(worst_succ, 3)),
                "_n_canaries": len(obs),
            }
    if any(lm_obs.values()):
        out["stage2_landmarks"] = {
            "_status": "auto-calibrated from pyfaceau-side parquets present in tests/golden/landmarks/",
        }
        for sev in ("normal", "paralyzed"):
            obs = lm_obs[sev]
            if not obs:
                continue
            worst_mean = max(o.mean_per_landmark_px for o in obs if not np.isnan(o.mean_per_landmark_px))
            worst_p95 = max(o.p95_per_landmark_px for o in obs if not np.isnan(o.p95_per_landmark_px))
            worst_max = max(o.max_per_landmark_px for o in obs if not np.isnan(o.max_per_landmark_px))
            out["stage2_landmarks"][sev] = {
                "mean_max_px": float(round(worst_mean + 1.0, 2)),
                "p95_max_px": float(round(worst_p95 + 2.0, 2)),
                "max_max_px": float(round(worst_max + 5.0, 2)),
                "_observed_worst_mean_px": float(round(worst_mean, 2)),
                "_observed_worst_p95_px": float(round(worst_p95, 2)),
                "_observed_worst_max_px": float(round(worst_max, 2)),
                "_n_canaries": len(obs),
            }

    # ------ Stage 3 calibration ------
    aus_dir = GOLDEN_ROOT / "aus"
    observed: dict[tuple[str, str], dict[str, list[float]]] = {}
    if aus_dir.exists():
        for c in CANARIES:
            for side in ("left", "right"):
                py_path = aus_dir / f"{c.id}_{side}" / "pyfaceau.parquet"
                cpp_path = aus_dir / f"{c.id}_{side}" / "cpp.parquet"
                if not (py_path.exists() and cpp_path.exists()):
                    continue
                py = pd.read_parquet(py_path).set_index("frame", drop=True)
                cpp = pd.read_parquet(cpp_path).set_index("frame", drop=True)
                cmp = compare_au_frames(py, cpp)
                bucket = (c.threshold_bucket, "")  # severity bucket
                for au in AU_COLUMNS:
                    diff = AU_DIFFICULTY[au]
                    if diff == "informational":
                        continue
                    key = (c.threshold_bucket, diff)
                    observed.setdefault(key, {"r": [], "mae": []})
                    r = cmp.per_au_pearson.get(au, float("nan"))
                    mae = cmp.per_au_mae.get(au, float("nan"))
                    if not np.isnan(r):
                        observed[key]["r"].append(r)
                    if not np.isnan(mae):
                        observed[key]["mae"].append(mae)
    stage3 = {"normal": {}, "paralyzed": {}}
    for sev in ("normal", "paralyzed"):
        for diff in ("easy", "medium", "hard"):
            d = observed.get((sev, diff), {"r": [], "mae": []})
            if d["r"]:
                worst_r = min(d["r"])
                worst_mae = max(d["mae"])
                stage3[sev][diff] = {
                    "pearson_r_min": float(round(worst_r - 0.05, 3)),
                    "mae_max": float(round(worst_mae + 0.10, 3)),
                    "_observed_worst_r": float(round(worst_r, 3)),
                    "_observed_worst_mae": float(round(worst_mae, 3)),
                    "_n_samples": len(d["r"]),
                }
            else:
                stage3[sev][diff] = {
                    "pearson_r_min": -1.0, "mae_max": 999.0,
                    "_status": "no observations",
                }
        stage3[sev]["informational"] = {"pearson_r_min": -1.0, "mae_max": 999.0}
    out["stage3_aus"] = stage3

    # ------ Stage 5 calibration (feature drift across canary rows) ------
    py_path = GOLDEN_ROOT / "features_pyfaceau.parquet"
    cpp_path = GOLDEN_ROOT / "features_cpp.parquet"
    if py_path.exists() and cpp_path.exists():
        pdf = pd.read_parquet(py_path)
        cdf = pd.read_parquet(cpp_path)
        merged = pdf.merge(cdf, on=["_patient_id", "_side"], suffixes=("_py", "_cpp"))
        feat_cols = [c for c in pdf.columns if c not in {"_patient_id", "_side", "_y"}]
        # per-row: worst absolute diff over all features
        worst_per_row = []
        per_row_warn_count = []
        soft = 0.20
        for _, row in merged.iterrows():
            row_diffs = []
            for f in feat_cols:
                a = row[f"{f}_py"]; b = row[f"{f}_cpp"]
                if pd.isna(a) or pd.isna(b):
                    continue
                row_diffs.append(abs(float(a) - float(b)))
            if row_diffs:
                worst_per_row.append(max(row_diffs))
                per_row_warn_count.append(sum(1 for d in row_diffs if d > soft))
        worst_overall = max(worst_per_row) if worst_per_row else 0.0
        worst_warn_count = max(per_row_warn_count) if per_row_warn_count else 0
        hard_cap = float(round(worst_overall + 0.10, 3))
        out["stage5_features"] = {
            "shared": {
                "single_feature_drift_hard_cap": hard_cap,
                "soft_warning_threshold": float(round(0.6 * hard_cap, 3)),
                "max_features_above_warning": int(worst_warn_count + 2),
                "_observed_worst_diff": float(round(worst_overall, 3)),
                "_observed_worst_warn_count": int(worst_warn_count),
            },
        }
    else:
        out["stage5_features"] = {"shared": {
            "single_feature_drift_hard_cap": 999.0,
            "soft_warning_threshold": 999.0,
            "max_features_above_warning": 999,
            "_status": "no feature parquets to calibrate from",
        }}

    # ------ Stage 4 calibration (peak frame agreement) ------
    pf_path = GOLDEN_ROOT / "peak_frames.json"
    if pf_path.exists():
        locked = json.loads(pf_path.read_text())
        py_pks = locked["pyfaceau_peak_frames"]
        cpp_pks = locked.get("cpp_peak_frames", {})
        per_canary_frac = []
        for pid in py_pks:
            if pid not in cpp_pks: continue
            py_p = py_pks[pid]; cpp_p = cpp_pks[pid]
            valid = [(a, abs(int(py_p[a]) - int(cpp_p[a])))
                     for a in py_p if py_p[a] is not None and cpp_p.get(a) is not None]
            if not valid: continue
            within = sum(1 for _, d in valid if d <= 3)
            per_canary_frac.append(within / len(valid))
        worst_frac = min(per_canary_frac) if per_canary_frac else 1.0
        out["stage4_peak_frames"] = {
            "shared": {
                "max_abs_frame_diff": 3,
                "fraction_within_tolerance_min": float(round(max(0.0, worst_frac - 0.05), 3)),
                "_observed_worst_canary_frac": float(round(worst_frac, 3)),
                "_n_canaries_compared": len(per_canary_frac),
            },
        }
    else:
        out["stage4_peak_frames"] = {"shared": {
            "max_abs_frame_diff": 3,
            "fraction_within_tolerance_min": 0.0,
            "_status": "no peak_frames.json to calibrate from",
        }}

    # ------ Stage 6a calibration (inference parity over canaries) ------
    pred_py = GOLDEN_ROOT / "predictions_pyfaceau.json"
    pred_cpp = GOLDEN_ROOT / "predictions_cpp.json"
    if pred_py.exists() and pred_cpp.exists():
        a = json.loads(pred_py.read_text())
        b = json.loads(pred_cpp.read_text())
        # alignment is by index (both came from same features rows)
        n_total = len(a["y_pred"])
        n_agree = sum(1 for x, y in zip(a["y_pred"], b["y_pred"]) if x == y)
        observed_agree = n_agree / n_total if n_total else 1.0
        out["stage6a_inference_parity"] = {
            "shared": {
                "agreement_min": float(round(max(0.5, observed_agree - 0.05), 3)),
                "_observed_agreement": float(round(observed_agree, 3)),
                "_n_canary_rows": n_total,
            },
        }
    else:
        out["stage6a_inference_parity"] = {"shared": {
            "agreement_min": 0.0,
            "_status": "no predictions_*.json to calibrate from",
        }}

    # ------ Stage 6b: read from retrain_bands.json if present ------
    # observed acc ± 0.005 — TIGHT bands, justified empirically:
    # the runpy-wrapper fix (commit TBD) eliminated the "5pp stochasticity"
    # that we previously attributed to SMOTE/threshold-CV RNG. The actual
    # cause was a broken PYTHONSTARTUP override that silently let Optuna
    # run every time. With Optuna correctly skipped (use_known_optimal=True
    # actually applied), 3 back-to-back retrains of mid_paper produced
    # 0.8889 → 0.8889 → 0.8889 — bit-exact.
    #
    # ±0.005 = ~1 patient out of 200 tolerance, well below the n_test=54
    # single-patient resolution of 0.0185. Failures will indicate real
    # library drift, not normal noise.
    rb_path = GOLDEN_ROOT / "retrain_bands.json"
    if rb_path.exists():
        rb = json.loads(rb_path.read_text())
        out["stage6b_retrain_bands"] = {
            "_status": "auto-calibrated from retrain_bands.json (re-measure with --stage retrain_bands)",
            "_relaxation_acc": 0.005,
            "_relaxation_rationale": (
                "±0.005 ≈ tighter than 1-patient-on-n_test=54 resolution "
                "(0.0185). With use_known_optimal=True actually applied via "
                "runpy wrapper (vs broken PYTHONSTARTUP), retrains are "
                "bit-exact deterministic across runs."
            ),
        }
        for zone in ("mid", "upper", "lower"):
            zone_entry: dict = {}
            for source_label in ("paper", "current_pyfaceau"):
                key = f"{zone}_{source_label}"
                m = rb.get(key, {})
                acc = m.get("accuracy")
                if acc is None:
                    continue
                zone_entry[f"{source_label}_acc_min"] = float(round(max(0.0, acc - 0.005), 4))
                zone_entry[f"{source_label}_acc_max"] = float(round(min(1.0, acc + 0.005), 4))
                zone_entry[f"_{source_label}_observed_acc"] = float(round(acc, 4))
            if zone_entry:
                out["stage6b_retrain_bands"][zone] = zone_entry
    else:
        out["stage6b_retrain_bands"] = {
            "_status": "no retrain_bands.json yet — run `python tests/update_goldens.py --stage retrain_bands` (slow, ~20 min)",
            "mid":   {"acc_min": 0.0, "acc_max": 1.0},
            "upper": {"acc_min": 0.0, "acc_max": 1.0},
            "lower": {"acc_min": 0.0, "acc_max": 1.0},
        }

    out_path = GOLDEN_ROOT / "metric_bands.yaml"
    out_path.write_text(_yaml.safe_dump(out, sort_keys=True, default_flow_style=False))
    print(f"  metric_bands.yaml: calibrated from current observations + headroom")
    print(f"    Stage3 r mins: " + ", ".join(
        f"{sev}/{diff}={out['stage3_aus'][sev][diff].get('pearson_r_min', '-')}"
        for sev in ('normal','paralyzed') for diff in ('easy','medium','hard')))
    print(f"    Stage5 hard cap: {out['stage5_features']['shared'].get('single_feature_drift_hard_cap')}")
    print(f"    Stage4 frac min: {out['stage4_peak_frames']['shared'].get('fraction_within_tolerance_min')}")
    print(f"    Stage6a agreement min: {out['stage6a_inference_parity']['shared'].get('agreement_min')}")
    return [out_path]


# ---------------------------------------------------------------------------
# Run-history bookkeeping
# ---------------------------------------------------------------------------


def _git_sha(path: Path) -> str:
    try:
        return subprocess.check_output(
            ["git", "-C", str(path), "rev-parse", "HEAD"],
            stderr=subprocess.DEVNULL,
        ).decode().strip()
    except Exception:
        return "unknown"


def _pip_freeze_sha(python: Path) -> str:
    try:
        out = subprocess.check_output(
            [str(python), "-m", "pip", "freeze"], stderr=subprocess.DEVNULL,
        )
        import hashlib
        return hashlib.sha256(out).hexdigest()
    except Exception:
        return "unknown"


def append_history(args: argparse.Namespace, written: list[Path]) -> None:
    history = GOLDEN_ROOT / "golden_history.md"
    if not history.exists():
        history.write_text("# Golden Update History\n\nAppend-only log of every `update_goldens.py` run.\n\n")
    git_sha = _git_sha(S3_ROOT)
    pip_sha = _pip_freeze_sha(Path(sys.executable))
    timestamp = dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")
    entry = [
        f"## {timestamp}",
        f"- **Stage(s):** {args.stage}",
        f"- **Reason:** {args.reason}",
        f"- **Git SHA:** `{git_sha}`",
        f"- **pip-freeze SHA256:** `{pip_sha}`",
        f"- **Files written:** {len(written)}",
        "",
    ]
    with history.open("a") as f:
        f.write("\n".join(entry) + "\n")


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------


def stage_batch_processor_subset(args: argparse.Namespace) -> list[Path]:
    """Run facial_au_batch_processor on the 2-canary Tier-0 subset and lock
    its output. Closes the gap between the AU CSVs (S2O) and the
    combined_results.csv (S3O) — the locked goldens should detect any
    change in find_peak_frame, baseline-frame logic, or per-action
    aggregation.

    NOT in the default 'all' order because (a) it requires running the
    analyzer on real AU CSVs (~10s), and (b) most goldens regens won't
    affect this output.
    """
    SUBSET_IDS = ["IMG_0942", "IMG_2380"]
    out_path = GOLDEN_ROOT / "batch_processor_subset.parquet"

    from facial_au_batch_processor import FacialAUBatchProcessor
    import tempfile as _tempfile

    canaries = [c for c in CANARIES if c.id in SUBSET_IDS]
    for c in canaries:
        for side in ("left", "right"):
            if not c.pyfaceau_csv(side).exists():
                print(f"  SKIP: missing pyfaceau CSV for {c.id} {side}")
                return []

    with _tempfile.TemporaryDirectory() as td:
        td_path = Path(td)
        processor = FacialAUBatchProcessor(output_dir=str(td_path))
        for c in canaries:
            processor.add_patient(
                left_csv=str(c.pyfaceau_csv("left")),
                right_csv=str(c.pyfaceau_csv("right")),
                video_path=str(c.video("left")) if c.video("left").exists() else None,
                patient_id=c.id,
            )
        processor.process_all(extract_frames=False, max_workers=1)
        produced_csv = td_path / "combined_results.csv"
        if not produced_csv.exists():
            print(f"  ERROR: combined_results.csv not produced")
            return []
        df = pd.read_csv(produced_csv, low_memory=False)
        df = df[df["Patient ID"].isin(SUBSET_IDS)].sort_values("Patient ID").reset_index(drop=True)
        # Drop columns that legitimately vary across runs (timestamps, paths)
        skip_cols = {
            "Processing Status", "Processing Time", "Output Path",
            "Frame Path", "Timestamp", "Output Folder",
        }
        keep_cols = [c for c in df.columns if c not in skip_cols]
        df = df[keep_cols]
        # Stringify Patient ID to ensure parquet stability
        if "Patient ID" in df.columns:
            df["Patient ID"] = df["Patient ID"].astype(str)
        stable_dataframe(df).to_parquet(out_path, index=False, compression="zstd")
        print(f"  Locked batch processor subset output ({len(df)} rows × {len(df.columns)} cols)")
        for _, row in df.iterrows():
            pks = {col.split("_")[0]: int(row[col]) for col in df.columns if col.endswith("_Max Frame") and pd.notna(row[col])}
            print(f"    {row['Patient ID']}: peak frames {pks}")
    return [out_path]


def stage_gpu_divergence(args: argparse.Namespace) -> list[Path]:
    """Measure CPU vs GPU AU output divergence on IMG_0942 left and lock
    the upper band per AU.

    Lives outside the default 'all' order because:
      - Slow (~1 min: needs two pyfaceau Pipelines on 30 frames each)
      - Only relevant when GPU code changes — most goldens regens won't
        need to re-measure this
    Run explicitly: `python tests/update_goldens.py --stage gpu_divergence ...`
    """
    out_path = GOLDEN_ROOT / "gpu_divergence_baseline.json"
    canary_id = "IMG_0942"
    side = "left"
    canary = next((c for c in CANARIES if c.id == canary_id), None)
    if canary is None or not canary.video(side).exists():
        print(f"  SKIP: {canary_id} {side} video missing")
        return []

    from pyfaceau.processor import OpenFaceProcessor
    from pyfaceau.config import CLNF_CONFIG

    max_frames = 30

    def _run(use_gpu: bool):
        saved = CLNF_CONFIG.get("use_gpu", False)
        try:
            CLNF_CONFIG["use_gpu"] = use_gpu
            proc = OpenFaceProcessor(verbose=False)
            df = proc.pipeline.process_video(
                str(canary.video(side)), output_csv=None, max_frames=max_frames
            )
            return df.reset_index(drop=True)
        finally:
            CLNF_CONFIG["use_gpu"] = saved

    print(f"  CPU run on {canary_id} {side} (max {max_frames} frames)...")
    cpu_df = _run(False)
    print(f"  GPU run on {canary_id} {side} (max {max_frames} frames)...")
    gpu_df = _run(True)

    common = cpu_df["frame"].isin(gpu_df["frame"]) & gpu_df["frame"].isin(cpu_df["frame"])
    cpu_ok = cpu_df.loc[common & (cpu_df["success"] == 1) & (gpu_df["success"] == 1)]
    gpu_ok = gpu_df.loc[common & (cpu_df["success"] == 1) & (gpu_df["success"] == 1)]

    au_cols = [c for c in cpu_df.columns if c.startswith("AU") and c.endswith("_r")]
    per_au: dict[str, float] = {}
    for au in au_cols:
        a = cpu_ok[au].astype(float).to_numpy()
        b = gpu_ok[au].astype(float).to_numpy()
        per_au[au] = float(np.mean(np.abs(a - b)))

    # Lock upper band as observed_mae + 0.05 (room for tiny drift)
    upper_band = {au: round(min(0.5, mae + 0.05), 4) for au, mae in per_au.items()}

    obj = {
        "canary": f"{canary_id}_{side}",
        "max_frames": max_frames,
        "n_compared_frames": int(len(cpu_ok)),
        "observed_per_au_mae": {au: round(v, 4) for au, v in per_au.items()},
        "max_acceptable_per_au_mae": upper_band,
        "_relaxation": "+0.05 over observed, capped at 0.5",
    }
    out_path.write_text(json.dumps(obj, indent=2, sort_keys=True) + "\n")
    print(f"  Locked GPU divergence baseline ({len(per_au)} AUs)")
    print(f"    worst observed MAE: {max(per_au.values()):.4f} ({max(per_au, key=per_au.get)})")
    return [out_path]


def stage_production_predictions(args: argparse.Namespace) -> list[Path]:
    """Lock per-canary × side × zone production-detector severities.

    Calls ParalysisDetector(zone).detect(row, side) for each canary using the
    current pyfaceau combined_results.csv row. Locks the predicted severity
    string ('Normal' | 'Partial' | 'Complete' | 'Error') so test_tier1_production_inference
    can detect future drift in the production inference path.

    Catches a different class of regression than stage_predictions (which
    uses the training-side feature pipeline): production goes through
    extract_features_for_detection() per single row, not the bulk
    extract_features() loop.
    """
    out_path = GOLDEN_ROOT / "production_predictions.json"
    if not PYFACEAU_COMBINED_CSV.exists():
        print(f"  SKIP: missing {PYFACEAU_COMBINED_CSV}")
        return []
    from paralysis_detector import ParalysisDetector

    detectors = {z: ParalysisDetector(z) for z in ("mid", "upper", "lower")}
    df = pd.read_csv(PYFACEAU_COMBINED_CSV, low_memory=False)
    canary_ids = {c.id for c in CANARIES}

    obj: dict[str, dict] = {}
    for _, row in df.iterrows():
        pid = row.get("Patient ID")
        if pid not in canary_ids:
            continue
        rd = row.to_dict()
        per_canary: dict = {}
        for zone, det in detectors.items():
            for side in ("left", "right"):
                result_str, conf, _details = det.detect(rd, side)
                per_canary[f"{zone}_{side}"] = result_str
                # also lock confidence to a coarse bucket for human eyeballing;
                # not used by the test (which compares result_str only) but
                # surfaces big confidence shifts in the diff.
                per_canary[f"{zone}_{side}_conf_bucket"] = round(conf, 1)
        obj[pid] = per_canary
        # quick visual summary
        sev_str = " ".join(
            f"{z}/{s[0]}={per_canary[f'{z}_{s}'][0]}"
            for z in ("mid",) for s in ("left", "right")
        )
        print(f"  {pid:>22s}: {sev_str}")
    out_path.write_text(json.dumps(obj, indent=2, sort_keys=True) + "\n")
    print(f"  Locked production predictions for {len(obj)} canaries")
    return [out_path]


def stage_retrain_bands(args: argparse.Namespace) -> list[Path]:
    """Measure current retrain test accuracy per (zone × source) and write
    bands to golden/retrain_bands.json. Bands = observed acc ± 0.04.

    SLOW (~3-5 min per measurement, 6 measurements = 15-30 min total). Run
    only when a Tier 2 baseline needs refresh — e.g. after sklearn upgrade.

    Writes golden/retrain_bands.json which stage_metric_bands then reads to
    populate metric_bands.yaml's stage6b_retrain_bands section.

    Crucially: this never modifies the production combined_results.csv on
    disk — it overrides INPUT_FILES['results_csv'] in the subprocess via
    PYTHONSTARTUP so production state is undisturbed (so other tests can
    keep running concurrently).
    """
    out_path = GOLDEN_ROOT / "retrain_bands.json"
    paper_csv = S3_ROOT / "paper_combined_results.csv"
    if not (paper_csv.exists() and PYFACEAU_COMBINED_CSV.exists()):
        print(f"  SKIP: missing CSV inputs (paper={paper_csv.exists()}, current={PYFACEAU_COMBINED_CSV.exists()})")
        return []

    measurements: dict[str, dict[str, float]] = {}
    log_dir = HERE / "_retrain_bands_logs"
    log_dir.mkdir(exist_ok=True)

    for source_label, src in (("paper", paper_csv), ("current_pyfaceau", PYFACEAU_COMBINED_CSV)):
        for zone in ("mid", "upper", "lower"):
            # Per-run helper: enable use_known_optimal AND redirect INPUT_FILES.
            # CRITICAL: must be invoked as a wrapper (python wrapper.py),
            # NOT as PYTHONSTARTUP (which only fires in interactive mode and
            # is silently ignored for `python script.py`). This was the bug
            # that caused the original 5pp Tier 2 stochasticity:
            # use_known_optimal was never actually being applied, so Optuna
            # ran every time with TPESampler nondeterminism.
            wrapper = HERE / f"_retrain_bands_wrapper_{zone}_{source_label}.py"
            wrapper.write_text(
                f"# Auto-generated by update_goldens.stage_retrain_bands\n"
                f"import sys, runpy\n"
                f"sys.path.insert(0, {str(S3_ROOT)!r})\n"
                f"import paralysis_config\n"
                f"paralysis_config.INPUT_FILES['results_csv'] = {str(src)!r}\n"
                f"for z in ('mid','upper','lower'):\n"
                f"    paralysis_config.ZONE_CONFIG[z]['training']['hyperparameter_tuning']['use_known_optimal'] = True\n"
                f"# Now invoke the training pipeline as if it were the entry script\n"
                f"sys.argv = ['paralysis_training_pipeline.py', {zone!r}]\n"
                f"runpy.run_path({str(S3_ROOT / 'paralysis_training_pipeline.py')!r}, run_name='__main__')\n"
            )
            env = {**os.environ, "PYTHONHASHSEED": "42", "OMP_NUM_THREADS": "1"}
            log_path = log_dir / f"{zone}_{source_label}.log"
            t0 = time.perf_counter()
            try:
                with log_path.open("w") as f:
                    rc = subprocess.run(
                        [sys.executable, str(wrapper)],
                        cwd=str(S3_ROOT), env=env, stdout=f, stderr=subprocess.STDOUT,
                        timeout=900,
                    ).returncode
            finally:
                wrapper.unlink(missing_ok=True)
            dt = time.perf_counter() - t0
            acc = None
            bal = None
            txt = log_path.read_text()
            m = re.search(r"Overall Accuracy:\s*([\d.]+)", txt)
            if m:
                acc = float(m.group(1))
            m = re.search(r"Balanced Accuracy:\s*([\d.]+)", txt)
            if m:
                bal = float(m.group(1))
            key = f"{zone}_{source_label}"
            measurements[key] = {
                "zone": zone,
                "source": source_label,
                "exit_code": int(rc),
                "accuracy": acc,
                "balanced_accuracy": bal,
                "elapsed_sec": round(dt, 1),
                "log": str(log_path.relative_to(S3_ROOT)),
            }
            print(f"  {key:>30s}: acc={acc} bal={bal} ({dt:.1f}s, exit {rc})")

    out_path.write_text(json.dumps(measurements, indent=2, sort_keys=True) + "\n")
    return [out_path]


STAGES: dict[str, callable] = {
    "aus":                    stage_aus,
    "landmarks":              stage_landmarks,
    "peak_frames":            stage_peak_frames,
    "features":               stage_features,
    "predictions":            stage_predictions,
    "production_predictions": stage_production_predictions,
    "test_split":             stage_test_split,
    "metric_bands":           stage_metric_bands,
    "retrain_bands":          stage_retrain_bands,
    "gpu_divergence":         stage_gpu_divergence,
    "batch_processor_subset": stage_batch_processor_subset,
}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--stage", required=True,
        help=f"One of: {sorted(STAGES) + ['all']}",
    )
    parser.add_argument("--reason", required=True, help="Why are we updating? Logged to golden_history.md")
    args = parser.parse_args()

    GOLDEN_ROOT.mkdir(parents=True, exist_ok=True)
    (GOLDEN_ROOT / ".gitkeep").touch()

    if args.stage == "all":
        # Order matters: features depends on combined_results being on disk;
        # predictions depends on features parquets existing.
        # retrain_bands not in default 'all' (it's slow ~20min); run it
        # separately when sklearn/xgboost upgrade.
        order = [
            "aus", "landmarks", "peak_frames", "test_split",
            "features", "predictions", "production_predictions",
            "metric_bands",
        ]
    else:
        order = [args.stage]

    written: list[Path] = []
    for stage in order:
        if stage not in STAGES:
            parser.error(f"unknown stage: {stage}; valid: {sorted(STAGES)}")
        print(f"\n=== stage: {stage} ===")
        written.extend(STAGES[stage](args))

    print(f"\n=== writing checksums.json over {len(list(GOLDEN_ROOT.rglob('*')))} golden files ===")
    sums = write_checksums(GOLDEN_ROOT)
    print(f"  {len(sums)} entries")

    append_history(args, written)
    print(f"\nDone. Goldens at: {GOLDEN_ROOT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
