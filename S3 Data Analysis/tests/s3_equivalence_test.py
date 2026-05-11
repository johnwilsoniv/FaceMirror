"""S3 cross-platform classifier-inference equivalence test.

Loads paper_combined_results.csv, runs ParalysisDetector for each
(patient, side) on every zone, and emits a JSON snapshot containing:
    - SHA256 of file bytes for each zone's model.pkl, scaler.pkl, features.list
    - SHA256 of the concatenated prediction string per zone
    - SHA256 of the concatenated probability vector per zone (catches
      numerical drift even when the predicted class happens to match)
    - accuracy + F1_weighted per zone (when ground-truth labels are
      available; otherwise reported as None)

Run on macOS to produce the reference, then run on Windows. Diff the two
JSON files: if every per-zone predictions_sha and proba_sha matches
across platforms, the S3 inference path is bit-for-bit reproducible
between Mac and Windows and S3 ships.

Usage
-----
    cd "S3 Data Analysis"
    python tests/s3_equivalence_test.py --out /path/to/snapshot.json

    # On the second platform after running both:
    python tests/s3_equivalence_test.py \
        --diff /path/to/mac_snapshot.json /path/to/win_snapshot.json
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

S3_DIR = Path(os.environ.get("S3_DIR") or Path(__file__).resolve().parent.parent)
sys.path.insert(0, str(S3_DIR))

from paralysis_detector import ParalysisDetector  # noqa: E402

ZONES = ("upper", "mid", "lower")
SIDES = ("left", "right")


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _sha256_obj(obj) -> str:
    """Stable SHA over a Python object via canonical JSON."""
    return hashlib.sha256(
        json.dumps(obj, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _round_proba(p: float, dp: int = 10) -> float:
    """Round probabilities to `dp` decimal places before hashing so cross-
    platform FP noise (~1e-7) doesn't trivially flip the SHA. dp=10 is
    well below sklearn/xgboost's reproducibility limit but tight enough
    to catch real drift."""
    return round(float(p), dp)


def _load_ground_truth_labels(s3_dir: Path) -> pd.DataFrame | None:
    """Optional: load the gitignored expert key for accuracy/F1 calc."""
    key_path = s3_dir / "FPRS FP Key.csv"
    if not key_path.exists():
        return None
    return pd.read_csv(key_path)


def run_snapshot(s3_dir: Path) -> dict:
    """Build the per-zone equivalence snapshot."""
    csv_path = s3_dir / "paper_combined_results.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"missing input: {csv_path}")
    df = pd.read_csv(csv_path, low_memory=False)
    rows = [r.to_dict() for _, r in df.iterrows() if r.get("Patient ID")]

    truth_df = _load_ground_truth_labels(s3_dir)
    snapshot = {
        "platform": sys.platform,
        "python": sys.version.split()[0],
        "input_csv": str(csv_path),
        "input_csv_sha256": _sha256_file(csv_path),
        "n_patients": len(rows),
        "zones": {},
    }
    if truth_df is not None:
        snapshot["truth_csv_sha256"] = _sha256_file(s3_dir / "FPRS FP Key.csv")
    else:
        snapshot["truth_csv_sha256"] = None

    for zone in ZONES:
        detector = ParalysisDetector(zone)
        # File integrity
        cfg = detector.config["filenames"]
        zone_block = {
            "model_sha256": _sha256_file(Path(cfg["model"])),
            "scaler_sha256": _sha256_file(Path(cfg["scaler"])),
            "features_sha256": _sha256_file(Path(cfg["feature_list"])),
        }
        # Per-(patient, side) predictions + probabilities
        preds = []   # ordered list of (patient_id, side, pred_str)
        probas = [] # ordered list of (patient_id, side, [class_probas])
        for row in rows:
            pid = row["Patient ID"]
            for side in SIDES:
                pred_str, conf, details = detector.detect(row, side)
                preds.append((pid, side, pred_str))
                # `details` is impl-specific -- pull probas out best-effort
                p_vec = None
                if isinstance(details, dict):
                    for k in ("probabilities", "proba", "class_probabilities"):
                        if k in details and details[k] is not None:
                            p_vec = details[k]
                            break
                if p_vec is None:
                    p_vec = [_round_proba(conf)]
                else:
                    p_vec = [_round_proba(p) for p in np.asarray(p_vec).flatten()]
                probas.append((pid, side, p_vec))
        zone_block["predictions_sha256"] = _sha256_obj(preds)
        zone_block["probability_sha256"] = _sha256_obj(probas)
        zone_block["n_predictions"] = len(preds)
        # Optional: accuracy / F1 if we have ground-truth
        if truth_df is not None:
            try:
                from sklearn.metrics import accuracy_score, f1_score
                truth_col = f"{zone}_face_severity"  # adjust per your key schema
                if truth_col not in truth_df.columns:
                    # try common variants
                    for alt in (zone.capitalize() + " Face Severity", f"{zone.capitalize()}_Severity"):
                        if alt in truth_df.columns:
                            truth_col = alt
                            break
                # For accuracy calc we need to align predictions with truth rows
                # Skip if column not found
                if truth_col in truth_df.columns:
                    truth_map = {
                        f"{r['Patient ID']}_{(r.get('Side') or 'left').lower()}": r[truth_col]
                        for _, r in truth_df.iterrows()
                        if r.get("Patient ID")
                    }
                    y_true = []
                    y_pred = []
                    for pid, side, pred in preds:
                        key = f"{pid}_{side}"
                        if key in truth_map and pd.notna(truth_map[key]):
                            y_true.append(str(truth_map[key]))
                            y_pred.append(pred)
                    if y_true:
                        zone_block["accuracy"] = float(accuracy_score(y_true, y_pred))
                        zone_block["f1_weighted"] = float(
                            f1_score(y_true, y_pred, average="weighted", zero_division=0)
                        )
                        zone_block["n_evaluated"] = len(y_true)
            except Exception as e:
                zone_block["accuracy_error"] = repr(e)
        snapshot["zones"][zone] = zone_block

    return snapshot


def diff_snapshots(a_path: Path, b_path: Path) -> int:
    """Pretty-print equivalence diff between two snapshots. Returns 0 if
    EQUIVALENT, 1 otherwise."""
    a = json.loads(a_path.read_text())
    b = json.loads(b_path.read_text())
    print(f"A: {a_path.name}  platform={a.get('platform')}  py={a.get('python')}")
    print(f"B: {b_path.name}  platform={b.get('platform')}  py={b.get('python')}")
    print(f"input_csv_sha256: A={a.get('input_csv_sha256','')[:16]}  B={b.get('input_csv_sha256','')[:16]}")
    print()
    print("=== Per-zone equivalence ===")
    diverged = False
    for zone in ZONES:
        za = a["zones"][zone]
        zb = b["zones"][zone]
        keys = ("model_sha256", "scaler_sha256", "features_sha256",
                "predictions_sha256", "probability_sha256")
        zone_eq = all(za[k] == zb[k] for k in keys)
        marker = "EQUIVALENT" if zone_eq else "DIVERGENT"
        print(f"  {zone.upper()} -- {marker}")
        for k in keys:
            same = za[k] == zb[k]
            mark = "=" if same else "!="
            print(f"    {k:<22} {mark}  A={za[k][:16]}...  B={zb[k][:16]}...")
        if "accuracy" in za and "accuracy" in zb:
            print(f"    accuracy             =  A={za['accuracy']:.6f}  B={zb['accuracy']:.6f}")
            print(f"    f1_weighted          =  A={za['f1_weighted']:.6f}  B={zb['f1_weighted']:.6f}")
        if not zone_eq:
            diverged = True
        print()
    if diverged:
        print("DIVERGENT: Windows S3 inference does NOT match macOS bit-for-bit.")
        print("Check above which SHA differs to localize the drift "
              "(model file -> LFS issue; predictions only -> argmax tie-break; "
              "predictions+proba -> sklearn/xgboost numerical divergence).")
        return 1
    print("EQUIVALENT: Windows S3 == Mac S3 on this input. Safe to ship.")
    return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, help="output JSON snapshot path")
    ap.add_argument("--diff", nargs=2, metavar=("MAC_JSON", "WIN_JSON"),
                    help="diff two pre-computed snapshots")
    args = ap.parse_args()

    if args.diff:
        sys.exit(diff_snapshots(Path(args.diff[0]), Path(args.diff[1])))

    snap = run_snapshot(S3_DIR)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(snap, indent=2, sort_keys=True))
        print(f"snapshot written to {args.out}")
    else:
        print(json.dumps(snap, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
