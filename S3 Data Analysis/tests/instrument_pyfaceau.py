"""Run pyfaceau on canary videos and capture per-frame bbox + 68 landmarks
into parquet files under tests/golden/{bbox,landmarks}/<canary>_<side>/pyfaceau.parquet.

Why a separate script (not part of update_goldens.py): pyfaceau processing is
slow (~30-60s per ~600-frame video; 10 canaries × 2 sides ≈ 15-20 min total).
We don't want every regen of fast goldens (peak frames, features, predictions)
to incur that cost. This script is run once after pyfaceau changes; the
fast `update_goldens.py` then snapshots the resulting parquets.

Usage:
    python tests/instrument_pyfaceau.py --canary IMG_0942 --side left
    python tests/instrument_pyfaceau.py --all          # all 10 canaries × both sides
    python tests/instrument_pyfaceau.py --tier0        # just IMG_0942 + IMG_2380

Per-frame schema (parquet):
    frame:int           1-indexed (matches C++)
    success:int         0/1; 0 means face detection or CLNF failed
    bbox_x1, bbox_y1, bbox_x2, bbox_y2:float    bbox corners (NaN if no face)
    x_0..x_67:float     2D landmark x coordinates (NaN if no landmarks)
    y_0..y_67:float     2D landmark y coordinates

The bbox values come from the face_detection step (PyMTCNN). Landmark values
come from the landmark_detection step (pyclnf CLNF). Frame numbering matches
C++ (1-indexed) for clean joins with C++ outputs.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))

from conftest import CANARIES, GOLDEN_ROOT, TIER0_CANARIES, Canary  # noqa: E402
from _pipeline_helpers import stable_dataframe  # noqa: E402


# ---------------------------------------------------------------------------
# Per-video runner
# ---------------------------------------------------------------------------


def _build_processor():
    """Initialize an OpenFaceProcessor with default weights — same path the
    production S2 pipeline takes."""
    from pyfaceau.processor import OpenFaceProcessor
    return OpenFaceProcessor(verbose=False)


def _empty_landmarks() -> dict:
    out = {}
    for i in range(68):
        out[f"x_{i}"] = float("nan")
        out[f"y_{i}"] = float("nan")
    return out


def _empty_bbox() -> dict:
    return {
        "bbox_x1": float("nan"),
        "bbox_y1": float("nan"),
        "bbox_x2": float("nan"),
        "bbox_y2": float("nan"),
    }


def _frame_record(frame_idx: int, debug: dict | None) -> dict:
    """Convert pipeline._process_frame's debug_info to a flat row."""
    rec: dict = {"frame": frame_idx}
    rec["success"] = 0
    rec.update(_empty_bbox())
    rec.update(_empty_landmarks())

    if not debug:
        return rec
    fd = debug.get("face_detection") or {}
    bbox = fd.get("bbox")
    if bbox is not None:
        try:
            rec["bbox_x1"] = float(bbox[0])
            rec["bbox_y1"] = float(bbox[1])
            rec["bbox_x2"] = float(bbox[2])
            rec["bbox_y2"] = float(bbox[3])
        except (TypeError, IndexError):
            pass
    ld = debug.get("landmark_detection") or {}
    lmks = ld.get("landmarks_68")
    if lmks is not None:
        arr = np.asarray(lmks)
        if arr.shape == (68, 2):
            for i in range(68):
                rec[f"x_{i}"] = float(arr[i, 0])
                rec[f"y_{i}"] = float(arr[i, 1])
            rec["success"] = 1
    return rec


def instrument_one(canary: Canary, side: str, processor=None, force: bool = False) -> Path | None:
    """Process one (canary, side) video and write golden parquets."""
    video_path = canary.video(side)
    if not video_path.exists():
        print(f"  SKIP {canary.id} {side}: video not at {video_path}")
        return None

    bbox_lmk_dir = GOLDEN_ROOT / "landmarks" / f"{canary.id}_{side}"
    bbox_lmk_dir.mkdir(parents=True, exist_ok=True)
    out_path = bbox_lmk_dir / "pyfaceau.parquet"
    if out_path.exists() and not force:
        print(f"  EXISTS {canary.id} {side}: {out_path.relative_to(GOLDEN_ROOT)} (use --force to regen)")
        return out_path

    if processor is None:
        processor = _build_processor()

    pipeline = processor.pipeline
    # Reset per-video state — same call the production code makes between videos
    if hasattr(pipeline, "_reset_per_video_state"):
        pipeline._reset_per_video_state()

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"  ERROR {canary.id} {side}: cannot open video")
        return None
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)

    rows: list[dict] = []
    t0 = time.perf_counter()
    # frame numbering: C++ FeatureExtraction outputs 1-indexed frames; match it
    frame_idx = 1
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        timestamp = (frame_idx - 1) / fps
        result = pipeline._process_frame(frame, frame_idx - 1, timestamp, return_debug=True)
        rows.append(_frame_record(frame_idx, result.get("debug_info")))
        frame_idx += 1
    cap.release()

    df = pd.DataFrame(rows)
    df = stable_dataframe(df)
    df.to_parquet(out_path, index=False, compression="zstd")
    elapsed = time.perf_counter() - t0
    n_success = int((df["success"] == 1).sum())
    print(
        f"  {canary.id:>22s} {side:5s}  {len(df)} frames in {elapsed:.1f}s "
        f"({len(df)/elapsed:.1f} fps), {n_success} successful"
    )
    return out_path


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    grp = parser.add_mutually_exclusive_group(required=True)
    grp.add_argument("--all", action="store_true", help="Process all 10 canaries × both sides")
    grp.add_argument("--tier0", action="store_true", help="Process Tier 0 canaries only (IMG_0942 + IMG_2380)")
    grp.add_argument("--canary", help="Process a specific canary id")
    parser.add_argument("--side", choices=("left", "right", "both"), default="both",
                        help="Which side(s) to process (default: both)")
    parser.add_argument("--force", action="store_true", help="Regenerate even if golden parquet exists")
    args = parser.parse_args()

    if args.all:
        targets = list(CANARIES)
    elif args.tier0:
        targets = list(TIER0_CANARIES)
    else:
        targets = [c for c in CANARIES if c.id == args.canary]
        if not targets:
            parser.error(f"unknown canary id: {args.canary}; valid: {[c.id for c in CANARIES]}")

    sides = ["left", "right"] if args.side == "both" else [args.side]

    print(f"Initializing PyFaceAU processor (one-time)...")
    processor = _build_processor()
    print(f"Processing {len(targets)} canaries × {len(sides)} sides = {len(targets) * len(sides)} videos\n")

    written = 0
    for c in targets:
        for side in sides:
            res = instrument_one(c, side, processor=processor, force=args.force)
            if res is not None:
                written += 1
    print(f"\nWrote {written} parquet files. Now run:")
    print(f"  python tests/update_goldens.py --stage landmarks --reason 'instrument_pyfaceau ran'")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
