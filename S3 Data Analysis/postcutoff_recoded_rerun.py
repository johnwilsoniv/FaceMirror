#!/usr/bin/env python3
"""
postcutoff_recoded_rerun.py — apply v1.3.15-equivalent cutoffs to AU17 / AU26
columns of v1.3.14 output (the Windows team's recoded_rerun batch).

Why
---
The Windows team is reprocessing recoded_per_frame/ on pyfaceau v1.3.14
(which still has the AU17 cutoff skipped and the AU26 cutoff overridden
to 0.12). Pyfaceau v1.3.15 fixes both — applies AU17's stored 0.20 cutoff
and removes the AU26 override (so the model's stored 0.30 applies).

Re-running the Windows batch on v1.3.15 would take another ~7 hours.
Instead this script post-processes the v1.3.14 output to closely match
what v1.3.15 would have produced. ~5 minutes for 222 CSVs.

What gets adjusted
------------------
- AU17_r: pyfaceau v1.3.14 stored RAW values (skip_au17_cutoff=True).
  Apply the model's stored 0.20 percentile cutoff. EXACT match to v1.3.15.

- AU26_r: pyfaceau v1.3.14 already applied a 0.12 percentile cutoff.
  We can't fully undo + reapply 0.30 without the original raw values.
  Pragmatic approximation: apply an ADDITIONAL 0.20 percentile cutoff on
  the current values. The 20-canary sweep showed this matches the
  stored-0.30 result very closely in r / MAE / intercept terms (within
  noise of the bit-perfect v1.3.15 result).

All other AU columns are passed through unchanged.

Input / output filename pattern is preserved exactly so downstream Mac
analysis scripts (which glob on `<patient>_<side>_mirrored_coded.csv`)
work without modification on the output dir.

Usage
-----
    # adjust paths if needed
    python postcutoff_recoded_rerun.py \\
        --input  /path/to/recoded_rerun \\
        --output /path/to/recoded_rerun_v1315

Outputs a status report at the end:
    n processed, n skipped, n errored, per-AU mean shift summary.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd


def apply_cutoff(values: np.ndarray, success_mask: np.ndarray,
                 percentile: float) -> np.ndarray:
    """Pyfaceau-equivalent cutoff: subtract the Nth-percentile value of the
    SUCCESS-filtered series from every frame, clip to [0, 5]. Mirrors
    pipeline.finalize_predictions().
    """
    if percentile <= 0:
        return values.copy()
    valid = values[success_mask]
    if len(valid) < 10:
        return values.copy()
    sorted_v = np.sort(valid)
    idx = int(len(sorted_v) * percentile)
    offset = sorted_v[idx]
    return np.clip(values - offset, 0.0, 5.0)


def process_one_csv(in_path: Path, out_path: Path) -> dict:
    """Apply cutoffs to AU17 and AU26; pass everything else through."""
    df = pd.read_csv(in_path)

    # success mask: prefer the success column; fall back to non-NaN AU
    if 'success' in df.columns:
        success = df['success'].astype(str).str.lower().isin(
            ['true', '1', '1.0']).values
        if not success.any():
            # fall back if the bool parsing failed
            success = ~df['AU17_r'].isna().values
    else:
        success = ~df['AU17_r'].isna().values

    stats = {'file': in_path.name, 'n_frames': len(df)}

    # AU17 — exact: v1.3.14 stored raw, apply stored 0.20 cutoff
    if 'AU17_r' in df.columns:
        au17_before = df['AU17_r'].copy()
        df['AU17_r'] = apply_cutoff(df['AU17_r'].astype(float).values,
                                    success, 0.20)
        stats['au17_mean_shift'] = float(df['AU17_r'].mean()
                                         - au17_before.mean())

    # AU26 — approximate: v1.3.14 already applied 0.12, apply +0.20 more.
    # Sweep showed this matches stored-0.30 cleanly.
    if 'AU26_r' in df.columns:
        au26_before = df['AU26_r'].copy()
        df['AU26_r'] = apply_cutoff(df['AU26_r'].astype(float).values,
                                    success, 0.20)
        stats['au26_mean_shift'] = float(df['AU26_r'].mean()
                                         - au26_before.mean())

    df.to_csv(out_path, index=False)
    return stats


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--input', required=True, type=Path,
                    help='Input dir of v1.3.14 recoded_rerun CSVs')
    ap.add_argument('--output', required=True, type=Path,
                    help='Output dir for v1.3.15-equivalent CSVs')
    ap.add_argument('--pattern', default='*_coded.csv',
                    help='Glob pattern for input CSVs (default: *_coded.csv)')
    ap.add_argument('--force', action='store_true',
                    help='Overwrite existing output CSVs')
    args = ap.parse_args()

    if not args.input.is_dir():
        print(f'ERROR: input dir not found: {args.input}', file=sys.stderr)
        sys.exit(1)

    args.output.mkdir(parents=True, exist_ok=True)

    csvs = sorted(args.input.glob(args.pattern))
    print(f'Found {len(csvs)} CSVs in {args.input}')
    print(f'Writing to {args.output}')
    print()

    processed, skipped, errored = 0, 0, 0
    all_stats = []

    for in_csv in csvs:
        out_csv = args.output / in_csv.name
        if out_csv.exists() and not args.force:
            skipped += 1
            continue
        try:
            stats = process_one_csv(in_csv, out_csv)
            all_stats.append(stats)
            processed += 1
            if processed <= 5 or processed % 50 == 0:
                print(f'  [{processed}/{len(csvs)}] {in_csv.name}: '
                      f'AU17 shift {stats.get("au17_mean_shift", float("nan")):+.3f}, '
                      f'AU26 shift {stats.get("au26_mean_shift", float("nan")):+.3f}')
        except Exception as e:
            errored += 1
            print(f'  ERROR {in_csv.name}: {e}', file=sys.stderr)

    print()
    print(f'Processed: {processed}')
    print(f'Skipped (already done; pass --force to overwrite): {skipped}')
    print(f'Errored: {errored}')
    if all_stats:
        au17_shifts = [s['au17_mean_shift'] for s in all_stats
                       if 'au17_mean_shift' in s]
        au26_shifts = [s['au26_mean_shift'] for s in all_stats
                       if 'au26_mean_shift' in s]
        print()
        print('Aggregate shifts (negative = baseline subtracted as expected):')
        print(f'  AU17: mean shift {np.mean(au17_shifts):+.3f}, '
              f'median {np.median(au17_shifts):+.3f}, '
              f'range [{np.min(au17_shifts):+.3f}, {np.max(au17_shifts):+.3f}]')
        print(f'  AU26: mean shift {np.mean(au26_shifts):+.3f}, '
              f'median {np.median(au26_shifts):+.3f}, '
              f'range [{np.min(au26_shifts):+.3f}, {np.max(au26_shifts):+.3f}]')


if __name__ == '__main__':
    main()
