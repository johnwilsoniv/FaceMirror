#!/usr/bin/env python3
"""
Build combined results CSV from recoded_rerun_dual_v1316/ per-frame CSVs.

Inputs:
  - 222 per-frame CSVs at S3 Data Analysis/recoded_rerun_dual_v1316/
    Schema: frame, timestamp, success, 17 AU{n}_r, 17 AU{n}_r_static, action
    Filename: {patient_id}_{left|right}_mirrored_coded.csv

Output:
  - S3 Data Analysis/recoded_rerun_dual_v1316_combined_results.csv
    One row per patient. Columns:
      Patient ID
      {task}_Max Frame                       (per task; computed on Left side default-mode)
      {task}_{side} AU{n}_r                  (max within task, default mode)
      {task}_{side} AU{n}_r_static           (max within task, static mode)
      {task}_{side} AU{n}_r_static_p10       (10th-pct within task, static mode)

The two static aggregations are both kept because (a) max_static is the
voluntary-task signal in static mode (used in pilot 15 Phase A head-to-head
comparison vs default), and (b) p10_static is the resting-tone "floor" signal
validated in pilots 13/14 and used in pilot 16 hypertonicity work.

Default-mode p10 isn't emitted: it's near-zero by construction (running median
sets the baseline).
"""
from pathlib import Path
import pandas as pd
import numpy as np
import re
import sys

DATA_DIR = Path("/Users/johnwilsoniv/Documents/SplitFace Open3/"
                "S3 Data Analysis/recoded_rerun_dual_v1316")
OUT_CSV = Path("/Users/johnwilsoniv/Documents/SplitFace Open3/"
               "S3 Data Analysis/recoded_rerun_dual_v1316_combined_results.csv")

AU_ORDER = ['AU01', 'AU02', 'AU04', 'AU05', 'AU06', 'AU07', 'AU09',
            'AU10', 'AU12', 'AU14', 'AU15', 'AU17', 'AU20', 'AU23',
            'AU25', 'AU26', 'AU45']

# 14 canonical tasks (matches recoded_combined_results.csv layout)
TASKS = ['BC', 'BK', 'BL', 'BS', 'ES', 'ET', 'FR', 'LT', 'PL', 'RE',
         'SE', 'SO', 'SS', 'WN']

FILENAME_RE = re.compile(r'^(.+)_(left|right)_mirrored_coded\.csv$')


def parse_filename(p: Path):
    m = FILENAME_RE.match(p.name)
    if not m:
        return None, None
    return m.group(1), m.group(2).capitalize()


def aggregate_one_side(df: pd.DataFrame):
    """For one (patient, side) per-frame DataFrame, return dict of per-task
    aggregations keyed without side prefix. Caller adds the side prefix when
    placing into the wide row.
    """
    out = {}
    # Strip whitespace; pyfaceau output can have stray spaces around the label
    df = df.copy()
    df['action'] = df['action'].astype(str).str.strip()
    for task in TASKS:
        task_frames = df[df['action'] == task]
        if len(task_frames) == 0:
            for au in AU_ORDER:
                out[f'{task} {au}_r'] = np.nan
                out[f'{task} {au}_r_static'] = np.nan
                out[f'{task} {au}_r_static_p10'] = np.nan
            out[f'{task}_Max Frame'] = np.nan
            continue

        au_r_cols = [f'{au}_r' for au in AU_ORDER]
        sum_r = task_frames[au_r_cols].sum(axis=1)
        max_idx = sum_r.idxmax()
        out[f'{task}_Max Frame'] = int(task_frames.loc[max_idx, 'frame'])

        for au in AU_ORDER:
            r_vals = task_frames[f'{au}_r'].astype(float).values
            s_vals = task_frames[f'{au}_r_static'].astype(float).values
            out[f'{task} {au}_r'] = float(np.max(r_vals))
            out[f'{task} {au}_r_static'] = float(np.max(s_vals))
            out[f'{task} {au}_r_static_p10'] = float(np.percentile(s_vals, 10))
    return out


def main():
    csvs = sorted(DATA_DIR.glob('*_mirrored_coded.csv'))
    print(f'Found {len(csvs)} per-frame CSVs in {DATA_DIR}')
    if len(csvs) != 222:
        print(f'  WARNING: expected 222 files, got {len(csvs)}')

    by_patient = {}
    for p in csvs:
        pid, side = parse_filename(p)
        if pid is None:
            print(f'  WARNING: unrecognized filename {p.name}, skipping')
            continue
        by_patient.setdefault(pid, {})[side] = p

    print(f'Patients: {len(by_patient)} (expect 111)')

    incomplete = [(pid, list(sides.keys())) for pid, sides in by_patient.items()
                  if len(sides) != 2]
    if incomplete:
        print(f'  WARNING: {len(incomplete)} patients missing a side:')
        for pid, slist in incomplete[:5]:
            print(f'    {pid}: only have {slist}')

    rows = []
    for i, (pid, sides) in enumerate(sorted(by_patient.items())):
        if i % 25 == 0:
            print(f'  [{i+1}/{len(by_patient)}] {pid}')

        row = {'Patient ID': pid}
        per_side_out = {}
        for side, path in sides.items():
            df = pd.read_csv(path)
            per_side_out[side] = aggregate_one_side(df)

        # Max Frame from Left side (or Right if Left missing)
        ref_side = 'Left' if 'Left' in per_side_out else 'Right'
        for task in TASKS:
            row[f'{task}_Max Frame'] = per_side_out[ref_side].get(
                f'{task}_Max Frame', np.nan)

        for side, side_out in per_side_out.items():
            for task in TASKS:
                for au in AU_ORDER:
                    for suffix in ('_r', '_r_static', '_r_static_p10'):
                        src_key = f'{task} {au}{suffix}'
                        dst_key = f'{task}_{side} {au}{suffix}'
                        row[dst_key] = side_out.get(src_key, np.nan)
        rows.append(row)

    out_df = pd.DataFrame(rows)
    out_df.to_csv(OUT_CSV, index=False)
    print(f'\nWrote {len(out_df)} rows × {len(out_df.columns)} cols to {OUT_CSV}')

    # Spot checks vs Windows team's reported values
    print(f'\nSpot checks vs Windows team report:')
    spot = [
        ('IMG_0443', 'Left', 'BL', 'AU17',
         'static p10 should be near 0 (control)'),
        ('IMG_0443', 'Left', 'BL', 'AU45',
         'static max ~1.12 per spot-check, default ~0.88'),
        ('20240903_iOS', 'Left', 'BL', 'AU17',
         'static max ~2.05, p10 ~0.79 (synkinetic hypertonus)'),
        ('20240903_iOS', 'Left', 'BL', 'AU45',
         'static ~1.46 (vs default 0.52)'),
    ]
    for pid, side, task, au, note in spot:
        match = out_df[out_df['Patient ID'] == pid]
        if len(match) == 0:
            # Try the longer iOS form (20240903_HHMMSS_iOS)
            match = out_df[out_df['Patient ID'].str.contains(pid, regex=False)]
        if len(match) == 0:
            print(f'  {pid} {side} {task} {au}: PATIENT NOT FOUND ({note})')
            continue
        if len(match) > 1:
            print(f'  {pid} {side} {task} {au}: {len(match)} matching patients;'
                  f' showing first ({match.iloc[0]["Patient ID"]})')
        m = match.iloc[0]
        r = m.get(f'{task}_{side} {au}_r', np.nan)
        s_max = m.get(f'{task}_{side} {au}_r_static', np.nan)
        s_p10 = m.get(f'{task}_{side} {au}_r_static_p10', np.nan)
        print(f'  {m["Patient ID"][:30]:<30} {side} {task} {au}: '
              f'default_max={r:.3f}  static_max={s_max:.3f}  '
              f'static_p10={s_p10:.3f}  [{note}]')

    # Column count sanity
    expected_cols = 1 + len(TASKS) + len(TASKS) * 2 * len(AU_ORDER) * 3
    print(f'\nColumn count: {len(out_df.columns)} (expected {expected_cols})')


if __name__ == '__main__':
    main()
