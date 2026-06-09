#!/usr/bin/env python3
"""Bridge: curated per-frame data -> baseline-referenced ANALYSIS MATRIX.

One row per (patient, hemiface, action), with every panel AU expressed three ways:
  {AU}_act  mean AU over the action's CURATED representative frames
  {AU}_bl   mean AU over the patient's CURATED resting-baseline frames
  {AU}_dev  = act - bl   (the "deviation from rest" — the substrate every phenotype
             metric reads: synkinesis = involuntary off-target dev, paralysis =
             reduced on-target dev, hypertonicity lives in the baseline itself)

Hygiene mirrors the auto-curator ground truth:
  - only status=done actions (human-confirmed frames)
  - not_performed EXCLUDED (the task wasn't done -> no representative frames)
  - abnormal KEPT, flagged (the task WAS performed, just atypically)
  - BL excluded as a row (it is the reference); bl_quality carried on every row
    (neutral / elevated / smiling) so a caveated baseline is visible downstream.

PRIVACY: the output CSV is patient data -> gitignored (pilot15_*.csv). Only this
script is committed.

Run:  python3 "S3 Data Analysis/build_analysis_matrix.py"
"""
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# config/data_manager live in the S2.5 worktree
_CUR = os.path.abspath(os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "..", ".claude", "worktrees", "pilot15-static-mode-audit", "S2.5 Frame Curator"))
if _CUR not in sys.path:
    sys.path.insert(0, _CUR)
import config                       # noqa: E402
from data_manager import DataManager  # noqa: E402

OUT = Path(__file__).resolve().parent / "pilot15_curated_analysis_matrix.csv"
AUS = list(config.AU_ORDER)


def frame_means(df, aucols, frames):
    sub = df[df['frame'].isin(frames)]
    return sub[aucols].mean()


def build():
    dm = DataManager()
    rows = []
    skipped = []
    for pid, node in dm.curation['patients'].items():
        if not isinstance(node, dict):
            continue
        bl = node.get('BL')
        if not isinstance(bl, dict) or bl.get('status') != 'done':
            skipped.append((pid, 'no curated BL'))
            continue
        bl_frames = set(int(f) for f in bl.get('kept', []))
        if not bl_frames:
            skipped.append((pid, 'empty BL'))
            continue
        bl_quality = dm.bl_quality(pid) or 'neutral'
        for side in ('left', 'right'):
            csv = config.PER_FRAME_DIR / f'{pid}_{side}_mirrored_coded.csv'
            if not csv.exists():
                continue
            df = pd.read_csv(csv)
            aucols = [f'{a}_r' for a in AUS if f'{a}_r' in df.columns]
            bl_mean = frame_means(df, aucols, bl_frames)
            for action, st in node.items():
                if action == 'BL' or not isinstance(st, dict):
                    continue
                if st.get('status') != 'done':
                    continue
                flags = st.get('flags', [])
                if 'not_performed' in flags:
                    continue
                kept = set(int(f) for f in st.get('kept', []))
                if not kept:
                    continue
                act_mean = frame_means(df, aucols, kept)
                dev = act_mean - bl_mean
                row = {'pid': pid, 'side': side, 'action': action,
                       'abnormal': 'abnormal' in flags, 'bl_quality': bl_quality,
                       'bl_n': len(bl_frames), 'act_n': len(kept)}
                for a in AUS:
                    c = f'{a}_r'
                    if c in df.columns:
                        row[f'{a}_dev'] = round(float(dev[c]), 4)
                        row[f'{a}_act'] = round(float(act_mean[c]), 4)
                        row[f'{a}_bl'] = round(float(bl_mean[c]), 4)
                rows.append(row)
    m = pd.DataFrame(rows)
    m.to_csv(OUT, index=False)
    return m, skipped


def report(m, skipped):
    print(f"\nAnalysis matrix: {len(m)} rows -> {OUT.name}")
    print(f"  patients={m['pid'].nunique()}  actions={sorted(m['action'].unique())}")
    print(f"  abnormal-flagged rows={int(m['abnormal'].sum())}  "
          f"bl_quality={dict(m.drop_duplicates('pid')['bl_quality'].value_counts())}")
    if skipped:
        print(f"  skipped patients: {skipped}")

    # ---- SANITY: each task should drive its OWN target AUs ABOVE rest. The mean
    # target-AU deviation should be strongly positive, on a large fraction of rows.
    print("\nSanity — target-AU deviation (a clean bridge = strongly positive):")
    print(f"  {'act':>4} {'n':>4} {'target AUs':<18} {'mean Σdev':>9} {'% rows>0':>9}")
    for action in ['RE', 'ES', 'ET', 'SS', 'BS', 'SE', 'SO', 'WN', 'FR', 'BK', 'BC']:
        sub = m[m['action'] == action]
        if sub.empty:
            continue
        tgt = config.SIGNAL_AUS.get(action) if hasattr(config, 'SIGNAL_AUS') else None
        tgt = tgt or config.FACS_TASK_AUS.get(action, [])
        cols = [f'{a}_dev' for a in tgt if f'{a}_dev' in sub.columns]
        if not cols:
            continue
        s = sub[cols].sum(axis=1)
        print(f"  {action:>4} {len(sub):>4} {'+'.join(tgt):<18} "
              f"{s.mean():>9.2f} {100*(s > 0).mean():>8.0f}%")


if __name__ == '__main__':
    m, skipped = build()
    report(m, skipped)
