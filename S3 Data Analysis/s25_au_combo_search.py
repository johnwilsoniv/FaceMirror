"""
S25 AU-combo search — does a DYAD or TRIAD of AUs identify the key expression
better than the full in-set sum, for the hard actions (BS/SS/SO/LT/...)?

Motivation: the current rule sums ALL in-set AUs as the "is the action happening"
signal. For smile/speech complexes that may dilute the signal with weak/noisy AUs.
A focused 2-3 AU combination might track the expression more cleanly.

Method (same discipline as s25_auto_curator):
  - For in-set actions: search subsets (size 1,2,3, + full) WITHIN the FACS in-set.
  - For no-panel actions (LT/PL/BC): search all 17 AUs (no anatomical prior).
  - Each combo's signal = SUM of its AUs (also try MIN for specificity).
  - Score by LEAVE-ONE-PATIENT-OUT CV F0.5 with inner param selection on train.
  - Adopt a combo ONLY if it beats the full-sum baseline on held-out patients.

This is a DIAGNOSTIC: it reports, it does not modify the curator.
"""
import json
import os
import sys
from collections import defaultdict
from itertools import combinations

import numpy as np
import pandas as pd

_CUR = ("/Users/johnwilsoniv/Documents/SplitFace Open3/.claude/worktrees/"
        "pilot15-static-mode-audit/S2.5 Frame Curator")
if _CUR not in sys.path:
    sys.path.insert(0, _CUR)
os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')

import config
from data_manager import DataManager
import s25_auto_curator as ac

HARD = ['SO', 'LT', 'BC']                # no-good-panel-AU actions to find a proxy for
ALL_AUS = list(config.AU_ORDER)
# Defining AU absent from the 17-panel (SO=orbicularis oris/AU18, LT=lower-lip
# depressor/AU16, BC=cheek puff/AU33-34) -> NO anatomical prior, so search all 17
# AUs for a proxy dyad/triad rather than restricting to the FACS in-set guess.
SEARCH_ALL = {'SO', 'LT', 'BC', 'PL'}

# rule type per action (reuse what the auto-curator settled on)
PLATEAU = {'BS', 'SS'}


def smooth(x, w):
    x = np.asarray(x, float)
    if w <= 1 or len(x) < w:
        return x
    return np.convolve(x, np.ones(int(w)) / int(w), mode='same')


def longest_run(mask):
    best, i = None, 0
    while i < len(mask):
        if mask[i]:
            j = i
            while j < len(mask) and mask[j]:
                j += 1
            if best is None or (j - i) > (best[1] - best[0]):
                best = (i, j)
            i = j
        else:
            i += 1
    return best


def predict(task, au45, pos, action, prm):
    """Keep mask from a precomputed task signal + the rule params."""
    eye = action in config.EYE_TASKS
    if action in PLATEAU:
        ss = smooth(task, prm['smooth'])
        peak = ss.max() if ss.size else 0.0
        if peak <= 0:
            return np.ones(len(task), bool)
        run = longest_run(ss >= prm['rel'] * peak)
        keep = np.zeros(len(task), bool)
        if run:
            keep[run[0]:run[1]] = True
        if not eye:
            keep &= (au45 < prm['blink'])
        return keep
    # frac-of-peak + position
    peak = task.max() if task.size else 0.0
    if peak <= 0:
        return np.ones(len(task), bool)
    keep = task >= prm['frac'] * peak
    if prm.get('pos', 0.0) > 0.0:
        keep &= (pos >= prm['pos'])
    if not eye:
        keep &= (au45 <= prm['blink'])
    return keep


def grid(action):
    if action in PLATEAU:
        for rel in [0.5, 0.6, 0.7, 0.8]:
            for sm in [3, 5]:
                for b in [1.5, 2.0, 1e9]:
                    yield {'rel': rel, 'smooth': sm, 'blink': b}
    else:
        for fr in [0.3, 0.45, 0.6, 0.75]:
            for p in [0.0, 0.3, 0.5]:
                for b in [0.8, 1.0, 2.0, 1e9]:
                    yield {'frac': fr, 'pos': p, 'blink': b}


def f05(tp, fp, fn):
    p = tp / (tp + fp) if tp + fp else 0
    r = tp / (tp + fn) if tp + fn else 0
    return (1.25 * p * r / (0.25 * p + r)) if (0.25 * p + r) else 0


def build_instances(dm, action):
    """[(pid, per-AU matrix df, au45, pos, human_mask_frames, kept)] for status=done."""
    insts = []
    for pid, node in dm.curation['patients'].items():
        if not isinstance(node, dict):
            continue
        st = node.get(action)
        if not isinstance(st, dict) or st.get('status') != 'done':
            continue
        if 'not_performed' in st.get('flags', []):
            continue                       # not a valid frame-selection instance
        if action in config.STRONGER_SIDE_ACTIONS and config.FACS_TASK_AUS.get(action):
            sub = ac.stronger_side_sub(pid, action)
        else:
            df = dm.get_frame_df(pid)
            sub = df[df['action'] == action].sort_values('frame') if df is not None else None
        if sub is None or sub.empty:
            continue
        frames = sub['frame'].astype(int).values
        aumat = {au: sub[f'{au}_r'].values for au in ALL_AUS if f'{au}_r' in sub.columns}
        au45 = sub['AU45_r'].values if 'AU45_r' in sub.columns else np.zeros(len(sub))
        n = len(sub)
        pos = np.linspace(0, 1, n) if n > 1 else np.array([1.0])
        kept = set(st.get('kept', []))
        hum = np.array([f in kept for f in frames])
        insts.append((pid, aumat, au45, pos, hum))
    return insts


def cv_score(insts, action, aus):
    """Leave-one-patient-out CV F0.5 for a given AU subset (sum signal)."""
    pids = [i[0] for i in insts]
    held = []
    for test in pids:
        train = [i for i in insts if i[0] != test]
        tst = [i for i in insts if i[0] == test]
        # inner: best params on train
        best, bestf = None, -1
        for prm in grid(action):
            tp = fp = fn = 0
            for _, aumat, au45, pos, hum in train:
                task = sum(aumat[a] for a in aus if a in aumat)
                if np.isscalar(task):
                    continue
                pred = predict(task, au45, pos, action, prm)
                tp += int((pred & hum).sum()); fp += int((pred & ~hum).sum())
                fn += int((~pred & hum).sum())
            f = f05(tp, fp, fn)
            if f > bestf:
                bestf, best = f, prm
        # score held-out
        tp = fp = fn = 0
        for _, aumat, au45, pos, hum in tst:
            task = sum(aumat[a] for a in aus if a in aumat)
            pred = predict(task, au45, pos, action, best)
            tp += int((pred & hum).sum()); fp += int((pred & ~hum).sum())
            fn += int((~pred & hum).sum())
        held.append(f05(tp, fp, fn))
    return float(np.mean(held)) if held else float('nan')


def main():
    dm = DataManager()
    for action in HARD:
        insts = build_instances(dm, action)
        if len(insts) < 3:
            print(f"\n{action}: only {len(insts)} instances — skip"); continue
        # candidate AU pool
        inset = config.FACS_TASK_AUS.get(action, [])
        if action in SEARCH_ALL:
            pool = list(ALL_AUS)                  # no anatomical prior -> search all 17
        else:
            pool = inset if inset else list(ALL_AUS)
        pool = [a for a in pool if a != 'AU45'] or pool   # AU45 handled as blink
        full = inset if inset else ALL_AUS               # baseline = current signal

        baseline = cv_score(insts, action, full)
        results = {}
        for k in (1, 2, 3):
            best_combo, best_f = None, -1
            for combo in combinations(pool, k):
                f = cv_score(insts, action, list(combo))
                if f > best_f:
                    best_f, best_combo = f, combo
            results[k] = (best_combo, best_f)

        print(f"\n=== {action} (n={len(insts)}, pool={'in-set' if inset else 'ALL-17'}) ===")
        print(f"  baseline full-sum {full}: CV F0.5 = {baseline:.3f}")
        for k, name in [(1, 'single'), (2, 'dyad'), (3, 'triad')]:
            combo, f = results[k]
            delta = f - baseline
            mark = '  <== beats baseline' if delta > 0.01 else ''
            print(f"  best {name:6}: {'+'.join(combo):<22} CV F0.5 = {f:.3f} "
                  f"(Δ{delta:+.3f}){mark}")


if __name__ == '__main__':
    main()
