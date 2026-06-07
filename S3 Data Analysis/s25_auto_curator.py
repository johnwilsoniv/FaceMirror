"""
S25 Auto-Curator — derive an interpretable frame-selection rule from the human
ground truth produced in the S2.5 Frame Curator.

DESIGN PRINCIPLES (from the curating clinician, NOT from raw correlations):
  1. Keep frames where the REQUESTED ACTION is being performed at/near maximal
     expression  ->  task-AU activation near the per-instance peak.
  2. Reject BLINKS (AU45 high)  ->  EXCEPT for eye-closure tasks (ES/ET/BK)
     where eye closure is the target itself.

We deliberately do NOT encode "reject high AU17 during BS" style rules: those are
spurious correlations (a frame where the patient wasn't smiling happens to have
some other AU up), not the clinician's reasoning. Encoding them overfits n=10.

Archetypes:
  - REST (BL): keep quiet frames; reject blinks and gross movement.
  - EXPRESSION w/ task AUs (BS,SS,RE,ES,ET,SE,SO,WN,FR): keep near-peak task
    activation; reject blinks unless the task is eye-closure.
  - NO PANEL AU (PL,BC,LT): defining AU (AU18/AU33-34/AU16) absent from the
    17-AU panel -> fall back to held-expression position; flagged as low-confidence
    auto-curation.

Validation: leave-one-patient-out CV. Objective: F0.5 (precision-leaning, per the
"purer phenotype" choice) with an implicit recall floor from the beta weighting.
"""
import json
import os
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

# The curator's config/data_manager live in the S2.5 worktree; add to path so
# this script can run from anywhere.
_CURATOR_DIR = ("/Users/johnwilsoniv/Documents/SplitFace Open3/.claude/worktrees/"
                "pilot15-static-mode-audit/S2.5 Frame Curator")
if _CURATOR_DIR not in sys.path:
    sys.path.insert(0, _CURATOR_DIR)
os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')

import config
from data_manager import DataManager

# FACS-correct task AUs per action. The 8 analyzed actions reuse the validated
# in-set; the 6 others get their defining AU where it exists in the panel.
PANEL = set(config.AU_ORDER)
FACS_TASK_AUS = {
    'BS': ['AU06', 'AU07', 'AU10', 'AU12', 'AU14', 'AU23', 'AU25'],
    'SS': ['AU06', 'AU07', 'AU10', 'AU12', 'AU14', 'AU23', 'AU25', 'AU45'],
    'RE': ['AU01', 'AU02', 'AU05'],
    'ES': ['AU07', 'AU45'],
    'ET': ['AU04', 'AU06', 'AU07', 'AU09', 'AU10', 'AU14', 'AU23', 'AU26', 'AU45'],
    'SE': ['AU06', 'AU07', 'AU10', 'AU12', 'AU14', 'AU25', 'AU26'],
    'SO': ['AU01', 'AU04', 'AU06', 'AU07', 'AU10', 'AU12', 'AU14', 'AU25', 'AU26'],
    'WN': ['AU09'],          # nose wrinkler (defining)
    'FR': ['AU04'],          # brow lowerer (defining)
    'BK': ['AU45'],          # eye closure
    # No panel AU for these (defining AU absent): position-only fallback.
    'PL': [],                # pucker = AU18 (absent)
    'BC': [],                # blow cheeks = AU33/34 (absent)
    'LT': [],                # lower teeth = AU16 (absent)
}
EYE_TASKS = {'ES', 'ET', 'BK'}          # high AU45 is the TARGET, never a blink
# Position-only fallback: defining AU absent AND no CV-confirmed proxy. BC moved
# OUT (its AU12+AU17 proxy beats position under CV); LT stays (DLI/AU16 absent and
# AU15/DAO did not beat position-only).
NO_PANEL = {'PL', 'LT'}
REST = 'BL'


def _side_df(pid, side):
    p = config.PER_FRAME_DIR / f'{pid}_{side}_mirrored_coded.csv'
    if not p.exists():
        return None
    df = pd.read_csv(p)
    df['action'] = df['action'].astype(str).str.strip()
    return df


def stronger_side_sub(pid, action):
    """Return the (sorted) per-frame sub-frame from the hemiface with the higher
    key-AU peak for this action. For frame TIMING the stronger side shows the
    clearest onset/plateau; on a paretic side the AU barely moves. ~40% of our
    patients are right-dominant, so always-left reads the wrong side for them."""
    best, best_pk = None, -1.0
    aus = [a for a in FACS_TASK_AUS.get(action, [])]
    for side in ('left', 'right'):
        df = _side_df(pid, side)
        if df is None:
            continue
        sub = df[df['action'] == action].sort_values('frame').copy()
        if sub.empty:
            continue
        cols = [f'{a}_r' for a in aus if f'{a}_r' in sub.columns]
        pk = sub[cols].sum(axis=1).max() if cols else 0.0
        if pk > best_pk:
            best_pk, best = pk, sub
    return best


def load_ground_truth(dm, use_stronger_side=True):
    """Per (pid, action) frames with the human keep label, for status=done only."""
    rows = []
    for pid, node in dm.curation['patients'].items():
        if not isinstance(node, dict):
            continue
        for action, st in node.items():
            if not isinstance(st, dict) or st.get('status') != 'done':
                continue
            # EXCLUDE 'not_performed': the action wasn't done, so its kept set isn't a
            # representative-frame selection (it's leftover auto/whole-range or empty)
            # -> not valid frame-selection ground truth. Compliance is a separate gate.
            # KEEP 'abnormal': the action WAS performed (abnormally); kept is valid.
            if 'not_performed' in st.get('flags', []):
                continue
            if (use_stronger_side and action in STRONGER_SIDE_ACTIONS
                    and FACS_TASK_AUS.get(action)):
                sub = stronger_side_sub(pid, action)
            else:
                df = dm.get_frame_df(pid)      # default left (CV-best elsewhere)
                sub = (df[df['action'] == action].sort_values('frame').copy()
                       if df is not None else None)
            if sub is None or sub.empty:
                continue
            kept = set(st.get('kept', []))
            rows.append((pid, action, sub, kept))
    return rows


# Validated focused task signals (CV-confirmed dyads that beat the full in-set
# sum on the n=10 ground truth). Only the solid n=10 actions; SO/LT stay full-sum
# until the +30 expansion confirms their combos.
SIGNAL_AUS = {
    'BS': ['AU10', 'AU25'],
    'SS': ['AU12', 'AU23'],
    'SO': ['AU17', 'AU25', 'AU26'],  # orbicularis oris (AU18) absent -> open-"O" articulation
                                     # proxy: jaw-drop+lips-part+chin (CV 0.82, +0.23 vs in-set)
    'BC': ['AU12', 'AU17'],    # cheek puff (AU33/34) absent -> CV-confirmed proxy dyad (+0.06)
}


def frame_features(sub, action):
    """Return arrays: task_activation, au45, pos, total_activity (per frame)."""
    aus = [au for au in (SIGNAL_AUS.get(action) or FACS_TASK_AUS.get(action, []))
           if f'{au}_r' in sub.columns]
    cols = [f'{au}_r' for au in aus]
    task = sub[cols].sum(axis=1).values if cols else np.zeros(len(sub))
    au45 = sub['AU45_r'].values if 'AU45_r' in sub.columns else np.zeros(len(sub))
    n = len(sub)
    pos = np.linspace(0, 1, n) if n > 1 else np.array([1.0])
    allcols = [f'{au}_r' for au in config.AU_ORDER if f'{au}_r' in sub.columns]
    total = sub[allcols].sum(axis=1).values
    return task, au45, pos, total


def _smooth(x, w):
    if w <= 1 or len(x) < w:
        return np.asarray(x, float)
    k = np.ones(w) / w
    return np.convolve(np.asarray(x, float), k, mode='same')


def _longest_run(mask):
    """[i0,i1) of the longest contiguous True run in a bool array (or None)."""
    best = None
    i = 0
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


def predict_keep(sub, action, params):
    """Apply the interpretable rule -> boolean keep mask for the frames in `sub`."""
    task, au45, pos, total = frame_features(sub, action)
    n = len(sub)
    if action == REST:
        # rest: keep quiet frames (low overall activity) and no blink
        return (au45 <= params['blink']) & (total <= params['rest_move'])
    if action in NO_PANEL:
        # no measurable task AU -> held-expression position proxy
        keep = pos >= params['pos']
        if action not in EYE_TASKS:
            keep &= (au45 <= params['blink'])
        return keep

    # ---- expression archetype ----
    if params.get('mode') == 'plateau':
        # Sustained-plateau detector, encoding the clinician's 3 concepts:
        #  (1) delayed onset / (3) early-stop -> take the LONGEST contiguous run
        #      held above threshold on the SMOOTHED key-AU trace (not every
        #      transient frame above it);
        #  (2) blink -> drop high-AU45 frames within the run (gentle for smiles);
        #  compliance gate -> if the smoothed peak is below abs_floor, the action
        #      was weak/non-performed: tighten to a higher relative threshold.
        ss = _smooth(task, int(params.get('smooth', 5)))
        peak = ss.max() if ss.size else 0.0
        if peak <= 0:
            return np.ones(n, dtype=bool)
        rel = params['rel']
        weak = peak < params.get('abs_floor', 0.0)
        if weak:
            rel = max(rel, params.get('rel_strict', 0.75))
        run = _longest_run(ss >= rel * peak)
        keep = np.zeros(n, dtype=bool)
        if run is not None:
            keep[run[0]:run[1]] = True
        if action not in EYE_TASKS:
            keep &= (au45 < params['blink'])
        return keep

    # legacy frac-of-peak mode (kept for the actions where it already wins)
    peak = task.max() if task.size else 0.0
    if peak <= 0:
        return np.ones(n, dtype=bool)
    keep = task >= params['frac'] * peak
    if params.get('pos', 0.0) > 0.0:
        keep &= (pos >= params['pos'])
    if action not in EYE_TASKS:
        keep &= (au45 <= params['blink'])
    return keep


def score(pred, human_mask, beta=0.5):
    tp = int((pred & human_mask).sum())
    fp = int((pred & ~human_mask).sum())
    fn = int((~pred & human_mask).sum())
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    b2 = beta * beta
    denom = b2 * prec + rec
    fbeta = (1 + b2) * prec * rec / denom if denom else 0.0
    return prec, rec, fbeta, tp, fp, fn


# Parameter grids per archetype.
GRID_FRAC = [round(x, 2) for x in np.arange(0.30, 0.96, 0.05)]
GRID_BLINK = [0.5, 0.8, 1.0, 1.5, 2.0, 1e9]      # 1e9 == no blink filter
GRID_POS = [round(x, 2) for x in np.arange(0.0, 0.71, 0.1)]
GRID_RESTMOVE = [3.0, 5.0, 8.0, 12.0, 1e9]
# plateau detector grids
GRID_REL = [round(x, 2) for x in np.arange(0.45, 0.81, 0.05)]
GRID_SMOOTH = [3, 5, 7]
GRID_ABSFLOOR = [0.0, 3.0, 5.0, 8.0]             # compliance gate (raw AU-sum)
GRID_RELSTRICT = [0.75, 0.85]                    # tightened rel when weak
# Actions where the plateau detector + focused dyad BEAT frac-of-peak under
# leave-one-patient-out CV: BS (AU10+AU25) and SS (AU12+AU23), both ~0.84-0.86.
# SO/SE are brief phonemes -> frac-of-peak. CV-decided, confirmed by combo search.
PLATEAU_ACTIONS = {'BS', 'SS'}

# Stronger-side (higher key-AU peak) reading helped BK and both smiles (BS,SS)
# under CV but slightly hurt RE (already near-perfect on the default side).
# Apply it ONLY where it was proven, not universally.
STRONGER_SIDE_ACTIONS = {'BK', 'BS', 'SS'}


def candidate_params(action):
    """Yield (params dict) candidates appropriate to the action's archetype."""
    if action == REST:
        for b in GRID_BLINK:
            for m in GRID_RESTMOVE:
                yield {'blink': b, 'rest_move': m}
    elif action in NO_PANEL:
        for p in GRID_POS:
            for b in (GRID_BLINK if action not in EYE_TASKS else [1e9]):
                yield {'pos': p, 'blink': b}
    elif action in PLATEAU_ACTIONS:
        blinks = [1.5, 2.0, 1e9]                  # gentle for smiles (per decision)
        for rel in GRID_REL:
            for sm in GRID_SMOOTH:
                for af in GRID_ABSFLOOR:
                    for b in blinks:
                        rstrict = [rel] if af == 0.0 else GRID_RELSTRICT
                        for rs in rstrict:
                            yield {'mode': 'plateau', 'rel': rel, 'smooth': sm,
                                   'abs_floor': af, 'rel_strict': rs, 'blink': b}
    else:
        blinks = GRID_BLINK if action not in EYE_TASKS else [1e9]
        for fr in GRID_FRAC:
            for b in blinks:
                for p in GRID_POS:           # position term: 0.0 == disabled
                    yield {'frac': fr, 'blink': b, 'pos': p}


def main():
    dm = DataManager()
    rows = load_ground_truth(dm)
    by_action = defaultdict(list)
    for pid, action, sub, kept in rows:
        by_action[action].append((pid, sub, kept))

    print(f"Ground truth: {len(rows)} (pid,action) · "
          f"{len(set(r[0] for r in rows))} patients\n")

    print(f"{'act':>4} {'inst':>4} {'best params':<26} "
          f"{'CV prec':>7} {'CV rec':>6} {'CV F0.5':>7} {'cur F0.5':>8}")
    results = {}
    for action in ['BL', 'RE', 'ES', 'ET', 'BS', 'SS', 'SE', 'SO',
                   'WN', 'FR', 'BK', 'PL', 'BC', 'LT']:
        insts = by_action.get(action, [])
        if not insts:
            continue
        pids = sorted(set(p for p, _, _ in insts))

        # ----- leave-one-patient-out CV: for each held-out patient, pick the
        # params that maximize pooled F0.5 on the OTHER patients, then score the
        # held-out one. Report mean held-out precision/recall/F0.5. -----
        cands = list(candidate_params(action))
        heldout = []
        for test_pid in pids:
            train = [(s, k) for p, s, k in insts if p != test_pid]
            test = [(s, k) for p, s, k in insts if p == test_pid]
            if not train or not test:
                continue
            best, best_f = None, -1
            for prm in cands:
                tp = fp = fn = 0
                for sub, kept in train:
                    pred = predict_keep(sub, action, prm)
                    hum = sub['frame'].isin(kept).values
                    _, _, _, a, b, c = score(pred, hum)
                    tp += a; fp += b; fn += c
                prec = tp / (tp + fp) if tp + fp else 0
                rec = tp / (tp + fn) if tp + fn else 0
                f = (1.25 * prec * rec / (0.25 * prec + rec)) if (0.25*prec+rec) else 0
                if f > best_f:
                    best_f, best = f, prm
            # score held-out patient with train-selected params
            tp = fp = fn = 0
            for sub, kept in test:
                pred = predict_keep(sub, action, best)
                hum = sub['frame'].isin(kept).values
                _, _, _, a, b, c = score(pred, hum)
                tp += a; fp += b; fn += c
            prec = tp / (tp + fp) if tp + fp else 0
            rec = tp / (tp + fn) if tp + fn else 0
            f = (1.25 * prec * rec / (0.25 * prec + rec)) if (0.25*prec+rec) else 0
            heldout.append((prec, rec, f))

        cv_p = np.mean([h[0] for h in heldout]) if heldout else float('nan')
        cv_r = np.mean([h[1] for h in heldout]) if heldout else float('nan')
        cv_f = np.mean([h[2] for h in heldout]) if heldout else float('nan')

        # final params: refit on ALL instances (what we'd ship)
        best, best_f = None, -1
        for prm in cands:
            tp = fp = fn = 0
            for _, sub, kept in insts:
                pred = predict_keep(sub, action, prm)
                hum = sub['frame'].isin(kept).values
                _, _, _, a, b, c = score(pred, hum)
                tp += a; fp += b; fn += c
            prec = tp / (tp + fp) if tp + fp else 0
            rec = tp / (tp + fn) if tp + fn else 0
            f = (1.25 * prec * rec / (0.25 * prec + rec)) if (0.25*prec+rec) else 0
            if f > best_f:
                best_f, best = f, prm
        results[action] = {'params': best, 'cv_prec': cv_p, 'cv_rec': cv_r,
                           'cv_f05': cv_f, 'fit_f05': best_f, 'n_inst': len(insts)}

        # current plateau selector F0.5 for comparison
        tp = fp = fn = 0
        for pid_, sub, kept in insts:
            auto = set(dm.auto_keep_frames(pid_, action))
            hum = sub['frame'].isin(kept).values
            pred = sub['frame'].isin(auto).values
            _, _, _, a, b, c = score(pred, hum)
            tp += a; fp += b; fn += c
        cprec = tp / (tp + fp) if tp + fp else 0
        crec = tp / (tp + fn) if tp + fn else 0
        cur_f = (1.25 * cprec * crec / (0.25 * cprec + crec)) if (0.25*cprec+crec) else 0

        pstr = ', '.join(f'{k}={v}' for k, v in best.items())
        print(f"{action:>4} {len(insts):>4} {pstr:<26} "
              f"{cv_p:>7.2f} {cv_r:>6.2f} {cv_f:>7.2f} {cur_f:>8.2f}")

    # save fitted params
    out = {a: dict(r['params']) for a, r in results.items()}
    for a, aus in SIGNAL_AUS.items():           # record the focused signal
        if a in out:
            out[a]['signal_aus'] = aus
    Path('/tmp/s25_auto_params.json').write_text(json.dumps(out, indent=2))
    print("\nfitted params -> /tmp/s25_auto_params.json")
    return results


if __name__ == '__main__':
    main()
