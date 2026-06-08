#!/usr/bin/env python3
"""
Pilot 15 — Asymmetry-only discriminator test.

Question: does |L_static_p10 - R_static_p10| at BL discriminate hypertonus from
anatomy on its own, without needing the multi-prong combinator?

Premise: post-paralysis aberrant regeneration is usually unilateral, anatomy
is usually bilateral. So asymmetry should be informative.

Counterexamples to keep in mind:
  - IMG_0422 control has AU17 |L-R| = 2.045 (extreme anatomic asymmetry)
  - Bilateral hypertonus exists but is uncommon

Method:
  Per AU, compute |L_static_p10 - R_static_p10| at BL distributions for:
    (a) controls (n=13)
    (b) FP-Key Hypertonicity-flagged cases (target: AU14 buccinator)
    (c) FP-Key Mentalis-Synkinesis-flagged cases (target: AU17)
    (d) FP-Key Oral-Ocular-Synkinesis cases (target: AU45)
    (e) Other cases (no relevant FP flag for that AU)

  If hypertonic cases have systematically higher asymmetry than controls AND
  non-flagged cases for the corresponding AU, asymmetry is a real signal.

Outputs: pilot15_asymmetry_distribution.csv + console table.
"""
from pathlib import Path
import pandas as pd
import numpy as np
from scipy.stats import mannwhitneyu

DATA = Path("/Users/johnwilsoniv/Documents/SplitFace Open3/S3 Data Analysis")
COMBINED = DATA / "recoded_rerun_dual_v1316_combined_results.csv"
KEY = DATA / "FPRS FP Key.csv"
CTRL_DIR = Path("/Users/johnwilsoniv/Documents/SplitFace Open3/"
                "S Data/Normal Cohort")

OUT = DATA / "pilot15_asymmetry_distribution.csv"

AU_OF_INTEREST = ['AU04', 'AU07', 'AU14', 'AU15', 'AU17', 'AU23', 'AU45']

# Map AU → relevant FP-Key columns (which patients should show this AU's tonus)
AU_TO_FP = {
    'AU14': ('Hypertonicity Left', 'Hypertonicity Right'),     # buccinator
    'AU17': ('Mentalis Synkinesis Left', 'Mentalis Synkinesis Right'),
    'AU45': ('Oral-Ocular Synkinesis Left', 'Oral-Ocular Synkinesis Right'),
    'AU07': ('Hypertonicity Left', 'Hypertonicity Right'),     # orbicularis oculi (proxy)
    'AU04': (None, None),  # no FP-Key field for corrugator hypertonus
    'AU15': (None, None),  # no FP-Key field for DAO
    'AU23': (None, None),  # no FP-Key field for lip pressor
}


def patient_au_asym(combined, pid, au, side_p10_cols=None):
    """Per (patient, AU) asymmetry: |L_static_p10 - R_static_p10| at BL."""
    row = combined[combined['Patient ID'].astype(str) == pid]
    if len(row) == 0:
        return np.nan
    row = row.iloc[0]
    l = float(row.get(f'BL_Left {au}_r_static_p10', np.nan))
    r = float(row.get(f'BL_Right {au}_r_static_p10', np.nan))
    if pd.isna(l) or pd.isna(r):
        return np.nan
    return abs(l - r)


def fp_flag(key, pid, left_col, right_col, side='any'):
    """Return whether FP Key flags this patient for a given column pair."""
    if left_col is None:
        return None
    row = key[key['Patient'].astype(str) == pid]
    if len(row) == 0:
        return False
    row = row.iloc[0]
    l = str(row.get(left_col, '')).strip()
    r = str(row.get(right_col, '')).strip()
    has_l = l in ('Yes', 'Complete', 'Partial')
    has_r = r in ('Yes', 'Complete', 'Partial')
    if side == 'any': return has_l or has_r
    if side == 'left': return has_l
    if side == 'right': return has_r
    return False


def main():
    combined = pd.read_csv(COMBINED)
    key = pd.read_csv(KEY, encoding='utf-8-sig')
    controls = sorted(p.stem for p in CTRL_DIR.glob('IMG_*.MOV'))
    cases = [pid for pid in combined['Patient ID'].astype(str).tolist()
             if pid not in set(controls)]
    print(f'Controls: {len(controls)}, Cases: {len(cases)}')

    rows = []
    print('\n' + '=' * 90)
    print(f'{"AU":<6}{"target FP flag":<35}{"cohort":<14}'
          f'{"n":>5}{"med":>8}{"p75":>8}{"p90":>8}{"p95":>8}{"max":>8}')
    print('=' * 90)

    for au in AU_OF_INTEREST:
        l_col, r_col = AU_TO_FP[au]

        # (a) controls
        ctrl_asym = [patient_au_asym(combined, pid, au) for pid in controls]
        ctrl_asym = pd.Series(ctrl_asym).dropna()

        # (b) FP-flagged cases for this AU
        if l_col is not None:
            flagged = [pid for pid in cases if fp_flag(key, pid, l_col, r_col)]
            unflagged = [pid for pid in cases if not fp_flag(key, pid, l_col, r_col)]
            flagged_asym = pd.Series([patient_au_asym(combined, pid, au)
                                       for pid in flagged]).dropna()
            unflagged_asym = pd.Series([patient_au_asym(combined, pid, au)
                                         for pid in unflagged]).dropna()
        else:
            flagged = []
            flagged_asym = pd.Series([])
            unflagged = cases
            unflagged_asym = pd.Series([patient_au_asym(combined, pid, au)
                                         for pid in unflagged]).dropna()

        target = (f'{l_col}/{r_col}' if l_col else '(no FP target)')[:34]

        for cohort_label, vals in [
            ('control', ctrl_asym),
            ('FP-flagged', flagged_asym),
            ('FP-unflagged', unflagged_asym),
        ]:
            if len(vals) == 0:
                continue
            med = float(vals.median())
            p75 = float(vals.quantile(0.75))
            p90 = float(vals.quantile(0.9))
            p95 = float(vals.quantile(0.95))
            mx = float(vals.max())
            print(f'{au:<6}{target:<35}{cohort_label:<14}{len(vals):>5}'
                  f'{med:>8.2f}{p75:>8.2f}{p90:>8.2f}{p95:>8.2f}{mx:>8.2f}')
            rows.append({
                'au': au, 'fp_target': target, 'cohort': cohort_label,
                'n': len(vals), 'median': med, 'p75': p75, 'p90': p90,
                'p95': p95, 'max': mx,
            })
        print()

    # Statistical test: do FP-flagged cases have higher asym than controls
    # AND than FP-unflagged cases?
    print('=' * 90)
    print('STATISTICAL TEST: Mann-Whitney U (one-sided, FP-flagged > comparator)')
    print('=' * 90)
    print(f'{"AU":<6}{"comparison":<35}{"U":>10}{"p":>10}{"signif?":>10}')
    print('-' * 71)
    for au in AU_OF_INTEREST:
        l_col, r_col = AU_TO_FP[au]
        if l_col is None:
            continue
        flagged = [pid for pid in cases if fp_flag(key, pid, l_col, r_col)]
        unflagged = [pid for pid in cases if not fp_flag(key, pid, l_col, r_col)]
        ctrl_asym = pd.Series([patient_au_asym(combined, pid, au)
                               for pid in controls]).dropna()
        flagged_asym = pd.Series([patient_au_asym(combined, pid, au)
                                   for pid in flagged]).dropna()
        unflagged_asym = pd.Series([patient_au_asym(combined, pid, au)
                                     for pid in unflagged]).dropna()
        for comp_label, comp_vals in [
            ('flagged > controls', ctrl_asym),
            ('flagged > unflagged-cases', unflagged_asym),
        ]:
            if len(flagged_asym) < 3 or len(comp_vals) < 3:
                continue
            try:
                u, p = mannwhitneyu(flagged_asym, comp_vals, alternative='greater')
                signif = '*' if p < 0.05 else 'ns'
                print(f'{au:<6}{comp_label:<35}{u:>10.1f}{p:>10.4f}{signif:>10}')
            except Exception as e:
                print(f'{au:<6}{comp_label:<35}  err: {e}')

    # Compute simple discriminator: asym > control p95 = predicted positive
    # Sens/spec per AU vs FP-Key flag.
    print('\n' + '=' * 90)
    print('SIMPLE DISCRIMINATOR: asym > control_p95 → predicted hypertonic')
    print('=' * 90)
    print(f'{"AU":<6}{"ctrl_p95":>10}{"thr_used":>10}{"sens":>8}{"spec":>8}'
          f'{"tp":>5}{"fp":>5}{"fn":>5}{"tn":>5}{"n":>5}')
    print('-' * 70)
    for au in AU_OF_INTEREST:
        l_col, r_col = AU_TO_FP[au]
        if l_col is None:
            continue
        ctrl_asym = pd.Series([patient_au_asym(combined, pid, au)
                               for pid in controls]).dropna()
        if len(ctrl_asym) == 0:
            continue
        # Use control p95 as threshold (or control max if p95 too low)
        ctrl_p95 = float(ctrl_asym.quantile(0.95))
        ctrl_max = float(ctrl_asym.max())
        thr = max(ctrl_p95, 0.3)  # don't go below clinical relevance
        tp = fp = fn = tn = 0
        for pid in cases:
            asym = patient_au_asym(combined, pid, au)
            if pd.isna(asym):
                continue
            label = fp_flag(key, pid, l_col, r_col)
            pred = asym > thr
            if label and pred: tp += 1
            elif label and not pred: fn += 1
            elif not label and pred: fp += 1
            else: tn += 1
        sens = tp / (tp + fn) if (tp + fn) > 0 else 0
        spec = tn / (tn + fp) if (tn + fp) > 0 else 0
        print(f'{au:<6}{ctrl_p95:>10.3f}{thr:>10.3f}{sens:>8.2f}{spec:>8.2f}'
              f'{tp:>5}{fp:>5}{fn:>5}{tn:>5}{tp+fp+fn+tn:>5}')

    pd.DataFrame(rows).to_csv(OUT, index=False)
    print(f'\nWrote {OUT.name}')


if __name__ == '__main__':
    main()
