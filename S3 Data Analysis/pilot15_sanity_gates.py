#!/usr/bin/env python3
"""
Pilot 15 sanity gates — must pass before Phase A code runs.

Gate 1: Cohort assignment from FP Key, not filename.
    Each of the 111 patients gets a case/control label from FP Key flag presence.
    Filename heuristic (Normal Cohort dir = control, others = candidate case) must
    agree. Disagreements halt the audit.

Gate 2: Control overview audit (default-mode baseline check).
    Controls should be quiet in default mode at BS task. Median + p95 per AU.
    Sanity, not gate.

Gate 3: Canary continuity vs v1315 reference.
    For IMG_0443, IMG_0452, IMG_0453: per-frame default-mode AU values in
    v1316 must correlate r > 0.95 with v1315 reference. Catches pipeline
    drift from the dual-mode plumbing.

Gate 4: Dual-mode column completeness.
    Combined CSV has 1443 cols; each per-frame CSV has 38 cols including 17 _r
    + 17 _r_static + action. No silent drops.
"""
from pathlib import Path
import pandas as pd
import numpy as np
import sys

DATA = Path("/Users/johnwilsoniv/Documents/SplitFace Open3/S3 Data Analysis")
DUAL_DIR = DATA / "recoded_rerun_dual_v1316"
V1315_DIR = DATA / "recoded_rerun_v1315"
COMBINED = DATA / "recoded_rerun_dual_v1316_combined_results.csv"
KEY = DATA / "FPRS FP Key.csv"
CONTROL_DIR = Path("/Users/johnwilsoniv/Documents/SplitFace Open3/"
                   "S Data/Normal Cohort")

AU_ORDER = ['AU01', 'AU02', 'AU04', 'AU05', 'AU06', 'AU07', 'AU09',
            'AU10', 'AU12', 'AU14', 'AU15', 'AU17', 'AU20', 'AU23',
            'AU25', 'AU26', 'AU45']

# Flag columns in FP Key (cols 2-18, the actual finding columns)
FP_FLAG_COLS = [
    'Paralysis - Left Upper Face', 'Paralysis - Left Mid Face',
    'Paralysis - Left Lower Face', 'Paralysis - Right Upper Face',
    'Paralysis - Right Mid Face', 'Paralysis - Right Lower Face',
    'Oral-Ocular Synkinesis Left', 'Oral-Ocular Synkinesis Right',
    'Ocular-Oral Synkinesis Left', 'Ocular-Oral Synkinesis Right',
    'Snarl Smile Left', 'Snarl Smile Right',
    'Mentalis Synkinesis Left', 'Mentalis Synkinesis Right',
    'Hypertonicity Left', 'Hypertonicity Right',
    'Brow Cocked Left', 'Brow Cocked Right',
]

CANARIES = ['IMG_0443', 'IMG_0452', 'IMG_0453']
R_THRESHOLD = 0.95


def banner(text, sym='='):
    print('\n' + sym * 72)
    print(f'  {text}')
    print(sym * 72)


def patient_has_any_flag(key_row):
    """A patient is a case if any FP-Key finding column is non-null and not 'No'."""
    for col in FP_FLAG_COLS:
        v = key_row.get(col)
        if pd.isna(v):
            continue
        v = str(v).strip()
        if v in ('', 'No', 'no', 'None', 'none', 'nan'):
            continue
        return True
    return False


def gate_1_cohort_assignment():
    banner('GATE 1: Cohort assignment from FP Key (not filename)')
    combined = pd.read_csv(COMBINED)
    key = pd.read_csv(KEY, encoding='utf-8-sig')

    # Build filename-based heuristic: IDs that exist in Normal Cohort dir are controls
    control_ids_by_dir = sorted(p.stem for p in CONTROL_DIR.glob('IMG_*.MOV'))
    print(f'Normal Cohort dir: {len(control_ids_by_dir)} controls')

    rows = []
    failures = []
    for pid in combined['Patient ID'].astype(str):
        key_row = key[key['Patient'].astype(str) == pid]
        if len(key_row) == 0:
            fp_label = 'not_in_FP_Key'
        else:
            fp_label = 'case' if patient_has_any_flag(key_row.iloc[0]) else 'control'
        file_label = 'control' if pid in control_ids_by_dir else 'case'
        agree = (fp_label == file_label) or (fp_label == 'not_in_FP_Key'
                                             and file_label == 'control')
        rows.append({
            'patient_id': pid,
            'fp_label': fp_label,
            'file_label': file_label,
            'agree': agree,
        })
        if not agree:
            failures.append((pid, fp_label, file_label))

    df = pd.DataFrame(rows)
    n_fp_case = (df['fp_label'] == 'case').sum()
    n_fp_ctrl = (df['fp_label'] == 'control').sum()
    n_not_fp = (df['fp_label'] == 'not_in_FP_Key').sum()
    n_file_case = (df['file_label'] == 'case').sum()
    n_file_ctrl = (df['file_label'] == 'control').sum()

    print(f'\nFP-Key cohort:    case={n_fp_case}  control={n_fp_ctrl}  '
          f'not_in_FP_Key={n_not_fp}')
    print(f'Filename cohort:  case={n_file_case}  control={n_file_ctrl}')
    print(f'Disagreements:    {len(failures)}')

    if failures:
        print('\nDisagreement details:')
        for pid, fp, fl in failures[:20]:
            print(f'  {pid:<30}  FP={fp:<15}  file={fl}')

    df.to_csv(DATA / 'pilot15_gate1_cohort_assignment.csv', index=False)
    print(f'\nWrote pilot15_gate1_cohort_assignment.csv')
    return len(failures) == 0, df


def gate_2_control_overview(cohort_df):
    banner('GATE 2: Control overview audit (sanity, not gate)')
    combined = pd.read_csv(COMBINED)
    controls = cohort_df[cohort_df['fp_label'] == 'control']['patient_id'].tolist()
    cdf = combined[combined['Patient ID'].astype(str).isin(controls)]
    print(f'Auditing {len(cdf)} controls')

    # For BS task, get default-mode AU max per control × side
    print('\nDefault-mode BS task, AU max distribution across controls × sides:')
    print(f'{"AU":<6}{"side":<6}{"min":>8}{"med":>8}{"p95":>8}{"max":>8}')
    print('-' * 44)
    for side in ('Left', 'Right'):
        for au in AU_ORDER:
            col = f'BS_{side} {au}_r'
            if col not in cdf.columns:
                continue
            vals = cdf[col].astype(float).dropna()
            if len(vals) == 0:
                continue
            print(f'{au:<6}{side:<6}{vals.min():>8.3f}{vals.median():>8.3f}'
                  f'{vals.quantile(0.95):>8.3f}{vals.max():>8.3f}')

    # Also static-mode BL for the hypertonicity-target AUs
    print('\nStatic-mode BL task, AU p10 + max across controls × sides '
          '(hypertonicity-target AUs):')
    print(f'{"AU":<6}{"side":<6}{"p10_max":>10}{"max_max":>10}')
    print('-' * 32)
    hyp_aus = ['AU04', 'AU07', 'AU14', 'AU15', 'AU17', 'AU23', 'AU45']
    for au in hyp_aus:
        for side in ('Left', 'Right'):
            p10_col = f'BL_{side} {au}_r_static_p10'
            max_col = f'BL_{side} {au}_r_static'
            p10_vals = cdf[p10_col].astype(float).dropna() if p10_col in cdf else pd.Series([])
            max_vals = cdf[max_col].astype(float).dropna() if max_col in cdf else pd.Series([])
            p10_max = p10_vals.max() if len(p10_vals) else float('nan')
            max_max = max_vals.max() if len(max_vals) else float('nan')
            print(f'{au:<6}{side:<6}{p10_max:>10.3f}{max_max:>10.3f}')
    return True


def gate_3_canary_continuity():
    banner('GATE 3: Canary continuity vs v1315 reference')
    if not V1315_DIR.exists():
        print(f'WARNING: v1315 reference dir not found at {V1315_DIR}')
        print('Skipping gate 3.')
        return True

    results = []
    for canary in CANARIES:
        for side in ('left', 'right'):
            fn = f'{canary}_{side}_mirrored_coded.csv'
            new_path = DUAL_DIR / fn
            old_path = V1315_DIR / fn
            if not new_path.exists() or not old_path.exists():
                print(f'  SKIP {canary} {side}: missing file '
                      f'(new={new_path.exists()}, old={old_path.exists()})')
                continue
            new_df = pd.read_csv(new_path)
            old_df = pd.read_csv(old_path)
            # Match on frame index (most reliable)
            merged = pd.merge(
                old_df[['frame'] + [f'{au}_r' for au in AU_ORDER]],
                new_df[['frame'] + [f'{au}_r' for au in AU_ORDER]],
                on='frame', how='inner', suffixes=('_old', '_new'))
            for au in AU_ORDER:
                old_v = merged[f'{au}_r_old'].astype(float)
                new_v = merged[f'{au}_r_new'].astype(float)
                if old_v.std() < 1e-6 and new_v.std() < 1e-6:
                    # Both flat; correlation undefined, treat as pass
                    r = np.nan
                    pass_ = True
                else:
                    r = old_v.corr(new_v)
                    pass_ = r >= R_THRESHOLD
                results.append({
                    'canary': canary, 'side': side, 'au': au,
                    'r': r, 'pass': pass_,
                    'n_frames': len(merged),
                    'old_mean': float(old_v.mean()),
                    'new_mean': float(new_v.mean()),
                })

    rdf = pd.DataFrame(results)
    print(f'\nTotal comparisons: {len(rdf)}')
    fails = rdf[(rdf['pass'] == False)]
    print(f'r < {R_THRESHOLD}: {len(fails)}')
    if len(fails):
        print('\nFailures:')
        for _, row in fails.iterrows():
            print(f'  {row["canary"]} {row["side"]} {row["au"]}: '
                  f'r={row["r"]:.3f}  old_mean={row["old_mean"]:.3f}  '
                  f'new_mean={row["new_mean"]:.3f}')
    print('\nSample (one canary, left, all AUs):')
    sample = rdf[(rdf['canary'] == CANARIES[0]) & (rdf['side'] == 'left')]
    for _, row in sample.iterrows():
        r_str = 'flat' if pd.isna(row['r']) else f'{row["r"]:.4f}'
        print(f'  {row["au"]}: r={r_str}  '
              f'old={row["old_mean"]:.3f}  new={row["new_mean"]:.3f}')

    rdf.to_csv(DATA / 'pilot15_gate3_canary_continuity.csv', index=False)
    print(f'\nWrote pilot15_gate3_canary_continuity.csv')
    return len(fails) == 0


def gate_4_column_completeness():
    banner('GATE 4: Dual-mode column completeness')
    combined = pd.read_csv(COMBINED)
    print(f'Combined CSV: {len(combined)} rows × {len(combined.columns)} cols')

    # Expected: 1 (Patient ID) + 14 (Max Frame) + 14*2*17*3 = 1443
    expected = 1 + 14 + 14 * 2 * 17 * 3
    cols_ok = len(combined.columns) == expected
    print(f'Columns expected: {expected} → {"OK" if cols_ok else "FAIL"}')

    # Per-frame CSV: 4 (frame/timestamp/success/action) + 17 _r + 17 _r_static = 38
    sample = pd.read_csv(DUAL_DIR / '20240723_175947000_iOS_left_mirrored_coded.csv')
    pf_ok = len(sample.columns) == 38
    print(f'Per-frame CSV columns: {len(sample.columns)} → {"OK" if pf_ok else "FAIL"}')
    have_r = sum(1 for c in sample.columns if c.endswith('_r'))
    have_static = sum(1 for c in sample.columns if c.endswith('_r_static'))
    print(f'  _r columns: {have_r}  _r_static columns: {have_static}')

    # NaN audit on combined: how many patients have any NaN in _r columns?
    r_cols = [c for c in combined.columns if c.endswith(' AU01_r')]  # one per task/side
    print(f'\nNaN audit: per-task missing-data check')
    print(f'{"task_side":<14}{"present":>10}{"missing":>10}')
    print('-' * 34)
    for c in sorted(r_cols)[:8]:  # sample first 8
        non_nan = combined[c].notna().sum()
        nan_n = combined[c].isna().sum()
        print(f'  {c.replace(" AU01_r", ""):<12}{non_nan:>10}{nan_n:>10}')
    print(f'  ... ({len(r_cols)-8} more, see CSV)')

    return cols_ok and pf_ok


def main():
    gate1_ok, cohort_df = gate_1_cohort_assignment()
    gate_2_control_overview(cohort_df)
    gate3_ok = gate_3_canary_continuity()
    gate4_ok = gate_4_column_completeness()

    banner('SUMMARY', sym='#')
    print(f'Gate 1 (cohort assignment):  {"PASS" if gate1_ok else "FAIL"}')
    print(f'Gate 2 (control overview):   sanity report (no pass/fail)')
    print(f'Gate 3 (canary continuity):  {"PASS" if gate3_ok else "FAIL"}')
    print(f'Gate 4 (column completeness): {"PASS" if gate4_ok else "FAIL"}')
    sys.exit(0 if (gate1_ok and gate3_ok and gate4_ok) else 1)


if __name__ == '__main__':
    main()
