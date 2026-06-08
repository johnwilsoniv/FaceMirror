#!/usr/bin/env python3
"""
Pilot 15 — Tone vs Anatomy Discriminator.

Question: when a patient shows elevated static_p10 at BL on some AU, is it
(a) imaging artifact, (b) anatomical face shape, or (c) genuine elevated tone?

Five signals combined into a heuristic classifier:

  S1 — bilateral_asym        |L-R| / max(L,R) of static_p10 at BL
                              high asym → tonus suspected (often unilateral)
  S2 — frame_quality         success_rate of CLNF in BL frames (per side)
                              low quality → imaging artifact suspected
  S3 — cluster_coherence     # of OTHER AUs in same anatomic cluster
                              that are also elevated
                              high → anatomy (face shape affects clusters);
                              low (single AU) → focal muscle (tonus suspected)
  S4 — cross_task_persist    fraction of voluntary tasks (BS/SS/RE/ES/ET/SE/SO)
                              where AU is also above control p95 in static
                              high → chronic (anatomy or tonus, NOT voluntary)
  S5 — range_compression     (static_max - static_p10) within voluntary task
                              vs control distribution of same range
                              compressed → tonus (less voluntary headroom)

Heuristic classifier:
  if S2 < 0.7:                                       IMAGING_ARTIFACT
  elif S3 >= 2:                                      ANATOMY (cluster pattern)
  elif S1 > 0.5 and S4 > 0.7 and S5_compressed:      HYPERTONUS
  elif S1 < 0.2 and S4 > 0.7:                        ANATOMY (symmetric chronic)
  else:                                              AMBIGUOUS

Validation:
  - Controls (13): no patient × AU should classify HYPERTONUS
  - FP-flagged hypertonic cases: should classify HYPERTONUS on the relevant AUs

Outputs:
  - pilot15_discriminator_per_finding.csv   (long-format, per (patient, AU))
  - pilot15_discriminator_validation.csv    (validation against controls + FP)
"""
from pathlib import Path
import pandas as pd
import numpy as np

DATA = Path("/Users/johnwilsoniv/Documents/SplitFace Open3/S3 Data Analysis")
DUAL_DIR = DATA / "recoded_rerun_dual_v1316"
COMBINED = DATA / "recoded_rerun_dual_v1316_combined_results.csv"
KEY = DATA / "FPRS FP Key.csv"
CTRL_DIR = Path("/Users/johnwilsoniv/Documents/SplitFace Open3/"
                "S Data/Normal Cohort")

OUT_FINDINGS = DATA / "pilot15_discriminator_per_finding.csv"
OUT_VALIDATION = DATA / "pilot15_discriminator_validation.csv"

AU_ORDER = ['AU01', 'AU02', 'AU04', 'AU05', 'AU06', 'AU07', 'AU09',
            'AU10', 'AU12', 'AU14', 'AU15', 'AU17', 'AU20', 'AU23',
            'AU25', 'AU26', 'AU45']
HYP_AUS = ['AU04', 'AU07', 'AU14', 'AU15', 'AU17', 'AU23', 'AU45']
WORKING_TASKS = ['BS', 'SS', 'RE', 'ES', 'ET', 'SE', 'SO']

# Anatomic clusters (face shape affects these together)
CLUSTERS = {
    'lower_face_morphology': {'AU14', 'AU17', 'AU23', 'AU25'},
    'brow_morphology':       {'AU01', 'AU02', 'AU04', 'AU05'},
    'cheek_eyelid':          {'AU06', 'AU07'},
    'mouth_wide':            {'AU10', 'AU12', 'AU15', 'AU20'},
    'nose':                  {'AU09'},
    'blink':                 {'AU45'},
}

# Discriminator thresholds
ELEV_P10_THR   = 0.5    # candidate finding threshold
ASYM_HIGH      = 0.5    # tonus-suspect asymmetry
ASYM_LOW       = 0.2    # anatomy-suspect symmetry
QUAL_LOW       = 0.7    # below this → imaging artifact
PERSIST_HIGH   = 0.7    # fraction of voluntary tasks elevated → chronic
CLUSTER_HIGH   = 2      # ≥2 other AUs in same cluster elevated → anatomy
RANGE_COMPR_PCT = 5     # below ctrl p5 of range → compressed


def get_au_cluster(au):
    for name, members in CLUSTERS.items():
        if au in members:
            return name
    return None


def load_per_side_success(controls_and_cases):
    """Compute success_rate per (patient, side, task) by reading per-frame CSVs."""
    succ = {}
    for csv in DUAL_DIR.glob('*_mirrored_coded.csv'):
        # filename: {pid}_{side}_mirrored_coded.csv
        stem = csv.stem.replace('_mirrored_coded', '')
        parts = stem.rsplit('_', 1)
        if len(parts) != 2:
            continue
        pid, side = parts
        side = side.capitalize()
        df = pd.read_csv(csv, usecols=['action', 'success'])
        df['action'] = df['action'].astype(str).str.strip()
        for task, tdf in df.groupby('action'):
            if task in ('', 'nan'):
                continue
            succ[(pid, side, task)] = float(tdf['success'].mean())
    return succ


def load_control_range_distributions(combined, controls):
    """For each (task, AU): compute control distribution of (static_max - static_p10)
    range. Used as reference for S5 range_compression."""
    cdf = combined[combined['Patient ID'].astype(str).isin(controls)]
    out = {}  # (task, side, au) -> {p5, p10, mean, p90}
    for task in WORKING_TASKS:
        for side in ('Left', 'Right'):
            for au in AU_ORDER:
                p10c = f'{task}_{side} {au}_r_static_p10'
                maxc = f'{task}_{side} {au}_r_static'
                if p10c not in cdf.columns or maxc not in cdf.columns:
                    continue
                p10v = cdf[p10c].astype(float)
                maxv = cdf[maxc].astype(float)
                rng = (maxv - p10v).dropna()
                if len(rng) < 3:
                    continue
                out[(task, side, au)] = {
                    'p5':  float(rng.quantile(0.05)),
                    'p10': float(rng.quantile(0.1)),
                    'mean': float(rng.mean()),
                    'p90': float(rng.quantile(0.9)),
                }
    return out


def load_control_static_p95_per_task(combined, controls):
    """Per-(task, side, AU): control static p95 (used as S4 elevation threshold
    in voluntary tasks)."""
    cdf = combined[combined['Patient ID'].astype(str).isin(controls)]
    out = {}
    for task in WORKING_TASKS + ['BL']:
        for side in ('Left', 'Right'):
            for au in AU_ORDER:
                col = f'{task}_{side} {au}_r_static'
                if col not in cdf.columns:
                    continue
                vals = cdf[col].astype(float).dropna()
                if len(vals) < 3:
                    continue
                out[(task, side, au)] = float(vals.quantile(0.95))
    return out


def compute_features(row, side, succ_map, ctrl_p95, ctrl_range):
    """For one (patient, side) at BL, compute features for each elevated AU."""
    pid = str(row['Patient ID'])
    findings = []
    for au in AU_ORDER:
        # Trigger: BL static_p10 on either side ≥ threshold
        l_col = f'BL_Left {au}_r_static_p10'
        r_col = f'BL_Right {au}_r_static_p10'
        l_p10 = float(row.get(l_col, np.nan))
        r_p10 = float(row.get(r_col, np.nan))
        max_p10 = max(l_p10 if not np.isnan(l_p10) else 0,
                      r_p10 if not np.isnan(r_p10) else 0)
        if max_p10 < ELEV_P10_THR:
            continue

        # S1 — bilateral asymmetry
        if not (np.isnan(l_p10) or np.isnan(r_p10)):
            asym = abs(l_p10 - r_p10) / max(max_p10, 0.01)
        else:
            asym = float('nan')

        # S2 — frame quality (worst of two sides)
        l_succ = succ_map.get((pid, 'Left', 'BL'), 1.0)
        r_succ = succ_map.get((pid, 'Right', 'BL'), 1.0)
        quality = min(l_succ, r_succ)

        # S3 — cluster coherence
        cluster = get_au_cluster(au)
        coherence = 0
        if cluster:
            for other_au in CLUSTERS[cluster]:
                if other_au == au:
                    continue
                ol = float(row.get(f'BL_Left {other_au}_r_static_p10', np.nan))
                orr = float(row.get(f'BL_Right {other_au}_r_static_p10', np.nan))
                other_max = max(ol if not np.isnan(ol) else 0,
                                orr if not np.isnan(orr) else 0)
                if other_max >= ELEV_P10_THR:
                    coherence += 1

        # S4 — cross-task persistence
        # For each voluntary task: is patient's static value above control p95?
        n_persist = 0
        n_total = 0
        for task in WORKING_TASKS:
            for side_check in ('Left', 'Right'):
                ctrl_thr = ctrl_p95.get((task, side_check, au))
                if ctrl_thr is None:
                    continue
                pat_val = float(row.get(f'{task}_{side_check} {au}_r_static',
                                        np.nan))
                if np.isnan(pat_val):
                    continue
                n_total += 1
                if pat_val > ctrl_thr:
                    n_persist += 1
        persist = n_persist / n_total if n_total else float('nan')

        # S5 — range compression in BS task (most voluntary smile)
        range_compressed = False
        for task in ('BS', 'SS'):  # try smile tasks first
            for side_check in ('Left', 'Right'):
                ref = ctrl_range.get((task, side_check, au))
                if ref is None:
                    continue
                pat_p10 = float(row.get(f'{task}_{side_check} {au}_r_static_p10',
                                        np.nan))
                pat_max = float(row.get(f'{task}_{side_check} {au}_r_static',
                                        np.nan))
                if np.isnan(pat_p10) or np.isnan(pat_max):
                    continue
                pat_range = pat_max - pat_p10
                if pat_range < ref['p5']:
                    range_compressed = True
                    break
            if range_compressed:
                break

        # Classify
        if quality < QUAL_LOW:
            cls = 'IMAGING_ARTIFACT'
        elif coherence >= CLUSTER_HIGH:
            cls = 'ANATOMY'
        elif (not np.isnan(asym) and asym > ASYM_HIGH
              and not np.isnan(persist) and persist > PERSIST_HIGH
              and range_compressed):
            cls = 'HYPERTONUS'
        elif (not np.isnan(asym) and asym < ASYM_LOW
              and not np.isnan(persist) and persist > PERSIST_HIGH):
            cls = 'ANATOMY'
        else:
            cls = 'AMBIGUOUS'

        findings.append({
            'patient_id': pid, 'au': au,
            'left_p10': l_p10, 'right_p10': r_p10, 'max_p10': max_p10,
            'S1_bilateral_asym': asym,
            'S2_frame_quality': quality,
            'S3_cluster_coherence': coherence,
            'S3_cluster_name': cluster,
            'S4_cross_task_persist': persist,
            'S5_range_compressed': range_compressed,
            'classification': cls,
        })
    return findings


def fp_hyp_label(key, pid):
    """Returns 'left', 'right', 'both', or None for FP-Key Hypertonicity flag."""
    row = key[key['Patient'].astype(str) == pid]
    if len(row) == 0:
        return None
    row = row.iloc[0]
    l = str(row.get('Hypertonicity Left', '')).strip()
    r = str(row.get('Hypertonicity Right', '')).strip()
    has_l = l in ('Yes', 'Complete', 'Partial')
    has_r = r in ('Yes', 'Complete', 'Partial')
    if has_l and has_r: return 'both'
    if has_l: return 'left'
    if has_r: return 'right'
    return None


def main():
    print('Loading data...')
    combined = pd.read_csv(COMBINED)
    key = pd.read_csv(KEY, encoding='utf-8-sig')
    controls = sorted(p.stem for p in CTRL_DIR.glob('IMG_*.MOV'))
    cases = [pid for pid in combined['Patient ID'].astype(str).tolist()
             if pid not in set(controls)]
    print(f'Controls: {len(controls)}, Cases: {len(cases)}')

    print('Loading per-side success rates from per-frame CSVs...')
    succ_map = load_per_side_success(combined['Patient ID'].astype(str).tolist())
    print(f'  {len(succ_map)} (patient, side, task) entries')

    print('Building control reference distributions...')
    ctrl_p95 = load_control_static_p95_per_task(combined, controls)
    ctrl_range = load_control_range_distributions(combined, controls)
    print(f'  ctrl_p95: {len(ctrl_p95)} entries')
    print(f'  ctrl_range: {len(ctrl_range)} entries')

    print('\nComputing features and classifying findings...')
    all_findings = []
    for _, row in combined.iterrows():
        pid = str(row['Patient ID'])
        cohort = 'control' if pid in set(controls) else 'case'
        findings = compute_features(row, None, succ_map, ctrl_p95, ctrl_range)
        for f in findings:
            f['cohort'] = cohort
            f['fp_hyp_side'] = fp_hyp_label(key, pid) if cohort == 'case' else None
            all_findings.append(f)

    fdf = pd.DataFrame(all_findings)
    fdf.to_csv(OUT_FINDINGS, index=False)
    print(f'Wrote {OUT_FINDINGS.name}: {len(fdf)} findings')

    # ---- Validation: classification distribution by cohort ----
    print('\n' + '=' * 72)
    print('VALIDATION 1: classification distribution by cohort')
    print('=' * 72)
    pivot = fdf.groupby(['cohort', 'classification']).size().unstack(fill_value=0)
    print(pivot.to_string())

    # ---- Validation: controls should never be HYPERTONUS ----
    print('\n' + '=' * 72)
    print('VALIDATION 2: any control classified as HYPERTONUS?')
    print('=' * 72)
    ctrl_hyp = fdf[(fdf['cohort'] == 'control') &
                   (fdf['classification'] == 'HYPERTONUS')]
    if len(ctrl_hyp) == 0:
        print('PASS: 0 controls classified as HYPERTONUS')
    else:
        print(f'FAIL: {len(ctrl_hyp)} controls classified as HYPERTONUS:')
        print(ctrl_hyp[['patient_id', 'au', 'left_p10', 'right_p10',
                        'S1_bilateral_asym', 'S2_frame_quality',
                        'S3_cluster_coherence', 'S4_cross_task_persist',
                        'S5_range_compressed']].to_string(index=False))

    # ---- Validation: FP-flagged hypertonic cases ----
    print('\n' + '=' * 72)
    print('VALIDATION 3: FP-Key Hypertonicity-flagged cases')
    print('=' * 72)
    fp_pos = [pid for pid in cases if fp_hyp_label(key, pid) is not None]
    print(f'FP-flagged hypertonic cases: {len(fp_pos)}')
    sub = fdf[fdf['patient_id'].isin(fp_pos)]
    print(f'\nFindings on FP-flagged hypertonic cases (any AU): {len(sub)}')
    print('Classification distribution:')
    print(sub['classification'].value_counts().to_string())

    # AU14 (buccinator = "Hypertonicity" target per FP Key) on these cases
    print('\nAU14 (buccinator) findings on FP-flagged hypertonic cases:')
    sub14 = sub[sub['au'] == 'AU14']
    if len(sub14) == 0:
        print('  (no AU14 elevation found on these cases above threshold)')
    else:
        for _, r in sub14.iterrows():
            print(f'  {r["patient_id"]:<28} L={r["left_p10"]:.2f} '
                  f'R={r["right_p10"]:.2f} '
                  f'asym={r["S1_bilateral_asym"]:.2f} '
                  f'qual={r["S2_frame_quality"]:.2f} '
                  f'coh={r["S3_cluster_coherence"]} '
                  f'persist={r["S4_cross_task_persist"]:.2f} '
                  f'compr={r["S5_range_compressed"]} '
                  f'→ {r["classification"]}')

    # AU17 on mentalis-synkinesis-flagged cases (a related hypertonus pattern)
    print('\nAU17 (mentalis) findings on Mentalis Synkinesis-flagged cases:')
    ms_pos = []
    for pid in cases:
        krow = key[key['Patient'].astype(str) == pid]
        if len(krow) == 0:
            continue
        krow = krow.iloc[0]
        l = str(krow.get('Mentalis Synkinesis Left', '')).strip()
        r = str(krow.get('Mentalis Synkinesis Right', '')).strip()
        if l == 'Yes' or r == 'Yes':
            ms_pos.append(pid)
    print(f'(Mentalis Synkinesis-flagged cases: {len(ms_pos)})')
    sub17 = fdf[(fdf['patient_id'].isin(ms_pos)) & (fdf['au'] == 'AU17')]
    if len(sub17) == 0:
        print('  (no AU17 elevation above threshold on MS cases)')
    else:
        cls_counts = sub17['classification'].value_counts()
        print(f'  Classifications: {dict(cls_counts)}')

    # ---- Per-control breakdown ----
    print('\n' + '=' * 72)
    print('VALIDATION 4: per-control classification detail')
    print('=' * 72)
    ctrl_findings = fdf[fdf['cohort'] == 'control']
    if len(ctrl_findings) == 0:
        print('(no control findings above threshold)')
    else:
        print(f'{"patient":<14}{"AU":<6}{"L p10":>7}{"R p10":>7}{"asym":>7}'
              f'{"qual":>7}{"coh":>5}{"per":>6}{"cmp":>5}  cls')
        print('-' * 80)
        for _, r in ctrl_findings.sort_values(['patient_id', 'au']).iterrows():
            asym_s = f'{r["S1_bilateral_asym"]:.2f}' if not pd.isna(r["S1_bilateral_asym"]) else '  na'
            persist_s = f'{r["S4_cross_task_persist"]:.2f}' if not pd.isna(r["S4_cross_task_persist"]) else '  na'
            print(f'{r["patient_id"]:<14}{r["au"]:<6}{r["left_p10"]:>7.2f}'
                  f'{r["right_p10"]:>7.2f}{asym_s:>7}{r["S2_frame_quality"]:>7.2f}'
                  f'{r["S3_cluster_coherence"]:>5}{persist_s:>6}'
                  f'{"yes" if r["S5_range_compressed"] else " no":>5}  '
                  f'{r["classification"]}')

    # ---- Save validation summary ----
    val_rows = []
    val_rows.append({'metric': 'controls_findings', 'value': len(ctrl_findings)})
    val_rows.append({'metric': 'controls_classified_HYPERTONUS', 'value': len(ctrl_hyp)})
    val_rows.append({'metric': 'fp_flagged_hyp_cases', 'value': len(fp_pos)})
    val_rows.append({'metric': 'fp_hyp_findings_total', 'value': len(sub)})
    val_rows.append({'metric': 'fp_hyp_findings_classified_HYPERTONUS',
                     'value': int((sub['classification'] == 'HYPERTONUS').sum())})
    val_rows.append({'metric': 'fp_hyp_AU14_findings', 'value': len(sub14)})
    pd.DataFrame(val_rows).to_csv(OUT_VALIDATION, index=False)
    print(f'\nWrote {OUT_VALIDATION.name}')


if __name__ == '__main__':
    main()
