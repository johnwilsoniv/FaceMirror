#!/usr/bin/env python3
"""
Pilot 15 Phase A — Head-to-head pilots 7/8/9 framework in default vs static modes.

Cohort: directory-based (S Data/Normal Cohort/ = controls, rest = cases).

Outputs:
  - pilot15_phaseA_pilot7_metrics.csv      (BS task, both modes)
  - pilot15_phaseA_pilot8_metrics.csv      (all voluntary tasks, both modes)
  - pilot15_phaseA_pilot9_signatures.csv   (pilot 9 signatures + FP Key concordance, both modes)
  - pilot15_phaseA_findings_long.csv       (long-format finding rows for downstream use)

Comparison is at the (task × mode) granularity. The key Phase A question:
  - Does static mode improve, hurt, or wash on voluntary-task signatures?
"""
from pathlib import Path
import pandas as pd
import numpy as np
from collections import Counter, defaultdict
from sklearn.covariance import LedoitWolf
from scipy.stats import norm

DATA = Path("/Users/johnwilsoniv/Documents/SplitFace Open3/S3 Data Analysis")
COMBINED = DATA / "recoded_rerun_dual_v1316_combined_results.csv"
KEY = DATA / "FPRS FP Key.csv"
CONTROL_DIR = Path("/Users/johnwilsoniv/Documents/SplitFace Open3/"
                   "S Data/Normal Cohort")

OUT_P7 = DATA / "pilot15_phaseA_pilot7_metrics.csv"
OUT_P8 = DATA / "pilot15_phaseA_pilot8_metrics.csv"
OUT_P9 = DATA / "pilot15_phaseA_pilot9_signatures.csv"
OUT_FINDINGS = DATA / "pilot15_phaseA_findings_long.csv"

AU_ORDER = ['AU01', 'AU02', 'AU04', 'AU05', 'AU06', 'AU07', 'AU09',
            'AU10', 'AU12', 'AU14', 'AU15', 'AU17', 'AU20', 'AU23',
            'AU25', 'AU26', 'AU45']

# Pilot 8's 13 voluntary tasks (BL excluded per pilot 8 design)
TASKS_VOLUNTARY = ['BS', 'SS', 'RE', 'FR', 'ES', 'ET', 'BK', 'WN', 'SE', 'SO',
                   'PL', 'BC', 'LT']

# Modes: (column suffix, label)
MODES = [('_r', 'default'), ('_r_static', 'static')]

# Thresholds (match pilot 7)
SD_FLOOR = 0.1
Z_PERMISSIVE = 2.0
Z_STRICT = 3.0
PCT_HIGH = 95
PCT_LOW = 5
PCT_STRICT_HIGH = 99
PCT_STRICT_LOW = 1
FDR_Q = 0.05

# Pilot 9 signatures — same definitions
SIGNATURES = [
    {'id': 'oral_ocular', 'name': 'Oral-Ocular Synkinesis',
     'must_have': [('+AU45', 'BS')],
     'must_not_have': [('+AU45', 'ES'), ('+AU45', 'ET')],
     'fp_key_columns': ['Oral-Ocular Synkinesis Left',
                        'Oral-Ocular Synkinesis Right']},
    {'id': 'ocular_oral', 'name': 'Ocular-Oral Synkinesis',
     'must_have': [('+AU12', 'ET')],
     'must_not_have': [],
     'fp_key_columns': ['Ocular-Oral Synkinesis Left',
                        'Ocular-Oral Synkinesis Right']},
    {'id': 'frontalis_zygomatic', 'name': 'Frontalis-Zygomatic Synkinesis',
     'must_have': [('+AU06', 'RE')],
     'must_not_have': [],
     'fp_key_columns': []},
    {'id': 'mentalis_synkinesis', 'name': 'Mentalis Synkinesis',
     'must_have_any_of': [('+AU17', 'RE'), ('+AU17', 'ET'), ('+AU17', 'ES')],
     'fp_key_columns': ['Mentalis Synkinesis Left',
                        'Mentalis Synkinesis Right']},
    {'id': 'snarl_pattern', 'name': 'Snarl Smile Pattern',
     'must_have_any_of': [('+AU09', 'BS'), ('+AU09', 'SS')],
     'fp_key_columns': ['Snarl Smile Left', 'Snarl Smile Right']},
    {'id': 'paretic_smile', 'name': 'Paretic Smile',
     'must_have_any_of': [('-AU12', 'BS'), ('-AU12', 'SS')],
     'fp_key_columns': []},
    {'id': 'brow_paresis', 'name': 'Brow Paresis',
     'must_have_any_of': [('-AU01', 'RE'), ('-AU02', 'RE')],
     'fp_key_columns': ['Paralysis - Left Upper Face',
                        'Paralysis - Right Upper Face']},
]


def load_cohort(combined):
    control_ids = sorted(p.stem for p in CONTROL_DIR.glob('IMG_*.MOV'))
    all_ids = combined['Patient ID'].astype(str).tolist()
    case_ids = [pid for pid in all_ids if pid not in set(control_ids)]
    return control_ids, case_ids


def build_X(combined, patient_ids, task, suffix):
    df = combined[combined['Patient ID'].astype(str).isin(patient_ids)]
    rows, hemis = [], []
    for _, row in df.iterrows():
        pid = str(row['Patient ID'])
        for side in ('Left', 'Right'):
            au_cols = [f'{task}_{side} {au}{suffix}' for au in AU_ORDER]
            missing = [c for c in au_cols if c not in df.columns]
            if missing:
                return None, None
            v = row[au_cols].values.astype(float)
            if np.isnan(v).all():
                continue
            v = np.clip(np.nan_to_num(v, nan=0.0), 0.0, None)
            rows.append(v)
            hemis.append(f'{pid}|{side}')
    if not rows:
        return None, None
    return np.array(rows), hemis


def fit_control_distribution(X_ctrl):
    dist = {}
    for j, au in enumerate(AU_ORDER):
        vals = X_ctrl[:, j]
        sd = float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0
        dist[au] = {
            'mean': float(np.mean(vals)),
            'sd': sd,
            'sd_floored': max(sd, SD_FLOOR),
            'p5': float(np.percentile(vals, PCT_LOW)),
            'p95': float(np.percentile(vals, PCT_HIGH)),
            'p1': float(np.percentile(vals, PCT_STRICT_LOW)),
            'p99': float(np.percentile(vals, PCT_STRICT_HIGH)),
        }
    return dist


def z_scores(X, dist):
    Z = np.zeros_like(X)
    for j, au in enumerate(AU_ORDER):
        Z[:, j] = (X[:, j] - dist[au]['mean']) / dist[au]['sd_floored']
    return Z


def derive_findings(z_row, x_row, dist):
    findings = []
    for j, au in enumerate(AU_ORDER):
        z = z_row[j]
        v = x_row[j]
        d = dist[au]
        if z > 0:
            direction = 'elevation'
            perm_z = z >= Z_PERMISSIVE
            perm_pct = v > d['p95']
            strict_z = z >= Z_STRICT
            strict_pct = v > d['p99']
            sign = '+'
        else:
            direction = 'paresis'
            perm_z = z <= -Z_PERMISSIVE
            perm_pct = v < d['p5']
            strict_z = z <= -Z_STRICT
            strict_pct = v < d['p1']
            sign = '-'
        permissive = bool(perm_z or perm_pct)
        strict = bool(strict_z and strict_pct)
        if permissive:
            findings.append({
                'au': au, 'direction': direction,
                'finding_id': f'{sign}{au}',
                'value': float(v), 'z': float(z),
                'permissive': True, 'strict': bool(strict),
            })
    return findings


def mahalanobis(X, X_ctrl):
    lw = LedoitWolf().fit(X_ctrl)
    diffs = X - lw.location_
    cov_inv = np.linalg.pinv(lw.covariance_)
    return np.sqrt(np.einsum('ij,jk,ik->i', diffs, cov_inv, diffs))


def parse_hemi(h):
    pid, side = h.split('|')
    return pid, side


def fp_positive(key_df, pid, sig):
    cols = sig.get('fp_key_columns', [])
    if not cols:
        return None
    row = key_df[key_df['Patient'].astype(str) == pid]
    if len(row) == 0:
        return False
    row = row.iloc[0]
    any_assessed = False
    for col in cols:
        if col not in key_df.columns:
            continue
        v = row[col]
        if pd.notna(v):
            any_assessed = True
            s = str(v).strip()
            if s in ('Yes', 'Complete', 'Partial'):
                return True
    return False if any_assessed else None


def evaluate_signature(sig, finding_tasks_by_pid):
    """For each patient, eval whether the signature fires.
    finding_tasks_by_pid: {pid: {finding_id: set_of_tasks}}
    """
    results = {}
    for pid, finding_tasks in finding_tasks_by_pid.items():
        if 'must_have' in sig:
            ok = all(t in finding_tasks.get(fid, set())
                     for fid, t in sig['must_have'])
            if 'must_not_have' in sig:
                ok = ok and all(t not in finding_tasks.get(fid, set())
                                for fid, t in sig['must_not_have'])
            results[pid] = bool(ok)
        elif 'must_have_any_of' in sig:
            results[pid] = any(t in finding_tasks.get(fid, set())
                               for fid, t in sig['must_have_any_of'])
    return results


# ---------- Main ----------

def run_one_mode(combined, key, controls, cases, suffix, mode_label, findings_long):
    """For one mode, run pilots 7/8/9 framework. Returns dict of metrics."""
    metrics = {
        'mode': mode_label,
        'per_task': {},   # task -> dict of metrics
        'signatures': {}, # sig_id -> dict
        'finding_tasks_by_pid': defaultdict(lambda: defaultdict(set)),
    }

    for task in TASKS_VOLUNTARY:
        X_ctrl, hemi_ctrl = build_X(combined, controls, task, suffix)
        X_case, hemi_case = build_X(combined, cases, task, suffix)
        if X_ctrl is None or X_case is None:
            continue

        dist = fit_control_distribution(X_ctrl)
        Z_ctrl = z_scores(X_ctrl, dist)
        Z_case = z_scores(X_case, dist)

        findings_ctrl = [derive_findings(Z_ctrl[i], X_ctrl[i], dist)
                         for i in range(len(X_ctrl))]
        findings_case = [derive_findings(Z_case[i], X_case[i], dist)
                         for i in range(len(X_case))]

        # Accumulate findings into per-patient cross-task index
        for hemi, flist in zip(hemi_case, findings_case):
            pid, side = parse_hemi(hemi)
            for f in flist:
                metrics['finding_tasks_by_pid'][pid][f['finding_id']].add(task)
                findings_long.append({
                    'mode': mode_label, 'task': task,
                    'patient_id': pid, 'side': side, 'cohort': 'case',
                    'finding_id': f['finding_id'], 'au': f['au'],
                    'direction': f['direction'], 'value': f['value'],
                    'z': f['z'], 'strict': f['strict'],
                })
        for hemi, flist in zip(hemi_ctrl, findings_ctrl):
            pid, side = parse_hemi(hemi)
            for f in flist:
                findings_long.append({
                    'mode': mode_label, 'task': task,
                    'patient_id': pid, 'side': side, 'cohort': 'control',
                    'finding_id': f['finding_id'], 'au': f['au'],
                    'direction': f['direction'], 'value': f['value'],
                    'z': f['z'], 'strict': f['strict'],
                })

        # Mahalanobis
        md_ctrl = mahalanobis(X_ctrl, X_ctrl)
        md_case = mahalanobis(X_case, X_ctrl)

        # Per-task aggregates
        n_ctrl_findings = sum(len(f) for f in findings_ctrl)
        n_case_findings = sum(len(f) for f in findings_case)
        avg_ctrl = n_ctrl_findings / max(len(X_ctrl), 1)
        avg_case = n_case_findings / max(len(X_case), 1)
        # Discrimination ratio
        disc = avg_case / max(avg_ctrl, 0.01)

        metrics['per_task'][task] = {
            'n_ctrl_hemi': len(X_ctrl), 'n_case_hemi': len(X_case),
            'n_ctrl_findings': n_ctrl_findings,
            'n_case_findings': n_case_findings,
            'avg_ctrl_findings_per_hemi': avg_ctrl,
            'avg_case_findings_per_hemi': avg_case,
            'discrimination_ratio': disc,
            'md_ctrl_mean': float(md_ctrl.mean()),
            'md_case_mean': float(md_case.mean()),
            'md_ctrl_p95': float(np.percentile(md_ctrl, 95)),
            'md_case_above_ctrl_p95_pct': float(
                (md_case > np.percentile(md_ctrl, 95)).mean() * 100),
        }

    # Pilot 9 signatures
    for sig in SIGNATURES:
        preds = evaluate_signature(sig, metrics['finding_tasks_by_pid'])
        # Validate vs FP Key on CASE patients only
        tp = fp = fn = tn = 0
        n_assessed = 0
        for pid in cases:
            label = fp_positive(key, pid, sig)
            if label is None:
                continue
            n_assessed += 1
            pred = preds.get(pid, False)
            if label and pred: tp += 1
            elif label and not pred: fn += 1
            elif not label and pred: fp += 1
            else: tn += 1
        sens = tp / (tp + fn) if (tp + fn) > 0 else None
        spec = tn / (tn + fp) if (tn + fp) > 0 else None
        case_count = sum(1 for pid in cases if preds.get(pid, False))
        ctrl_count = sum(1 for pid in controls if preds.get(pid, False))
        metrics['signatures'][sig['id']] = {
            'name': sig['name'],
            'sensitivity': sens, 'specificity': spec,
            'tp': tp, 'fp': fp, 'fn': fn, 'tn': tn,
            'n_assessed': n_assessed,
            'case_count_pred_pos': case_count,
            'ctrl_count_pred_pos': ctrl_count,
        }
    return metrics


def main():
    print('Loading combined CSV...')
    combined = pd.read_csv(COMBINED)
    key = pd.read_csv(KEY, encoding='utf-8-sig')
    controls, cases = load_cohort(combined)
    print(f'Controls: {len(controls)}  Cases: {len(cases)}')

    findings_long = []
    metrics = {}
    for suffix, label in MODES:
        print(f'\nRunning mode: {label} (suffix={suffix})')
        metrics[label] = run_one_mode(combined, key, controls, cases,
                                       suffix, label, findings_long)

    # ---- Output: per-task metrics (pilot 8) ----
    rows = []
    for label in ('default', 'static'):
        for task, m in metrics[label]['per_task'].items():
            rows.append({'task': task, 'mode': label, **m})
    pilot8 = pd.DataFrame(rows)
    pilot8.to_csv(OUT_P8, index=False)
    print(f'\nWrote {OUT_P8.name}: {len(pilot8)} rows')

    # ---- Output: pilot 7 (BS only) summary table ----
    p7 = pilot8[pilot8['task'] == 'BS'].copy()
    p7.to_csv(OUT_P7, index=False)
    print(f'Wrote {OUT_P7.name}: {len(p7)} rows (BS only)')

    # ---- Output: signatures (pilot 9) ----
    rows = []
    for label in ('default', 'static'):
        for sid, m in metrics[label]['signatures'].items():
            rows.append({'signature_id': sid, 'mode': label, **m})
    pilot9 = pd.DataFrame(rows)
    pilot9.to_csv(OUT_P9, index=False)
    print(f'Wrote {OUT_P9.name}: {len(pilot9)} rows')

    # ---- Output: findings long ----
    fdf = pd.DataFrame(findings_long)
    fdf.to_csv(OUT_FINDINGS, index=False)
    print(f'Wrote {OUT_FINDINGS.name}: {len(fdf)} rows')

    # ---- Print headline comparisons ----
    print('\n' + '=' * 72)
    print('HEADLINE: per-task discrimination ratio (case/control findings)')
    print('=' * 72)
    pivot = pilot8.pivot(index='task', columns='mode',
                         values='discrimination_ratio')
    pivot['ratio_static_vs_default'] = pivot['static'] / pivot['default']
    pivot = pivot.reindex(TASKS_VOLUNTARY)
    print(pivot.to_string(float_format=lambda x: f'{x:.2f}'))

    print('\n' + '=' * 72)
    print('HEADLINE: per-task control finding rate (lower = cleaner reference)')
    print('=' * 72)
    pivot_ctrl = pilot8.pivot(index='task', columns='mode',
                              values='avg_ctrl_findings_per_hemi')
    pivot_ctrl = pivot_ctrl.reindex(TASKS_VOLUNTARY)
    print(pivot_ctrl.to_string(float_format=lambda x: f'{x:.2f}'))

    print('\n' + '=' * 72)
    print('HEADLINE: per-task Mahalanobis case mean (higher = better separation)')
    print('=' * 72)
    pivot_md = pilot8.pivot(index='task', columns='mode',
                            values='md_case_mean')
    pivot_md = pivot_md.reindex(TASKS_VOLUNTARY)
    print(pivot_md.to_string(float_format=lambda x: f'{x:.2f}'))

    print('\n' + '=' * 72)
    print('HEADLINE: pilot 9 signatures sens/spec by mode')
    print('=' * 72)
    print(f'{"signature":<30}{"mode":<10}{"sens":>8}{"spec":>8}'
          f'{"tp":>5}{"fp":>5}{"fn":>5}{"tn":>5}{"n":>5}')
    print('-' * 84)
    for sid in [s['id'] for s in SIGNATURES]:
        for label in ('default', 'static'):
            m = metrics[label]['signatures'][sid]
            sens = f'{m["sensitivity"]:.2f}' if m['sensitivity'] is not None else 'na'
            spec = f'{m["specificity"]:.2f}' if m['specificity'] is not None else 'na'
            print(f'{sid[:29]:<30}{label:<10}{sens:>8}{spec:>8}'
                  f'{m["tp"]:>5}{m["fp"]:>5}{m["fn"]:>5}{m["tn"]:>5}'
                  f'{m["n_assessed"]:>5}')


if __name__ == '__main__':
    main()
