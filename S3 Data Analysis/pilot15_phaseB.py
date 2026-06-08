#!/usr/bin/env python3
"""
Pilot 15 Phase B — Face-shape leak characterization in controls.

Question: when we strip the running median, do controls show non-zero AU
baselines that are anatomy rather than behavior? If so:
  - Which controls?
  - Which AUs?
  - Bilateral or asymmetric?
  - Is it 1-2 outlier faces or systematic across the cohort?

Outputs:
  - pilot15_phaseB_per_control_anatomy.csv  (per control × side: static BL p10 + max for all 17 AUs)
  - pilot15_phaseB_per_au_distribution.csv  (per-AU: default vs static distribution stats)
  - pilot15_phaseB_asymmetry_audit.csv      (per-control |L-R| values per AU, both modes)
  - pilot15_phaseB_default_vs_static_BL.csv (per-AU control BL: default mean vs static mean)
"""
from pathlib import Path
import pandas as pd
import numpy as np

DATA = Path("/Users/johnwilsoniv/Documents/SplitFace Open3/S3 Data Analysis")
COMBINED = DATA / "recoded_rerun_dual_v1316_combined_results.csv"
CONTROL_DIR = Path("/Users/johnwilsoniv/Documents/SplitFace Open3/"
                   "S Data/Normal Cohort")

AU_ORDER = ['AU01', 'AU02', 'AU04', 'AU05', 'AU06', 'AU07', 'AU09',
            'AU10', 'AU12', 'AU14', 'AU15', 'AU17', 'AU20', 'AU23',
            'AU25', 'AU26', 'AU45']

# Hypertonicity-target AUs from pilot 10 (the AUs we most care about for static-mode tonus)
HYP_AUS = ['AU04', 'AU07', 'AU14', 'AU15', 'AU17', 'AU23', 'AU45']

OUT_ANATOMY = DATA / "pilot15_phaseB_per_control_anatomy.csv"
OUT_DIST = DATA / "pilot15_phaseB_per_au_distribution.csv"
OUT_ASYM = DATA / "pilot15_phaseB_asymmetry_audit.csv"
OUT_DEF_VS_STATIC = DATA / "pilot15_phaseB_default_vs_static_BL.csv"


def banner(text, sym='='):
    print('\n' + sym * 72)
    print(f'  {text}')
    print(sym * 72)


def main():
    combined = pd.read_csv(COMBINED)
    controls = sorted(p.stem for p in CONTROL_DIR.glob('IMG_*.MOV'))
    cdf = combined[combined['Patient ID'].astype(str).isin(controls)].copy()
    print(f'Controls: {len(cdf)}')

    # ---------- 3.1: Per-control anatomy fingerprint ----------
    banner("Phase B.1 — Per-control anatomy fingerprint (static BL, all 17 AUs)")
    rows = []
    for _, row in cdf.iterrows():
        pid = str(row['Patient ID'])
        for side in ('Left', 'Right'):
            r = {'patient_id': pid, 'side': side}
            for au in AU_ORDER:
                p10_col = f'BL_{side} {au}_r_static_p10'
                max_col = f'BL_{side} {au}_r_static'
                def_col = f'BL_{side} {au}_r'
                r[f'{au}_static_p10'] = float(row.get(p10_col, np.nan))
                r[f'{au}_static_max'] = float(row.get(max_col, np.nan))
                r[f'{au}_default_max'] = float(row.get(def_col, np.nan))
            rows.append(r)
    anatomy = pd.DataFrame(rows)
    anatomy.to_csv(OUT_ANATOMY, index=False)
    print(f'Wrote {OUT_ANATOMY.name}: {len(anatomy)} rows')

    # ---------- 3.2: Identify outlier controls per AU ----------
    banner("Phase B.2 — Outlier controls per AU (static BL p10 ≥ 0.5)")
    print(f'{"AU":<6}{"side":<6}{"n_outliers":>11}{"max_p10":>10}'
          f'   outlier_patients')
    print('-' * 72)
    outlier_records = []
    for au in HYP_AUS:
        for side in ('Left', 'Right'):
            col = f'{au}_static_p10'
            sub = anatomy[anatomy['side'] == side]
            outliers = sub[sub[col] >= 0.5][['patient_id', col]].sort_values(
                col, ascending=False)
            max_p10 = sub[col].max()
            ids = ', '.join(f'{r.patient_id}({r[col]:.2f})'
                            for _, r in outliers.head(5).iterrows())
            print(f'{au:<6}{side:<6}{len(outliers):>11}{max_p10:>10.2f}   {ids}')
            for _, r in outliers.iterrows():
                outlier_records.append({
                    'au': au, 'side': side, 'patient_id': r['patient_id'],
                    'static_p10': r[col],
                })

    # ---------- 3.3: Per-AU distribution stats ----------
    banner("Phase B.3 — Per-AU control distribution (default BL vs static BL)")
    rows = []
    for au in AU_ORDER:
        for side in ('Left', 'Right'):
            def_col = f'BL_{side} {au}_r'
            stat_col = f'BL_{side} {au}_r_static'
            stat_p10_col = f'BL_{side} {au}_r_static_p10'
            d = cdf[def_col].astype(float).dropna()
            s = cdf[stat_col].astype(float).dropna()
            sp10 = cdf[stat_p10_col].astype(float).dropna()
            rows.append({
                'au': au, 'side': side,
                'default_mean': float(d.mean()), 'default_sd': float(d.std()),
                'default_p95': float(d.quantile(0.95)),
                'static_mean': float(s.mean()), 'static_sd': float(s.std()),
                'static_p95': float(s.quantile(0.95)),
                'static_p10_mean': float(sp10.mean()),
                'static_p10_sd': float(sp10.std()),
                'static_p10_p95': float(sp10.quantile(0.95)),
                'leak_score': float(sp10.quantile(0.95)),  # the "what threshold can a sig use" stat
            })
    dist = pd.DataFrame(rows)
    dist.to_csv(OUT_DIST, index=False)
    print(f'Wrote {OUT_DIST.name}: {len(dist)} rows')

    # Print a focused table for the AUs we most care about
    print(f'\n{"AU":<6}{"side":<6}{"def_mean":>10}{"def_p95":>10}'
          f'{"stat_mean":>11}{"stat_p95":>10}{"stat_p10_p95":>14}')
    print('-' * 67)
    for au in HYP_AUS:
        for side in ('Left', 'Right'):
            r = dist[(dist['au'] == au) & (dist['side'] == side)].iloc[0]
            print(f'{au:<6}{side:<6}{r["default_mean"]:>10.3f}'
                  f'{r["default_p95"]:>10.3f}{r["static_mean"]:>11.3f}'
                  f'{r["static_p95"]:>10.3f}{r["static_p10_p95"]:>14.3f}')

    # ---------- 3.4: Asymmetry-in-controls check ----------
    banner("Phase B.4 — Asymmetry in controls (|Left - Right| at BL)")
    rows = []
    for _, row in cdf.iterrows():
        pid = str(row['Patient ID'])
        for au in AU_ORDER:
            l_def = row.get(f'BL_Left {au}_r', np.nan)
            r_def = row.get(f'BL_Right {au}_r', np.nan)
            l_stat = row.get(f'BL_Left {au}_r_static', np.nan)
            r_stat = row.get(f'BL_Right {au}_r_static', np.nan)
            l_stat_p10 = row.get(f'BL_Left {au}_r_static_p10', np.nan)
            r_stat_p10 = row.get(f'BL_Right {au}_r_static_p10', np.nan)
            rows.append({
                'patient_id': pid, 'au': au,
                'default_diff': abs(l_def - r_def) if pd.notna(l_def) and pd.notna(r_def) else np.nan,
                'static_diff': abs(l_stat - r_stat) if pd.notna(l_stat) and pd.notna(r_stat) else np.nan,
                'static_p10_diff': abs(l_stat_p10 - r_stat_p10) if pd.notna(l_stat_p10) and pd.notna(r_stat_p10) else np.nan,
            })
    asym = pd.DataFrame(rows)
    asym.to_csv(OUT_ASYM, index=False)
    print(f'Wrote {OUT_ASYM.name}: {len(asym)} rows')

    # Per-AU summary
    print(f'\n{"AU":<6}{"def_p95":>10}{"def_max":>10}{"stat_p95":>10}'
          f'{"stat_max":>10}{"sp10_p95":>10}{"sp10_max":>10}')
    print('-' * 66)
    for au in AU_ORDER:
        sub = asym[asym['au'] == au]
        d = sub['default_diff'].dropna()
        s = sub['static_diff'].dropna()
        sp10 = sub['static_p10_diff'].dropna()
        if len(d) == 0:
            continue
        print(f'{au:<6}{d.quantile(0.95):>10.3f}{d.max():>10.3f}'
              f'{s.quantile(0.95):>10.3f}{s.max():>10.3f}'
              f'{sp10.quantile(0.95):>10.3f}{sp10.max():>10.3f}')

    # ---------- 3.5: Default-mode-as-designed sanity ----------
    banner("Phase B.5 — Default mode as-designed sanity (control BL should be ~0)")
    rows = []
    for au in AU_ORDER:
        for side in ('Left', 'Right'):
            def_col = f'BL_{side} {au}_r'
            stat_col = f'BL_{side} {au}_r_static_p10'
            d = cdf[def_col].astype(float).dropna()
            s = cdf[stat_col].astype(float).dropna()
            rows.append({
                'au': au, 'side': side,
                'default_BL_max_value': float(d.max()),
                'default_BL_median_value': float(d.median()),
                'static_p10_BL_max_value': float(s.max()),
                'static_p10_BL_median_value': float(s.median()),
                'leak_factor': float(s.max()) / max(float(d.max()), 0.01),
            })
    sanity = pd.DataFrame(rows)
    sanity.to_csv(OUT_DEF_VS_STATIC, index=False)
    print(f'Wrote {OUT_DEF_VS_STATIC.name}: {len(sanity)} rows')

    print(f'\nWhat the running median compensates: '
          f'controls\' default BL max VS static p10 max')
    print(f'{"AU":<6}{"side":<6}{"def_BL_max":>12}{"static_p10_max":>16}'
          f'{"def_BL_med":>12}{"static_p10_med":>16}')
    print('-' * 68)
    for au in HYP_AUS + ['AU01', 'AU02', 'AU09', 'AU12']:
        for side in ('Left', 'Right'):
            r = sanity[(sanity['au'] == au) & (sanity['side'] == side)].iloc[0]
            print(f'{au:<6}{side:<6}{r["default_BL_max_value"]:>12.3f}'
                  f'{r["static_p10_BL_max_value"]:>16.3f}'
                  f'{r["default_BL_median_value"]:>12.3f}'
                  f'{r["static_p10_BL_median_value"]:>16.3f}')

    # ---------- Headline summary ----------
    banner("Phase B HEADLINE SUMMARY", sym='#')

    n_outliers_per_au = {}
    for au in HYP_AUS:
        for side in ('Left', 'Right'):
            col = f'{au}_static_p10'
            n = (anatomy[anatomy['side'] == side][col] >= 0.5).sum()
            n_outliers_per_au[f'{au}_{side}'] = n

    print('Outlier-control count per (AU, side) at static_p10 ≥ 0.5:')
    for k, v in sorted(n_outliers_per_au.items(), key=lambda x: -x[1]):
        flag = '  <- FACE-SHAPE LEAK' if v >= 3 else ''
        print(f'  {k:<14} {v}/13 controls{flag}')

    print('\nMost-affected controls (count of AU×side cells where they are outliers):')
    pid_count = {}
    for _, r in anatomy.iterrows():
        for au in HYP_AUS:
            v = r.get(f'{au}_static_p10', 0)
            if v >= 0.5:
                pid_count[r['patient_id']] = pid_count.get(r['patient_id'], 0) + 1
    for pid, n in sorted(pid_count.items(), key=lambda x: -x[1])[:10]:
        print(f'  {pid:<20} {n} cells')

    print(f'\nMax static asymmetry observed in any control × AU at BL:')
    max_asym = asym['static_diff'].max()
    max_row = asym[asym['static_diff'] == max_asym].iloc[0]
    print(f'  {max_row["au"]} {max_row["patient_id"]}: |L-R| = {max_asym:.3f}')


if __name__ == '__main__':
    main()
