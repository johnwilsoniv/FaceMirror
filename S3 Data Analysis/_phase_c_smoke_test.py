"""End-to-end Phase C smoke test against synthetic data.

Generates a small synthetic combined_results.csv with all the AU columns the
feature modules need, plus a synthetic expert key with realistic Yes/None/Not
Assessed mixes. Then runs the training pipeline (no Optuna tuning) for one
representative type per pattern (coupling, baseline_resting, baseline_asymmetry).

Run with the Open3 .venv (needs imblearn/seaborn/etc.):
    .venv/bin/python S3 Data Analysis/_phase_c_smoke_test.py
"""
import logging
import os
import shutil
import sys
import tempfile

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)

logging.basicConfig(level=logging.WARNING, format='%(levelname)s: %(message)s')

ACTIONS = ['BL', 'RE', 'ET', 'ES', 'BS', 'SS', 'SO', 'SE', 'PL', 'FR', 'BK', 'WN', 'BC', 'LT']
AUS = ['AU01_r', 'AU02_r', 'AU06_r', 'AU10_r', 'AU12_r', 'AU14_r',
       'AU15_r', 'AU17_r', 'AU25_r', 'AU45_r']


def synth_results(n_patients=80, seed=42):
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(n_patients):
        patient_id = f"PT{i:04d}"
        # Inject a "synkinesis-positive" half whose AUs covary differently
        positive = i % 2 == 0
        row = {'Patient ID': patient_id}
        for action in ACTIONS:
            for side in ('Left', 'Right'):
                for au in AUS:
                    base = float(rng.uniform(0.0, 1.5))
                    # Add a coupling signature: AU12/AU45/AU17 elevated on Left if positive
                    if positive and side == 'Left' and au in ('AU12_r', 'AU45_r', 'AU17_r'):
                        base += float(rng.uniform(0.3, 0.8))
                    row[f"{action}_{side} {au}"] = base
                    row[f"{action}_{side} {au} (Normalized)"] = base * float(rng.uniform(0.4, 0.8))
        rows.append(row)
    return pd.DataFrame(rows)


def synth_expert(patient_ids, seed=43):
    rng = np.random.default_rng(seed)
    rows = []
    for i, pid in enumerate(patient_ids):
        positive = i % 2 == 0
        row = {'Patient': pid}
        # ~50% positive on the synkinesis cols where i is even
        for col in [
            'Oral-Ocular Synkinesis Left', 'Oral-Ocular Synkinesis Right',
            'Snarl Smile Left', 'Snarl Smile Right',
            'Mentalis Synkinesis Left', 'Mentalis Synkinesis Right',
            'Hypertonicity Left', 'Hypertonicity Right',
        ]:
            row[col] = 'Yes' if positive and rng.random() > 0.2 else 'None'
        # Ocular-Oral has a "Not Assessed" tier
        for col in ['Ocular-Oral Synkinesis Left', 'Ocular-Oral Synkinesis Right']:
            r = rng.random()
            row[col] = 'Not Assessed' if r < 0.25 else ('Yes' if (positive and r > 0.5) else 'None')
        # Brow Cocked — rare
        for col in ['Brow Cocked Left', 'Brow Cocked Right']:
            row[col] = 'Yes' if (positive and rng.random() < 0.15) else 'None'
        rows.append(row)
    return pd.DataFrame(rows)


def main():
    tmpdir = tempfile.mkdtemp(prefix='syn_smoke_')
    print(f"Temp dir: {tmpdir}")

    results_df = synth_results(n_patients=80)
    expert_df = synth_expert(results_df['Patient ID'].tolist())

    results_csv = os.path.join(tmpdir, 'combined_results.csv')
    expert_csv = os.path.join(tmpdir, 'expert_key.csv')
    results_df.to_csv(results_csv, index=False)
    expert_df.to_csv(expert_csv, index=False)

    # Run the pipeline programmatically (skip tuning, skip artifact saves to keep test isolated).
    from synkinesis_training_pipeline import main as pipeline_main

    rc = pipeline_main([
        '--type', 'all',
        '--results', results_csv,
        '--expert', expert_csv,
        '--no-tune',
        '--no-save',
        '--log-level', 'WARNING',
    ])

    shutil.rmtree(tmpdir, ignore_errors=True)
    print(f"\nPipeline exit code: {rc}")
    if rc == 0:
        print("PHASE C SMOKE TEST: PASS")
    else:
        print("PHASE C SMOKE TEST: FAIL")
    return rc


if __name__ == '__main__':
    sys.exit(main())
