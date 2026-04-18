# synkinesis_training_pipeline.py
#
# CLI entrypoint for binary synkinesis training. Trains one type or all six.
#
# Usage:
#     python synkinesis_training_pipeline.py --type all
#     python synkinesis_training_pipeline.py --type ocular_oral --no-tune
#     python synkinesis_training_pipeline.py --type brow_cocked \
#         --results /path/to/combined_results.csv

import argparse
import logging
import os
import sys
import time
import traceback

import pandas as pd

from synkinesis_config import (
    ANALYSIS_DIR,
    SYNKINESIS_CONFIG,
    ensure_artifact_dirs,
    get_all_types,
)
from synkinesis_trainer import train_one_type

logger = logging.getLogger(__name__)


def _aggregate_label_review(types_run):
    """Build a cross-type roll-up of label review candidates. Patients flagged
    in multiple types are higher-priority review targets — could indicate label
    noise, multi-synkinesis presentations the experts coded inconsistently, or
    structurally hard cases the models should be re-checked against."""
    aggregates = {}
    for type_key in types_run:
        cfg = SYNKINESIS_CONFIG.get(type_key, {})
        cand_path = cfg.get('filenames', {}).get('review_candidates_csv')
        if not cand_path or not os.path.exists(cand_path):
            continue
        try:
            df = pd.read_csv(cand_path)
        except Exception:
            continue
        for _, row in df.iterrows():
            key = (str(row['patient_id']), str(row['side']))
            agg = aggregates.setdefault(key, {
                'patient_id': row['patient_id'],
                'side': row['side'],
                'flagged_types': [],
                'directions': set(),
                'max_disagreement': 0.0,
            })
            direction = 'FN' if row['true_label'] == 1 else 'FP'
            agg['flagged_types'].append(f'{type_key}({direction}:{row["model_proba"]:.2f})')
            agg['directions'].add(direction)
            agg['max_disagreement'] = max(agg['max_disagreement'], float(row['disagreement_score']))

    if not aggregates:
        return None

    rows = []
    for key, agg in aggregates.items():
        rows.append({
            'patient_id': agg['patient_id'],
            'side': agg['side'],
            'n_types_flagged': len(agg['flagged_types']),
            'directions': '/'.join(sorted(agg['directions'])),
            'max_disagreement': round(agg['max_disagreement'], 3),
            'flagged_in': '; '.join(agg['flagged_types']),
        })
    df = pd.DataFrame(rows).sort_values(
        ['n_types_flagged', 'max_disagreement'], ascending=[False, False]
    )
    out_path = os.path.join(ANALYSIS_DIR, 'synkinesis', 'aggregate_label_review.csv')
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    df.to_csv(out_path, index=False)
    return out_path, df


def _print_summary(results):
    if not results:
        print("\nNo runs completed.")
        return
    print("\n" + "=" * 105)
    print("SYNKINESIS TRAINING SUMMARY (threshold via OOF on train; metrics on held-out test)")
    print("=" * 105)
    print(f"{'Type':<16}{'N_tr':>6}{'Pos':>5}{'N_te':>6}{'Pos':>5}{'Feat':>6}"
          f"{'Thr':>7}{'F1+':>8}{'AP':>8}{'AUC':>8}{'BalAcc':>9}{'Review':>8}{'Status':>9}")
    print("-" * 105)
    for type_key, payload in results.items():
        if payload.get('error'):
            print(f"{type_key:<16}{'—':>6}{'—':>5}{'—':>6}{'—':>5}{'—':>6}"
                  f"{'—':>7}{'—':>8}{'—':>8}{'—':>8}{'—':>9}{'—':>8}{'FAILED':>9}")
            continue
        m = payload['metrics']
        print(
            f"{type_key:<16}"
            f"{m.get('n_train', 0):>6d}"
            f"{m.get('n_pos_train', 0):>5d}"
            f"{m.get('n_test', 0):>6d}"
            f"{m.get('n_pos_test', 0):>5d}"
            f"{m.get('n_features_selected', 0):>6d}"
            f"{m.get('threshold', 0.0):>7.3f}"
            f"{m.get('f1_positive', 0.0):>8.3f}"
            f"{m.get('average_precision', 0.0):>8.3f}"
            f"{m.get('roc_auc', float('nan')):>8.3f}"
            f"{m.get('balanced_accuracy', 0.0):>9.3f}"
            f"{m.get('n_review_candidates', 0):>8d}"
            f"{'OK':>9}"
        )
    print("=" * 105)


def main(argv=None):
    parser = argparse.ArgumentParser(description="Binary synkinesis training pipeline.")
    parser.add_argument(
        '--type', '-t', default='all',
        help=f"One of {get_all_types()} or 'all' (default: all).",
    )
    parser.add_argument('--results', help='Path to combined_results.csv (overrides config).')
    parser.add_argument('--expert', help='Path to expert key CSV (overrides config).')
    parser.add_argument('--no-tune', action='store_true',
                        help='Skip Optuna; use baseline params from config.')
    parser.add_argument('--no-save', action='store_true',
                        help='Run training but do not write any artifacts.')
    parser.add_argument('--log-level', default='INFO')
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format='%(asctime)s %(levelname)s %(name)s: %(message)s',
    )
    ensure_artifact_dirs()

    requested = list(SYNKINESIS_CONFIG) if args.type == 'all' else [args.type]
    invalid = [t for t in requested if t not in SYNKINESIS_CONFIG]
    if invalid:
        parser.error(f"Unknown type(s): {invalid}. Choose from {get_all_types()} or 'all'.")

    overall_start = time.time()
    results = {}
    for type_key in requested:
        start = time.time()
        try:
            metrics = train_one_type(
                type_key,
                results_csv=args.results,
                expert_csv=args.expert,
                skip_tuning=args.no_tune,
                save_artifacts=not args.no_save,
            )
            results[type_key] = {'metrics': metrics, 'elapsed': time.time() - start}
        except Exception as e:
            logger.error(f"[{type_key}] FAILED: {e}")
            traceback.print_exc()
            results[type_key] = {'error': str(e), 'elapsed': time.time() - start}

    _print_summary(results)

    if not args.no_save:
        agg = _aggregate_label_review([k for k, v in results.items() if not v.get('error')])
        if agg is not None:
            agg_path, agg_df = agg
            print("\n" + "=" * 105)
            print(f"LABEL SENSITIVITY — patients flagged in 2+ types (top 15 of {len(agg_df)})")
            print(f"Full report: {agg_path}")
            print("-" * 105)
            multi = agg_df[agg_df['n_types_flagged'] >= 2].head(15)
            if len(multi):
                print(multi.to_string(index=False))
            else:
                print("  (none — all flags isolated to single types)")
            print("=" * 105)

    print(f"\nTotal elapsed: {time.time() - overall_start:.1f}s")
    failures = [k for k, v in results.items() if v.get('error')]
    return 1 if failures else 0


if __name__ == '__main__':
    sys.exit(main())
