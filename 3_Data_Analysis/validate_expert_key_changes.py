# validate_expert_key_changes.py
import argparse
import pandas as pd
import numpy as np
import logging
import os
from datetime import datetime

from paralysis_config import ZONE_CONFIG, INPUT_FILES, REVIEW_CONFIG
from impact_predictor import ImpactPredictor
from consistency_checker import ConsistencyChecker
import joblib

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(name)s - %(message)s'
)
logger = logging.getLogger(__name__)


def compare_expert_keys(original_path, modified_path):
    """Compare two expert key files and identify changes"""
    logger.info("Comparing expert keys...")

    # Read both files
    original_df = pd.read_csv(original_path, dtype=str, keep_default_na=False, na_values=[''])
    modified_df = pd.read_csv(modified_path, dtype=str, keep_default_na=False, na_values=[''])

    # Ensure same structure
    if not original_df.columns.equals(modified_df.columns):
        logger.warning("Column mismatch between files")
        return None

    # Find changes
    changes = []

    # Assuming 'Patient' or 'Patient ID' column exists
    patient_col = 'Patient' if 'Patient' in original_df.columns else 'Patient ID'

    # Merge on patient ID to compare
    comparison_df = original_df.merge(
        modified_df,
        on=patient_col,
        suffixes=('_orig', '_mod'),
        how='outer'
    )

    # Check each zone's columns
    for zone, config in ZONE_CONFIG.items():
        expert_cols = config.get('expert_columns', {})

        for side in ['left', 'right']:
            col_name = expert_cols.get(side)
            if not col_name:
                continue

            orig_col = f"{col_name}_orig"
            mod_col = f"{col_name}_mod"

            if orig_col in comparison_df.columns and mod_col in comparison_df.columns:
                # Find rows where values changed
                mask = comparison_df[orig_col] != comparison_df[mod_col]
                changed_rows = comparison_df[mask]

                for _, row in changed_rows.iterrows():
                    changes.append({
                        'Patient ID': row[patient_col],
                        'Zone': zone,
                        'Side': side.capitalize(),
                        'Original_Label': row[orig_col],
                        'New_Label': row[mod_col],
                        'Column': col_name
                    })

    changes_df = pd.DataFrame(changes)
    logger.info(f"Found {len(changes_df)} changes")

    return changes_df


def validate_changes(changes_df, output_dir):
    """Validate the impact of changes"""

    os.makedirs(output_dir, exist_ok=True)

    validation_results = {}

    # Group changes by zone
    zones_affected = changes_df['Zone'].unique()

    for zone in zones_affected:
        zone_changes = changes_df[changes_df['Zone'] == zone]
        zone_name = ZONE_CONFIG[zone].get('name', zone.capitalize() + ' Face')

        logger.info(f"\nValidating {len(zone_changes)} changes for {zone_name}")

        # Load zone data
        config = ZONE_CONFIG[zone]
        filenames = config.get('filenames', {})

        # Load model and data
        model = joblib.load(filenames.get('model'))
        scaler = joblib.load(filenames.get('scaler'))
        feature_names = joblib.load(filenames.get('feature_list'))

        # Load training data
        import importlib
        module_name = f"{zone}_face_features"
        feature_module = importlib.import_module(module_name)
        prepare_data_func = getattr(feature_module, 'prepare_data')

        features, targets, metadata = prepare_data_func()

        if features is None:
            logger.error(f"Failed to load data for {zone}")
            continue

        # Initialize impact predictor
        predictor = ImpactPredictor(zone, model, scaler, feature_names, config)

        # Prepare proposed changes in correct format
        proposed_changes = []
        for _, change in zone_changes.iterrows():
            proposed_changes.append({
                'Patient ID': change['Patient ID'],
                'Side': change['Side'],
                'New_Label': change['New_Label']
            })

        proposed_changes_df = pd.DataFrame(proposed_changes)

        # Estimate impact
        impact_summary = predictor.estimate_performance_impact(
            features, targets, proposed_changes_df, metadata
        )

        validation_results[zone] = impact_summary

        # Save zone-specific report
        zone_report_path = os.path.join(output_dir, f'{zone}_validation_report.txt')
        write_zone_validation_report(zone, zone_changes, impact_summary, zone_report_path)

    # Create overall validation report
    overall_report_path = os.path.join(output_dir, 'validation_summary.txt')
    write_overall_validation_report(changes_df, validation_results, overall_report_path)

    # Check for new inconsistencies
    logger.info("\nChecking for new inconsistencies...")
    check_new_inconsistencies(changes_df, output_dir)

    return validation_results


def write_zone_validation_report(zone, changes_df, impact_summary, output_path):
    """Write validation report for a specific zone"""
    zone_name = ZONE_CONFIG[zone].get('name', zone.capitalize() + ' Face')

    report_lines = []
    report_lines.append(f"VALIDATION REPORT - {zone_name}")
    report_lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report_lines.append("=" * 60)
    report_lines.append("")

    # Change summary
    report_lines.append("CHANGE SUMMARY")
    report_lines.append("-" * 30)
    report_lines.append(f"Total changes: {len(changes_df)}")

    # Change types
    change_types = changes_df.apply(
        lambda row: f"{row['Original_Label']} -> {row['New_Label']}",
        axis=1
    ).value_counts()

    report_lines.append("\nChange types:")
    for change_type, count in change_types.items():
        report_lines.append(f"  {change_type}: {count}")

    # Impact summary
    report_lines.append("\n\nIMPACT ANALYSIS")
    report_lines.append("-" * 30)
    report_lines.append(f"Expected F1 improvement: {impact_summary.get('expected_f1_improvement', 0):.4f}")
    report_lines.append(
        f"Model agrees with changes: {impact_summary.get('model_agrees', 0)}/{impact_summary.get('total_changes', 0)}")
    report_lines.append(f"Average confidence in changes: {impact_summary.get('avg_confidence_in_changes', 0):.3f}")

    # Distribution impact
    report_lines.append("\nDistribution impact:")
    dist_shift = impact_summary.get('distribution_shift', {})
    for label, shift in dist_shift.items():
        label_name = {0: 'None', 1: 'Partial', 2: 'Complete'}.get(label, f'Class_{label}')
        report_lines.append(f"  {label_name}: {shift:+.3%}")

    # Recommendations
    report_lines.append("\nRECOMMENDATIONS")
    report_lines.append("-" * 30)
    for rec in impact_summary.get('recommendations', []):
        report_lines.append(f"- {rec}")

    # Write report
    with open(output_path, 'w') as f:
        f.write('\n'.join(report_lines))


def write_overall_validation_report(changes_df, validation_results, output_path):
    """Write overall validation summary"""
    report_lines = []
    report_lines.append("EXPERT KEY VALIDATION SUMMARY")
    report_lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report_lines.append("=" * 60)
    report_lines.append("")

    # Overall statistics
    report_lines.append("OVERALL STATISTICS")
    report_lines.append("-" * 30)
    report_lines.append(f"Total changes across all zones: {len(changes_df)}")
    report_lines.append(f"Zones affected: {', '.join(changes_df['Zone'].unique())}")

    # Per-zone summary
    report_lines.append("\n\nPER-ZONE SUMMARY")
    report_lines.append("-" * 30)

    overall_improvement = 0.0
    for zone, results in validation_results.items():
        zone_name = ZONE_CONFIG[zone].get('name', zone.capitalize() + ' Face')
        improvement = results.get('expected_f1_improvement', 0)
        overall_improvement += improvement

        report_lines.append(f"\n{zone_name}:")
        report_lines.append(f"  Expected F1 improvement: {improvement:.4f}")
        report_lines.append(f"  Significant: {'Yes' if improvement > 0.01 else 'No'}")

    # Overall recommendation
    report_lines.append("\n\nOVERALL RECOMMENDATION")
    report_lines.append("-" * 30)

    if overall_improvement > 0.01:
        report_lines.append("✓ Changes are likely to improve overall model performance")
        report_lines.append(f"  Expected total F1 improvement: {overall_improvement:.4f}")
    else:
        report_lines.append("⚠ Changes may not significantly improve performance")
        report_lines.append("  Consider reviewing the proposed changes")

    # Write report
    with open(output_path, 'w') as f:
        f.write('\n'.join(report_lines))

    # Also print summary to console
    print("\n" + "=" * 60)
    print("VALIDATION COMPLETE")
    print("=" * 60)
    print(f"Overall expected improvement: {overall_improvement:.4f}")
    print(f"Recommendation: {'PROCEED' if overall_improvement > 0.01 else 'REVIEW FURTHER'}")
    print(f"\nDetailed reports saved to: {os.path.dirname(output_path)}")


def check_new_inconsistencies(changes_df, output_dir):
    """Check if changes introduce new inconsistencies"""
    # This is a simplified check - you might want to expand this
    inconsistency_report = []

    # Group by patient to check cross-zone consistency
    patient_changes = changes_df.groupby('Patient ID')

    for patient_id, patient_df in patient_changes:
        if len(patient_df) > 1:
            # Patient has changes in multiple zones/sides
            new_labels = patient_df[['Zone', 'Side', 'New_Label']].values

            # Check for inconsistencies
            # e.g., Complete in one zone but None in another
            labels = [row[2] for row in new_labels]
            if 'Complete' in labels and 'None' in labels:
                inconsistency_report.append({
                    'Patient_ID': patient_id,
                    'Issue': 'Complete-None inconsistency across zones',
                    'Details': str(new_labels)
                })

    if inconsistency_report:
        report_df = pd.DataFrame(inconsistency_report)
        report_path = os.path.join(output_dir, 'new_inconsistencies.csv')
        report_df.to_csv(report_path, index=False)
        logger.warning(f"Found {len(report_df)} potential new inconsistencies. See {report_path}")


def main():
    parser = argparse.ArgumentParser(description='Validate expert key changes')
    parser.add_argument('--original', type=str, required=True,
                        help='Path to original expert key CSV')
    parser.add_argument('--modified', type=str, required=True,
                        help='Path to modified expert key CSV')
    parser.add_argument('--output', type=str,
                        default='validation_results',
                        help='Output directory for validation results')

    args = parser.parse_args()

    # Verify files exist
    if not os.path.exists(args.original):
        logger.error(f"Original file not found: {args.original}")
        return

    if not os.path.exists(args.modified):
        logger.error(f"Modified file not found: {args.modified}")
        return

    # Compare files
    changes_df = compare_expert_keys(args.original, args.modified)

    if changes_df is None or changes_df.empty:
        logger.info("No changes detected between files")
        return

    # Save changes summary
    os.makedirs(args.output, exist_ok=True)
    changes_path = os.path.join(args.output, 'changes_summary.csv')
    changes_df.to_csv(changes_path, index=False)
    logger.info(f"Changes summary saved to: {changes_path}")

    # Validate changes
    validation_results = validate_changes(changes_df, args.output)


if __name__ == '__main__':
    main()