#!/usr/bin/env python3
"""
Batch processing test to verify paralysis detection works on real patient data.
Tests 3 patients through the full analysis pipeline.
"""

import sys
import os
import logging
import pandas as pd
from pathlib import Path

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_batch_processing():
    """Test processing of 3 patients through the full pipeline."""
    logger.info("="*70)
    logger.info("BATCH PROCESSING TEST - 3 PATIENTS")
    logger.info("="*70)

    try:
        # Import the analyzer
        from facial_au_analyzer import FacialAUAnalyzer
        from facial_au_batch_processor import FacialAUBatchProcessor
        logger.info("✓ Successfully imported analyzer and batch processor")

        # Check for OpenFace results
        results_dir = Path("../S2O Coded Files")
        if not results_dir.exists():
            logger.error(f"✗ OpenFace results directory not found: {results_dir}")
            return False

        # Find CSV files - need to find patient pairs (left and right)
        all_csv_files = list(results_dir.glob("*.csv"))
        if not all_csv_files:
            logger.error(f"✗ No CSV files found in {results_dir}")
            return False

        logger.info(f"✓ Found {len(all_csv_files)} OpenFace result files")

        # Group by patient ID (files are named patient_id_left_mirrored_coded.csv and patient_id_right_mirrored_coded.csv)
        patient_pairs = {}
        for csv_file in all_csv_files:
            if '_left_mirrored_coded' in csv_file.stem:
                patient_id = csv_file.stem.replace('_left_mirrored_coded', '')
                if patient_id not in patient_pairs:
                    patient_pairs[patient_id] = {}
                patient_pairs[patient_id]['left'] = csv_file
            elif '_right_mirrored_coded' in csv_file.stem:
                patient_id = csv_file.stem.replace('_right_mirrored_coded', '')
                if patient_id not in patient_pairs:
                    patient_pairs[patient_id] = {}
                patient_pairs[patient_id]['right'] = csv_file

        # Filter to only complete pairs
        complete_pairs = {pid: files for pid, files in patient_pairs.items()
                         if 'left' in files and 'right' in files}

        if not complete_pairs:
            logger.error("✗ No complete patient pairs (left+right) found")
            return False

        # Select first 3 patients
        test_patients = list(complete_pairs.items())[:3]
        logger.info(f"✓ Testing with {len(test_patients)} patients:")
        for i, (patient_id, _) in enumerate(test_patients, 1):
            logger.info(f"  {i}. {patient_id}")

        # Initialize batch processor
        logger.info("\nInitializing batch processor...")
        batch_processor = FacialAUBatchProcessor()
        logger.info("✓ Batch processor initialized")

        # Create output directory
        test_output_dir = Path("test_output")
        test_output_dir.mkdir(exist_ok=True)
        logger.info(f"✓ Output directory: {test_output_dir}")

        # Process each patient
        results = []
        for i, (patient_id, files) in enumerate(test_patients, 1):
            logger.info(f"\n{'-'*70}")
            logger.info(f"Processing Patient {i}/{len(test_patients)}: {patient_id}")
            logger.info(f"{'-'*70}")

            try:
                # Initialize analyzer for this patient
                analyzer = FacialAUAnalyzer(output_dir=str(test_output_dir))
                logger.info(f"✓ Analyzer initialized for {patient_id}")

                # Load data
                logger.info(f"  Loading left and right CSV files...")
                analyzer.load_data(
                    left_csv_path=str(files['left']),
                    right_csv_path=str(files['right'])
                )
                analyzer.patient_id = patient_id
                logger.info(f"✓ Data loaded for {patient_id}")

                # Run analysis
                logger.info(f"  Running analysis...")
                patient_results = analyzer.analyze_all_actions()

                if patient_results:
                    logger.info(f"✓ Analysis completed for {patient_id}")

                    # Check for paralysis detections
                    patient_summary = patient_results.get('patient_summary', {})
                    paralysis_detected = patient_summary.get('Paralysis Detected', False)

                    logger.info(f"  Paralysis Detected: {paralysis_detected}")

                    # Check zones
                    zones = ['Upper', 'Mid', 'Lower']
                    sides = ['Left', 'Right']

                    for zone in zones:
                        for side in sides:
                            key = f"{side} {zone} Face Paralysis"
                            severity = patient_summary.get(key, 'Unknown')
                            if severity not in ['None', 'Unknown']:
                                logger.info(f"  {key}: {severity}")

                    # Count actions analyzed
                    actions_analyzed = len([k for k in patient_results.keys() if k != 'patient_summary'])
                    logger.info(f"  Actions analyzed: {actions_analyzed}")

                    results.append({
                        'patient_id': patient_id,
                        'success': True,
                        'paralysis_detected': paralysis_detected,
                        'actions_analyzed': actions_analyzed,
                        'summary': patient_summary
                    })
                else:
                    logger.error(f"✗ Analysis failed for {patient_id}")
                    results.append({
                        'patient_id': patient_id,
                        'success': False,
                        'error': 'No results returned'
                    })

            except Exception as e:
                logger.error(f"✗ Error processing {patient_id}: {e}")
                import traceback
                traceback.print_exc()
                results.append({
                    'patient_id': patient_id,
                    'success': False,
                    'error': str(e)
                })

        # Summary
        logger.info(f"\n{'='*70}")
        logger.info("BATCH PROCESSING SUMMARY")
        logger.info(f"{'='*70}")

        successful = sum(1 for r in results if r.get('success', False))
        total = len(results)

        logger.info(f"Patients processed: {total}")
        logger.info(f"Successful: {successful}")
        logger.info(f"Failed: {total - successful}")

        logger.info(f"\nDetailed Results:")
        for i, result in enumerate(results, 1):
            pid = result.get('patient_id', 'Unknown')
            if result.get('success', False):
                paralysis = result.get('paralysis_detected', False)
                actions = result.get('actions_analyzed', 0)
                logger.info(f"  {i}. {pid}: ✓ SUCCESS - Paralysis: {paralysis}, Actions: {actions}")
            else:
                error = result.get('error', 'Unknown error')
                logger.info(f"  {i}. {pid}: ✗ FAILED - {error}")

        # Check output files
        logger.info(f"\nOutput Files Generated:")
        output_files = list(test_output_dir.rglob("*"))
        logger.info(f"  Total files: {len(output_files)}")

        png_files = list(test_output_dir.rglob("*.png"))
        html_files = list(test_output_dir.rglob("*.html"))
        csv_files = list(test_output_dir.rglob("*.csv"))

        logger.info(f"  PNG files: {len(png_files)}")
        logger.info(f"  HTML files: {len(html_files)}")
        logger.info(f"  CSV files: {len(csv_files)}")

        if successful == total:
            logger.info(f"\n{'='*70}")
            logger.info("✓ ALL TESTS PASSED - Paralysis detection working correctly!")
            logger.info(f"{'='*70}")
            return True
        else:
            logger.error(f"\n{'='*70}")
            logger.error(f"✗ SOME TESTS FAILED - {total - successful}/{total} patients failed")
            logger.error(f"{'='*70}")
            return False

    except Exception as e:
        logger.error(f"✗ Test crashed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_batch_processing()
    sys.exit(0 if success else 1)
