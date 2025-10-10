#!/usr/bin/env python3
"""
Test script to verify paralysis detection still works after synkinesis cleanup.
Tests the main components that were modified.
"""

import sys
import logging

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

def test_imports():
    """Test that all imports work correctly."""
    logger.info("Testing imports...")
    try:
        from facial_au_constants import (
            ALL_AU_COLUMNS, AU_NAMES, FACIAL_ZONES,
            ASYMMETRY_THRESHOLDS, EXPERT_KEY_MAPPING,
            standardize_paralysis_label, standardize_binary_label,
            SEVERITY_ABBREVIATIONS, PARALYSIS_FINDINGS_KEYS,
            ZONE_SPECIFIC_ACTIONS, PATIENT_SUMMARY_COLUMNS
        )
        logger.info("  ✓ facial_au_constants imports successful")
        logger.info(f"  ✓ PATIENT_SUMMARY_COLUMNS has {len(PATIENT_SUMMARY_COLUMNS)} columns")

        from paralysis_utils import (
            calculate_ratio, calculate_percent_diff,
            standardize_paralysis_labels, process_binary_target,
            prepare_data_generalized
        )
        logger.info("  ✓ paralysis_utils imports successful")

        from facial_au_visualizer import FacialAUVisualizer
        logger.info("  ✓ facial_au_visualizer import successful")

        return True
    except ImportError as e:
        logger.error(f"  ✗ Import failed: {e}")
        return False

def test_visualizer_initialization():
    """Test that visualizer initializes without synkinesis components."""
    logger.info("Testing FacialAUVisualizer initialization...")
    try:
        from facial_au_visualizer import FacialAUVisualizer
        viz = FacialAUVisualizer()

        # Check that synkinesis attributes are NOT present
        if hasattr(viz, 'synkinesis_patterns'):
            logger.error("  ✗ Visualizer still has synkinesis_patterns attribute!")
            return False
        if hasattr(viz, 'synkinesis_types'):
            logger.error("  ✗ Visualizer still has synkinesis_types attribute!")
            return False
        if hasattr(viz, 'hypertonicity_aus'):
            logger.error("  ✗ Visualizer still has hypertonicity_aus attribute!")
            return False

        # Check that paralysis attributes ARE present
        if not hasattr(viz, 'au_names'):
            logger.error("  ✗ Visualizer missing au_names attribute!")
            return False
        if not hasattr(viz, 'facial_zones'):
            logger.error("  ✗ Visualizer missing facial_zones attribute!")
            return False

        logger.info("  ✓ Visualizer initialized correctly (no synkinesis components)")
        return True
    except Exception as e:
        logger.error(f"  ✗ Visualizer initialization failed: {e}")
        return False

def test_constants_content():
    """Test that constants don't contain synkinesis references."""
    logger.info("Testing constants content...")
    try:
        from facial_au_constants import PATIENT_SUMMARY_COLUMNS

        # Check that synkinesis columns are NOT in summary
        synk_keywords = ['synkinesis', 'hypertonicity', 'mentalis', 'ocular', 'oral', 'snarl', 'smile', 'brow', 'cocked']
        found_synk = []
        for col in PATIENT_SUMMARY_COLUMNS:
            col_lower = col.lower()
            for keyword in synk_keywords:
                if keyword in col_lower:
                    found_synk.append(col)
                    break

        if found_synk:
            logger.error(f"  ✗ Found synkinesis columns in PATIENT_SUMMARY_COLUMNS: {found_synk}")
            return False

        # Check that paralysis columns ARE present
        paralysis_cols = [c for c in PATIENT_SUMMARY_COLUMNS if 'paralysis' in c.lower()]
        if len(paralysis_cols) < 6:
            logger.error(f"  ✗ Expected at least 6 paralysis columns, found {len(paralysis_cols)}")
            return False

        logger.info(f"  ✓ PATIENT_SUMMARY_COLUMNS clean ({len(paralysis_cols)} paralysis columns, 0 synkinesis columns)")
        return True
    except Exception as e:
        logger.error(f"  ✗ Constants test failed: {e}")
        return False

def test_utils_functions():
    """Test that utils functions work correctly."""
    logger.info("Testing paralysis_utils functions...")
    try:
        from paralysis_utils import (
            calculate_ratio, calculate_percent_diff,
            standardize_paralysis_labels
        )
        import pandas as pd

        # Test calculate_ratio
        s1 = pd.Series([1.0, 2.0, 3.0])
        s2 = pd.Series([2.0, 2.0, 1.0])
        ratios = calculate_ratio(s1, s2)
        if len(ratios) != 3:
            logger.error("  ✗ calculate_ratio returned wrong length")
            return False
        logger.info("  ✓ calculate_ratio works")

        # Test calculate_percent_diff
        pct_diffs = calculate_percent_diff(s1, s2)
        if len(pct_diffs) != 3:
            logger.error("  ✗ calculate_percent_diff returned wrong length")
            return False
        logger.info("  ✓ calculate_percent_diff works")

        # Test standardize_paralysis_labels
        test_labels = ['None', 'Partial', 'Complete', 'mild', 'severe']
        results = [standardize_paralysis_labels(lbl) for lbl in test_labels]
        expected = ['None', 'Partial', 'Complete', 'Partial', 'Complete']
        if results != expected:
            logger.error(f"  ✗ standardize_paralysis_labels failed: {results} != {expected}")
            return False
        logger.info("  ✓ standardize_paralysis_labels works")

        return True
    except Exception as e:
        logger.error(f"  ✗ Utils functions test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run all tests."""
    logger.info("="*60)
    logger.info("PARALYSIS DETECTION TEST AFTER SYNKINESIS CLEANUP")
    logger.info("="*60)

    tests = [
        ("Imports", test_imports),
        ("Visualizer Initialization", test_visualizer_initialization),
        ("Constants Content", test_constants_content),
        ("Utils Functions", test_utils_functions),
    ]

    results = {}
    for test_name, test_func in tests:
        logger.info("")
        try:
            results[test_name] = test_func()
        except Exception as e:
            logger.error(f"Test '{test_name}' crashed: {e}")
            results[test_name] = False

    # Summary
    logger.info("")
    logger.info("="*60)
    logger.info("TEST SUMMARY")
    logger.info("="*60)

    passed = sum(1 for v in results.values() if v)
    total = len(results)

    for test_name, result in results.items():
        status = "✓ PASS" if result else "✗ FAIL"
        logger.info(f"{status}: {test_name}")

    logger.info("")
    logger.info(f"Results: {passed}/{total} tests passed")

    if passed == total:
        logger.info("✓ All tests PASSED - Paralysis detection working correctly!")
        return 0
    else:
        logger.error("✗ Some tests FAILED - Review errors above")
        return 1

if __name__ == "__main__":
    sys.exit(main())
