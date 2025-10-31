#!/usr/bin/env python3
"""
Quick Diagnostic Script for Performance Recovery
Executes immediate fixes and diagnostics
"""

import os
import shutil
import subprocess
import sys
from pathlib import Path

def print_section(title):
    """Print formatted section header"""
    print(f"\n{'='*70}")
    print(f"  {title}")
    print('='*70)

def clear_cache():
    """Clear the data cache directory"""
    print_section("ACTION 1: Clearing Data Cache")

    cache_dir = Path(__file__).parent / '.cache'

    if cache_dir.exists():
        print(f"Found cache directory: {cache_dir}")
        print(f"Cache size: {sum(f.stat().st_size for f in cache_dir.rglob('*') if f.is_file()) / 1024:.1f} KB")

        response = input("Delete cache? (y/n): ")
        if response.lower() == 'y':
            shutil.rmtree(cache_dir)
            print("✓ Cache cleared successfully")
            return True
        else:
            print("✗ Cache not cleared")
            return False
    else:
        print("ℹ No cache directory found (this is fine)")
        return True

def check_au04_in_config():
    """Check if AU04 is in upper face config"""
    print_section("ACTION 2: Checking Upper Face AU Configuration")

    config_path = Path(__file__).parent / 'paralysis_config.py'

    if not config_path.exists():
        print(f"✗ Config file not found: {config_path}")
        return False

    with open(config_path, 'r') as f:
        content = f.read()

    # Look for upper face AU configuration
    in_upper = False
    for i, line in enumerate(content.split('\n')):
        if "'upper'" in line.lower() and '{' in line:
            in_upper = True
        if in_upper and "'aus'" in line:
            print(f"Line {i+1}: {line.strip()}")
            if 'AU04' in line:
                print("\n⚠️  FOUND: AU04 is in upper face configuration!")
                print("   This differs from published code (which used only AU01, AU02)")
                print("\n   Recommendation: Remove AU04_r from upper face config")

                response = input("\nRemove AU04_r from upper face config? (y/n): ")
                if response.lower() == 'y':
                    # Create backup
                    backup_path = config_path.with_suffix('.py.backup')
                    shutil.copy(config_path, backup_path)
                    print(f"✓ Created backup: {backup_path}")

                    # Remove AU04_r
                    new_content = content.replace(
                        "'aus': ['AU01_r', 'AU02_r', 'AU04_r']",
                        "'aus': ['AU01_r', 'AU02_r']"
                    )

                    with open(config_path, 'w') as f:
                        f.write(new_content)

                    print("✓ Removed AU04_r from upper face config")
                    print("  Config now matches published version")
                    return True
                else:
                    print("✗ AU04_r not removed")
                    return False
            else:
                print("\n✓ AU04 not in upper face configuration (matches published)")
                return True

    print("✗ Could not find upper face AU configuration")
    return False

def compare_config_to_published():
    """Compare current config to published config"""
    print_section("DIAGNOSTIC: Config Comparison")

    current_dir = Path(__file__).parent
    published_dir = Path.home() / 'Documents/open2GR/3_Data_Analysis'

    current_config = current_dir / 'paralysis_config.py'
    published_config = published_dir / 'paralysis_config.py'

    if not published_config.exists():
        print(f"⚠️  Published config not found at: {published_config}")
        return

    print("Comparing configurations...")
    print(f"Current:   {current_config}")
    print(f"Published: {published_config}")

    # Key settings to check
    settings_to_check = [
        ("Lower class weights", "ZONE_CONFIG['lower']['training']['class_weights']"),
        ("Mid class weights", "ZONE_CONFIG['mid']['training']['class_weights']"),
        ("Upper class weights", "ZONE_CONFIG['upper']['training']['class_weights']"),
        ("Lower Optuna trials", "ZONE_CONFIG['lower']['training']['hyperparameter_tuning']['optuna']['n_trials']"),
        ("Mid Optuna trials", "ZONE_CONFIG['mid']['training']['hyperparameter_tuning']['optuna']['n_trials']"),
        ("Upper Optuna trials", "ZONE_CONFIG['upper']['training']['hyperparameter_tuning']['optuna']['n_trials']"),
    ]

    print("\nKey Configuration Values:")
    print("-" * 70)

    # This is a simplified check - would need actual config parsing for full comparison
    with open(current_config, 'r') as f:
        current_content = f.read()

    for setting_name, config_path in settings_to_check:
        # Simple string search (not robust, but quick)
        print(f"  {setting_name}: Check manually in config file")

def verify_data_files():
    """Verify input data files exist and are accessible"""
    print_section("DIAGNOSTIC: Data File Verification")

    files_to_check = [
        ("Combined Results", "~/Documents/SplitFace/S3O Results/combined_results.csv"),
        ("Expert Key", "~/Documents/SplitFace/FPRS FP Key.csv"),
        ("Coded Files", "~/Documents/SplitFace/S2O Coded Files/"),
    ]

    all_found = True
    for name, path_str in files_to_check:
        path = Path(path_str).expanduser()

        if path.exists():
            if path.is_file():
                size_mb = path.stat().st_size / (1024 * 1024)
                print(f"✓ {name}: Found ({size_mb:.1f} MB)")
            else:
                file_count = len(list(path.glob('*.csv')))
                print(f"✓ {name}: Found ({file_count} files)")
        else:
            print(f"✗ {name}: NOT FOUND at {path}")
            all_found = False

    return all_found

def run_quick_train_upper():
    """Offer to run quick upper face training"""
    print_section("ACTION 3: Quick Upper Face Retrain")

    print("This will retrain ONLY the upper face model to test if AU04 removal helps.")
    print("Estimated time: 10-15 minutes")

    response = input("\nRun upper face training now? (y/n): ")

    if response.lower() == 'y':
        script_dir = Path(__file__).parent
        training_script = script_dir / 'paralysis_training_pipeline.py'

        if not training_script.exists():
            print(f"✗ Training script not found: {training_script}")
            return False

        print("\nStarting training...")
        print("Command: python3 paralysis_training_pipeline.py upper")
        print("-" * 70)

        try:
            result = subprocess.run(
                [sys.executable, str(training_script), 'upper'],
                cwd=str(script_dir),
                check=True
            )

            print("-" * 70)
            print("✓ Training completed!")
            print("\nCheck the output above for performance metrics.")
            print("Look for accuracy and F1 scores.")
            return True

        except subprocess.CalledProcessError as e:
            print(f"✗ Training failed with error: {e}")
            return False
        except KeyboardInterrupt:
            print("\n✗ Training interrupted by user")
            return False
    else:
        print("✗ Training skipped")
        print("  You can run manually: python3 paralysis_training_pipeline.py upper")
        return False

def print_next_steps():
    """Print recommended next steps"""
    print_section("NEXT STEPS")

    print("""
Based on the diagnostic results:

1. If cache was cleared:
   → Run full training: python3 paralysis_training_pipeline.py

2. If AU04 was removed from upper face:
   → Check upper face results
   → If improved, AU04 was causing issues
   → If still poor, continue with deeper diagnostics

3. Compare results to benchmark:
   Lower Face Target:  Accuracy 0.84, F1 0.82, Partial F1 0.46
   Mid Face Target:    Accuracy 0.93, F1 0.92, Partial F1 0.67
   Upper Face Target:  Accuracy 0.83, F1 0.83, Partial F1 0.40

4. If still poor performance:
   → Check COMPREHENSIVE_PERFORMANCE_RECOVERY_PLAN.md
   → Follow PHASE 2 diagnostics

Full training command (all zones):
    python3 paralysis_training_pipeline.py

Or train zones individually:
    python3 paralysis_training_pipeline.py lower
    python3 paralysis_training_pipeline.py mid
    python3 paralysis_training_pipeline.py upper
""")

def main():
    """Main diagnostic execution"""
    print("\n" + "="*70)
    print("  PERFORMANCE RECOVERY DIAGNOSTIC TOOL")
    print("  Same Cohort, Different Performance - Quick Fix Attempt")
    print("="*70)

    print("""
This script will:
1. Clear data cache (if exists)
2. Check for AU04 in upper face config
3. Offer to retrain upper face for quick validation
4. Provide next steps based on results
""")

    response = input("\nProceed with diagnostics? (y/n): ")
    if response.lower() != 'y':
        print("Diagnostic cancelled")
        return

    # Run diagnostics
    results = {}

    # Action 1: Clear cache
    results['cache_cleared'] = clear_cache()

    # Action 2: Check AU04
    results['au04_handled'] = check_au04_in_config()

    # Diagnostic: Compare configs
    compare_config_to_published()

    # Diagnostic: Verify data files
    results['data_files_ok'] = verify_data_files()

    # Action 3: Offer to retrain upper
    if results['au04_handled'] or results['cache_cleared']:
        results['upper_retrained'] = run_quick_train_upper()

    # Print next steps
    print_next_steps()

    # Summary
    print_section("DIAGNOSTIC SUMMARY")
    print(f"Cache Cleared:      {'✓' if results.get('cache_cleared') else '✗'}")
    print(f"AU04 Handled:       {'✓' if results.get('au04_handled') else '✗'}")
    print(f"Data Files OK:      {'✓' if results.get('data_files_ok') else '✗'}")
    print(f"Upper Retrained:    {'✓' if results.get('upper_retrained') else '✗ (not attempted)'}")

    print("\nFor complete recovery plan, see:")
    print("  COMPREHENSIVE_PERFORMANCE_RECOVERY_PLAN.md")

if __name__ == '__main__':
    main()
