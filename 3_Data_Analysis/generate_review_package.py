# generate_review_package.py (v2.3.4 - Fix FutureWarning and ReviewAdvisor Warning)
import argparse
import sys
import pandas as pd
import numpy as np
import os
import logging
import joblib
from datetime import datetime
import json
import importlib

try:
    from review_advisor import ReviewAdvisor
    from consistency_checker import ConsistencyChecker
    from impact_predictor import ImpactPredictor
except ImportError as e:
    logging.critical(f"Failed to import helper classes (ReviewAdvisor, ConsistencyChecker, ImpactPredictor): {e}")
    logging.critical("Please ensure these .py files are in the same directory or in PYTHONPATH.")
    sys.exit(1)

logger = logging.getLogger(__name__)

ITEM_CONFIG_GLOBAL = {}
INPUT_FILES_GLOBAL = {}
REVIEW_CONFIG_GLOBAL = {}
ITEM_CLASS_MAP_GLOBAL = {}
ANALYSIS_DIR_GLOBAL = ""
FEATURE_MODULE_SUFFIX_GLOBAL = ""
_MAIN_CALL_COUNT_GLOBAL = 0


def setup_global_configs(analysis_type):
    global ITEM_CONFIG_GLOBAL, INPUT_FILES_GLOBAL, REVIEW_CONFIG_GLOBAL, \
        ITEM_CLASS_MAP_GLOBAL, ANALYSIS_DIR_GLOBAL, FEATURE_MODULE_SUFFIX_GLOBAL

    if analysis_type == 'paralysis':
        from paralysis_config import ZONE_CONFIG, INPUT_FILES, REVIEW_CONFIG, ANALYSIS_DIR
        from paralysis_utils import PARALYSIS_MAP
        ITEM_CONFIG_GLOBAL = ZONE_CONFIG
        INPUT_FILES_GLOBAL = INPUT_FILES
        REVIEW_CONFIG_GLOBAL = REVIEW_CONFIG
        ITEM_CLASS_MAP_GLOBAL = PARALYSIS_MAP
        ANALYSIS_DIR_GLOBAL = ANALYSIS_DIR
        FEATURE_MODULE_SUFFIX_GLOBAL = "_face_features"
        logger.info("Loaded configurations for PARALYSIS pipeline.")
    elif analysis_type == 'synkinesis':
        from synkinesis_config import SYNKINESIS_CONFIG, INPUT_FILES as SK_INPUT_FILES, \
            REVIEW_CONFIG as SK_REVIEW_CONFIG, ANALYSIS_DIR as SK_ANALYSIS_DIR, \
            CLASS_NAMES as SYNKINESIS_MAP  # Assuming CLASS_NAMES is the synkinesis map
        ITEM_CONFIG_GLOBAL = SYNKINESIS_CONFIG
        INPUT_FILES_GLOBAL = SK_INPUT_FILES
        REVIEW_CONFIG_GLOBAL = SK_REVIEW_CONFIG
        ITEM_CLASS_MAP_GLOBAL = SYNKINESIS_MAP  # Use the imported CLASS_NAMES
        ANALYSIS_DIR_GLOBAL = SK_ANALYSIS_DIR
        FEATURE_MODULE_SUFFIX_GLOBAL = "_features"
        logger.info("Loaded configurations for SYNKINESIS pipeline.")
    else:
        msg = f"Invalid analysis type for configuration setup: {analysis_type}"
        logger.error(msg)
        raise ValueError(msg)


def load_item_data(item_key, analysis_type):
    if not ITEM_CONFIG_GLOBAL or not ANALYSIS_DIR_GLOBAL:
        logger.error("Global configurations not set. Call setup_global_configs() for current analysis_type first.")
        raise RuntimeError("Global configurations not set for load_item_data.")

    item_config = ITEM_CONFIG_GLOBAL.get(item_key)
    if not item_config:
        raise ValueError(f"Config for item '{item_key}' not found in ITEM_CONFIG_GLOBAL (for {analysis_type}).")

    item_name_display = item_config.get('name', item_key.capitalize())
    logger.info(f"Loading data for {item_name_display} ({analysis_type})...")

    filenames = item_config.get('filenames', {})
    model_path = filenames.get('model')
    scaler_path = filenames.get('scaler')
    feature_list_path = filenames.get('feature_list')

    if not all([model_path, scaler_path, feature_list_path]):
        raise ValueError(f"Missing model artifacts paths for {item_key}")
    if not all(os.path.exists(p) for p in [model_path, scaler_path, feature_list_path] if p):
        missing_files = [p for p in [model_path, scaler_path, feature_list_path] if p and not os.path.exists(p)]
        raise FileNotFoundError(f"Artifact files missing for {item_key}: {', '.join(missing_files)}")

    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    feature_names = joblib.load(feature_list_path)

    review_candidates_path = filenames.get('review_candidates_csv')
    review_candidates_df = pd.DataFrame()
    if review_candidates_path and os.path.exists(review_candidates_path):
        try:
            review_candidates_df = pd.read_csv(review_candidates_path, keep_default_na=False, na_values=[''])
            if 'Patient ID' in review_candidates_df.columns:
                review_candidates_df['Patient ID'] = review_candidates_df['Patient ID'].astype(str).str.strip()
            if 'Side' in review_candidates_df.columns:
                review_candidates_df['Side'] = review_candidates_df['Side'].astype(str).str.strip()
            logger.info(f"Loaded review candidates from: {review_candidates_path}")
        except Exception as e:
            logger.error(f"Error loading review_candidates_csv '{review_candidates_path}': {e}")
            review_candidates_df = pd.DataFrame()
    else:
        logger.warning(f"Review candidates CSV not found for {item_key} at {review_candidates_path}. Using empty DF.")

    error_details_path = filenames.get('error_details_csv')
    error_details_df = pd.DataFrame()
    if error_details_path and os.path.exists(error_details_path):
        try:
            error_details_df = pd.read_csv(error_details_path, keep_default_na=False, na_values=[''])
            logger.info(f"Loaded error details from: {error_details_path}")
            if 'Patient ID' in error_details_df.columns:
                error_details_df['Patient ID'] = error_details_df['Patient ID'].astype(str).str.strip()
            if 'Side' in error_details_df.columns:
                error_details_df['Side'] = error_details_df['Side'].astype(str).str.strip()
        except Exception as e:
            logger.error(f"Error loading error_details_csv '{error_details_path}': {e}")
            error_details_df = pd.DataFrame()
    else:
        constructed_path = os.path.join(ANALYSIS_DIR_GLOBAL, item_key, f"{item_key}_error_details.csv")
        if error_details_path != constructed_path and os.path.exists(constructed_path):
            logger.warning(
                f"Configured error_details_csv '{error_details_path}' not found. Trying fallback '{constructed_path}'")
            try:
                error_details_df = pd.read_csv(constructed_path, keep_default_na=False, na_values=[''])
                logger.info(f"Loaded error details from fallback: {constructed_path}")
                if 'Patient ID' in error_details_df.columns:
                    error_details_df['Patient ID'] = error_details_df['Patient ID'].astype(str).str.strip()
                if 'Side' in error_details_df.columns:
                    error_details_df['Side'] = error_details_df['Side'].astype(str).str.strip()
            except Exception as e:
                logger.error(f"Error loading error_details_csv from fallback '{constructed_path}': {e}")
                error_details_df = pd.DataFrame()
        else:
            logger.warning(
                f"Error details CSV not found (checked config path: '{error_details_path}', fallback: '{constructed_path}'). Using empty DF.")

    return {
        'model': model, 'scaler': scaler, 'feature_names': feature_names,
        'review_candidates': review_candidates_df, 'error_details': error_details_df,
        'config': item_config
    }


def load_item_training_data(item_key, analysis_type):
    if not ITEM_CONFIG_GLOBAL or not INPUT_FILES_GLOBAL or not FEATURE_MODULE_SUFFIX_GLOBAL:
        logger.error("Global configs not set for training data. Call setup_global_configs() first.")
        raise RuntimeError("Global configs not set for training data loading.")

    item_config = ITEM_CONFIG_GLOBAL.get(item_key)
    if not item_config:
        raise ValueError(f"Config for item '{item_key}' not found for training data loading.")

    item_name_display = item_config.get('name', item_key.capitalize())
    module_name = f"{item_key}{FEATURE_MODULE_SUFFIX_GLOBAL}"

    try:
        logger.debug(f"Attempting to import module: {module_name}")
        feature_module = importlib.import_module(module_name)
        prepare_data_func = getattr(feature_module, 'prepare_data')
    except ModuleNotFoundError:
        logger.error(
            f"Feature module {module_name}.py not found for item {item_key} ({analysis_type}). Check PYTHONPATH/filename.")
        raise
    except AttributeError:
        logger.error(f"Function 'prepare_data' not found in {module_name}.py for item {item_key} ({analysis_type}).")
        raise

    results_csv = INPUT_FILES_GLOBAL.get('results_csv')
    expert_csv = INPUT_FILES_GLOBAL.get('expert_key_csv')

    if not results_csv or not os.path.exists(results_csv):
        raise FileNotFoundError(f"Input results CSV file not found: {results_csv}")
    if not expert_csv or not os.path.exists(expert_csv):
        raise FileNotFoundError(f"Input expert key CSV file not found: {expert_csv}")

    logger.info(f"[{item_name_display} ({analysis_type})] Preparing training data via {module_name}.prepare_data...")
    features, targets_arr, metadata = prepare_data_func(
        results_file=results_csv, expert_file=expert_csv,
    )

    if features is None or targets_arr is None or metadata is None:
        logger.error(f"[{item_name_display}] Data preparation returned None.")
        return pd.DataFrame(), pd.Series(name='Target', dtype=object), pd.DataFrame()

    targets_series = pd.Series(name='Target', dtype=object)

    if isinstance(targets_arr, np.ndarray):
        if targets_arr.size > 0:
            if np.issubdtype(targets_arr.dtype, np.integer):
                try:
                    targets_series = pd.Series(targets_arr, name='Target', dtype=int)
                except Exception as e:
                    logger.error(
                        f"[{item_name_display}] Error converting int array to int Series: {e}. Falling back to object dtype.")
                    targets_series = pd.Series(targets_arr, name='Target', dtype=object)
            elif np.issubdtype(targets_arr.dtype, np.floating):
                if np.isnan(targets_arr).any():
                    targets_series = pd.Series(targets_arr, name='Target', dtype=object)
                else:
                    if np.array_equal(targets_arr, targets_arr.astype(int)):
                        targets_series = pd.Series(targets_arr, name='Target').astype(int)
                    else:
                        targets_series = pd.Series(targets_arr, name='Target', dtype=object)
            elif targets_arr.dtype == object:
                targets_series = pd.Series(targets_arr, name='Target', dtype=object)
            else:
                targets_series = pd.Series(targets_arr, name='Target', dtype=object)
    elif isinstance(targets_arr, pd.Series):
        targets_series = targets_arr
    else:
        logger.error(
            f"[{item_name_display}] Unexpected type for targets from prepare_data: {type(targets_arr)}. Returning empty Series.")

    if not features.empty and not targets_series.empty:
        if len(targets_series) == len(features):
            targets_series.index = features.index
        elif len(targets_series) != len(features):
            logger.warning(
                f"[{item_name_display}] Length mismatch features ({len(features)}) vs targets ({len(targets_series)}). Index not aligned.")

    if 'Patient ID' in metadata.columns: metadata['Patient ID'] = metadata['Patient ID'].astype(str).str.strip()
    if 'Side' in metadata.columns: metadata['Side'] = metadata['Side'].astype(str).str.strip()
    return features, targets_series, metadata


def generate_review_package_for_item(item_key, analysis_type, output_dir):
    item_config = ITEM_CONFIG_GLOBAL.get(item_key)
    if not item_config:
        logger.error(f"No item_config found for item '{item_key}' in generate_review_package_for_item.")
        return

    item_name_display = item_config.get('name', item_key.capitalize())
    logger.info(f"Generating review package for {item_name_display} ({analysis_type})")
    os.makedirs(output_dir, exist_ok=True)

    try:
        item_data_loaded = load_item_data(item_key, analysis_type)
        features_train, targets_train, metadata_train = load_item_training_data(item_key, analysis_type)
    except Exception as e:
        logger.error(f"Failed to load data for item '{item_key}' ({analysis_type}): {e}", exc_info=True)
        metadata_file = os.path.join(output_dir, f'package_metadata_{item_key}_{analysis_type}.json')
        package_metadata = {'item_key': item_key, 'item_name': item_name_display, 'analysis_type': analysis_type,
                            'generated': datetime.now().isoformat(), 'status': f'failed - data loading error: {str(e)}'}
        try:
            with open(metadata_file, 'w') as f:
                json.dump(package_metadata, f, indent=2)
        except Exception as json_e:
            logger.error(f"Failed to write failure metadata for {item_key}: {json_e}")
        return

    if features_train.empty or targets_train.empty or metadata_train.empty:
        logger.error(f"[{item_name_display}] Training data empty after loading. Cannot proceed.")
        metadata_file = os.path.join(output_dir, f'package_metadata_{item_key}_{analysis_type}.json')
        package_metadata = {'item_key': item_key, 'item_name': item_name_display, 'analysis_type': analysis_type,
                            'generated': datetime.now().isoformat(), 'status': 'failed - empty training data'}
        try:
            with open(metadata_file, 'w') as f:
                json.dump(package_metadata, f, indent=2)
        except Exception as json_e:
            logger.error(f"Failed to write empty training data metadata for {item_key}: {json_e}")
        return

    advisor = ReviewAdvisor(item_key, item_config, ITEM_CLASS_MAP_GLOBAL, REVIEW_CONFIG_GLOBAL)
    consistency_checker = ConsistencyChecker(item_key, item_config, ITEM_CLASS_MAP_GLOBAL,
                                             similarity_threshold=REVIEW_CONFIG_GLOBAL.get('similarity_threshold',
                                                                                           0.95))
    impact_predictor = ImpactPredictor(
        item_key, item_data_loaded['model'], item_data_loaded['scaler'],
        item_data_loaded['feature_names'], item_config, ITEM_CLASS_MAP_GLOBAL
    )

    logger.info(f"[{item_name_display}] Checking label consistency...")
    inconsistency_df = consistency_checker.find_label_inconsistencies(features_train, targets_train, metadata_train)
    inconsistency_summary = consistency_checker.create_inconsistency_summary(features_train, targets_train,
                                                                             metadata_train,
                                                                             inconsistency_df=inconsistency_df)

    # --- Create training_ground_truth_df ---
    training_ground_truth_df = pd.DataFrame()
    if not metadata_train.empty and not targets_train.empty:
        targets_train_series = targets_train.copy()
        if isinstance(targets_train_series, np.ndarray):
            # Ensure index alignment if metadata_train has a specific index from features_train
            if metadata_train.index.size == targets_train_series.size:
                targets_train_series = pd.Series(targets_train_series, index=metadata_train.index,
                                                 name='Expert_Label_Num_Train')
            else:
                logger.warning(
                    f"[{item_name_display}] Index size mismatch between metadata_train and targets_train (ndarray). Using default RangeIndex for targets.")
                targets_train_series = pd.Series(targets_train_series, name='Expert_Label_Num_Train')

        elif not targets_train_series.index.equals(metadata_train.index):
            logger.warning(
                f"[{item_name_display}] Index mismatch targets_train (Series) vs metadata_train. Attempting to reindex targets_train.")
            try:
                # Reindex targets_train_series to match metadata_train's index
                targets_train_series = targets_train_series.reindex(metadata_train.index)
                targets_train_series.name = 'Expert_Label_Num_Train'  # Ensure name consistency
            except Exception as ve:
                logger.error(
                    f"[{item_name_display}] Failed to reindex targets_train_series: {ve}. Skipping training_ground_truth_df population.")
                # Create an empty series with the correct index to prevent downstream errors if possible
                targets_train_series = pd.Series(dtype=object, index=metadata_train.index,
                                                 name='Expert_Label_Num_Train')

        if 'Patient ID' in metadata_train.columns and 'Side' in metadata_train.columns and not targets_train_series.empty:
            training_ground_truth_df = metadata_train[['Patient ID', 'Side']].copy()
            # Map numeric targets_train_series (if numeric) to string names
            if pd.api.types.is_numeric_dtype(targets_train_series.dtype):
                training_ground_truth_df['Expert_Label_Name_train_gt'] = targets_train_series.map(ITEM_CLASS_MAP_GLOBAL)
            else:  # Assume it's already string names (e.g. if prepare_data returned string labels)
                training_ground_truth_df['Expert_Label_Name_train_gt'] = targets_train_series
            # Fix FutureWarning for chained assignment
            training_ground_truth_df['Expert_Label_Name_train_gt'] = training_ground_truth_df[
                'Expert_Label_Name_train_gt'].fillna('Unknown')
            logger.info(
                f"[{item_name_display}] Created training_ground_truth_df with {len(training_ground_truth_df)} entries.")
        else:
            logger.warning(
                f"[{item_name_display}] Could not create training_ground_truth_df due to missing Patient ID/Side in metadata_train or problematic targets_train_series.")
    else:
        logger.warning(
            f"[{item_name_display}] metadata_train or targets_train is empty. Cannot create training_ground_truth_df.")
    # --- End of training_ground_truth_df creation ---

    logger.info(f"[{item_name_display}] Generating prioritized review list...")
    review_df = advisor.generate_prioritized_review_list(
        item_data_loaded['review_candidates'],
        item_data_loaded['error_details'],
        inconsistency_summary,
        training_ground_truth_df  # Pass the newly created DataFrame
    )

    logger.info(f"[{item_name_display}] Identifying high-impact changes...")
    high_impact_df = pd.DataFrame()
    if review_df is not None and not review_df.empty:
        current_labels_for_impact = targets_train.copy()
        if not pd.api.types.is_numeric_dtype(current_labels_for_impact.dtype):
            numerical_map = {name: num for num, name in ITEM_CLASS_MAP_GLOBAL.items()}
            mapped_targets = current_labels_for_impact.map(numerical_map)
            if mapped_targets.isnull().any():
                unmapped_values = current_labels_for_impact[mapped_targets.isnull()].unique()
                logger.warning(
                    f"[{item_name_display}] Unmapped labels for ImpactPredictor: {unmapped_values}. Filling with -1.")
                current_labels_for_impact = mapped_targets.fillna(-1).astype(int)
            else:
                current_labels_for_impact = mapped_targets.astype(int)
        if pd.api.types.is_numeric_dtype(current_labels_for_impact.dtype) and not np.issubdtype(
                current_labels_for_impact.dtype, np.integer):
            logger.info(
                f"[{item_name_display}] Casting non-integer numeric targets (dtype: {current_labels_for_impact.dtype}) to int for ImpactPredictor.")
            try:
                if current_labels_for_impact.isnull().any(): current_labels_for_impact = current_labels_for_impact.fillna(
                    -1)
                current_labels_for_impact = current_labels_for_impact.astype(int)
            except Exception as e:
                logger.error(f"[{item_name_display}] Could not cast numeric targets to int: {e}. Using pd.to_numeric.")
                current_labels_for_impact = pd.to_numeric(current_labels_for_impact, errors='coerce').fillna(-1).astype(
                    int)

        final_numeric_targets_for_impact = current_labels_for_impact.values
        if not np.issubdtype(final_numeric_targets_for_impact.dtype, np.integer):
            logger.error(
                f"CRITICAL: Final targets for ImpactPredictor for {item_key} are not integer dtype ({final_numeric_targets_for_impact.dtype}) after all processing. Aborting impact prediction.")
        else:
            high_impact_df = impact_predictor.identify_high_impact_changes(
                review_df, features_train, final_numeric_targets_for_impact, metadata_train,
                top_k=REVIEW_CONFIG_GLOBAL.get('impact_predictor', {}).get('top_k_high_impact', 50)
            )
    else:
        logger.warning(f"[{item_name_display}] review_df is empty/None. Skipping high-impact change ID.")

    logger.info(f"[{item_name_display}] Exporting review package...")
    file_prefix = f"{item_key}_{analysis_type}"
    review_file = os.path.join(output_dir, f'review_recommendations_{file_prefix}.xlsx')
    advisor.export_review_spreadsheet(review_df, review_file)
    consistency_file = os.path.join(output_dir, f'consistency_report_{file_prefix}.txt')
    consistency_checker.generate_consistency_report(inconsistency_df, None, consistency_file)
    review_report_file = os.path.join(output_dir, f'review_report_{file_prefix}.txt')
    advisor.create_review_report(review_df, review_report_file)
    high_impact_file = os.path.join(output_dir, f'high_impact_changes_{file_prefix}.csv')
    if high_impact_df is not None and not high_impact_df.empty:
        high_impact_df.to_csv(high_impact_file, index=False)
    else:
        pd.DataFrame().to_csv(high_impact_file, index=False)

    metadata_file_path = os.path.join(output_dir, f'package_metadata_{file_prefix}.json')
    package_metadata = {
        'item_key': item_key, 'item_name': item_name_display, 'analysis_type': analysis_type,
        'generated': datetime.now().isoformat(),
        'total_review_cases': len(review_df) if review_df is not None else 0,
        'tier_breakdown': review_df[
            'Review_Tier'].value_counts().to_dict() if review_df is not None and not review_df.empty and 'Review_Tier' in review_df else {},
        'inconsistency_pairs': len(inconsistency_df) if inconsistency_df is not None else 0,
        'high_impact_suggestions': len(high_impact_df) if high_impact_df is not None else 0,
        'files': {'review_spreadsheet': os.path.basename(review_file),
                  'consistency_report': os.path.basename(consistency_file),
                  'review_report': os.path.basename(review_report_file),
                  'high_impact_changes': os.path.basename(high_impact_file)}
    }
    with open(metadata_file_path, 'w') as f:
        json.dump(package_metadata, f, indent=2)
    logger.info(f"[{item_name_display}] Review package generated successfully in {output_dir}")

    print(f"\nReview Package Summary for {item_name_display} ({analysis_type}):")
    if review_df is not None and not review_df.empty:
        print(f"  Total cases for review: {len(review_df)}")
        if 'Review_Tier' in review_df:
            tiers = sorted(review_df['Review_Tier'].unique())
            for tier_val in tiers:
                tier_config_info = REVIEW_CONFIG_GLOBAL.get('review_tiers', {}).get(tier_val, {})
                tier_name = tier_config_info.get('name', f'Tier {tier_val}')
                count = len(review_df[review_df['Review_Tier'] == tier_val])
                print(f"  {tier_name}: {count}")
    else:
        print(f"  Total cases for review: 0")
    print(f"  Inconsistency pairs found: {len(inconsistency_df) if inconsistency_df is not None else 0}")
    print(f"  High-impact suggestions: {len(high_impact_df) if high_impact_df is not None else 0}")
    print(f"\nFiles created in: {output_dir}")


def main():
    global _MAIN_CALL_COUNT_GLOBAL, ITEM_CONFIG_GLOBAL
    _MAIN_CALL_COUNT_GLOBAL += 1

    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(name)s - %(message)s', force=True)
    logger.info(f"ENTERING generate_review_package.main() - CALL #{_MAIN_CALL_COUNT_GLOBAL}. Args: {sys.argv}")

    parser = argparse.ArgumentParser(description='Generate review package for expert key optimization')
    parser.add_argument('--analysis-type', type=str, choices=['paralysis', 'synkinesis'], required=False, default=None,
                        help='Type of analysis pipeline to run. If not specified, both are run.')
    parser.add_argument('--item', type=str, help='Specific zone (paralysis) or type (synkinesis) to analyze.')
    parser.add_argument('--all-items', action='store_true',
                        help='Generate for all zones/types of the specified analysis-type(s).')
    parser.add_argument('--output', type=str, default=None, help='Base output directory.')

    args = parser.parse_args()
    logger.info(
        f"Arguments parsed: analysis_type={args.analysis_type}, item={args.item}, all_items={args.all_items}, output={args.output}")

    analysis_pipelines_to_run = []
    if args.analysis_type:
        analysis_pipelines_to_run.append(args.analysis_type)
    else:
        analysis_pipelines_to_run = ['paralysis', 'synkinesis']
        logger.info("No --analysis-type specified, attempting both paralysis and synkinesis pipelines.")

    base_output_dir_arg = args.output
    if base_output_dir_arg is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        base_output_dir_arg = f"review_packages_{timestamp}"
        logger.info(f"No output directory specified. Using default base: {base_output_dir_arg}")
    os.makedirs(base_output_dir_arg, exist_ok=True)

    for current_analysis_type in analysis_pipelines_to_run:
        logger.info(f"\n{'=' * 30} PROCESSING PIPELINE: {current_analysis_type.upper()} {'=' * 30}")
        pipeline_base_output_dir = os.path.join(base_output_dir_arg, current_analysis_type)
        os.makedirs(pipeline_base_output_dir, exist_ok=True)

        try:
            setup_global_configs(current_analysis_type)
        except Exception as e:
            logger.critical(f"Configuration setup failed for {current_analysis_type}: {e}", exc_info=True)
            logger.critical(f"Skipping {current_analysis_type} pipeline.")
            continue

        if current_analysis_type == 'paralysis':
            from paralysis_config import get_all_zones as get_all_items_func
        elif current_analysis_type == 'synkinesis':
            from synkinesis_config import get_all_synkinesis_types as get_all_items_func
        else:
            logger.error(f"Internal error: Unhandled analysis type '{current_analysis_type}' for get_all_items_func.")
            continue

        items_to_process_for_pipeline = []
        if args.all_items or (args.item is None and not args.analysis_type):
            items_to_process_for_pipeline = list(get_all_items_func())
        elif args.item:
            if args.analysis_type is None or args.analysis_type == current_analysis_type:
                if args.item in ITEM_CONFIG_GLOBAL:
                    items_to_process_for_pipeline = [args.item]
                else:
                    logger.error(
                        f"Specified item '{args.item}' not found in config for {current_analysis_type}. Available: {list(ITEM_CONFIG_GLOBAL.keys())}")

        elif args.analysis_type == current_analysis_type and not args.item and not args.all_items:
            items_to_process_for_pipeline = list(get_all_items_func())

        if not items_to_process_for_pipeline:
            logger.info(f"No specific items to process for {current_analysis_type} based on arguments.")
            if args.item and args.analysis_type and args.analysis_type != current_analysis_type:
                logger.info(
                    f"Item '{args.item}' was specified for analysis_type '{args.analysis_type}', but current pipeline is '{current_analysis_type}'. Skipping item for this pipeline.")
            elif args.item and args.item not in ITEM_CONFIG_GLOBAL and (
                    args.analysis_type is None or args.analysis_type == current_analysis_type):
                logger.info(
                    f"Item '{args.item}' was specified but is not a valid item for the '{current_analysis_type}' pipeline.")
            # Removed redundant log for: elif not args.all_items and not args.item and args.analysis_type == current_analysis_type:

        for item_key_loop in items_to_process_for_pipeline:
            item_specific_output_dir = os.path.join(pipeline_base_output_dir, item_key_loop)
            logger.info(
                f"Processing item: {item_key_loop} ({current_analysis_type}). Output dir: {item_specific_output_dir}")
            try:
                generate_review_package_for_item(item_key_loop, current_analysis_type, item_specific_output_dir)
            except Exception as e:
                logger.error(
                    f"Failed to generate review package for item '{item_key_loop}' ({current_analysis_type}): {e}",
                    exc_info=True)

    logger.info(f"All processing finished (Call #{_MAIN_CALL_COUNT_GLOBAL}).")


if __name__ == '__main__':
    script_dir = os.path.dirname(os.path.abspath(__file__))
    if script_dir not in sys.path: sys.path.insert(0, script_dir)
    parent_dir = os.path.dirname(script_dir)
    if parent_dir not in sys.path: sys.path.insert(0, parent_dir)
    main()