# paralysis_training_pipeline.py

import logging
import os
import sys
import pandas as pd
import numpy as np
import warnings

from sklearn.model_selection import train_test_split

# Suppress specific warnings globally for the pipeline run
# This is also set in other files but good to have at the entry point.
os.environ['PYTHONWARNINGS'] = 'ignore::UserWarning,ignore::FutureWarning'
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", message=".*use_label_encoder.*", category=UserWarning)  # XGBoost specific
warnings.filterwarnings("ignore", message=".*is_sparse.*", category=UserWarning)  # Pandas sparse warning
warnings.filterwarnings("ignore", message=".*X does not have valid feature names.*",
                        category=UserWarning)  # Scaler warning if no names

# Enable scikit-learn metadata routing globally at the very start
try:
    import sklearn

    sklearn.set_config(enable_metadata_routing=True)
    print("INFO: Scikit-learn metadata routing enabled globally via training_pipeline.py.")
except Exception as e_sklearn_config:
    print(f"WARNING: Could not enable scikit-learn metadata routing: {e_sklearn_config}")

# Import from project structure (direct imports as files are in the same directory)
try:
    from paralysis_config import (
        ZONE_CONFIG, LOGGING_CONFIG, INPUT_FILES,  # CLASS_NAMES removed, use PARALYSIS_MAP from utils
        ADVANCED_TRAINING_CONFIG, MODEL_DIR, LOG_DIR, ANALYSIS_DIR,
        get_performance_targets, print_config_summary, validate_config  # Added validate_config
    )
    from paralysis_utils import (
        prepare_data_generalized, generate_review_candidates,
        perform_error_analysis, analyze_critical_errors, analyze_partial_errors,
        PARALYSIS_MAP  # Use PARALYSIS_MAP as the global class names map
    )

    CONFIG_LOADED = True
    UTILS_LOADED = True
except ImportError as e:
    print(f"CRITICAL: Failed to import base configs or utils - {e}")
    CONFIG_LOADED = False
    UTILS_LOADED = False
    # Fallback definitions (essential for script to not crash immediately if imports fail)
    ZONE_CONFIG = {}
    LOGGING_CONFIG = {'level': 'INFO', 'format': '%(asctime)s - %(levelname)s - %(name)s - %(message)s'}
    INPUT_FILES = {}
    ADVANCED_TRAINING_CONFIG = {}
    MODEL_DIR = 'models'
    LOG_DIR = 'logs'
    ANALYSIS_DIR = 'analysis_results'
    PARALYSIS_MAP = {0: 'None', 1: 'Partial', 2: 'Complete'}  # Default if config fails


    def prepare_data_generalized(*args, **kwargs):
        return None, None, None


    def generate_review_candidates(*args, **kwargs):
        return pd.DataFrame()


    def get_performance_targets(*args, **kwargs):
        return {}


    def print_config_summary():
        print("Config summary (unavailable due to import error).")


    def validate_config():
        return ["Config validation unavailable due to import error."]


    def perform_error_analysis(*args, **kwargs):
        pass


    def analyze_critical_errors(*args, **kwargs):
        pass


    def analyze_partial_errors(*args, **kwargs):
        pass

from paralysis_model_trainer import train_model_workflow
from paralysis_training_helpers import (
    setup_logging_for_zone, save_model_artifacts,
    log_performance_summary, analyze_optuna_study,
    apply_optimized_thresholds
)

# Global logger for the pipeline script itself, distinct from zone-specific loggers
pipeline_script_logger = logging.getLogger("ParalysisTrainingPipeline")
if not pipeline_script_logger.hasHandlers():  # Setup only if not already configured (e.g. by another import)
    script_handler = logging.StreamHandler()
    script_formatter = logging.Formatter(
        LOGGING_CONFIG.get('format', '%(asctime)s - %(levelname)s - %(name)s - %(message)s'))
    script_handler.setFormatter(script_formatter)
    pipeline_script_logger.addHandler(script_handler)
    pipeline_script_logger.setLevel(LOGGING_CONFIG.get('level', 'INFO').upper())


def run_zone_training_pipeline(zone_key_pipe, zone_config_all_pipe, input_files_all_pipe, class_names_global_pipe):
    """
    Runs the full training and evaluation pipeline for a single specified zone.
    """
    zone_specific_config_pipe = zone_config_all_pipe[zone_key_pipe]
    zone_name_display_pipe = zone_specific_config_pipe.get('name', zone_key_pipe.capitalize() + ' Face')
    filenames_cfg_pipe = zone_specific_config_pipe.get('filenames', {})
    log_file_pipe = filenames_cfg_pipe.get('training_log', os.path.join(LOG_DIR, f'{zone_key_pipe}_training.log'))

    # Setup zone-specific logging
    # This will reconfigure the root logger for this zone's file output
    setup_logging_for_zone(log_file_pipe, LOGGING_CONFIG.get('level', 'INFO'), LOGGING_CONFIG.get('format'))
    logger = logging.getLogger(__name__)  # Get logger for this module, now configured by setup_logging_for_zone

    logger.info(f"{'=' * 60}\nStarting Training Pipeline for Zone: {zone_name_display_pipe}\n{'=' * 60}")

    # Log key configuration settings for this zone run
    training_params_log_pipe = zone_specific_config_pipe.get('training', {})
    fs_params_log_pipe = zone_specific_config_pipe.get('feature_selection', {})
    smote_variant_log = training_params_log_pipe.get('smote', {}).get('variant', 'N/A')
    smoteenn_enabled_log = training_params_log_pipe.get('smote', {}).get('use_smoteenn_after', False)
    if smoteenn_enabled_log and smote_variant_log not in ['regular', '', None]:
        smote_variant_log_display = f"{smote_variant_log} (overridden to 'regular' for SMOTEENN base)"
    else:
        smote_variant_log_display = smote_variant_log

    logger.info(
        f"Key Configs - FS Enabled: {fs_params_log_pipe.get('enabled')}, FS Top N: {fs_params_log_pipe.get('top_n_features')}")
    logger.info(f"Key Configs - Optuna: {training_params_log_pipe.get('hyperparameter_tuning', {}).get('enabled')}, "
                f"SMOTE: {training_params_log_pipe.get('smote', {}).get('enabled')} (Variant: {smote_variant_log_display}, SMOTEENN: {smoteenn_enabled_log}), "
                f"ClassWeights: {training_params_log_pipe.get('class_weights', {})}, "
                f"ThresholdOpt: {training_params_log_pipe.get('threshold_optimization', {}).get('enabled')}")
    logger.info(f"Key Configs - Ensemble (Voting): {training_params_log_pipe.get('use_ensemble')}")

    logger.info(f"[{zone_name_display_pipe}] Preparing data...")
    results_csv_pipe = input_files_all_pipe.get('results_csv')
    expert_csv_pipe = input_files_all_pipe.get('expert_key_csv')

    # prepare_data_generalized now returns the initial full feature set
    features_df_pipe, targets_arr_pipe, metadata_df_pipe = prepare_data_generalized(
        zone_key=zone_key_pipe, results_file_path=results_csv_pipe, expert_file_path=expert_csv_pipe,
        base_config_dict=zone_config_all_pipe, input_files_global_dict=input_files_all_pipe,
        class_names_global_dict=class_names_global_pipe
    )
    if features_df_pipe is None or features_df_pipe.empty or targets_arr_pipe is None or len(targets_arr_pipe) == 0:
        logger.error(f"[{zone_name_display_pipe}] Data preparation failed or returned no data. Aborting zone.")
        return

    feature_names_list_pipe_initial = features_df_pipe.columns.tolist()  # Full list of features from prepare_data
    logger.info(
        f"[{zone_name_display_pipe}] Loaded {len(features_df_pipe)} samples with {len(feature_names_list_pipe_initial)} initial features.")

    unique_all_pipe, counts_all_pipe = np.unique(targets_arr_pipe, return_counts=True)
    class_dist_str_pipe = {class_names_global_pipe.get(k, str(k)): v for k, v in zip(unique_all_pipe, counts_all_pipe)}
    logger.info(f"[{zone_name_display_pipe}] Overall class distribution: {class_dist_str_pipe}")

    # Train-test split
    # sklearn is imported at the top, so direct use is fine.
    # from sklearn.model_selection import train_test_split # No longer needed here
    test_size_cfg_pipe = training_params_log_pipe.get('test_size', 0.25)
    random_state_cfg_pipe = training_params_log_pipe.get('random_state', 42)

    min_samples_for_stratify_pipe = 2  # Default for StratifiedShuffleSplit
    # Ensure all classes in targets_arr_pipe have at least min_samples_for_stratify_pipe for stratification
    can_stratify = False
    if len(unique_all_pipe) > 1:
        can_stratify = all(c >= min_samples_for_stratify_pipe for c in counts_all_pipe)

    stratify_arr_pipe = targets_arr_pipe if can_stratify else None
    if not can_stratify and len(unique_all_pipe) > 1:
        logger.warning(
            f"[{zone_name_display_pipe}] Cannot stratify train/test split due to low class counts {dict(zip(unique_all_pipe, counts_all_pipe))}. Using non-stratified split.")

    if metadata_df_pipe is None:  # Should ideally not happen if prepare_data is robust
        logger.warning(
            f"[{zone_name_display_pipe}] metadata_df_pipe is None from prepare_data. Creating dummy metadata for split.")
        metadata_df_pipe = pd.DataFrame(index=features_df_pipe.index)  # Match index of features_df_pipe
        # Add dummy 'Patient ID' and 'Side' if missing, using the index
        if 'Patient ID' not in metadata_df_pipe.columns:
            metadata_df_pipe['Patient ID'] = "UnknownID_Split_" + metadata_df_pipe.index.astype(str)
        if 'Side' not in metadata_df_pipe.columns:
            metadata_df_pipe['Side'] = "UnknownSide_Split"

    X_train_pipe, X_test_pipe, y_train_pipe, y_test_pipe, metadata_train_pipe, metadata_test_pipe = train_test_split(
        features_df_pipe, targets_arr_pipe, metadata_df_pipe,
        test_size=test_size_cfg_pipe, random_state=random_state_cfg_pipe, stratify=stratify_arr_pipe
    )
    logger.info(
        f"[{zone_name_display_pipe}] Train set: {X_train_pipe.shape[0]} samples, Test set: {X_test_pipe.shape[0]} samples")

    # --- Call the main model training workflow ---
    # train_model_workflow now returns selected_feature_names_mw as the 7th item
    trained_model_pipe, scaler_pipe, feat_importance_df_pipe, train_analysis_df_pipe, \
        optimal_thresholds_pipe, optuna_study_pipe, selected_feature_names_final_pipe = train_model_workflow(
        zone_key_pipe,
        X_train_pipe,  # DataFrame with initial full features
        y_train_pipe,
        X_test_pipe,  # DataFrame with initial full features (for context, not direct training by workflow)
        y_test_pipe,  # For context
        feature_names_list_pipe_initial,  # Pass the initial full list of feature names
        zone_specific_config_pipe,
        class_names_global_pipe
    )
    if trained_model_pipe is None:
        logger.error(f"[{zone_name_display_pipe}] Model training workflow failed. Aborting zone.")
        return

    logger.info(
        f"[{zone_name_display_pipe}] Final model trained using {len(selected_feature_names_final_pipe)} features: {selected_feature_names_final_pipe[:10]}...")

    # Save the *selected* feature list using the path from config
    selected_feature_list_path = filenames_cfg_pipe.get('feature_list')
    if selected_feature_list_path and selected_feature_names_final_pipe:
        try:
            os.makedirs(os.path.dirname(selected_feature_list_path), exist_ok=True)
            with open(selected_feature_list_path, 'w') as f_sel:
                for feature_name_sel in selected_feature_names_final_pipe:
                    f_sel.write(f"{feature_name_sel}\n")
            logger.info(
                f"[{zone_name_display_pipe}] Selected feature list ({len(selected_feature_names_final_pipe)}) saved to {selected_feature_list_path}")
        except Exception as e_save_selfl:
            logger.error(f"[{zone_name_display_pipe}] Failed to save selected feature list: {e_save_selfl}")

    logger.info(f"[{zone_name_display_pipe}] Evaluating final model on test set using selected features...")
    # X_test_pipe is a DataFrame with initial full features.
    # 1. Transform with the fitted scaler (which was fit on full features of X_train_pipe).
    # 2. Select the same features that the model was trained on.
    X_test_scaled_full_pipe = scaler_pipe.transform(X_test_pipe)

    # Create a DataFrame to easily select columns by name
    X_test_df_scaled_full_pipe = pd.DataFrame(X_test_scaled_full_pipe, columns=feature_names_list_pipe_initial,
                                              index=X_test_pipe.index)

    # Select the final features used for model training
    try:
        X_test_scaled_selected_pipe = X_test_df_scaled_full_pipe[selected_feature_names_final_pipe].values
    except KeyError as e_key_test_select:
        logger.error(f"[{zone_name_display_pipe}] Key error selecting features for test set: {e_key_test_select}. "
                     f"Available columns: {X_test_df_scaled_full_pipe.columns.tolist()[:10]}... "
                     f"Needed: {selected_feature_names_final_pipe[:10]}...")
        logger.error(
            f"[{zone_name_display_pipe}] Ensure selected_feature_names_final_pipe are present in feature_names_list_pipe_initial.")
        return  # Abort if features don't match

    y_pred_test_final_pipe = trained_model_pipe.predict(X_test_scaled_selected_pipe)
    optimal_thresholds_applied_pipe = False
    if optimal_thresholds_pipe:
        y_pred_test_final_pipe = apply_optimized_thresholds(trained_model_pipe, X_test_scaled_selected_pipe,
                                                            optimal_thresholds_pipe)
        optimal_thresholds_applied_pipe = True
        logger.info(f"[{zone_name_display_pipe}] Applied optimized thresholds to test set predictions.")

    y_proba_test_final_pipe = trained_model_pipe.predict_proba(X_test_scaled_selected_pipe)

    # Log performance summary
    log_performance_summary(
        zone_key_pipe, zone_name_display_pipe, y_test_pipe, y_pred_test_final_pipe, y_proba_test_final_pipe,
        class_names_global_pipe, get_performance_targets, optimal_thresholds_applied_pipe
    )

    # Save model artifacts (feat_importance_df_pipe is already based on selected features from trainer)
    save_model_artifacts(
        zone_key_pipe, filenames_cfg_pipe, trained_model_pipe, scaler_pipe,
        feat_importance_df_pipe, optimal_thresholds_pipe, optuna_study_pipe
    )

    # Analyze Optuna study if available
    if optuna_study_pipe:
        analyze_optuna_study(zone_key_pipe, zone_name_display_pipe, zone_specific_config_pipe)

    logger.info(f"[{zone_name_display_pipe}] Generating error analysis reports...")
    test_analysis_data_pipe = {
        'Expert': y_test_pipe,
        'Prediction': y_pred_test_final_pipe,
        'Patient ID': metadata_test_pipe[
            'Patient ID'].values if 'Patient ID' in metadata_test_pipe.columns and not metadata_test_pipe.empty else 'N/A',
        'Side': metadata_test_pipe[
            'Side'].values if 'Side' in metadata_test_pipe.columns and not metadata_test_pipe.empty else 'N/A'
    }
    for i_cls_pipe_key, i_cls_pipe_name in class_names_global_pipe.items():
        i_cls_pipe = int(i_cls_pipe_key)
        c_name_pipe = i_cls_pipe_name.replace(' ', '_')
        if i_cls_pipe < y_proba_test_final_pipe.shape[1]:
            test_analysis_data_pipe[f'Prob_{c_name_pipe}'] = y_proba_test_final_pipe[:, i_cls_pipe]
    test_analysis_df_for_errors_pipe = pd.DataFrame(test_analysis_data_pipe)

    error_report_dir_pipe = os.path.join(ANALYSIS_DIR, zone_key_pipe, 'error_reports')
    os.makedirs(error_report_dir_pipe, exist_ok=True)

    perform_error_analysis(test_analysis_df_for_errors_pipe, error_report_dir_pipe, f"{zone_key_pipe}_test_set",
                           zone_name_display_pipe, class_names_global_pipe)
    analyze_critical_errors(test_analysis_df_for_errors_pipe, error_report_dir_pipe, f"{zone_key_pipe}_test_set",
                            zone_name_display_pipe, class_names_global_pipe)
    analyze_partial_errors(test_analysis_df_for_errors_pipe, error_report_dir_pipe, f"{zone_key_pipe}_test_set",
                           zone_name_display_pipe, class_names_global_pipe)

    # Review Candidate Generation
    review_cfg_pipe = zone_specific_config_pipe.get('training', {}).get('review_analysis', {})
    if review_cfg_pipe.get('enabled', False) and UTILS_LOADED:
        logger.info(f"[{zone_name_display_pipe}] Generating review candidates...")
        review_csv_path_pipe = filenames_cfg_pipe.get('review_candidates_csv',
                                                      os.path.join(ANALYSIS_DIR, zone_key_pipe,
                                                                   f"{zone_key_pipe}_review_candidates.csv"))
        if review_csv_path_pipe:
            review_dir = os.path.dirname(review_csv_path_pipe)
            if review_dir: os.makedirs(review_dir, exist_ok=True)

        if train_analysis_df_pipe is None or train_analysis_df_pipe.empty:
            logger.warning(f"[{zone_name_display_pipe}] Training analysis data for review candidates is empty.")
        else:
            # Ensure metadata_train_pipe's index aligns with train_analysis_df_pipe's index
            # train_analysis_df_pipe is based on X_train_pipe (original indices)
            metadata_train_aligned_pipe = metadata_train_pipe.copy()
            if not metadata_train_aligned_pipe.index.equals(train_analysis_df_pipe.index):
                logger.warning(
                    f"[{zone_name_display_pipe}] Re-aligning metadata_train index for review candidates generation.")
                metadata_train_aligned_pipe = metadata_train_aligned_pipe.reindex(train_analysis_df_pipe.index)
                # Fill NaNs that might arise from reindexing if indices were mismatched
                if 'Patient ID' in metadata_train_aligned_pipe.columns and metadata_train_aligned_pipe[
                    'Patient ID'].isnull().any():
                    metadata_train_aligned_pipe['Patient ID'].fillna(f"UnknownID_RevReidx_{zone_key_pipe}",
                                                                     inplace=True)
                if 'Side' in metadata_train_aligned_pipe.columns and metadata_train_aligned_pipe['Side'].isnull().any():
                    metadata_train_aligned_pipe['Side'].fillna(f"UnknownSide_RevReidx_{zone_key_pipe}", inplace=True)

            # X_train_orig and X_test_orig for generate_review_candidates should be the DataFrames
            # with the *selected features* if influence analysis or similar feature-based logic inside it relies on them.
            # X_train_pipe and X_test_pipe are DataFrames with full initial features.
            X_train_selected_df_for_review = X_train_pipe[
                selected_feature_names_final_pipe] if selected_feature_names_final_pipe else X_train_pipe
            X_test_selected_df_for_review = X_test_pipe[
                selected_feature_names_final_pipe] if selected_feature_names_final_pipe else X_test_pipe

            review_candidates_pipe = generate_review_candidates(
                zone=zone_key_pipe, model=trained_model_pipe, scaler=scaler_pipe,  # scaler was fit on full features
                feature_names=selected_feature_names_final_pipe,  # Pass selected feature names
                config=zone_specific_config_pipe,
                X_train_orig=X_train_selected_df_for_review,  # DataFrame with selected features
                y_train_orig=y_train_pipe,
                X_test_orig=X_test_selected_df_for_review,  # DataFrame with selected features
                y_test_orig=y_test_pipe,
                metadata_train=metadata_train_aligned_pipe,  # Aligned metadata
                training_analysis_df=train_analysis_df_pipe,  # From workflow
                class_names_map=class_names_global_pipe
            )
            if review_candidates_pipe is not None and not review_candidates_pipe.empty:
                if review_csv_path_pipe:
                    review_candidates_pipe.to_csv(review_csv_path_pipe, index=False, float_format='%.4f')
                    logger.info(f"Saved {len(review_candidates_pipe)} review candidates to {review_csv_path_pipe}")
                else:
                    logger.warning(f"[{zone_name_display_pipe}] Review CSV path not defined, cannot save candidates.")
            else:
                logger.info(f"[{zone_name_display_pipe}] No review candidates generated.")

    logger.info(f"[{zone_name_display_pipe}] Training pipeline finished for zone {zone_key_pipe}!\n{'=' * 60}")


if __name__ == "__main__":
    if not CONFIG_LOADED or not UTILS_LOADED:
        pipeline_script_logger.critical("Essential config or utils could not be loaded. Aborting.")
        sys.exit(1)

    # Validate configuration from paralysis_config.py
    config_validation_errors = validate_config()
    if config_validation_errors:
        pipeline_script_logger.error("Configuration validation errors found:")
        for error in config_validation_errors:
            pipeline_script_logger.error(f"  - {error}")
        pipeline_script_logger.error("Please fix configuration errors before proceeding. Aborting.")
        sys.exit(1)
    else:
        pipeline_script_logger.info("Configuration validation passed.")

    # Optuna availability check
    optuna_globally_needed_main = any(
        z_cfg.get('training', {}).get('hyperparameter_tuning', {}).get('enabled', False) and
        z_cfg.get('training', {}).get('hyperparameter_tuning', {}).get('method') == 'optuna'
        for z_key, z_cfg in ZONE_CONFIG.items() if isinstance(z_cfg, dict)
    )
    if optuna_globally_needed_main:
        try:
            import optuna  # Check again to be sure

            pipeline_script_logger.info("Optuna is available and configured for use in at least one zone.")
        except ImportError:
            pipeline_script_logger.critical(
                "Optuna is configured for use but NOT installed. Please install it: pip install optuna. Aborting.")
            sys.exit(1)

    try:
        print_config_summary()  # Print summary using function from config file
    except Exception as e_summary_print_main:
        pipeline_script_logger.warning(f"Could not print config summary: {e_summary_print_main}")

    # Zone selection logic (from command line or default)
    if len(sys.argv) > 1:
        zones_to_run_main = [z for z in sys.argv[1:] if z in ZONE_CONFIG]
        if not zones_to_run_main:
            pipeline_script_logger.error(
                f"No valid zones specified from command line: {sys.argv[1:]}. Valid zones: {list(ZONE_CONFIG.keys())}")
            sys.exit(1)
    else:
        # Default to all zones defined in ZONE_CONFIG if none specified
        zones_to_run_main = list(ZONE_CONFIG.keys())
        if not zones_to_run_main:
            pipeline_script_logger.error(f"No zones found in ZONE_CONFIG. Nothing to train.")
            sys.exit(1)
        pipeline_script_logger.info(
            f"No zones specified via command line. Running all configured zones: {zones_to_run_main}")

    # Loop through selected zones and run the pipeline
    for zone_key_main_loop in zones_to_run_main:
        pipeline_script_logger.info(f"--- Processing Zone: {zone_key_main_loop.upper()} ---")
        try:
            # PARALYSIS_MAP (from utils, sourced from config.CLASS_NAMES) is passed as class_names_global_pipe
            run_zone_training_pipeline(zone_key_main_loop, ZONE_CONFIG, INPUT_FILES, PARALYSIS_MAP)
        except KeyboardInterrupt:
            pipeline_script_logger.error(f"Pipeline interrupted by user for zone {zone_key_main_loop}.")
            print(f"\n✗ Training interrupted by user for Zone: {zone_key_main_loop.upper()}")
            break  # Exit loop on interrupt
        except Exception as e_pipeline_main:
            pipeline_script_logger.error(f"Pipeline failed for zone {zone_key_main_loop}: {e_pipeline_main}",
                                         exc_info=True)
            print(f"\n✗ Processing Failed for Zone: {zone_key_main_loop.upper()} due to: {e_pipeline_main}")
            # Optionally, `continue` to next zone or `break` to stop all processing
            continue

    pipeline_script_logger.info(f"\n{'=' * 80}\nAll Specified Zone Processing Finished!\n{'=' * 80}")