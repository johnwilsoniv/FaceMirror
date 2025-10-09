# paralysis_model_trainer.py

import logging
import numpy as np
import pandas as pd
import os
from copy import deepcopy
import gc

import warnings

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import train_test_split, StratifiedKFold
    from sklearn.calibration import CalibratedClassifierCV
    from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, VotingClassifier
    import xgboost as xgb

    try:
        import optuna

        OPTUNA_AVAILABLE = True
    except ImportError:
        OPTUNA_AVAILABLE = False

from paralysis_config import (
    ADVANCED_TRAINING_CONFIG, DATA_AUGMENTATION_CONFIG,
    PERFORMANCE_CONFIG, MODEL_DIR
)
from paralysis_utils import calculate_entropy, calculate_margin, PARALYSIS_MAP

from paralysis_training_helpers import (
    apply_data_augmentation, apply_smote_and_cleaning,
    create_optuna_objective, optimize_class_thresholds, apply_optimized_thresholds,
    get_class_specific_info, calculate_class_weights_for_model,
    get_common_feature_importance_df
)

logger = logging.getLogger(__name__)


def train_model_workflow(zone_key_mw, X_train_df_mw, y_train_arr_mw, X_test_df_mw, y_test_arr_mw,
                         feature_names_mw_initial, zone_config_mw, class_names_map_global_mw):
    if PERFORMANCE_CONFIG.get('memory_optimization', {}).get('enable_gc', True):
        gc.enable()

    zone_name_disp_mw = zone_config_mw.get('name', zone_key_mw.capitalize() + ' Face')
    training_params_mw = zone_config_mw.get('training', {})
    feature_sel_cfg_mw = zone_config_mw.get('feature_selection', {})
    smote_cfg_mw = training_params_mw.get('smote', {})
    tuning_cfg_mw = training_params_mw.get('hyperparameter_tuning', {})
    model_params_default_xgb_mw = training_params_mw.get('model_params', {}).copy()
    random_state_mw = training_params_mw.get('random_state', 42)
    calib_cfg_mw = training_params_mw.get('calibration', {})  # Used for method, split size for prefit
    # adv_calib_cfg_mw = ADVANCED_TRAINING_CONFIG.get('calibration', {}) # Not directly used now for CV type
    n_jobs_workflow = PERFORMANCE_CONFIG.get('parallel_processing', {}).get('n_jobs', 1)

    num_classes_mw = len(class_names_map_global_mw)
    class_specific_info_mw = get_class_specific_info(class_names_map_global_mw)
    # partial_class_index_mw = class_specific_info_mw['partial_class_index'] # Keep for Optuna if needed

    logger.info(f"[{zone_name_disp_mw}] Starting model training workflow...")
    logger.info(f"[{zone_name_disp_mw}] Initial features: {len(feature_names_mw_initial)}")

    scaler_mw = StandardScaler()
    X_train_scaled_mw_full = scaler_mw.fit_transform(X_train_df_mw)

    selected_feature_names_mw = feature_names_mw_initial
    X_train_scaled_mw_selected = X_train_scaled_mw_full.copy()

    if feature_sel_cfg_mw.get('enabled', False):
        logger.info(f"[{zone_name_disp_mw}] Performing preliminary feature selection...")
        top_n_features_fs = feature_sel_cfg_mw.get('top_n_features', len(feature_names_mw_initial))
        fs_model = RandomForestClassifier(n_estimators=100, random_state=random_state_mw, n_jobs=n_jobs_workflow,
                                          class_weight='balanced')
        X_fs_input, y_fs_input = X_train_scaled_mw_full, y_train_arr_mw
        if smote_cfg_mw.get('enabled', False) and not smote_cfg_mw.get('apply_per_fold_in_tuning', False):
            logger.info(f"[{zone_name_disp_mw}] Applying SMOTE before preliminary feature selection.")
            X_fs_input_smoted, y_fs_input_smoted = apply_smote_and_cleaning(
                X_train_scaled_mw_full, y_train_arr_mw, smote_cfg_mw, random_state_mw, zone_name_disp_mw + "-FS_SMOTE"
            )
            if X_fs_input_smoted.shape[0] > 0 and y_fs_input_smoted.shape[0] > 0:
                X_fs_input, y_fs_input = X_fs_input_smoted, y_fs_input_smoted
            else:
                logger.warning(f"[{zone_name_disp_mw}] SMOTE for FS resulted in empty data, using original for FS.")
        fs_model.fit(X_fs_input, y_fs_input)
        importances_fs = fs_model.feature_importances_
        indices_fs = np.argsort(importances_fs)[::-1]
        selected_feature_indices_fs = indices_fs[:top_n_features_fs]
        if len(selected_feature_indices_fs) > 0:
            selected_feature_names_mw = [feature_names_mw_initial[i] for i in selected_feature_indices_fs]
            X_train_scaled_mw_selected = X_train_scaled_mw_full[:, selected_feature_indices_fs]
            logger.info(
                f"[{zone_name_disp_mw}] Selected {len(selected_feature_names_mw)} features out of {len(feature_names_mw_initial)}.")
        else:
            logger.warning(
                f"[{zone_name_disp_mw}] Feature selection resulted in no features. Using all initial features.")
    else:
        logger.info(
            f"[{zone_name_disp_mw}] Preliminary feature selection disabled. Using all {len(feature_names_mw_initial)} features.")

    feature_names_mw = selected_feature_names_mw

    best_xgb_params_mw = model_params_default_xgb_mw.copy()
    if 'num_class' not in best_xgb_params_mw and num_classes_mw > 1: best_xgb_params_mw['num_class'] = num_classes_mw
    if 'objective' not in best_xgb_params_mw:
        best_xgb_params_mw['objective'] = 'multi:softprob' if num_classes_mw > 2 else 'binary:logistic'
    if 'eval_metric' not in best_xgb_params_mw:
        best_xgb_params_mw['eval_metric'] = 'mlogloss' if num_classes_mw > 2 else 'logloss'
    if best_xgb_params_mw.get('objective') == 'binary:logistic' and 'num_class' in best_xgb_params_mw:
        del best_xgb_params_mw['num_class']

    optuna_study_object_mw = None
    if tuning_cfg_mw.get('enabled', False) and tuning_cfg_mw.get('method') == 'optuna' and OPTUNA_AVAILABLE:
        # ... (Optuna setup and run - unchanged from previous full version)
        logger.info(f"[{zone_name_disp_mw}] Starting Optuna for XGBoost (base for VotingClassifier) parameters...")
        optuna_settings_mw = tuning_cfg_mw.get('optuna', {})
        n_trials_mw = optuna_settings_mw.get('n_trials', 100)
        sampler_opt_mw = optuna.samplers.TPESampler(seed=random_state_mw) if OPTUNA_AVAILABLE and hasattr(
            optuna.samplers, 'TPESampler') else None
        pruner_name_mw = optuna_settings_mw.get('pruner')
        pruner_opt_mw = None
        if OPTUNA_AVAILABLE:
            if pruner_name_mw == 'HyperbandPruner' and hasattr(optuna.pruners, 'HyperbandPruner'):
                pruner_opt_mw = optuna.pruners.HyperbandPruner()
            elif pruner_name_mw == 'MedianPruner' and hasattr(optuna.pruners, 'MedianPruner'):
                pruner_opt_mw = optuna.pruners.MedianPruner()
            elif pruner_name_mw == 'PercentilePruner' and hasattr(optuna.pruners, 'PercentilePruner'):
                pruner_opt_mw = optuna.pruners.PercentilePruner(
                    percentile=optuna_settings_mw.get('pruner_percentile', 25.0))
        study_mw = optuna.create_study(
            direction=optuna_settings_mw.get('direction', 'maximize'),
            sampler=sampler_opt_mw, pruner=pruner_opt_mw, study_name=f"{zone_key_mw}_xgb_voting_base_opt"
        )
        class_weights_config_for_optuna = training_params_mw.get('class_weights', {})
        objective_fn_mw = create_optuna_objective(
            X_train_scaled_mw_selected, y_train_arr_mw,
            optuna_settings_mw, best_xgb_params_mw,
            class_weights_config_for_optuna,
            smote_cfg_mw,
            num_classes_mw, class_specific_info_mw['partial_class_index'], random_state_mw, zone_name_disp_mw
        )
        optuna_callbacks_list = []
        log_freq_optuna = max(1, n_trials_mw // 10 if n_trials_mw >= 100 else (
            n_trials_mw // 5 if n_trials_mw >= 20 else 1))

        def optuna_logging_callback(study, trial):
            if trial.number > 0 and trial.number % log_freq_optuna == 0:
                current_best_val_cb_str = f"{study.best_value:.4f}" if study.best_trial and study.best_value is not None else "N/A"
                logger.info(
                    f"[{zone_name_disp_mw}] Optuna Trial {trial.number}/{n_trials_mw} - Current Best Score: {current_best_val_cb_str}")

        optuna_callbacks_list.append(optuna_logging_callback)
        study_mw.optimize(objective_fn_mw, n_trials=n_trials_mw, n_jobs=1, callbacks=optuna_callbacks_list)
        optuna_study_object_mw = study_mw
        if study_mw.best_trial:
            score_str_mw = f"{study_mw.best_value:.4f}" if study_mw.best_value is not None else "N/A"
            logger.info(
                f"[{zone_name_disp_mw}] Optuna Best trial: #{study_mw.best_trial.number}, Score: {score_str_mw}")
            logger.info(f"[{zone_name_disp_mw}] Optuna Best XGB base params: {study_mw.best_params}")
            best_xgb_params_mw.update(study_mw.best_params)
        else:
            logger.warning(f"[{zone_name_disp_mw}] Optuna did not find a best trial. Using default XGBoost params.")
    else:
        logger.info(
            f"[{zone_name_disp_mw}] Optuna disabled or not available. Using default XGBoost params for VotingClassifier base.")

    # Data for final model fitting (after FS, before SMOTE/Aug for main model training portion)
    X_for_final_model_and_calib_mw, y_for_final_model_and_calib_mw = X_train_scaled_mw_selected.copy(), y_train_arr_mw.copy()

    # Apply SMOTE (and potentially augmentation) to this data
    X_smoted_augmented_mw, y_smoted_augmented_mw = apply_smote_and_cleaning(
        X_for_final_model_and_calib_mw, y_for_final_model_and_calib_mw, smote_cfg_mw, random_state_mw, zone_name_disp_mw
    )
    if DATA_AUGMENTATION_CONFIG.get('enabled', False):
        logger.info(f"[{zone_name_disp_mw}] Applying data augmentation (post-SMOTE)...")
        try:
            X_smoted_augmented_mw, y_smoted_augmented_mw = apply_data_augmentation(
                X_smoted_augmented_mw, y_smoted_augmented_mw, DATA_AUGMENTATION_CONFIG, random_state_mw
            )
            unique_aug_mw, counts_aug_mw = np.unique(y_smoted_augmented_mw, return_counts=True)
            logger.info(
                f"[{zone_name_disp_mw}] Training distribution after augmentation: {dict(zip(unique_aug_mw, counts_aug_mw))}")
        except Exception as e_aug_mw:
            logger.warning(f"[{zone_name_disp_mw}] Data augmentation failed: {e_aug_mw}")

    # --- Data Split for Calibration (cv='prefit' for CalibratedClassifierCV) ---
    # We will fit the main VotingClassifier on X_train_main_model_mw, y_train_main_model_mw
    # and then calibrate it using X_calib_set_mw, y_calib_set_mw.
    X_train_main_model_mw = X_smoted_augmented_mw
    y_train_main_model_mw = y_smoted_augmented_mw
    X_calib_set_mw, y_calib_set_mw = None, None  # Initialize

    calib_method_mw = calib_cfg_mw.get('method', 'isotonic')
    calib_split_size_mw = calib_cfg_mw.get('calibration_split_size', 0.2)  # From zone_config or default
    min_samples_per_class_prefit_mw = calib_cfg_mw.get('min_samples_per_class_prefit', 10)  # From zone_config

    # Check if we have enough data to make a calibration split
    unique_y_smoted_aug, counts_y_smoted_aug = np.unique(y_smoted_augmented_mw, return_counts=True)
    can_stratify_calib_split = len(unique_y_smoted_aug) >= 2 and all(c >= 2 for c in counts_y_smoted_aug)

    # Crude check for enough samples for split
    # Need at least min_samples_per_class_prefit in calib set, and enough left for training.
    # And also enough samples for stratification if possible.
    min_total_for_calib_split = int(
        min_samples_per_class_prefit_mw / calib_split_size_mw) if calib_split_size_mw > 0 else float('inf')
    min_total_for_calib_split = max(min_total_for_calib_split,
                                    num_classes_mw * min_samples_per_class_prefit_mw)  # Ensure all classes can have min samples

    if len(y_smoted_augmented_mw) > min_total_for_calib_split and 0 < calib_split_size_mw < 1:
        try:
            X_train_main_model_mw, X_calib_set_mw, y_train_main_model_mw, y_calib_set_mw = train_test_split(
                X_smoted_augmented_mw, y_smoted_augmented_mw,
                test_size=calib_split_size_mw,
                random_state=random_state_mw,
                stratify=(y_smoted_augmented_mw if can_stratify_calib_split else None)
            )
            # Validate calibration set
            if X_calib_set_mw is not None and y_calib_set_mw is not None:
                unique_cal_labels_check, counts_cal_labels_check = np.unique(y_calib_set_mw, return_counts=True)
                if len(unique_cal_labels_check) < num_classes_mw or any(
                        c < min_samples_per_class_prefit_mw for c in counts_cal_labels_check if c > 0):
                    logger.warning(
                        f"[{zone_name_disp_mw}] Prefit calibration set invalid post-split (dist: {dict(zip(unique_cal_labels_check, counts_cal_labels_check))}, min: {min_samples_per_class_prefit_mw}). Calibration will be skipped or use internal CV if configured.")
                    X_train_main_model_mw, y_train_main_model_mw = X_smoted_augmented_mw, y_smoted_augmented_mw  # Revert to full data
                    X_calib_set_mw, y_calib_set_mw = None, None  # Nullify calib set
                else:
                    logger.info(
                        f"[{zone_name_disp_mw}] Created calibration set for 'prefit'. Main model train shape: {X_train_main_model_mw.shape}, Calib set shape: {X_calib_set_mw.shape}")
            else:  # Should not happen if split is successful
                X_train_main_model_mw, y_train_main_model_mw = X_smoted_augmented_mw, y_smoted_augmented_mw
                X_calib_set_mw, y_calib_set_mw = None, None
        except ValueError as ve_calib_split:  # e.g. not enough samples for stratification
            logger.warning(
                f"[{zone_name_disp_mw}] Could not create 'prefit' calibration split due to: {ve_calib_split}. Calibration might be skipped or use internal CV.")
            X_train_main_model_mw, y_train_main_model_mw = X_smoted_augmented_mw, y_smoted_augmented_mw
            X_calib_set_mw, y_calib_set_mw = None, None
    else:
        logger.info(
            f"[{zone_name_disp_mw}] Not enough samples ({len(y_smoted_augmented_mw)}) or invalid split size ({calib_split_size_mw}) for 'prefit' calibration split. Will attempt CV calibration or skip.")
        # X_train_main_model_mw and y_train_main_model_mw remain as the full (smoted, augmented) set
        # X_calib_set_mw, y_calib_set_mw remain None

    # --- Final Model Training (VotingClassifier on X_train_main_model_mw) ---
    logger.info(
        f"[{zone_name_disp_mw}] Training final Voting Ensemble model on {X_train_main_model_mw.shape[0]} samples...")
    # (Instantiate base models and VotingClassifier - same as before, but using X_train_main_model_mw, y_train_main_model_mw)
    adv_ensemble_opts_mw = ADVANCED_TRAINING_CONFIG.get('ensemble_options', {})
    rf_params_cfg_mw = adv_ensemble_opts_mw.get('random_forest_params', {}).copy()
    et_params_cfg_mw = adv_ensemble_opts_mw.get('extra_trees_params', {}).copy()
    voting_weights_cfg_mw = adv_ensemble_opts_mw.get('weights', {'xgb': 0.6, 'rf': 0.2, 'et': 0.2})
    voting_type_cfg_mw = adv_ensemble_opts_mw.get('voting_type', 'soft')

    xgb_final_params_voter = best_xgb_params_mw.copy()
    if 'early_stopping_rounds' in xgb_final_params_voter: del xgb_final_params_voter['early_stopping_rounds']
    xgb_final_params_voter['use_label_encoder'] = False
    if num_classes_mw <= 2 and xgb_final_params_voter.get('objective') == 'multi:softprob':
        xgb_final_params_voter['objective'] = 'binary:logistic';
        xgb_final_params_voter['eval_metric'] = 'logloss'
        if 'num_class' in xgb_final_params_voter: del xgb_final_params_voter['num_class']
    elif num_classes_mw > 2 and xgb_final_params_voter.get('objective') == 'binary:logistic':
        xgb_final_params_voter['objective'] = 'multi:softprob';
        xgb_final_params_voter['eval_metric'] = 'mlogloss'
        xgb_final_params_voter['num_class'] = num_classes_mw

    xgb_base_final_main = xgb.XGBClassifier(**xgb_final_params_voter)
    class_weights_map_main_model = calculate_class_weights_for_model(y_train_main_model_mw,
                                                                     training_params_mw.get('class_weights', {}))
    sample_weights_xgb_main_model = np.array(
        [class_weights_map_main_model.get(int(lbl), 1.0) for lbl in y_train_main_model_mw])

    logger.info(f"[{zone_name_disp_mw}] Fitting main XGBoost base model separately with sample_weight...")
    xgb_base_final_main.fit(X_train_main_model_mw, y_train_main_model_mw, sample_weight=sample_weights_xgb_main_model)

    rf_params_final_voter = rf_params_cfg_mw.copy();
    rf_params_final_voter['random_state'] = random_state_mw
    rf_params_final_voter['class_weight'] = class_weights_map_main_model if class_weights_map_main_model else 'balanced'
    rf_base_final_main = RandomForestClassifier(**rf_params_final_voter)

    et_params_final_voter = et_params_cfg_mw.copy();
    et_params_final_voter['random_state'] = random_state_mw
    et_params_final_voter['class_weight'] = class_weights_map_main_model if class_weights_map_main_model else 'balanced'
    et_base_final_main = ExtraTreesClassifier(**et_params_final_voter)

    estimators_list_main_model = [('xgb', xgb_base_final_main), ('rf', rf_base_final_main), ('et', et_base_final_main)]

    voting_model_uncalibrated_mw = VotingClassifier(
        estimators=estimators_list_main_model, voting=voting_type_cfg_mw,
        weights=[voting_weights_cfg_mw.get('xgb', 0.6), voting_weights_cfg_mw.get('rf', 0.2),
                 voting_weights_cfg_mw.get('et', 0.2)],
        n_jobs=n_jobs_workflow
    )
    logger.info(f"[{zone_name_disp_mw}] Fitting main VotingClassifier (RF and ET bases)...")
    voting_model_uncalibrated_mw.fit(X_train_main_model_mw, y_train_main_model_mw)  # XGB is pre-fit

    # --- Calibration ---
    final_model_mw = voting_model_uncalibrated_mw  # Default to uncalibrated

    # Option 1: Calibrate using the dedicated calibration set if available (cv='prefit')
    if X_calib_set_mw is not None and y_calib_set_mw is not None and len(X_calib_set_mw) > 0:
        logger.info(
            f"[{zone_name_disp_mw}] Attempting 'prefit' calibration using dedicated set ({X_calib_set_mw.shape[0]} samples). Method: '{calib_method_mw}'...")
        try:
            # Calibrate the already fitted voting_model_uncalibrated_mw
            calibrated_model_prefit_mw = CalibratedClassifierCV(
                estimator=deepcopy(voting_model_uncalibrated_mw),
                # Pass a copy to avoid modifying original if fit fails
                method=calib_method_mw,
                cv='prefit',
                n_jobs=n_jobs_workflow  # n_jobs for potential internal operations if method involves it
            )
            calibrated_model_prefit_mw.fit(X_calib_set_mw, y_calib_set_mw)  # Fit calibrators on X_calib_set_mw
            final_model_mw = calibrated_model_prefit_mw
            logger.info(f"[{zone_name_disp_mw}] Model calibrated with 'prefit' on dedicated calibration set.")
        except Exception as e_calib_prefit_mw:
            logger.warning(
                f"[{zone_name_disp_mw}] 'Prefit' calibration failed: {e_calib_prefit_mw}. Using uncalibrated model.",
                exc_info=True)
            # final_model_mw remains voting_model_uncalibrated_mw

    # Option 2: Fallback to CV-based calibration if prefit was not possible or failed
    # This uses the full X_smoted_augmented_mw, y_smoted_augmented_mw for calibration CV
    elif isinstance(calib_cfg_mw.get('cv'), int) and calib_cfg_mw.get('cv') > 1:
        calib_cv_folds_mw = calib_cfg_mw.get('cv')
        logger.info(
            f"[{zone_name_disp_mw}] Attempting CV-based calibration. Method: '{calib_method_mw}', CV Folds: {calib_cv_folds_mw}...")
        try:
            unique_y_cv_cal_mw, counts_y_cv_cal_mw = np.unique(X_smoted_augmented_mw,
                                                               return_counts=True)  # Use full data for CV calib
            if len(unique_y_cv_cal_mw) >= (2 if num_classes_mw > 1 else 1) and \
                    all(c >= calib_cv_folds_mw for c in counts_y_cv_cal_mw):

                xgb_template_calib_cv = xgb.XGBClassifier(**xgb_final_params_voter)
                rf_template_calib_cv = RandomForestClassifier(**rf_params_final_voter)
                et_template_calib_cv = ExtraTreesClassifier(**et_params_final_voter)

                estimators_template_calib_cv = [('xgb', xgb_template_calib_cv), ('rf', rf_template_calib_cv),
                                                ('et', et_template_calib_cv)]
                voting_model_template_for_calib_cv = VotingClassifier(
                    estimators=estimators_template_calib_cv, voting=voting_type_cfg_mw,
                    weights=[voting_weights_cfg_mw.get('xgb', 0.6), voting_weights_cfg_mw.get('rf', 0.2),
                             voting_weights_cfg_mw.get('et', 0.2)],
                    n_jobs=n_jobs_workflow
                )
                calib_cv_obj_mw = StratifiedKFold(n_splits=calib_cv_folds_mw, shuffle=True,
                                                  random_state=random_state_mw)

                calibrated_model_cv_mw = CalibratedClassifierCV(
                    estimator=voting_model_template_for_calib_cv,
                    method=calib_method_mw, cv=calib_cv_obj_mw, n_jobs=n_jobs_workflow
                )

                fit_params_for_calib_cv_internal = {}
                # This is where sample_weight for XGBoost needs to be specified for CalibratedClassifierCV's internal fits
                # Sample weights should be based on y_smoted_augmented_mw
                class_weights_map_for_calib_cv = calculate_class_weights_for_model(y_smoted_augmented_mw,
                                                                                   training_params_mw.get(
                                                                                       'class_weights', {}))
                sample_weights_xgb_for_calib_cv = np.array(
                    [class_weights_map_for_calib_cv.get(int(lbl), 1.0) for lbl in y_smoted_augmented_mw])

                if not np.all(sample_weights_xgb_for_calib_cv == 1.0):
                    # This attempts to pass sample_weight to the xgb component of the VotingClassifier template
                    # This specific routing ('estimator__xgb__sample_weight') might still be an issue if CalibratedClassifierCV doesn't propagate it deeply enough.
                    fit_params_for_calib_cv_internal['estimator__xgb__sample_weight'] = sample_weights_xgb_for_calib_cv
                    logger.info(
                        f"[{zone_name_disp_mw}] Passing sample_weight for XGB to CalibratedClassifierCV for internal CV fits.")

                calibrated_model_cv_mw.fit(X_smoted_augmented_mw, y_smoted_augmented_mw,
                                           **fit_params_for_calib_cv_internal)
                final_model_mw = calibrated_model_cv_mw
                logger.info(f"[{zone_name_disp_mw}] VotingClassifier calibrated with internal cv={calib_cv_folds_mw}.")
            else:
                logger.warning(
                    f"[{zone_name_disp_mw}] Internal CV calibration skipped: not enough samples for {calib_cv_folds_mw} folds. Using uncalibrated model.")
        except Exception as e_calib_cv_mw:
            logger.warning(
                f"[{zone_name_disp_mw}] Internal CV calibration failed: {e_calib_cv_mw}. Using uncalibrated model.",
                exc_info=True)
    else:
        logger.info(f"[{zone_name_disp_mw}] No suitable calibration data or configuration. Using uncalibrated model.")

    # --- Threshold Optimization (unchanged from previous full version) ---
    optimal_thresholds_map_mw = None
    threshold_opt_cfg_mw = training_params_mw.get('threshold_optimization', {})
    if threshold_opt_cfg_mw.get('enabled', False):
        logger.info(f"[{zone_name_disp_mw}] Optimizing prediction thresholds...")
        # Use the full data that the main model was trained on (or the portion before calib split if prefit was used)
        # For simplicity, using X_smoted_augmented_mw, y_smoted_augmented_mw
        # A more robust approach uses a hold-out set not seen by `final_model_mw` at all.
        data_for_thresh_opt_X = X_smoted_augmented_mw
        data_for_thresh_opt_y = y_smoted_augmented_mw
        if X_calib_set_mw is not None and y_calib_set_mw is not None and len(X_calib_set_mw) > 0:
            # If a calib set was made, that's a good candidate for threshold opt as final_model_mw (uncalibrated part) hasn't seen it
            logger.info(f"[{zone_name_disp_mw}] Using dedicated calibration set for threshold optimization.")
            data_for_thresh_opt_X = X_calib_set_mw
            data_for_thresh_opt_y = y_calib_set_mw

        if len(data_for_thresh_opt_X) > 100:  # Arbitrary min samples for threshold opt
            try:
                optimal_thresholds_map_mw = optimize_class_thresholds(
                    final_model_mw, data_for_thresh_opt_X, data_for_thresh_opt_y,
                    class_names_map_global_mw, zone_config_mw
                )
            except Exception as e_thresh_opt_setup_mw:
                logger.warning(
                    f"[{zone_name_disp_mw}] Threshold optimization failed: {e_thresh_opt_setup_mw}. Skipping.",
                    exc_info=True)
        else:
            logger.warning(
                f"[{zone_name_disp_mw}] Not enough data ({len(data_for_thresh_opt_X)} samples) for threshold optimization. Skipping.")

    # --- Prepare Training Analysis Data (unchanged from previous full version) ---
    training_analysis_df_mw = pd.DataFrame()
    try:
        X_train_scaled_for_analysis_full = scaler_mw.transform(X_train_df_mw)  # Original X_train (full features) scaled
        df_temp_for_selection = pd.DataFrame(X_train_scaled_for_analysis_full, columns=feature_names_mw_initial)
        X_train_scaled_for_analysis_selected = df_temp_for_selection[
            feature_names_mw].values  # Select features used by model

        y_pred_train_analysis_mw = final_model_mw.predict(X_train_scaled_for_analysis_selected)
        y_proba_train_analysis_mw = final_model_mw.predict_proba(X_train_scaled_for_analysis_selected)
        analysis_data_mw = {'Expert_Label': y_train_arr_mw, 'Predicted_Label': y_pred_train_analysis_mw}
        for i_cls_mw_key, i_cls_mw_name in class_names_map_global_mw.items():
            i_cls_mw = int(i_cls_mw_key);
            c_name_mw = i_cls_mw_name.replace(' ', '_')
            if i_cls_mw < y_proba_train_analysis_mw.shape[1]:
                analysis_data_mw[f'Prob_{c_name_mw}'] = y_proba_train_analysis_mw[:, i_cls_mw]
        analysis_index_mw = X_train_df_mw.index
        training_analysis_df_mw = pd.DataFrame(analysis_data_mw, index=analysis_index_mw)
        training_analysis_df_mw['Entropy'] = [calculate_entropy(p_row) for p_row in y_proba_train_analysis_mw]
        training_analysis_df_mw['Margin'] = [calculate_margin(p_row) for p_row in y_proba_train_analysis_mw]
        training_analysis_df_mw['Prob_True_Label'] = [
            y_proba_train_analysis_mw[idx, int(lbl)] if 0 <= int(lbl) < y_proba_train_analysis_mw.shape[1] else 0.0
            for idx, lbl in enumerate(y_train_arr_mw)
        ]
        training_analysis_df_mw['Is_Correct'] = (
                    training_analysis_df_mw['Expert_Label'] == training_analysis_df_mw['Predicted_Label'])
        logger.info(
            f"[{zone_name_disp_mw}] Accuracy on original training set (for analysis, using selected features): {training_analysis_df_mw['Is_Correct'].mean():.4f}")
    except Exception as e_train_an_mw:
        logger.error(f"[{zone_name_disp_mw}] Training set uncertainty analysis failed: {e_train_an_mw}", exc_info=True)

    feature_importance_df_mw = get_common_feature_importance_df(final_model_mw, feature_names_mw, zone_name_disp_mw)

    if PERFORMANCE_CONFIG.get('memory_optimization', {}).get('enable_gc', True):
        gc.collect()

    return final_model_mw, scaler_mw, feature_importance_df_mw, training_analysis_df_mw, optimal_thresholds_map_mw, optuna_study_object_mw, feature_names_mw