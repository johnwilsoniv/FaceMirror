# synkinesis_training.py (v1.10.4 - Output Error Details as CSV)

import pandas as pd
import numpy as np
import logging
import os
import joblib
import importlib
import json
import sklearn.metrics  # Keep this for completeness

from sklearn.model_selection import train_test_split, StratifiedKFold, KFold
from sklearn.metrics import (
    classification_report, confusion_matrix, f1_score,
    balanced_accuracy_score, cohen_kappa_score, roc_auc_score,
    precision_recall_curve, auc, average_precision_score,
    brier_score_loss, log_loss, precision_score, recall_score
)
from sklearn.preprocessing import StandardScaler
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import VotingClassifier, RandomForestClassifier
import xgboost as xgb
from imblearn.over_sampling import SMOTE, BorderlineSMOTE, ADASYN
from imblearn.combine import SMOTEENN

# Import Optuna
try:
    import optuna
    from optuna.samplers import TPESampler
    from optuna.pruners import HyperbandPruner, MedianPruner, PercentilePruner

    OPTUNA_AVAILABLE = True
    optuna.logging.set_verbosity(optuna.logging.WARNING)
except ImportError:
    OPTUNA_AVAILABLE = False
    TPESampler, HyperbandPruner, MedianPruner, PercentilePruner = None, None, None, None

# Import central config for synkinesis
try:
    from synkinesis_config import (
        SYNKINESIS_CONFIG, LOGGING_CONFIG, INPUT_FILES, CLASS_NAMES,
        ADVANCED_TRAINING_CONFIG, REVIEW_CONFIG, DATA_AUGMENTATION_CONFIG,
        get_synkinesis_config, get_all_synkinesis_types,
        LOG_DIR, ANALYSIS_DIR
    )

    CONFIG_LOADED = True
except ImportError as e:
    print(f"CRITICAL: Failed to import from synkinesis_config.py - {e}")
    CONFIG_LOADED = False
    SYNKINESIS_CONFIG = {}
    LOGGING_CONFIG = {'level': 'INFO', 'format': '%(asctime)s - %(levelname)s - %(name)s - %(message)s'}
    INPUT_FILES = {}
    CLASS_NAMES = {0: 'None', 1: 'Synkinesis'}
    ADVANCED_TRAINING_CONFIG = {}
    REVIEW_CONFIG = {}
    DATA_AUGMENTATION_CONFIG = {}
    LOG_DIR = 'logs'
    ANALYSIS_DIR = 'analysis_results'

try:
    from paralysis_utils import (
        generate_review_candidates,
        calculate_entropy, calculate_margin,
    )

    UTILS_LOADED = True
except ImportError as e:
    print(f"WARNING: Failed to import from paralysis_utils.py - {e}. Some review features might be unavailable.")
    UTILS_LOADED = False


    def generate_review_candidates(*args, **kwargs):
        return pd.DataFrame()


    def calculate_entropy(*args, **kwargs):
        return 0.0


    def calculate_margin(*args, **kwargs):
        return 1.0

logger = logging.getLogger(__name__)


def _setup_logging(log_file, level_str, log_format_str):
    log_level = getattr(logging, level_str.upper(), logging.INFO)
    log_dir_path = os.path.dirname(log_file)
    if log_dir_path:
        os.makedirs(log_dir_path, exist_ok=True)

    root_logger = logging.getLogger()
    if root_logger.hasHandlers():
        root_logger.handlers.clear()

    logging.basicConfig(level=log_level, format=log_format_str,
                        handlers=[logging.StreamHandler(), logging.FileHandler(log_file, mode='w')],
                        force=True)


def _calculate_class_weights(y_true, strategy='balanced'):
    unique_classes, counts = np.unique(y_true, return_counts=True)
    if not unique_classes.size: return {}

    if strategy == 'balanced':
        total_samples = len(y_true)
        weights = {cls: total_samples / (len(unique_classes) * count) if count > 0 else 1.0
                   for cls, count in zip(unique_classes, counts)}
    elif strategy == 'balanced_subsample':
        max_count = max(counts) if counts.size > 0 else 1
        weights = {cls: max_count / count if count > 0 else 1.0
                   for cls, count in zip(unique_classes, counts)}
    elif isinstance(strategy, dict):
        weights = strategy
    else:
        weights = {cls: 1.0 for cls in unique_classes}
    return weights


def _get_sampling_strategy(y_fold_train, configured_strategy, min_samples_per_class_recalc=30):
    if isinstance(configured_strategy, dict):
        resolved_strategy = {}
        unique_y, counts_y = np.unique(y_fold_train, return_counts=True)
        y_dist = dict(zip(unique_y, counts_y))

        for cls_label, target_count in configured_strategy.items():
            if cls_label not in y_dist:
                continue
            if isinstance(target_count, (int, float)):
                resolved_strategy[cls_label] = max(int(target_count), y_dist[cls_label],
                                                   1)
            else:
                resolved_strategy[cls_label] = y_dist[cls_label]
        return resolved_strategy if resolved_strategy else 'auto'

    if configured_strategy == 'auto':
        return 'auto'
    return 'auto'


def _get_optuna_suggestion(trial, param_name, param_config_list):
    param_type = param_config_list[0]
    options = param_config_list[3] if len(param_config_list) > 3 else {}
    if param_type == 'float':
        return trial.suggest_float(param_name, param_config_list[1], param_config_list[2], **options)
    elif param_type == 'int':
        return trial.suggest_int(param_name, param_config_list[1], param_config_list[2], **options)
    elif param_type == 'categorical':
        return trial.suggest_categorical(param_name, param_config_list[1])
    else:
        raise ValueError(f"Unsupported Optuna parameter type: {param_type}")


def _create_optuna_objective_synkinesis(X_train_scaled_np, y_train_np,
                                        smote_config_obj, optuna_config_obj,
                                        model_params_base, random_state_obj,
                                        synk_type_name_obj):
    if not OPTUNA_AVAILABLE:
        raise ImportError("Optuna is not available for hyperparameter tuning.")

    param_distributions = optuna_config_obj.get('param_distributions', {})
    cv_folds_optuna = optuna_config_obj.get('cv_folds', 3)
    scoring_metric_optuna = optuna_config_obj.get('scoring', 'average_precision').lower()
    optuna_early_stop_rounds = optuna_config_obj.get('optuna_early_stopping_rounds')

    smote_enabled_obj = smote_config_obj.get('enabled', False)
    smote_per_fold_obj = smote_config_obj.get('apply_per_fold_in_tuning', False) and smote_enabled_obj
    k_neighbors_smote_obj = smote_config_obj.get('k_neighbors', 5)
    sampling_strategy_smote_obj = smote_config_obj.get('sampling_strategy', 'auto')
    min_samples_for_smote_recalc_obj = smote_config_obj.get('min_samples_per_class', 5)
    smote_variant_obj = smote_config_obj.get('variant', 'regular').lower()

    def objective(trial):
        xgb_params_trial = {
            'objective': 'binary:logistic',
            'eval_metric': model_params_base.get('eval_metric', 'aucpr'),
            'random_state': random_state_obj,
        }

        for param_name, param_config_list in param_distributions.items():
            xgb_params_trial[param_name] = _get_optuna_suggestion(trial, param_name, param_config_list)

        if optuna_early_stop_rounds:
            xgb_params_trial['early_stopping_rounds'] = optuna_early_stop_rounds
        elif 'early_stopping_rounds' in model_params_base:
            xgb_params_trial['early_stopping_rounds'] = model_params_base['early_stopping_rounds']

        fold_scores = []
        unique_labels_train, counts_train = np.unique(y_train_np, return_counts=True)
        min_samples_for_stratify = cv_folds_optuna

        if len(unique_labels_train) < 2 or any(c < min_samples_for_stratify for c in counts_train):
            cv_obj = KFold(n_splits=cv_folds_optuna, shuffle=True, random_state=random_state_obj)
        else:
            cv_obj = StratifiedKFold(n_splits=cv_folds_optuna, shuffle=True, random_state=random_state_obj)

        for fold_idx, (train_idx, val_idx) in enumerate(cv_obj.split(X_train_scaled_np, y_train_np)):
            X_fold_train, y_fold_train = X_train_scaled_np[train_idx], y_train_np[train_idx]
            X_fold_val, y_fold_val = X_train_scaled_np[val_idx], y_train_np[val_idx]

            X_to_fit, y_to_fit = X_fold_train, y_fold_train
            fit_params_for_xgb = {'verbose': False}

            if smote_per_fold_obj:
                try:
                    unique_fold, counts_fold = np.unique(y_fold_train, return_counts=True)
                    min_class_count_fold = min(counts_fold) if counts_fold.size > 0 else 0
                    actual_k_smote = min(k_neighbors_smote_obj,
                                         max(1, min_class_count_fold - 1)) if min_class_count_fold > 1 else 0

                    if actual_k_smote >= 1 and len(unique_fold) > 1:
                        current_fold_smote_strategy = _get_sampling_strategy(
                            y_fold_train, sampling_strategy_smote_obj, min_samples_for_smote_recalc_obj
                        )
                        if smote_variant_obj == 'borderline':
                            smote_instance = BorderlineSMOTE(k_neighbors=actual_k_smote, random_state=random_state_obj,
                                                             sampling_strategy=current_fold_smote_strategy)
                        elif smote_variant_obj == 'adasyn':
                            smote_instance = ADASYN(n_neighbors=actual_k_smote, random_state=random_state_obj,
                                                    sampling_strategy=current_fold_smote_strategy)
                        elif smote_variant_obj == 'smoteenn':
                            smote_base = SMOTE(k_neighbors=actual_k_smote, random_state=random_state_obj,
                                               sampling_strategy=current_fold_smote_strategy)
                            smote_instance = SMOTEENN(smote=smote_base, random_state=random_state_obj)
                        else:
                            smote_instance = SMOTE(k_neighbors=actual_k_smote, random_state=random_state_obj,
                                                   sampling_strategy=current_fold_smote_strategy)
                        X_to_fit, y_to_fit = smote_instance.fit_resample(X_fold_train, y_fold_train)
                except Exception as e_smote:
                    pass

            model_fold = xgb.XGBClassifier(**xgb_params_trial)
            if xgb_params_trial.get('early_stopping_rounds') and len(
                    np.unique(y_fold_val)) > 1:
                fit_params_for_xgb['eval_set'] = [(X_fold_val, y_fold_val)]

            try:
                model_fold.fit(X_to_fit, y_to_fit, **fit_params_for_xgb)
                current_score_val = 0.0
                y_proba_fold_val = model_fold.predict_proba(X_fold_val)  # Get probabilities once

                if scoring_metric_optuna == 'average_precision':
                    if len(np.unique(y_fold_val)) < 2:
                        current_score_val = 0.0
                    else:
                        current_score_val = average_precision_score(y_fold_val, y_proba_fold_val[:, 1], pos_label=1)
                elif scoring_metric_optuna == 'f1':
                    y_pred_fold_val = (y_proba_fold_val[:, 1] >= 0.5).astype(int)  # Assume 0.5 for Optuna F1
                    current_score_val = f1_score(y_fold_val, y_pred_fold_val, pos_label=1, zero_division=0)
                elif scoring_metric_optuna == 'roc_auc':
                    if len(np.unique(y_fold_val)) < 2:
                        current_score_val = 0.0
                    else:
                        current_score_val = roc_auc_score(y_fold_val, y_proba_fold_val[:, 1])
                else:  # Default to Average Precision
                    if len(np.unique(y_fold_val)) < 2:
                        current_score_val = 0.0
                    else:
                        current_score_val = average_precision_score(y_fold_val, y_proba_fold_val[:, 1], pos_label=1)

                fold_scores.append(current_score_val)
                trial.report(current_score_val, fold_idx)
                if trial.should_prune():
                    raise optuna.TrialPruned()
            except optuna.TrialPruned:
                raise
            except Exception as e_fit:
                fold_scores.append(0.0)

        avg_score_trial = np.mean(fold_scores) if fold_scores else 0.0
        return avg_score_trial

    return objective


def train_synkinesis_model(synk_type_key, X_train_df, y_train_series, X_test_df, y_test_series,
                           feature_names_list, type_specific_config,
                           eval_metrics_list_arg, metadata_test_df_for_errors):  # Added metadata_test_df_for_errors
    synk_name = type_specific_config.get('name', synk_type_key.capitalize())
    training_params_cfg = type_specific_config.get('training', {})
    smote_cfg_train = training_params_cfg.get('smote', {})
    tuning_cfg_train = training_params_cfg.get('hyperparameter_tuning', {})
    base_model_params_train = training_params_cfg.get('model_params', {}).copy()
    random_state_train = training_params_cfg.get('random_state', 42)

    if 'random_state' not in base_model_params_train: base_model_params_train['random_state'] = random_state_train
    if 'early_stopping_rounds' not in base_model_params_train: base_model_params_train[
        'early_stopping_rounds'] = training_params_cfg.get('early_stopping_rounds', 15)
    if 'eval_metric' not in base_model_params_train: base_model_params_train[
        'eval_metric'] = 'aucpr'

    scaler = StandardScaler()
    X_train_scaled_arr = scaler.fit_transform(X_train_df)
    X_test_scaled_arr = scaler.transform(X_test_df)
    y_train_arr = y_train_series.values if isinstance(y_train_series, pd.Series) else np.array(y_train_series)
    y_test_arr = y_test_series.values if isinstance(y_test_series, pd.Series) else np.array(y_test_series)

    best_params_for_final_model = base_model_params_train.copy()

    if tuning_cfg_train.get('enabled', False) and tuning_cfg_train.get('method') == 'optuna':
        if not OPTUNA_AVAILABLE:
            logger.error(f"[{synk_name}] Optuna specified but not available. Using default parameters.")
        else:
            optuna_settings_train = tuning_cfg_train.get('optuna', {})
            n_trials_optuna = optuna_settings_train.get('n_trials', 50)
            sampler_type_optuna = optuna_settings_train.get('sampler', 'TPESampler')
            pruner_type_optuna = optuna_settings_train.get('pruner', 'MedianPruner')
            sampler_optuna = TPESampler(
                seed=random_state_train) if sampler_type_optuna == 'TPESampler' and TPESampler else None
            pruner_map_optuna = {'HyperbandPruner': HyperbandPruner() if HyperbandPruner else None,
                                 'MedianPruner': MedianPruner() if MedianPruner else None,
                                 'PercentilePruner': PercentilePruner(percentile=25.0) if PercentilePruner else None}
            pruner_optuna = pruner_map_optuna.get(pruner_type_optuna)
            study = optuna.create_study(direction=optuna_settings_train.get('direction', 'maximize'),
                                        sampler=sampler_optuna, pruner=pruner_optuna,
                                        study_name=f"{synk_type_key}_optuna_study")
            objective_function = _create_optuna_objective_synkinesis(X_train_scaled_arr, y_train_arr, smote_cfg_train,
                                                                     optuna_settings_train, base_model_params_train,
                                                                     random_state_train, synk_name)
            study.optimize(objective_function, n_trials=n_trials_optuna, callbacks=None, n_jobs=1)
            best_params_for_final_model.update(study.best_params)
            model_save_dir_opt = os.path.dirname(
                type_specific_config.get('filenames', {}).get('model', 'models/synkinesis/default_model.pkl'))
            os.makedirs(model_save_dir_opt, exist_ok=True)
            study_path_opt = os.path.join(model_save_dir_opt, f"{synk_type_key}_optuna_study.pkl")
            try:
                joblib.dump(study, study_path_opt)
            except Exception as e_save_study:
                logger.error(f"[{synk_name}] Failed to save Optuna study: {e_save_study}")

    if 'early_stopping_rounds' not in best_params_for_final_model: best_params_for_final_model[
        'early_stopping_rounds'] = training_params_cfg.get('early_stopping_rounds', 15)
    if 'eval_metric' not in best_params_for_final_model: best_params_for_final_model[
        'eval_metric'] = base_model_params_train.get('eval_metric', 'aucpr')

    X_train_for_final_fit, y_train_for_final_fit = X_train_scaled_arr.copy(), y_train_arr.copy()
    apply_global_smote = smote_cfg_train.get('enabled', False)
    if smote_cfg_train.get('apply_per_fold_in_tuning', False) and tuning_cfg_train.get('enabled', False):
        apply_global_smote = smote_cfg_train.get('apply_to_full_train_if_not_per_fold', True)

    if apply_global_smote:
        try:
            smote_variant_final = smote_cfg_train.get('variant', 'regular').lower()
            k_neighbors_final_cfg = smote_cfg_train.get('k_neighbors', 5)
            sampling_strategy_final = smote_cfg_train.get('sampling_strategy', 'auto')
            min_samples_final_recalc = smote_cfg_train.get('min_samples_per_class', 5)
            unique_final_y, counts_final_y = np.unique(y_train_arr, return_counts=True)
            min_class_count_final = min(counts_final_y) if counts_final_y.size > 0 else 0
            actual_k_final = min(k_neighbors_final_cfg,
                                 max(1, min_class_count_final - 1)) if min_class_count_final > 1 else 0
            if actual_k_final >= 1 and len(unique_final_y) > 1:
                final_smote_strategy_resolved = _get_sampling_strategy(y_train_arr, sampling_strategy_final,
                                                                       min_samples_final_recalc)
                if smote_variant_final == 'borderline':
                    smote_final_obj = BorderlineSMOTE(k_neighbors=actual_k_final, random_state=random_state_train,
                                                      sampling_strategy=final_smote_strategy_resolved)
                elif smote_variant_final == 'adasyn':
                    smote_final_obj = ADASYN(n_neighbors=actual_k_final, random_state=random_state_train,
                                             sampling_strategy=final_smote_strategy_resolved)
                elif smote_variant_final == 'smoteenn':
                    smote_base_final = SMOTE(k_neighbors=actual_k_final, random_state=random_state_train,
                                             sampling_strategy=final_smote_strategy_resolved)
                    smote_final_obj = SMOTEENN(smote=smote_base_final, random_state=random_state_train)
                else:
                    smote_final_obj = SMOTE(k_neighbors=actual_k_final, random_state=random_state_train,
                                            sampling_strategy=final_smote_strategy_resolved)
                X_train_for_final_fit, y_train_for_final_fit = smote_final_obj.fit_resample(X_train_scaled_arr,
                                                                                            y_train_arr)
        except Exception as e_smote_final:
            logger.error(
                f"[{synk_name}] Global SMOTE failed: {e_smote_final}. Using original training data for final model.",
                exc_info=True)
            X_train_for_final_fit, y_train_for_final_fit = X_train_scaled_arr.copy(), y_train_arr.copy()

    if 'scale_pos_weight' not in best_params_for_final_model:
        counts_final_train = np.bincount(y_train_for_final_fit.astype(int))
        if len(counts_final_train) > 1 and counts_final_train[1] > 0:
            auto_spw = counts_final_train[0] / counts_final_train[1]
            best_params_for_final_model['scale_pos_weight'] = auto_spw
        elif len(counts_final_train) <= 1:
            best_params_for_final_model['scale_pos_weight'] = 1.0

    X_calib_set, y_calib_set = None, None
    X_base_train_for_calib, y_base_train_for_calib = X_train_for_final_fit, y_train_for_final_fit
    adv_calib_cfg = ADVANCED_TRAINING_CONFIG.get('calibration', {})
    type_calib_cfg = training_params_cfg.get('calibration', {})
    calib_method_final = type_calib_cfg.get('method', adv_calib_cfg.get('method', 'isotonic'))
    calib_cv_setting_final = type_calib_cfg.get('cv', adv_calib_cfg.get('cv', 'prefit'))
    calib_split_size_final = type_calib_cfg.get('calibration_split_size',
                                                adv_calib_cfg.get('calibration_split_size', 0.2))
    min_samples_per_class_for_calib_prefit = type_specific_config.get('training', {}).get('calibration', {}).get(
        'min_samples_per_class_prefit', adv_calib_cfg.get('min_samples_per_class_prefit', 10))

    if calib_cv_setting_final == 'prefit' and \
            len(np.unique(y_train_for_final_fit)) > 1 and \
            y_train_for_final_fit.shape[0] * calib_split_size_final >= min_samples_per_class_for_calib_prefit * len(
        np.unique(y_train_for_final_fit)) and \
            y_train_for_final_fit.shape[0] * (
            1 - calib_split_size_final) >= min_samples_per_class_for_calib_prefit * len(
        np.unique(y_train_for_final_fit)):
        try:
            unique_y_cal_split, counts_y_cal_split = np.unique(y_train_for_final_fit, return_counts=True)
            can_stratify_calib = len(unique_y_cal_split) > 1 and \
                                 all(c * calib_split_size_final >= 1 for c in counts_y_cal_split if c > 0) and \
                                 all(c * (1 - calib_split_size_final) >= 1 for c in counts_y_cal_split if c > 0)
            stratify_calib_arr = y_train_for_final_fit if can_stratify_calib else None
            X_base_train_for_calib, X_calib_set, y_base_train_for_calib, y_calib_set = train_test_split(
                X_train_for_final_fit, y_train_for_final_fit, test_size=calib_split_size_final,
                random_state=random_state_train, stratify=stratify_calib_arr
            )
            if X_calib_set is not None and y_calib_set is not None:
                unique_calib_final, counts_calib_final = np.unique(y_calib_set, return_counts=True)
                present_classes_meet_min_samples = True
                if counts_calib_final.size > 0:
                    for cls_idx, count in enumerate(counts_calib_final):
                        if count < min_samples_per_class_for_calib_prefit: present_classes_meet_min_samples = False; break
                if len(unique_calib_final) < 2 or not present_classes_meet_min_samples:
                    X_calib_set, y_calib_set = None, None
                    X_base_train_for_calib, y_base_train_for_calib = X_train_for_final_fit, y_train_for_final_fit
            else:
                X_calib_set, y_calib_set = None, None;
                X_base_train_for_calib, y_base_train_for_calib = X_train_for_final_fit, y_train_for_final_fit
        except ValueError as ve_calib_split:
            X_calib_set, y_calib_set = None, None;
            X_base_train_for_calib, y_base_train_for_calib = X_train_for_final_fit, y_train_for_final_fit
    else:
        X_base_train_for_calib, y_base_train_for_calib = X_train_for_final_fit, y_train_for_final_fit
        X_calib_set, y_calib_set = None, None

    fit_params_xgb_final_direct_fit = {'verbose': False}
    current_base_model_params_for_direct_fit = best_params_for_final_model.copy()
    if current_base_model_params_for_direct_fit.get('early_stopping_rounds'):
        eval_set_x_base, eval_set_y_base = X_test_scaled_arr, y_test_arr
        if calib_cv_setting_final == 'prefit' and X_calib_set is not None and y_calib_set is not None and len(
                np.unique(y_calib_set)) > 1:
            eval_set_x_base, eval_set_y_base = X_calib_set, y_calib_set
        elif len(np.unique(y_test_arr)) > 1:
            pass
        else:
            if 'early_stopping_rounds' in current_base_model_params_for_direct_fit: del \
                current_base_model_params_for_direct_fit['early_stopping_rounds']
        if 'early_stopping_rounds' in current_base_model_params_for_direct_fit: fit_params_xgb_final_direct_fit[
            'eval_set'] = [(eval_set_x_base, eval_set_y_base)]

    xgb_final_model_base = xgb.XGBClassifier(**current_base_model_params_for_direct_fit)
    xgb_final_model_base.fit(X_base_train_for_calib, y_base_train_for_calib, **fit_params_xgb_final_direct_fit)
    model_to_calibrate = xgb_final_model_base

    if training_params_cfg.get('use_ensemble', False):
        ensemble_opts_cfg = ADVANCED_TRAINING_CONFIG.get('ensemble_options', {})
        rf_params_ensemble = ensemble_opts_cfg.get('random_forest_params', {})
        if 'class_weight' not in rf_params_ensemble or rf_params_ensemble['class_weight'] == 'balanced_subsample':
            rf_params_ensemble['class_weight'] = _calculate_class_weights(y_base_train_for_calib, 'balanced_subsample')
        xgb_ensemble_template_params = best_params_for_final_model.copy()
        if 'early_stopping_rounds' in xgb_ensemble_template_params: del xgb_ensemble_template_params[
            'early_stopping_rounds']
        if 'eval_metric' in xgb_ensemble_template_params: del xgb_ensemble_template_params['eval_metric']
        xgb_for_ensemble_template = xgb.XGBClassifier(**xgb_ensemble_template_params)
        rf_model_ensemble_template = RandomForestClassifier(random_state=random_state_train, **rf_params_ensemble,
                                                            n_jobs=-1)
        ensemble_weights_cfg = ensemble_opts_cfg.get('weights', {'xgboost': 0.7, 'random_forest': 0.3})
        voting_clf = VotingClassifier(
            estimators=[('xgb', xgb_for_ensemble_template), ('rf', rf_model_ensemble_template)],
            voting=ensemble_opts_cfg.get('voting_type', 'soft'),
            weights=[ensemble_weights_cfg.get('xgboost', 0.7), ensemble_weights_cfg.get('random_forest', 0.3)])
        voting_clf.fit(X_base_train_for_calib, y_base_train_for_calib)
        model_to_calibrate = voting_clf

    final_model_output = model_to_calibrate
    was_calibrated_explicitly = False
    try:
        if calib_method_final in ['isotonic', 'sigmoid']:
            if calib_cv_setting_final == 'prefit':
                if X_calib_set is not None and y_calib_set is not None and len(
                        np.unique(y_calib_set)) >= 2:
                    calibrated_model_obj = CalibratedClassifierCV(estimator=model_to_calibrate,
                                                                  method=calib_method_final, cv='prefit')
                    calibrated_model_obj.fit(X_calib_set, y_calib_set)
                    final_model_output = calibrated_model_obj;
                    was_calibrated_explicitly = True
            elif isinstance(calib_cv_setting_final, int) and calib_cv_setting_final > 1:
                estimator_template_for_calib_cv = None;
                current_model_params_template = best_params_for_final_model.copy()
                if 'early_stopping_rounds' in current_model_params_template: del current_model_params_template[
                    'early_stopping_rounds']
                if 'eval_set' in current_model_params_template: del current_model_params_template[
                    'eval_set']
                if 'eval_metric' in current_model_params_template: del current_model_params_template[
                    'eval_metric']
                if training_params_cfg.get('use_ensemble', False):
                    ensemble_opts_cfg = ADVANCED_TRAINING_CONFIG.get('ensemble_options', {});
                    rf_params_tpl = ensemble_opts_cfg.get('random_forest_params', {}).copy()
                    xgb_tpl = xgb.XGBClassifier(**current_model_params_template);
                    rf_tpl = RandomForestClassifier(random_state=random_state_train, **rf_params_tpl, n_jobs=-1)
                    estimator_template_for_calib_cv = VotingClassifier(estimators=[('xgb', xgb_tpl), ('rf', rf_tpl)],
                                                                       voting=ensemble_opts_cfg.get('voting_type',
                                                                                                    'soft'),
                                                                       weights=ensemble_opts_cfg.get('weights',
                                                                                                     {'xgboost': 0.7,
                                                                                                      'random_forest': 0.3}))
                else:
                    estimator_template_for_calib_cv = xgb.XGBClassifier(**current_model_params_template)

                if estimator_template_for_calib_cv and len(
                        np.unique(y_train_for_final_fit)) > 1:
                    class_weights_for_calib_cv = _calculate_class_weights(y_train_for_final_fit,
                                                                          strategy='balanced_subsample')
                    sample_weights_for_calib_cv = np.array(
                        [class_weights_for_calib_cv.get(y_val, 1.0) for y_val in y_train_for_final_fit])
                    fit_params_for_calib_cv_fit = {
                        'sample_weight': sample_weights_for_calib_cv}

                    calibrated_model_cv_obj = CalibratedClassifierCV(estimator=estimator_template_for_calib_cv,
                                                                     method=calib_method_final,
                                                                     cv=calib_cv_setting_final)
                    calibrated_model_cv_obj.fit(X_train_for_final_fit, y_train_for_final_fit,
                                                **fit_params_for_calib_cv_fit)
                    final_model_output = calibrated_model_cv_obj;
                    was_calibrated_explicitly = True
    except Exception as e_calib:
        logger.warning(f"[{synk_name}] Calibration failed: {e_calib}. Using model from before calibration attempt.",
                       exc_info=True)

    if was_calibrated_explicitly and model_to_calibrate is not final_model_output:
        try:
            if hasattr(model_to_calibrate, "predict_proba") and hasattr(final_model_output, "predict_proba"):
                proba_uncalibrated_all = model_to_calibrate.predict_proba(X_test_scaled_arr)
                proba_calibrated_all = final_model_output.predict_proba(X_test_scaled_arr)
                if proba_uncalibrated_all.shape[1] > 1 and proba_calibrated_all.shape[1] > 1:
                    proba_uncalibrated = proba_uncalibrated_all[:, 1];
                    proba_calibrated = proba_calibrated_all[:, 1]
                    logger.info(
                        f"[{synk_name}] Avg Test Probs (Class 1) - Uncalibrated: {np.mean(proba_uncalibrated):.4f}, Calibrated: {np.mean(proba_calibrated):.4f}")
                    y_pred_diag_temp_calib_05 = final_model_output.predict(X_test_scaled_arr)
                    tp_indices_diag = np.where((y_test_arr == 1) & (y_pred_diag_temp_calib_05 == 1))[0]
                    fn_indices_diag = np.where((y_test_arr == 1) & (y_pred_diag_temp_calib_05 == 0))[0]
                    if len(tp_indices_diag) > 0: logger.info(
                        f"[{synk_name}] Avg Test Probs for TPs (calibrated model @0.5 thresh) - Uncalibrated: {np.mean(proba_uncalibrated[tp_indices_diag]):.4f}, Calibrated: {np.mean(proba_calibrated[tp_indices_diag]):.4f}")
                    if len(fn_indices_diag) > 0: logger.info(
                        f"[{synk_name}] Avg Test Probs for FNs (calibrated model @0.5 thresh) - Uncalibrated: {np.mean(proba_uncalibrated[fn_indices_diag]):.4f}, Calibrated: {np.mean(proba_calibrated[fn_indices_diag]):.4f}")
        except Exception as e_calib_log:
            logger.warning(f"[{synk_name}] Could not log calibration impact probabilities: {e_calib_log}")

    optimal_threshold = 0.5
    optimal_f1 = 0.0
    y_proba_test_full = np.array([])  # Full probabilities for all classes on test set
    error_report_data = []  # For CSV error details

    if len(np.unique(y_test_arr)) > 1:
        try:
            y_proba_test_full = final_model_output.predict_proba(X_test_scaled_arr)
            y_proba_test_class1 = y_proba_test_full[:, 1]  # Probability for positive class (class 1)
            precisions_ot, recalls_ot, thresholds_ot_prc = precision_recall_curve(y_test_arr,
                                                                                  y_proba_test_class1,
                                                                                  pos_label=1)
            if len(thresholds_ot_prc) > 0:
                prec_for_f1 = precisions_ot[:-1];
                rec_for_f1 = recalls_ot[:-1]
                f1_scores_ot_calc = np.zeros_like(thresholds_ot_prc, dtype=float)
                valid_idx_ot = (prec_for_f1 + rec_for_f1) > 0
                if np.any(valid_idx_ot):
                    f1_scores_ot_calc[valid_idx_ot] = 2 * (prec_for_f1[valid_idx_ot] * rec_for_f1[valid_idx_ot]) / (
                            prec_for_f1[valid_idx_ot] + rec_for_f1[valid_idx_ot])
                    optimal_f1_idx = np.argmax(f1_scores_ot_calc)
                    optimal_threshold = thresholds_ot_prc[optimal_f1_idx]
                    optimal_f1 = f1_scores_ot_calc[optimal_f1_idx]
                else:
                    logger.warning(f"[{synk_name}] No valid P/R pairs for optimal threshold. Defaulting.")
            else:
                logger.warning(f"[{synk_name}] No thresholds from PRC. Defaulting threshold.")
        except Exception as e_thresh_eval:
            logger.error(f"[{synk_name}] Error in optimal threshold eval: {e_thresh_eval}. Defaulting.", exc_info=True)
    else:
        logger.warning(f"[{synk_name}] Test set has only one class. Optimal threshold skipped. Defaulting.")

    logger.info(
        f"[{synk_name}] Optimal F1 Score on Test (from PRC): {optimal_f1:.4f} at Threshold: {optimal_threshold:.4f}")

    # --- DETAILED LOGGING AND ERROR CSV DATA COLLECTION AT OPTIMAL THRESHOLD ---
    if optimal_threshold is not None and y_test_arr is not None and len(y_test_arr) > 0 and y_proba_test_full.size > 0:
        logger.info(f"[{synk_name}] Evaluating Test Set Performance at Optimal Threshold: {optimal_threshold:.4f}")
        y_pred_test_optimal = (y_proba_test_full[:, 1] >= optimal_threshold).astype(
            int)  # Predictions based on class 1 proba

        class_0_name = CLASS_NAMES.get(0, "Class 0");
        class_1_name = CLASS_NAMES.get(1, "Class 1")
        report_optimal = classification_report(y_test_arr, y_pred_test_optimal,
                                               target_names=[class_0_name, class_1_name], zero_division=0)
        logger.info(
            f"[{synk_name}] Test Set Classification Report (at Optimal Threshold {optimal_threshold:.4f}):\n{report_optimal}")
        cm_optimal = confusion_matrix(y_test_arr, y_pred_test_optimal, labels=[0, 1])
        logger.info(
            f"[{synk_name}] Test Set Confusion Matrix (at Optimal Threshold {optimal_threshold:.4f}):\n{cm_optimal}")

        precision_optimal_cls1 = precision_score(y_test_arr, y_pred_test_optimal, pos_label=1, zero_division=0)
        recall_optimal_cls1 = recall_score(y_test_arr, y_pred_test_optimal, pos_label=1, zero_division=0)
        f1_optimal_recalc_cls1 = f1_score(y_test_arr, y_pred_test_optimal, pos_label=1, zero_division=0)
        bal_acc_optimal = balanced_accuracy_score(y_test_arr, y_pred_test_optimal)

        logger.info(
            f"[{synk_name}] Test Metrics for Class 1 ('{class_1_name}') at Optimal Threshold ({optimal_threshold:.4f}): "
            f"F1: {f1_optimal_recalc_cls1:.4f}, Recall: {recall_optimal_cls1:.4f}, Precision: {precision_optimal_cls1:.4f}")
        logger.info(
            f"[{synk_name}] Overall Balanced Accuracy at Optimal Threshold ({optimal_threshold:.4f}): {bal_acc_optimal:.4f}")

        if cm_optimal.size == 4:
            tn, fp, fn, tp = cm_optimal.ravel()
            logger.info(
                f"[{synk_name}] TP: {tp}, FN: {fn}, FP: {fp}, TN: {tn} (at optimal threshold {optimal_threshold:.4f})")

        # --- Collect individual error details for CSV ---
        test_indices = X_test_df.index
        for i in range(len(y_test_arr)):
            true_label_num = y_test_arr[i]
            pred_label_num = y_pred_test_optimal[i]
            if true_label_num != pred_label_num:
                original_idx = test_indices[i]
                patient_id = metadata_test_df_for_errors.loc[
                    original_idx, 'Patient ID'] if 'Patient ID' in metadata_test_df_for_errors.columns and original_idx in metadata_test_df_for_errors.index else 'Unknown_ID'
                side = metadata_test_df_for_errors.loc[
                    original_idx, 'Side'] if 'Side' in metadata_test_df_for_errors.columns and original_idx in metadata_test_df_for_errors.index else 'Unknown_Side'

                instance_probas = y_proba_test_full[i]  # Probabilities for this instance

                error_entry = {
                    'Patient ID': patient_id,
                    'Side': side,
                    'Expert_Label_Name': CLASS_NAMES.get(true_label_num, str(true_label_num)),
                    'Predicted_Label_Name': CLASS_NAMES.get(pred_label_num, str(pred_label_num)),
                    'Is_Correct': False,
                    'Optimal_Threshold_Used': optimal_threshold
                }
                for class_val, class_name_str in CLASS_NAMES.items():
                    safe_class_name = class_name_str.replace(" ", "_")
                    if class_val < len(instance_probas):
                        error_entry[f'Prob_{safe_class_name}'] = instance_probas[class_val]
                    else:
                        error_entry[f'Prob_{safe_class_name}'] = 0.0
                error_report_data.append(error_entry)
        # --- End of error data collection ---

    feature_importance_df = pd.DataFrame()
    importances_arr = None;
    model_for_importance_extraction = final_model_output
    if isinstance(model_for_importance_extraction, CalibratedClassifierCV):
        if hasattr(model_for_importance_extraction,
                   'estimator') and model_for_importance_extraction.estimator is not None:
            model_for_importance_extraction = model_for_importance_extraction.estimator
        elif hasattr(model_for_importance_extraction,
                     'calibrated_classifiers_') and model_for_importance_extraction.calibrated_classifiers_:
            first_calibrator_object = model_for_importance_extraction.calibrated_classifiers_[0]
            if hasattr(first_calibrator_object,
                       'base_estimator_') and first_calibrator_object.base_estimator_ is not None:
                model_for_importance_extraction = first_calibrator_object.base_estimator_
    if isinstance(model_for_importance_extraction, VotingClassifier):
        if hasattr(model_for_importance_extraction,
                   'named_estimators_') and 'xgb' in model_for_importance_extraction.named_estimators_:
            actual_xgb_in_ensemble = model_for_importance_extraction.named_estimators_['xgb']
            if hasattr(actual_xgb_in_ensemble, 'feature_importances_'):
                importances_arr = actual_xgb_in_ensemble.feature_importances_
        elif hasattr(model_for_importance_extraction, 'estimators_') and \
                len(model_for_importance_extraction.estimators_) > 0 and \
                hasattr(model_for_importance_extraction.estimators_[0], 'feature_importances_'):
            importances_arr = model_for_importance_extraction.estimators_[0].feature_importances_
    elif hasattr(model_for_importance_extraction, 'feature_importances_'):
        importances_arr = model_for_importance_extraction.feature_importances_

    if importances_arr is not None and feature_names_list:
        if len(importances_arr) == len(feature_names_list):
            feature_importance_df = pd.DataFrame(
                {'feature': feature_names_list, 'importance': importances_arr}).sort_values('importance',
                                                                                            ascending=False)

    training_analysis_df = pd.DataFrame()
    if UTILS_LOADED and REVIEW_CONFIG.get('enabled', True):  # Assuming REVIEW_CONFIG is defined globally or passed
        try:
            y_pred_train_analysis_05 = final_model_output.predict(X_train_scaled_arr)
            y_proba_train_analysis = final_model_output.predict_proba(X_train_scaled_arr)
            analysis_data = {'Expert_Label': y_train_arr, 'Predicted_Label': y_pred_train_analysis_05}
            class_0_name_safe = CLASS_NAMES.get(0, "None").replace(" ", "_")
            class_1_name_safe = CLASS_NAMES.get(1, "Synkinesis").replace(" ", "_")
            if y_proba_train_analysis.shape[1] >= 2:
                analysis_data[f'Prob_{class_0_name_safe}'] = y_proba_train_analysis[:, 0]
                analysis_data[f'Prob_{class_1_name_safe}'] = y_proba_train_analysis[:, 1]
            elif y_proba_train_analysis.shape[1] == 1:
                analysis_data[f'Prob_{class_0_name_safe}'] = y_proba_train_analysis[:, 0]
                analysis_data[f'Prob_{class_1_name_safe}'] = 1.0 - y_proba_train_analysis[:, 0]
            analysis_index = X_train_df.index if X_train_df.index.equals(
                pd.RangeIndex(start=0, stop=len(y_train_arr), step=1)) or X_train_df.index.size == len(
                y_train_arr) else pd.RangeIndex(start=0, stop=len(y_train_arr), step=1)  # Safer index handling
            training_analysis_df = pd.DataFrame(analysis_data, index=analysis_index)
            training_analysis_df['Entropy'] = [calculate_entropy(p_row) for p_row in y_proba_train_analysis]
            if f'Prob_{class_1_name_safe}' in training_analysis_df and f'Prob_{class_0_name_safe}' in training_analysis_df:
                training_analysis_df['Margin'] = np.abs(
                    training_analysis_df[f'Prob_{class_1_name_safe}'] - training_analysis_df[
                        f'Prob_{class_0_name_safe}'])
            elif y_proba_train_analysis.shape[1] >= 2:
                training_analysis_df['Margin'] = np.abs(y_proba_train_analysis[:, 1] - y_proba_train_analysis[:, 0])
            else:
                training_analysis_df['Margin'] = 1.0
            true_label_probs = []
            for idx, true_label in enumerate(y_train_arr):
                if 0 <= true_label < y_proba_train_analysis.shape[1]:
                    true_label_probs.append(y_proba_train_analysis[idx, int(true_label)])
                else:
                    true_label_probs.append(0.0)
            training_analysis_df['Prob_True_Label'] = true_label_probs
            training_analysis_df['Is_Correct'] = (
                        training_analysis_df['Expert_Label'] == training_analysis_df['Predicted_Label'])
        except Exception as e_train_analysis:
            logger.error(f"[{synk_name}] Training set uncertainty analysis failed: {e_train_analysis}", exc_info=True)

    df_error_details = pd.DataFrame(error_report_data)

    return final_model_output, scaler, feature_importance_df, training_analysis_df, optimal_threshold, df_error_details  # Return optimal_threshold


def train_synkinesis_type(synk_type_key):
    if not CONFIG_LOADED: print(
        f"CRITICAL: Base synkinesis_config.py failed to load. Cannot train {synk_type_key}."); return
    if not UTILS_LOADED: print(
        f"WARNING: paralysis_utils.py failed to load. Review generation for {synk_type_key} might be impacted.")

    type_config = get_synkinesis_config(synk_type_key)
    if not type_config: print(f"ERROR: Synkinesis type '{synk_type_key}' not found in SYNKINESIS_CONFIG."); return

    synk_name_display = type_config.get('name', synk_type_key.capitalize())
    filenames_cfg = type_config.get('filenames', {})
    training_params_main = type_config.get('training', {})
    review_cfg_main = training_params_main.get('review_analysis', REVIEW_CONFIG)
    log_file_path_type = filenames_cfg.get('training_log', os.path.join(LOG_DIR, f'{synk_type_key}_training.log'))
    _setup_logging(log_file_path_type, LOGGING_CONFIG.get('level', 'INFO'), LOGGING_CONFIG.get('format'))

    logger.info(f"{'=' * 60}\nStarting Training for Synkinesis Type: {synk_name_display} (v1.10.4)\n{'=' * 60}")

    try:
        logger.info(f"[{synk_name_display}] Loading and preparing data...")
        feature_module_name = f"{synk_type_key}_features"
        try:
            feature_module = importlib.import_module(feature_module_name)
            prepare_data_func = getattr(feature_module, 'prepare_data')
        except ModuleNotFoundError:
            logger.error(f"[{synk_name_display}] Feature module '{feature_module_name}.py' not found. Skipping.");
            return
        except AttributeError:
            logger.error(f"[{synk_name_display}] 'prepare_data' func not in '{feature_module_name}.py'. Skipping.");
            return

        results_csv_path = INPUT_FILES.get('results_csv');
        expert_csv_path = INPUT_FILES.get('expert_key_csv')
        features_df, targets_arr, metadata_df = prepare_data_func(results_file=results_csv_path,
                                                                  expert_file=expert_csv_path)

        if features_df is None or features_df.empty: logger.error(
            f"[{synk_name_display}] No features. Aborting."); return
        if targets_arr is None or len(targets_arr) == 0: logger.error(
            f"[{synk_name_display}] No targets. Aborting."); return
        if len(features_df) != len(targets_arr): logger.error(
            f"[{synk_name_display}] Features/targets len mismatch. Aborting."); return
        if metadata_df is None:
            metadata_df = pd.DataFrame(index=features_df.index)
        elif len(metadata_df) != len(features_df):
            metadata_df = metadata_df.reindex(features_df.index)

        feature_names_final = features_df.columns.tolist()

        # Ensure metadata_df has 'Patient ID' and 'Side' if possible before split
        if 'Patient ID' not in metadata_df.columns:
            logger.warning(
                f"[{synk_name_display}] 'Patient ID' not in metadata_df from prepare_data. Error details will have 'Unknown_ID'.")
            metadata_df['Patient ID'] = 'Unknown_ID_Placeholder'  # Add placeholder
        if 'Side' not in metadata_df.columns:
            logger.warning(
                f"[{synk_name_display}] 'Side' not in metadata_df from prepare_data. Error details will have 'Unknown_Side'.")
            metadata_df['Side'] = 'Unknown_Side_Placeholder'  # Add placeholder

        stratify_split_arr = targets_arr if len(np.unique(targets_arr)) > 1 and all(
            c >= 2 for c in np.unique(targets_arr, return_counts=True)[1]) else None

        X_train, X_test, y_train, y_test, metadata_train, metadata_test = train_test_split(
            features_df, targets_arr, metadata_df,  # Pass full metadata_df here
            test_size=training_params_main.get('test_size', 0.25),
            random_state=training_params_main.get('random_state', 42),
            stratify=stratify_split_arr
        )
        eval_metrics_list = ADVANCED_TRAINING_CONFIG.get('evaluation_metrics', ['f1_score', 'roc_auc_score'])

        final_trained_model, final_scaler, df_feature_importance, df_training_analysis, optimal_threshold_for_review, df_error_details_generated = train_synkinesis_model(
            synk_type_key, X_train, y_train, X_test, y_test,
            feature_names_final, type_config, eval_metrics_list,
            metadata_test  # Pass the metadata_test split
        )
        logger.info(
            f"[{synk_name_display}] Optimal threshold from train_synkinesis_model: {optimal_threshold_for_review:.4f}")

        if final_trained_model is None: logger.error(f"[{synk_name_display}] Model training failed. Aborting."); return

        model_file_path = filenames_cfg.get('model');
        scaler_file_path = filenames_cfg.get('scaler');
        importance_file_path = filenames_cfg.get('importance')
        error_details_csv_path = filenames_cfg.get('error_details_csv')  # Get CSV path

        for path_val_save in [model_file_path, scaler_file_path, importance_file_path,
                              error_details_csv_path]:  # Add error_details_csv_path
            if path_val_save: os.makedirs(os.path.dirname(path_val_save), exist_ok=True)

        if model_file_path: joblib.dump(final_trained_model, model_file_path)
        if scaler_file_path: joblib.dump(final_scaler, scaler_file_path)

        if df_feature_importance is not None and not df_feature_importance.empty and importance_file_path:
            df_feature_importance.to_csv(importance_file_path, index=False)

        # Save the error details CSV
        if error_details_csv_path:
            if df_error_details_generated is not None and not df_error_details_generated.empty:
                df_error_details_generated.to_csv(error_details_csv_path, index=False, float_format='%.4f')
                logger.info(
                    f"[{synk_name_display}] Saved {len(df_error_details_generated)} error details to {error_details_csv_path}")
            else:
                logger.info(
                    f"[{synk_name_display}] No error details generated or returned empty. CSV not created at {error_details_csv_path}")
                # Optionally create an empty CSV with headers if needed downstream
                # header_cols = ['Patient ID', 'Side', 'Expert_Label_Name', 'Predicted_Label_Name', 'Is_Correct', 'Optimal_Threshold_Used'] + [f'Prob_{cn.replace(" ","_")}' for cn in CLASS_NAMES.values()]
                # pd.DataFrame(columns=header_cols).to_csv(error_details_csv_path, index=False)
        else:
            logger.warning(
                f"[{synk_name_display}] 'error_details_csv' path not defined in config. Error details CSV not saved.")

        if review_cfg_main.get('enabled', False) and UTILS_LOADED:
            logger.info(
                f"[{synk_name_display}] Generating review candidates using optimal threshold: {optimal_threshold_for_review:.4f}...")
            review_csv_path = filenames_cfg.get('review_candidates_csv')
            if review_csv_path and not os.path.isabs(review_csv_path) and not review_csv_path.startswith(ANALYSIS_DIR):
                review_csv_path = os.path.join(ANALYSIS_DIR, synk_type_key, os.path.basename(review_csv_path))

            if review_csv_path:
                os.makedirs(os.path.dirname(review_csv_path), exist_ok=True)
                if df_training_analysis is None or df_training_analysis.empty:
                    logger.warning(
                        f"[{synk_name_display}] Training analysis DataFrame empty. Cannot generate review candidates.")
                else:
                    metadata_train_for_review = metadata_train.copy()
                    df_training_analysis_for_review = df_training_analysis.copy()
                    if not metadata_train_for_review.index.equals(df_training_analysis_for_review.index):
                        try:
                            metadata_train_for_review = metadata_train_for_review.reindex(
                                df_training_analysis_for_review.index)
                        except Exception as e_reindex:
                            logger.error(f"Failed to reindex metadata_train: {e_reindex}.")

                    review_candidates_df = generate_review_candidates(
                        zone=synk_type_key, model=final_trained_model, scaler=final_scaler,
                        feature_names=feature_names_final,
                        config=type_config, X_train_orig=X_train, y_train_orig=y_train, X_test_orig=X_test,
                        y_test_orig=y_test,
                        metadata_train=metadata_train_for_review, training_analysis_df=df_training_analysis_for_review,
                        class_names_map=CLASS_NAMES,
                        decision_threshold=optimal_threshold_for_review,
                        top_k_influence=review_cfg_main.get('top_k_influence', 30),
                        entropy_quantile=review_cfg_main.get('entropy_quantile', 0.9),
                        margin_quantile=review_cfg_main.get('margin_quantile', 0.1),
                        true_label_prob_threshold=review_cfg_main.get('true_label_prob_threshold', 0.4)
                    )
                    if review_candidates_df is not None and not review_candidates_df.empty:
                        review_candidates_df.to_csv(review_csv_path, index=False, float_format='%.4f')
                        logger.info(f"Saved {len(review_candidates_df)} review candidates to {review_csv_path}")
                    else:
                        logger.info(f"[{synk_name_display}] No review candidates generated or returned empty.")
        logger.info(f"[{synk_name_display}] Training complete!\n{'=' * 60}")
    except Exception as e_pipeline:
        logger.error(f"[{synk_name_display}] Pipeline failed for {synk_type_key}: {str(e_pipeline)}", exc_info=True)
        logger.info(f"{'=' * 60}")


def analyze_model_performance_synkinesis(synk_type_key):
    if not OPTUNA_AVAILABLE:
        return
    type_config = get_synkinesis_config(synk_type_key)
    if not type_config:
        return
    synk_name_display = type_config.get('name', synk_type_key.capitalize())
    filenames_cfg_analyzer = type_config.get('filenames', {})
    model_file_path_analyzer = filenames_cfg_analyzer.get('model', f'models/synkinesis/{synk_type_key}/model.pkl')
    model_base_dir_analyzer = os.path.dirname(model_file_path_analyzer)
    if not model_base_dir_analyzer: model_base_dir_analyzer = "."
    study_path_analyzer = os.path.join(model_base_dir_analyzer, f"{synk_type_key}_optuna_study.pkl")
    analysis_plots_dir = os.path.join(ANALYSIS_DIR, synk_type_key, 'optuna_plots')
    os.makedirs(analysis_plots_dir, exist_ok=True)
    current_logger = logging.getLogger(__name__)

    if os.path.exists(study_path_analyzer):
        try:
            study_loaded = joblib.load(study_path_analyzer)
            plot_opt_hist = ADVANCED_TRAINING_CONFIG.get('monitoring', {}).get('plot_optimization_history', False)
            if plot_opt_hist:
                try:
                    import plotly;
                    from optuna.visualization import plot_optimization_history, plot_param_importances
                    # fig_history = plot_optimization_history(study_loaded) # Plotly images can be large
                    # history_plot_path = os.path.join(analysis_plots_dir, f"{synk_type_key}_optuna_history.png")
                    # fig_history.write_image(history_plot_path)
                    # if len(study_loaded.trials) > 1 and study_loaded.best_trial:
                    #     fig_importance = plot_param_importances(study_loaded)
                    #     importance_plot_path = os.path.join(analysis_plots_dir, f"{synk_type_key}_optuna_param_importance.png")
                    # fig_importance.write_image(importance_plot_path)
                except ImportError:
                    pass
                except Exception as e_plot:
                    pass
        except Exception as e_load_study:
            current_logger.error(f"Could not analyze Optuna study for {synk_name_display}: {e_load_study}")


if __name__ == "__main__":
    if not CONFIG_LOADED:
        print("CRITICAL: synkinesis_config.py could not be loaded. Aborting main training script.")
    else:
        main_script_log_file = os.path.join(LOG_DIR, "synkinesis_main_pipeline_orchestrator.log")
        _setup_logging(main_script_log_file, LOGGING_CONFIG.get('level', 'INFO'), LOGGING_CONFIG.get('format'))
        main_logger = logging.getLogger(__name__)
        main_logger.info("Synkinesis Training Pipeline Script Started (v1.10.4).")

        optuna_globally_needed = any(
            s_cfg.get('training', {}).get('hyperparameter_tuning', {}).get('enabled', False) and
            s_cfg.get('training', {}).get('hyperparameter_tuning', {}).get('method') == 'optuna'
            for s_cfg in SYNKINESIS_CONFIG.values() if isinstance(s_cfg, dict)
        )

        if optuna_globally_needed and not OPTUNA_AVAILABLE:
            main_logger.critical(
                "Optuna is required but not installed. Please install with: pip install optuna plotly kaleido")

        synkinesis_types_to_run = get_all_synkinesis_types()

        if not synkinesis_types_to_run:
            main_logger.warning("No synkinesis types found in SYNKINESIS_CONFIG to process.")
        else:
            main_logger.info(f"Synkinesis types to process: {synkinesis_types_to_run}")

        for sk_type in synkinesis_types_to_run:
            main_logger.info(f"\n>>> Processing Synkinesis Type: {sk_type.upper()} <<<")
            train_synkinesis_type(sk_type)
            analyze_model_performance_synkinesis(sk_type)
            main_logger.info(f"\n>>> Finished Processing Synkinesis Type: {sk_type.upper()} <<<")

        main_logger.info("Synkinesis Training Pipeline Script Finished All Types.")