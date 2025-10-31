# paralysis_training_helpers.py

import logging
import os
import numpy as np
import pandas as pd
import joblib
from copy import deepcopy
# import scipy.stats # Not directly used here, but by utils

import warnings

os.environ['PYTHONWARNINGS'] = 'ignore::UserWarning'
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    from sklearn.model_selection import StratifiedKFold, KFold
    from sklearn.metrics import f1_score, balanced_accuracy_score
    from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, VotingClassifier
    import xgboost as xgb
    from imblearn.over_sampling import SMOTE, BorderlineSMOTE, ADASYN
    from imblearn.combine import SMOTEENN
    from imblearn.under_sampling import TomekLinks, EditedNearestNeighbours

    try:
        import optuna

        OPTUNA_AVAILABLE = True
        optuna.logging.set_verbosity(optuna.logging.WARNING)
    except ImportError:
        OPTUNA_AVAILABLE = False

from paralysis_config import (
    ADVANCED_TRAINING_CONFIG, MODEL_DIR, ANALYSIS_DIR, PERFORMANCE_CONFIG
)
from paralysis_utils import PARALYSIS_MAP

logger = logging.getLogger(__name__)

def setup_logging_for_zone(log_file, level_str, log_format_str):
    log_level = getattr(logging, level_str.upper(), logging.INFO)
    log_dir_path = os.path.dirname(log_file)
    if log_dir_path: os.makedirs(log_dir_path, exist_ok=True)
    root_logger = logging.getLogger()
    if root_logger.hasHandlers(): root_logger.handlers.clear()
    logging.basicConfig(level=log_level, format=log_format_str,
                        handlers=[logging.StreamHandler(), logging.FileHandler(log_file, mode='w')],
                        force=True)
    logger.info(f"Logging for current zone initialized. Level: {level_str}. File: {log_file}")

def apply_data_augmentation(X, y, aug_config, random_state=42):
    if not aug_config.get('enabled', False):
        return X, y
    if isinstance(X, pd.DataFrame):
        X_np = X.values
    else:
        X_np = X.copy()
    y_np = y.copy()
    np.random.seed(random_state)  # Ensure reproducibility
    methods = aug_config.get('methods', {})
    aug_factor = aug_config.get('augmentation_factor', 0.3)
    unique_classes, class_counts = np.unique(y_np, return_counts=True)
    minority_classes = np.array([])
    if class_counts.size > 0:  # Check if class_counts is not empty
        median_count = np.median(class_counts)
        minority_classes = unique_classes[class_counts < median_count]
        if not minority_classes.size and unique_classes.size > 1:  # If all classes have same count or only one class
            minority_classes = unique_classes[class_counts == np.min(class_counts)]  # Target the smallest class(es)
    if aug_config.get('apply_to_minority_only', True) and minority_classes.size > 0:
        mask = np.isin(y_np, minority_classes)
        X_to_aug_np, y_to_aug = X_np[mask], y_np[mask]
    else:
        X_to_aug_np, y_to_aug = X_np, y_np
    if X_to_aug_np.shape[0] == 0: return X_np, y_np  # No samples to augment
    n_aug = int(X_to_aug_np.shape[0] * aug_factor)
    if n_aug == 0: return X_np, y_np  # Augmentation factor too small or no samples
    aug_indices = np.random.choice(X_to_aug_np.shape[0], n_aug, replace=True)
    X_aug_np, y_aug_resampled = X_to_aug_np[aug_indices].copy(), y_to_aug[aug_indices].copy()
    if methods.get('noise_injection', {}).get('enabled', False):
        noise_level = methods['noise_injection'].get('noise_level', 0.02)
        noise_type = methods['noise_injection'].get('noise_type', 'gaussian')
        if noise_type == 'gaussian':
            noise = np.random.normal(0, noise_level, X_aug_np.shape)
        else:
            noise = (np.random.random(X_aug_np.shape) - 0.5) * 2 * noise_level  # Uniform
        X_aug_np += noise
    if methods.get('feature_perturbation', {}).get('enabled', False):
        perturb_factor = methods['feature_perturbation'].get('perturbation_factor', 0.05)
        perturb_prob = methods['feature_perturbation'].get('perturbation_probability', 0.3)
        perturb_mask = np.random.random(X_aug_np.shape) < perturb_prob
        perturbation_values = np.random.uniform(-perturb_factor, perturb_factor, X_aug_np.shape)
        X_aug_np[perturb_mask] *= (1 + perturbation_values[perturb_mask])
    if methods.get('mixup', {}).get('enabled', False):
        alpha = methods['mixup'].get('alpha', 0.2)
        mix_prob = methods['mixup'].get('probability', 0.5)
        num_mixup_samples = int(X_aug_np.shape[0] * mix_prob)
        if num_mixup_samples > 0:
            mixup_indices_in_aug_pool = np.random.choice(X_aug_np.shape[0], num_mixup_samples, replace=False)
            for i_mix in mixup_indices_in_aug_pool:
                current_class_label_mix = y_aug_resampled[i_mix]
                same_class_indices_in_aug_pool_for_mix = np.where(y_aug_resampled == current_class_label_mix)[0]
                valid_indices_for_j_mix = [idx_mix for idx_mix in same_class_indices_in_aug_pool_for_mix if
                                           idx_mix != i_mix]
                if not valid_indices_for_j_mix: continue
                j_mix_idx_in_aug_pool = np.random.choice(valid_indices_for_j_mix)
                lam_mix = np.random.beta(alpha, alpha)
                X_aug_np[i_mix] = lam_mix * X_aug_np[i_mix] + (1 - lam_mix) * X_aug_np[j_mix_idx_in_aug_pool]
    X_combined_np = np.vstack([X_np, X_aug_np])
    y_combined = np.hstack([y_np, y_aug_resampled])
    return X_combined_np, y_combined

def get_smote_sampling_strategy(y_data, configured_strategy, adaptive_params=None,
                                min_samples_for_recalc=50, majority_class_label=0):
    unique_classes, counts = np.unique(y_data, return_counts=True)
    if not counts.size: return 'auto'
    current_dist = dict(zip(unique_classes, counts))
    if configured_strategy == 'adaptive' and adaptive_params:
        sampling_dict = {}
        majority_count = current_dist.get(majority_class_label, max(counts) if counts.size > 0 else 0)
        explicit_targets = adaptive_params.get('explicit_target_counts', {})
        if explicit_targets:
            for cls_label_str, target_val in explicit_targets.items():
                try:
                    cls = int(cls_label_str)
                except ValueError:
                    continue
                if cls not in current_dist: continue
                if target_val == 'auto':
                    sampling_dict[cls] = current_dist[cls] if cls == majority_class_label else majority_count
                elif isinstance(target_val, (int, float)):
                    sampling_dict[cls] = max(int(target_val), current_dist[cls],
                                             min_samples_for_recalc if cls != majority_class_label else current_dist[
                                                 cls])
                else:
                    sampling_dict[cls] = current_dist[cls]
        else:
            min_after_smote = adaptive_params.get('min_samples_after_smote', min_samples_for_recalc)
            for cls_label, count_val in current_dist.items():
                cls = int(cls_label)
                if cls == majority_class_label: sampling_dict[cls] = count_val; continue
                target_count = count_val;
                ratio = None
                # Assuming PARALYSIS_MAP has 1: 'Partial', 2: 'Complete' for these specific ratios
                partial_val_map = [k for k, v in PARALYSIS_MAP.items() if str(v).lower() == 'partial']
                complete_val_map = [k for k, v in PARALYSIS_MAP.items() if str(v).lower() == 'complete']
                if partial_val_map and cls == partial_val_map[0]:
                    ratio = adaptive_params.get('target_ratio_partial_to_majority')
                elif complete_val_map and cls == complete_val_map[0]:
                    ratio = adaptive_params.get('target_ratio_complete_to_majority')
                if ratio is not None: target_count = max(count_val, int(majority_count * ratio))
                target_count = max(target_count, min_after_smote)
                sampling_dict[cls] = int(target_count)
        for cls_label_iter in current_dist:
            cls_iter = int(cls_label_iter)
            if cls_iter not in sampling_dict:
                sampling_dict[cls_iter] = current_dist[cls_label_iter]
            else:
                sampling_dict[cls_iter] = max(sampling_dict[cls_iter], current_dist[cls_label_iter])
        logger.debug(f"Adaptive SMOTE strategy: Original {current_dist} -> Target {sampling_dict}")
        return sampling_dict if sampling_dict else 'auto'
    return configured_strategy


def apply_smote_and_cleaning(X_train, y_train, smote_config, random_state, zone_name_log="Zone"):
    if not smote_config.get('enabled', False): return X_train, y_train
    logger.info(f"[{zone_name_log}] Applying SMOTE/cleaning...")
    if isinstance(X_train, pd.DataFrame):
        X_train_np = X_train.values
    else:
        X_train_np = X_train.copy()
    y_train_np = y_train.copy()
    X_resampled_np, y_resampled = X_train_np, y_train_np
    n_jobs_parallel = PERFORMANCE_CONFIG.get('parallel_processing', {}).get('n_jobs', 1)
    try:
        variant = smote_config.get('variant', 'regular').lower()
        k_neighbors_cfg = smote_config.get('k_neighbors', 5)
        sampling_strategy_cfg = smote_config.get('sampling_strategy', 'auto')
        adaptive_params = smote_config.get('adaptive_strategy_params', {})
        min_samples_recalc = smote_config.get('min_samples_per_class', 50)
        use_smoteenn = smote_config.get('use_smoteenn_after', False)
        use_tomek = smote_config.get('use_tomek_after', False) and not use_smoteenn
        unique_y, counts_y = np.unique(y_train_np, return_counts=True)
        if not counts_y.size: logger.info(
            f"[{zone_name_log}] SMOTE not applicable (empty y_train). Using original data."); return X_train, y_train
        min_class_count = min(counts_y) if counts_y.size > 0 else 0
        actual_k_neighbors = min(k_neighbors_cfg, max(1, min_class_count - 1)) if min_class_count > 1 else 0
        if actual_k_neighbors >= 1 and len(unique_y) > 1:
            current_smote_strategy = get_smote_sampling_strategy(y_train_np, sampling_strategy_cfg, adaptive_params,
                                                                 min_samples_recalc)
            if use_smoteenn:
                logger.info(f"[{zone_name_log}] Applying SMOTEENN...")
                enn_strategy = smote_config.get('enn_sampling_strategy', 'auto');
                enn_kind_sel = smote_config.get('enn_kind_sel', 'mode');
                enn_n_neighbors = smote_config.get('enn_n_neighbors', 3)
                enn = EditedNearestNeighbours(sampling_strategy=enn_strategy, kind_sel=enn_kind_sel,
                                              n_jobs=n_jobs_parallel, n_neighbors=enn_n_neighbors)
                original_variant_for_log = variant
                if original_variant_for_log not in ['regular', '', None]: logger.warning(
                    f"[{zone_name_log}] Configured SMOTE variant is '{original_variant_for_log}', but SMOTEENN is enabled. The 'smote' part of SMOTEENN will use regular SMOTE.")
                smote_for_enn_instance = SMOTE(k_neighbors=actual_k_neighbors, random_state=random_state,
                                               sampling_strategy=current_smote_strategy)
                logger.info(
                    f"[{zone_name_log}] SMOTEENN internal SMOTE type: regular SMOTE, k={actual_k_neighbors}, strategy={current_smote_strategy}")
                smote_enn_cleaner = SMOTEENN(random_state=random_state, smote=smote_for_enn_instance, enn=enn)
                X_resampled_np, y_resampled = smote_enn_cleaner.fit_resample(X_train_np, y_train_np)
                logger.info(f"[{zone_name_log}] After SMOTEENN: X shape {X_resampled_np.shape}")
            else:
                logger.info(
                    f"[{zone_name_log}] SMOTE params (primary): k={actual_k_neighbors}, variant={variant}, strategy={current_smote_strategy}")
                smote_instance = None
                if variant == 'borderline':
                    kind = adaptive_params.get('borderline_kind', 'borderline-1')
                    smote_instance = BorderlineSMOTE(k_neighbors=actual_k_neighbors, random_state=random_state,
                                                     sampling_strategy=current_smote_strategy, kind=kind,
                                                     n_jobs=n_jobs_parallel)
                elif variant == 'adasyn':
                    smote_instance = ADASYN(n_neighbors=actual_k_neighbors, random_state=random_state,
                                            sampling_strategy=current_smote_strategy, n_jobs=n_jobs_parallel)
                else:  # regular SMOTE
                    smote_instance = SMOTE(k_neighbors=actual_k_neighbors, random_state=random_state,
                                           sampling_strategy=current_smote_strategy, n_jobs=n_jobs_parallel)
                X_resampled_np, y_resampled = smote_instance.fit_resample(X_train_np, y_train_np)
                logger.info(f"[{zone_name_log}] After primary SMOTE ({variant}): X shape {X_resampled_np.shape}")
                if use_tomek:
                    logger.info(f"[{zone_name_log}] Applying Tomek Links...")
                    tomek_cleaner = TomekLinks(sampling_strategy='auto', n_jobs=n_jobs_parallel)
                    X_resampled_np, y_resampled = tomek_cleaner.fit_resample(X_resampled_np, y_resampled)
                    logger.info(f"[{zone_name_log}] After Tomek Links: X shape {X_resampled_np.shape}")
            unique_smote_final, counts_smote_final = np.unique(y_resampled, return_counts=True)
            logger.info(
                f"[{zone_name_log}] Final training distribution after SMOTE/cleaning: {dict(zip(unique_smote_final, counts_smote_final))}")
        else:
            logger.info(
                f"[{zone_name_log}] SMOTE not applicable (k_neighbors < 1 or single class). Using original data.")
    except Exception as e_smote_clean:
        logger.error(f"[{zone_name_log}] SMOTE/cleaning failed: {e_smote_clean}. Using original data.", exc_info=True)
        if isinstance(X_train, pd.DataFrame): return X_train.values, y_train
        return X_train, y_train
    return X_resampled_np, y_resampled

def get_optuna_suggestion(trial, param_name, param_config_list):
    param_type = param_config_list[0]
    options = param_config_list[3] if len(param_config_list) > 3 and isinstance(param_config_list[3], dict) else {}
    if param_type == 'float':
        return trial.suggest_float(param_name, param_config_list[1], param_config_list[2], **options)
    elif param_type == 'int':
        return trial.suggest_int(param_name, param_config_list[1], param_config_list[2], **options)
    elif param_type == 'categorical':
        return trial.suggest_categorical(param_name, param_config_list[1])
    else:
        raise ValueError(f"Unsupported Optuna parameter type: {param_type}")


def create_optuna_objective(
        X_train_scaled_np, y_train_np,
        optuna_config, model_params_base_xgb,
        class_weights_config,  # This is the class_weights map from config
        smote_config,
        num_classes, partial_class_index, random_state, zone_name,
        optuna_n_jobs=1):  # Add parameter to know how many parallel trials are running
    if not OPTUNA_AVAILABLE:
        raise ImportError("Optuna is not available.")

    param_distributions_xgb = optuna_config.get('param_distributions', {})
    cv_folds_optuna_config = optuna_config.get('cv_folds', 5)
    scoring_type = optuna_config.get('scoring', 'f1_macro').lower()

    smote_enabled_in_objective = smote_config.get('enabled', False) and \
                                 smote_config.get('apply_per_fold_in_tuning', False)

    adv_ensemble_opts = ADVANCED_TRAINING_CONFIG.get('ensemble_options', {})
    rf_params_optuna_cfg = adv_ensemble_opts.get('random_forest_params', {}).copy()
    et_params_optuna_cfg = adv_ensemble_opts.get('extra_trees_params', {}).copy()
    voting_weights_optuna_cfg = adv_ensemble_opts.get('weights', {'xgb': 0.6, 'rf': 0.2, 'et': 0.2})
    voting_type_optuna_cfg = adv_ensemble_opts.get('voting_type', 'soft')

    # Multi-core optimization: Calculate optimal thread count for XGBoost
    # With optuna_n_jobs parallel trials, distribute available cores among them
    n_jobs_parallel = PERFORMANCE_CONFIG.get('parallel_processing', {}).get('n_jobs', -1)
    if n_jobs_parallel == -1:
        n_jobs_parallel = os.cpu_count() or 1

    # Each Optuna trial gets an equal share of cores for its XGBoost model
    # This prevents thread over-subscription and maintains good parallelism
    xgb_nthread = max(1, n_jobs_parallel // optuna_n_jobs) if optuna_n_jobs > 0 else n_jobs_parallel
    logger.debug(f"[{zone_name}] XGBoost will use {xgb_nthread} threads per trial ({optuna_n_jobs} parallel trials)")

    # n_jobs for base models within Optuna fold. VotingClassifier n_jobs is for predict/proba.
    # Base models like RF, ET, XGB can use their own n_jobs if set in their params.
    # Keep VotingClassifier's n_jobs for predict/proba at 1 to avoid oversubscription if base models are parallel.
    n_jobs_voting_predict_optuna = 1

    def objective(trial):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")

            xgb_params_trial = {
                'objective': model_params_base_xgb.get('objective', 'multi:softprob'),  # Base objective
                'num_class': num_classes,
                'random_state': random_state,
                'eval_metric': model_params_base_xgb.get('eval_metric', 'mlogloss'),  # Base eval_metric
                'tree_method': model_params_base_xgb.get('tree_method', 'hist'),
                'nthread': xgb_nthread,  # Optimal thread count for parallel Optuna trials
                'verbosity': 0  # Suppress XGBoost verbosity during Optuna
            }
            # Update with objective/eval_metric based on num_classes if not already set
            if num_classes <= 2 and xgb_params_trial['objective'] == 'multi:softprob':
                xgb_params_trial['objective'] = 'binary:logistic'
                xgb_params_trial['eval_metric'] = 'logloss'  # or 'auc', 'error'
                if 'num_class' in xgb_params_trial: del xgb_params_trial['num_class']

            for param_name_iter, param_config_list_iter in param_distributions_xgb.items():
                xgb_params_trial[param_name_iter] = get_optuna_suggestion(trial, param_name_iter,
                                                                          param_config_list_iter)

            fold_scores_agg = []
            unique_labels_overall, counts_overall = np.unique(y_train_np, return_counts=True)
            min_samples_for_stratify = cv_folds_optuna_config

            cv_obj_outer_optuna = None
            if len(unique_labels_overall) < 2 or any(c < min_samples_for_stratify for c in counts_overall if c > 0):
                cv_obj_outer_optuna = KFold(n_splits=cv_folds_optuna_config, shuffle=True, random_state=random_state)
            else:
                cv_obj_outer_optuna = StratifiedKFold(n_splits=cv_folds_optuna_config, shuffle=True,
                                                      random_state=random_state)

            for fold_idx, (train_idx, val_idx) in enumerate(cv_obj_outer_optuna.split(X_train_scaled_np, y_train_np)):
                X_f_train, y_f_train = X_train_scaled_np[train_idx], y_train_np[train_idx]
                X_f_val, y_f_val = X_train_scaled_np[val_idx], y_train_np[val_idx]

                X_f_train_resampled, y_f_train_resampled = X_f_train.copy(), y_f_train.copy()
                if smote_enabled_in_objective:
                    try:
                        X_f_train_resampled, y_f_train_resampled = apply_smote_and_cleaning(
                            X_f_train, y_f_train, smote_config, random_state, f"{zone_name}-OptFold{fold_idx}"
                        )
                    except Exception as e_smote_fold_obj:
                        logger.warning(
                            f"[{zone_name} Optuna Trial {trial.number} Fold {fold_idx}] SMOTE failed: {e_smote_fold_obj}. Using original fold data.")

                fold_class_weights_map = calculate_class_weights_for_model(y_f_train_resampled, class_weights_config)

                xgb_fold = xgb.XGBClassifier(**xgb_params_trial)
                sample_weights_xgb_fold = np.array(
                    [fold_class_weights_map.get(int(lbl), 1.0) for lbl in y_f_train_resampled])

                # XGBoost early stopping inside Optuna fold:
                # If 'n_estimators' is tuned, XGBoost's own early stopping might interfere or be redundant
                # with Optuna's pruner. For simplicity, rely on Optuna's pruner and tuned 'n_estimators'.
                # If xgb_params_trial contains 'early_stopping_rounds' from Optuna search, it would need an eval_set.
                # We will not provide one to xgb_fold.fit() to keep it simpler.
                xgb_fit_params_fold_opt = {}  # No early stopping here for XGB's fit

                try:
                    xgb_fold.fit(X_f_train_resampled, y_f_train_resampled, sample_weight=sample_weights_xgb_fold,
                                 **xgb_fit_params_fold_opt)
                except Exception as e_xgb_fit_optuna:
                    logger.error(
                        f"[{zone_name} Optuna Trial {trial.number} Fold {fold_idx}] XGBoost fit failed: {e_xgb_fit_optuna}. Skipping fold.")
                    fold_scores_agg.append(0.0)  # Penalize this fold heavily
                    continue  # Skip to next fold

                rf_fold_params_this = rf_params_optuna_cfg.copy()
                rf_fold_params_this['random_state'] = random_state
                rf_fold_params_this['class_weight'] = fold_class_weights_map  # RF uses map directly
                rf_fold = RandomForestClassifier(**rf_fold_params_this)

                et_fold_params_this = et_params_optuna_cfg.copy()
                et_fold_params_this['random_state'] = random_state
                et_fold_params_this['class_weight'] = fold_class_weights_map  # ET uses map directly
                et_fold = ExtraTreesClassifier(**et_fold_params_this)

                estimators_fold = [
                    ('xgb', xgb_fold),  # PRE-FITTED
                    ('rf', rf_fold),  # Unfitted
                    ('et', et_fold)  # Unfitted
                ]

                model_fold_obj = VotingClassifier(
                    estimators=estimators_fold,
                    voting=voting_type_optuna_cfg,
                    weights=[voting_weights_optuna_cfg.get('xgb', 0.6), voting_weights_optuna_cfg.get('rf', 0.2),
                             voting_weights_optuna_cfg.get('et', 0.2)],
                    n_jobs=n_jobs_voting_predict_optuna
                )

                try:
                    model_fold_obj.fit(X_f_train_resampled, y_f_train_resampled)  # Fits only RF and ET

                    y_pred_f_val = model_fold_obj.predict(X_f_val)

                    current_score_fold = 0.0
                    if scoring_type == 'f1_macro':
                        current_score_fold = f1_score(y_f_val, y_pred_f_val, average='macro', zero_division=0)
                    elif scoring_type == 'balanced_accuracy':
                        current_score_fold = balanced_accuracy_score(y_f_val, y_pred_f_val, adjusted=False)
                    elif scoring_type == 'f1_weighted':
                        current_score_fold = f1_score(y_f_val, y_pred_f_val, average='weighted', zero_division=0)
                    elif scoring_type == 'f1_partial' and partial_class_index != -1:  # Check if partial_class_index is valid
                        present_labels_fold_val = np.unique(np.concatenate((y_f_val, y_pred_f_val)))
                        all_possible_labels_fold_val = list(range(num_classes))
                        if partial_class_index in present_labels_fold_val:  # Check if partial class is in this fold's y_val or y_pred_f_val
                            f1_all_classes_fold_val = f1_score(y_f_val, y_pred_f_val, average=None,
                                                               labels=all_possible_labels_fold_val, zero_division=0)
                            if partial_class_index < len(f1_all_classes_fold_val):
                                current_score_fold = f1_all_classes_fold_val[partial_class_index]
                            else:  # Should not happen if labels are correct
                                current_score_fold = 0.0  # Default if partial_class_index out of bounds
                        else:  # Partial class not present in this fold's validation, score is 0 for this metric
                            current_score_fold = 0.0
                    else:
                        current_score_fold = f1_score(y_f_val, y_pred_f_val, average='macro', zero_division=0)
                        if scoring_type not in ['f1_macro', 'balanced_accuracy', 'f1_weighted', 'f1_partial']:
                            logger.warning(
                                f"Unknown scoring type '{scoring_type}' in Optuna objective, defaulting to f1_macro.")

                    fold_scores_agg.append(current_score_fold)
                    trial.report(current_score_fold, fold_idx)
                    if trial.should_prune():
                        raise optuna.TrialPruned()
                except optuna.TrialPruned:
                    raise
                except Exception as e_fit_fold:
                    logger.error(
                        f"[{zone_name} Optuna Trial {trial.number} Fold {fold_idx}] VotingClassifier Fit/Eval Error: {e_fit_fold}",
                        exc_info=False)
                    fold_scores_agg.append(0.0)

        avg_score_trial = np.mean(fold_scores_agg) if fold_scores_agg else 0.0
        return avg_score_trial

    return objective

def optimize_class_thresholds(model, X_val, y_val, class_names_map, zone_config):
    threshold_config_main = zone_config.get('training', {}).get('threshold_optimization', {})
    if not threshold_config_main.get('enabled', False): return None
    y_proba = model.predict_proba(X_val)
    n_classes = y_proba.shape[1]
    optimal_thresholds_map = {}
    # Determine partial class index from class_names_map
    partial_class_idx_thresh = -1
    for k_map, v_map in class_names_map.items():
        if str(v_map).lower() == 'partial': partial_class_idx_thresh = int(k_map); break

    for class_idx in range(n_classes):
        class_name_iter = class_names_map.get(class_idx, f"Class_{class_idx}")
        current_range_key = f'class_{class_idx}_range'
        current_step_key = f'class_{class_idx}_step_size'
        # Use specific range for partial class if defined
        if class_idx == partial_class_idx_thresh and 'partial_class_range' in threshold_config_main:
            current_range = threshold_config_main.get('partial_class_range')
            current_step = threshold_config_main.get('partial_class_step_size', threshold_config_main.get('step_size',
                                                                                                          0.05))  # Use partial_step_size or global step_size
        else:  # Fallback to generic or default if specific not found
            current_range = threshold_config_main.get(current_range_key, [0.2, 0.8])  # Wider default range
            current_step = threshold_config_main.get(current_step_key, threshold_config_main.get('step_size', 0.05))

        best_thresh_iter, best_f1_iter = 0.5, 0.0  # Default threshold to 0.5
        thresholds_to_check_list = np.arange(current_range[0], current_range[1] + current_step, current_step)
        thresholds_to_check_list = thresholds_to_check_list[thresholds_to_check_list <= (current_range[1] + 1e-5)]

        for threshold_val_iter in thresholds_to_check_list:
            y_pred_this_class_binary = (y_proba[:, class_idx] >= threshold_val_iter).astype(int)
            y_true_this_class_binary = (y_val == class_idx).astype(int)
            f1_current_class = f1_score(y_true_this_class_binary, y_pred_this_class_binary, zero_division=0)
            if f1_current_class > best_f1_iter:
                best_f1_iter, best_thresh_iter = f1_current_class, threshold_val_iter
            elif f1_current_class == best_f1_iter and abs(threshold_val_iter - 0.5) < abs(best_thresh_iter - 0.5):
                # Prefer threshold closer to 0.5 if F1 is the same
                best_thresh_iter = threshold_val_iter

        optimal_thresholds_map[class_idx] = {'threshold': best_thresh_iter, 'f1_score': best_f1_iter,
                                             'class_name': class_name_iter}
        logger.info(
            f"  Optimal threshold for {class_name_iter}: {best_thresh_iter:.3f} (F1 for this class OvR: {best_f1_iter:.4f})")
    return optimal_thresholds_map


def apply_optimized_thresholds(model, X, thresholds_map):
    if thresholds_map is None: return model.predict(X)
    y_proba = model.predict_proba(X)
    y_pred_final = np.zeros(len(X), dtype=int)
    for i_apply in range(len(X)):
        eligible_classes_list = []
        for class_idx_apply_key_str, thresh_info_apply in thresholds_map.items():  # Iterate through map
            class_idx_apply = int(class_idx_apply_key_str)  # Ensure key is int
            if class_idx_apply < y_proba.shape[1] and y_proba[i_apply, class_idx_apply] >= thresh_info_apply[
                'threshold']:
                eligible_classes_list.append((class_idx_apply, y_proba[i_apply, class_idx_apply]))
        if eligible_classes_list:
            eligible_classes_list.sort(key=lambda item: item[1], reverse=True)  # Sort by probability desc
            y_pred_final[i_apply] = eligible_classes_list[0][0]  # Highest prob among eligible
        else:  # If no class meets its threshold, predict based on highest original probability
            y_pred_final[i_apply] = np.argmax(y_proba[i_apply])
    return y_pred_final

def save_model_artifacts(zone_key, zone_filenames_config, model, scaler, feature_importance_df, optimal_thresholds,
                         optuna_study=None):
    logger.info(f"[{zone_key}] Saving model artifacts...")
    model_path = zone_filenames_config.get('model')
    scaler_path = zone_filenames_config.get('scaler')
    importance_path = zone_filenames_config.get('importance')
    paths_to_check = [model_path, scaler_path, importance_path]
    study_path_save = None  # Initialize
    if optuna_study and model_path:  # Ensure model_path exists for deriving directory
        model_dir_for_study = os.path.dirname(model_path) if os.path.dirname(model_path) else MODEL_DIR
        study_path_save = os.path.join(model_dir_for_study, f"{zone_key}_optuna_study.pkl")
        paths_to_check.append(study_path_save)
    for p_iter in paths_to_check:
        if p_iter:  # Ensure path is not None or empty
            dir_to_make = os.path.dirname(p_iter)
            if dir_to_make: os.makedirs(dir_to_make, exist_ok=True)
    if model_path and model: joblib.dump(model, model_path); logger.info(f"Model saved: {model_path}")
    if scaler_path and scaler: joblib.dump(scaler, scaler_path); logger.info(f"Scaler saved: {scaler_path}")
    if importance_path and feature_importance_df is not None and not feature_importance_df.empty:
        feature_importance_df.to_csv(importance_path, index=False)
        logger.info(f"Feature importance saved: {importance_path}")
    if optimal_thresholds and model_path:  # Ensure model_path exists for deriving directory
        thresh_dir_for_save = os.path.dirname(model_path) if os.path.dirname(model_path) else MODEL_DIR
        thresh_path_save = os.path.join(thresh_dir_for_save, f"{zone_key}_optimal_thresholds.pkl")
        joblib.dump(optimal_thresholds, thresh_path_save)
        logger.info(f"Optimal thresholds saved: {thresh_path_save}")
    if optuna_study and study_path_save:  # study_path_save is defined if optuna_study and model_path are valid
        try:
            joblib.dump(optuna_study, study_path_save)
            logger.info(f"Optuna study saved: {study_path_save}")
        except Exception as e_save_study_helper:  # More specific exception handling
            logger.error(f"Failed to save Optuna study to {study_path_save}: {e_save_study_helper}", exc_info=True)

def log_performance_summary(zone_key_log, zone_name_disp_log, y_test_arr_log, y_pred_test_log, y_proba_test_log,
                            class_names_map_log, get_perf_targets_func_log, optimal_thresholds_used_log=False):
    from sklearn.metrics import classification_report, confusion_matrix, cohen_kappa_score, matthews_corrcoef, \
        precision_score, recall_score, roc_auc_score
    from sklearn.preprocessing import label_binarize
    num_classes_log = len(class_names_map_log)
    # Ensure target_names are correctly ordered based on numeric keys if map is {0:'A', 1:'B'}
    target_names_ordered = [class_names_map_log.get(i, str(i)) for i in sorted(class_names_map_log.keys()) if
                            isinstance(i, int)]
    if not target_names_ordered or len(
            target_names_ordered) != num_classes_log:  # Fallback if keys aren't simple integers 0..N-1
        target_names_ordered = list(class_names_map_log.values())

    report_str_log = classification_report(y_test_arr_log, y_pred_test_log, target_names=target_names_ordered,
                                           zero_division=0, labels=list(range(num_classes_log)))
    logger.info(
        f"[{zone_name_disp_log}] Test Set Classification Report {'(with Optimized Thresholds)' if optimal_thresholds_used_log else ''}:\n{report_str_log}")
    cm_log = confusion_matrix(y_test_arr_log, y_pred_test_log, labels=list(range(num_classes_log)))
    logger.info(f"[{zone_name_disp_log}] Confusion Matrix (rows=True, cols=Pred):\n{cm_log}")
    bal_acc_log = balanced_accuracy_score(y_test_arr_log, y_pred_test_log)
    kappa_log = cohen_kappa_score(y_test_arr_log, y_pred_test_log)
    logger.info(f"[{zone_name_disp_log}] ===== PERFORMANCE SUMMARY =====")
    logger.info(f"[{zone_name_disp_log}] Overall Accuracy: {(y_pred_test_log == y_test_arr_log).mean():.4f}")
    logger.info(f"[{zone_name_disp_log}] Balanced Accuracy: {bal_acc_log:.4f}")
    logger.info(f"[{zone_name_disp_log}] Cohen's Kappa: {kappa_log:.4f}")
    try:
        mcc_log = matthews_corrcoef(y_test_arr_log, y_pred_test_log)
        logger.info(f"[{zone_name_disp_log}] Matthews Correlation Coefficient: {mcc_log:.4f}")
    except Exception:
        pass
    f1_per_log = f1_score(y_test_arr_log, y_pred_test_log, average=None, labels=list(range(num_classes_log)),
                          zero_division=0)
    prec_per_log = precision_score(y_test_arr_log, y_pred_test_log, average=None, labels=list(range(num_classes_log)),
                                   zero_division=0)
    rec_per_log = recall_score(y_test_arr_log, y_pred_test_log, average=None, labels=list(range(num_classes_log)),
                               zero_division=0)
    logger.info(f"[{zone_name_disp_log}] Per-class metrics:")
    for idx_log in range(num_classes_log):
        cn_log = class_names_map_log.get(idx_log, f"Class_{idx_log}")
        f1_v_log = f1_per_log[idx_log] if idx_log < len(f1_per_log) else np.nan
        p_v_log = prec_per_log[idx_log] if idx_log < len(prec_per_log) else np.nan
        r_v_log = rec_per_log[idx_log] if idx_log < len(rec_per_log) else np.nan
        logger.info(f"  {cn_log}: F1={f1_v_log:.4f}, Precision={p_v_log:.4f}, Recall={r_v_log:.4f}")
    if get_perf_targets_func_log:
        try:
            targets_log = get_perf_targets_func_log(zone_key_log)  # zone_key_log is the string key e.g. 'lower'
            logger.info(f"[{zone_name_disp_log}] Performance vs Targets:")
            ba_target_val = targets_log.get('balanced_accuracy', 'N/A')
            ba_target_str_log = f"{float(ba_target_val):.2f}" if isinstance(ba_target_val, (int, float)) else str(
                ba_target_val)
            logger.info(f"  Balanced Accuracy: {bal_acc_log:.4f} (target: {ba_target_str_log})")
            partial_idx_for_target = -1
            for k_map, v_map in class_names_map_log.items():
                if str(v_map).lower() == 'partial': partial_idx_for_target = int(k_map); break
            if partial_idx_for_target != -1 and partial_idx_for_target < num_classes_log and partial_idx_for_target < len(
                    f1_per_log):
                f1_partial_actual_log = f1_per_log[partial_idx_for_target]
                f1p_target_val = targets_log.get('f1_partial', 'N/A')
                f1p_target_str_log = f"{float(f1p_target_val):.2f}" if isinstance(f1p_target_val,
                                                                                  (int, float)) else str(f1p_target_val)
                logger.info(
                    f"  F1 for '{class_names_map_log.get(partial_idx_for_target, 'Partial')}': {f1_partial_actual_log:.4f} (target: {f1p_target_str_log})")
        except Exception as e_perf_target_log:
            logger.debug(f"[{zone_name_disp_log}] Could not check performance targets: {e_perf_target_log}")
    try:
        y_test_bin_log = label_binarize(y_test_arr_log, classes=list(range(num_classes_log)))
        if num_classes_log > 1 and y_test_bin_log.ndim > 1 and y_test_bin_log.shape[1] == num_classes_log and \
                y_proba_test_log.shape[1] == num_classes_log:
            for i_cls_log in range(num_classes_log):
                if np.sum(y_test_bin_log[:, i_cls_log]) > 0 and len(
                        np.unique(y_test_bin_log[:, i_cls_log])) > 1:  # Check for positive samples and variance
                    auc_cls_log = roc_auc_score(y_test_bin_log[:, i_cls_log], y_proba_test_log[:, i_cls_log])
                    logger.info(
                        f"[{zone_name_disp_log}] AUC for class {class_names_map_log.get(i_cls_log, str(i_cls_log))}: {auc_cls_log:.4f}")
                else:
                    logger.debug(
                        f"[{zone_name_disp_log}] Skipping AUC for class {class_names_map_log.get(i_cls_log, str(i_cls_log))}: insufficient variance or no positive samples in test set for this class.")
        elif num_classes_log == 2 and y_test_bin_log.ndim == 1:  # Binary case, label_binarize might return 1D
            if len(np.unique(y_test_arr_log)) > 1:  # Check original y_test for variance
                # For binary, roc_auc_score uses probabilities of the positive class (class 1)
                auc_binary_log = roc_auc_score(y_test_arr_log, y_proba_test_log[:, 1])
                logger.info(f"[{zone_name_disp_log}] AUC (binary): {auc_binary_log:.4f}")
            else:
                logger.info(
                    f"[{zone_name_disp_log}] Binary classification but only one class present in y_test; AUC not applicable.")
        elif num_classes_log <= 1 or (y_test_bin_log.ndim == 1 and len(np.unique(y_test_bin_log)) <= 1):
            logger.info(
                f"[{zone_name_disp_log}] Only one class effectively present in test set; multi-class OvR AUC not applicable. Unique binarized labels: {np.unique(y_test_bin_log)}")
    except ValueError as ve_auc:
        logger.warning(f"[{zone_name_disp_log}] Could not calculate AUC: {ve_auc}")  # Catches "Only one class present"
    except Exception as e_auc_log:
        logger.warning(f"[{zone_name_disp_log}] Could not calculate AUC due to other error: {e_auc_log}")


def analyze_optuna_study(zone_key_an, zone_name_disp_an, zone_config_an):
    if not OPTUNA_AVAILABLE: logger.info(f"[{zone_name_disp_an}] Optuna not available for study analysis."); return
    filenames_an = zone_config_an.get('filenames', {})
    model_file_path_an = filenames_an.get('model', os.path.join(MODEL_DIR, f"{zone_key_an}_model.pkl"))
    model_base_dir_an = MODEL_DIR
    if model_file_path_an:
        dir_part_an = os.path.dirname(model_file_path_an)
        if dir_part_an: model_base_dir_an = dir_part_an
    study_path_an = os.path.join(model_base_dir_an, f"{zone_key_an}_optuna_study.pkl")
    if os.path.exists(study_path_an):
        try:
            study_an = joblib.load(study_path_an)
            logger.info(f"\n[{zone_name_disp_an}] Optuna Study Analysis:")
            completed_an = [t for t in study_an.trials if t.state == optuna.trial.TrialState.COMPLETE]
            pruned_an = [t for t in study_an.trials if t.state == optuna.trial.TrialState.PRUNED]
            failed_an = [t for t in study_an.trials if t.state == optuna.trial.TrialState.FAIL]
            logger.info(
                f"  - Total trials: {len(study_an.trials)}, Completed: {len(completed_an)}, Pruned: {len(pruned_an)}, Failed: {len(failed_an)}")
            if study_an.best_trial:
                score_str_an = f"{study_an.best_value:.4f}" if study_an.best_value is not None else "N/A"
                logger.info(f"  - Best trial: #{study_an.best_trial.number}, Score: {score_str_an}")
                logger.info(f"  - Best params (for XGB base): {study_an.best_params}")
            else:
                logger.info("  - No best trial found (e.g., all trials failed or were pruned).")
            plot_history_an = ADVANCED_TRAINING_CONFIG.get('monitoring', {}).get('plot_optimization_history', False)
            if plot_history_an and len(completed_an) > 0:
                try:
                    from optuna.visualization import plot_optimization_history, plot_param_importances
                    plots_dir_an = os.path.join(ANALYSIS_DIR, zone_key_an, 'optuna_plots')
                    os.makedirs(plots_dir_an, exist_ok=True)
                    fig_hist_an = plot_optimization_history(study_an)
                    hist_path_an = os.path.join(plots_dir_an, f"{zone_key_an}_optuna_history.png")
                    if fig_hist_an: fig_hist_an.write_image(hist_path_an); logger.info(
                        f"Saved Optuna history plot: {hist_path_an}")
                    if len(completed_an) > 1 and study_an.best_trial:  # param_importances needs multiple completed trials
                        fig_imp_an = plot_param_importances(study_an)
                        imp_path_an = os.path.join(plots_dir_an, f"{zone_key_an}_optuna_param_importance.png")
                        if fig_imp_an: fig_imp_an.write_image(imp_path_an); logger.info(
                            f"Saved Optuna importance plot: {imp_path_an}")
                except ImportError:
                    logger.warning(
                        "Cannot generate Optuna plots: optuna.visualization or dependencies (plotly, kaleido) missing.")
                except Exception as e_plot_an:
                    logger.warning(f"Could not generate Optuna plots for {zone_name_disp_an}: {e_plot_an}")
        except Exception as e_load_study_an:
            logger.error(
                f"Could not load/analyze Optuna study for {zone_name_disp_an} from {study_path_an}: {e_load_study_an}",
                exc_info=True)
    else:
        logger.info(f"[{zone_name_disp_an}] Optuna study file not found at {study_path_an}")

def calculate_class_weights(y_true, strategy='balanced', class_map_config=None):
    if isinstance(class_map_config, dict) and class_map_config:
        return {int(k): float(v) for k, v in class_map_config.items() if str(k).isdigit()}
    unique_classes_cw, counts_cw = np.unique(y_true, return_counts=True)
    if not unique_classes_cw.size: return {}
    weights_cw = {}
    if strategy == 'balanced':
        total_samples_cw = len(y_true)
        weights_cw = {cls: total_samples_cw / (len(unique_classes_cw) * count_val) if count_val > 0 else 1.0
                      for cls, count_val in zip(unique_classes_cw, counts_cw)}
    elif strategy == 'balanced_subsample':
        max_count_cw = max(counts_cw) if counts_cw.size > 0 else 1
        weights_cw = {cls: max_count_cw / count_val if count_val > 0 else 1.0
                      for cls, count_val in zip(unique_classes_cw, counts_cw)}
    else:
        weights_cw = {cls: 1.0 for cls in unique_classes_cw}
    return {int(k): float(v) for k, v in weights_cw.items()}


def calculate_class_weights_for_model(y_true_cw_model, class_weight_config_cw_model):
    if isinstance(class_weight_config_cw_model, str):  # e.g. "balanced"
        return calculate_class_weights(y_true_cw_model, strategy=class_weight_config_cw_model)
    # Assumes class_weight_config_cw_model is a dictionary or None
    return calculate_class_weights(y_true_cw_model, class_map_config=class_weight_config_cw_model)

def get_class_specific_info(class_names_map_csi):
    partial_class_index_csi = -1
    if not isinstance(class_names_map_csi, dict):
        logger.error(
            f"get_class_specific_info: class_names_map_csi is not a dictionary: {type(class_names_map_csi)}. Using default for partial index.")
        return {'partial_class_index': -1}
    for class_val_csi, class_name_str_csi in class_names_map_csi.items():
        if isinstance(class_name_str_csi, str) and class_name_str_csi.lower() == 'partial':
            try:
                partial_class_index_csi = int(class_val_csi);
                break
            except ValueError:
                logger.warning(f"Could not convert partial class key '{class_val_csi}' to int for class_names_map_csi.")
    return {'partial_class_index': partial_class_index_csi}


def get_common_feature_importance_df(model_fi, feature_names_fi, zone_name_fi="Zone"):
    from sklearn.calibration import CalibratedClassifierCV
    from sklearn.ensemble import VotingClassifier
    import xgboost as xgb
    fi_df = pd.DataFrame()
    estimator_to_inspect = model_fi
    if isinstance(estimator_to_inspect, CalibratedClassifierCV):
        if hasattr(estimator_to_inspect,
                   'estimator_') and estimator_to_inspect.estimator_ is not None:  # sklearn >= 1.0 style
            estimator_to_inspect = estimator_to_inspect.estimator_
            logger.debug(f"[{zone_name_fi}] FI: Unwrapped CalibratedClassifierCV via estimator_.")
        elif hasattr(estimator_to_inspect,
                     'calibrated_classifiers_') and estimator_to_inspect.calibrated_classifiers_:  # Older or CV'd
            # If CalibratedClassifierCV used cv=int, it has multiple base estimators.
            # We'd ideally average importances or take from the first.
            # If estimator was prefit, calibrated_classifiers_[0].estimator might be the one.
            base_calib_est = estimator_to_inspect.calibrated_classifiers_[0]
            if hasattr(base_calib_est, 'estimator_') and base_calib_est.estimator_ is not None:
                estimator_to_inspect = base_calib_est.estimator_
            elif hasattr(base_calib_est,
                         'estimator') and base_calib_est.estimator is not None:  # Fallback for slightly older
                estimator_to_inspect = base_calib_est.estimator
            elif hasattr(base_calib_est, 'base_estimator'):  # Older versions
                estimator_to_inspect = base_calib_est.base_estimator
            logger.debug(f"[{zone_name_fi}] FI: Unwrapped CalibratedClassifierCV via calibrated_classifiers_.")
        else:
            logger.warning(
                f"[{zone_name_fi}] FI: Could not fully unwrap CalibratedClassifierCV of type {type(model_fi)}.")

    xgb_model_to_use = None
    if isinstance(estimator_to_inspect, VotingClassifier):
        logger.debug(f"[{zone_name_fi}] FI: Inspected estimator is VotingClassifier.")
        if hasattr(estimator_to_inspect, 'named_estimators_') and 'xgb' in estimator_to_inspect.named_estimators_:
            xgb_model_to_use = estimator_to_inspect.named_estimators_['xgb']
            logger.debug(f"[{zone_name_fi}] FI: Found 'xgb' in named_estimators_ of VotingClassifier.")
        elif hasattr(estimator_to_inspect, 'estimators_') and estimator_to_inspect.estimators_:
            for est_instance_fi in estimator_to_inspect.estimators_:
                if isinstance(est_instance_fi, xgb.XGBClassifier):
                    xgb_model_to_use = est_instance_fi;
                    logger.debug(f"[{zone_name_fi}] FI: Found XGBClassifier instance in VotingClassifier.estimators_.");
                    break
        if not xgb_model_to_use and hasattr(estimator_to_inspect,
                                            'estimators'):  # Original list of (name, est_template) tuples
            for name, est_template in estimator_to_inspect.estimators:
                if name == 'xgb' and isinstance(est_template, xgb.XGBClassifier):
                    logger.warning(
                        f"[{zone_name_fi}] FI: Using XGB template from VotingClassifier.estimators list. This might be an unfitted template if VotingClassifier was not yet fit or if 'xgb' was pre-fitted and not in estimators_.");
                    xgb_model_to_use = est_template;
                    break
    elif isinstance(estimator_to_inspect, xgb.XGBClassifier):
        logger.debug(f"[{zone_name_fi}] FI: Inspected estimator is directly XGBClassifier.")
        xgb_model_to_use = estimator_to_inspect

    if xgb_model_to_use and hasattr(xgb_model_to_use, 'feature_importances_'):
        importances_val = xgb_model_to_use.feature_importances_
        if feature_names_fi and len(importances_val) == len(feature_names_fi):
            fi_df = pd.DataFrame({'feature': feature_names_fi, 'importance': importances_val}).sort_values('importance',
                                                                                                           ascending=False)
            logger.info(f"[{zone_name_fi}] Successfully extracted feature importances from XGBoost model component.")
        else:
            logger.warning(
                f"[{zone_name_fi}] FI: Length mismatch for XGB feature importance: FI len {len(importances_val)}, feature_names len {len(feature_names_fi if feature_names_fi else [])}.")
    elif fi_df.empty and hasattr(estimator_to_inspect, 'feature_importances_'):
        importances_val_other = estimator_to_inspect.feature_importances_
        if feature_names_fi and len(importances_val_other) == len(feature_names_fi):
            fi_df = pd.DataFrame({'feature': feature_names_fi, 'importance': importances_val_other}).sort_values(
                'importance', ascending=False)
            logger.warning(
                f"[{zone_name_fi}] FI: Extracted FI from general estimator of type: {type(estimator_to_inspect)} as XGB was not found or had issues.")
        else:
            logger.warning(
                f"[{zone_name_fi}] FI: General estimator {type(estimator_to_inspect)} had FI, but length mismatch with feature names.")

    if fi_df.empty: logger.error(
        f"[{zone_name_fi}] FAILED to extract any feature importances from model {type(model_fi)}.")
    return fi_df