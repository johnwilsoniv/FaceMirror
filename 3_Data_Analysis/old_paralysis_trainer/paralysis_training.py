# paralysis_training.py (v5 - SMOTE Strategy & Calibration Fixes)

import pandas as pd
import numpy as np
import logging
import os
import joblib
import importlib
from sklearn.model_selection import train_test_split, StratifiedKFold, KFold
from sklearn.metrics import (
    classification_report, confusion_matrix, f1_score,
    balanced_accuracy_score, cohen_kappa_score, roc_auc_score
)
from sklearn.preprocessing import StandardScaler, label_binarize
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import VotingClassifier, RandomForestClassifier
import xgboost as xgb
from imblearn.over_sampling import SMOTE, BorderlineSMOTE, ADASYN
from imblearn.combine import SMOTEENN  # SMOTETomek was not used, SMOTEENN added from config

# Import Optuna with enhanced features
try:
    import optuna
    from optuna.samplers import TPESampler
    from optuna.pruners import HyperbandPruner, MedianPruner, PercentilePruner

    OPTUNA_AVAILABLE = True
    optuna.logging.set_verbosity(optuna.logging.WARNING)
except ImportError:
    OPTUNA_AVAILABLE = False
    TPESampler = None
    HyperbandPruner = None
    MedianPruner = None
    PercentilePruner = None

# Import central config and utils
try:
    from paralysis_config import ZONE_CONFIG, LOGGING_CONFIG, INPUT_FILES, CLASS_NAMES, ADVANCED_TRAINING_CONFIG
except ImportError as e:
    print(f"CRITICAL: Failed to import from paralysis_config.py - {e}")
    ZONE_CONFIG = {}
    LOGGING_CONFIG = {'level': 'INFO', 'format': '%(asctime)s - %(levelname)s - %(name)s - %(message)s'}
    INPUT_FILES = {}
    CLASS_NAMES = {0: 'None', 1: 'Partial', 2: 'Complete'}  # Basic default
    ADVANCED_TRAINING_CONFIG = {}

try:
    from paralysis_utils import (
        generate_review_candidates,
        calculate_entropy, calculate_margin,
        PARALYSIS_MAP
    )

    UTILS_LOADED = True
except ImportError as e:
    print(f"CRITICAL: Failed to import from paralysis_utils.py - {e}")
    UTILS_LOADED = False


    def generate_review_candidates(*args, **kwargs):
        return pd.DataFrame()


    def calculate_entropy(*args, **kwargs):
        return 0.0


    def calculate_margin(*args, **kwargs):
        return 1.0


    PARALYSIS_MAP = {0: 'None', 1: 'Partial', 2: 'Complete'}

logger = logging.getLogger(__name__)


def _setup_logging(log_file, level):
    """Setup logging configuration"""
    log_level = getattr(logging, level.upper(), logging.INFO)
    log_format = LOGGING_CONFIG.get('format', '%(asctime)s - %(levelname)s - %(name)s - %(message)s')
    log_dir = os.path.dirname(log_file)
    if log_dir:
        os.makedirs(log_dir, exist_ok=True)
    root_logger = logging.getLogger()
    if root_logger.hasHandlers():
        root_logger.handlers.clear()
    logging.basicConfig(level=log_level, format=log_format,
                        handlers=[logging.StreamHandler(), logging.FileHandler(log_file, mode='w')],
                        force=True)


def _calculate_class_weights(y, weight_strategy='balanced'):
    """Calculate class weights based on class distribution"""
    unique_classes, counts = np.unique(y, return_counts=True)
    total_samples = len(y)

    if not unique_classes.size:  # Handle empty y
        return {}

    if weight_strategy == 'balanced':
        weights = {}
        for cls, count in zip(unique_classes, counts):
            weights[cls] = total_samples / (len(unique_classes) * count) if count > 0 else 1.0
    elif weight_strategy == 'balanced_subsample':
        weights = {}
        max_count = max(counts) if counts.size > 0 else 1
        for cls, count in zip(unique_classes, counts):
            weights[cls] = max_count / count if count > 0 else 1.0
    elif isinstance(weight_strategy, dict):
        weights = weight_strategy
    else:  # Default to equal weights if strategy unknown
        weights = {cls: 1.0 for cls in unique_classes}

    return weights


def _get_sampling_strategy(y, configured_strategy, min_samples_per_class_for_recalc=50, majority_class_label=0):
    """
    Get sampling strategy for SMOTE variants.
    Prioritizes dictionary if provided.
    If 'auto', returns 'auto'.
    If string (e.g. 'custom_recalc'), uses min_samples_per_class_for_recalc.
    `majority_class_label` is used if the dictionary strategy has 'auto' for majority.
    """
    if isinstance(configured_strategy, dict):
        # Resolve 'auto' for majority class if present in dict
        # Ensure all keys in configured_strategy are present in y
        # This part might need adjustment if a class in dict is NOT in y_fold_train
        # For now, we assume dict keys are a subset of or equal to y's classes
        resolved_strategy = {}
        unique_y, counts_y = np.unique(y, return_counts=True)
        y_dist = dict(zip(unique_y, counts_y))

        # Find true majority count if 'auto' is used for it
        majority_count = y_dist.get(majority_class_label, max(counts_y) if counts_y.size > 0 else 0)

        for cls, target in configured_strategy.items():
            if cls not in y_dist:  # Class in strategy not in current data fold
                logger.warning(
                    f"SMOTE strategy class {cls} not in current y-fold data. Skipping this class for strategy.")
                continue
            if target == 'auto':
                # Typically, 'auto' for majority means keep as is, for minority means oversample to majority.
                # Here, if it's majority, we set it to its current count (no change from SMOTE's perspective)
                # If it's minority and 'auto', SMOTE will handle it. Let's be explicit.
                resolved_strategy[cls] = y_dist[cls] if cls == majority_class_label else majority_count
            elif isinstance(target, (int, float)):
                # Ensure target isn't less than current count for that class
                resolved_strategy[cls] = max(int(target), y_dist[cls])
            else:
                logger.warning(f"Unsupported target '{target}' for class {cls} in SMOTE dict. Using current count.")
                resolved_strategy[cls] = y_dist[cls]
        logger.debug(f"Resolved dictionary sampling strategy: {resolved_strategy}")
        return resolved_strategy

    if configured_strategy == 'auto':
        return 'auto'  # SMOTE handles this: oversamples all minority to majority

    # If configured_strategy is a string like 'custom_behavior_flag' or similar,
    # indicating a rule-based calculation is desired based on min_samples_per_class_for_recalc.
    unique_classes, counts = np.unique(y, return_counts=True)
    if not counts.size: return 'auto'  # Or {} if preferred for no change

    sampling_dict = {}
    max_count_overall = max(counts)  # Max count in the current fold/dataset

    for cls, count in zip(unique_classes, counts):
        target_samples = count  # Default to current count
        if count < min_samples_per_class_for_recalc:
            # Bring up to min_samples_per_class_for_recalc, but not more than 50% of overall majority (or overall majority itself)
            target_samples = max(count, min(min_samples_per_class_for_recalc, int(max_count_overall * 0.5)))
        elif count < max_count_overall * 0.5:  # If not caught by above, but still < 50% of majority
            target_samples = max(count, int(max_count_overall * 0.5))

        if target_samples > count:  # Only add to dict if oversampling is needed for this class
            sampling_dict[cls] = target_samples
        # else: # Optionally, explicitly state current count if no oversampling
        #     sampling_dict[cls] = count

    logger.debug(f"Recalculated sampling strategy: {sampling_dict if sampling_dict else 'auto'}")
    return sampling_dict if sampling_dict else 'auto'


def _get_optuna_suggestion(trial, param_name, param_config):
    """Get parameter suggestion from Optuna trial"""
    param_type = param_config[0]

    if param_type == 'float':
        low, high = param_config[1], param_config[2]
        options = param_config[3] if len(param_config) > 3 else {}
        return trial.suggest_float(param_name, low, high, **options)
    elif param_type == 'int':
        low, high = param_config[1], param_config[2]
        options = param_config[3] if len(param_config) > 3 else {}
        return trial.suggest_int(param_name, low, high, **options)
    elif param_type == 'categorical':
        choices = param_config[1]
        return trial.suggest_categorical(param_name, choices)
    else:
        raise ValueError(f"Unsupported parameter type: {param_type}")


def _create_enhanced_optuna_objective(X_train_scaled, y_train,
                                      smote_config, class_names_map,
                                      optuna_config, model_params_config,
                                      random_state, zone_name):
    """Enhanced Optuna objective with better handling of class imbalance"""

    if not OPTUNA_AVAILABLE:
        raise ImportError("Optuna is not available")

    num_classes = len(class_names_map)
    param_distributions = optuna_config.get('param_distributions', {})
    cv_folds = optuna_config.get('cv_folds', 3)
    scoring_metric = optuna_config.get('scoring', 'f1_macro')
    optuna_early_stopping_rounds_val = optuna_config.get('optuna_early_stopping_rounds')  # Renamed to avoid clash

    # Get sampling configuration
    smote_enabled = smote_config.get('enabled', False)
    smote_per_fold = smote_config.get('apply_per_fold_in_tuning', False) and smote_enabled
    k_neighbors_config = smote_config.get('k_neighbors', 5)
    configured_sampling_strategy = smote_config.get('sampling_strategy', 'auto')
    min_samples_for_smote_recalc = smote_config.get('min_samples_per_class', 50)
    smote_variant = smote_config.get('variant', 'regular')

    # Calculate base class weights (used if not using SMOTE or if SMOTE fails)
    # This uses the original y_train distribution
    base_class_weights = _calculate_class_weights(y_train, 'balanced_subsample')

    def objective(trial):
        xgb_params = {
            'objective': model_params_config.get('objective', 'multi:softprob'),
            'num_class': num_classes,
            'eval_metric': 'mlogloss',  # Standard for multi:softprob
            'random_state': random_state,
        }

        for param_name, param_config_list in param_distributions.items():
            if param_name != 'scale_pos_weight':
                xgb_params[param_name] = _get_optuna_suggestion(trial, param_name, param_config_list)

        if 'scale_pos_weight' in param_distributions and num_classes == 2:
            xgb_params['scale_pos_weight'] = _get_optuna_suggestion(
                trial, 'scale_pos_weight', param_distributions['scale_pos_weight']
            )

        if optuna_early_stopping_rounds_val:  # Use the renamed variable
            xgb_params['early_stopping_rounds'] = optuna_early_stopping_rounds_val

        fold_scores = []
        diversity_scores = []
        per_class_f1_scores_agg = {i: [] for i in range(num_classes)}  # Renamed

        unique_labels_overall, counts_overall = np.unique(y_train, return_counts=True)
        min_samples_for_stratified = cv_folds

        if len(unique_labels_overall) < 2 or any(c < min_samples_for_stratified for c in counts_overall):
            cv = KFold(n_splits=cv_folds, shuffle=True, random_state=random_state)
            logger.warning(
                f"[{zone_name} Trial {trial.number}] Using KFold due to class distribution in y_train for CV split.")
        else:
            cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=random_state)

        for fold_idx, (train_idx, val_idx) in enumerate(cv.split(X_train_scaled, y_train)):
            X_fold_train, y_fold_train = X_train_scaled[train_idx], y_train[train_idx]
            X_fold_val, y_fold_val = X_train_scaled[val_idx], y_train[val_idx]

            y_fold_train_original_dist = dict(zip(*np.unique(y_fold_train, return_counts=True)))  # For logging

            # Dynamic class weights for the current fold before SMOTE
            # These weights are applied to the potentially SMOTEd data
            current_fold_class_weights = _calculate_class_weights(y_fold_train, 'balanced_subsample')

            if smote_per_fold:
                try:
                    unique_fold, counts_fold = np.unique(y_fold_train, return_counts=True)
                    min_class_count_in_fold = min(counts_fold) if counts_fold.size > 0 else 0

                    # k_neighbors must be less than the number of samples in the smallest class
                    actual_k = min(k_neighbors_config,
                                   min_class_count_in_fold - 1) if min_class_count_in_fold > 1 else 1

                    if actual_k >= 1 and len(unique_fold) > 1:  # SMOTE possible
                        # Get the actual sampling strategy for this fold
                        current_fold_sampling_strategy = _get_sampling_strategy(
                            y_fold_train, configured_sampling_strategy, min_samples_for_smote_recalc
                        )
                        logger.debug(
                            f"[{zone_name} Trial {trial.number} Fold {fold_idx}] SMOTE params: k={actual_k}, strategy={current_fold_sampling_strategy}, variant={smote_variant}")

                        if smote_variant == 'borderline':
                            smote_obj = BorderlineSMOTE(k_neighbors=actual_k, random_state=random_state,
                                                        sampling_strategy=current_fold_sampling_strategy)
                        elif smote_variant == 'adasyn':
                            smote_obj = ADASYN(n_neighbors=actual_k, random_state=random_state,
                                               sampling_strategy=current_fold_sampling_strategy)
                        # elif smote_variant == 'smoteenn': # Not used here, handled globally
                        #     smote_obj = SMOTEENN(smote=SMOTE(k_neighbors=actual_k, random_state=random_state, sampling_strategy=current_fold_sampling_strategy), random_state=random_state)
                        else:  # regular SMOTE
                            smote_obj = SMOTE(k_neighbors=actual_k, random_state=random_state,
                                              sampling_strategy=current_fold_sampling_strategy)

                        X_fold_train_smoted, y_fold_train_smoted = smote_obj.fit_resample(X_fold_train, y_fold_train)
                        # After SMOTE, class weights might need to be re-evaluated or set to 1 if classes are balanced
                        # For XGBoost, if classes are balanced by SMOTE, sample_weight might be less critical or even counterproductive
                        # Let's recalculate weights based on the SMOTEd distribution for this fold.
                        current_fold_class_weights = _calculate_class_weights(y_fold_train_smoted, 'balanced_subsample')
                        X_fold_train, y_fold_train = X_fold_train_smoted, y_fold_train_smoted  # Update for training

                        y_fold_train_smoted_dist = dict(zip(*np.unique(y_fold_train, return_counts=True)))
                        logger.debug(
                            f"[{zone_name} Trial {trial.number} Fold {fold_idx}] y_fold_train dist original: {y_fold_train_original_dist}, after SMOTE: {y_fold_train_smoted_dist}")

                    else:  # SMOTE not possible or not beneficial
                        logger.debug(
                            f"[{zone_name} Trial {trial.number} Fold {fold_idx}] SMOTE not applied (actual_k={actual_k}, unique_classes={len(unique_fold)}). Using original fold data and weights.")

                except Exception as e:
                    logger.warning(
                        f"[{zone_name} Trial {trial.number} Fold {fold_idx}] SMOTE failed: {e}. Using original fold data and weights.")
                    # current_fold_class_weights remains based on original y_fold_train

            # Calculate sample weights using the determined class_weights for this fold (either from original or SMOTEd y_fold_train)
            sample_weights_for_fit = np.array([current_fold_class_weights.get(y_val, 1.0) for y_val in y_fold_train])

            model = xgb.XGBClassifier(**xgb_params)
            fit_params = {'sample_weight': sample_weights_for_fit, 'verbose': False}

            if xgb_params.get('early_stopping_rounds') and len(np.unique(y_fold_val)) > 1:
                fit_params['eval_set'] = [(X_fold_val, y_fold_val)]

            try:
                model.fit(X_fold_train, y_fold_train, **fit_params)
                y_pred_fold = model.predict(X_fold_val)
                # y_proba_fold = model.predict_proba(X_fold_val) # Not used for main score

                f1_w = f1_score(y_fold_val, y_pred_fold, average='weighted', zero_division=0)
                f1_m = f1_score(y_fold_val, y_pred_fold, average='macro', zero_division=0)
                f1_pc = f1_score(y_fold_val, y_pred_fold, average=None, zero_division=0)

                for i_cls, f1_val_cls in enumerate(f1_pc):
                    if i_cls in per_class_f1_scores_agg:
                        per_class_f1_scores_agg[i_cls].append(f1_val_cls)

                unique_preds_fold = len(np.unique(y_pred_fold))
                unique_actual_fold = len(np.unique(y_fold_val))
                div_penalty_fold = unique_preds_fold / max(unique_actual_fold, 1)
                bal_acc_fold = balanced_accuracy_score(y_fold_val, y_pred_fold)

                current_score = 0.0
                if scoring_metric == 'composite':
                    min_class_f1_fold = np.min(f1_pc[f1_pc > 0]) if any(f1_pc > 0) else 0
                    current_score = (
                            0.25 * f1_w + 0.25 * f1_m + 0.20 * bal_acc_fold +
                            0.15 * min_class_f1_fold + 0.15 * div_penalty_fold
                    )
                elif scoring_metric == 'f1_partial_class' and 1 < len(f1_pc):  # Assuming class 1 is 'Partial'
                    current_score = f1_pc[1]
                elif scoring_metric == 'f1_macro':
                    current_score = f1_m
                elif scoring_metric == 'f1_weighted':
                    current_score = f1_w
                elif scoring_metric == 'balanced_accuracy':
                    current_score = bal_acc_fold
                else:
                    current_score = f1_m  # Default

                fold_scores.append(current_score)
                diversity_scores.append(div_penalty_fold)

                trial.report(current_score, fold_idx)
                if trial.should_prune():
                    raise optuna.TrialPruned()

            except optuna.TrialPruned:
                raise
            except Exception as e:
                logger.error(f"[{zone_name} Trial {trial.number} Fold {fold_idx}] Error in objective: {e}",
                             exc_info=True)
                fold_scores.append(0.0)  # Penalize error
                diversity_scores.append(0.0)

        avg_score_trial = np.mean(fold_scores) if fold_scores else 0.0
        avg_diversity_trial = np.mean(diversity_scores) if diversity_scores else 0.0

        # Log per-class F1 means for this trial
        avg_per_class_f1_log = {cls: np.mean(scores) if scores else 0.0 for cls, scores in
                                per_class_f1_scores_agg.items()}
        logger.info(f"[{zone_name} Trial {trial.number}] Score: {avg_score_trial:.4f}, "
                    f"Diversity: {avg_diversity_trial:.4f}, Avg Per-class F1: {avg_per_class_f1_log}")

        if avg_diversity_trial < 0.5 and num_classes > 1:  # Heavy penalty if only one class predicted
            avg_score_trial *= 0.5
            logger.warning(
                f"[{zone_name} Trial {trial.number}] Low diversity penalty applied. New score: {avg_score_trial:.4f}")

        return avg_score_trial

    return objective


def train_enhanced_model(zone, X_train_df, y_train_arr, X_test_df, y_test_arr,
                         feature_names, config, calibration_split_size=0.2):
    """Enhanced model training with SMOTE strategy fix, calibration fix, and feature importance retrieval fix."""

    class_names_map = PARALYSIS_MAP
    num_classes = len(class_names_map)
    zone_name = config.get('name', zone.capitalize() + ' Face')
    training_params = config.get('training', {})
    smote_config = training_params.get('smote', {})
    tuning_config = training_params.get('hyperparameter_tuning', {})
    model_params = training_params.get('model_params', {})
    random_state = training_params.get('random_state', 42)
    adv_calib_cfg = ADVANCED_TRAINING_CONFIG.get('calibration', {})

    logger.info(f"[{zone_name}] Starting enhanced model training...")

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_df)
    X_test_scaled = scaler.transform(X_test_df)

    unique_train_overall, counts_train_overall = np.unique(y_train_arr, return_counts=True)
    logger.info(
        f"[{zone_name}] Initial Training class distribution: {dict(zip(unique_train_overall, counts_train_overall))}")

    base_class_weights_for_final_model = _calculate_class_weights(y_train_arr, 'balanced_subsample')
    logger.info(f"[{zone_name}] Base class weights for final model (pre-SMOTE): {base_class_weights_for_final_model}")

    X_train_to_tune = X_train_scaled.copy()
    y_train_to_tune = y_train_arr.copy()

    best_params_from_tuning = model_params.copy()

    if tuning_config.get('enabled', False) and tuning_config.get('method') == 'optuna':
        if not OPTUNA_AVAILABLE:
            logger.error(f"[{zone_name}] Optuna not available, using default parameters from config.")
        else:
            logger.info(f"[{zone_name}] Starting Optuna hyperparameter optimization...")
            optuna_cfg = tuning_config.get('optuna', {})
            n_trials = optuna_cfg.get('n_trials', 100)
            sampler_type = optuna_cfg.get('sampler', 'TPESampler')
            pruner_type = optuna_cfg.get('pruner', 'MedianPruner')

            sampler = TPESampler(seed=random_state) if sampler_type == 'TPESampler' and TPESampler else None
            pruner_map = {
                'HyperbandPruner': HyperbandPruner() if HyperbandPruner else None,
                'MedianPruner': MedianPruner() if MedianPruner else None,
                'PercentilePruner': PercentilePruner(percentile=25.0) if PercentilePruner else None,
            }
            pruner = pruner_map.get(pruner_type)

            study = optuna.create_study(
                direction=optuna_cfg.get('direction', 'maximize'),
                sampler=sampler, pruner=pruner, study_name=f"{zone_name}_optimization"
            )

            objective_fn = _create_enhanced_optuna_objective(
                X_train_to_tune, y_train_to_tune,
                smote_config, class_names_map,
                optuna_cfg, model_params,
                random_state, zone_name
            )

            def optuna_callback(study, trial):
                if trial.number > 0 and trial.number % 10 == 0:
                    logger.info(f"[{zone_name}] Optuna Trial {trial.number} - Current Best: {study.best_value:.4f}")

            study.optimize(objective_fn, n_trials=n_trials, callbacks=[optuna_callback] if n_trials > 20 else None,
                           n_jobs=1)

            logger.info(f"[{zone_name}] Optuna Best trial: {study.best_trial.number}, Value: {study.best_value:.4f}")
            logger.info(f"[{zone_name}] Optuna Best params: {study.best_params}")
            best_params_from_tuning.update(study.best_params)

            model_save_dir = os.path.dirname(config.get('filenames', {}).get('model', 'models/default_model.pkl'))
            os.makedirs(model_save_dir, exist_ok=True)
            study_path = os.path.join(model_save_dir, f"{zone}_optuna_study.pkl")
            try:
                joblib.dump(study, study_path); logger.info(f"[{zone_name}] Optuna study saved to {study_path}")
            except Exception as e:
                logger.error(f"[{zone_name}] Failed to save Optuna study: {e}")

    X_train_final = X_train_scaled.copy()
    y_train_final = y_train_arr.copy()

    if smote_config.get('enabled'):
        logger.info(f"[{zone_name}] Applying global SMOTE for final model training...")
        try:
            smote_variant_final = smote_config.get('variant', 'regular')
            k_neighbors_final_cfg = smote_config.get('k_neighbors', 5)
            configured_strategy_final = smote_config.get('sampling_strategy', 'auto')
            min_samples_final_recalc = smote_config.get('min_samples_per_class', 50)

            unique_final_y, counts_final_y = np.unique(y_train_arr, return_counts=True)
            min_class_count_final = min(counts_final_y) if counts_final_y.size > 0 else 0
            actual_k_final = min(k_neighbors_final_cfg, min_class_count_final - 1) if min_class_count_final > 1 else 1

            if actual_k_final >= 1 and len(unique_final_y) > 1:
                final_sampling_strategy = _get_sampling_strategy(
                    y_train_arr, configured_strategy_final, min_samples_final_recalc
                )
                logger.info(
                    f"[{zone_name}] Global SMOTE params: k={actual_k_final}, strategy={final_sampling_strategy}, variant={smote_variant_final}")

                if smote_variant_final == 'borderline':
                    smote_final_obj = BorderlineSMOTE(k_neighbors=actual_k_final, random_state=random_state,
                                                      sampling_strategy=final_sampling_strategy)
                elif smote_variant_final == 'adasyn':
                    smote_final_obj = ADASYN(n_neighbors=actual_k_final, random_state=random_state,
                                             sampling_strategy=final_sampling_strategy)
                elif smote_variant_final == 'smoteenn':
                    smote_final_obj = SMOTEENN(
                        smote=SMOTE(k_neighbors=actual_k_final, random_state=random_state,
                                    sampling_strategy=final_sampling_strategy),
                        random_state=random_state
                    )
                else:
                    smote_final_obj = SMOTE(k_neighbors=actual_k_final, random_state=random_state,
                                            sampling_strategy=final_sampling_strategy)

                X_train_final, y_train_final = smote_final_obj.fit_resample(X_train_scaled, y_train_arr)
                logger.info(f"[{zone_name}] Global SMOTE applied. New training shape: {X_train_final.shape}")
                unique_smote, counts_smote = np.unique(y_train_final, return_counts=True)
                logger.info(
                    f"[{zone_name}] Post-Global SMOTE training distribution: {dict(zip(unique_smote, counts_smote))}")
            else:
                logger.info(
                    f"[{zone_name}] Global SMOTE not applied (actual_k={actual_k_final}, unique_classes={len(unique_final_y)}). Using original training data.")

        except Exception as e:
            logger.error(f"[{zone_name}] Global SMOTE failed: {e}. Using original training data for final model.")
            X_train_final, y_train_final = X_train_scaled.copy(), y_train_arr.copy()

    logger.info(f"[{zone_name}] Training final model with parameters: {best_params_from_tuning}")

    xgb_final_model_params = {
        'objective': best_params_from_tuning.get('objective', 'multi:softprob'),
        'num_class': num_classes,
        'eval_metric': 'mlogloss',
        'random_state': random_state,
    }
    for p_name, p_val in best_params_from_tuning.items():
        if p_name not in xgb_final_model_params:
            xgb_final_model_params[p_name] = p_val

    final_model_class_weights = _calculate_class_weights(y_train_final, 'balanced_subsample')
    logger.info(f"[{zone_name}] Class weights for final model training (on y_train_final): {final_model_class_weights}")
    final_model_sample_weights = np.array([final_model_class_weights.get(y, 1.0) for y in y_train_final])

    X_train_for_base, y_train_for_base = X_train_final, y_train_final
    sample_weights_for_base = final_model_sample_weights

    X_calib, y_calib = None, None
    calibration_method = adv_calib_cfg.get('method', 'isotonic')
    calibration_cv_setting = adv_calib_cfg.get('cv', 'prefit')

    optuna_cfg_for_calib_split = tuning_config.get('optuna', {}) if tuning_config.get('enabled') else {}
    cv_folds_for_calib_check = optuna_cfg_for_calib_split.get('cv_folds', 3)

    if calibration_cv_setting == 'prefit' and \
            len(np.unique(y_train_final)) > 1 and \
            len(y_train_final) > cv_folds_for_calib_check * 2:
        try:
            unique_y_final_classes, unique_y_final_counts = np.unique(y_train_final, return_counts=True)
            can_stratify_calib = len(unique_y_final_classes) > 1 and \
                                 all(c >= 2 for c in unique_y_final_counts) and \
                                 all(int(c * calibration_split_size) >= 1 for c in unique_y_final_counts if c > 0)

            stratify_calib = y_train_final if can_stratify_calib else None

            X_train_for_base, X_calib, y_train_for_base, y_calib = train_test_split(
                X_train_final, y_train_final, test_size=calibration_split_size,
                random_state=random_state, stratify=stratify_calib
            )
            base_model_class_weights_after_split = _calculate_class_weights(y_train_for_base, 'balanced_subsample')
            sample_weights_for_base = np.array(
                [base_model_class_weights_after_split.get(y, 1.0) for y in y_train_for_base])
            logger.info(
                f"[{zone_name}] Split training data for 'prefit' calibration. Base train: {X_train_for_base.shape}, Calib: {X_calib.shape if X_calib is not None else 'None'}")
            logger.info(
                f"[{zone_name}] Base model training class dist: {dict(zip(*np.unique(y_train_for_base, return_counts=True)))}")
            if X_calib is not None and y_calib is not None and len(y_calib) > 0:
                logger.info(
                    f"[{zone_name}] Calibration set class dist: {dict(zip(*np.unique(y_calib, return_counts=True)))}")
            else:
                logger.warning(
                    f"[{zone_name}] Calibration set (X_calib, y_calib) is empty or None after split. Prefit calibration may not occur as expected.")
                X_calib, y_calib = None, None  # Ensure they are None if split failed to produce a usable calib set

        except ValueError as ve:
            logger.warning(f"[{zone_name}] Stratified split for calibration failed ({ve}). Using non-stratified split.")
            X_train_for_base, X_calib, y_train_for_base, y_calib = train_test_split(
                X_train_final, y_train_final, test_size=calibration_split_size, random_state=random_state
            )
            base_model_class_weights_after_split = _calculate_class_weights(y_train_for_base, 'balanced_subsample')
            sample_weights_for_base = np.array(
                [base_model_class_weights_after_split.get(y, 1.0) for y in y_train_for_base])
            if X_calib is not None and y_calib is not None and len(y_calib) > 0:
                logger.info(
                    f"[{zone_name}] Calibration set (non-stratified) class dist: {dict(zip(*np.unique(y_calib, return_counts=True)))}")
            else:
                logger.warning(
                    f"[{zone_name}] Calibration set (X_calib, y_calib) is empty or None after non-stratified split. Prefit calibration may not occur as expected.")
                X_calib, y_calib = None, None

    base_model_for_calibration = xgb.XGBClassifier(**xgb_final_model_params)

    fit_params_final = {'sample_weight': sample_weights_for_base, 'verbose': False}
    early_stopping_final = training_params.get('early_stopping_rounds')
    if early_stopping_final and 'early_stopping_rounds' not in xgb_final_model_params:
        xgb_final_model_params['early_stopping_rounds'] = early_stopping_final

    if xgb_final_model_params.get('early_stopping_rounds'):
        eval_set_x, eval_set_y = X_test_scaled, y_test_arr
        can_use_calib_for_es = X_calib is not None and y_calib is not None and len(np.unique(y_calib)) > 1
        can_use_test_for_es = len(np.unique(y_test_arr)) > 1

        if calibration_cv_setting == 'prefit' and can_use_calib_for_es:
            eval_set_x, eval_set_y = X_calib, y_calib
            logger.info(f"[{zone_name}] Using calibration set for XGBoost early stopping.")
        elif can_use_test_for_es:
            logger.info(f"[{zone_name}] Using test set for XGBoost early stopping.")
        else:
            logger.warning(
                f"[{zone_name}] Cannot use early stopping: no suitable eval set with >1 class found (Calib unique: {len(np.unique(y_calib)) if y_calib is not None and len(y_calib) > 0 else 'N/A'}, Test unique: {len(np.unique(y_test_arr))}).")
            if 'early_stopping_rounds' in xgb_final_model_params: del xgb_final_model_params['early_stopping_rounds']

        if 'early_stopping_rounds' in xgb_final_model_params:
            fit_params_final['eval_set'] = [(eval_set_x, eval_set_y)]

    logger.info(f"[{zone_name}] Fitting base XGBoost model...")
    base_model_for_calibration.fit(X_train_for_base, y_train_for_base, **fit_params_final)

    if training_params.get('use_ensemble', False):
        logger.info(f"[{zone_name}] Creating and training ensemble model...")
        rf_params_cfg = ADVANCED_TRAINING_CONFIG.get('ensemble_options', {}).get('random_forest_params', {})
        rf_model = RandomForestClassifier(
            n_estimators=rf_params_cfg.get('n_estimators', 300),
            max_depth=rf_params_cfg.get('max_depth', xgb_final_model_params.get('max_depth')),
            class_weight=rf_params_cfg.get('class_weight', 'balanced_subsample'),
            random_state=random_state,
            n_jobs=-1
        )
        rf_model.fit(X_train_for_base, y_train_for_base)

        ensemble_weights = ADVANCED_TRAINING_CONFIG.get('ensemble_options', {}).get('weights', {'xgboost': 0.7,
                                                                                                'random_forest': 0.3})
        voting_model = VotingClassifier(
            estimators=[('xgb', base_model_for_calibration), ('rf', rf_model)],
            voting=ADVANCED_TRAINING_CONFIG.get('ensemble_options', {}).get('voting_type', 'soft'),
            weights=[ensemble_weights.get('xgboost', 0.7), ensemble_weights.get('random_forest', 0.3)]
        )
        logger.info(f"[{zone_name}] Fitting VotingClassifier ensemble...")
        voting_model.fit(X_train_for_base, y_train_for_base)
        base_model_for_calibration = voting_model

    final_model_to_evaluate = base_model_for_calibration
    logger.info(
        f"[{zone_name}] Calibrating model using method: '{calibration_method}', cv: '{calibration_cv_setting}'...")

    try:
        if calibration_cv_setting == 'prefit':
            if X_calib is not None and y_calib is not None and len(np.unique(y_calib)) > 1:
                calibrated_model = CalibratedClassifierCV(
                    estimator=base_model_for_calibration,
                    method=calibration_method,
                    cv='prefit'
                )
                calibrated_model.fit(X_calib, y_calib)
                final_model_to_evaluate = calibrated_model
                logger.info(f"[{zone_name}] Model calibrated successfully using 'prefit' on calibration set.")
            else:
                logger.warning(
                    f"[{zone_name}] 'prefit' calibration skipped: Calibration set not suitable or not available.")
        elif isinstance(calibration_cv_setting, int) and calibration_cv_setting > 1:
            if len(np.unique(y_train_final)) > 1:
                if training_params.get('use_ensemble', False):
                    rf_params_cfg_calib = ADVANCED_TRAINING_CONFIG.get('ensemble_options', {}).get(
                        'random_forest_params', {})
                    xgb_estimator_calib = xgb.XGBClassifier(**xgb_final_model_params)
                    rf_estimator_calib = RandomForestClassifier(
                        n_estimators=rf_params_cfg_calib.get('n_estimators', 300),
                        max_depth=rf_params_cfg_calib.get('max_depth', xgb_final_model_params.get('max_depth')),
                        class_weight=rf_params_cfg_calib.get('class_weight', 'balanced_subsample'),
                        random_state=random_state, n_jobs=-1
                    )
                    ensemble_weights_calib = ADVANCED_TRAINING_CONFIG.get('ensemble_options', {}).get('weights',
                                                                                                      {'xgboost': 0.7,
                                                                                                       'random_forest': 0.3})
                    estimator_for_calib_cv = VotingClassifier(
                        estimators=[('xgb', xgb_estimator_calib), ('rf', rf_estimator_calib)],
                        voting=ADVANCED_TRAINING_CONFIG.get('ensemble_options', {}).get('voting_type', 'soft'),
                        weights=[ensemble_weights_calib.get('xgboost', 0.7),
                                 ensemble_weights_calib.get('random_forest', 0.3)]
                    )
                else:
                    estimator_for_calib_cv = xgb.XGBClassifier(**xgb_final_model_params)

                calibrated_model_cv = CalibratedClassifierCV(
                    estimator=estimator_for_calib_cv,
                    method=calibration_method,
                    cv=calibration_cv_setting
                )
                calibrated_model_cv.fit(X_train_final, y_train_final, sample_weight=final_model_sample_weights)
                final_model_to_evaluate = calibrated_model_cv
                logger.info(f"[{zone_name}] Model calibrated successfully using cv={calibration_cv_setting}.")
            else:
                logger.warning(
                    f"[{zone_name}] Calibration with cv={calibration_cv_setting} skipped: y_train_final has <2 classes.")
        else:
            logger.warning(
                f"[{zone_name}] Unknown calibration CV setting: '{calibration_cv_setting}'. Calibration skipped.")

    except Exception as e:
        logger.warning(f"[{zone_name}] Calibration failed: {e}. Using uncalibrated model.", exc_info=True)

    logger.info(f"[{zone_name}] Evaluating final model on test set...")
    y_pred_test = final_model_to_evaluate.predict(X_test_scaled)
    y_proba_test = final_model_to_evaluate.predict_proba(X_test_scaled)

    report = classification_report(y_test_arr, y_pred_test, target_names=list(class_names_map.values()),
                                   zero_division=0)
    logger.info(f"[{zone_name}] Test Set Classification Report:\n{report}")
    cm = confusion_matrix(y_test_arr, y_pred_test)
    logger.info(f"[{zone_name}] Confusion Matrix:\n{cm}")

    bal_acc_test = balanced_accuracy_score(y_test_arr, y_pred_test)
    kappa_test = cohen_kappa_score(y_test_arr, y_pred_test)
    logger.info(f"[{zone_name}] Balanced Accuracy: {bal_acc_test:.4f}")
    logger.info(f"[{zone_name}] Cohen's Kappa: {kappa_test:.4f}")

    unique_preds_test = len(np.unique(y_pred_test))
    logger.info(f"[{zone_name}] Unique predictions on test set: {unique_preds_test}/{num_classes}")
    if unique_preds_test == 1 and num_classes > 1:
        logger.warning(f"[{zone_name}] FINAL MODEL IS ONLY PREDICTING ONE CLASS ON THE TEST SET!")

    try:
        y_test_bin = label_binarize(y_test_arr, classes=list(range(num_classes)))
        if y_test_bin.shape[1] == num_classes:
            for i in range(num_classes):
                if np.sum(y_test_bin[:, i]) > 0 and y_proba_test.shape[1] > i:
                    auc = roc_auc_score(y_test_bin[:, i], y_proba_test[:, i])
                    logger.info(f"[{zone_name}] AUC for class {class_names_map.get(i, str(i))}: {auc:.4f}")
    except Exception as e:
        logger.warning(f"[{zone_name}] Could not calculate per-class AUC: {e}")

    # --- Feature Importance Retrieval (FIXED for TypeError) ---
    feature_importance_df = pd.DataFrame()
    actual_model_to_get_importance_from = final_model_to_evaluate
    if hasattr(final_model_to_evaluate, 'estimator') and final_model_to_evaluate.estimator is not None:
        actual_model_to_get_importance_from = final_model_to_evaluate.estimator
    elif hasattr(final_model_to_evaluate, 'base_estimator') and final_model_to_evaluate.base_estimator is not None:
        actual_model_to_get_importance_from = final_model_to_evaluate.base_estimator

    importances = None
    if isinstance(actual_model_to_get_importance_from, (xgb.XGBClassifier, RandomForestClassifier)):
        if hasattr(actual_model_to_get_importance_from, 'feature_importances_'):
            importances = actual_model_to_get_importance_from.feature_importances_
    elif isinstance(actual_model_to_get_importance_from, VotingClassifier):
        # Use named_estimators_ which is a dictionary of name:fitted_estimator
        if hasattr(actual_model_to_get_importance_from, 'named_estimators_') and \
                'xgb' in actual_model_to_get_importance_from.named_estimators_ and \
                hasattr(actual_model_to_get_importance_from.named_estimators_['xgb'], 'feature_importances_'):
            importances = actual_model_to_get_importance_from.named_estimators_['xgb'].feature_importances_
        else:
            logger.warning(
                f"[{zone_name}] Could not retrieve 'xgb' estimator or its importances from VotingClassifier.")
    else:
        logger.warning(
            f"[{zone_name}] Unknown model type for feature importance: {type(actual_model_to_get_importance_from)}")

    if importances is not None and feature_names:
        if len(importances) == len(feature_names):
            feature_importance_df = pd.DataFrame({'feature': feature_names, 'importance': importances}).sort_values(
                'importance', ascending=False)
            logger.info(f"[{zone_name}] Top 10 features:\n{feature_importance_df.head(10)}")
        else:
            logger.warning(
                f"[{zone_name}] Mismatch between feature importances length ({len(importances)}) and feature names length ({len(feature_names)} from prepare_data). Skipping importance logging.")
    elif feature_names is None:
        logger.warning(f"[{zone_name}] Feature names are None. Skipping importance logging.")
    # --- End of Feature Importance Fix ---

    training_analysis_df = pd.DataFrame()
    if UTILS_LOADED:
        try:
            y_pred_train_analysis = final_model_to_evaluate.predict(X_train_scaled)
            y_proba_train_analysis = final_model_to_evaluate.predict_proba(X_train_scaled)

            analysis_data = {'Expert_Label': y_train_arr, 'Predicted_Label': y_pred_train_analysis}
            for i, c_name in class_names_map.items():
                safe_c_name = c_name.replace(' ', '_')
                if i < y_proba_train_analysis.shape[1]: analysis_data[f'Prob_{safe_c_name}'] = y_proba_train_analysis[:,
                                                                                               i]

            training_analysis_df = pd.DataFrame(analysis_data, index=X_train_df.index if X_train_df.index.size == len(
                y_train_arr) else None)
            if training_analysis_df.index is None:
                training_analysis_df = pd.DataFrame(analysis_data)

            training_analysis_df['Entropy'] = [calculate_entropy(p) for p in y_proba_train_analysis]
            training_analysis_df['Margin'] = [calculate_margin(p) for p in y_proba_train_analysis]
            training_analysis_df['Prob_True_Label'] = [
                y_proba_train_analysis[idx, lbl] if 0 <= lbl < y_proba_train_analysis.shape[1] else 0.0
                for idx, lbl in enumerate(y_train_arr)
            ]
            training_analysis_df['Is_Correct'] = (
                        training_analysis_df['Expert_Label'] == training_analysis_df['Predicted_Label'])
            logger.info(
                f"[{zone_name}] Accuracy on original training set (for analysis): {training_analysis_df['Is_Correct'].mean():.4f}")

        except Exception as e:
            logger.error(f"[{zone_name}] Training set uncertainty analysis failed: {e}", exc_info=True)

    return final_model_to_evaluate, scaler, feature_importance_df, training_analysis_df

def train_zone_with_validation(zone, quick_validation_only=False, validation_data=None):
    """Enhanced train_zone function with validation capabilities and fixes."""

    if not UTILS_LOADED:
        print(f"ERROR: Cannot train {zone}, paralysis_utils failed to load.")
        return

    if zone not in ZONE_CONFIG:
        print(f"ERROR: Zone '{zone}' not found in paralysis_config.py")
        return

    config = ZONE_CONFIG[zone]
    zone_name = config.get('name', zone.capitalize() + ' Face')
    filenames = config.get('filenames', {})
    training_params = config.get('training', {})
    review_cfg = training_params.get('review_analysis', {})

    log_file = filenames.get('training_log', os.path.join('logs', f'{zone}_training.log'))
    _setup_logging(log_file, LOGGING_CONFIG.get('level', 'INFO'))

    logger.info(f"{'=' * 60}\nStarting Enhanced Training for Zone: {zone_name}\n{'=' * 60}")
    tune_cfg = training_params.get('hyperparameter_tuning', {})
    smote_cfg = training_params.get('smote', {})
    fs_cfg = config.get('feature_selection', {})  # Get FS config
    logger.info(f"Configuration:\n  - Tuning: {tune_cfg.get('enabled', False)} ({tune_cfg.get('method', 'none')})\n"
                f"  - SMOTE: {smote_cfg.get('enabled', False)} (variant: {smote_cfg.get('variant', 'regular')}, strategy: {smote_cfg.get('sampling_strategy', 'auto')})\n"
                f"  - Feature Selection: {fs_cfg.get('enabled', False)} (Top N: {fs_cfg.get('top_n_features', 'N/A')})\n"  # Log FS details
                f"  - Ensemble: {training_params.get('use_ensemble', False)}\n"
                f"  - Review Analysis: {review_cfg.get('enabled', False)}")

    try:
        logger.info(f"[{zone_name}] Loading data...")
        module_name = f"{zone}_face_features"  # Assumes file like 'lower_face_features.py'
        feature_module = importlib.import_module(module_name)
        prepare_data_func = getattr(feature_module, 'prepare_data')

        results_csv = INPUT_FILES.get('results_csv');
        expert_csv = INPUT_FILES.get('expert_key_csv')
        if not results_csv or not os.path.exists(results_csv): logger.error(
            f"Results CSV missing: {results_csv}"); return
        if not expert_csv or not os.path.exists(expert_csv): logger.error(f"Expert CSV missing: {expert_csv}"); return

        if quick_validation_only:
            # ... (quick validation logic remains same)
            logger.info(f"[{zone_name}] Running quick validation mode...")
            if validation_data:
                X_val, y_val = validation_data
                model_path = filenames.get('model');
                scaler_path = filenames.get('scaler')
                if os.path.exists(model_path) and os.path.exists(scaler_path):
                    model = joblib.load(model_path);
                    scaler = joblib.load(scaler_path)
                    X_val_scaled = scaler.transform(X_val)
                    y_pred = model.predict(X_val_scaled)
                    report = classification_report(y_val, y_pred, zero_division=0)
                    logger.info(f"[{zone_name}] Quick Validation Report:\n{report}")
                else:
                    logger.error(f"[{zone_name}] Model/scaler not found for quick validation.")
            return

        features_df, targets_arr, metadata_df = prepare_data_func(results_file=results_csv, expert_file=expert_csv)

        if features_df is None or features_df.empty: logger.error(f"[{zone_name}] No features extracted."); return
        if targets_arr is None or len(targets_arr) == 0: logger.error(f"[{zone_name}] No targets found."); return
        if len(features_df) != len(targets_arr): logger.error(
            f"[{zone_name}] Features/targets length mismatch."); return

        feature_names = features_df.columns.tolist()  # Get feature names AFTER prepare_data (and potential FS)
        logger.info(
            f"[{zone_name}] Loaded {len(features_df)} samples with {len(feature_names)} features: {feature_names[:10]}...")  # Log some features

        unique_all, counts_all = np.unique(targets_arr, return_counts=True)
        class_dist = {PARALYSIS_MAP.get(k, str(k)): v for k, v in zip(unique_all, counts_all)}
        logger.info(f"[{zone_name}] Overall class distribution from prepare_data: {class_dist}")

        if counts_all.size > 0 and min(counts_all) > 0:
            imbalance_ratio = max(counts_all) / min(counts_all)
            if imbalance_ratio > 10: logger.warning(
                f"[{zone_name}] Severe class imbalance! Ratio: {imbalance_ratio:.2f}")
        else:
            logger.warning(f"[{zone_name}] Cannot calculate imbalance ratio due to class counts: {counts_all}")

        stratify_split = targets_arr if len(unique_all) > 1 and all(
            c >= 2 for c in counts_all) else None  # Ensure at least 2 samples per class for stratify
        if stratify_split is None and len(unique_all) > 1:
            logger.warning(
                f"[{zone_name}] Cannot stratify train/test split due to low class counts. Proceeding without stratification.")

        X_train_df, X_test_df, y_train_arr, y_test_arr, metadata_train, metadata_test = train_test_split(
            features_df, targets_arr, metadata_df,
            test_size=training_params.get('test_size', 0.25),
            random_state=training_params.get('random_state', 42),
            stratify=stratify_split
        )
        logger.info(f"[{zone_name}] Train set: {X_train_df.shape[0]} samples, Test set: {X_test_df.shape[0]} samples")

        final_model, scaler, feature_importance, training_analysis_df = train_enhanced_model(
            zone, X_train_df, y_train_arr, X_test_df, y_test_arr,
            feature_names, config  # Pass current feature_names
        )

        if final_model is None: logger.error(f"[{zone_name}] Model training failed."); return

        logger.info(f"[{zone_name}] Saving model artifacts...")
        model_path = filenames.get('model');
        scaler_path = filenames.get('scaler')
        feature_list_path = filenames.get('feature_list')  # This is saved by prepare_data
        importance_path = filenames.get('importance')

        for path_val in [model_path, scaler_path, importance_path]:  # feature_list_path handled by prepare_data
            if path_val: os.makedirs(os.path.dirname(path_val), exist_ok=True)

        joblib.dump(final_model, model_path)
        joblib.dump(scaler, scaler_path)
        if not feature_importance.empty and importance_path:
            feature_importance.to_csv(importance_path, index=False)
            logger.info(f"[{zone_name}] Feature importance saved to {importance_path}")
        elif importance_path:
            logger.info(f"[{zone_name}] Feature importance is empty or path not specified. Not saved.")

        logger.info(f"[{zone_name}] Artifacts saved successfully.")

        if review_cfg.get('enabled', False) and UTILS_LOADED:
            logger.info(f"[{zone_name}] Generating review candidates...")
            review_csv_path = filenames.get('review_candidates_csv',
                                            os.path.join('analysis_results', f"{zone}_review_candidates.csv"))
            os.makedirs(os.path.dirname(review_csv_path), exist_ok=True)

            if training_analysis_df is None or training_analysis_df.empty:
                logger.warning(f"[{zone_name}] No training analysis available for review candidates.")
            else:
                if not metadata_train.index.equals(
                        training_analysis_df.index) and training_analysis_df.index.size == metadata_train.index.size:
                    # If indexes are different but same length, try to reset and align. Risky.
                    logger.warning(
                        f"[{zone_name}] Metadata_train and training_analysis_df index mismatch but same length. Attempting reindex based on position. Ensure this is intended.")
                    metadata_train = metadata_train.reset_index(drop=True)
                    training_analysis_df = training_analysis_df.reset_index(drop=True)
                elif not metadata_train.index.equals(training_analysis_df.index):
                    logger.warning(
                        f"[{zone_name}] Metadata_train and training_analysis_df index mismatch and different lengths. Reindexing metadata_train.")
                    metadata_train = metadata_train.reindex(
                        training_analysis_df.index)  # This might introduce NaNs if indexes don't align

                review_candidates_df = generate_review_candidates(
                    zone=zone, model=final_model, scaler=scaler, feature_names=feature_names,
                    config=config, X_train_orig=X_train_df, y_train_orig=y_train_arr,
                    X_test_orig=X_test_df, y_test_orig=y_test_arr,
                    metadata_train=metadata_train, training_analysis_df=training_analysis_df,
                    class_names_map=PARALYSIS_MAP,
                    top_k_influence=review_cfg.get('top_k_influence', 50),
                    entropy_quantile=review_cfg.get('entropy_quantile', 0.9),
                    margin_quantile=review_cfg.get('margin_quantile', 0.1),
                    true_label_prob_threshold=review_cfg.get('true_label_prob_threshold', 0.4)
                )
                if review_candidates_df is not None and not review_candidates_df.empty:
                    review_candidates_df.to_csv(review_csv_path, index=False, float_format='%.4f')
                    logger.info(
                        f"[{zone_name}] Saved {len(review_candidates_df)} review candidates to {review_csv_path}")
                else:
                    logger.info(f"[{zone_name}] No review candidates generated.")

        logger.info(f"[{zone_name}] Training complete!\n{'=' * 60}")

    except Exception as e:
        logger.error(f"[{zone_name}] Training failed: {str(e)}", exc_info=True)
        logger.info(f"{'=' * 60}")


def analyze_model_performance(zone):
    """Analyze saved model performance and generate detailed reports"""
    # ... (analyze_model_performance logic remains same) ...
    if zone not in ZONE_CONFIG:
        print(f"ERROR: Zone '{zone}' not found")
        return

    config = ZONE_CONFIG[zone]
    zone_name = config.get('name', zone.capitalize() + ' Face')
    model_base_dir = os.path.dirname(config.get('filenames', {}).get('model', 'models/default.pkl'))
    study_path = os.path.join(model_base_dir, f"{zone}_optuna_study.pkl")

    if os.path.exists(study_path):
        try:
            study = joblib.load(study_path)
            print(f"\n{zone_name} Optuna Study Analysis:")
            print(f"  - Total trials: {len(study.trials)}")
            if study.best_trial:
                print(f"  - Best trial number: {study.best_trial.number}")
                print(f"  - Best value: {study.best_value:.4f}")
                print(f"  - Best params: {study.best_params}")
            else:
                print("  - No best trial found in study.")

            plot_optimization = ADVANCED_TRAINING_CONFIG.get('monitoring', {}).get('plot_optimization_history', False)
            if plot_optimization:
                import matplotlib.pyplot as plt
                from optuna.visualization import plot_optimization_history, plot_param_importances

                fig_history = plot_optimization_history(study)
                history_plot_path = os.path.join(model_base_dir, f"{zone}_optuna_history.png")
                fig_history.write_image(history_plot_path)
                print(f"  - Saved optimization history plot: {history_plot_path}")

                if len(study.trials) > 1 and study.best_trial:  # Param importances make sense if there are multiple trials and a best one
                    fig_importance = plot_param_importances(study)
                    importance_plot_path = os.path.join(model_base_dir, f"{zone}_optuna_param_importance.png")
                    fig_importance.write_image(importance_plot_path)
                    print(f"  - Saved parameter importance plot: {importance_plot_path}")
        except ImportError:
            print("    Cannot generate Optuna plots: matplotlib or plotly might be missing.")
        except Exception as e:
            print(f"    Could not analyze Optuna study or generate plots: {e}")
    else:
        print(f"\n{zone_name} Optuna Study Analysis: Study file not found at {study_path}")


if __name__ == "__main__":
    if not UTILS_LOADED:
        print("CRITICAL: paralysis_utils could not be loaded. Training cannot run.")
    else:
        optuna_needed = any(
            zone_cfg.get('training', {}).get('hyperparameter_tuning', {}).get('enabled') and
            zone_cfg.get('training', {}).get('hyperparameter_tuning', {}).get('method') == 'optuna'
            for zone_cfg in ZONE_CONFIG.values()
        )

        if optuna_needed and not OPTUNA_AVAILABLE:
            print(
                "CRITICAL: Optuna is required for one or more zones but not installed. Please install with: pip install optuna plotly")  # Added plotly for Optuna viz
        else:
            if optuna_needed: print("Optuna is available and will be used for hyperparameter tuning if configured.")

            zones_to_train = ['lower', 'mid', 'upper']
            # zones_to_train = ['mid'] # Example: focus on one zone

            for zone_key in zones_to_train:
                if zone_key in ZONE_CONFIG:
                    print(f"\n{'=' * 60}\nStarting Main Training Process for {zone_key.upper()} FACE\n{'=' * 60}")
                    train_zone_with_validation(zone_key)
                    analyze_model_performance(zone_key)
                    print(f"\nFinished Main Training Process for {zone_key.upper()} FACE")
                else:
                    print(f"Warning: Zone '{zone_key}' not found in config. Skipping.")