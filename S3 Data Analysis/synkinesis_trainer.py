# synkinesis_trainer.py
#
# Per-type binary trainer for synkinesis classifiers. Reuses the modern
# paralysis training helpers (SMOTE/SMOTEENN, save_model_artifacts) where
# they are pattern-agnostic, but owns its own Optuna objective, threshold
# optimizer, and performance summary so that the binary path stays clean.

import logging
import os
import warnings
from copy import deepcopy

import joblib
import numpy as np
import pandas as pd

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    import xgboost as xgb
    from sklearn.calibration import CalibratedClassifierCV
    from sklearn.ensemble import (
        ExtraTreesClassifier,
        RandomForestClassifier,
        VotingClassifier,
    )
    from sklearn.metrics import (
        average_precision_score,
        balanced_accuracy_score,
        classification_report,
        confusion_matrix,
        f1_score,
        precision_recall_curve,
        roc_auc_score,
    )
    from sklearn.model_selection import StratifiedKFold, train_test_split
    from sklearn.preprocessing import StandardScaler

    try:
        import optuna
        from optuna.pruners import HyperbandPruner, MedianPruner
        from optuna.samplers import TPESampler
        OPTUNA_AVAILABLE = True
        optuna.logging.set_verbosity(optuna.logging.WARNING)
    except ImportError:
        OPTUNA_AVAILABLE = False
        TPESampler = HyperbandPruner = MedianPruner = None

from paralysis_training_helpers import (
    apply_smote_and_cleaning,
    get_optuna_suggestion,
)
from paralysis_utils import hybrid_feature_selection
from synkinesis_config import (
    ADVANCED_TRAINING_CONFIG,
    SYNKINESIS_CONFIG,
    ensure_artifact_dirs,
)
from synkinesis_data import prepare_type_data

logger = logging.getLogger(__name__)


def _setup_logging(log_file, level='INFO'):
    log_dir = os.path.dirname(log_file)
    if log_dir:
        os.makedirs(log_dir, exist_ok=True)
    root = logging.getLogger()
    if root.hasHandlers():
        root.handlers.clear()
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
        handlers=[logging.StreamHandler(), logging.FileHandler(log_file, mode='w')],
        force=True,
    )


def _sample_weight_from_class_weights(y, class_weights):
    return np.array([class_weights.get(int(label), 1.0) for label in y], dtype=float)


def _scale_pos_weight_from_class_weights(class_weights):
    """XGB uses scale_pos_weight = (negative_count / positive_count) * adjustment.
    We translate class_weights {0: w0, 1: w1} into scale_pos_weight = w1/w0."""
    w0 = float(class_weights.get(0, 1.0))
    w1 = float(class_weights.get(1, 1.0))
    return max(w1 / w0, 0.1)


def _build_xgb(params, random_state, class_weights=None):
    model_params = {
        'objective': 'binary:logistic',
        'eval_metric': 'aucpr',
        'tree_method': 'hist',
        'random_state': random_state,
        'verbosity': 0,
    }
    model_params.update(params)
    if class_weights is not None and 'scale_pos_weight' not in model_params:
        model_params['scale_pos_weight'] = _scale_pos_weight_from_class_weights(class_weights)
    return xgb.XGBClassifier(**model_params)


def _build_voting_ensemble(xgb_params, ensemble_cfg, random_state, class_weights=None):
    rf_params = dict(ensemble_cfg.get('random_forest_params', {}))
    et_params = dict(ensemble_cfg.get('extra_trees_params', {}))
    weights_cfg = ensemble_cfg.get('weights', {'xgb': 0.6, 'rf': 0.2, 'et': 0.2})

    # sklearn ≥1.7 VotingClassifier no longer reliably forwards sample_weight
    # to base estimators wrapped in CalibratedClassifierCV. Configure imbalance
    # handling at the base-estimator level instead.
    if class_weights is not None and 'class_weight' not in rf_params:
        rf_params['class_weight'] = 'balanced_subsample'
    if class_weights is not None and 'class_weight' not in et_params:
        et_params['class_weight'] = 'balanced_subsample'

    xgb_clf = _build_xgb(xgb_params, random_state, class_weights=class_weights)
    rf_clf = RandomForestClassifier(random_state=random_state, **rf_params)
    et_clf = ExtraTreesClassifier(random_state=random_state, **et_params)

    return VotingClassifier(
        estimators=[('xgb', xgb_clf), ('rf', rf_clf), ('et', et_clf)],
        voting=ensemble_cfg.get('voting_type', 'soft'),
        weights=[weights_cfg.get('xgb', 0.6), weights_cfg.get('rf', 0.2), weights_cfg.get('et', 0.2)],
        n_jobs=1,
    )


def _create_optuna_objective(X_train_scaled, y_train, smote_cfg, optuna_cfg, class_weights, random_state):
    param_dists = optuna_cfg.get('param_distributions', {})
    cv_folds = optuna_cfg.get('cv_folds', 5)
    apply_smote_per_fold = smote_cfg.get('apply_per_fold_in_tuning', True)

    def objective(trial):
        params = {
            name: get_optuna_suggestion(trial, name, cfg)
            for name, cfg in param_dists.items()
        }
        skf = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=random_state)
        scores = []
        for fold_idx, (tr_idx, val_idx) in enumerate(skf.split(X_train_scaled, y_train)):
            X_tr = X_train_scaled[tr_idx]
            y_tr = y_train[tr_idx]
            X_val = X_train_scaled[val_idx]
            y_val = y_train[val_idx]

            if apply_smote_per_fold:
                X_tr_res, y_tr_res = apply_smote_and_cleaning(
                    X_tr, y_tr, smote_cfg, random_state, zone_name_log=f"trial{trial.number}_fold{fold_idx}"
                )
            else:
                X_tr_res, y_tr_res = X_tr, y_tr

            model = _build_xgb(params, random_state, class_weights=class_weights)
            model.fit(X_tr_res, y_tr_res)
            proba_pos = model.predict_proba(X_val)[:, 1]
            scores.append(average_precision_score(y_val, proba_pos))

            trial.report(np.mean(scores), step=fold_idx)
            if trial.should_prune():
                raise optuna.exceptions.TrialPruned()
        return float(np.mean(scores))

    return objective


def _optimize_threshold(y_true, proba_pos, search_range, step):
    """Pick the threshold that maximizes positive-class F1. Tolerates NaN entries
    in proba_pos (skipped from optimization)."""
    y_arr = np.asarray(y_true)
    p_arr = np.asarray(proba_pos)
    valid = ~np.isnan(p_arr)
    if not valid.any():
        return 0.5, 0.0
    y_v = y_arr[valid]
    p_v = p_arr[valid]
    lo, hi = search_range
    candidates = np.arange(lo, hi + 1e-9, step)
    best_f1 = -1.0
    best_thr = 0.5
    for thr in candidates:
        y_pred = (p_v >= thr).astype(int)
        f1 = f1_score(y_v, y_pred, pos_label=1, zero_division=0)
        if f1 > best_f1:
            best_f1 = f1
            best_thr = float(thr)
    return best_thr, best_f1


def _oof_predict_proba(X, y, smote_cfg, ensemble_cfg, xgb_params, class_weights,
                       random_state, n_splits=5, name="type"):
    """Out-of-fold probabilities via stratified K-fold with per-fold SMOTE.

    Used to optimize the decision threshold without burning a held-out validation
    slice — more stable for small minority classes (brow_cocked has only ~13 train
    positives) than a single 20% val cut. The voting ensemble is fit per fold
    without calibration; calibration is monotonic so the F1-optimal threshold
    carries over to the calibrated production probabilities (modulo a small shift).
    """
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    oof = np.full(len(y), np.nan, dtype=float)
    for fold_idx, (tr_idx, val_idx) in enumerate(skf.split(X, y)):
        X_tr = X[tr_idx]
        y_tr = y[tr_idx]
        X_val_fold = X[val_idx]
        try:
            X_tr_res, y_tr_res = apply_smote_and_cleaning(
                X_tr, y_tr, smote_cfg, random_state, zone_name_log=f'{name}_oof_f{fold_idx}'
            )
            voting = _build_voting_ensemble(xgb_params, ensemble_cfg, random_state, class_weights=class_weights)
            voting.fit(X_tr_res, y_tr_res)
            oof[val_idx] = voting.predict_proba(X_val_fold)[:, 1]
        except Exception as e:
            logger.warning(f"[{name}] OOF fold {fold_idx} failed ({e}); leaving fold probas as NaN.")
    return oof


def _save_review_candidates(name, output_path, train_meta, y_train, train_oof,
                            test_meta, y_test, test_proba, threshold,
                            confident_threshold=0.5):
    """Identify samples where the model strongly disagrees with the expert label.

    Useful for spot-checking possibly-mislabeled patients in the FPRS key. A high
    `disagreement_score` means: for a labeled positive the model assigns very low
    probability (likely false negative or label noise), or for a labeled negative
    the model assigns very high probability (likely false positive or missed
    positive in the labels). Candidates are sorted by descending disagreement.
    """
    rows = []
    for split_label, meta_df, y_arr, proba_arr in (
        ('train_oof', train_meta, y_train, train_oof),
        ('test', test_meta, y_test, test_proba),
    ):
        if meta_df is None or len(meta_df) == 0:
            continue
        meta_df = meta_df.reset_index(drop=True)
        for i in range(len(y_arr)):
            p = proba_arr[i]
            if np.isnan(p):
                continue
            true = int(y_arr[i])
            if true == 1:
                disagreement = 1.0 - float(p)
                disagreement_type = 'labeled_positive_model_says_negative'
            else:
                disagreement = float(p)
                disagreement_type = 'labeled_negative_model_says_positive'
            if disagreement >= confident_threshold:
                rows.append({
                    'patient_id': meta_df.iloc[i]['Patient ID'],
                    'side': meta_df.iloc[i]['Side'],
                    'split': split_label,
                    'true_label': true,
                    'model_proba': float(p),
                    'threshold_used': float(threshold),
                    'disagreement_score': disagreement,
                    'disagreement_type': disagreement_type,
                })
    if not rows:
        logger.info(f"[{name}] No label review candidates above {confident_threshold}.")
        return pd.DataFrame()
    df = pd.DataFrame(rows).sort_values('disagreement_score', ascending=False)
    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        df.to_csv(output_path, index=False)
        logger.info(f"[{name}] {len(df)} review candidates → {output_path}")
    return df


def _binary_performance_summary(name, y_true, proba_pos, threshold):
    y_pred = (proba_pos >= threshold).astype(int)
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    report = classification_report(y_true, y_pred, target_names=['No', 'Yes'],
                                   zero_division=0, labels=[0, 1])
    logger.info(f"[{name}] Threshold = {threshold:.3f}")
    logger.info(f"[{name}] Confusion matrix (rows=True, cols=Pred):\n{cm}")
    logger.info(f"[{name}] Classification report:\n{report}")
    metrics = {
        'threshold': threshold,
        'accuracy': float((y_pred == y_true).mean()),
        'balanced_accuracy': float(balanced_accuracy_score(y_true, y_pred)),
        'f1_positive': float(f1_score(y_true, y_pred, pos_label=1, zero_division=0)),
        'average_precision': float(average_precision_score(y_true, proba_pos)),
    }
    if len(np.unique(y_true)) == 2:
        try:
            metrics['roc_auc'] = float(roc_auc_score(y_true, proba_pos))
        except ValueError:
            metrics['roc_auc'] = float('nan')
    logger.info(
        f"[{name}] Acc={metrics['accuracy']:.3f} BalAcc={metrics['balanced_accuracy']:.3f} "
        f"F1+={metrics['f1_positive']:.3f} AP={metrics['average_precision']:.3f} "
        f"AUC={metrics.get('roc_auc', float('nan')):.3f}"
    )
    return metrics


def _save_artifacts(filenames, model, scaler, feature_names, importance_df, threshold, optuna_study):
    for path in filenames.values():
        if path:
            d = os.path.dirname(path)
            if d:
                os.makedirs(d, exist_ok=True)

    if filenames.get('model'):
        joblib.dump(model, filenames['model'])
        logger.info(f"Model → {filenames['model']}")
    if filenames.get('scaler'):
        joblib.dump(scaler, filenames['scaler'])
        logger.info(f"Scaler → {filenames['scaler']}")
    if filenames.get('feature_list'):
        with open(filenames['feature_list'], 'w') as f:
            for name in feature_names:
                f.write(f"{name}\n")
        logger.info(f"Feature list ({len(feature_names)}) → {filenames['feature_list']}")
    if filenames.get('importance') and importance_df is not None and not importance_df.empty:
        importance_df.to_csv(filenames['importance'], index=False)
        logger.info(f"Importance → {filenames['importance']}")
    if filenames.get('optimal_threshold'):
        joblib.dump({'threshold': float(threshold)}, filenames['optimal_threshold'])
        logger.info(f"Threshold ({threshold:.3f}) → {filenames['optimal_threshold']}")
    if filenames.get('optuna_study') and optuna_study is not None:
        try:
            joblib.dump(optuna_study, filenames['optuna_study'])
            logger.info(f"Optuna study → {filenames['optuna_study']}")
        except Exception as e:
            logger.warning(f"Could not save Optuna study: {e}")


def train_one_type(synk_type, results_csv=None, expert_csv=None,
                   skip_tuning=False, save_artifacts=True):
    """Train a single binary synkinesis classifier end-to-end. Returns metrics dict."""
    if synk_type not in SYNKINESIS_CONFIG:
        raise KeyError(f"Unknown type: {synk_type}")

    cfg = deepcopy(SYNKINESIS_CONFIG[synk_type])
    name = cfg['name']
    fn = cfg['filenames']
    training_cfg = cfg['training']
    feature_sel_cfg = cfg['feature_selection']

    ensure_artifact_dirs()
    if save_artifacts and fn.get('training_log'):
        _setup_logging(fn['training_log'])

    random_state = training_cfg.get('random_state', 42)
    test_size = training_cfg.get('test_size', 0.25)
    thr_cfg = training_cfg.get('threshold_optimization', {})

    logger.info(f"=== [{name}] Training pipeline starting ===")
    features_df, y, metadata = prepare_type_data(synk_type, results_csv, expert_csv)
    if len(np.unique(y)) < 2:
        raise ValueError(f"[{name}] Single-class data — cannot train binary classifier.")

    # Stratified train/test split. Threshold optimization uses cross-validated
    # out-of-fold predictions on the train set (see _oof_predict_proba below) so
    # no val slice is needed and the test set stays sealed until final reporting.
    indices = np.arange(len(y))
    train_idx, test_idx = train_test_split(
        indices, test_size=test_size, random_state=random_state, stratify=y
    )
    X_train = features_df.iloc[train_idx].reset_index(drop=True)
    X_test = features_df.iloc[test_idx].reset_index(drop=True)
    y_train = y[train_idx]
    y_test = y[test_idx]
    meta_train = metadata.iloc[train_idx].reset_index(drop=True)
    meta_test = metadata.iloc[test_idx].reset_index(drop=True)

    # Hybrid feature selection on the training partition only (test remains
    # unseen during selection).
    if feature_sel_cfg.get('enabled', True):
        n_top = feature_sel_cfg.get('top_n_features', 30)
        logger.info(f"[{name}] Hybrid FS targeting top {n_top} features (n_train={len(y_train)})...")
        selected_df = hybrid_feature_selection(X_train, y_train, feature_sel_cfg, n_top)
        selected_features = selected_df.columns.tolist()
        logger.info(f"[{name}] FS selected {len(selected_features)} features.")
    else:
        selected_features = features_df.columns.tolist()
    X_train = X_train[selected_features]
    X_test = X_test[selected_features]

    # Scaler fit on train only.
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Optuna search.
    optuna_cfg = training_cfg.get('hyperparameter_tuning', {}).get('optuna', {})
    smote_cfg = training_cfg.get('smote', {})
    class_weights = training_cfg.get('class_weights', {0: 1.0, 1: 1.0})
    best_xgb_params = {}
    optuna_study = None

    tuning_requested = (not skip_tuning
                        and training_cfg.get('hyperparameter_tuning', {}).get('enabled', True))
    if tuning_requested and not OPTUNA_AVAILABLE:
        logger.warning(
            f"[{name}] Hyperparameter tuning requested but optuna is not installed — "
            f"falling back to baseline params from config. "
            f"Install with: pip install optuna"
        )
    if tuning_requested and OPTUNA_AVAILABLE:
        n_trials = optuna_cfg.get('n_trials', 100)
        sampler = TPESampler(seed=random_state)
        pruner_name = optuna_cfg.get('pruner', 'HyperbandPruner')
        pruner = HyperbandPruner() if pruner_name == 'HyperbandPruner' else MedianPruner()
        objective = _create_optuna_objective(
            X_train_scaled, y_train, smote_cfg, optuna_cfg, class_weights, random_state
        )
        study = optuna.create_study(direction='maximize', sampler=sampler, pruner=pruner)
        logger.info(f"[{name}] Optuna {n_trials} trials, scoring=AP, pruner={pruner_name}...")
        study.optimize(objective, n_trials=n_trials, n_jobs=1, show_progress_bar=False)
        best_xgb_params = study.best_params
        optuna_study = study
        logger.info(f"[{name}] Best AP={study.best_value:.4f} params={best_xgb_params}")
    else:
        best_xgb_params = dict(training_cfg.get('model_params', {}))
        for k in ('objective', 'eval_metric', 'tree_method'):
            best_xgb_params.pop(k, None)

    # Final SMOTE on training set only (val and test stay clean).
    X_train_resampled, y_train_resampled = apply_smote_and_cleaning(
        X_train_scaled, y_train, smote_cfg, random_state, zone_name_log=name
    )

    # Build calibrated voting ensemble and fit. Imbalance is handled at the
    # base-estimator level (XGB scale_pos_weight + RF/ET balanced_subsample),
    # so no sample_weight is passed here.
    ensemble_cfg = ADVANCED_TRAINING_CONFIG.get('ensemble_options', {})
    voting = _build_voting_ensemble(best_xgb_params, ensemble_cfg, random_state, class_weights=class_weights)

    calib_cfg = training_cfg.get('calibration', {})
    method = calib_cfg.get('method', 'isotonic')
    cv_setting = calib_cfg.get('cv', 5)

    logger.info(f"[{name}] Fitting voting ensemble (calibration={method}, cv={cv_setting})...")
    if cv_setting == 'prefit':
        X_fit, X_calib, y_fit, y_calib = train_test_split(
            X_train_resampled, y_train_resampled,
            test_size=calib_cfg.get('calibration_split_size', 0.2),
            random_state=random_state, stratify=y_train_resampled,
        )
        voting.fit(X_fit, y_fit)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=FutureWarning)
            calibrated = CalibratedClassifierCV(voting, method=method, cv='prefit')
            calibrated.fit(X_calib, y_calib)
    else:
        calibrated = CalibratedClassifierCV(voting, method=method, cv=int(cv_setting))
        calibrated.fit(X_train_resampled, y_train_resampled)

    # Threshold optimization via cross-validated out-of-fold probabilities on the
    # train set. Falls back to threshold=0.5 if OOF entirely failed.
    train_oof = np.full(len(y_train), np.nan, dtype=float)
    if thr_cfg.get('enabled', True):
        n_splits = thr_cfg.get('cv_folds', 5)
        train_oof = _oof_predict_proba(
            X_train_scaled, y_train, smote_cfg, ensemble_cfg,
            best_xgb_params, class_weights, random_state,
            n_splits=n_splits, name=name,
        )
        thr_range = thr_cfg.get('positive_class_range', [0.15, 0.7])
        step = thr_cfg.get('step_size', 0.02)
        threshold, oof_f1 = _optimize_threshold(y_train, train_oof, thr_range, step)
        valid_oof = (~np.isnan(train_oof)).sum()
        logger.info(
            f"[{name}] Threshold optimization via {n_splits}-fold OOF "
            f"(valid={valid_oof}/{len(train_oof)}): thr={threshold:.3f} (F1+={oof_f1:.3f})"
        )
    else:
        threshold = 0.5

    # Test set evaluation only — never used to set the threshold.
    test_proba = calibrated.predict_proba(X_test_scaled)[:, 1]
    metrics = _binary_performance_summary(name, y_test, test_proba, threshold)
    metrics['n_train'] = int(len(y_train))
    metrics['n_test'] = int(len(y_test))
    metrics['n_pos_train'] = int((y_train == 1).sum())
    metrics['n_pos_test'] = int((y_test == 1).sum())
    metrics['n_features_selected'] = len(selected_features)

    # Label sensitivity: flag patients where the model strongly disagrees with
    # the expert label (a useful pre-publication audit of the FPRS key).
    if save_artifacts:
        review_path = fn.get('review_candidates_csv')
        review_df = _save_review_candidates(
            name, review_path, meta_train, y_train, train_oof,
            meta_test, y_test, test_proba, threshold,
        )
        metrics['n_review_candidates'] = int(len(review_df))

    importance_df = None
    try:
        # Pull feature importance from the underlying XGB inside the first calibrated estimator.
        first = calibrated.calibrated_classifiers_[0].estimator
        if hasattr(first, 'estimators_') and first.estimators_:
            xgb_est = next((e for e in first.estimators_ if isinstance(e, xgb.XGBClassifier)), None)
            if xgb_est is not None and hasattr(xgb_est, 'feature_importances_'):
                importance_df = pd.DataFrame({
                    'feature': selected_features,
                    'importance': xgb_est.feature_importances_,
                }).sort_values('importance', ascending=False)
    except Exception as e:
        logger.debug(f"[{name}] Could not extract feature importance: {e}")

    if save_artifacts:
        _save_artifacts(fn, calibrated, scaler, selected_features, importance_df, threshold, optuna_study)

    logger.info(f"=== [{name}] Done ===")
    return metrics
