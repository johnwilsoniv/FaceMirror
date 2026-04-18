# synkinesis_config.py
#
# Per-type configuration for binary coupling-pattern detectors. Mirrors the
# structure of paralysis_config.py (ZONE_CONFIG_DEFAULTS + per-type overrides)
# so the existing training stack can drive both pipelines.
#
# Ported from the historical S3 SYN Data Analysis module and modernized to
# match the current paralysis training defaults: HyperbandPruner, adaptive
# SMOTE → SMOTEENN cleanup, isotonic calibration with integer CV,
# XGBoost+RF VotingClassifier ensemble, and F1-maximize threshold optimization.

import os

import config_paths

MODEL_DIR = str(config_paths.get_models_dir())
OUTPUT_BASE = str(config_paths.get_output_base_dir())
LOG_DIR = os.path.join(OUTPUT_BASE, 'logs')
ANALYSIS_DIR = os.path.join(OUTPUT_BASE, 'analysis_results')
SYN_MODEL_PARENT_DIR = os.path.join(MODEL_DIR, 'synkinesis')

os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(ANALYSIS_DIR, exist_ok=True)

CLASS_NAMES = {0: 'No', 1: 'Yes'}
LOGGING_CONFIG = {
    'level': 'INFO',
    'format': '%(asctime)s - %(levelname)s - %(name)s - %(message)s'
}
INPUT_FILES = {
    'results_csv': os.path.expanduser('~/Documents/SplitFace/S3O Results/combined_results.csv'),
    'expert_key_csv': os.path.join(os.path.dirname(__file__), 'FPRS FP Key.csv')
}

# Patients excluded from synkinesis training/evaluation due to data-quality issues
# (e.g., did not follow recording protocol consistently). The label key still
# contains these patients; the data loader filters them after merging.
EXCLUDED_PATIENTS = [
    'IMG_3148',  # poor protocol adherence — AU patterns not interpretable
]

OPTUNA_PARAM_DIST_BINARY = {
    'learning_rate': ['float', 0.005, 0.2, {'log': True}],
    'max_depth': ['int', 3, 7],
    'n_estimators': ['int', 100, 600],
    'min_child_weight': ['int', 1, 7],
    'gamma': ['float', 0.0, 0.4],
    'subsample': ['float', 0.6, 1.0],
    'colsample_bytree': ['float', 0.6, 1.0],
    'reg_alpha': ['float', 1e-3, 1.0, {'log': True}],
    'reg_lambda': ['float', 1e-3, 2.0, {'log': True}],
    'scale_pos_weight': ['float', 1.0, 15.0],
}

OPTUNA_PARAM_DIST_BINARY_RARE = {
    'learning_rate': ['float', 0.005, 0.1, {'log': True}],
    'max_depth': ['int', 3, 5],
    'n_estimators': ['int', 100, 400],
    'min_child_weight': ['int', 3, 10],
    'gamma': ['float', 0.05, 0.5],
    'subsample': ['float', 0.5, 0.8],
    'colsample_bytree': ['float', 0.5, 0.8],
    'reg_alpha': ['float', 0.05, 1.0, {'log': True}],
    'reg_lambda': ['float', 0.05, 2.0, {'log': True}],
    'scale_pos_weight': ['float', 5.0, 25.0],
}

TYPE_CONFIG_DEFAULTS = {
    'feature_extraction': {
        'use_normalized': True,
        'percent_diff_cap': 200.0,
        'min_value': 0.0001,
        'include_aggregates': True,
        # Multiplicative trigger×coupled interaction features that distinguish
        # true coupling-pattern synkinesis from contralateral compensation.
        # Only meaningful for types with both trigger_aus and coupled_aus
        # populated (the four coupling/context_coupling types).
        'include_paralysis_conditioned': True,
    },
    'feature_selection': {
        'enabled': True,
        'top_n_features': 40,
        'method': 'rf_importance_in_workflow',
        'variance_threshold': 0.01,
    },
    'training': {
        'test_size': 0.25,
        'random_state': 42,
        'use_ensemble': True,
        'model_params': {
            'objective': 'binary:logistic',
            'eval_metric': 'aucpr',
            'learning_rate': 0.05,
            'max_depth': 5,
            'min_child_weight': 1,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'gamma': 0.1,
            'n_estimators': 300,
            'tree_method': 'hist',
        },
        'hyperparameter_tuning': {
            'enabled': True,
            'method': 'optuna',
            'optuna': {
                'n_trials': 150,
                'cv_folds': 5,
                'direction': 'maximize',
                'scoring': 'average_precision',
                'sampler': 'TPESampler',
                'pruner': 'HyperbandPruner',
                'param_distributions': OPTUNA_PARAM_DIST_BINARY,
                'optuna_early_stopping_rounds': 25,
                'patience': 15,
            }
        },
        'smote': {
            'enabled': True,
            'variant': 'borderline',
            'k_neighbors': 5,
            # Binary task — 'auto' means oversample minority to match majority.
            # Imbalanced types override with explicit {1: N} target counts.
            'sampling_strategy': 'auto',
            'adaptive_strategy_params': {
                'borderline_kind': 'borderline-1',
                'min_samples_after_smote': 50,
            },
            'apply_per_fold_in_tuning': True,
            'min_samples_per_class': 30,
            'use_smoteenn_after': True,
            'use_tomek_after': False,
            'enn_sampling_strategy': 'auto',
            'enn_kind_sel': 'mode',
        },
        'calibration': {
            'method': 'isotonic',
            'cv': 5,
            'ensemble': True,
            'n_jobs': -1,
        },
        'class_weights': {0: 1.0, 1: 2.0},
        'threshold_optimization': {
            'enabled': True,
            'method': 'f1_maximize',
            'positive_class_range': [0.15, 0.7],
            'step_size': 0.02,
        },
        'review_analysis': {
            'enabled': True,
            'top_k_influence': 30,
            'entropy_quantile': 0.9,
            'margin_quantile': 0.1,
            'true_label_prob_threshold': 0.4,
        },
    },
}


def _filenames(type_key):
    type_dir = os.path.join(SYN_MODEL_PARENT_DIR, type_key)
    analysis_dir = os.path.join(ANALYSIS_DIR, 'synkinesis', type_key)
    return {
        'model': os.path.join(type_dir, 'model.pkl'),
        'scaler': os.path.join(type_dir, 'scaler.pkl'),
        'feature_list': os.path.join(type_dir, 'features.list'),
        'importance': os.path.join(type_dir, 'feature_importance.csv'),
        'optuna_study': os.path.join(type_dir, 'optuna_study.pkl'),
        'optimal_threshold': os.path.join(type_dir, 'optimal_threshold.pkl'),
        'training_log': os.path.join(LOG_DIR, f'{type_key}_training.log'),
        'analysis_log': os.path.join(analysis_dir, f'{type_key}_analysis.log'),
        'critical_errors_report': os.path.join(analysis_dir, f'{type_key}_critical_errors_report.txt'),
        'review_candidates_csv': os.path.join(analysis_dir, f'{type_key}_review_candidates.csv'),
        'pr_curve_png': os.path.join(analysis_dir, f'{type_key}_precision_recall_curve.png'),
    }


SYNKINESIS_CONFIG = {
    'ocular_oral': {
        'name': 'Ocular-Oral',
        'pattern': 'coupling',
        'actions': ['ET', 'ES', 'RE', 'BK'],
        'trigger_aus': ['AU01_r', 'AU02_r', 'AU45_r'],
        'coupled_aus': ['AU12_r', 'AU25_r', 'AU14_r'],
        'context_aus': [],
        'expert_columns': {
            'left': 'Ocular-Oral Synkinesis Left',
            'right': 'Ocular-Oral Synkinesis Right',
        },
        'target_columns': {
            'left': 'Target_Left_Ocular_Oral',
            'right': 'Target_Right_Ocular_Oral',
        },
        'filenames': _filenames('ocular_oral'),
    },
    'oral_ocular': {
        'name': 'Oral-Ocular',
        'pattern': 'coupling',
        'actions': ['BS', 'SS', 'SO', 'SE', 'PL', 'LT'],
        'trigger_aus': ['AU12_r', 'AU25_r'],
        'coupled_aus': ['AU06_r', 'AU45_r'],
        'context_aus': [],
        'expert_columns': {
            'left': 'Oral-Ocular Synkinesis Left',
            'right': 'Oral-Ocular Synkinesis Right',
        },
        'target_columns': {
            'left': 'Target_Left_Oral_Ocular',
            'right': 'Target_Right_Oral_Ocular',
        },
        'filenames': _filenames('oral_ocular'),
    },
    'snarl_smile': {
        'name': 'Snarl-Smile',
        'pattern': 'coupling',
        'actions': ['BS'],
        'trigger_aus': ['AU12_r'],
        'coupled_aus': ['AU10_r', 'AU14_r', 'AU15_r'],
        'context_aus': [],
        'expert_columns': {
            'left': 'Snarl Smile Left',
            'right': 'Snarl Smile Right',
        },
        'target_columns': {
            'left': 'Target_Left_Snarl_Smile',
            'right': 'Target_Right_Snarl_Smile',
        },
        'filenames': _filenames('snarl_smile'),
    },
    'mentalis': {
        'name': 'Mentalis',
        'pattern': 'context_coupling',
        'actions': ['ET', 'ES', 'BS', 'SS', 'SO', 'SE', 'RE', 'PL', 'FR', 'BK', 'WN', 'BC', 'LT'],
        # AU17 (Chin Raiser) is the parasitic synkinesis; AU12 (smile) and AU15
        # (depressor) are the intended muscle activations whose ratio to AU17
        # quantifies the synkinetic coupling. Modeled as trigger/coupled so the
        # standard helper produces the cross-AU ratios.
        'trigger_aus': ['AU12_r', 'AU15_r'],
        'coupled_aus': ['AU17_r'],
        'context_aus': [],
        'asymmetry_actions': ['BS', 'SE'],  # Actions where bilateral AU17 asymmetry is most informative
        'expert_columns': {
            'left': 'Mentalis Synkinesis Left',
            'right': 'Mentalis Synkinesis Right',
        },
        'target_columns': {
            'left': 'Target_Left_Mentalis',
            'right': 'Target_Right_Mentalis',
        },
        'filenames': _filenames('mentalis'),
    },
    'hypertonicity': {
        'name': 'Hypertonicity',
        'pattern': 'baseline_resting',
        'actions': ['BL'],
        'trigger_aus': [],
        'coupled_aus': [],
        'interest_aus': ['AU12_r', 'AU14_r'],
        'context_aus': [],
        'reference_action': 'BS',
        'expert_columns': {
            'left': 'Hypertonicity Left',
            'right': 'Hypertonicity Right',
        },
        'target_columns': {
            'left': 'Target_Left_Hypertonicity',
            'right': 'Target_Right_Hypertonicity',
        },
        'filenames': _filenames('hypertonicity'),
    },
    'brow_cocked': {
        'name': 'Brow Cocked',
        'pattern': 'baseline_asymmetry',
        'actions': ['BL', 'RE'],
        'trigger_aus': [],
        'coupled_aus': [],
        'interest_aus': ['AU01_r', 'AU02_r'],
        'context_aus': [],
        'expert_columns': {
            'left': 'Brow Cocked Left',
            'right': 'Brow Cocked Right',
        },
        'target_columns': {
            'left': 'Target_Left_BrowCocked',
            'right': 'Target_Right_BrowCocked',
        },
        'filenames': _filenames('brow_cocked'),
    },
}


_TYPE_OVERRIDES = {
    # Well-balanced positives (n_pos >= 30 per side). Use modern adaptive defaults.
    'oral_ocular': {
        'feature_selection': {'top_n_features': 40},
        'training': {'class_weights': {0: 1.0, 1: 2.0}},
    },
    'snarl_smile': {
        'feature_selection': {'top_n_features': 15},
        'training': {
            'class_weights': {0: 1.0, 1: 2.5},
            'model_params': {'reg_alpha': 0.5, 'gamma': 0.5},
        },
    },
    'mentalis': {
        'feature_selection': {'top_n_features': 30},
        'training': {'class_weights': {0: 1.0, 1: 2.0}},
    },
    'hypertonicity': {
        'feature_selection': {'top_n_features': 10},
        'training': {'class_weights': {0: 1.0, 1: 1.5}},
    },
    # Mid-imbalance (~12-20 positives one side).
    'ocular_oral': {
        'feature_selection': {'top_n_features': 40},
        'training': {
            'class_weights': {0: 1.0, 1: 4.0},
            'smote': {
                'variant': 'adasyn',
                'k_neighbors': 3,
                'sampling_strategy': {1: 70},
                'min_samples_per_class': 5,
                'use_smoteenn_after': False,  # preserve scarce minority
            },
        },
    },
    # Severe imbalance (n_pos 9-12). Constrained search, ADASYN, no SMOTEENN.
    'brow_cocked': {
        'feature_selection': {'top_n_features': 10},
        'training': {
            'class_weights': {0: 1.0, 1: 8.0},
            'hyperparameter_tuning': {
                'optuna': {
                    'n_trials': 200,
                    'param_distributions': OPTUNA_PARAM_DIST_BINARY_RARE,
                }
            },
            'smote': {
                'variant': 'adasyn',
                'k_neighbors': 3,
                'sampling_strategy': {1: 100},
                'min_samples_per_class': 5,
                'use_smoteenn_after': False,
            },
            'calibration': {'method': 'sigmoid', 'cv': 'prefit'},
            'threshold_optimization': {'positive_class_range': [0.05, 0.5]},
        },
    },
}


def _deep_merge(base, override):
    out = {}
    for key, value in base.items():
        if isinstance(value, dict):
            out[key] = _deep_merge(value, override.get(key, {}) if isinstance(override, dict) else {})
        else:
            out[key] = value
    if isinstance(override, dict):
        for key, value in override.items():
            if key not in out:
                out[key] = value
            elif isinstance(value, dict) and isinstance(out[key], dict):
                out[key] = _deep_merge(out[key], value)
            else:
                out[key] = value
    return out


for _type_key, _type_data in SYNKINESIS_CONFIG.items():
    merged = {
        'feature_extraction': {**TYPE_CONFIG_DEFAULTS['feature_extraction']},
        'feature_selection': {**TYPE_CONFIG_DEFAULTS['feature_selection']},
        'training': _deep_merge(TYPE_CONFIG_DEFAULTS['training'], {}),
    }
    overrides = _TYPE_OVERRIDES.get(_type_key, {})
    for section in ('feature_extraction', 'feature_selection', 'training'):
        if section in overrides:
            merged[section] = _deep_merge(merged[section], overrides[section])
    for key, value in _type_data.items():
        if key not in ('feature_extraction', 'feature_selection', 'training'):
            merged[key] = value
    SYNKINESIS_CONFIG[_type_key] = merged


REVIEW_CONFIG = {
    'similarity_threshold': 0.95,
    'consistency_checks': {'cross_type': True, 'temporal': True, 'feature_based': True},
    'priority_weights': {'confidence': 0.4, 'inconsistency': 0.3, 'influence': 0.3},
    'export_format': 'xlsx',
    'include_features': True,
    'max_similar_patients': 5,
    'change_limits': {
        'max_changes_per_tier': {1: 20, 2: 40, 3: 80, 4: 150},
        'max_distribution_shift': 0.05,
    },
    'review_tiers': {
        1: {'name': 'High Confidence Errors (FP/FN)', 'priority': 'highest'},
        2: {'name': 'Consistency Issues', 'priority': 'high'},
        3: {'name': 'High Uncertainty', 'priority': 'medium'},
        4: {'name': 'General Review Pool', 'priority': 'low'},
    },
}

ADVANCED_TRAINING_CONFIG = {
    'ensemble_options': {
        'voting_type': 'soft',
        'weights': {'xgb': 0.6, 'rf': 0.2, 'et': 0.2},
        'random_forest_params': {
            'n_estimators': 100, 'max_depth': None,
            'min_samples_split': 2, 'min_samples_leaf': 1,
            'n_jobs': -1, 'oob_score': False, 'bootstrap': True,
        },
        'extra_trees_params': {
            'n_estimators': 100, 'max_depth': None,
            'min_samples_split': 2, 'min_samples_leaf': 1,
            'n_jobs': -1, 'bootstrap': False,
        },
    },
    'evaluation_metrics': [
        'accuracy', 'balanced_accuracy', 'f1_macro', 'f1_per_class',
        'cohen_kappa', 'matthews_corrcoef', 'auc_roc', 'auc_pr',
        'confusion_matrix', 'classification_report', 'precision_recall_curve',
    ],
    'monitoring': {
        'save_intermediate_results': True,
        'plot_optimization_history': True,
        'calculate_feature_importance': True,
        'save_model_checkpoints': True,
    },
}


def get_type_config(type_key):
    if type_key not in SYNKINESIS_CONFIG:
        raise KeyError(f"Unknown type '{type_key}'. Available: {list(SYNKINESIS_CONFIG)}")
    return SYNKINESIS_CONFIG[type_key]


def get_all_types():
    return list(SYNKINESIS_CONFIG.keys())


def ensure_artifact_dirs():
    for type_key in SYNKINESIS_CONFIG:
        os.makedirs(os.path.join(SYN_MODEL_PARENT_DIR, type_key), exist_ok=True)
        os.makedirs(os.path.join(ANALYSIS_DIR, 'synkinesis', type_key), exist_ok=True)
